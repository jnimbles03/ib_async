"""Inbound API frames must be rejected when the peer's length prefix is too large.

The socket peer (whoever accepts the client's TCP connection) chooses the
uint32 length prefix. These tests are the required behavior:

* A declared length greater than ``MAX_INBOUND_FRAME`` must close the socket
  and discard the receive buffer as soon as that length is known, including
  when the 4-byte prefix is split across reads and when a valid frame is
  already in the same buffer.
* Body bytes that arrive while that frame is incomplete must not be retained
  past ``MAX_INBOUND_FRAME``.
* A frame that is fully delivered and larger than the cap must be dropped
  before the decoder runs.
* A normal handshake and a small application frame must still be parsed.

The oversized-frame tests fail while the client buffers until the declared
length arrives. They pass once that cap is enforced.
"""

import asyncio
import contextlib
import struct

import pytest

from ib_async.client import Client

# Largest inbound frame the peer may cause the client to retain.
# Handshake and ordinary API messages are far below this. A uint32 length
# above it (including multi-gigabyte values) must be refused.
MAX_INBOUND_FRAME = 8 * 1024 * 1024

pytestmark = pytest.mark.asyncio


class Probe:
    """Wrapper stand-in. Records the callbacks the real client invokes."""

    def __init__(self):
        self.client: Client | None = None
        self.handshake_version: int | None = None
        self.times: list[int] = []
        self.high_water = 0

    def connectAck(self):
        assert self.client is not None
        self.handshake_version = self.client.serverVersion()

    def currentTime(self, time: int):
        self.times.append(time)

    def error(self, reqId, errorCode, errorString, advancedOrderRejectJson=""):
        return None

    def tcpDataProcessed(self):
        assert self.client is not None
        self.high_water = max(self.high_water, len(self.client._data))


def _frame(payload: bytes) -> bytes:
    return struct.pack(">I", len(payload)) + payload


def _make_client() -> tuple[Probe, Client]:
    probe = Probe()
    client = Client(probe)
    probe.client = client
    return probe, client


async def _read_client_hello(reader: asyncio.StreamReader) -> bytes:
    """Read the API greeting ``Client.connectAsync`` writes after connect."""
    buf = b""
    while True:
        chunk = await asyncio.wait_for(reader.read(4096), timeout=2)
        assert chunk, "client closed before sending the API hello"
        buf += chunk
        if not buf.startswith(b"API\0"):
            return buf
        if len(buf) >= 8:
            declared = struct.unpack(">I", buf[4:8])[0]
            if len(buf) >= 8 + declared:
                return buf


async def _until_peer_handled(client: Client, previous_recv: int, timeout: float = 2):
    """Wait until this process has read more socket bytes, or the transport is gone."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        if client._numBytesRecv > previous_recv or not client.conn.isConnected():
            # Let a close scheduled from inside data_received finish.
            await asyncio.sleep(0)
            return
        await asyncio.sleep(0.01)
    raise AssertionError(
        "client did not read the peer bytes "
        f"(recv={client._numBytesRecv}, transport={client.conn.isConnected()})"
    )


async def _drive(client: Client, peer):
    """Connect ``client`` to a local peer and run ``peer`` before tearing down.

    Snapshots taken inside ``peer`` happen while the socket is still the one
    ``connectAsync`` opened. Cancelling that task calls ``disconnect`` and
    ``reset``, which clears a buffer the peer had already observed.
    """
    result: dict = {}
    finished = asyncio.Event()

    async def handler(reader, writer):
        try:
            hello = await _read_client_hello(reader)
            result["hello"] = hello
            await peer(reader, writer, result)
        except Exception as exc:
            result["peer_error"] = exc
        finally:
            result["done"] = True
            writer.close()
            with contextlib.suppress(Exception):
                await writer.wait_closed()
            finished.set()

    server = await asyncio.start_server(handler, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    connect_task = asyncio.create_task(
        client.connectAsync("127.0.0.1", port, clientId=7, timeout=None)
    )
    try:
        await asyncio.wait_for(finished.wait(), timeout=30)
    finally:
        connect_task.cancel()
        with contextlib.suppress(BaseException):
            await connect_task
        server.close()
        await server.wait_closed()
    if peer_error := result.get("peer_error"):
        raise peer_error
    return result


async def _send_and_let_client_read(client: Client, writer, data: bytes):
    previous = client._numBytesRecv
    writer.write(data)
    try:
        await asyncio.wait_for(writer.drain(), timeout=2)
    except (
        asyncio.TimeoutError,
        ConnectionError,
        BrokenPipeError,
        OSError,
        RuntimeError,
    ):
        return False
    try:
        await _until_peer_handled(client, previous)
    except AssertionError:
        return False
    return True


async def test_huge_declared_length_is_rejected_before_any_body():
    """A multi-gigabyte length prefix alone must drop the connection.

    No body is sent. Keeping the socket open retains every later byte until
    that many bytes arrive.
    """
    probe, client = _make_client()

    async def peer(reader, writer, result):
        declared = 0xFFFFFFFF
        assert declared > MAX_INBOUND_FRAME
        kept_reading = await _send_and_let_client_read(
            client, writer, struct.pack(">I", declared)
        )
        result["kept_reading"] = kept_reading
        result["transport_open"] = client.conn.isConnected()
        result["buffered"] = len(client._data)
        result["high_water"] = probe.high_water

    result = await _drive(client, peer)
    assert result["hello"].startswith(b"API\0")
    assert result["transport_open"] is False, result
    assert result["buffered"] == 0, result
    assert probe.high_water <= 4, result


async def test_split_prefix_and_streamed_body_stay_within_cap():
    """The length may arrive one byte at a time, and the body may be streamed.

    Bytes received after the prefix is known must not push the receive buffer
    past the cap, and the socket must be closed without the rest of the frame.
    """
    probe, client = _make_client()
    declared = 0x7FFFFFFF
    assert declared > MAX_INBOUND_FRAME

    async def peer(reader, writer, result):
        header = struct.pack(">I", declared)
        # Incomplete prefix: the length is not known yet, so the socket may stay up.
        await _send_and_let_client_read(client, writer, header[:2])
        result["after_partial_prefix_open"] = client.conn.isConnected()
        result["after_partial_prefix_buffered"] = len(client._data)

        await _send_and_let_client_read(client, writer, header[2:])
        # Stream well past the cap. A client that only stops once the declared
        # frame is complete will retain every chunk.
        chunk = b"A" * (256 * 1024)
        body_sent = 0
        target = MAX_INBOUND_FRAME + len(chunk)
        while body_sent < target and client.conn.isConnected():
            wrote = await _send_and_let_client_read(client, writer, chunk)
            if not wrote:
                break
            body_sent += len(chunk)
        result["body_sent"] = body_sent
        result["transport_open"] = client.conn.isConnected()
        result["buffered"] = len(client._data)
        result["high_water"] = probe.high_water

    result = await _drive(client, peer)
    assert result["hello"].startswith(b"API\0")
    # Two bytes are not a length yet.
    assert result["after_partial_prefix_open"] is True
    assert result["after_partial_prefix_buffered"] <= 4
    assert result["high_water"] <= MAX_INBOUND_FRAME, (
        f"high_water={result['high_water']} buffered={result['buffered']} "
        f"body_sent={result['body_sent']}"
    )
    assert result["transport_open"] is False, result
    assert result["buffered"] == 0, result
    assert result["body_sent"] < declared, result


async def test_oversized_frame_glued_to_a_valid_frame_is_rejected():
    """A legal handshake and an oversized frame in one read share the parser loop.

    The handshake must be honored. The following frame must not be kept or decoded.
    """
    probe, client = _make_client()
    handshake = _frame(b"178\x0020250101 12:00:00\x00")
    # Complete application frame whose declared size is over the cap, glued on
    # so both frames are in the receive buffer together.
    payload = b"49\x001\x001700000000\x00"
    # Longer than the cap by more than one socket chunk, so the incomplete
    # frame is still sitting in the receive buffer after a full chunk and,
    # once the last bytes arrive, would be decoded if the cap is not applied
    # before the decoder.
    payload = payload + (b"Z" * (MAX_INBOUND_FRAME + 300_000 - len(payload)))
    assert len(payload) > MAX_INBOUND_FRAME + 256 * 1024
    oversized = struct.pack(">I", len(payload)) + payload

    async def peer(reader, writer, result):
        # Deliver the handshake first so a single 256 KiB recv cannot be the
        # only reason the oversized body is visible, then deliver the body in
        # chunks. The declared length is over the cap from the first body byte.
        await _send_and_let_client_read(client, writer, handshake)
        result["version_after_handshake"] = probe.handshake_version
        sent = 0
        view = oversized
        while sent < len(view) and client.conn.isConnected():
            nxt = view[sent : sent + 256 * 1024]
            wrote = await _send_and_let_client_read(client, writer, nxt)
            if not wrote:
                break
            sent += len(nxt)
        result["body_sent"] = sent
        result["transport_open"] = client.conn.isConnected()
        result["buffered"] = len(client._data)
        result["decoded_times"] = list(probe.times)
        result["high_water"] = probe.high_water

    result = await _drive(client, peer)
    assert result["hello"].startswith(b"API\0")
    assert result["version_after_handshake"] == 178, result
    assert result["high_water"] <= MAX_INBOUND_FRAME, (
        f"high_water={result['high_water']} buffered={result['buffered']} "
        f"body_sent={result['body_sent']} times={result['decoded_times']}"
    )
    assert result["decoded_times"] == [], result
    assert result["transport_open"] is False, result
    assert result["buffered"] == 0, result


async def test_small_handshake_and_message_are_parsed():
    """Frames inside the cap still travel through the real decoder."""
    probe, client = _make_client()
    handshake = _frame(b"178\x0020250101 12:00:00\x00")
    clock = _frame(b"49\x001\x001700000000\x00")
    assert len(handshake) < MAX_INBOUND_FRAME
    assert len(clock) < MAX_INBOUND_FRAME

    async def peer(reader, writer, result):
        await _send_and_let_client_read(client, writer, handshake + clock)
        result["version"] = probe.handshake_version
        result["times"] = list(probe.times)
        result["transport_open"] = client.conn.isConnected()
        result["buffered"] = len(client._data)

    result = await _drive(client, peer)
    assert result["hello"].startswith(b"API\0")
    assert result["version"] == 178
    assert result["times"] == [1700000000]
    assert result["transport_open"] is True
    assert result["buffered"] == 0
