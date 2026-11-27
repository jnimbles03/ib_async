"""Signed generic ticks must survive the live market-data path.

Interactive Brokers delivers these values on incoming message 45
(TICK_GENERIC) as reqId, tick type, and a float. Index future premium
(tick 31, requested with generic tick list 162) is the points by which
the index future is over the cash index, so a futures discount is
negative. Halted (tick 49) uses -1 for "status unavailable" and 0 for
"not halted". None of the generic ticks below document a non-positive
value as a missing size.

The tests drive ``IB.reqMktData`` and the socket decoder. They expect
the numeric value from the gateway to be stored on the ticker and on
the tick, including when it is zero or negative.
"""

import struct

import pytest

import ib_async as ibi
from ib_async.objects import IBDefaults
from ib_async.wrapper import GENERIC_TICK_MAP

# Distinct from every value the tests deliver. A handler that substitutes
# the empty size for a non-positive tick fails these checks.
EMPTY_SIZE_SENTINEL = -1_000_000_000.0

# Generic tick list ids from the TWS API tick-type table. Halted (49) and
# delayed halted (90) are not requested this way: halted is a default tick,
# and delayed halted follows reqMarketDataType(3).
GENERIC_TICK_LIST = "104,106,162,236,293,294,295,411,460"

# Values that are real for at least one generic tick and that the
# positive-only clamp erases: a discount, a documented status, and zero.
NON_POSITIVE_SAMPLES = (-12.5, -1.0, 0.0)
POSITIVE_SAMPLE = 4.5


class RecordingTransport:
    def __init__(self):
        self.payloads: list[bytes] = []

    def write(self, data: bytes):
        self.payloads.append(data)

    def write_eof(self):
        return None

    def close(self):
        return None

    def messages(self) -> list[list[str]]:
        decoded = []
        for blob in self.payloads:
            size = struct.unpack(">I", blob[:4])[0]
            fields = blob[4 : 4 + size].decode().split("\0")
            fields.pop()
            decoded.append(fields)
        return decoded


def gateway_frame(*fields: str) -> bytes:
    """Length-prefixed TWS message, including the trailing NUL."""
    body = ("\0".join(fields) + "\0").encode()
    return struct.pack(">I", len(body)) + body


def connected_ib(defaults: IBDefaults | None = None) -> tuple[ibi.IB, RecordingTransport]:
    """An IB client that records outbound API messages and accepts inbound frames.

    No socket is opened. ``reqMktData`` still runs the production send path,
    and inbound bytes still run ``Connection.data_received`` ->
    ``Client._onSocketHasData`` -> ``Decoder.interpret``.
    """
    ib = ibi.IB(defaults or IBDefaults())
    transport = RecordingTransport()
    client = ib.client
    client.connState = client.CONNECTED
    client._apiReady = True
    client._serverVersion = client.MaxClientVersion
    client.decoder.serverVersion = client.MaxClientVersion
    client._reqIdSeq = 1000
    client.conn.transport = transport
    return ib, transport


def subscribe(ib: ibi.IB, generic_tick_list: str) -> ibi.Ticker:
    contract = ibi.Future(
        conId=1,
        symbol="ES",
        lastTradeDateOrContractMonth="202603",
        exchange="CME",
        currency="USD",
    )
    return ib.reqMktData(contract, genericTickList=generic_tick_list)


def deliver_tick_generic(
    ib: ibi.IB, req_id: int, tick_type: int, value: float, *, split: bool = False
) -> None:
    """Feed one TICK_GENERIC message (id 45, version, reqId, tickType, value)."""
    frame = gateway_frame("45", "1", str(req_id), str(tick_type), repr(value))
    if split:
        ib.client.conn.data_received(frame[:6])
        assert ib.client._data, "a short read must wait for the rest of the frame"
        ib.client.conn.data_received(frame[6:])
    else:
        ib.client.conn.data_received(frame)


def req_id_of(ib: ibi.IB, ticker: ibi.Ticker) -> int:
    return ib.wrapper.ticker2ReqId["mktData"][ticker]


def test_request_for_index_future_premium_uses_generic_tick_162():
    ib, transport = connected_ib()
    ticker = subscribe(ib, GENERIC_TICK_LIST)
    assert ticker.indexFuturePremium != ticker.indexFuturePremium  # unset nan

    outbound = [
        fields
        for fields in transport.messages()
        if fields and fields[0] == "1"
    ]
    assert len(outbound) == 1
    assert GENERIC_TICK_LIST in outbound[0]
    assert req_id_of(ib, ticker) == 1000


def test_discounted_index_future_premium_is_kept():
    """A futures discount must stay negative on the ticker and on the tick."""
    ib, _transport = connected_ib()
    ib.reqMarketDataType(1)
    ticker = subscribe(ib, "162")
    req_id = req_id_of(ib, ticker)
    published: list[float] = []
    ticker.updateEvent += lambda updated: published.append(updated.indexFuturePremium)

    deliver_tick_generic(ib, req_id, 31, POSITIVE_SAMPLE)
    assert ticker.indexFuturePremium == POSITIVE_SAMPLE
    assert published[-1] == POSITIVE_SAMPLE

    deliver_tick_generic(ib, req_id, 31, -12.5, split=True)
    assert ticker.indexFuturePremium == -12.5
    assert published[-1] == -12.5
    tick = ticker.ticks[-1]
    assert tick.tickType == 31
    assert tick.price == -12.5
    assert tick.size == 0

    # Flat to the index is a real premium of zero, not an empty size.
    deliver_tick_generic(ib, req_id, 31, 0.0)
    assert ticker.indexFuturePremium == 0.0
    assert ticker.ticks[-1].price == 0.0
    assert published[-1] == 0.0


def test_discounted_premium_stays_negative_when_empty_size_is_customized():
    ib, _transport = connected_ib(IBDefaults(emptySize=EMPTY_SIZE_SENTINEL))
    ticker = subscribe(ib, "162")
    req_id = req_id_of(ib, ticker)

    deliver_tick_generic(ib, req_id, 31, -12.5)
    assert ticker.indexFuturePremium == -12.5
    assert ticker.ticks[-1].price == -12.5

    deliver_tick_generic(ib, req_id, 31, 0.0)
    assert ticker.indexFuturePremium == 0.0
    assert ticker.ticks[-1].price == 0.0


def test_halted_statuses_are_not_rewritten():
    """Halted -1 means unavailable and 0 means not halted. Both must be kept."""
    ib, _transport = connected_ib(IBDefaults(emptySize=EMPTY_SIZE_SENTINEL))
    ib.reqMarketDataType(3)
    ticker = subscribe(ib, "")
    req_id = req_id_of(ib, ticker)

    for tick_type, field in ((49, "halted"), (90, "delayedHalted")):
        for value in (-1.0, 0.0, 1.0, 2.0):
            deliver_tick_generic(ib, req_id, tick_type, value)
            assert getattr(ticker, field) == value
            assert ticker.ticks[-1].tickType == tick_type
            assert ticker.ticks[-1].price == value


@pytest.mark.parametrize("tick_type", sorted(GENERIC_TICK_MAP))
@pytest.mark.parametrize("value", NON_POSITIVE_SAMPLES + (POSITIVE_SAMPLE,))
def test_every_generic_tick_keeps_the_gateway_value(tick_type: int, value: float):
    ib, _transport = connected_ib(IBDefaults(emptySize=EMPTY_SIZE_SENTINEL))
    ticker = subscribe(ib, GENERIC_TICK_LIST)
    req_id = req_id_of(ib, ticker)
    field = GENERIC_TICK_MAP[tick_type]

    deliver_tick_generic(ib, req_id, tick_type, value, split=True)

    assert getattr(ticker, field) == value
    assert ticker.ticks[-1] == ibi.TickData(ticker.ticks[-1].time, tick_type, value, 0)
    assert ticker in ib.pendingTickers()
