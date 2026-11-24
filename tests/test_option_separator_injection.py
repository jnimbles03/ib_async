"""Option lists must stay the pairs the caller supplied.

Client.send encodes every list field as ``tag=value;`` with no escaping
(ib_async/client.py). That is the same encoding the TWS API uses for one
protocol field (ibapi 9.81.1.post1, ibapi/tag_value.py and ibapi/client.py):
each pair is concatenated, then the whole string is one NUL-terminated field.

A semicolon inside one tag or value therefore becomes another pair, and an
equals sign inside a tag becomes a different tag. These tests drive the
request methods that pass a list into send. They fail while that join is
unescaped. They pass when the request is refused before any socket write, or
when the bytes that were written still decode to exactly the supplied pairs.
"""

from __future__ import annotations

import asyncio
import struct

import pytest

import ib_async as ibi

# Marker that is always the first pair, so the option field can be found
# after the rest of the message has been split on NUL.
_APP_TAG = "appChosen"
_APP_VALUE = "keep"
_USER_TAG = "userLimit"
_USER_VALUE = "10"


def _contract() -> ibi.Contract:
    # conId is set so ticker registration can hash the contract. The option
    # field does not depend on this id.
    return ibi.Stock("SPY", "SMART", "USD", conId=1)


def _pairs(options: list[ibi.TagValue]) -> list[tuple[str, str]]:
    return [(item.tag, item.value) for item in options]


def _protocol_pairs(field: str) -> list[tuple[str, str]]:
    """Inverse of the tag=value; join used for one option field.

    The official client builds that field by appending ``str(TagValue)``, and
    ``TagValue.__str__`` is ``tag + "=" + value + ";"``. Splitting on ";" and
    then on the first "=" is the matching decode: two explicitly supplied
    pairs and one value that contains ";" plus another pair encode to the
    same field.
    """
    pairs: list[tuple[str, str]] = []
    for part in field.split(";"):
        if part == "":
            continue
        tag, sep, value = part.partition("=")
        pairs.append((tag, value if sep else ""))
    return pairs


def _option_field(fields: list[str]) -> str:
    matches = [field for field in fields if field.startswith(f"{_APP_TAG}=")]
    assert len(matches) == 1, fields
    return matches[0]


def _decode_frame(blob: bytes) -> list[str]:
    assert len(blob) >= 4
    size = struct.unpack(">I", blob[:4])[0]
    assert size == len(blob) - 4
    body = blob[4:].decode()
    parts = body.split("\0")
    assert parts[-1] == ""
    return parts[:-1]


class _Session:
    """Connected client whose socket peer records the framed request."""

    def __init__(self) -> None:
        self.ib = ibi.IB()
        self._chunks: list[bytes] = []
        self._written: list[bytes] = []

    async def start(self) -> None:
        async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
            while True:
                data = await reader.read(65536)
                if not data:
                    break
                self._chunks.append(data)
            writer.close()

        self._server = await asyncio.start_server(handle, "127.0.0.1", 0)
        port = self._server.sockets[0].getsockname()[1]
        await self.ib.client.conn.connectAsync("127.0.0.1", port)
        self.ib.client.connState = ibi.Client.CONNECTED
        self.ib.client._serverVersion = ibi.Client.MaxClientVersion
        self.ib.client._apiReady = True
        self.ib.client._reqIdSeq = 1
        original = self.ib.client.conn.sendMsg

        def send_msg(msg: bytes) -> None:
            self._written.append(msg)
            original(msg)

        self.ib.client.conn.sendMsg = send_msg  # type: ignore[method-assign]

    async def close(self) -> None:
        self.ib.client.conn.disconnect()
        self._server.close()
        await self._server.wait_closed()

    async def submit(self, send) -> list[str] | None:
        """Run one request.

        Returns the decoded protocol fields, or None when send refused the
        options before writing. A refusal that still writes fails the test.
        """
        self._chunks.clear()
        self._written.clear()
        try:
            send(self.ib)
        except ConnectionError:
            raise
        except Exception:  # noqa: BLE001 - refusal may use any error type
            await asyncio.sleep(0)
            assert self._written == [], "options were refused after a socket write"
            assert self._chunks == [], "options were refused after bytes were sent"
            return None
        await asyncio.sleep(0.05)
        assert self._written, "request was neither sent nor refused"
        framed = b"".join(self._written)
        assert b"".join(self._chunks) == framed
        return _decode_frame(framed)


def _submit(send) -> list[str] | None:
    async def once():
        session = _Session()
        await session.start()
        try:
            return await session.submit(send)
        finally:
            await session.close()

    return asyncio.run(once())


def _assert_options_stay_intact(send, options: list[ibi.TagValue]) -> None:
    safe = [ibi.TagValue(_APP_TAG, _APP_VALUE), ibi.TagValue(_USER_TAG, _USER_VALUE)]
    safe_fields = _submit(lambda ib: send(ib, safe))
    assert safe_fields is not None, "a list of ordinary options must still be sent"
    safe_field = _option_field(safe_fields)
    assert safe_field == "".join(f"{tag}={value};" for tag, value in _pairs(safe))
    assert _protocol_pairs(safe_field) == _pairs(safe)

    fields = _submit(lambda ib: send(ib, options))
    if fields is None:
        return
    assert _protocol_pairs(_option_field(fields)) == _pairs(options)


def _place_order(ib: ibi.IB, options: list[ibi.TagValue]) -> None:
    order = ibi.LimitOrder("BUY", 1, 10, orderMiscOptions=options)
    ib.placeOrder(_contract(), order)


def _scanner_options(ib: ibi.IB, options: list[ibi.TagValue]) -> None:
    ib.reqScannerSubscription(
        ibi.ScannerSubscription(
            instrument="STK", locationCode="STK.US", scanCode="TOP_PERC_GAIN"
        ),
        options,
        [ibi.TagValue("otherFilter", "untouched")],
    )


def _scanner_filters(ib: ibi.IB, options: list[ibi.TagValue]) -> None:
    ib.reqScannerSubscription(
        ibi.ScannerSubscription(
            instrument="STK", locationCode="STK.US", scanCode="TOP_PERC_GAIN"
        ),
        [ibi.TagValue("otherOption", "untouched")],
        options,
    )


def _real_time_bars(ib: ibi.IB, options: list[ibi.TagValue]) -> None:
    ib.reqRealTimeBars(_contract(), 5, "TRADES", False, options)


def _market_data(ib: ibi.IB, options: list[ibi.TagValue]) -> None:
    ib.reqMktData(_contract(), "", False, False, options)


def _market_depth(ib: ibi.IB, options: list[ibi.TagValue]) -> None:
    ib.reqMktDepth(_contract(), 5, False, options)


def _historical_bars(ib: ibi.IB, options: list[ibi.TagValue]) -> None:
    ib.client.reqHistoricalData(
        1,
        _contract(),
        "",
        "1 D",
        "1 min",
        "TRADES",
        True,
        1,
        False,
        options,
    )


def _fundamental_data(ib: ibi.IB, options: list[ibi.TagValue]) -> None:
    ib.client.reqFundamentalData(1, _contract(), "ReportSnapshot", options)


def _implied_volatility(ib: ibi.IB, options: list[ibi.TagValue]) -> None:
    ib.client.calculateImpliedVolatility(1, _contract(), 1.0, 100.0, options)


def _option_price(ib: ibi.IB, options: list[ibi.TagValue]) -> None:
    ib.client.calculateOptionPrice(1, _contract(), 0.2, 100.0, options)


def _news_article(ib: ibi.IB, options: list[ibi.TagValue]) -> None:
    ib.client.reqNewsArticle(1, "BZ", "article", options)


def _historical_news(ib: ibi.IB, options: list[ibi.TagValue]) -> None:
    ib.client.reqHistoricalNews(1, 265598, "BZ", "", "", 5, options)


def _historical_ticks(ib: ibi.IB, options: list[ibi.TagValue]) -> None:
    ib.client.reqHistoricalTicks(
        1,
        _contract(),
        "",
        "20260101 00:00:00",
        10,
        "TRADES",
        True,
        False,
        options,
    )


_OPERATIONS = [
    _place_order,
    _scanner_options,
    _scanner_filters,
    _real_time_bars,
    _market_data,
    _market_depth,
    _historical_bars,
    _fundamental_data,
    _implied_volatility,
    _option_price,
    _news_article,
    _historical_news,
    _historical_ticks,
]


def _value_with_extra_pair() -> list[ibi.TagValue]:
    # The value is the caller's single option. The text after the semicolon
    # is a second pair the caller did not pass as its own TagValue.
    return [
        ibi.TagValue(_APP_TAG, _APP_VALUE),
        ibi.TagValue(_USER_TAG, f"{_USER_VALUE};injectedScan=1"),
    ]


def _tag_with_extra_pair() -> list[ibi.TagValue]:
    return [
        ibi.TagValue(_APP_TAG, _APP_VALUE),
        ibi.TagValue(f"{_USER_TAG};injectedScan=1", _USER_VALUE),
    ]


def _tag_with_equals() -> list[ibi.TagValue]:
    return [
        ibi.TagValue(_APP_TAG, _APP_VALUE),
        ibi.TagValue(f"{_USER_TAG}=injectedScan", _USER_VALUE),
    ]


@pytest.mark.parametrize("send", _OPERATIONS, ids=lambda fn: fn.__name__)
def test_semicolon_in_value_does_not_add_an_option(send):
    _assert_options_stay_intact(send, _value_with_extra_pair())


@pytest.mark.parametrize("send", _OPERATIONS, ids=lambda fn: fn.__name__)
def test_semicolon_in_tag_does_not_add_an_option(send):
    _assert_options_stay_intact(send, _tag_with_extra_pair())


@pytest.mark.parametrize("send", _OPERATIONS, ids=lambda fn: fn.__name__)
def test_equals_in_tag_does_not_change_the_tag(send):
    _assert_options_stay_intact(send, _tag_with_equals())
