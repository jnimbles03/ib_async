"""Missing option greeks must stay empty through market data and arithmetic.

A computation tick uses -1 and -2 for values the model did not produce.
Volatility, delta, gamma, and price already become empty. Vega and theta
have to follow the same rule, including when two contracts are combined
or one contract is scaled into a hedge.
"""

import asyncio
import struct

import pytest

from ib_async import IB, Option
from ib_async.wrapper import GREEKS_TICK_MAP

MISSING = {
    "tickAttrib": 1,
    "impliedVol": -1,
    "delta": -2,
    "optPrice": -1,
    "pvDividend": -1,
    "gamma": -2,
    "vega": -2,
    "theta": -2,
    "undPrice": -1,
}

# Values on either side of the sentinel, including a negative theta larger
# than two and a negative vega, must be kept. Only the exact sentinel is empty.
REAL = {
    "tickAttrib": 0,
    "impliedVol": 0.25,
    "delta": 0.5,
    "optPrice": 4.0,
    "pvDividend": 0.0,
    "gamma": 0.0625,
    "vega": 0.25,
    "theta": -3.5,
    "undPrice": 100.0,
}

ZERO = {
    "tickAttrib": 0,
    "impliedVol": 0.0,
    "delta": 0.0,
    "optPrice": 0.0,
    "pvDividend": 0.0,
    "gamma": 0.0,
    "vega": 0.0,
    "theta": 0.0,
    "undPrice": 100.0,
}


class _CaptureTransport:
    def __init__(self):
        self.payloads: list[bytes] = []

    def write(self, data: bytes) -> None:
        self.payloads.append(data)

    def write_eof(self) -> None:
        return None

    def close(self) -> None:
        return None


def _decode_request(payload: bytes) -> list[str]:
    size = struct.unpack(">I", payload[:4])[0]
    body = payload[4 : 4 + size].decode()
    fields = body.split("\0")
    fields.pop()
    return fields


def _ready_session() -> tuple[IB, _CaptureTransport]:
    """Open the API session the socket callbacks expect, without a gateway."""
    ib = IB()
    transport = _CaptureTransport()
    ib.client.conn.transport = transport
    ib.client.connState = ib.client.CONNECTED
    ib.client._apiReady = True
    ib.client._serverVersion = ib.client.MaxClientVersion
    ib.client.decoder.serverVersion = ib.client.MaxClientVersion
    return ib, transport


def _option(strike: float) -> Option:
    # reqMktData keys the ticker by conId, which a qualified contract has.
    contract = Option("SPY", "20261218", strike, "C", "SMART", currency="USD")
    contract.conId = int(strike)
    return contract


def _computation_fields(req_id: int, tick_type: int, values: dict) -> list:
    # Field order is the option-computation message the decoder unpacks:
    # message id, request id, tick type, attribute, then the eight floats.
    return [
        21,
        req_id,
        tick_type,
        values["tickAttrib"],
        values["impliedVol"],
        values["delta"],
        values["optPrice"],
        values["pvDividend"],
        values["gamma"],
        values["vega"],
        values["theta"],
        values["undPrice"],
    ]


def _deliver(ib: IB, fields: list, *, split: bool = False) -> None:
    """Feed one length-prefixed socket message through the connection callback."""
    body = ("\0".join(str(field) for field in fields) + "\0").encode()
    packet = struct.pack(">I", len(body)) + body
    if split:
        ib.client.conn.data_received(packet[:5])
        ib.client.conn.data_received(packet[5:])
    else:
        ib.client.conn.data_received(packet)


def _assert_empty(comp) -> None:
    assert comp is not None
    assert comp.tickAttrib == MISSING["tickAttrib"]
    assert comp.impliedVol is None
    assert comp.delta is None
    assert comp.optPrice is None
    assert comp.pvDividend is None
    assert comp.gamma is None
    assert comp.vega is None
    assert comp.theta is None
    assert comp.undPrice is None


def _assert_real(comp, values: dict) -> None:
    assert comp is not None
    assert comp.tickAttrib == values["tickAttrib"]
    assert comp.impliedVol == values["impliedVol"]
    assert comp.delta == values["delta"]
    assert comp.optPrice == values["optPrice"]
    assert comp.pvDividend == values["pvDividend"]
    assert comp.gamma == values["gamma"]
    assert comp.vega == values["vega"]
    assert comp.theta == values["theta"]
    assert comp.undPrice == values["undPrice"]


@pytest.mark.parametrize("tick_type", sorted(GREEKS_TICK_MAP))
async def test_market_data_missing_vega_and_theta_are_empty(tick_type):
    ib, transport = _ready_session()
    try:
        contract = _option(500)
        ticker = ib.reqMktData(contract)
        requests = [_decode_request(payload) for payload in transport.payloads]
        assert any(
            fields[0] == "1" and contract.symbol in fields for fields in requests
        )

        req_id = ib.wrapper.ticker2ReqId["mktData"][ticker]
        published = []
        ib.pendingTickersEvent += lambda tickers: published.append(set(tickers))
        before = ib.client._numMsgRecv
        _deliver(ib, _computation_fields(req_id, tick_type, MISSING))

        assert ib.client._numMsgRecv == before + 1
        assert published == [{ticker}]
        comp = getattr(ticker, GREEKS_TICK_MAP[tick_type])
        _assert_empty(comp)
    finally:
        ib.disconnect()


async def test_split_socket_packet_keeps_missing_model_greeks_empty():
    ib, _transport = _ready_session()
    try:
        ticker = ib.reqMktData(_option(500))
        req_id = ib.wrapper.ticker2ReqId["mktData"][ticker]
        _deliver(
            ib,
            _computation_fields(req_id, 13, MISSING),
            split=True,
        )
        _assert_empty(ticker.modelGreeks)
    finally:
        ib.disconnect()


@pytest.mark.parametrize("tick_type", sorted(GREEKS_TICK_MAP))
async def test_market_data_keeps_real_greeks_around_the_sentinel(tick_type):
    ib, _transport = _ready_session()
    try:
        ticker = ib.reqMktData(_option(500))
        req_id = ib.wrapper.ticker2ReqId["mktData"][ticker]
        attribute = GREEKS_TICK_MAP[tick_type]
        _deliver(ib, _computation_fields(req_id, tick_type, REAL))
        _assert_real(getattr(ticker, attribute), REAL)

        _deliver(ib, _computation_fields(req_id, tick_type, ZERO))
        _assert_real(getattr(ticker, attribute), ZERO)
    finally:
        ib.disconnect()


async def test_spread_and_hedge_math_drops_missing_vega_and_theta():
    ib, _transport = _ready_session()
    try:
        long_ticker = ib.reqMktData(_option(500))
        short_ticker = ib.reqMktData(_option(510))
        long_id = ib.wrapper.ticker2ReqId["mktData"][long_ticker]
        short_id = ib.wrapper.ticker2ReqId["mktData"][short_ticker]
        _deliver(ib, _computation_fields(long_id, 13, REAL))
        _deliver(ib, _computation_fields(short_id, 13, MISSING))
        long_greeks = long_ticker.modelGreeks
        short_greeks = short_ticker.modelGreeks
        _assert_real(long_greeks, REAL)
        _assert_empty(short_greeks)

        spread = long_greeks - short_greeks
        assert spread.delta == REAL["delta"]
        assert spread.gamma == REAL["gamma"]
        assert spread.vega == REAL["vega"]
        assert spread.theta == REAL["theta"]

        combined = long_greeks + short_greeks
        assert combined.vega == REAL["vega"]
        assert combined.theta == REAL["theta"]

        both_missing = short_greeks + short_greeks
        assert both_missing.vega == 0
        assert both_missing.theta == 0

        flat = short_greeks * 4
        assert flat.vega == 0
        assert flat.theta == 0
        short_hedge = short_greeks * -3
        assert short_hedge.vega == 0
        assert short_hedge.theta == 0
        fractional = short_greeks * 2.5
        assert fractional.vega == 0
        assert fractional.theta == 0

        scaled = long_greeks * 4
        assert scaled.vega == REAL["vega"] * 4
        assert scaled.theta == REAL["theta"] * 4
        scaled_short = long_greeks * -3
        assert scaled_short.vega == REAL["vega"] * -3
        assert scaled_short.theta == REAL["theta"] * -3
    finally:
        ib.disconnect()


async def test_calculation_replies_drop_missing_vega_and_theta():
    ib, transport = _ready_session()
    try:
        contract = _option(500)
        implied_id = ib.client._reqIdSeq
        implied_task = asyncio.create_task(
            ib.calculateImpliedVolatilityAsync(contract, 4.0, 100.0, [])
        )
        for _ in range(50):
            if implied_id in ib.wrapper._futures:
                break
            await asyncio.sleep(0)
        else:
            raise AssertionError("implied volatility request was not registered")

        _deliver(ib, _computation_fields(implied_id, 13, MISSING))
        implied = await implied_task
        _assert_empty(implied)

        price_id = ib.client._reqIdSeq
        price_task = asyncio.create_task(
            ib.calculateOptionPriceAsync(contract, 0.25, 100.0, [])
        )
        for _ in range(50):
            if price_id in ib.wrapper._futures:
                break
            await asyncio.sleep(0)
        else:
            raise AssertionError("option price request was not registered")

        _deliver(ib, _computation_fields(price_id, 13, REAL))
        priced = await price_task
        _assert_real(priced, REAL)

        requests = [_decode_request(payload) for payload in transport.payloads]
        assert any(
            fields[0] == "54" and contract.symbol in fields for fields in requests
        )
        assert any(
            fields[0] == "55" and contract.symbol in fields for fields in requests
        )
    finally:
        ib.disconnect()
