"""NUL characters in outbound fields must be rejected before any bytes are sent.

The IB socket protocol separates fields with NUL. ``Client.send`` copies string
fields, contract fields, and tag values through unchanged, so a NUL inside one
of them splits that field and shifts every later field. These tests are the
spec for closing that gap: they fail while a NUL is still written on the wire,
and they pass once every such value raises ``ValueError`` first.

A refusal has to happen before the connection is written and before the
message is queued. The caller's order and contract objects keep the values
they were given.
"""

from __future__ import annotations

import struct
from types import SimpleNamespace

import pytest

import ib_async as ibi

NUL = "\x00"


class _Capture:
    def __init__(self) -> None:
        self.writes: list[bytes] = []

    def write(self, data: bytes) -> None:
        self.writes.append(bytes(data))

    def write_eof(self) -> None:
        pass

    def close(self) -> None:
        pass


def _client(server_version: int = 178) -> ibi.Client:
    client = ibi.Client(SimpleNamespace())
    client.connState = ibi.Client.CONNECTED
    client._serverVersion = server_version
    client._apiReady = True
    client.conn.transport = _Capture()
    return client


def _frames(client: ibi.Client) -> list[bytes]:
    return client.conn.transport.writes


def _fields(frame: bytes) -> list[str]:
    """Split one length-prefixed frame the way ``Client._onSocketHasData`` does."""
    size = struct.unpack(">I", frame[:4])[0]
    payload = frame[4 : 4 + size]
    assert len(payload) == size
    parts = payload.decode("latin1").split(NUL)
    assert parts[-1] == ""
    return parts[:-1]


def _assert_refused(client: ibi.Client, send_call) -> None:
    """The call must raise ``ValueError`` before a frame is queued or written."""
    try:
        send_call()
    except ValueError:
        refused = True
    else:
        refused = False

    writes = _frames(client)
    if writes or client._msgQ or not refused:
        detail = ""
        if writes:
            fields = _fields(writes[0])
            detail = f" first frame has {len(fields)} fields: {fields[:40]!r}"
        pytest.fail(
            "outbound message containing a NUL was not refused before send;"
            f" raised={refused} writes={len(writes)} queued={len(client._msgQ)}.{detail}"
        )


def _stock() -> ibi.Contract:
    return ibi.Stock("SPY", "SMART", "USD")


def _staged_order(**kwargs) -> ibi.Order:
    order = ibi.LimitOrder(
        "BUY",
        10,
        123.45,
        tif="DAY",
        account="DU1234567",
        orderRef="staged-ref",
        transmit=False,
    )
    for name, value in kwargs.items():
        setattr(order, name, value)
    return order


def test_clean_place_order_keeps_transmit_off_on_the_wire():
    """A normal staged order is still sent, and transmit stays the object's flag."""
    client = _client()
    order = _staged_order()
    client.placeOrder(42, _stock(), order)

    frames = _frames(client)
    assert len(frames) == 1
    fields = _fields(frames[0])
    ref_at = fields.index("staged-ref")
    assert fields[ref_at + 1] == "0"
    assert fields[16] == "BUY"
    assert fields[17] == "10"
    assert fields[23] == "DU1234567"
    assert order.transmit is False


def test_clean_string_contract_and_tags_are_sent():
    client = _client()
    client.send("SAFE", _stock(), [ibi.TagValue("tag", "value")])
    assert len(_frames(client)) == 1
    assert not client._msgQ


@pytest.mark.parametrize("server_version", [176, 178])
def test_null_in_order_ref_cannot_flip_transmit(server_version):
    client = _client(server_version)
    order = _staged_order(orderRef=f"staged-ref{NUL}1")
    contract = _stock()

    _assert_refused(client, lambda: client.placeOrder(42, contract, order))

    assert order.transmit is False
    assert order.orderRef == f"staged-ref{NUL}1"
    assert order.action == "BUY"
    assert order.totalQuantity == 10
    assert order.account == "DU1234567"
    assert contract.symbol == "SPY"


@pytest.mark.parametrize(
    "field",
    [
        "action",
        "orderType",
        "tif",
        "ocaGroup",
        "account",
        "openClose",
        "goodAfterTime",
        "goodTillDate",
        "faGroup",
        "faMethod",
        "faPercentage",
        "modelCode",
        "designatedLocation",
        "rule80A",
        "settlingFirm",
        "algoId",
        "clearingAccount",
        "clearingIntent",
        "scaleTable",
        "activeStartTime",
        "activeStopTime",
        "hedgeType",
        "hedgeParam",
        "extOperator",
        "advancedErrorOverride",
        "manualOrderTime",
        "mifid2DecisionMaker",
        "mifid2DecisionAlgo",
        "mifid2ExecutionTrader",
        "mifid2ExecutionAlgo",
    ],
)
def test_place_order_rejects_null_in_order_strings(field):
    client = _client()
    injected = f"value{NUL}1"
    kwargs = {field: injected}
    if field == "hedgeParam":
        kwargs["hedgeType"] = "P"
    if field == "orderType":
        kwargs["orderType"] = f"LMT{NUL}1"
    order = _staged_order(**kwargs)
    original = getattr(order, field)

    _assert_refused(client, lambda: client.placeOrder(7, _stock(), order))
    assert getattr(order, field) == original
    assert order.transmit is False


def test_place_order_rejects_null_in_fa_profile_before_server_177():
    """Server versions below 177 still encode ``faProfile`` on the wire."""
    client = _client(176)
    order = _staged_order(faProfile=f"profile{NUL}1")

    _assert_refused(client, lambda: client.placeOrder(7, _stock(), order))
    assert order.faProfile == f"profile{NUL}1"
    assert order.transmit is False


def test_place_order_rejects_null_in_soft_dollar_tier():
    client = _client()
    order = _staged_order()
    order.softDollarTier = ibi.SoftDollarTier(name=f"tier{NUL}1", val="fee")

    _assert_refused(client, lambda: client.placeOrder(7, _stock(), order))
    assert order.softDollarTier.name == f"tier{NUL}1"
    assert order.softDollarTier.val == "fee"
    assert order.transmit is False


def test_place_order_rejects_null_in_soft_dollar_value():
    client = _client()
    order = _staged_order()
    order.softDollarTier = ibi.SoftDollarTier(name="tier", val=f"fee{NUL}1")

    _assert_refused(client, lambda: client.placeOrder(7, _stock(), order))
    assert order.softDollarTier.val == f"fee{NUL}1"


@pytest.mark.parametrize(
    "field",
    [
        "symbol",
        "secType",
        "lastTradeDateOrContractMonth",
        "right",
        "multiplier",
        "exchange",
        "primaryExchange",
        "currency",
        "localSymbol",
        "tradingClass",
        "secIdType",
        "secId",
    ],
)
def test_place_order_rejects_null_in_contract_fields(field):
    client = _client()
    contract = _stock()
    injected = f"SPY{NUL}SELL"
    setattr(contract, field, injected)

    _assert_refused(client, lambda: client.placeOrder(7, contract, _staged_order()))
    assert getattr(contract, field) == injected


def test_place_order_rejects_null_in_combo_leg_strings():
    client = _client()
    contract = ibi.Contract(
        symbol="SPY", secType="BAG", exchange="SMART", currency="USD"
    )
    contract.comboLegs = [
        ibi.ComboLeg(
            conId=1,
            ratio=1,
            action=f"BUY{NUL}SELL",
            exchange="SMART",
            designatedLocation=f"away{NUL}1",
        )
    ]
    order = _staged_order()
    order.orderComboLegs = [ibi.OrderComboLeg(price=1.5)]
    order.smartComboRoutingParams = [ibi.TagValue("tag", f"val{NUL}1")]

    _assert_refused(client, lambda: client.placeOrder(7, contract, order))
    assert contract.comboLegs[0].action == f"BUY{NUL}SELL"
    assert order.transmit is False


def test_place_order_rejects_null_in_algo_param_tag_and_value():
    client = _client()
    order = _staged_order(algoStrategy="VWAP")
    order.algoParams = [ibi.TagValue(f"tag{NUL}1", f"val{NUL}1")]

    _assert_refused(client, lambda: client.placeOrder(7, _stock(), order))
    assert order.algoParams[0].tag == f"tag{NUL}1"
    assert order.algoParams[0].value == f"val{NUL}1"
    assert order.transmit is False


def test_place_order_rejects_null_in_order_misc_options():
    client = _client()
    order = _staged_order()
    order.orderMiscOptions = [ibi.TagValue("customer", f"x{NUL}1")]

    _assert_refused(client, lambda: client.placeOrder(7, _stock(), order))
    assert order.orderMiscOptions[0].value == f"x{NUL}1"


def test_place_order_rejects_null_in_condition_strings():
    client = _client()
    order = _staged_order()
    order.conditions = [
        ibi.ExecutionCondition(secType="STK", exch="SMART", symbol=f"SPY{NUL}1"),
        ibi.TimeCondition(time=f"20260101-12:00:00{NUL}1"),
    ]

    _assert_refused(client, lambda: client.placeOrder(7, _stock(), order))
    assert order.conditions[0].symbol == f"SPY{NUL}1"
    assert order.conditions[1].time == f"20260101-12:00:00{NUL}1"
    assert order.transmit is False


@pytest.mark.parametrize("payload", [NUL, f"{NUL}1", f"ref{NUL}", f"ref{NUL}1{NUL}0"])
def test_send_rejects_null_in_a_string_field(payload):
    client = _client()
    _assert_refused(client, lambda: client.send(1, payload, 2))


def test_send_rejects_null_inside_a_contract():
    client = _client()
    contract = _stock()
    contract.symbol = f"SPY{NUL}SELL"
    _assert_refused(client, lambda: client.send(1, contract))
    assert contract.symbol == f"SPY{NUL}SELL"


def test_send_rejects_null_inside_tag_values():
    client = _client()
    tags = [ibi.TagValue(f"t{NUL}1", "v"), ibi.TagValue("ok", f"v{NUL}1")]
    _assert_refused(client, lambda: client.send(tags))
    assert tags[0].tag == f"t{NUL}1"
    assert tags[1].value == f"v{NUL}1"


def test_send_rejects_null_when_make_empty_is_disabled():
    client = _client()
    _assert_refused(client, lambda: client.send(f"keep{NUL}1", makeEmpty=False))


def test_cancel_order_rejects_null_in_manual_time():
    client = _client()
    manual = f"20260101-12:00:00{NUL}1"
    _assert_refused(client, lambda: client.cancelOrder(9, manual))


def test_req_mkt_data_rejects_null_in_symbol_and_ticks_and_options():
    client = _client()
    contract = _stock()
    contract.symbol = f"SPY{NUL}1"
    options = [ibi.TagValue("opt", f"v{NUL}1")]
    _assert_refused(
        client,
        lambda: client.reqMktData(1, contract, f"233{NUL}1", False, False, options),
    )
    assert contract.symbol == f"SPY{NUL}1"


def test_req_account_updates_rejects_null_account():
    client = _client()
    _assert_refused(client, lambda: client.reqAccountUpdates(True, f"DU123{NUL}1"))


def test_req_executions_rejects_null_in_filter_strings():
    client = _client()
    filt = ibi.ExecutionFilter(
        acctCode=f"DU123{NUL}1",
        time=f"20260101-00:00:00{NUL}1",
        symbol=f"SPY{NUL}1",
        secType=f"STK{NUL}1",
        exchange=f"SMART{NUL}1",
        side=f"BUY{NUL}1",
    )
    _assert_refused(client, lambda: client.reqExecutions(3, filt))
    assert filt.symbol == f"SPY{NUL}1"
    assert filt.acctCode == f"DU123{NUL}1"


def test_historical_requests_reject_null_strings():
    client = _client()
    contract = _stock()
    _assert_refused(
        client,
        lambda: client.reqHistoricalData(
            1,
            contract,
            f"20260101 12:00:00{NUL}1",
            "1 D",
            "1 min",
            "TRADES",
            True,
            1,
            False,
            [ibi.TagValue("tag", f"v{NUL}1")],
        ),
    )
    _assert_refused(
        client,
        lambda: client.reqHistoricalTicks(
            2,
            contract,
            f"20260101 12:00:00{NUL}1",
            "",
            10,
            "TRADES",
            True,
            True,
            [ibi.TagValue("tag", "v")],
        ),
    )
    _assert_refused(
        client,
        lambda: client.reqHeadTimeStamp(3, contract, f"TRADES{NUL}1", True, 1),
    )


def test_scanner_and_news_reject_null_strings():
    client = _client()
    sub = ibi.ScannerSubscription(instrument=f"STK{NUL}1", locationCode="STK.US")
    _assert_refused(
        client,
        lambda: client.reqScannerSubscription(
            1,
            sub,
            [ibi.TagValue("opt", f"v{NUL}1")],
            [ibi.TagValue("flt", "v")],
        ),
    )
    assert sub.instrument == f"STK{NUL}1"
    _assert_refused(
        client,
        lambda: client.reqNewsArticle(2, "BZ", f"article{NUL}1", []),
    )
    _assert_refused(
        client,
        lambda: client.reqHistoricalNews(
            3,
            265598,
            f"BZ{NUL}FLY",
            "20260101-00:00:00",
            "20260102-00:00:00",
            10,
            [],
        ),
    )


def test_account_fa_and_group_requests_reject_null_strings():
    client = _client()
    _assert_refused(
        client,
        lambda: client.reqAccountSummary(1, f"All{NUL}1", "NetLiquidation"),
    )
    _assert_refused(client, lambda: client.requestFA(f"<xml{NUL}1>"))
    _assert_refused(client, lambda: client.replaceFA(2, 1, f"<xml{NUL}1>"))
    _assert_refused(
        client,
        lambda: client.reqAccountUpdatesMulti(3, f"DU123{NUL}1", "", False),
    )
    _assert_refused(client, lambda: client.reqPositionsMulti(4, f"DU123{NUL}1", ""))
    _assert_refused(client, lambda: client.reqPnL(5, f"DU123{NUL}1", ""))
    _assert_refused(client, lambda: client.reqPnLSingle(6, "DU123", f"model{NUL}1", 1))
    _assert_refused(client, lambda: client.subscribeToGroupEvents(7, f"group{NUL}1"))
    _assert_refused(client, lambda: client.updateDisplayGroup(8, f"SPY{NUL}1"))


def test_verification_and_lookup_requests_reject_null_strings():
    client = _client()
    _assert_refused(client, lambda: client.verifyRequest(f"name{NUL}1", "1"))
    _assert_refused(client, lambda: client.verifyMessage(f"data{NUL}1"))
    _assert_refused(
        client,
        lambda: client.verifyAndAuthRequest("name", "1", f"key{NUL}1"),
    )
    _assert_refused(client, lambda: client.reqMatchingSymbols(1, f"SPY{NUL}1"))
    _assert_refused(client, lambda: client.reqSmartComponents(2, f"SMART{NUL}1"))
    _assert_refused(
        client,
        lambda: client.reqSecDefOptParams(3, f"SPY{NUL}1", "", "STK", 0),
    )


def test_market_data_helpers_reject_null_in_contract_or_options():
    client = _client()
    contract = _stock()
    contract.localSymbol = f"SPY{NUL}1"
    _assert_refused(
        client,
        lambda: client.reqMktDepth(1, contract, 5, False, [ibi.TagValue("t", "v")]),
    )
    _assert_refused(
        client,
        lambda: client.reqRealTimeBars(
            2, _stock(), 5, f"TRADES{NUL}1", True, [ibi.TagValue("t", f"v{NUL}1")]
        ),
    )
    _assert_refused(
        client,
        lambda: client.reqFundamentalData(
            3, _stock(), f"ReportSnapshot{NUL}1", [ibi.TagValue("t", "v")]
        ),
    )
    _assert_refused(
        client,
        lambda: client.reqTickByTickData(4, _stock(), f"Last{NUL}1", 0, False),
    )
    _assert_refused(
        client,
        lambda: client.calculateImpliedVolatility(
            5, _stock(), 1.0, 100.0, [ibi.TagValue("t", f"v{NUL}1")]
        ),
    )
    _assert_refused(
        client,
        lambda: client.exerciseOptions(6, _stock(), 1, 1, f"DU123{NUL}1", False),
    )
    _assert_refused(client, lambda: client.reqHistogramData(7, _stock(), True, "1 day"))
    poisoned = _stock()
    poisoned.tradingClass = f"SPY{NUL}1"
    _assert_refused(
        client, lambda: client.reqHistogramData(8, poisoned, True, "1 day")
    )


def test_wsh_event_request_rejects_null_in_filter_and_dates():
    client = _client()
    data = ibi.WshEventData(
        conId=1,
        filter=f"watch{NUL}1",
        startDate=f"20260101{NUL}1",
        endDate="20260102",
    )
    _assert_refused(client, lambda: client.reqWshEventData(1, data))
    assert data.filter == f"watch{NUL}1"
    assert data.startDate == f"20260101{NUL}1"


def test_contract_details_rejects_null_in_symbol():
    client = _client()
    contract = _stock()
    contract.symbol = f"SPY{NUL}1"
    _assert_refused(client, lambda: client.reqContractDetails(1, contract))
    assert contract.symbol == f"SPY{NUL}1"
