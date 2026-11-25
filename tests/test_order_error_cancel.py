"""Broker modify/cancel failures must not mark a live order cancelled.

These tests fail while ``Wrapper.error`` treats every non-warning code as a
hard cancel. They describe the behavior required after that is fixed: record
the error, keep the last status the broker reported, and emit ``cancelledEvent``
only when the broker rejected or cancelled the order.

The socket bytes are the TWS API framing this client parses: a 4-byte
big-endian length, then NUL-terminated fields (``Client._onSocketHasData``),
message id 4 for errors (``Decoder.errorMsg``) and message id 3 for order
status (``Decoder`` handler 3).
"""

import struct

import ib_async as ibi

CLIENT_ID = 1
SERVER_VERSION = 176
PERM_ID = 44001

# Published TWS messages for a modify that did not cancel the prior order.
# https://www.interactivebrokers.com/docs/tws-api/doc/error/codes
MODIFY_FAILURES = [
    (104, "Can't modify a filled order."),
    (134, "Modify order failed."),
    (440, "Order modify failed. Algorithm cannot be modified."),
    (
        444,
        "Volatility Combo order is not yet acknowledged. Cannot submit changes at this time.",
    ),
    (
        10241,
        "Order Quantity is expressed in monetary terms. Modification is not supported via API. Please use desktop version to revise this order.",
    ),
    (
        10242,
        "Fractional-sized order cannot be modified via API. Please use desktop version to revise this order.",
    ),
    (10326, "OCA group revision is not allowed"),
    (10327, "OCA group type revision is not allowed"),
    (10341, "Parent order id cannot be modified"),
]

# Published TWS messages for a cancel the broker did not complete.
CANCEL_FAILURES = [
    (136, "This order cannot be cancelled."),
    (
        137,
        "VWAP orders can only be cancelled up to three minutes before the start time.",
    ),
    (
        161,
        f"Cancel attempted when order is not in a cancellable state. Order permId ={PERM_ID}",
    ),
    (
        10148,
        "OrderId {order_id} that needs to be cancelled can not be cancelled, state:Filled",
    ),
]


def _frame(*fields) -> bytes:
    payload = "".join(f"{field}\0" for field in fields).encode()
    return struct.pack(">I", len(payload)) + payload


def _push(ib: ibi.IB, *frames: bytes) -> None:
    ib.client._onSocketHasData(b"".join(frames))


def _order_status(
    order_id: int,
    status: str,
    *,
    filled: float = 0,
    remaining: float = 100,
    avg: float = 0,
) -> bytes:
    # Message id 3, no version field (Decoder registers orderStatus with skip=1).
    return _frame(
        3,
        order_id,
        status,
        filled,
        remaining,
        avg,
        PERM_ID,
        0,
        avg,
        CLIENT_ID,
        "",
        0,
    )


def _error(order_id: int, code: int, text: str, advanced: str = "") -> bytes:
    # Message id 4. Server version >= 166 appends advancedOrderRejectJson.
    return _frame(4, 2, order_id, code, text, advanced)


def _session() -> ibi.IB:
    ib = ibi.IB()
    ib.wrapper.clientId = CLIENT_ID
    ib.client.decoder.serverVersion = SERVER_VERSION
    # placeOrder/cancelOrder refuse to run until the API socket is up.
    # No transport is attached, so the outbound bytes are not written.
    ib.client.connState = ib.client.CONNECTED
    ib.client._apiReady = True
    return ib


def _watch(trade: ibi.Trade, ib: ibi.IB, *, replace: bool = True):
    cancelled: list[ibi.Trade] = []
    replacements: list[ibi.Trade] = []
    errors: list[tuple] = []

    def on_cancelled(done: ibi.Trade) -> None:
        cancelled.append(done)
        if not replace:
            return
        replacement = ibi.LimitOrder(
            done.order.action, done.order.totalQuantity, done.order.lmtPrice
        )
        replacement.orderId = done.order.orderId + 1000
        replacements.append(ib.placeOrder(done.contract, replacement))

    trade.cancelledEvent += on_cancelled
    ib.errorEvent += lambda *args: errors.append(args)
    return cancelled, replacements, errors


def _working_order(ib: ibi.IB, order_id: int, status: str = ibi.OrderStatus.Submitted):
    contract = ibi.Stock("SPY", "SMART", "USD")
    order = ibi.LimitOrder("BUY", 100, 500.0, orderId=order_id)
    trade = ib.placeOrder(contract, order)
    _push(ib, _order_status(order_id, status, remaining=100))
    assert trade.orderStatus.status == status
    assert not trade.isDone()
    return contract, order, trade


def _assert_still_live(
    ib: ibi.IB,
    trade: ibi.Trade,
    status: str,
    code: int,
    cancelled: list,
    replacements: list,
    errors: list,
    *,
    remaining: float = 100,
    filled: float = 0,
) -> None:
    assert trade.orderStatus.status == status
    assert trade.orderStatus.remaining == remaining
    assert trade.orderStatus.filled == filled
    assert not trade.isDone()
    assert trade.isActive()
    assert [open_trade.order.orderId for open_trade in ib.openTrades()] == [
        trade.order.orderId
    ]
    assert cancelled == []
    assert replacements == []
    assert any(entry.errorCode == code for entry in trade.log)
    assert any(args[0] == trade.order.orderId and args[1] == code for args in errors)


def test_refused_modify_same_packet_does_not_place_a_second_order():
    """A still-working status can follow the error in the same TCP packet.

    The cancel event is synchronous, so a replacement submitted from it is
    already live before that later status is applied.
    """
    ib = _session()
    _contract, order, trade = _working_order(ib, 7)
    cancelled, replacements, _errors = _watch(trade, ib)
    order.lmtPrice = 501.0
    ib.placeOrder(trade.contract, order)

    _push(
        ib,
        _error(7, 134, "Modify order failed."),
        _order_status(7, ibi.OrderStatus.Submitted, remaining=100),
    )

    assert cancelled == []
    assert replacements == []
    assert trade.orderStatus.status == ibi.OrderStatus.Submitted
    assert [item.order.orderId for item in ib.openTrades()] == [7]


def test_refused_cancel_same_packet_does_not_place_a_second_order():
    ib = _session()
    _contract, order, trade = _working_order(ib, 8)
    cancelled, replacements, _errors = _watch(trade, ib)
    ib.cancelOrder(order)
    assert trade.orderStatus.status == ibi.OrderStatus.PendingCancel

    _push(
        ib,
        _error(8, 136, "This order cannot be cancelled."),
        _order_status(8, ibi.OrderStatus.Submitted, remaining=100),
    )

    assert cancelled == []
    assert replacements == []
    assert trade.orderStatus.status == ibi.OrderStatus.Submitted
    assert [item.order.orderId for item in ib.openTrades()] == [8]


def test_modify_failures_keep_the_broker_status(code, text):
    ib = _session()
    _contract, order, trade = _working_order(ib, 20 + (code % 50))
    cancelled, replacements, errors = _watch(trade, ib)
    order.lmtPrice = 501.0
    ib.placeOrder(trade.contract, order)
    order_id = order.orderId

    _push(ib, _error(order_id, code, text, advanced='{"text":"modify rejected"}'))

    _assert_still_live(
        ib,
        trade,
        ibi.OrderStatus.Submitted,
        code,
        cancelled,
        replacements,
        errors,
    )
    assert trade.advancedError == '{"text":"modify rejected"}'


def test_cancel_failures_restore_the_broker_status(code, text):
    ib = _session()
    _contract, order, trade = _working_order(ib, 70 + (code % 40))
    cancelled, replacements, errors = _watch(trade, ib)
    ib.cancelOrder(order)
    assert trade.orderStatus.status == ibi.OrderStatus.PendingCancel
    order_id = order.orderId
    message = text.format(order_id=order_id)

    _push(ib, _error(order_id, code, message))

    _assert_still_live(
        ib,
        trade,
        ibi.OrderStatus.Submitted,
        code,
        cancelled,
        replacements,
        errors,
    )


def test_cancel_rejected_while_presubmitted_stays_presubmitted():
    ib = _session()
    _contract, order, trade = _working_order(ib, 11, ibi.OrderStatus.PreSubmitted)
    cancelled, replacements, errors = _watch(trade, ib)
    ib.cancelOrder(order)

    _push(
        ib,
        _error(
            order.orderId,
            161,
            f"Cancel attempted when order is not in a cancellable state. Order permId ={PERM_ID}",
        ),
    )

    _assert_still_live(
        ib,
        trade,
        ibi.OrderStatus.PreSubmitted,
        161,
        cancelled,
        replacements,
        errors,
    )


def test_modify_failed_before_acknowledgement_stays_pending():
    ib = _session()
    contract = ibi.Stock("SPY", "SMART", "USD")
    order = ibi.LimitOrder("BUY", 100, 500.0, orderId=12)
    trade = ib.placeOrder(contract, order)
    assert trade.orderStatus.status == ibi.OrderStatus.PendingSubmit
    cancelled, replacements, errors = _watch(trade, ib)
    order.lmtPrice = 501.0
    ib.placeOrder(contract, order)

    _push(ib, _error(12, 134, "Modify order failed."))

    _assert_still_live(
        ib,
        trade,
        ibi.OrderStatus.PendingSubmit,
        134,
        cancelled,
        replacements,
        errors,
        remaining=0,
    )


def test_broker_can_fill_after_refused_modify():
    ib = _session()
    _contract, order, trade = _working_order(ib, 13)
    cancelled, replacements, _errors = _watch(trade, ib)
    filled: list[ibi.Trade] = []
    trade.filledEvent += lambda done: filled.append(done)
    order.lmtPrice = 501.0
    ib.placeOrder(trade.contract, order)

    _push(ib, _error(13, 134, "Modify order failed."))
    _push(
        ib,
        _order_status(
            13, ibi.OrderStatus.Filled, filled=100, remaining=0, avg=500.25
        ),
    )

    assert cancelled == []
    assert replacements == []
    assert filled == [trade]
    assert trade.orderStatus.status == ibi.OrderStatus.Filled
    assert trade.isDone()
    assert ib.openTrades() == []


def test_broker_cancel_after_refused_cancel_emits_once():
    ib = _session()
    _contract, order, trade = _working_order(ib, 14)
    cancelled, replacements, _errors = _watch(trade, ib)
    ib.cancelOrder(order)

    _push(ib, _error(14, 136, "This order cannot be cancelled."))
    assert cancelled == []
    _push(ib, _order_status(14, ibi.OrderStatus.Cancelled, remaining=100))

    assert cancelled == [trade]
    assert [item.order.orderId for item in replacements] == [1014]
    assert trade.orderStatus.status == ibi.OrderStatus.Cancelled
    assert [item.order.orderId for item in ib.openTrades()] == [1014]


def test_broker_cancelled_error_marks_the_order_cancelled():
    ib = _session()
    _contract, _order, trade = _working_order(ib, 15)
    cancelled, replacements, errors = _watch(trade, ib)

    _push(ib, _error(15, 202, "Order cancelled - Reason:"))

    assert trade.orderStatus.status == ibi.OrderStatus.Cancelled
    assert trade.isDone()
    assert cancelled == [trade]
    assert [item.order.orderId for item in replacements] == [1015]
    assert ib.openTrades() == replacements
    assert any(entry.errorCode == 202 for entry in trade.log)
    assert any(args[1] == 202 for args in errors)


def test_new_order_reject_marks_the_order_cancelled():
    ib = _session()
    contract = ibi.Stock("SPY", "SMART", "USD")
    order = ibi.LimitOrder("BUY", 100, 500.0, orderId=16)
    trade = ib.placeOrder(contract, order)
    cancelled, _replacements, _errors = _watch(trade, ib, replace=False)

    _push(ib, _error(16, 201, "Order rejected - Reason:"))

    assert trade.orderStatus.status == ibi.OrderStatus.Cancelled
    assert trade.isDone()
    assert cancelled == [trade]
    assert ib.openTrades() == []


def test_submit_new_order_failed_marks_the_order_cancelled():
    ib = _session()
    contract = ibi.Stock("SPY", "SMART", "USD")
    order = ibi.LimitOrder("BUY", 100, 500.0, orderId=17)
    trade = ib.placeOrder(contract, order)
    cancelled, _replacements, _errors = _watch(trade, ib, replace=False)

    _push(ib, _error(17, 133, "Submit new order failed."))

    assert trade.orderStatus.status == ibi.OrderStatus.Cancelled
    assert cancelled == [trade]


def test_confirmed_cancel_error_while_pending_cancel():
    ib = _session()
    _contract, order, trade = _working_order(ib, 18)
    cancelled, _replacements, _errors = _watch(trade, ib, replace=False)
    ib.cancelOrder(order)

    _push(ib, _error(18, 202, "Order cancelled - Reason:"))

    assert trade.orderStatus.status == ibi.OrderStatus.Cancelled
    assert cancelled == [trade]
    assert ib.openTrades() == []


def test_modify_warning_does_not_cancel():
    ib = _session()
    _contract, order, trade = _working_order(ib, 19)
    cancelled, replacements, _errors = _watch(trade, ib)
    order.orderType = "STP"
    ib.placeOrder(trade.contract, order)

    _push(
        ib,
        _error(19, 329, "Order modify failed. Cannot change to the new order type."),
    )

    assert trade.orderStatus.status == ibi.OrderStatus.ValidationError
    assert not trade.isDone()
    assert cancelled == []
    assert replacements == []
    assert trade in ib.openTrades()


def test_still_processing_warning_does_not_cancel():
    ib = _session()
    _contract, order, trade = _working_order(ib, 21)
    cancelled, replacements, _errors = _watch(trade, ib)
    order.lmtPrice = 501.0
    ib.placeOrder(trade.contract, order)

    _push(
        ib,
        _error(21, 2102, "Unable to modify this order as it is still being processed."),
    )

    assert trade.orderStatus.status == ibi.OrderStatus.ValidationError
    assert not trade.isDone()
    assert cancelled == []
    assert replacements == []


def pytest_generate_tests(metafunc):
    if metafunc.definition.name == "test_modify_failures_keep_the_broker_status":
        metafunc.parametrize("code,text", MODIFY_FAILURES)
    elif metafunc.definition.name == "test_cancel_failures_restore_the_broker_status":
        metafunc.parametrize("code,text", CANCEL_FAILURES)
