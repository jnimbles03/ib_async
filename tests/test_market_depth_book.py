"""Market-depth books must follow row insert, update, and delete.

A depth subscription receives ``updateMktDepth`` (message 12) or
``updateMktDepthL2`` (message 13). Operation 0 inserts a level at that row
and shifts every existing row at that position and below up by one.
Operation 1 replaces the level already at that row. Operation 2 removes the
row and shifts every following row down by one so the positions stay
contiguous. ``ticker.domBids`` and ``ticker.domAsks`` are those rows in
position order, and the position dicts use the same keys.

These tests fail until that bookkeeping is in place.
"""

import struct

from ib_async import IB, DOMLevel, Stock
from ib_async.client import Client

ASK = 0
BID = 1
INSERT = 0
UPDATE = 1
DELETE = 2


def _num(value: float) -> str:
    return format(float(value), ".10g")


def _frame(*payloads: bytes) -> bytes:
    return b"".join(struct.pack(">I", len(payload)) + payload for payload in payloads)


def _depth(req_id, position, operation, side, price, size) -> bytes:
    fields = [
        "12",
        "1",
        str(req_id),
        str(position),
        str(operation),
        str(side),
        _num(price),
        _num(size),
        "",
    ]
    return "\0".join(fields).encode()


def _depth_l2(
    req_id, position, market_maker, operation, side, price, size, is_smart
) -> bytes:
    fields = [
        "13",
        "1",
        str(req_id),
        str(position),
        market_maker,
        str(operation),
        str(side),
        _num(price),
        _num(size),
        "1" if is_smart else "0",
        "",
    ]
    return "\0".join(fields).encode()


def _level(price, size, market_maker="") -> DOMLevel:
    return DOMLevel(float(price), float(size), market_maker)


class DepthSession:
    """Subscribe with ``reqMktDepth`` and deliver gateway messages on the socket."""

    def __init__(self, symbol: str, con_id: int, *, is_smart: bool = False):
        self.ib = IB()
        self.ib.client.connState = Client.CONNECTED
        self.ib.client._apiReady = True
        self.ib.client._serverVersion = Client.MaxClientVersion
        self.ib.client._reqIdSeq = 100 + con_id
        self.sent = []
        self.ib.client.send = self._capture_send
        self.contract = Stock(symbol, "ISLAND", "USD", conId=con_id)
        self.ticker = self.ib.reqMktDepth(
            self.contract, numRows=10, isSmartDepth=is_smart
        )
        self.req_id = self.ib.wrapper.ticker2ReqId["mktDepth"][self.ticker]
        self.published = []
        self.ticker.updateEvent.connect(self._on_update, keep_ref=True)
        assert self.sent and self.sent[0][0] == 10
        assert self.sent[0][2] == self.req_id

    def _capture_send(self, *fields, makeEmpty=True):
        self.sent.append(fields)

    def _on_update(self, ticker):
        self.published.append((list(ticker.domBids), list(ticker.domAsks)))

    def feed(self, *payloads: bytes):
        self.ib.client._onSocketHasData(_frame(*payloads))

    def depth(self, position, operation, side, price, size):
        self.feed(_depth(self.req_id, position, operation, side, price, size))

    def depth_l2(self, position, market_maker, operation, side, price, size, is_smart):
        self.feed(
            _depth_l2(
                self.req_id,
                position,
                market_maker,
                operation,
                side,
                price,
                size,
                is_smart,
            )
        )


def _assert_side(session: DepthSession, side: int, levels: list[DOMLevel]):
    if side == BID:
        book = session.ticker.domBids
        mapping = session.ticker.domBidsDict
        published = session.published[-1][0]
    else:
        book = session.ticker.domAsks
        mapping = session.ticker.domAsksDict
        published = session.published[-1][1]

    assert book == levels
    assert published == levels
    assert set(mapping) == set(range(len(levels)))
    assert [mapping[i] for i in range(len(levels))] == levels


def test_insert_at_occupied_bid_row_shifts_the_rows_below():
    session = DepthSession("QQQ", 1)
    session.depth(0, INSERT, BID, 50, 1)
    session.depth(1, INSERT, BID, 40, 2)
    _assert_side(session, BID, [_level(50, 1), _level(40, 2)])

    # A new row 1 must push the previous row 1 down. Row order is not price order.
    session.depth(1, INSERT, BID, 60, 3)
    _assert_side(session, BID, [_level(50, 1), _level(60, 3), _level(40, 2)])
    assert session.ticker.domAsks == []


def test_insert_at_bid_row_zero_keeps_the_previous_level():
    session = DepthSession("QQQ", 2)
    session.depth(0, INSERT, BID, 100, 10)
    session.depth(1, INSERT, BID, 99, 20)
    session.depth(0, INSERT, BID, 101, 5)
    _assert_side(
        session,
        BID,
        [_level(101, 5), _level(100, 10), _level(99, 20)],
    )


def test_delete_shifts_following_bids_and_later_insert_stays_in_row_order():
    session = DepthSession("QQQ", 3)
    session.depth(0, INSERT, BID, 100, 10)
    session.depth(1, INSERT, BID, 99, 20)
    session.depth(0, INSERT, BID, 101, 5)
    session.depth(0, DELETE, BID, 0, 0)
    _assert_side(session, BID, [_level(100, 10), _level(99, 20)])

    session.depth(0, INSERT, BID, 102, 7)
    _assert_side(
        session,
        BID,
        [_level(102, 7), _level(100, 10), _level(99, 20)],
    )

    # The row that slid down to position 1 is the one an update at 1 must change.
    session.depth(1, UPDATE, BID, 100, 15)
    _assert_side(
        session,
        BID,
        [_level(102, 7), _level(100, 15), _level(99, 20)],
    )


def test_middle_delete_closes_the_hole_before_the_next_update():
    session = DepthSession("QQQ", 4)
    session.depth(0, INSERT, BID, 50, 1)
    session.depth(1, INSERT, BID, 40, 2)
    session.depth(2, INSERT, BID, 30, 3)
    session.depth(1, DELETE, BID, 0, 0)
    _assert_side(session, BID, [_level(50, 1), _level(30, 3)])

    session.depth(1, UPDATE, BID, 30, 9)
    _assert_side(session, BID, [_level(50, 1), _level(30, 9)])


def test_ask_l2_insert_and_delete_shift_rows_and_market_makers():
    session = DepthSession("QQQ", 5, is_smart=False)
    session.depth_l2(0, "ISLAND", INSERT, ASK, 10, 1, False)
    session.depth_l2(1, "NASDAQ", INSERT, ASK, 12, 2, False)
    session.depth_l2(0, "ARCA", INSERT, ASK, 11, 4, False)
    _assert_side(
        session,
        ASK,
        [_level(11, 4, "ARCA"), _level(10, 1, "ISLAND"), _level(12, 2, "NASDAQ")],
    )

    session.depth_l2(0, "", DELETE, ASK, 0, 0, False)
    _assert_side(
        session,
        ASK,
        [_level(10, 1, "ISLAND"), _level(12, 2, "NASDAQ")],
    )

    session.depth_l2(0, "BATS", INSERT, ASK, 9, 8, False)
    _assert_side(
        session,
        ASK,
        [
            _level(9, 8, "BATS"),
            _level(10, 1, "ISLAND"),
            _level(12, 2, "NASDAQ"),
        ],
    )
    assert session.ticker.domBids == []


def test_smart_depth_insert_and_delete_use_the_same_row_shifts():
    session = DepthSession("SPY", 6, is_smart=True)
    session.depth_l2(0, "SMART", INSERT, BID, 70, 1, True)
    session.depth_l2(1, "SMART", INSERT, BID, 69, 2, True)
    session.depth_l2(0, "ISLAND", INSERT, BID, 71, 3, True)
    _assert_side(
        session,
        BID,
        [_level(71, 3, "ISLAND"), _level(70, 1, "SMART"), _level(69, 2, "SMART")],
    )
    session.depth_l2(1, "", DELETE, BID, 0, 0, True)
    _assert_side(session, BID, [_level(71, 3, "ISLAND"), _level(69, 2, "SMART")])


def test_update_replaces_one_row_and_leaves_the_others():
    session = DepthSession("QQQ", 7)
    session.depth(0, INSERT, ASK, 10, 1)
    session.depth(1, INSERT, ASK, 11, 2)
    session.depth(0, UPDATE, ASK, 10, 9)
    _assert_side(session, ASK, [_level(10, 9), _level(11, 2)])


def test_insert_at_the_end_appends():
    session = DepthSession("QQQ", 8)
    session.depth(0, INSERT, BID, 5, 1)
    session.depth(1, INSERT, BID, 4, 2)
    _assert_side(session, BID, [_level(5, 1), _level(4, 2)])


def test_delete_of_a_missing_row_leaves_the_book():
    session = DepthSession("QQQ", 9)
    session.depth(0, INSERT, BID, 5, 1)
    session.depth(1, INSERT, BID, 4, 2)
    session.depth(5, DELETE, BID, 0, 0)
    _assert_side(session, BID, [_level(5, 1), _level(4, 2)])


def test_bid_and_ask_books_stay_independent():
    session = DepthSession("QQQ", 10)
    session.depth(0, INSERT, BID, 20, 1)
    session.depth(1, INSERT, BID, 19, 1)
    session.depth(0, INSERT, ASK, 21, 3)
    session.depth(0, INSERT, ASK, 22, 4)
    _assert_side(session, BID, [_level(20, 1), _level(19, 1)])
    _assert_side(
        session,
        ASK,
        [_level(22, 4), _level(21, 3)],
    )


def test_two_depth_subscriptions_do_not_share_a_book():
    first = DepthSession("QQQ", 11)
    second = DepthSession("SPY", 12)
    first.depth(0, INSERT, BID, 100, 1)
    first.depth(1, INSERT, BID, 99, 1)
    second.depth(0, INSERT, BID, 50, 8)
    second.depth(0, INSERT, BID, 51, 9)
    _assert_side(first, BID, [_level(100, 1), _level(99, 1)])
    _assert_side(second, BID, [_level(51, 9), _level(50, 8)])


def test_several_depth_messages_in_one_packet_apply_in_order():
    session = DepthSession("QQQ", 13)
    req_id = session.req_id
    session.feed(
        _depth(req_id, 0, INSERT, BID, 100, 10),
        _depth(req_id, 1, INSERT, BID, 99, 20),
        _depth(req_id, 0, INSERT, BID, 101, 5),
        _depth(req_id, 0, DELETE, BID, 0, 0),
        _depth(req_id, 0, INSERT, BID, 102, 7),
    )
    _assert_side(
        session,
        BID,
        [_level(102, 7), _level(100, 10), _level(99, 20)],
    )
