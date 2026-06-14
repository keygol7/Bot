from bot.timeutil import parse_iso8601


def test_parses_z_suffix():
    assert parse_iso8601("2026-09-27T13:00:00Z") is not None


def test_parses_offset():
    a = parse_iso8601("2026-09-27T13:00:00+00:00")
    b = parse_iso8601("2026-09-27T13:00:00Z")
    assert a == b


def test_two_dates_far_apart():
    jun = parse_iso8601("2026-06-16T21:40:00Z")
    sep = parse_iso8601("2026-09-27T13:00:00Z")
    assert abs(sep - jun) > 30 * 86400   # ~100 days -> clearly different events


def test_none_and_garbage():
    assert parse_iso8601(None) is None
    assert parse_iso8601("") is None
    assert parse_iso8601("not a date") is None
