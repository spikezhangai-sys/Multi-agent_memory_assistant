from datetime import UTC, datetime, timedelta

import pytest

from driftscope.retrieval.rule_time_parser import RuleBasedQueryTimeParser


@pytest.fixture
def now() -> datetime:
    return datetime(2026, 5, 15, 12, 0, tzinfo=UTC)


@pytest.fixture
def parser() -> RuleBasedQueryTimeParser:
    return RuleBasedQueryTimeParser()


def _delta_days(parser, query: str, now: datetime) -> float:
    hint = parser.parse(query=query, now=now)
    assert hint is not None, f"expected a hint for {query!r}"
    return (now - hint.center).total_seconds() / 86400.0


def test_yesterday(parser, now):
    assert _delta_days(parser, "What did I eat yesterday?", now) == pytest.approx(1.0)


def test_today(parser, now):
    assert _delta_days(parser, "What's on my plate today?", now) == pytest.approx(0.0)


def test_tomorrow(parser, now):
    assert _delta_days(parser, "What's tomorrow's schedule?", now) == pytest.approx(-1.0)


def test_last_week(parser, now):
    assert _delta_days(parser, "Recap last week's meetings.", now) == pytest.approx(7.0)


def test_last_month(parser, now):
    assert _delta_days(parser, "Spending last month?", now) == pytest.approx(30.0)


def test_last_year(parser, now):
    assert _delta_days(parser, "Where did I travel last year?", now) == pytest.approx(365.0)


def test_n_days_ago(parser, now):
    assert _delta_days(parser, "Where was I 5 days ago?", now) == pytest.approx(5.0)


def test_n_weeks_ago(parser, now):
    assert _delta_days(parser, "What did I do 3 weeks ago?", now) == pytest.approx(21.0)


def test_n_months_ago(parser, now):
    assert _delta_days(parser, "What did I buy 2 months ago?", now) == pytest.approx(60.0)


def test_a_couple_of_days_ago(parser, now):
    assert _delta_days(parser, "Note from a couple of days ago", now) == pytest.approx(2.0)


def test_a_week_ago(parser, now):
    assert _delta_days(parser, "What did I post a week ago?", now) == pytest.approx(7.0)


def test_a_month_ago(parser, now):
    assert _delta_days(parser, "Where did I dine a month ago?", now) == pytest.approx(30.0)


def test_a_year_ago(parser, now):
    assert _delta_days(parser, "Vacation a year ago?", now) == pytest.approx(365.0)


def test_recently_uses_wide_window(parser, now):
    # mempalace tunes "recently" as offset=14, half_window=14 — center 14d back, span 28d.
    assert _delta_days(parser, "What have I been working on recently?", now) == pytest.approx(14.0)
    hint = parser.parse(query="What have I been working on recently?", now=now)
    assert (hint.end - hint.start) == timedelta(days=28)


def test_case_insensitive(parser, now):
    assert _delta_days(parser, "WHAT DID I EAT YESTERDAY?", now) == pytest.approx(1.0)


def test_no_temporal_phrase_returns_none(parser, now):
    assert parser.parse(query="How tall is the Eiffel Tower?", now=now) is None


def test_empty_query_returns_none(parser, now):
    assert parser.parse(query="", now=now) is None
    assert parser.parse(query="   ", now=now) is None


def test_n_ago_takes_priority_over_unit_phrase(parser, now):
    # "5 days ago" should beat the looser "yesterday" path; both shouldn't fire.
    assert _delta_days(parser, "5 days ago, was that yesterday?", now) == pytest.approx(5.0)


def test_window_widths_match_unit(parser, now):
    yesterday = parser.parse(query="yesterday", now=now)
    last_week = parser.parse(query="last week", now=now)
    last_month = parser.parse(query="last month", now=now)
    assert (yesterday.end - yesterday.start) == timedelta(days=2)
    assert (last_week.end - last_week.start) == timedelta(days=6)
    assert (last_month.end - last_month.start) == timedelta(days=14)
