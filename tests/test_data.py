"""Checks that the data is loaded correctly before any analysis is trusted."""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from load import ATT_ALL, england_monthly, load_all  # noqa: E402


@pytest.fixture(scope="module")
def loaded():
    data, totals = load_all()
    return data, totals, england_monthly(data)


def test_one_row_per_provider_per_month(loaded):
    data, _, _ = loaded
    assert not data.duplicated(["month", "org_code"]).any()


def test_no_total_rows_left_in_provider_data(loaded):
    data, _, _ = loaded
    assert not data["org_code"].str.upper().eq("TOTAL").any()
    assert not data["period_raw"].astype(str).str.strip().str.upper().eq("TOTAL").any()


def test_months_are_consecutive(loaded):
    _, _, eng = loaded
    gaps = eng.index.to_series().diff().dropna().dt.days
    assert gaps.between(28, 31).all()


@pytest.mark.parametrize("column", ["attendances", "dta_12h_plus"])
def test_rebuilt_england_totals_match_published_totals(loaded, column):
    """Every month's published TOTAL row must equal the sum of the providers."""
    _, totals, eng = loaded
    assert totals, "no published TOTAL rows found"
    for month, row in totals.items():
        published = (sum(float(row[c]) for c in ATT_ALL) if column == "attendances"
                     else float(row["dta_12h_plus"]))
        assert eng.loc[month, column] == pytest.approx(published), month.date()


def test_performance_is_a_valid_percentage(loaded):
    _, _, eng = loaded
    assert eng["pct_4h_all"].between(0, 100).all()
    assert eng["pct_4h_t1"].between(0, 100).all()
