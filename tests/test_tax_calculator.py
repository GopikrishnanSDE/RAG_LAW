"""Hand-computed cases for the s.202 / s.156 / s.19 / s.58 calculator."""

import pytest

from rag_core.tax_calculator import TaxInputs, calculate, rebate_156, rupees, slab_tax


@pytest.mark.parametrize(
    "income, expected",
    [
        (400_000, 0),
        (800_000, 20_000),  # 5% of 4L
        (1_200_000, 60_000),  # + 10% of 4L
        (2_000_000, 200_000),  # + 15% of 4L + 20% of 4L
        (2_400_000, 300_000),  # + 25% of 4L
        (3_000_000, 480_000),  # + 30% of 6L
    ],
)
def test_slab_tax(income, expected):
    assert slab_tax(income)[0] == pytest.approx(expected)


def test_rebate_wipes_out_tax_up_to_12_lakh():
    assert rebate_156(1_200_000, 60_000) == 60_000


def test_marginal_relief_just_above_12_lakh():
    # Tax on 12.1L is 61,500, but can't exceed the 10,000 excess over 12L.
    tax, _ = slab_tax(1_210_000)
    assert tax == pytest.approx(61_500)
    assert tax - rebate_156(1_210_000, tax) == pytest.approx(10_000)


def test_no_marginal_relief_well_above_12_lakh():
    assert rebate_156(2_000_000, 200_000) == 0


def test_business_profit_20_lakh():
    r = calculate(TaxInputs(business_profit=2_000_000))
    assert r.supported
    assert r.total_income == 2_000_000
    assert r.tax == pytest.approx(200_000)
    # Same figure read as turnover: 8% presumptive = 1.6L -> nil tax
    assert r.alternatives[0]["total_income"] == pytest.approx(160_000)
    assert r.alternatives[0]["tax"] == 0


def test_business_turnover_uses_presumptive_8_percent():
    r = calculate(TaxInputs(person_type="individual", business_turnover=15_000_000))
    assert r.total_income == pytest.approx(1_200_000)  # 8% of 1.5 crore
    assert r.tax == 0  # rebate up to 12L
    assert r.alternatives[0]["total_income"] == pytest.approx(900_000)  # 6%


def test_business_turnover_mixed_digital():
    r = calculate(TaxInputs(business_turnover=10_000_000, digital_share=0.5))
    assert r.total_income == pytest.approx(700_000)  # 3L + 4L


def test_turnover_above_presumptive_limit_is_not_guessed():
    r = calculate(TaxInputs(business_turnover=25_000_000))
    assert not r.supported
    assert "s.58(2)" in r.reason


def test_low_cash_raises_limit_to_3_crore():
    r = calculate(TaxInputs(business_turnover=25_000_000, digital_share=1.0))
    assert r.supported
    assert r.total_income == pytest.approx(1_500_000)  # 6%


def test_salary_15_lakh():
    r = calculate(TaxInputs(salary=1_500_000))
    assert r.total_income == pytest.approx(1_425_000)  # less 75k standard deduction
    # 20k + 40k + 15% of 2.25L = 93,750
    assert r.tax == pytest.approx(93_750)


def test_salary_12_75_lakh_is_nil():
    r = calculate(TaxInputs(salary=1_275_000))
    assert r.total_income == pytest.approx(1_200_000)
    assert r.tax == 0


def test_profession_50_percent():
    r = calculate(TaxInputs(professional_receipts=4_000_000))
    assert r.total_income == pytest.approx(2_000_000)
    assert r.tax == pytest.approx(200_000)


def test_huf_gets_no_individual_rebate():
    r = calculate(TaxInputs(person_type="huf", business_profit=1_000_000))
    assert r.tax == pytest.approx(40_000)  # 20k + 10% of 2L, no s.156


def test_company_not_supported():
    assert not calculate(TaxInputs(person_type="company", business_profit=1_000_000)).supported


def test_old_regime_not_supported():
    r = calculate(TaxInputs(regime="old", salary=1_000_000))
    assert not r.supported
    assert "Finance Act" in r.reason


def test_rupee_grouping():
    assert rupees(2_000_000) == "₹20,00,000"
    assert rupees(93_750) == "₹93,750"
    assert rupees(-75_000) == "-₹75,000"
    assert rupees(500) == "₹500"


# --- amount grounding (query_analysis) -----------------------------------

from rag_core.query_analysis import _grounded, amounts_in  # noqa: E402


def test_amounts_in_parses_indian_units():
    assert amounts_in("my business earns 20 lakh") == [2_000_000]
    assert amounts_in("turnover 1.5 crore") == [15_000_000]
    assert amounts_in("salary ₹15,00,000") == [1_500_000]
    assert amounts_in("I sold my flat after 3 years") == [3]


def test_invented_amount_is_rejected():
    assert not _grounded(TaxInputs(other_income=2_000_000), "I sold my flat after 3 years, how is the gain taxed?")


def test_stated_amount_is_accepted():
    assert _grounded(TaxInputs(business_profit=2_000_000), "If a business earns 20 lakh rupees, how much tax?")


# --- one amount, one field (query_analysis) --------------------------------

from rag_core.query_analysis import _resolve_fields  # noqa: E402


def _fields(inp):
    return {k: v for k, v in inp.__dict__.items() if v is not None and k != "person_type"}


def test_duplicate_amount_goes_to_profession():
    q = "I'm an IT consultant with 30 lakh of gross receipts. How much tax do I owe?"
    inp = _resolve_fields(TaxInputs(business_turnover=3e6, business_profit=3e6), q)
    assert _fields(inp) == {"professional_receipts": 3e6}


def test_duplicate_amount_goes_to_turnover():
    q = "My shop turnover is 1.5 crore. How much tax?"
    inp = _resolve_fields(TaxInputs(business_turnover=1.5e7, business_profit=1.5e7), q)
    assert _fields(inp) == {"business_turnover": 1.5e7}


def test_salary_wins_for_salaried():
    q = "My salary is 15 lakh, how much tax?"
    inp = _resolve_fields(TaxInputs(salary=1.5e6, other_income=1.5e6), q)
    assert _fields(inp) == {"salary": 1.5e6}


def test_business_earns_stays_profit():
    q = "If a business earns 20 lakh rupees, how much tax should I pay?"
    inp = _resolve_fields(TaxInputs(business_profit=2e6), q)
    assert _fields(inp) == {"business_profit": 2e6}


def test_lowercase_it_is_not_a_profession():
    q = "My business earns 20 lakh, how is it taxed and how much do I pay?"
    inp = _resolve_fields(TaxInputs(business_profit=2e6, other_income=2e6), q)
    assert _fields(inp) == {"business_profit": 2e6}
