"""Deterministic income-tax computation from the Act's own rates.

Retrieval answers "what does the law say"; it can't be trusted with "how
much do I owe". A 7B model doing slab arithmetic invents numbers (it once
answered a ₹20 lakh question with a "₹50 lakh" threshold). So the LLM only
*extracts* the facts (see query_analysis.py) and this module does the maths,
with every step citing the provision it applies.

Everything here is transcribed from the corpus text, not from memory:
  - s.202(1)  new tax regime slab table (individual, HUF, AOP, BOI, AJP)
  - s.156(2)  rebate for resident individuals: nil tax up to ₹12 lakh,
              marginal relief just above it
  - s.19(1)   salary standard deduction, ₹75,000 under s.202(1)
  - s.58(2)   presumptive profits: business (Table Sl. 1) and specified
              professions (Table Sl. 3)

Deliberately out of scope, because the Act doesn't contain the numbers:
  - old-regime slab rates, surcharge and Health & Education Cess — all set by
    each year's Finance Act
  - firm and company rates, and special-rate income (capital gains, lottery)
The calculator returns supported=False with the reason instead of guessing.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# s.202(1) Table: (upper bound of slab, rate). None = no upper bound.
NEW_REGIME_SLABS = [
    (400_000, 0.00),
    (800_000, 0.05),
    (1_200_000, 0.10),
    (1_600_000, 0.15),
    (2_000_000, 0.20),
    (2_400_000, 0.25),
    (None, 0.30),
]

REBATE_INCOME_LIMIT = 1_200_000  # s.156(2)(a)
REBATE_MAX = 60_000  # s.156(2)(a)
STANDARD_DEDUCTION = 75_000  # s.19(1) Table Sl. 2(a)

# s.58(2) Table Sl. 1 — business, "eligible assessee"
BUSINESS_TURNOVER_LIMIT = 20_000_000  # ₹2 crore
BUSINESS_TURNOVER_LIMIT_LOW_CASH = 30_000_000  # ₹3 crore if cash <= 5%
BUSINESS_RATE_DIGITAL = 0.06
BUSINESS_RATE_OTHER = 0.08

# s.58(2) Table Sl. 3 — specified profession, "specified assessee"
PROFESSION_RECEIPTS_LIMIT = 5_000_000  # ₹50 lakh
PROFESSION_RECEIPTS_LIMIT_LOW_CASH = 7_500_000  # ₹75 lakh if cash <= 5%
PROFESSION_RATE = 0.50

SUPPORTED_PERSONS = {"individual", "huf"}

NOTES = [
    "Excludes surcharge and Health & Education Cess: those are levied by the "
    "annual Finance Act, not by the Income-tax Act, 2025.",
    "Covers income taxed at the normal slab rates only. Capital gains and other "
    "special-rate income are taxed separately (e.g. s.196, s.197).",
]


def rupees(x: float) -> str:
    """Indian digit grouping: 2000000 -> '₹20,00,000'."""
    n = round(x)
    sign = "-" if n < 0 else ""
    s = str(abs(n))
    if len(s) > 3:
        head, tail = s[:-3], s[-3:]
        groups = []
        while len(head) > 2:
            groups.insert(0, head[-2:])
            head = head[:-2]
        if head:
            groups.insert(0, head)
        s = ",".join(groups) + "," + tail
    return f"{sign}₹{s}"


@dataclass
class TaxInputs:
    person_type: str | None = None  # individual | huf | firm | company | other
    resident: bool | None = None
    regime: str | None = None  # new | old
    salary: float | None = None
    business_turnover: float | None = None
    business_profit: float | None = None
    professional_receipts: float | None = None
    digital_share: float | None = None  # share of receipts via banking/online, 0..1
    other_income: float | None = None


@dataclass
class TaxResult:
    supported: bool
    reason: str = ""
    assumptions: list[str] = field(default_factory=list)
    steps: list[dict] = field(default_factory=list)
    total_income: float = 0
    tax: float = 0
    alternatives: list[dict] = field(default_factory=list)
    notes: list[str] = field(default_factory=lambda: list(NOTES))
    sections: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def slab_tax(total_income: float) -> tuple[float, list[dict]]:
    """Tax under the s.202(1) table, with one step per slab that applies."""
    tax, lower, steps = 0.0, 0, []
    for upper, rate in NEW_REGIME_SLABS:
        if total_income <= lower:
            break
        top = total_income if upper is None else min(total_income, upper)
        part = (top - lower) * rate
        if rate:
            steps.append(
                {
                    "label": f"{int(rate * 100)}% on {rupees(lower + 1)} – {rupees(top)}",
                    "amount": part,
                    "citation": "s.202(1)",
                }
            )
        tax += part
        if upper is None or total_income <= upper:
            break
        lower = upper
    return tax, steps


def rebate_156(total_income: float, tax: float) -> float:
    """s.156(2) rebate for a resident individual under s.202(1)."""
    if total_income <= REBATE_INCOME_LIMIT:
        return min(tax, REBATE_MAX)  # (a)
    excess = total_income - REBATE_INCOME_LIMIT
    if tax > excess:
        return tax - excess  # (b) marginal relief: tax can't exceed the excess
    return 0.0


def _presumptive_business(turnover: float, digital_share: float | None) -> tuple[float | None, str]:
    limit = (
        BUSINESS_TURNOVER_LIMIT_LOW_CASH
        if digital_share is not None and digital_share >= 0.95
        else BUSINESS_TURNOVER_LIMIT
    )
    if turnover > limit:
        return None, (
            f"Turnover of {rupees(turnover)} is above the s.58(2) limit of {rupees(limit)}, so "
            "presumptive taxation isn't available; tax depends on actual profit."
        )
    if digital_share is None:
        return turnover * BUSINESS_RATE_OTHER, "8% of turnover (receipts assumed not via banking/online)"
    profit = turnover * digital_share * BUSINESS_RATE_DIGITAL + turnover * (1 - digital_share) * BUSINESS_RATE_OTHER
    return profit, f"6% of digital + 8% of other turnover ({int(digital_share * 100)}% digital)"


def _compute(inp: TaxInputs, total_income_parts: list[tuple[str, float, str]]) -> tuple[float, float, list[dict]]:
    """Slab tax and rebate on a list of (label, amount, citation) income parts."""
    steps = [{"label": label, "amount": amt, "citation": cite} for label, amt, cite in total_income_parts]
    total = sum(amt for _, amt, _ in total_income_parts)
    steps.append({"label": "Total income", "amount": total, "citation": ""})

    tax, slab_steps = slab_tax(total)
    steps += slab_steps
    steps.append({"label": "Tax at slab rates", "amount": tax, "citation": "s.202(1)"})

    if inp.person_type == "individual" and inp.resident is not False:
        rebate = rebate_156(total, tax)
        if rebate:
            label = (
                "Rebate (income up to ₹12,00,000)"
                if total <= REBATE_INCOME_LIMIT
                else "Rebate (marginal relief above ₹12,00,000)"
            )
            steps.append({"label": label, "amount": -rebate, "citation": "s.156(2)"})
            tax -= rebate
    return total, tax, steps


def calculate(inp: TaxInputs) -> TaxResult:
    assumptions: list[str] = []

    person = inp.person_type or "individual"
    if not inp.person_type:
        assumptions.append("Assumed you are an individual.")
    if person not in SUPPORTED_PERSONS:
        return TaxResult(
            supported=False,
            reason=(
                f"The calculator covers individuals and HUFs under the s.202 new tax regime. "
                f"Rates for a {person} aren't computed here."
            ),
        )
    inp.person_type = person

    if inp.regime == "old":
        return TaxResult(
            supported=False,
            reason=(
                "Old-regime slab rates are set by each year's Finance Act, which isn't part of "
                "this corpus, so they can't be computed from the Act."
            ),
        )
    if inp.regime is None:
        assumptions.append("Assumed the default new tax regime under s.202(1).")
    if person == "individual" and inp.resident is None:
        assumptions.append("Assumed you are resident in India (needed for the s.156 rebate).")

    parts: list[tuple[str, float, str]] = []
    sections = ["202"]
    alternatives: list[dict] = []

    if inp.salary:
        sd = min(STANDARD_DEDUCTION, inp.salary)
        parts.append(("Salary", inp.salary, ""))
        parts.append(("Standard deduction", -sd, "s.19(1)"))
        sections.append("19")

    if inp.business_profit:
        parts.append(("Business profit", inp.business_profit, ""))
        # "Earns" is ambiguous: show what presumptive taxation would give if
        # the same figure were turnover instead of profit.
        if not inp.business_turnover:
            alt_profit, how = _presumptive_business(inp.business_profit, inp.digital_share)
            if alt_profit is not None:
                alt_parts = [p for p in parts if p[0] != "Business profit"] + [
                    ("Presumptive business profit", alt_profit, "s.58(2)")
                ]
                alt_total, alt_tax, _ = _compute(TaxInputs(**{**inp.__dict__}), alt_parts)
                alternatives.append(
                    {
                        "label": f"If {rupees(inp.business_profit)} is turnover, not profit "
                        f"(presumptive, {how})",
                        "total_income": alt_total,
                        "tax": alt_tax,
                        "citation": "s.58(2)",
                    }
                )
                sections.append("58")
    elif inp.business_turnover:
        profit, how = _presumptive_business(inp.business_turnover, inp.digital_share)
        if profit is None:
            return TaxResult(supported=False, reason=how, assumptions=assumptions)
        parts.append((f"Presumptive business profit: {how}", profit, "s.58(2)"))
        sections.append("58")
        if inp.digital_share is None:
            alt = inp.business_turnover * BUSINESS_RATE_DIGITAL
            alt_parts = [p for p in parts if not p[0].startswith("Presumptive")] + [
                ("Presumptive business profit (6%)", alt, "s.58(2)")
            ]
            alt_total, alt_tax, _ = _compute(TaxInputs(**{**inp.__dict__}), alt_parts)
            alternatives.append(
                {
                    "label": "If all receipts came via banking/online (6% of turnover)",
                    "total_income": alt_total,
                    "tax": alt_tax,
                    "citation": "s.58(2)",
                }
            )
        assumptions.append(
            "Assumed you're eligible for and opt into presumptive taxation under s.58 "
            "(resident individual/HUF/non-LLP firm, no commission or agency income)."
        )

    if inp.professional_receipts:
        limit = (
            PROFESSION_RECEIPTS_LIMIT_LOW_CASH
            if inp.digital_share is not None and inp.digital_share >= 0.95
            else PROFESSION_RECEIPTS_LIMIT
        )
        if inp.professional_receipts > limit:
            return TaxResult(
                supported=False,
                reason=(
                    f"Professional receipts of {rupees(inp.professional_receipts)} exceed the s.58(2) "
                    f"limit of {rupees(limit)}; tax depends on actual profit."
                ),
                assumptions=assumptions,
            )
        parts.append(
            ("Presumptive professional income: 50% of gross receipts",
             inp.professional_receipts * PROFESSION_RATE, "s.58(2)")
        )
        sections.append("58")
        assumptions.append(
            "Assumed yours is a specified profession under s.62(4) (legal, medical, engineering, "
            "architectural, accountancy, technical consultancy, interior decoration, IT or company "
            "secretary) and that you opt into presumptive taxation under s.58."
        )

    if inp.other_income:
        parts.append(("Other income", inp.other_income, ""))

    if not parts:
        return TaxResult(
            supported=False,
            reason="No income amount was given, so there's nothing to compute.",
            assumptions=assumptions,
        )

    total, tax, steps = _compute(inp, parts)
    if person == "individual" and inp.resident is not False:
        sections.append("156")

    return TaxResult(
        supported=True,
        assumptions=assumptions,
        steps=steps,
        total_income=total,
        tax=max(tax, 0.0),
        alternatives=alternatives,
        sections=list(dict.fromkeys(sections)),
    )


def format_for_prompt(result: TaxResult) -> str:
    """Render a result as plain text for the LLM's context."""
    if not result.supported:
        return f"CALCULATOR: not computed. {result.reason}"
    lines = ["CALCULATOR RESULT (computed in code from the Act's rates — use these figures exactly):"]
    lines += [f"- Assumption: {a}" for a in result.assumptions]
    for s in result.steps:
        cite = f" [{s['citation']}]" if s["citation"] else ""
        lines.append(f"- {s['label']}: {rupees(s['amount'])}{cite}")
    lines.append(f"- Income-tax payable: {rupees(result.tax)}")
    for alt in result.alternatives:
        lines.append(
            f"- Alternative — {alt['label']}: total income {rupees(alt['total_income'])}, "
            f"tax {rupees(alt['tax'])} [{alt['citation']}]"
        )
    lines += [f"- Note: {n}" for n in result.notes]
    return "\n".join(lines)


def _cite(c: str) -> str:
    return f" (Section {c.removeprefix('s.')})" if c else ""


def format_answer(result: TaxResult) -> str:
    """The user-facing answer for a computed result, built in code.

    Calculation answers never go through the LLM: given the computed figures
    plus loosely related retrieved text, the 7B model ignored the calculator
    and did its own (wrong) arithmetic from an unrelated section.
    """
    lines = [f"Estimated income-tax: {rupees(result.tax)} (Section 202(1)).", "", "How it's worked out:"]
    for s in result.steps:
        lines.append(f"• {s['label']}: {rupees(s['amount'])}{_cite(s['citation'])}")
    lines.append(f"• Income-tax payable: {rupees(result.tax)}")
    if result.assumptions:
        lines += ["", "Assumptions:"] + [f"• {a}" for a in result.assumptions]
    if result.alternatives:
        lines.append("")
        for alt in result.alternatives:
            lines.append(
                f"{alt['label']}: total income {rupees(alt['total_income'])}, "
                f"tax {rupees(alt['tax'])}{_cite(alt['citation'])}."
            )
    lines += ["", "Not included:"] + [f"• {n}" for n in result.notes]
    return "\n".join(lines)
