from raggov.analyzers.grounding.claims import ClaimExtractor


def test_policy_language_with_spelled_out_numbers_is_treated_as_substantive() -> None:
    answer = (
        "Helio subscribers can request a refund within fourteen days of the initial purchase, "
        "and annual renewals are refundable only if premium features were not used after renewal."
    )

    claims = ClaimExtractor().extract(answer)

    assert claims


def test_non_policy_explanatory_text_can_remain_unchecked() -> None:
    answer = (
        "Python list comprehensions build a new list from an iterable by evaluating an output "
        "expression for each item in a for clause."
    )

    claims = ClaimExtractor().extract(answer)

    assert claims == []


def test_short_entity_answer_is_claim_worthy_for_citation_checks() -> None:
    claims = ClaimExtractor().extract("Paris.")

    assert claims == ["Paris."]
