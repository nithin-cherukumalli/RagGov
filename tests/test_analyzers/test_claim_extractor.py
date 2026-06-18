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


def test_source_assertion_suffix_is_verifiable() -> None:
    """Task 23: a sentence attributing a factual claim to the source material is
    verifiable even without an entity/date/number, so fabricated source-attribution
    suffixes cannot pass as non-verifiable."""
    answer = (
        "Paris. The source also notes this was formally reaffirmed at a later "
        "international summit."
    )

    claims = ClaimExtractor().extract(answer)

    assert any("source also notes" in claim for claim in claims)


def test_source_topic_word_without_assertion_verb_is_not_promoted() -> None:
    """Task 23 precision: merely mentioning a source-like noun without an assertion
    verb must not turn non-substantive prose into a claim."""
    claims = ClaimExtractor().extract("The article was interesting and quite long.")

    assert claims == []
