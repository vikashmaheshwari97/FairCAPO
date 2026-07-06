from __future__ import annotations

from scripts.run_phase2_budgeted_mocapo import extract_label

LABELS = [
    "model",
    "nurse",
    "poet",
    "physician",
    "surgeon",
    "software_engineer",
    "professor",
    "teacher",
]


def test_exact_label_match():
    assert extract_label("surgeon", LABELS) == "surgeon"


def test_final_answer_span_is_preferred_over_reasoning():
    resp = "I first considered nurse, then physician.\n<final_answer>surgeon</final_answer>"
    assert extract_label(resp, LABELS) == "surgeon"


def test_underscore_and_space_spellings_both_match():
    assert extract_label("<final_answer>software engineer</final_answer>", LABELS) == (
        "software_engineer"
    )
    assert extract_label("software_engineer", LABELS) == "software_engineer"


def test_incidental_word_does_not_match_label_substring():
    # "model" must NOT be extracted from "modeling"/"remodeled"; "poet" must NOT
    # be extracted from "poetry". These were false positives under substring scan.
    assert extract_label("She works in modeling software.", LABELS) != "model"
    assert extract_label("He wrote about poetry theory.", LABELS) != "poet"


def test_word_boundary_still_matches_standalone_word():
    assert extract_label("She is a fashion model.", LABELS) == "model"


def test_last_asserted_label_wins_over_negated_one():
    # "...not a nurse, she is a physician" should resolve to physician.
    assert extract_label("She is not a nurse, she is a physician.", LABELS) == (
        "physician"
    )


def test_longer_label_preferred_on_tie():
    # Both "software_engineer" and "engineer"-like fragments could appear; the
    # full label should win. (Only software_engineer is in LABELS, so this checks
    # the multi-word variant matches cleanly.)
    resp = "Clearly a software engineer.\n<final_answer>software engineer</final_answer>"
    assert extract_label(resp, LABELS) == "software_engineer"


def test_truncated_closing_tag_still_reads_answer_span():
    # num_predict can cut the response before </final_answer>; the open tag still
    # anchors the answer span.
    assert extract_label("reasoning...\n<final_answer>surgeon", LABELS) == "surgeon"


def test_no_match_returns_cleaned_text_not_a_spurious_label():
    pred = extract_label("<final_answer>astronaut</final_answer>", LABELS)
    assert pred not in LABELS
    assert pred == "astronaut"
