"""Unit tests for the entity-resolution lexical gate (pick #3).

`_lexically_related` PROPOSES merge candidates; the centroid cosine DISPOSES.
The live corpus500 dry-run showed the earlier "shared significant token" rule
over-proposed distinct same-domain entities ('X University'/'Y University',
'Ministry of X'/'Ministry of Y', 'Computed Tomography'/'Magnetic Resonance
Tomography') that ALSO score cosine >= 0.9 (same-document context), so cosine
could not veto them. The gate is now normalization/substring + acronym only —
high precision. These tests lock that in.
"""

from __future__ import annotations

import pytest

from src.services.kg import _lexically_related


@pytest.mark.parametrize(
    "a,b",
    [
        ("government", "Government"),  # case
        ("Cơ sở y tế", "CƠ SỞ Y TẾ"),  # case + diacritics
        ("Disney's", "Disney"),  # possessive (short form is >=80% of the long)
        ("National Geographic", "National Geographic's"),  # trailing 's
        ("the Financial Secretary", "Financial Secretary"),  # dropped article
        ("large language model", "LLM"),  # acronym
        ("LLM", "large language model"),  # acronym, symmetric
        ("United States", "US"),  # acronym via compact form
        ("United Nations", "UN"),  # acronym
    ],
)
def test_proposes_real_variants(a, b):
    assert _lexically_related(a, b) is True


@pytest.mark.parametrize(
    "a,b",
    [
        ("Doug", "Mike"),  # two people in one doc — the co-occurrence trap
        # real corpus500 dry-run over-merges the shared-token rule used to propose:
        ("Washington and Lee University", "Princeton University"),
        ("Ministry of Employment", "Ministry of Education"),
        ("Computed Tomography", "Magnetic Resonance Tomography"),
        ("Bank of America", "America First Bank"),  # distinct banks
        ("Ministry", "Ministry of Education"),  # generic ⊄ specific (80% guard)
        ("thành phẩm", "bán thành phẩm"),  # finished ⊄ semi-finished (meaning flip)
        ("Comptroller", "Comptroller Gould"),  # title ⊄ specific person
        ("Canada", "California"),  # no substring / acronym
        ("2022", "2023"),  # adjacent dates must never merge
        ("OCC", "Disney"),
    ],
)
def test_refuses_unrelated(a, b):
    assert _lexically_related(a, b) is False


def test_empty_and_degenerate():
    assert _lexically_related("", "Apple") is False
    assert _lexically_related("Apple", "") is False
    assert _lexically_related("...", "!!!") is False
