import numpy as np
import pytest

from src.research_stats import holm_adjust


def test_holm_keeps_original_order_and_monotonic_steps():
    np.testing.assert_allclose(holm_adjust([.04, .001, .03, .2]), [.09, .004, .09, .2])


def test_missing_or_unexecuted_tests_do_not_shrink_family():
    np.testing.assert_allclose(holm_adjust([.01, np.nan], family_size=3), [.03, 1])
    with pytest.raises(ValueError):
        holm_adjust([.01, .02], family_size=1)


def test_reject_invalid_p_values():
    with pytest.raises(ValueError):
        holm_adjust([-.1])
