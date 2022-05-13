import pandas as pd
import numpy as np
from mylsh import (_create_shingles_fromstring,
                  shingles_from_series)
import pytest

@pytest.mark.parametrize("text, n_grams, hashfunc, expected",
                        ['two words', 1, None, np.array(['two', 'words']),
                        ])
def test_create_shingles_from_string(text, n_grams, hashfunc, expected):
    assert _create_shingles_fromstring(text, n_grams, hashfunc) == expected
