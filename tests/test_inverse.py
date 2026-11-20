"""
Round-trip accuracy of the inverse lookup table.

Each encoded level decodes to the midpoint of the range of input values that encode to it,
so a decoded value is never further from the original than half the width of that range
(the quantization step at that input).
"""

from __future__ import annotations

import numpy as np
import pytest

from anscombe_transform import AnscombeTransformV2, AnscombeTransformV3
from anscombe_transform.codec import make_anscombe_lookup, make_inverse_lookup


def quantization_step(x: np.ndarray, lookup_table: np.ndarray) -> np.ndarray:
    """Number of input values that share the encoded level of each value in ``x``."""
    levels = lookup_table.astype(np.int64)
    wraps = np.flatnonzero(np.diff(levels) < 0)
    if wraps.size:  # levels that overflow the encoded dtype wrap around
        levels = levels[: wraps[0] + 1]
    return np.bincount(levels)[levels[x]]


def round_trip(codec: AnscombeTransformV2 | AnscombeTransformV3, x: np.ndarray) -> np.ndarray:
    if isinstance(codec, AnscombeTransformV3):
        return codec._decode(codec._encode(x))
    return codec.decode(codec.encode(x))


CODECS = [AnscombeTransformV2, AnscombeTransformV3]


@pytest.mark.parametrize("codec_class", CODECS)
@pytest.mark.parametrize("encoded_dtype", ["uint8", "uint16"])
@pytest.mark.parametrize(
    ("conversion_gain", "zero_level", "input_max"),
    [(1.0, 0, 1000), (20.0, 100, 20000)],
    ids=["raw counts", "gain 20"],
)
def test_round_trip_within_quantization_step(
    codec_class: type[AnscombeTransformV2 | AnscombeTransformV3],
    encoded_dtype: str,
    conversion_gain: float,
    zero_level: int,
    input_max: int,
) -> None:
    codec = codec_class(
        zero_level=zero_level,
        conversion_gain=conversion_gain,
        encoded_dtype=encoded_dtype,
        decoded_dtype="int16",
    )
    x = np.arange(input_max + 1, dtype="int16")
    decoded = round_trip(codec, x)

    lookup_table = make_anscombe_lookup(
        conversion_gain, zero_level=zero_level, output_type=encoded_dtype
    )
    step = quantization_step(x, lookup_table)
    error = np.abs(decoded.astype(np.int64) - x)
    # Midpoint of the step, rounded down: at most half the step away from any value in it.
    assert np.all(error <= step // 2)
    # Where every count has its own level, decoding is exact.
    assert np.all(error[step == 1] == 0)


@pytest.mark.parametrize("encoded_dtype", ["uint8", "uint16"])
def test_raw_counts_regression(encoded_dtype: str) -> None:
    """
    With conversion_gain=1 and zero_level=0 the forward table skips levels 1 and 3.
    Indexing the inverse by rank instead of by level decoded 20 counts as 26.
    """
    codec = AnscombeTransformV2(zero_level=0, conversion_gain=1.0, encoded_dtype=encoded_dtype)
    encoded = codec.encode(np.array([20], dtype="int16"))
    assert encoded[0] == 16
    decoded = codec.decode(encoded)
    assert abs(int(decoded[0]) - 20) <= 1


@pytest.mark.parametrize(
    ("conversion_gain", "zero_level"), [(1.0, 0), (1.0, 100), (3.0, 20), (20.0, 100)]
)
def test_inverse_is_indexed_by_level(conversion_gain: float, zero_level: int) -> None:
    lookup_table = make_anscombe_lookup(
        conversion_gain, zero_level=zero_level, output_type="uint16"
    )
    inverse = make_inverse_lookup(lookup_table)
    assert inverse.size == lookup_table.max() + 1
    # Every level that occurs decodes to an input inside its own range ...
    present = np.unique(lookup_table)
    assert np.array_equal(lookup_table[inverse[present]], present)
    # ... and skipped levels decode between their neighbours.
    assert np.all(np.diff(inverse.astype(np.int64)) >= 0)


def test_inverse_ignores_wrapped_levels() -> None:
    """
    At conversion_gain=1 the levels for large inputs overflow uint8 and wrap around.
    The inverse comes from the monotonic part of the table only.
    """
    lookup_table = make_anscombe_lookup(1.0, zero_level=0, output_type="uint8")
    assert np.diff(lookup_table.astype(np.int64)).min() < 0  # the table does wrap
    inverse = make_inverse_lookup(lookup_table)
    assert inverse.size == 256
    assert np.all(np.diff(inverse.astype(np.int64)) >= 0)
    # 20, 21 and 22 counts encode to level 16, which decodes to 21.
    assert np.array_equal(np.flatnonzero(lookup_table[:1000] == 16), [20, 21, 22])
    assert inverse[16] == 21
