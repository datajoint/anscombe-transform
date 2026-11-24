import numpy as np
import pytest

from anscombe_transform import AnscombeTransformV2
from anscombe_transform.codec import make_anscombe_lookup


def make_poisson_ramp_signals(
    rng: np.random.Generator, shape=(10, 1, 1), min_rate=1, max_rate=5, dtype="int16"
):
    assert isinstance(shape, tuple)
    assert len(shape) == 3
    x, y, times = shape
    output_array = np.zeros(shape, dtype=dtype)
    for x_ind in range(x):
        for y_ind in range(y):
            output_array[x_ind, y_ind, :] = sensitivity * rng.poisson(
                np.arange(min_rate, max_rate, (max_rate - min_rate) / times)
            )
    return output_array.astype("int16")


sensitivity = 100.0


@pytest.fixture
def test_data(dtype="int16") -> list[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)
    test2d = make_poisson_ramp_signals(rng, shape=(50, 1, 1), min_rate=1, max_rate=5, dtype=dtype)
    test2d_long = make_poisson_ramp_signals(
        rng, shape=(1, 50, 1), min_rate=1, max_rate=5, dtype=dtype
    )
    return [test2d, test2d_long]


def test_poisson_encode_decode(test_data: np.ndarray) -> None:
    codec = AnscombeTransformV2(
        zero_level=0,
        conversion_gain=sensitivity,
        encoded_dtype="uint8",
        decoded_dtype="int16",
    )
    lookup_table = make_anscombe_lookup(sensitivity, zero_level=0, output_type="uint8")
    for example_data in test_data:
        encoded = codec.encode(example_data)
        decoded = codec.decode(encoded)
        recoded = codec.decode(codec.encode(decoded))
        # Each level decodes to the midpoint of the inputs that share it, so a value is at
        # most half that range away. The range widens as counts grow, so a fixed tolerance
        # fails whenever the random data reach large counts.
        original = example_data.ravel()  # decode returns a flat array
        step = np.bincount(lookup_table)[lookup_table[original]]
        error = np.abs(np.ravel(decoded).astype(np.int64) - original)
        assert np.all(error <= step // 2)
        assert (decoded == recoded).all()
