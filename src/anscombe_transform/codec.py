"""
Codec implementations for Anscombe Transform.

This module provides Zarr v2 and v3 codec implementations for the Anscombe
variance-stabilizing transformation, designed for photon-limited data with
Poisson noise.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Literal, Self, TypedDict

import numcodecs
import numpy as np
import numpy.typing as npt
from zarr.abc.codec import ArrayArrayCodec
from zarr.core.array_spec import ArraySpec
from zarr.core.dtype import parse_dtype


class AnscombeCodecConfig(TypedDict):
    """Configuration dictionary for Anscombe codec parameters."""

    zero_level: int
    conversion_gain: float


class AnscomeCodecJSON_V2(AnscombeCodecConfig):
    """Zarr v2 JSON configuration with codec ID."""

    id: Literal["anscombe-v1"]


def make_anscombe_lookup(
    conversion_gain: float,
    input_max: int = 0x7FFF,
    zero_level: int = 0,
    beta: float = 0.5,
    output_type: str = "uint8",
) -> np.ndarray:
    """
    Compute the Anscombe lookup table.

    The lookup converts a linear grayscale image into a uniform variance image
    by applying the Anscombe variance-stabilizing transformation.

    Parameters
    ----------
    conversion_gain : float
        Estimated signal intensity increase per quantum (e.g. photon).
    input_max : int, optional
        The maximum value in the input data, by default 0x7FFF (32767).
    zero_level : int, optional
        Signal level when no photons are recorded, by default 0.
    beta : float, optional
        The grayscale quantization step expressed in units of noise std dev, by default 0.5.
    output_type : str, optional
        NumPy dtype string for output array, by default "uint8".

    Returns
    -------
    np.ndarray
        Lookup table array for Anscombe transformation.
    """
    xx = (np.r_[: input_max + 1] - zero_level) / conversion_gain  # input expressed in photon rates
    zero_slope = 1 / beta / np.sqrt(3 / 8)  # slope for negative values
    offset = zero_level * zero_slope / conversion_gain
    lookup_table = np.round(
        offset
        + (xx < 0) * (xx * zero_slope)
        + (xx >= 0) * (2.0 / beta * (np.sqrt(np.maximum(0, xx) + 3 / 8) - np.sqrt(3 / 8)))
    )
    lookup = lookup_table.astype(output_type)
    assert np.diff(lookup_table).min() >= 0, "non-monotonic lookup generated"
    return lookup


def make_inverse_lookup(lookup_table: np.ndarray, output_type="int16") -> np.ndarray:
    """
    Compute the inverse lookup table for a monotonic forward lookup table.

    The inverse table is indexed by encoded level: entry ``k`` holds the midpoint of the
    range of input values that encode to level ``k`` (rounded down for an integer
    ``output_type``). When the forward table skips a level
    (no input encodes to it, as happens at low conversion gains), its entry is interpolated
    linearly between the neighbouring levels. The table has ``lookup_table.max() + 1``
    entries; `lookup` clamps larger levels to the last entry.

    Parameters
    ----------
    lookup_table : np.ndarray
        Monotonic forward lookup table. If the table wraps around because its levels
        overflow the encoded dtype, only the part before the first wrap is used.
    output_type : str, optional
        NumPy dtype string for output array, by default "int16".

    Returns
    -------
    np.ndarray
        Inverse lookup table that maps encoded values back to original values.
    """
    levels = np.asarray(lookup_table).astype(np.int64)
    # Keep the monotonic part only: levels past an overflow of the encoded dtype wrap around.
    wraps = np.flatnonzero(np.diff(levels) < 0)
    if wraps.size:
        levels = levels[: wraps[0] + 1]
    present, first = np.unique(levels, return_index=True)  # first input of each level
    _, last = np.unique(levels[::-1], return_index=True)  # last input of each level
    last = levels.size - 1 - last
    midpoints = (first + last) / 2
    # Index by the level itself, not by its rank among the levels present.
    inverse = np.interp(np.arange(present[-1] + 1), present, midpoints)
    return inverse.astype(output_type)


def lookup(movie: np.ndarray, lookup_table: np.ndarray) -> np.ndarray:
    """
    Apply lookup table to movie with boundary clamping.

    Parameters
    ----------
    movie : np.ndarray
        Input array to transform.
    lookup_table : np.ndarray
        Lookup table for transformation.

    Returns
    -------
    np.ndarray
        Transformed array with values from lookup table.
    """
    return lookup_table[np.maximum(0, np.minimum(movie, lookup_table.size - 1))]


def encode(
    buf: np.ndarray, *, conversion_gain: float, zero_level: int, encoded_dtype: str
) -> np.ndarray:
    """
    Encode an array using the Anscombe transform.

    Parameters
    ----------
    buf : np.ndarray
        Input array to encode.
    conversion_gain : float
        Signal intensity increase per photon.
    zero_level : int
        Signal level when no photons are recorded.
    encoded_dtype : str
        NumPy dtype string for encoded output.

    Returns
    -------
    np.ndarray
        Encoded array with variance-stabilized values.
    """
    lut = make_anscombe_lookup(
        conversion_gain,
        output_type=encoded_dtype,
        zero_level=zero_level,
    )
    encoded = lookup(buf, lut)
    return encoded.astype(encoded_dtype)


def decode(
    buf: bytes | np.ndarray,
    *,
    conversion_gain: float,
    zero_level: int,
    encoded_dtype: npt.DtypeLike,
    decoded_dtype: npt.DTypeLike,
) -> np.ndarray:
    """
    Decode an array using the inverse Anscombe transform.

    Parameters
    ----------
    buf : bytes or np.ndarray
        Encoded buffer to decode.
    conversion_gain : float
        Signal intensity increase per photon.
    zero_level : int
        Signal level when no photons are recorded.
    encoded_dtype : numpy.typing.DtypeLike
        NumPy dtype of encoded data.
    decoded_dtype : numpy.typing.DtypeLike
        NumPy dtype for decoded output.

    Returns
    -------
    np.ndarray
        Decoded array with original value scale.
    """
    lookup_table = make_anscombe_lookup(
        conversion_gain,
        output_type=encoded_dtype,
        zero_level=zero_level,
    )
    inverse_table = make_inverse_lookup(lookup_table, output_type=decoded_dtype)
    decoded = np.frombuffer(buf, dtype=encoded_dtype)
    return lookup(decoded, inverse_table).astype(decoded_dtype)


@dataclass(frozen=True, slots=True)
class AnscombeTransformV2:
    """
    Zarr v2 codec for Anscombe Transform for photon-limited data.

    The codec assumes input data has linear encoding with Poisson noise,
    typically from photon-limited imaging modalities.

    Attributes
    ----------
    codec_id : str
        Codec identifier ("anscombe-v1").
    zero_level : int
        Signal level when no photons are recorded.
    conversion_gain : float
        Signal intensity increase per photon.
    encoded_dtype : str
        Data type for encoded values (default: "uint8").
    decoded_dtype : str
        Data type for decoded values (default: "int16").
    """

    codec_id: ClassVar[Literal["anscombe-v1"]] = "anscombe-v1"
    zero_level: int
    conversion_gain: float
    encoded_dtype: str = "uint8"
    decoded_dtype: str = "int16"

    def encode(self, buf: np.ndarray) -> np.ndarray:
        """
        Encode data using Anscombe transform.

        Parameters
        ----------
        buf : np.ndarray
            Input array to encode.

        Returns
        -------
        np.ndarray
            Encoded array.
        """
        return encode(
            buf,
            conversion_gain=self.conversion_gain,
            zero_level=self.zero_level,
            encoded_dtype=self.encoded_dtype,
        )

    def decode(self, buf: bytes, out: object | None = None) -> np.ndarray:
        """
        Decode data using inverse Anscombe transform.

        Parameters
        ----------
        buf : bytes
            Encoded buffer to decode.
        out : object or None, optional
            Output buffer (unused), by default None.

        Returns
        -------
        np.ndarray
            Decoded array.
        """
        return decode(
            buf,
            conversion_gain=self.conversion_gain,
            zero_level=self.zero_level,
            encoded_dtype=self.encoded_dtype,
            decoded_dtype=self.decoded_dtype,
        )

    def get_config(self) -> AnscomeCodecJSON_V2:
        """
        Get codec configuration dictionary.

        Returns
        -------
        dict
            Configuration dictionary with codec ID and parameters.
        """
        return {
            "id": self.codec_id,
            "zero_level": self.zero_level,
            "conversion_gain": self.conversion_gain,
        }

    @classmethod
    def from_config(cls, config: AnscomeCodecJSON_V2) -> Self:
        """
        Create codec instance from configuration dictionary.

        Parameters
        ----------
        config : dict
            Configuration dictionary with 'zero_level' and 'conversion_gain' keys.

        Returns
        -------
        AnscombeTransformV2
            New codec instance.
        """
        return cls(zero_level=config["zero_level"], conversion_gain=config["conversion_gain"])


numcodecs.register_codec(AnscombeTransformV2)


@dataclass(frozen=True, slots=True)
class AnscombeTransformV3(ArrayArrayCodec):
    """
    Zarr v3 codec for Anscombe Transform for photon-limited data.

    The codec assumes input data has linear encoding with Poisson noise,
    typically from photon-limited imaging modalities.

    Attributes
    ----------
    zero_level : int
        Signal level when no photons are recorded.
    conversion_gain : float
        Signal intensity increase per photon.
    encoded_dtype : str
        Data type for encoded values (default: "uint8").
    decoded_dtype : str
        Data type for decoded values (default: "int16").
    is_fixed_size : bool
        Whether the codec produces fixed-size output (default: True).
    """

    zero_level: int
    conversion_gain: float
    encoded_dtype: str = "uint8"
    decoded_dtype: str = "int16"
    is_fixed_size: bool = True

    @classmethod
    def from_dict(cls, data: dict) -> Self:
        """
        Create codec instance from configuration dictionary.

        Parameters
        ----------
        data : dict
            Configuration dictionary with 'configuration' key containing codec parameters.

        Returns
        -------
        AnscombeTransformV3
            New codec instance.
        """
        config = data.get("configuration", {})
        return cls(
            zero_level=config["zero_level"],
            conversion_gain=config["conversion_gain"],
            encoded_dtype=config.get("encoded_dtype", "uint8"),
            decoded_dtype=config.get("decoded_dtype", "int16"),
        )

    def to_dict(self) -> dict:
        """
        Convert codec to configuration dictionary.

        Returns
        -------
        dict
            Configuration dictionary with codec name and parameters.
        """
        return {
            "name": "anscombe-v1",
            "configuration": {
                "zero_level": self.zero_level,
                "conversion_gain": self.conversion_gain,
                "encoded_dtype": self.encoded_dtype,
                "decoded_dtype": self.decoded_dtype,
            },
        }

    def resolve_metadata(self, chunk_spec: ArraySpec) -> ArraySpec:
        """
        Resolve metadata for encoded output.

        Parameters
        ----------
        chunk_spec : ArraySpec
            Input chunk specification.

        Returns
        -------
        ArraySpec
            Output chunk specification with updated dtype.
        """
        return ArraySpec(
            shape=chunk_spec.shape,
            dtype=parse_dtype(np.dtype(self.encoded_dtype), zarr_format=3),
            fill_value=chunk_spec.fill_value,
            config=chunk_spec.config,
            prototype=chunk_spec.prototype,
        )

    def _encode(self, buf: np.ndarray) -> np.ndarray:
        """
        Encode data synchronously for direct use.

        Parameters
        ----------
        buf : np.ndarray
            Input array to encode.

        Returns
        -------
        np.ndarray
            Encoded array.
        """
        return encode(
            buf,
            conversion_gain=self.conversion_gain,
            zero_level=self.zero_level,
            encoded_dtype=self.encoded_dtype,
        )

    def _decode(self, buf: np.ndarray) -> np.ndarray:
        """
        Decode data synchronously for direct use.

        Parameters
        ----------
        buf : np.ndarray
            Encoded buffer to decode.

        Returns
        -------
        np.ndarray
            Decoded array.
        """
        return decode(
            buf.tobytes(),
            conversion_gain=self.conversion_gain,
            zero_level=self.zero_level,
            encoded_dtype=self.encoded_dtype,
            decoded_dtype=self.decoded_dtype,
        )

    async def _encode_single(
        self,
        chunk_array,
        chunk_spec,
    ):
        """
        Encode a single chunk using Anscombe transform.

        Parameters
        ----------
        chunk_array : NDBuffer
            Input chunk to encode.
        chunk_spec : ArraySpec
            Chunk specification.

        Returns
        -------
        NDBuffer
            Encoded chunk.
        """
        # Convert NDBuffer to numpy array
        data = chunk_array.as_numpy_array()

        # Apply encoding
        encoded = self._encode(data)

        # Return as NDBuffer
        return chunk_array.from_numpy_array(encoded)

    async def _decode_single(
        self,
        chunk_array,
        chunk_spec,
    ):
        """
        Decode a single chunk using inverse Anscombe transform.

        Parameters
        ----------
        chunk_array : NDBuffer
            Encoded chunk to decode.
        chunk_spec : ArraySpec
            Chunk specification.

        Returns
        -------
        NDBuffer
            Decoded chunk.
        """
        # Convert NDBuffer to numpy array
        data = chunk_array.as_numpy_array()

        # Apply decoding
        decoded = self._decode(data)

        # Reshape to original shape
        decoded = decoded.reshape(chunk_spec.shape)

        # Return as NDBuffer
        return chunk_array.from_numpy_array(decoded)


# Register codec with zarr
from zarr.registry import register_codec

register_codec("anscombe-v1", AnscombeTransformV3)
