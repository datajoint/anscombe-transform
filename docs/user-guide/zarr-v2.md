# Zarr V2 Integration

This guide covers using the Anscombe Transform codec with Zarr V2.

The codec requires `zero_level` and `conversion_gain` parameters. See [Codec Parameters](../concepts/parameters.md) for details.

## Basic Usage

```python
import zarr
import numpy as np
from anscombe_transform import AnscombeTransformV2

# Create data
data = np.random.poisson(lam=50, size=(100, 512, 512)).astype('int16')

# Create Zarr V2 array with Anscombe codec as compressor
arr = zarr.create_array(
    {},
    shape=data.shape,
    chunks=(10, 512, 512),
    dtype='int16',
    filters=AnscombeTransformV2(
        zero_level=100,
        conversion_gain=2.5
    ),
    zarr_format=2
)

# Write
arr[:] = data
```
