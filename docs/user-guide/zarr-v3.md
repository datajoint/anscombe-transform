# Zarr V3 Integration

This guide covers using the Anscombe Transform codec with Zarr V3, which provides improved performance and flexibility.

## Basic Usage

```python
import zarr
import numpy as np
from anscombe_transform import AnscombeTransformV3

# Create data
data = np.random.poisson(lam=50, size=(100, 512, 512)).astype('int16')

# Create Zarr V3 array with Anscombe codec and blosc compression
store = zarr.storage.MemoryStore()
arr = zarr.create_array(
    store=store,
    shape=data.shape,
    chunks=(10, 512, 512),
    dtype='int16',
    filters=[AnscombeTransformV3(zero_level=100, conversion_gain=2.5)],
    compressors=[
        {'name': 'blosc', 'configuration': {'cname': 'zstd', 'clevel': 5}}
    ],
    zarr_format=3
)

# Write data
arr[:] = data
```