# Quick Start

This guide will help you get started with the Anscombe Transform codec for compressing photon-limited movies.

## Basic Usage with Zarr V3

```python
import zarr
import numpy as np
from anscombe_transform import AnscombeTransformV3
from anscombe_transform.common import make_demo_data

# Set seed for reproducibility
np.random.seed(0)

# Generate test data
data = make_demo_data(zero_level=20.0, conversion_gain=30.0)

# Create a Zarr array with the Anscombe codec and blosc compression
store = zarr.storage.MemoryStore()
arr = zarr.create_array(
    store=store,
    shape=data.shape,
    chunks=(12, 160, 80),
    dtype='int16',
    filters=[AnscombeTransformV3(zero_level=20.0, conversion_gain=30.0)],
    compressors=[
        {'name': 'blosc', 'configuration': {'cname': 'zstd', 'clevel': 5}}
    ],
    zarr_format=3
)

# Write data
arr[:] = data

# Read data back
recovered = arr[:]

# The transformation is lossy, so we do not expect data to
# round-trip perfectly
print(f"Max difference: {np.abs(data - recovered).max()}")
#> Max difference: 59
```

## Estimating Parameters from Data

If you don't know the `zero_level` and `conversion_gain` parameters, you can estimate them from your data:

```python
import numpy as np
from anscombe_transform.estimate import compute_conversion_gain
from anscombe_transform import AnscombeTransformV3
from anscombe_transform.common import make_demo_data

# Set seed for reproducibility
np.random.seed(0)

# Generate test data
movie = make_demo_data(n_frames=100)

# Estimate parameters
result = compute_conversion_gain(movie)

print(f"Estimated conversion gain: {result['conversion_gain']:.3f}")
#> Estimated conversion gain: 29.866
print(f"Estimated zero level: {result['zero_level']:.3f}")
#> Estimated zero level: 17.814

# Use estimated parameters in codec
codec = AnscombeTransformV3(
    zero_level=int(result['zero_level']),
    conversion_gain=result['conversion_gain']
)
```

## Key Parameters

The Anscombe codec requires two key parameters:

- **`zero_level`**: Baseline signal with no photons. See [Parameters](../concepts/parameters.md#zero_level) for details.
- **`conversion_gain`**: Signal units per photon. See [Parameters](../concepts/parameters.md#conversion_gain) for details.

Optional parameters:

- **`encoded_dtype`**: Data type for encoded values (default: `uint8`). Use `uint8` for maximum compression.
- **`decoded_dtype`**: Data type for decoded values (default: inferred from data).
