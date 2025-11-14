# Anscombe Transform

[![PyPI version](https://badge.fury.io/py/anscombe-transform.svg)](https://badge.fury.io/py/anscombe-transform)
![tests](https://github.com/datajoint/anscombe-transform/actions/workflows/test.yml/badge.svg)

Zarr V2 and V3 codecs for compressing photon-limited movies using the Anscombe Transform.

## What is it?

This codec is designed for compressing movies with Poisson noise, which are produced by photon-limited modalities such as:

- Multiphoton microscopy
- Radiography
- Astronomy

## How it works

The codec re-quantizes grayscale data efficiently using a square-root-like transformation to equalize noise variance across grayscale levels: the [Anscombe Transform](https://en.wikipedia.org/wiki/Anscombe_transform). This results in:

- Fewer unique grayscale levels
- Significant improvements in data compressibility
- No sacrifice to signal accuracy

## Requirements

To use the codec, you need to provide two pieces of information:

1. **`zero_level`**: The input value corresponding to the absence of light
2. **`conversion_gain`** (also called `photon_sensitivity`): The conversion factor from signal levels to photon counts

The codec assumes that the video is linearly encoded with a potential offset and that these parameters can be accurately estimated from the data.

## Features

- ✅ Zarr V2 support via `numcodecs` interface
- ✅ Zarr V3 support via `ArrayArrayCodec` interface
- ✅ Automatic parameter estimation from data
- ✅ Lossless compression for photon-limited data
- ✅ Python 3.11+ support

## Quick Example

```python
import zarr
import numpy as np
from anscombe_transform import AnscombeTransformV3

# Create sample data with Poisson noise
data = np.random.poisson(lam=50, size=(100, 512, 512)).astype('int16')

# Create Zarr array with Anscombe codec and blosc compression
store = zarr.storage.MemoryStore()
arr = zarr.create_array(
    store=store,
    shape=data.shape,
    chunks=(10, 512, 512),
    dtype='int16',
    filters=[AnscombeTransformV3(zero_level=100, conversion_gain=2.5)],
    compressors=[{'name': 'blosc', 'configuration': {'cname': 'zstd', 'clevel': 5}}],
    zarr_format=3
)

# Write and read data
arr[:] = data
recovered = arr[:]
```

## Next Steps

- [Installation Guide](getting-started/installation.md)
- [Quick Start Tutorial](getting-started/quick-start.md)
- [User Guide](user-guide/overview.md)
- [API Reference](api/codec.md)
