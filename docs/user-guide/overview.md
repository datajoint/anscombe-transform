# Overview

## The Anscombe Transform

The Anscombe Transform is a variance-stabilizing transformation specifically designed for data with Poisson noise. In photon-limited imaging, the noise variance grows linearly with the signal mean (characteristic of Poisson statistics), 
which makes compression difficult because different intensity levels have different noise characteristics.

### The Problem

In photon-limited data:
- Low intensity regions have low noise variance
- High intensity regions have high noise variance
- This heteroscedastic noise makes efficient compression challenging

### The Solution

The Anscombe Transform applies a square-root-like transformation that:
1. Equalizes noise variance across all intensity levels
2. Reduces the number of unique grayscale values needed
3. Improves compressibility without losing signal accuracy

Mathematically, the transform is:

```
f(x) = 2 * sqrt(x + 3/8)
```

For our codec, we adapt this to account for camera parameters:

```
encoded = quantize(2 * sqrt((data - zero_level) / conversion_gain + 3/8))
```

## Codec Architecture

The codec is implemented in two versions to support both Zarr V2 and V3:

### Zarr V2: `AnscombeTransformV2`
- Implements the `numcodecs.Codec` interface
- Used as a compressor in Zarr V2 arrays
- Registered with ID `"anscombe-v1"`

### Zarr V3: `AnscombeTransformV3`
- Implements the `ArrayArrayCodec` interface
- Used as a filter before compression in Zarr V3 arrays
- Registered with the same ID `"anscombe-v1"`

Both share the same core `encode()` and `decode()` functions, ensuring consistent behavior.

## How It Works

### Encoding Pipeline

1. **Normalize**: Convert raw data to photon counts using `conversion_gain` and `zero_level`
2. **Transform**: Apply the Anscombe Transform to stabilize variance
3. **Quantize**: Discretize the transformed values to `encoded_dtype` (typically `uint8`)
4. **Compress**: Apply additional compression (e.g., Blosc, Zstd)

### Decoding Pipeline

1. **Decompress**: Uncompress the data
2. **Lookup**: Apply inverse transform via lookup table
3. **Denormalize**: Convert back to original units using `conversion_gain` and `zero_level`

### Lookup Tables

The codec uses pre-computed lookup tables for efficiency:
- **Forward lookup**: Maps input values to transformed values
- **Inverse lookup**: Maps transformed values back to original values

These tables are computed once during codec initialization and reused for all encode/decode operations.

## Performance Characteristics

### Compression Ratios

Typical compression ratios (Anscombe + Blosc/Zstd):
- **3-8x** for typical multiphoton microscopy data
- **6-10x** for astronomy data
- **3-6x** for radiography data

The exact ratio depends on:
- Signal-to-noise ratio of the data
- Spatial correlation in the images
- Choice of secondary compressor

### Speed

The codec is designed for speed:
- Encoding: ~500-1000 MB/s (single-threaded)
- Decoding: ~800-1500 MB/s (single-threaded)
- Scales well with chunk-based parallel processing

### Accuracy

The codec is designed to be **nearly lossless** for photon-limited data:
- Max absolute error: ~ 0.25 noise-sigma per pixel (for `beta=0.5`)
- Error scales with quantization (`beta` parameter)
- For default parameters (`beta=0.5`), the noise variance is increased by a ~1 % with respect to the original noise variance and no bias is introduced.

## When to Use This Codec

### Good Use Cases ✅

- Multiphoton microscopy movies
- Astronomy images with photon counting detectors
- Radiography/X-ray imaging
- Any data with Poisson noise where signal ≈ variance
- Data where you can estimate or know `conversion_gain` and `zero_level`

### Not Recommended ❌

- Data with non-Poisson or non-stationalry noise (e.g., pre-processed images)
- Data where detector parameters are unknown and can't be estimated
- Data that has been transformed with a non-linear function (e.g. gamma correction)

## Next Steps

- [Parameter Estimation Guide](parameter-estimation.md)
- [Zarr V2 Integration](zarr-v2.md)
- [Zarr V3 Integration](zarr-v3.md)
