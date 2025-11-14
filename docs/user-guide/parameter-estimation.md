# Parameter Estimation

To use the Anscombe Transform codec effectively, you need two key parameters: **`zero_level`** and **`conversion_gain`**.

See [Codec Parameters](../concepts/parameters.md) for detailed explanations of what these parameters mean and how they relate to your camera hardware.

This guide explains how to estimate these parameters from your data.

## The `compute_conversion_gain()` Function

The codec provides a built-in parameter estimation function:

```python
import numpy as np
from anscombe_transform.estimate import compute_conversion_gain
from anscombe_transform.common import make_demo_data

# Set seed for reproducibility
np.random.seed(0)

# Generate demo data
movie = make_demo_data(n_frames=100)

# Estimate parameters
result = compute_conversion_gain(movie)

print(f"Conversion gain: {result['conversion_gain']:.3f}")
#> Conversion gain: 29.866
print(f"Zero level: {result['zero_level']:.3f}")
#> Zero level: 17.814
```

### Input Requirements

The `compute_conversion_gain()` function expects:
- **Shape**: `(time, height, width)` - temporal axis must be first
- **Data type**: Integer or float
- **Minimum frames**: At least 10-20 frames for reliable estimation
- **Static scene**: Works best when the scene doesn't change much over time

### How It Works

The function uses the **noise transfer function** approach:

1. **Compute temporal variance**: Calculate sample-wise variance across time
2. **Compute temporal mean**: Calculate sample-wise mean across time
3. **Fit noise model**: Use HuberRegressor to fit `variance = slope * mean + intercept`

For Poisson noise: `variance = conversion_gain * (mean - zero_level)`

Therefore:
- `conversion_gain = slope`
- `zero_level = -intercept / slope`

### Return Value

The function returns a dictionary with:


```python
import numpy as np
from sklearn.linear_model import HuberRegressor

result = {
    'conversion_gain': float,    # The conversion gain (signal units per event)
    'zero_level': float,         # The baseline signal level
    'variance': np.ndarray,      # Computed sample-wise variance
    'model': HuberRegressor,     # The fitted regression model
    'counts': np.ndarray,        # Event counts per sample
    'min_intensity': int,        # Minimum intensity value
    'max_intensity': int,        # Maximum intensity value
}
```

## Validation

After estimating parameters, validate them:

```python
import numpy as np
from anscombe_transform import AnscombeTransformV3
from anscombe_transform.estimate import compute_conversion_gain
from anscombe_transform.common import make_demo_data

# Set seed for reproducibility
np.random.seed(0)

# Generate test movie data
movie = make_demo_data(n_frames=100)

# Estimate parameters
result = compute_conversion_gain(movie)

# Create codec with estimated parameters
codec = AnscombeTransformV3(
    zero_level=result['zero_level'],
    conversion_gain=result['conversion_gain']
)

# Print estimated parameters
print(f"Estimated zero level: {result['zero_level']:.1f}")
#> Estimated zero level: 17.8
print(f"Estimated conversion gain: {result['conversion_gain']:.3f}")
#> Estimated conversion gain: 29.866
```

## Example Workflow


```python
import numpy as np
from anscombe_transform.estimate import compute_conversion_gain
from anscombe_transform import AnscombeTransformV3
from anscombe_transform.common import make_demo_data
import zarr

# Set seed for reproducibility
np.random.seed(0)

# 1. Load temporal data (generate sample movie)
movie = make_demo_data(n_frames=100)

# 2. Estimate parameters
params = compute_conversion_gain(movie)
print(f"Conversion gain: {params['conversion_gain']:.3f} ADU/event")
#> Conversion gain: 29.866 ADU/event
print(f"Zero level: {params['zero_level']:.1f} ADU")
#> Zero level: 17.8 ADU

# 3. Create codec with estimated parameters
codec = AnscombeTransformV3(
    zero_level=params['zero_level'],
    conversion_gain=params['conversion_gain']
)

# 4. Create Zarr array in memory
store = zarr.storage.MemoryStore()
arr = zarr.create_array(
    store=store,
    shape=movie.shape,
    chunks=(10, 512, 512),
    dtype='int16',
    filters=[codec],
    zarr_format=3
)

# 5. Compress data
arr[:] = movie
```