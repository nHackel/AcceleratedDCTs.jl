# AcceleratedDCTs.jl v0.5.0 Release Notes

## FFTW Moved to Package Extension

FFTW is no longer a hard dependency. The FFTW-based DCT-I implementation has been moved into a package extension (`FFTWExt`), making the core package lighter and more modular.

### ⚠️ Breaking Change

Users must now explicitly load FFTW to get the optimized CPU DCT-I path:

**Before (v0.4.x)**:
```julia
using AcceleratedDCTs  # FFTW loaded automatically as a hard dependency
p = plan_dct1(rand(64, 64, 64))  # FFTW REDFT00 used implicitly
```

**After (v0.5.0)**:
```julia
using FFTW              # ← Required: activates FFTWExt
using AcceleratedDCTs
p = plan_dct1(rand(64, 64, 64))  # FFTW REDFT00 used via extension
```

Without `using FFTW`, `plan_dct1` on CPU `Array` falls back to the generic separable implementation, which is significantly slower.

### Extension Architecture

AcceleratedDCTs.jl now uses Julia's package extensions to keep all backend-specific code optional:

| Extension | Trigger | Backend | Plan Type |
|-----------|---------|---------|-----------|
| `VkDCTExt` | `using CUDA` | VkFFT REDFT00 | `VkFFTDCT1Plan` |
| `FFTWExt` | `using FFTW` | FFTW REDFT00 | `FFTWBasedDCT1Plan` |
| *(none)* | *(always)* | Separable Split-Radix | `DCT1Plan` |

Dispatch follows specificity: `VkDCTExt` > `FFTWExt` > generic fallback.

### Details

- `ext/FFTWExt.jl` methods properly qualified with `AcceleratedDCTs.plan_dct1` / `AcceleratedDCTs.plan_idct1` for correct dispatch.
- `_compute_idct1_scale` extracted to shared `src/utils.jl` to avoid code duplication across extensions.
- Comprehensive test suite added (`test/test_fftw_ext.jl`): dispatch verification, correctness, roundtrip, plan inversion, and Float32 support.
- Documentation updated throughout: README, tutorial, and implementation details.

### Contributors

- @nHackel — initial FFTW extension PR (#10)

### Full Changelog

See [changelog.md](./changelog.md) for the complete list of changes.
