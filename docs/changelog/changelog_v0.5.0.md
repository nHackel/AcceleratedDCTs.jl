# Changelog

All notable changes to AcceleratedDCTs.jl will be documented in this file.

## [v0.5.0] - 2026-03-29

### ⚠️ Breaking Changes
- **FFTW is no longer a hard dependency.** Users must explicitly `using FFTW` to activate the optimized CPU DCT-I path. Without it, `plan_dct1` on `Array` falls back to the slower separable implementation.

### Added
- `ext/FFTWExt.jl`: New package extension that provides FFTW-based DCT-I/IDCT-I via native `REDFT00` for CPU `Array` inputs. Activated automatically when `FFTW.jl` is loaded.
- `src/utils.jl`: Shared utilities module; `_compute_idct1_scale` extracted from extension to avoid code duplication.
- `test/test_fftw_ext.jl`: Comprehensive tests for the FFTW extension — dispatch verification, correctness (1D/2D/3D), roundtrip accuracy, plan inversion, `ldiv!`, `mul!`, and Float32 support.
- `test/test_idct1_optimized.jl` added to `test/runtests.jl`.
- Documentation: FFTW Extension section in `README.md`, `docs/src/10-tutorial.md`, and `docs/src/30-implementation.md`:
  - DCT-I plan dispatch table (`VkDCTExt` > `FFTWExt` > generic fallback)
  - Extension architecture overview
  - Performance warning when FFTW is not loaded

### Changed
- `Project.toml`: FFTW moved from `[deps]` to `[weakdeps]`; `FFTWExt` registered under `[extensions]`.
- `ext/FFTWExt.jl`: Methods properly qualified with `AcceleratedDCTs.plan_dct1` / `AcceleratedDCTs.plan_idct1` to ensure correct method dispatch (fixes silent fallback to generic path).
- `docs/src/30-implementation.md`: Restructured DCT-I Plans section with separate subsections for VkDCT, FFTW, and Generic/Separable backends.
- `docs/src/10-tutorial.md`: Updated Getting Started to explain FFTW as a weak dependency.

### Removed
- `src/dct1_fftw.jl`: Deleted. FFTW-based DCT-I code now lives exclusively in `ext/FFTWExt.jl`.
- FFTW removed from core `[deps]` (now a weak dependency).

### Contributors
- @nHackel — initial FFTW extension PR (#10)

## [v0.4.1] - 2026-02-23

### Added
- `VkDCT_jll` (v1.3.4) as a direct dependency, providing pre-compiled `libvkfft_dct` for GPU DCT-I. Users no longer need to manually compile the CUDA shim.
- `lib/VkDCT/build_tarballs.jl`: Yggdrasil recipe (reference copy) for building `VkDCT_jll`.
- `docs/design/vkdct_jll_integration.md`: Design document for the JLL integration.

### Changed
- `ext/VkDCTExt.jl`: Library loading now follows priority: `ENV["VKDCT_LIB"]` → `VkDCT_jll.libvkfft_dct` → local fallback.
- Updated `README.md`, `docs/src/index.md`, `docs/src/10-tutorial.md`, and `docs/src/30-implementation.md` to reflect zero-setup VkDCT integration.

### Fixed
- Removed dead links in `README.md` (Issue #5).
- Removed duplicate "3D Optimized" entry in `docs/src/index.md`.

## [v0.4.0] - 2026-02-01

### Added
- VkDCT extension (`ext/VkDCTExt.jl`) with VkFFT-based GPU DCT-I backend.
- `lib/VkDCT`: C++/CUDA shim wrapping VkFFT for 3D DCT-I (Float32 & Float64).
- Benchmarks for VkDCT extension.

## [v0.3.0]

### Added
- Separable split-radix DCT-I algorithm with permuted GPU strategy.
- FFTW-based DCT-I for CPU arrays.
- Mirror-based DCT-I (legacy).
