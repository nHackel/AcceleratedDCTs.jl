# Implementation Details

## KernelAbstractions.jl
The code is written using `KernelAbstractions`, meaning the **exact same code** runs on:
*   CPU (using `Base.Threads`)
*   NVIDIA GPU (using `CUDA.jl`)
*   AMD GPU (using `AMDGPU.jl`) - *conceptual support, untested*

We strictly avoid "scalar indexing" (accessing single elements from the host), ensuring high performance on GPUs.

## Plan-Based API (AbstractFFTs)
To maximize performance, we separate **resource allocation** (cheap on CPU, expensive on GPU) from **execution**.

### DCT-II/DCT-III Plans
*   **`plan_dct(x)`** / **`plan_idct(x)`**:
    *   Allocates temporary buffers (`tmp_real`, `tmp_comp`).
    *   Creates an internal FFT plan (`plan_rfft`).
    *   Pre-calculates twiddle factors (`cispi(...)`) on the device.

### DCT-I Plan Dispatch

`plan_dct1(x)` automatically selects the best backend based on the array type and loaded packages:

| Array Type | Loaded Package | Backend | Plan Type |
|:-----------|:---------------|:--------|:----------|
| `CuArray` (3D) | `CUDA.jl` | VkFFT REDFT00 | `VkFFTDCT1Plan` |
| `Array` | `FFTW.jl` | FFTW REDFT00 | `FFTWBasedDCT1Plan` |
| Any `AbstractArray` | *(always available)* | Separable Split-Radix | `DCT1Plan` |

Dispatch follows specificity: `VkDCTExt` > `FFTWExt` > generic fallback.

### DCT-I Plans (VkDCT Extension)
*   **`plan_dct1(x::CuArray)`** via **`VkDCTExt`**:
    *   **Backend**: Uses [`VkDCT_jll`](https://github.com/JuliaBinaryWrappers/VkDCT_jll.jl), a pre-compiled [VkFFT](https://github.com/DTolm/VkFFT)-based CUDA library. No manual compilation required.
    *   **Trigger**: Activated automatically when `CUDA.jl` is loaded.
    *   **Performance**: ~7x-15x faster than the separable backend due to hand-tuned kernels and reduced memory traffic.
    *   **Features**: Supports `Float32`/`Float64`, full 3D transforms.

### DCT-I Plans (FFTW Extension)
*   **`plan_dct1(x::Array)`** via **`FFTWExt`**:
    *   **Backend**: Uses FFTW's native `REDFT00` (real-even DFT), computing DCT-I directly in a single optimized call.
    *   **Trigger**: Activated when `FFTW.jl` is loaded (`using FFTW`).
    *   **Performance**: Significantly faster than the generic separable fallback on CPU.
    *   **Features**: Supports `Float32`/`Float64`, any dimensionality (1D, 2D, 3D, ...).

!!! warning "FFTW required for optimal CPU DCT-I"
    Without `using FFTW`, `plan_dct1` on CPU `Array` falls back to the generic
    separable implementation, which is significantly slower. The separable
    fallback itself still requires *some* FFT backend (e.g. FFTW) to be loaded
    for its internal `plan_fft!` calls. Always load FFTW for CPU usage:
    ```julia
    using FFTW
    using AcceleratedDCTs
    ```

### DCT-I Plans (Generic / Separable)
*   **`plan_dct1(x::AbstractArray)`** / **`plan_idct1(x::AbstractArray)`**:
    *   **Default fallback** for any array type without a specialized extension.
    *   Uses **Separable Split-Radix** algorithm:
        *   Maps $M$ points to size $N=M-1$ Complex FFT.
        *   **GPU Strategy**: Uses a **Permuted** approach to ensure Unit Stride memory access for internal FFTs across all dimensions, maximizing performance.
        *   Memory scaling: $O(M^D)$ (efficient for N-D).
    *   Dispatch is automatic.

### DCT-I Plans (Mirroring / Legacy)
*   **`plan_dct1_mirror(x)`** / **`plan_idct1_mirror(x)`**:
    *   Uses **Mirroring** strategy.
    *   Maps $M$ points to size $2M-2$ Real FFT.
    *   Memory scaling: $O((2M)^D)$ (expensive for N-D).
    *   Available as an alternative for validation.

### Execution
*   **`mul!(y, p, x)`**:
    *   Reuses all buffers.
    *   Zero memory allocation during execution.
