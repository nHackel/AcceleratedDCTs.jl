import FFTW
using AcceleratedDCTs
using Test
using LinearAlgebra

@testset "AcceleratedDCTs.jl" begin
    # Reference implementation tests (slow)
    include("test_reference.jl")

    # Existing batched implementation tests
    include("test_dct_batch.jl")

    # DCT Plan and cached buffer tests
    include("test_dct_plan.jl")

    # DCT-II/DCT-III
    include("test_dct_optimized.jl")

    # DCT-I (separable / generic fallback)
    include("test_dct1_separable.jl")

    # IDCT-I
    include("test_idct1_optimized.jl")

    # FFTW extension (loaded via `import FFTW` above)
    include("test_fftw_ext.jl")
end

