import FFTW
import AbstractFFTs
using Test
using AcceleratedDCTs
using AcceleratedDCTs: dct1, idct1, plan_dct1, plan_idct1
using LinearAlgebra: mul!, ldiv!

# Access the extension's types via Base.get_extension
const _FFTWExt = Base.get_extension(AcceleratedDCTs, :FFTWExt)
const FFTWBasedDCT1Plan = _FFTWExt.FFTWBasedDCT1Plan
const FFTWBasedIDCT1Plan = _FFTWExt.FFTWBasedIDCT1Plan

# Helper function to compute FFTW's DCT-I (REDFT00) directly
function fftw_dct1_ref(x::AbstractArray{T}) where T
    return FFTW.r2r(x, FFTW.REDFT00)
end

@testset "FFTWExt: FFTW-based DCT-I" begin
    @testset "Dispatch: plan_dct1(::Array) returns FFTWBasedDCT1Plan" begin
        for sz in [(16,), (8, 8), (4, 4, 4)]
            x = rand(Float64, sz...)
            p = plan_dct1(x)
            @test p isa FFTWBasedDCT1Plan
        end
    end

    @testset "Dispatch: plan_idct1(::Array) returns FFTWBasedIDCT1Plan" begin
        for sz in [(16,), (8, 8), (4, 4, 4)]
            x = rand(Float64, sz...)
            p = plan_idct1(x)
            @test p isa FFTWBasedIDCT1Plan
        end
    end

    @testset "Correctness: DCT-I (1D)" begin
        for M in [4, 5, 8, 16, 32, 64]
            x = rand(Float64, M)
            p = plan_dct1(x)
            y = p * x
            @test y ≈ fftw_dct1_ref(x) atol=1e-12
        end
    end

    @testset "Correctness: DCT-I (2D)" begin
        for (M1, M2) in [(4, 4), (8, 9), (16, 16)]
            x = rand(Float64, M1, M2)
            p = plan_dct1(x)
            y = p * x
            @test y ≈ fftw_dct1_ref(x) atol=1e-11
        end
    end

    @testset "Correctness: DCT-I (3D)" begin
        for sz in [(4, 4, 4), (4, 5, 6), (8, 8, 8)]
            x = rand(Float64, sz...)
            p = plan_dct1(x)
            y = p * x
            @test y ≈ fftw_dct1_ref(x) atol=1e-10
        end
    end

    @testset "Roundtrip Accuracy" begin
        for sz in [(16,), (8, 8), (4, 5, 6)]
            x = rand(Float64, sz...)
            y = dct1(x)
            x_rec = idct1(y)
            @test x_rec ≈ x atol=1e-12
        end
    end

    @testset "Plan inversion" begin
        x = rand(Float64, 8, 8)
        p_fwd = plan_dct1(x)
        p_inv = inv(p_fwd)
        @test p_inv isa FFTWBasedIDCT1Plan

        p_fwd2 = inv(p_inv)
        @test p_fwd2 isa FFTWBasedDCT1Plan

        y = p_fwd * x
        x_rec = p_inv * y
        @test x_rec ≈ x atol=1e-12
    end

    @testset "ldiv!" begin
        x = rand(Float64, 8, 8)
        p_fwd = plan_dct1(x)
        y = p_fwd * x

        x_rec = p_fwd \ y
        @test x_rec ≈ x atol=1e-12

        x_rec2 = similar(x)
        ldiv!(x_rec2, p_fwd, y)
        @test x_rec2 ≈ x atol=1e-12
    end

    @testset "mul! (in-place)" begin
        x = rand(Float64, 8, 8)
        p = plan_dct1(x)

        y = similar(x)
        mul!(y, p, x)
        @test y ≈ fftw_dct1_ref(x) atol=1e-11

        p_inv = plan_idct1(x)
        x_rec = similar(x)
        mul!(x_rec, p_inv, y)
        @test x_rec ≈ x atol=1e-12
    end

    @testset "Float32 support" begin
        x = rand(Float32, 8, 8, 8)
        p = plan_dct1(x)
        @test p isa FFTWBasedDCT1Plan

        y = p * x
        x_rec = idct1(y)
        @test x_rec ≈ x atol=1e-4
    end
end
