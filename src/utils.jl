# ============================================================================
# Shared Utilities
# ============================================================================

"""
    _compute_idct1_scale(sz, region)

Compute the normalization scale for IDCT-I.
For DCT-I, the inverse is DCT-I itself scaled by 1/∏(2*(Mi-1)) for each
dimension i in `region`.
"""
function _compute_idct1_scale(sz::NTuple{N, Int}, region) where N
    scale = one(Float64)
    for d in region
        scale *= 2 * (sz[d] - 1)
    end
    return 1 / scale
end
