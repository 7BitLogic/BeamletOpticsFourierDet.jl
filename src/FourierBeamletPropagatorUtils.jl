# ==============================================================================
# Numerical Validation, Observables Readout, BMO Detector Verbs & Hit Helpers
# ==============================================================================

# Replace ±Inf or NaN values with a finite fallback default
@inline safe_finite(x::T, default::T = zero(T)) where {T <: Real} = isfinite(x) ? x : default
@inline safe_finite(x::Real, default::Real) = safe_finite(promote(x, default)...)

# Ensure strictly positive finite values (e.g., beam waist radii), falling back otherwise
@inline safe_pos(x::T, fallback::T) where {T <: Real} = (isfinite(x) && x > zero(T)) ? x : fallback
@inline safe_pos(x::Real, fallback::Real) = safe_pos(promote(x, fallback)...)

# Guarded reciprocal 1/x that returns `default` on zero or non-finite inputs
@inline function safe_inv(x::T, default::T = zero(T)) where {T <: Real}
    (isfinite(x) && !iszero(x)) || return default
    ix = inv(x)
    return isfinite(ix) ? T(ix) : default
end
@inline safe_inv(x::Real, default::Real) = safe_inv(promote(x, default)...)

# Compute axial defocus distance Δz = C / (C² + (λ / π w²)²) from wavefront curvature C = 1/R and spot size w
@inline function _beamlet_defocus_distance(C::T, w::T, wl::T)::T where {T <: Real}
    (!isfinite(C) || !isfinite(w) || w <= zero(T) || !isfinite(wl) || wl <= zero(T)) && return zero(T)
    zr_inv = wl / (T(π) * w^2)
    denom = C^2 + zr_inv^2
    res = denom > zero(T) ? (C / denom) : zero(T)
    return isfinite(res) ? res : zero(T)
end
@inline _beamlet_defocus_distance(C::Real, w::Real, wl::Real) =
    _beamlet_defocus_distance(promote(float(C), float(w), float(wl))...)

# ==============================================================================
# Observable Extraction & Detector Readout
# ==============================================================================

# Effective pixel integration measure: Δx in 1D, Δx⋅Δz in 2D, and slice-averaged Δx⋅Δz / Ny in 3D
@inline _calculate_pixel_area(grid::SpatialGrid{T, 1}) where {T} = abs(step(grid.ranges[1]))
@inline _calculate_pixel_area(grid::SpatialGrid{T, 2}) where {T} = abs(step(grid.ranges[1])) * abs(step(grid.ranges[2]))
@inline _calculate_pixel_area(grid::SpatialGrid{T, 3}) where {T} = (abs(step(grid.ranges[1])) * abs(step(grid.ranges[3]))) / T(length(grid.ranges[2]))

"""
    DetectorObservables{T <: Real, A <: AbstractArray}

Container holding physical optical observables extracted from a synthesized electric field:
- `intensity::A`: Optical intensity distribution in W/m².
- `phase::A`: Wavefront phase distribution in radians.
- `stokes::NTuple{4, A}`: Four-tuple of Stokes parameter arrays `(S0, S1, S2, S3)` in W/m².
- `power::T`: Total integrated optical power across the detector aperture in Watts.
"""
mutable struct DetectorObservables{T <: Real, A <: AbstractArray}
    intensity::A          # Total irradiance I = |E|² / (2 Z) [W/m²]
    phase::A              # Wavefront phase of the dominant field component [rad]
    stokes::NTuple{4, A}  # Transverse Stokes parameters (S0, S1, S2, S3) [W/m²]
    power::T              # Integrated optical power ∑ I ⋅ pixel_area [W]
end

@inline DetectorObservables(intensity::AbstractArray{T}, phase::AbstractArray{T}, stokes::NTuple{4, AbstractArray{T}}, power::Real = zero(T)) where {T <: Real} =
    DetectorObservables{T, typeof(intensity)}(intensity, phase, stokes, T(power))

# Preallocate uninitialized observable arrays of shape `dims` for zero-allocation in-place readout
@inline function DetectorObservables(dims::Dims{D}, ::Type{T_val}) where {D, T_val <: Real}
    return DetectorObservables{T_val, Array{T_val, D}}(
        Array{T_val, D}(undef, dims),
        Array{T_val, D}(undef, dims),
        ntuple(_ -> Array{T_val, D}(undef, dims), Val(4)),
        zero(T_val)
    )
end

@inline DetectorObservables(dims::Dims{D}; T::Type{T_val} = Float64) where {D, T_val <: Real} = DetectorObservables(dims, T_val)
@inline DetectorObservables(dims::Int...; T::Type{T_val} = Float64) where {T_val <: Real} = DetectorObservables(dims, T_val)

"""
    calculate_observables!(obs::DetectorObservables{T}, [d::FourierBeamletPropagator], field_or_grid...; pixel_area = 1.0, Z = Z_vacuum)

In-place evaluation of intensity, phase, Stokes parameters `(S0, S1, S2, S3)`, and integrated
optical power into a preallocated [`DetectorObservables`](@ref) container. When `field` is
passed directly, operates with zero heap allocations.
"""
function calculate_observables!(
    obs::DetectorObservables{T}, d::FourierBeamletPropagator,
    field::AbstractArray{SVector{3, Complex{T}}, D};
    pixel_area::Real = 1.0, Z::Real = Z_vacuum
) where {T <: Real, D}
    # Validate that preallocated output arrays match the input field dimensions
    if size(obs.intensity) != size(field) || size(obs.phase) != size(field) || any(s -> size(s) != size(field), obs.stokes)
        throw(DimensionMismatch("DetectorObservables array dimensions $(size(obs.intensity)) do not match field dimensions $(size(field))."))
    end

    # First pass: find the Cartesian field component with largest total energy for phase reference
    inv_2Z = inv(2 * T(Z))
    sx, sy, sz = zero(T), zero(T), zero(T)
    @inbounds for i in eachindex(field)
        Ef = field[i]
        sx += abs2(Ef[1]); sy += abs2(Ef[2]); sz += abs2(Ef[3])
    end
    dom_idx = (sx >= sy && sx >= sz) ? 1 : (sy >= sz ? 2 : 3)

    # Construct orthonormal detector basis: horizontal ex, vertical ez, and longitudinal normal en
    ex, ez = _detector_local_axes(d, T)
    en = normalize(cross(ex, ez))
    I_arr, phi_arr = obs.intensity, obs.phase
    s0_arr, s1_arr, s2_arr, s3_arr = obs.stokes

    # Second pass: project E(r) onto (ex, ez, en) and evaluate intensity, phase, and Stokes parameters
    P_sum = zero(T)
    @inbounds for i in eachindex(field)
        Ef = field[i]
        Eh, Ev, En = dot(ex, Ef), dot(ez, Ef), dot(en, Ef)
        Eh_sq, Ev_sq, En_sq = abs2(Eh), abs2(Ev), abs2(En)

        # Total 3D irradiance includes both transverse (Eh, Ev) and longitudinal (En) energy density
        E_trans_sq = (Eh_sq + Ev_sq) * inv_2Z
        I_val = max(zero(T), E_trans_sq + En_sq * inv_2Z)
        I_arr[i] = I_val
        P_sum += I_val
        phi_arr[i] = angle(Ef[dom_idx])

        # Transverse Stokes vector: S1 = |Eh|²-|Ev|², S2 = 2 Re(Eh* Ev), S3 = 2 Im(Eh* Ev)
        Eh_conj_Ev = conj(Eh) * Ev
        s1_val = (Eh_sq - Ev_sq) * inv_2Z
        s2_val = (2 * real(Eh_conj_Ev)) * inv_2Z
        s3_val = (2 * imag(Eh_conj_Ev)) * inv_2Z
        # Guard against floating-point rounding violating S0² >= S1² + S2² + S3²
        s_norm_sq = s1_val^2 + s2_val^2 + s3_val^2
        s0_val = max(zero(T), E_trans_sq, sqrt(s_norm_sq))
        s0_val^2 < s_norm_sq && (s0_val = nextfloat(s0_val))

        s0_arr[i], s1_arr[i], s2_arr[i], s3_arr[i] = s0_val, s1_val, s2_val, s3_val
    end

    # Integrate total optical power across the grid
    obs.power = max(zero(T), P_sum * T(pixel_area))
    return obs
end

# Convenience dispatches for calculate_observables! with SpatialGrid, range tuples, or default detector frame
calculate_observables!(obs::DetectorObservables{T}, d::FourierBeamletPropagator, field::AbstractArray{SVector{3, Complex{T}}, D}, grid::SpatialGrid{T, D}; Z::Real = Z_vacuum) where {T <: Real, D} =
    calculate_observables!(obs, d, field; pixel_area = _calculate_pixel_area(grid), Z = Z)

calculate_observables!(obs::DetectorObservables{T}, d::FourierBeamletPropagator, grid::SpatialGrid{T, D}; Z::Real = Z_vacuum) where {T <: Real, D} =
    calculate_observables!(obs, d, synthesize_field(d, grid), grid; Z = Z)

calculate_observables!(obs::DetectorObservables{T}, d::FourierBeamletPropagator, ranges::NTuple{D, AbstractRange}; Z::Real = Z_vacuum) where {T <: Real, D} =
    calculate_observables!(obs, d, SpatialGrid{T, D}(ranges); Z = Z)

calculate_observables!(obs::DetectorObservables{T}, field::AbstractArray{SVector{3, Complex{T}}, D}, grid::Union{SpatialGrid{T, D}, Nothing} = nothing; pixel_area::Real = isnothing(grid) ? 1.0 : _calculate_pixel_area(grid), Z::Real = Z_vacuum) where {T <: Real, D} =
    calculate_observables!(obs, FourierBeamletPropagator(one(T)), field; pixel_area = pixel_area, Z = Z)

"""
    calculate_observables(d::FourierBeamletPropagator, grid_or_ranges; Z = Z_vacuum)
    calculate_observables([d::FourierBeamletPropagator], field, [grid]; pixel_area = 1.0, Z = Z_vacuum)

Computes optical intensity, wavefront phase, Stokes polarization parameters `(S0, S1, S2, S3)`,
and total integrated optical power, returning a new [`DetectorObservables`](@ref) container.
"""
calculate_observables(d::FourierBeamletPropagator, field::AbstractArray{SVector{3, Complex{T}}, D}; pixel_area::Real = 1.0, Z::Real = Z_vacuum) where {T <: Real, D} =
    calculate_observables!(DetectorObservables(size(field), T), d, field; pixel_area = pixel_area, Z = Z)

calculate_observables(d::FourierBeamletPropagator, field::AbstractArray{SVector{3, Complex{T}}, D}, grid::SpatialGrid{T, D}; Z::Real = Z_vacuum) where {T <: Real, D} =
    calculate_observables(d, field; pixel_area = _calculate_pixel_area(grid), Z = Z)

calculate_observables(d::FourierBeamletPropagator, grid::SpatialGrid{T, D}; Z::Real = Z_vacuum) where {T <: Real, D} =
    calculate_observables(d, synthesize_field(d, grid), grid; Z = Z)

calculate_observables(d::FourierBeamletPropagator, ranges::NTuple{D, AbstractRange}; Z::Real = Z_vacuum) where {D} =
    calculate_observables(d, SpatialGrid(ranges); Z = Z)

calculate_observables(field::AbstractArray{SVector{3, Complex{T}}, D}, grid::Union{SpatialGrid{T, D}, Nothing} = nothing; pixel_area::Real = isnothing(grid) ? 1.0 : _calculate_pixel_area(grid), Z::Real = Z_vacuum) where {T <: Real, D} =
    calculate_observables(FourierBeamletPropagator(one(T)), field; pixel_area = pixel_area, Z = Z)

# ==============================================================================
# Standard BeamletOptics Detector Readout Verbs (Drop-in Compatibility)
# ==============================================================================

# Extract cosine surface projection factor for centroid weighting across BMO v0.13 and v0.14+
@inline function _safe_projection_factor(hit::AbstractRayHit, ::Type{T}) where {T <: Real}
    hasfield(typeof(hit), :proj) && return T(getfield(hit, :proj))
    inter = intersection(hit.ray)
    return isnothing(inter) ? one(T) : T(abs(dot(direction(hit), normal3d(inter))))
end

# Compute local (x, z) half-widths of a detector mesh from its transformed vertices
function _shape_half_extents(m::AbstractMesh, ::Type{T}) where {T <: Real}
    local_verts = (vertices(m) .- position(m)') * orientation(m)
    ex1, ex3 = extrema(view(local_verts, :, 1)), extrema(view(local_verts, :, 3))
    return (safe_pos(T((ex1[2] - ex1[1]) / 2), T(0.5e-3)), safe_pos(T((ex3[2] - ex3[1]) / 2), T(0.5e-3)))
end
_shape_half_extents(::AbstractShape, ::Type{T}) where {T <: Real} = (T(0.5e-3), T(0.5e-3))

# Project a single ray hit onto local 2D detector coordinates (x, z)
function _append_hit_spots!(
    pts_2D::Vector{Point2{T}}, hit::AbstractRayHit, p0::SVector{3, T},
    ex::SVector{3, T}, ::SVector{3, T}, ez::SVector{3, T}, ::T, ::Int
) where {T <: Real}
    v = _extract_hit_position(hit, T) - p0
    push!(pts_2D, Point2{T}(dot(v, ex), dot(v, ez)))
    return pts_2D
end

# Project a ring of `num_spots` points spanned by ellipse axes (a1, a2) onto the detector plane
function _project_spot_ring!(
    pts_2D::Vector{Point2{T}}, origin::SVector{3, T}, dir::SVector{3, T},
    a1::SVector{3, T}, a2::SVector{3, T}, p0::SVector{3, T},
    ex::SVector{3, T}, ey::SVector{3, T}, ez::SVector{3, T}, num_spots::Int
) where {T <: Real}
    for ts in LinRange(zero(T), T(2π), num_spots)
        pt = origin + cos(ts) * a1 + sin(ts) * a2
        # Intersect ray parallel to `dir` through `pt` with the detector plane (p0, ey)
        dist = line_plane_distance3d(p0, ey, pt, dir)
        isnothing(dist) && continue
        v = (pt + T(dist) * dir) - p0
        push!(pts_2D, Point2{T}(dot(v, ex), dot(v, ez)))
    end
    return pts_2D
end

# Sample circular waist contour of a GaussianBeamletHit for spot_diagram
function _append_hit_spots!(
    pts_2D::Vector{Point2{T}}, hit::GaussianBeamletHit, p0::SVector{3, T},
    ex::SVector{3, T}, ey::SVector{3, T}, ez::SVector{3, T}, crop_factor::T, num_spots::Int
) where {T <: Real}
    w = T(hit.w_max) * crop_factor
    dir = _extract_hit_direction(hit, T)
    nd1 = SVector{3, T}(normal3d(dir))
    return _project_spot_ring!(pts_2D, _extract_hit_position(hit, T), dir, w * nd1, w * cross(dir, nd1), p0, ex, ey, ez, num_spots)
end

# Sample elliptical waist contour of an AstigmaticGaussianBeamletHit for spot_diagram
function _append_hit_spots!(
    pts_2D::Vector{Point2{T}}, hit::AstigmaticGaussianBeamletHit, p0::SVector{3, T},
    ex::SVector{3, T}, ey::SVector{3, T}, ez::SVector{3, T}, crop_factor::T, num_spots::Int
) where {T <: Real}
    @static if hasfield(AstigmaticGaussianBeamletHit, :l_seg)
        l_seg = safe_finite(T(hit.l_seg), zero(T))
    else
        l_seg = safe_finite(T(length(rays(hit.agb.c)[hit.id])), zero(T))
    end
    # Propagate complex parabasal vectors to detector and take real spatial ellipse semi-axes
    a1 = SVector{3, T}(real((hit.h1 + l_seg * hit.u1) * crop_factor))
    a2 = SVector{3, T}(real((hit.h2 + l_seg * hit.u2) * crop_factor))
    return _project_spot_ring!(pts_2D, _extract_hit_position(hit, T), _extract_hit_direction(hit, T), a1, a2, p0, ex, ey, ez, num_spots)
end

_append_hit_spots!(pts_2D::Vector{Point2{T}}, ::AbstractDetectorHit, ::SVector{3, T}, ::SVector{3, T}, ::SVector{3, T}, ::SVector{3, T}, ::T, ::Int) where {T <: Real} = pts_2D

"""
    spot_diagram(d::FourierBeamletPropagator; crop_factor::Real = 1.0, num_spots::Int = 50)

Returns `Vector{Point2{T}}` in local detector `(x, z)` coordinates representing captured ray
intersections and projected Gaussian beamlet waists, matching `BeamletOptics.spot_diagram(::Detector)`.
"""
function spot_diagram(d::FourierBeamletPropagator{T}; crop_factor::Real = 1.0, num_spots::Int = 50) where {T <: Real}
    isempty(d.hits) && throw(ErrorException("No hits available on detector."))
    # Set up detector-local coordinate frame (origin p0, transverse axes ex/ez, normal ey)
    p0 = SVector{3, T}(position(d))
    ex, ez = _detector_local_axes(d, T)
    o = orientation(d.shape)
    ey = SVector{3, T}(T(o[1, 2]), T(o[2, 2]), T(o[3, 2]))
    pts_2D = Point2{T}[]
    for hit in d.hits
        _append_hit_spots!(pts_2D, hit, p0, ex, ey, ez, T(crop_factor), num_spots)
    end
    return pts_2D
end

# Determine automatic 2D evaluation bounds (xmin, xmax, zmin, zmax) from captured hits or detector shape
function _local_lims_fourier(
    d::FourierBeamletPropagator{T};
    crop_factor::Union{Nothing, Real} = nothing,
    center::AbstractCenterAlgorithm = Centroid()
) where {T <: Real}
    isempty(d.hits) && throw(ErrorException("No hits available on detector."))
    hx_shape, hz_shape = _shape_half_extents(d.shape, T)
    p0 = SVector{3, T}(position(d))
    ex, ez = _detector_local_axes(d, T)

    if all(h -> h isa AbstractRayHit, d.hits)
        # Pure ray hits: center window around weighted centroid (or MinMax) of ray intersections
        cf = isnothing(crop_factor) ? one(T) : T(crop_factor)
        n_hits = length(d.hits)
        xs, zs, ws = Vector{T}(undef, n_hits), Vector{T}(undef, n_hits), Vector{T}(undef, n_hits)
        for (idx, hit) in enumerate(d.hits)
            v = _extract_hit_position(hit, T) - p0
            xs[idx], zs[idx], ws[idx] = dot(v, ex), dot(v, ez), _safe_projection_factor(hit, T)
        end
        x0, z0 = calc_center_point(center, xs, zs, ws)
        dx, dz = maximum(x -> abs(x - x0), xs), maximum(z -> abs(z - z0), zs)
        hwx, hwz = (dx > zero(T) ? dx : hx_shape) * cf, (dz > zero(T) ? dz : hz_shape) * cf
        return (x0 - hwx, x0 + hwx, z0 - hwz, z0 + hwz)
    elseif any(h -> h isa AbstractBeamletHit, d.hits)
        # Beamlet hits: enclose all beamlet spot radii w_max * crop_factor
        cf = isnothing(crop_factor) ? T(1.5) : T(crop_factor)
        xmin, xmax, zmin, zmax = T(Inf), T(-Inf), T(Inf), T(-Inf)
        for hit in d.hits
            (hit isa AbstractBeamletHit || hit isa AbstractRayHit) || continue
            v = _extract_hit_position(hit, T) - p0
            x, z = dot(v, ex), dot(v, ez)
            w = hit isa AbstractBeamletHit ? T(hit.w_max) * cf : zero(T)
            xmin, xmax = min(xmin, x - w), max(xmax, x + w)
            zmin, zmax = min(zmin, z - w), max(zmax, z + w)
        end
        (xmin < xmax && zmin < zmax) && return (xmin, xmax, zmin, zmax)
    end

    # Fallback to geometric mesh half-extents
    cf = isnothing(crop_factor) ? one(T) : T(crop_factor)
    return (-hx_shape * cf, hx_shape * cf, -hz_shape * cf, hz_shape * cf)
end

# 1D and 2D composite trapezoidal quadrature weights for optical_power integration
function _trapz_1d_weights(xs::AbstractVector{T}) where {T <: Real}
    w = zeros(T, length(xs))
    @inbounds for i in 1:(length(xs) - 1)
        h = (xs[i + 1] - xs[i]) / T(2)
        w[i] += h; w[i + 1] += h
    end
    return w
end

function _trapz_2d(xs::AbstractVector{T}, zs::AbstractVector{T}, I::AbstractMatrix{T}) where {T <: Real}
    wx, wz = _trapz_1d_weights(xs), _trapz_1d_weights(zs)
    total = zero(T)
    @inbounds for j in axes(I, 2), i in axes(I, 1)
        total += wx[i] * wz[j] * I[i, j]
    end
    return total
end

# Synthesize 2D field on (xs, zs) and convert SVector elements to BMO's Point3{Complex{T}} convention
function _electric_field_on_ranges(d::FourierBeamletPropagator{T}, xs::LinRange{T, Int}, zs::LinRange{T, Int}, L_box::Real) where {T <: Real}
    E_sv = synthesize_field(d, SpatialGrid{T, 2}((xs, zs)); L_box = L_box)
    return (xs, zs, Point3{Complex{T}}.(E_sv))
end

"""
    electric_field(d::FourierBeamletPropagator; n = 100, crop_factor = nothing,
                   center = Centroid(), x_min = Inf, x_max = Inf, z_min = Inf, z_max = Inf,
                   x0_shift = 0, z0_shift = 0, L_box = 1.0, kwargs...)

Evaluates the 2D complex electric field `(xs, zs, E)` on a regular `n × n` grid in the
detector's local `(x, z)` coordinates via NUFFT synthesis, providing drop-in compatibility
with `BeamletOptics.electric_field(::Detector)`. Returns `E::Matrix{Point3{Complex{T}}}`.
"""
function electric_field(
    d::FourierBeamletPropagator{T};
    n::Int = 100,
    crop_factor::Union{Nothing, Real} = nothing,
    center::AbstractCenterAlgorithm = Centroid(),
    x_min::Real = Inf, x_max::Real = Inf,
    z_min::Real = Inf, z_max::Real = Inf,
    x0_shift::Real = 0, z0_shift::Real = 0,
    L_box::Real = 1.0,
    kwargs...
) where {T <: Real}
    # Determine local grid limits and override with explicit bounds if finite
    n >= 4 || throw(ArgumentError("Grid resolution n must be >= 4 for NUFFT synthesis (got n = $n)."))
    _xmin, _xmax, _zmin, _zmax = _local_lims_fourier(d; crop_factor = crop_factor, center = center)
    (isfinite(x_min) && isfinite(x_max)) && ((_xmin, _xmax) = (T(x_min), T(x_max)))
    (isfinite(z_min) && isfinite(z_max)) && ((_zmin, _zmax) = (T(z_min), T(z_max)))
    xs = LinRange{T, Int}(_xmin + T(x0_shift), _xmax + T(x0_shift), n)
    zs = LinRange{T, Int}(_zmin + T(z0_shift), _zmax + T(z0_shift), n)
    return _electric_field_on_ranges(d, xs, zs, L_box)
end

"""
    intensity(d::FourierBeamletPropagator, Z::Number = Z_vacuum; kwargs...)

Calculates the 2D optical intensity distribution `(xs, zs, I)` in W/m² on `d`, matching
`BeamletOptics.intensity(::Detector)`.
"""
function intensity(d::FourierBeamletPropagator, Z::Number = Z_vacuum; kwargs...)
    xs, zs, E = electric_field(d; kwargs...)
    return (xs, zs, intensity.(E, Z))
end

"""
    optical_power(d::FourierBeamletPropagator; Z::Number = Z_vacuum, kwargs...)

Calculates the integrated optical power in Watts across the 2D evaluation grid on `d`,
matching `BeamletOptics.optical_power(::Detector)`.
"""
function optical_power(d::FourierBeamletPropagator{T}; Z::Number = Z_vacuum, kwargs...) where {T <: Real}
    xs, zs, I = intensity(d, Z; kwargs...)
    return _trapz_2d(xs, zs, I)
end

# BMO v0.14+ sampling grid dispatches (RegularGrid & RectilinearGrid)
@static if isdefined(BeamletOptics, :RegularGrid)
    electric_field(d::FourierBeamletPropagator, g::BeamletOptics.RegularGrid; L_box::Real = 1.0, kwargs...) =
        electric_field(d; n = g.num_points, crop_factor = g.crop_factor, center = g.center,
            x_min = something(g.x_min, Inf), x_max = something(g.x_max, Inf),
            z_min = something(g.z_min, Inf), z_max = something(g.z_max, Inf),
            x0_shift = g.x0_shift, z0_shift = g.z0_shift, L_box = L_box)

    function electric_field(d::FourierBeamletPropagator{T}, g::BeamletOptics.RectilinearGrid; L_box::Real = 1.0, kwargs...) where {T <: Real}
        xs = LinRange{T, Int}(T(first(g.xs)), T(last(g.xs)), length(g.xs))
        zs = LinRange{T, Int}(T(first(g.zs)), T(last(g.zs)), length(g.zs))
        return _electric_field_on_ranges(d, xs, zs, L_box)
    end

    function intensity(d::FourierBeamletPropagator, g::BeamletOptics.AbstractDetectorSampling, Z::Number = Z_vacuum; kwargs...)
        xs, zs, E = electric_field(d, g; kwargs...)
        return (xs, zs, intensity.(E, Z))
    end

    function optical_power(d::FourierBeamletPropagator{T}, g::BeamletOptics.AbstractDetectorSampling; Z::Number = Z_vacuum, kwargs...) where {T <: Real}
        xs, zs, I = intensity(d, g, Z; kwargs...)
        return _trapz_2d(xs, zs, I)
    end
end

# ==============================================================================
# Internal Detector Hit Accessors (Zero Type Piracy) & Smooth Polarization Basis
# ==============================================================================

# Optical path length accessors across ray and beamlet hit types (v0.13 and v0.14+)
@inline _hit_opl(hit::AbstractRayHit) = optical_path_length(hit)

@static if hasfield(GaussianBeamletHit, :segment)
    # BMO v0.14+: OPL is pre-split into geometric l0, optical path difference Δl0, and n * l_seg
    @inline _hit_opl(hit::GaussianBeamletHit{T}) where {T} =
        hit.l0 + hit.Δl0 + hit.segment.n * safe_finite(hit.l_seg, zero(T))
else
    # BMO v0.13: accumulate OPL along parent beams and intersected chief-ray segments
    @inline function _hit_opl(hit::GaussianBeamletHit{T}) where {T}
        chief_beam = hit.gauss.chief
        p_parent = !isnothing(chief_beam.parent) ? chief_beam.parent : hit.gauss.parent
        opl = !isnothing(p_parent) ? optical_path_length(p_parent)::T : zero(T)
        chief_rays = rays(chief_beam)
        for i in 1:clamp(hit.id, 0, length(chief_rays))
            ray = chief_rays[i]
            isnothing(intersection(ray)) && break
            opl += optical_path_length(ray)::T
        end
        return opl
    end
end

@static if hasfield(AstigmaticGaussianBeamletHit, :Δl0)
    @inline _hit_opl(hit::AstigmaticGaussianBeamletHit{T}) where {T} =
        hit.l0 + hit.Δl0 + hit.n * safe_finite(hit.l_seg, zero(T))
else
    @inline _hit_opl(hit::AstigmaticGaussianBeamletHit) =
        hit.l0 + optical_path_length(hit.agb.c.rays[hit.id]) + hit.Δl
end

# Vacuum wavelength accessors across BMO v0.13 and v0.14+ hit structures
@static if hasfield(RayHit, :λ)
    @inline _hit_wavelength(hit::WeightedRayHit) = wavelength(hit.ray)
    @inline _hit_wavelength(hit::AbstractRayHit) = wavelength(hit)
else
    @inline _hit_wavelength(hit::AbstractRayHit) = wavelength(hit.ray)
end

@static if hasfield(GaussianBeamletHit, :segment)
    @inline _hit_wavelength(hit::GaussianBeamletHit) = hit.segment.λ
else
    @inline _hit_wavelength(hit::GaussianBeamletHit) = wavelength(hit.gauss)
end

@static if hasfield(AstigmaticGaussianBeamletHit, :λ)
    @inline _hit_wavelength(hit::AstigmaticGaussianBeamletHit) = hit.λ
else
    @inline _hit_wavelength(hit::AstigmaticGaussianBeamletHit) = wavelength(hit.agb)
end

@inline _hit_wavelength(hit::FourierKPoint) = wavelength(hit)
@inline _hit_wavelength(hit::AbstractDetectorHit) = wavenumber(hit) > 0 ? (2π / wavenumber(hit)) : 1.0

# Complex electric-field polarization accessors at the beam waist / hit point
@inline _hit_polarization(hit::Union{PolarizedRayHit, FourierKPoint}, _ = nothing) = polarization(hit)
@inline _hit_polarization(hit::AbstractRayHit{T}, _ = nothing) where {T} = Point3{Complex{T}}(transverse_polarization(direction(hit)))

@static if hasfield(GaussianBeamletHit, :segment)
    # BMO v0.14+: recover waist electric field E0 from invariant E_amp = E0 * sqrt(n) * w0
    function _hit_polarization(hit::GaussianBeamletHit{T}, r_hit::SVector{3, T}) where {T}
        _, _, _, w0_eval = _beamlet_gauss_params(hit, r_hit)
        w0 = safe_pos(T(w0_eval), T(hit.w_max))
        amp = hit.E_amp / (sqrt(hit.segment.n) * w0)
        return Point3{Complex{T}}(transverse_polarization(direction(hit)) * amp)
    end
else
    function _hit_polarization(hit::GaussianBeamletHit{T}, _ = nothing) where {T}
        chief_ray = hit.gauss.chief.rays[hit.id]
        chief_ray isa PolarizedRay && return polarization(chief_ray)
        return Point3{Complex{T}}(transverse_polarization(direction(hit)) * hit.gauss.E0)
    end
end

@static if hasfield(AstigmaticGaussianBeamletHit, :E0)
    # BMO v0.14+: hit.E0 stores electric field vector scaled by 1/|gouy0|; multiply by |gouy0|
    _hit_polarization(hit::AstigmaticGaussianBeamletHit{T}, _ = nothing) where {T} = hit.E0 * T(abs(hit.gouy0))
else
    function _hit_polarization(hit::AstigmaticGaussianBeamletHit{T}, _ = nothing) where {T}
        chief_ray = hit.agb.c.rays[hit.id]
        chief_ray isa PolarizedRay && return polarization(chief_ray)
        return Point3{Complex{T}}(transverse_polarization(direction(hit)) * hit.E_ref_amp)
    end
end

"""
    transverse_polarization(dir::AbstractVector{T}) where {T}

Constructs a smooth, singularity-free unit vector transverse to propagation direction `dir`
over the forward hemisphere (Tom Duff orthonormal basis construction).
"""
@inline function transverse_polarization(dir::AbstractVector{T}) where {T}
    # Normalize propagation vector and place the stereographic branch cut at z = -1
    v = normalize(SVector{3, T}(dir[1], dir[2], dir[3]))
    if v[3] > -T(0.99999)
        inv_1_pz = one(T) / (one(T) + v[3])
        return Point3{T}(one(T) - v[1]^2 * inv_1_pz, -v[1] * v[2] * inv_1_pz, -v[1])
    else
        # Antipodal chart near v_z ≈ -1 to avoid division by zero
        inv_1_mz = one(T) / (one(T) - v[3])
        return Point3{T}(one(T) - v[1]^2 * inv_1_mz, -v[1] * v[2] * inv_1_mz, v[1])
    end
end
