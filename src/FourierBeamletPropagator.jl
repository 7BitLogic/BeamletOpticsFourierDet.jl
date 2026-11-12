# ==============================================================================
# FourierBeamletPropagator.jl
#
# Fourier-optics detector and propagator transforming incident rays and Gaussian
# beamlets into discrete plane-wave k-points (FourierKPoint) for NUFFT-based
# electromagnetic field synthesis. Compatible with BeamletOptics v0.13 and v0.14+.
# ==============================================================================

"""
    FourierKPoint{T <: Real} <: AbstractDetectorHit

Discrete frequency-space point representing an electromagnetic plane-wave component.

# Fields
- `k4::SVector{4, T}`: Four-wavevector `[kx, ky, kz, omega]` with spatial wavevector in rad/m and angular frequency in rad/s.
- `E::SVector{3, Complex{T}}`: Complex 3D electric field vector `[Ex, Ey, Ez]` in V/m.
- `weight::T`: Quadrature / spectral weight.

Provides property accessors `p.k` (3D wavevector) and `p.omega` (angular frequency).
"""
struct FourierKPoint{T <: Real} <: AbstractDetectorHit
    k4::SVector{4, T}          # 4-wavevector: [kx, ky, kz, ω]
    E::SVector{3, Complex{T}}  # Complex 3D electric fieldphasor [Ex, Ey, Ez]
    weight::T                  # Spectral quadrature weight

    @inline FourierKPoint{T}(k4::SVector{4, T}, E::SVector{3, Complex{T}}, weight::T = one(T)) where {T <: Real} =
        new{T}(k4, E, weight)
end

# Outer constructors for static vectors, generic vectors, and (k_3d, omega) pairs
@inline FourierKPoint(k4::SVector{4, T}, E::SVector{3, Complex{T}}, weight::T = one(T)) where {T <: Real} =
    FourierKPoint{T}(k4, E, weight)

@inline FourierKPoint{T}(k4::AbstractVector{<:Real}, E::AbstractVector{<:Number}, weight::Real = one(T)) where {T <: Real} =
    FourierKPoint{T}(SVector{4, T}(k4), SVector{3, Complex{T}}(E), T(weight))

@inline function FourierKPoint(k4::AbstractVector{<:Real}, E::AbstractVector{<:Number}, weight::Real = 1.0)
    # Promote element types across wavevector, complex field, and scalar weight
    T = promote_type(eltype(k4), real(eltype(E)), typeof(float(weight)))
    return FourierKPoint{T}(k4, E, weight)
end

@inline FourierKPoint{T}(k::AbstractVector{<:Real}, omega::Real, E::AbstractVector{<:Number}, weight::Real = one(T)) where {T <: Real} =
    FourierKPoint{T}(SVector{4, T}(k[1], k[2], k[3], omega), E, weight)

@inline FourierKPoint(k::AbstractVector{<:Real}, omega::Real, E::AbstractVector{<:Number}, weight::Real = 1.0) =
    FourierKPoint(SVector{4}(k[1], k[2], k[3], omega), E, weight)

# Virtual property accessors: `p.k` extracts spatial 3-vector, `p.omega` extracts angular frequency
@inline Base.getproperty(p::FourierKPoint{T}, s::Symbol) where {T} =
    s === :k ? SVector{3, T}(getfield(p, :k4)[1], getfield(p, :k4)[2], getfield(p, :k4)[3]) :
    s === :omega ? getfield(p, :k4)[4] : getfield(p, s)

@inline Base.propertynames(::FourierKPoint, ::Bool = false) = (:k4, :E, :weight, :k, :omega)

# BeamletOptics hit-interface methods for frequency-space k-points
@inline wavenumber(p::FourierKPoint) = norm(p.k)
@inline wavelength(p::FourierKPoint{T}) where {T} = (k_mag = norm(p.k); k_mag > 0 ? (T(2π) / k_mag) : one(T))
@inline direction(p::FourierKPoint{T}) where {T} = (k_mag = norm(p.k); k_mag > 0 ? (p.k / k_mag) : SVector{3, T}(zero(T), one(T), zero(T)))
@inline position(::FourierKPoint{T}) where {T} = SVector{3, T}(zero(T), zero(T), zero(T))
@inline hit_point(::FourierKPoint{T}) where {T} = Point3{T}(zero(T), zero(T), zero(T))
@inline polarization(p::FourierKPoint) = p.E

"""
    WeightedRayHit{T <: Real} <: AbstractRayHit{T}

Detector hit record storing an unpolarized geometric [`Ray`](@ref), its accumulated
optical path length `opl` in meters, and an explicit ray optical power `power` in Watts.
"""
struct WeightedRayHit{T <: Real} <: AbstractRayHit{T}
    ray::Ray{T}  # Underlying geometric ray segment
    opl::T       # Accumulated optical path length [m]
    power::T     # Ray power weight [W]
end

@inline WeightedRayHit(ray::Ray{T}, opl::Real, power::Real = one(T)) where {T <: Real} =
    WeightedRayHit{T}(ray, T(opl), T(power))

# Forward standard AbstractRayHit queries to the underlying geometric ray
@inline position(hit::WeightedRayHit) = position(hit.ray)
@inline direction(hit::WeightedRayHit) = direction(hit.ray)
@inline Base.length(hit::WeightedRayHit) = length(hit.ray)
@inline optical_path_length(hit::WeightedRayHit) = hit.opl
@inline wavelength(hit::WeightedRayHit) = wavelength(hit.ray)
@inline wavenumber(hit::WeightedRayHit) = wavenumber(hit.ray)
@inline refractive_index(hit::WeightedRayHit) = refractive_index(hit.ray)
# Evaluate 3D intersection point if ray length is finite, otherwise fall back to ray origin
@inline hit_point(hit::WeightedRayHit) =
    isfinite(length(hit.ray)) ? (position(hit.ray) + length(hit.ray) * direction(hit.ray)) : position(hit.ray)

"""
    AbstractFourierDetector{T} <: AbstractDetector{T}

Abstract supertype for Fourier-optics detectors based on frequency-space decomposition
and non-uniform fast Fourier transforms.
"""
abstract type AbstractFourierDetector{T} <: AbstractDetector{T} end

"""
    FourierBeamletPropagator{T <: Real, S <: AbstractShape{T}} <: AbstractFourierDetector{T}

Fourier-optics detector and propagator that captures ray and Gaussian beamlet hits and
evaluates their discrete angular spectrum (`Vector{FourierKPoint{T}}`) and spatial field.

# Constructors
- `FourierBeamletPropagator(edge_length::Real; stop::Bool = true, is_planar::Bool = false)`
- `FourierBeamletPropagator(x_edge::Real, z_edge::Real; stop::Bool = true, is_planar::Bool = true)`
- `FourierBeamletPropagator(dx::Real, dy::Real, dz::Real; stop::Bool = true, is_planar::Bool = false)`
- `FourierBeamletPropagator(shape::AbstractShape; stop::Bool = true, is_planar::Bool = false)`
- `FourierBeamletPropagator(object::AbstractObject; stop::Bool = true, is_planar::Bool = false)`
"""
mutable struct FourierBeamletPropagator{T <: Real, S <: AbstractShape{T}} <: AbstractFourierDetector{T}
    const shape::S                     # Geometric detector surface or volume mesh
    hits::Vector{<:AbstractDetectorHit} # Captured ray, beamlet, or k-space hits
    stop::Bool                         # True = absorb beams; false = passive observer
    is_planar::Bool                    # True = apply planar obliquity factor sqrt(cos θ)
end

# Construct directly from an existing BMO shape
FourierBeamletPropagator(shape::S; stop::Bool = true, is_planar::Bool = false) where {T <: Real, S <: AbstractShape{T}} =
    FourierBeamletPropagator{T, S}(shape, AbstractDetectorHit[], stop, is_planar)

# Single-argument constructor: creates either a 2D square screen or a 3D cubic volume
function FourierBeamletPropagator(edge_length::Real; stop::Bool = true, is_planar::Bool = false)
    s = Float64(edge_length)
    is_planar && return FourierBeamletPropagator(s, s; stop = stop, is_planar = true)
    return FourierBeamletPropagator(s, s, s; stop = stop, is_planar = false)
end

# Two-argument constructor: planar rectangular detector in the X-Z plane facing -Y
function FourierBeamletPropagator(x_edge::Real, z_edge::Real; stop::Bool = true, is_planar::Bool = true)
    # Choose quadratic or rectangular flat mesh and rotate normal to face incoming +Y beams
    mesh = Float64(x_edge) == Float64(z_edge) ? QuadraticFlatMesh(Float64(x_edge)) : RectangularFlatMesh(Float64(x_edge), Float64(z_edge))
    zrotate3d!(mesh, π)
    return FourierBeamletPropagator(mesh; stop = stop, is_planar = is_planar)
end

# Three-argument constructor: volumetric cuboid detector centered at the origin
function FourierBeamletPropagator(dx::Real, dy::Real, dz::Real; stop::Bool = true, is_planar::Bool = false)
    mesh = (dx == dy == dz) ? CubeMesh(Float64(dx)) : CuboidMesh(Float64(dx), Float64(dy), Float64(dz))
    # Shift vertices so the geometric center sits at [0, 0, 0] while resetting local pos
    translate3d!(mesh, [-dx / 2, -dy / 2, -dz / 2])
    mesh.pos = zeros(Float64, 3)
    return FourierBeamletPropagator(mesh; stop = stop, is_planar = is_planar)
end

# Clone geometry from any existing BMO object
FourierBeamletPropagator(object::AbstractObject; stop::Bool = true, is_planar::Bool = false) =
    FourierBeamletPropagator(deepcopy(shape(object)); stop = stop, is_planar = is_planar)

# Standard BMO detector container & inspector interface
shape(d::FourierBeamletPropagator) = d.shape
hits(d::FourierBeamletPropagator) = d.hits
hits!(d::FourierBeamletPropagator, new) = (d.hits = new)
stop(d::FourierBeamletPropagator) = d.stop
@static if isdefined(BeamletOptics, :hit_count)
    BeamletOptics.hit_count(d::FourierBeamletPropagator) = length(d.hits)
end
properties(d::FourierBeamletPropagator) = [
    default_properties(d);
    "Hits" => length(d.hits);
    "Stops beams" => stop(d);
    "Planar" => d.is_planar
]
Base.empty!(d::FourierBeamletPropagator) = (empty!(d.hits); d)
Base.isempty(d::FourierBeamletPropagator) = isempty(d.hits)
Base.length(d::FourierBeamletPropagator) = length(d.hits)

# Reentrant lock protecting concurrent hit insertions during multithreaded ray tracing
const _PROPAGATOR_HIT_LOCK = Base.ReentrantLock()

@inline function Base.push!(d::FourierBeamletPropagator, hit::H) where {H <: AbstractDetectorHit}
    lock(_PROPAGATOR_HIT_LOCK) do
        h = d.hits
        if isempty(h) && !(h isa Vector{H})
            # Narrow empty storage vector to the concrete type of the first hit
            d.hits = H[hit]
        elseif hit isa eltype(h)
            # Fast path for homogeneous hit types
            push!(h, hit)
        else
            # Widen container element type via typejoin when heterogeneous hits arrive
            T_super = typejoin(eltype(h), H)
            d.hits = push!(copyto!(Vector{T_super}(undef, length(h)), h), hit)
        end
    end
    return d
end

# ------------------------------------------------------------------------------
# interact3d methods (passive observer when stop = false; supports BMO 0.13 & 0.14+)
# ------------------------------------------------------------------------------

@static if hasfield(RayHit, :p)
    # BMO v0.14+: value-based RayHit / PolarizedRayHit storing hit point and projection factor
    @inline function _ray_hit(ray::AbstractRay{T}, opl) where {T}
        isfinite(length(ray)) && return (ray isa PolarizedRay ? PolarizedRayHit(ray, T(opl)) : RayHit(ray, T(opl)))
        # Direct `interact3d(d, ray)` call without prior surface intersection: use ray origin
        p, dir = Point3{T}(position(ray)), Point3{T}(direction(ray))
        inter = intersection(ray)
        proj = isnothing(inter) ? one(T) : T(abs(dot(dir, normal3d(inter))))
        return ray isa PolarizedRay ?
            PolarizedRayHit{T}(p, dir, T(opl), T(wavelength(ray)), T(refractive_index(ray)), proj, Point3{Complex{T}}(polarization(ray))) :
            RayHit{T}(p, dir, T(opl), T(wavelength(ray)), T(refractive_index(ray)), proj)
    end
else
    # BMO v0.13: reference-based RayHit / PolarizedRayHit wrapping (ray, opl)
    @inline _ray_hit(ray::PolarizedRay, opl) = PolarizedRayHit(ray, opl)
    @inline _ray_hit(ray::AbstractRay, opl) = RayHit(ray, opl)
end

@inline function interact3d(::AbstractSystem, d::FourierBeamletPropagator, beam::Beam, ray::AbstractRay)
    # Record ray hit with finite accumulated OPL; absorb or pass through based on `d.stop`
    Base.push!(d, _ray_hit(ray, safe_finite(optical_path_length(beam))))
    return d.stop ? nothing : BeamInteraction(nothing, ray)
end

function interact3d(::AbstractSystem, d::FourierBeamletPropagator, g::GaussianBeamlet{R}, id::Int = 1) where {R}
    @static if hasfield(GaussianBeamletHit, :segment)
        # BMO v0.14+: value-based GaussianBeamletHit constructor
        Base.push!(d, GaussianBeamletHit(g, id))
    else
        # BMO v0.13: compute segment offset l0, surface obliquity, and spot radius w_max
        ray = g.chief.rays[id]
        l0 = safe_finite(length(g) - length(ray), zero(R))
        p0, d0 = position(ray), direction(ray)
        inter = intersection(ray)
        sqrt_proj = isnothing(inter) ? one(R) : sqrt(abs(dot(d0, normal3d(inter))))
        w_at_det, _, _, _ = gauss_parameters(g, safe_finite(length(g), zero(R)))
        Base.push!(d, GaussianBeamletHit(g, l0, id, p0, d0, sqrt_proj, safe_pos(w_at_det, g.w0)))
    end
    # Return pass-through interaction for chief, waist, and divergence rays when stop == false
    return d.stop ? nothing : GaussianBeamletInteraction{R}(
        BeamInteraction(nothing, g.chief.rays[id]),
        BeamInteraction(nothing, g.waist.rays[id]),
        BeamInteraction(nothing, g.divergence.rays[id]),
    )
end

function interact3d(::AbstractSystem, d::FourierBeamletPropagator, agb::AstigmaticGaussianBeamlet{R}, id::Int = 1) where {R}
    @static if hasfield(AstigmaticGaussianBeamletHit, :E0)
        # BMO v0.14+: value-based AstigmaticGaussianBeamletHit constructor
        Base.push!(d, AstigmaticGaussianBeamletHit(agb, id))
    else
        # BMO v0.13: evaluate parabasal ray parameters, reference area, and accumulated OPL
        chief = rays(agb.c)[id]
        l0 = safe_finite(length(agb) - length(chief), zero(R))
        p0, d0, k0 = position(chief), direction(chief), 2π / wavelength(chief)
        inter = intersection(chief)
        sqrt_proj = isnothing(inter) ? one(R) : sqrt(abs(dot(d0, normal3d(inter))))
        h1, u1, h2, u2, _ = BeamletOptics.parabasal_ray_parameters(agb, p0, id)
        # Compute reference cross-sectional area and complex amplitude at beam origin
        p0n, in_ = BeamletOptics.point_on_beam(agb, 0.0)
        h1n, _, h2n, _, _ = BeamletOptics.parabasal_ray_parameters(agb, p0n, in_)
        area_ref = BeamletOptics._pseudo_cross2d(h1n, h2n, direction(rays(agb.c)[in_]))
        E_vec = polarization(rays(agb.c)[in_])
        E_ref_amp = Complex{R}(norm(E_vec) * cis(angle(E_vec[argmax(abs.(E_vec))])))
        # Sum optical path difference Δl = OPL - geometric length along preceding segments
        p_parent = agb.parent
        Δl = isnothing(p_parent) ? 0.0 : (optical_path_length(p_parent) - length(p_parent))
        for j in 1:(id - 1)
            rj = rays(agb.c)[j]
            Δl += optical_path_length(rj) - length(rj)
        end
        # Propagate parabasal heights to detector plane to bound the maximum spot radius
        l_eval = safe_finite(length(chief), zero(R))
        w_max = safe_pos(max(norm(h1 + l_eval * u1), norm(h2 + l_eval * u2)), max(norm(h1), norm(h2)))
        Base.push!(d, AstigmaticGaussianBeamletHit(agb, l0, id, p0, d0, h1, u1, h2, u2,
            area_ref, k0, Δl, refractive_index(agb, id), E_ref_amp, sqrt_proj, w_max))
    end
    # Continue tracing all 9 chief + parabasal rays when acting as a passive observer
    return d.stop ? nothing : AstigmaticGaussianBeamletInteraction{R}(
        map(b -> BeamInteraction(nothing, rays(b)[id]), (agb.c, agb.wxp, agb.wxm, agb.wyp, agb.wym, agb.dxp, agb.dxm, agb.dyp, agb.dym))...
    )
end

# Convenience interact3d dispatches for standalone rays, beams, and ray vectors
interact3d(d::FourierBeamletPropagator, ray::AbstractRay) = interact3d(StaticSystem((d,)), d, Beam(ray), ray)
interact3d(sys::AbstractSystem, d::FourierBeamletPropagator, ray::AbstractRay) = interact3d(sys, d, Beam(ray), ray)
interact3d(d::FourierBeamletPropagator, beam::Beam) = (for r in rays(beam) interact3d(d, r) end; d.stop ? nothing : beam)
interact3d(d::FourierBeamletPropagator, rs::AbstractVector{<:AbstractRay}) = (for r in rs interact3d(d, r) end; d.stop ? nothing : rs)
interact3d(::AbstractSystem, d::FourierBeamletPropagator, beam::Union{Beam, AbstractVector{<:AbstractRay}}) = interact3d(d, beam)
interact3d(d::FourierBeamletPropagator, b::Union{GaussianBeamlet, AstigmaticGaussianBeamlet}, id::Int = 1) = interact3d(StaticSystem((d,)), d, b, id)

# ------------------------------------------------------------------------------
# Angular Spectrum Decomposition (generate_fourier_spectrum)
# ------------------------------------------------------------------------------

# Fallback origin position when a ray/beamlet hit has infinite segment length
@inline _fallback_hit_position(hit::AbstractBeamletHit) = hit.p0
@inline _fallback_hit_position(hit::WeightedRayHit) = position(hit.ray)
@inline _fallback_hit_position(hit::AbstractRayHit) = hasfield(typeof(hit), :ray) ? position(hit.ray) : hit.p
@inline _fallback_hit_position(hit::AbstractDetectorHit) = position(hit)

# Extract finite 3D hit coordinates, substituting the ray origin if hit_point is ±Inf/NaN
@inline function _extract_hit_position(hit::AbstractDetectorHit, ::Type{T}) where {T <: Real}
    hp, p0 = hit_point(hit), _fallback_hit_position(hit)
    return SVector{3, T}(T(safe_finite(hp[1], p0[1])), T(safe_finite(hp[2], p0[2])), T(safe_finite(hp[3], p0[3])))
end

# Extract unit propagation direction vector of a detector hit
@inline _extract_hit_direction(hit::AbstractDetectorHit, ::Type{T}) where {T <: Real} =
    normalize(SVector{3, T}(direction(hit)))

# Choose k-space grid spacing dk_step and mode half-count M to cover ±4σ_k (clamped to M <= 32)
function _deduce_fourier_beamlet_sampling(sigma_k::T, L_box::T, dk::Union{Nothing, Real}) where {T <: Real}
    if !isnothing(dk)
        dk_step = T(dk)
    else
        # Nyquist spacing from box size L_box, widened automatically if M would exceed 32
        dk_box = T(2π) / L_box
        M_est = ceil(Int, 4 * sigma_k / dk_box)
        M_clamped = clamp(M_est, 1, 32)
        dk_step = (M_clamped == M_est) ? dk_box : (4 * sigma_k / T(M_clamped))
    end
    return (dk_step, max(1, ceil(Int, 4 * sigma_k / dk_step)))
end

@static if hasfield(GaussianBeamletHit, :segment)
    # BMO v0.14+: evaluate propagation distance along the stored beamlet segment
    @inline function _beamlet_seg_length(hit, r_hit::SVector{3, T}) where {T <: Real}
        l_seg = safe_finite(T(hit.l_seg), zero(T))
        return iszero(l_seg) ? safe_pos(T(norm(r_hit - SVector{3, T}(hit.p0))), l_seg) : l_seg
    end
    @inline _beamlet_gauss_params(hit::GaussianBeamletHit, r_hit::SVector{3, T}) where {T <: Real} =
        gauss_parameters(hit.segment, _beamlet_seg_length(hit, r_hit))
else
    # BMO v0.13: evaluate total distance along the GaussianBeamlet up to the hit point
    @inline function _beamlet_eval_length(hit, rays_vec, r_hit::SVector{3, T}) where {T <: Real}
        l_ray = (1 <= hit.id <= length(rays_vec)) ? T(length(rays_vec[hit.id])) : zero(T)
        l_eval = safe_finite(T(hit.l0), zero(T)) + safe_finite(l_ray, zero(T))
        return iszero(l_eval) ? safe_pos(T(norm(r_hit - SVector{3, T}(position(hit)))), l_eval) : l_eval
    end
    @inline _beamlet_gauss_params(hit::GaussianBeamletHit, r_hit::SVector{3, T}) where {T <: Real} =
        gauss_parameters(hit.gauss, _beamlet_eval_length(hit, hit.gauss.chief.rays, r_hit))
end

@static if hasfield(AstigmaticGaussianBeamletHit, :E0)
    # BMO v0.14+: extract spot size w, curvature radius R, and waist w0 from parabasal (h, u)
    @inline function _astig_mode_params(h, u, ::Type{T}) where {T <: Real}
        w = T(norm(h))
        invR = T(real(dot(u, h))) / w^2
        R = isapprox(invR, zero(T), atol = T(1e-20)) ? T(Inf) : inv(invR)
        # Optical invariant Im(u* ⋅ h) yields waist radius w0 = |Im(u* ⋅ h)| / |u|
        w0 = abs(T(imag(dot(u, h)))) / T(norm(u))
        return (w, R, w0)
    end
    @inline function _beamlet_astig_params(hit::AstigmaticGaussianBeamletHit, r_hit::SVector{3, T}) where {T <: Real}
        l_seg = _beamlet_seg_length(hit, r_hit)
        # Propagate both orthogonal parabasal modes to the hit location
        w1, R1, w01 = _astig_mode_params(hit.h1 + l_seg * hit.u1, hit.u1, T)
        w2, R2, w02 = _astig_mode_params(hit.h2 + l_seg * hit.u2, hit.u2, T)
        return (w1, w2, R1, R2, zero(T), w01, w02)
    end
else
    # BMO v0.13: delegate directly to BeamletOptics.gauss_parameters(agb, l_eval)
    @inline _beamlet_astig_params(hit::AstigmaticGaussianBeamletHit, r_hit::SVector{3, T}) where {T <: Real} =
        gauss_parameters(hit.agb, _beamlet_eval_length(hit, hit.agb.c.rays, r_hit))
end

# Compute longitudinal distance from beam waist (dz1, dz2) along both transverse axes
@inline function _beamlet_defocus_pair(hit::GaussianBeamletHit, k0_mag::T, r_hit::SVector{3, T}, w01::T, ::T) where {T <: Real}
    w, curv, _, _ = _beamlet_gauss_params(hit, r_hit)
    dz = _beamlet_defocus_distance(safe_finite(T(curv), zero(T)), safe_pos(T(w), w01), T(2π) / k0_mag)
    return (dz, dz)
end

@inline function _beamlet_defocus_pair(hit::AstigmaticGaussianBeamletHit, k0_mag::T, r_hit::SVector{3, T}, w01::T, w02::T) where {T <: Real}
    wl = T(2π) / k0_mag
    w1, w2, R1, R2, _, _, _ = _beamlet_astig_params(hit, r_hit)
    return (_beamlet_defocus_distance(safe_inv(T(R1), zero(T)), safe_pos(T(w1), w01), wl),
            _beamlet_defocus_distance(safe_inv(T(R2), zero(T)), safe_pos(T(w2), w02), wl))
end

@inline _beamlet_defocus_pair(::Any, ::T, ::SVector{3, T}, ::T, ::T) where {T <: Real} = (zero(T), zero(T))

# Decompose a Gaussian beamlet hit into a 2D disk of plane-wave k-points on the Ewald sphere
function _sample_beamlet_modes!(
    pts::Vector{FourierKPoint{T}}, hit, dir::SVector{3, T}, w01::T, w02::T, sigma_k::T, dk_step::T,
    M::Int, t1::SVector{3, T}, t2::SVector{3, T}, E0::SVector{3, Complex{T}},
    k0_mag::T, omega::T, r_hit::SVector{3, T}, opl_val::T = zero(T)
) where {T <: Real}
    # Circular spectral cutoff at 4σ_k and Gaussian envelope constants
    cutoff_sq = (4 * sigma_k + T(1e-10))^2
    w01_sq_4, w02_sq_4 = w01^2 / T(4), w02^2 / T(4)
    norm_factor = (w01 * w02 * dk_step^2) / (4 * T(π))
    dz1, dz2 = _beamlet_defocus_pair(hit, k0_mag, r_hit, w01, w02)
    inv_2k0 = inv(T(2) * k0_mag)
    # Iterate over transverse wavevector offsets (dk1, dk2) in the local (t1, t2) plane
    for m1 in -M:M, m2 in -M:M
        dk1, dk2 = m1 * dk_step, m2 * dk_step
        dk_perp_sq = dk1^2 + dk2^2
        dk_perp_sq <= cutoff_sq || continue
        # Project onto the dispersion sphere |k| = k0_mag
        k_vec = sqrt(max(zero(T), k0_mag^2 - dk_perp_sq)) * dir + dk1 * t1 + dk2 * t2
        k_hat = normalize(k_vec)
        # Combine accumulated OPL, spatial shift -k⋅r_hit, and quadratic defocus phase
        phase = cis(k0_mag * opl_val - dot(k_vec, r_hit) - (dz1 * dk1^2 + dz2 * dk2^2) * inv_2k0)
        g = exp(-(w01_sq_4 * dk1^2 + w02_sq_4 * dk2^2))
        # Enforce Gauss's law (k_hat ⋅ E_trans == 0) for each plane-wave mode
        E_trans = E0 - dot(k_hat, E0) * k_hat
        push!(pts, FourierKPoint{T}(k_vec, omega, (norm_factor * g * phase) * E_trans, T(g)))
    end
    return pts
end

# Compute hit position relative to detector center, projecting to POCA for volumetric detectors
@inline function _local_hit_offset(d::FourierBeamletPropagator, dir::SVector{3, T}, r_hit::SVector{3, T}) where {T <: Real}
    r_rel = r_hit - SVector{3, T}(position(d.shape))
    # For 3D volumetric detectors, shift along the ray to the point of closest approach (POCA)
    s_poca = d.is_planar ? zero(T) : dot(-r_rel, dir)
    return (r_rel + s_poca * dir, s_poca)
end

# Extract ray power weight for unpolarized, weighted, or polarized ray hits
@inline _hit_power(hit::PolarizedRayHit, ::Type{T}) where {T <: Real} = T(sum(abs2, polarization(hit)))
@inline _hit_power(hit::WeightedRayHit, ::Type{T}) where {T <: Real} = T(hit.power)
@inline _hit_power(hit::AbstractRayHit, ::Type{T}) where {T <: Real} = hasproperty(hit, :power) ? T(hit.power) : one(T)

# Convert a single geometric or polarized ray hit into one plane-wave FourierKPoint
function _hit_to_kpoints!(
    pts::Vector{FourierKPoint{T}}, hit::AbstractRayHit, L_box::T,
    d::FourierBeamletPropagator, ::Union{Nothing, Real} = nothing
) where {T <: Real}
    # Extract ray direction, wavelength, wavenumber k0, and angular frequency ω
    dir = _extract_hit_direction(hit, T)
    wl = T(_hit_wavelength(hit))
    k0_mag, omega = T(2π) / wl, T(2π * c_vacuum) / wl
    r_hit = _extract_hit_position(hit, T)
    # Determine unit polarization vector (using smooth Tom Duff basis if unpolarized)
    pol_raw = hit isa PolarizedRayHit ? polarization(hit) : transverse_polarization(dir)
    p_vec = SVector{3, Complex{T}}(pol_raw)
    p_norm = norm(p_vec)
    e_hat = p_norm > 0 ? (p_vec / p_norm) : SVector{3, Complex{T}}(transverse_polarization(dir))
    # Anchor phase to detector center (or POCA for volumetric detectors)
    r_loc, s_poca = _local_hit_offset(d, dir, r_hit)
    k0 = k0_mag * dir
    opl_val = safe_finite(T(_hit_opl(hit)), zero(T)) + s_poca
    phase = cis(k0_mag * opl_val - dot(k0, r_loc))
    # Scale electric field amplitude by polarization norm or sqrt(P_ray) * L_box
    amp = hit isa PolarizedRayHit ? p_norm : (sqrt(max(zero(T), _hit_power(hit, T))) * L_box)
    push!(pts, FourierKPoint{T}(k0, omega, (amp * phase) * e_hat, one(T)))
    return pts
end

# Extract principal waist radii (w01, w02) and transverse axes (t1, t2) for stigmatic beamlets
@inline function _beamlet_waist_and_axes(hit::GaussianBeamletHit, dir::SVector{3, T}, r_hit::SVector{3, T}, ::Type{T}) where {T <: Real}
    _, _, _, w0_eval = _beamlet_gauss_params(hit, r_hit)
    @static if hasfield(GaussianBeamletHit, :gauss)
        w0 = isapprox(w0_eval, hit.gauss.w0, rtol = 1e-12) ? T(hit.gauss.w0) : T(w0_eval)
    else
        w0 = T(w0_eval)
    end
    w = safe_pos(w0, T(hit.w_max))
    t1 = SVector{3, T}(transverse_polarization(dir))
    return (w, w, t1, normalize(cross(dir, t1)))
end

# Pick the larger of Re(z) and Im(z) for a complex 3D parabasal ray vector
@inline _dominant_real_part(z::AbstractVector, ::Type{T}) where {T <: Real} =
    (vr = SVector{3, T}(real(z)); vi = SVector{3, T}(imag(z)); norm(vr) >= norm(vi) ? vr : vi)

# Extract principal waist radii (w01, w02) and oriented transverse axes (t1, t2) for astigmatic beamlets
@inline function _beamlet_waist_and_axes(hit::AstigmaticGaussianBeamletHit, dir::SVector{3, T}, r_hit::SVector{3, T}, ::Type{T}) where {T <: Real}
    _, _, _, _, _, w01, w02 = _beamlet_astig_params(hit, r_hit)
    # Align first transverse axis t1 with the dominant component of parabasal mode 1
    v_c = _dominant_real_part(hit.h1, T)
    norm(v_c) <= T(1e-12) && (v_c = _dominant_real_part(hit.u1, T))
    proj = v_c - dot(dir, v_c) * dir
    t1 = norm(proj) > T(1e-12) ? normalize(proj) : SVector{3, T}(transverse_polarization(dir))
    return (safe_pos(T(w01), T(hit.w_max)), safe_pos(T(w02), T(hit.w_max)), t1, normalize(cross(dir, t1)))
end

@inline function _beamlet_waist_and_axes(hit::AbstractBeamletHit, dir::SVector{3, T}, ::SVector{3, T}, ::Type{T}) where {T <: Real}
    w = safe_pos(T(hit.w_max), T(2π / hit.k0))
    t1 = SVector{3, T}(transverse_polarization(dir))
    return (w, w, t1, normalize(cross(dir, t1)))
end

# Expand a Gaussian or astigmatic Gaussian beamlet hit into its angular spectrum k-points
function _hit_to_kpoints!(
    pts::Vector{FourierKPoint{T}}, hit::AbstractBeamletHit, L_box::T,
    d::FourierBeamletPropagator, dk::Union{Nothing, Real} = nothing
) where {T <: Real}
    # Evaluate wave parameters and hit geometry in world coordinates
    wl = T(_hit_wavelength(hit))
    k0_mag, omega = T(2π) / wl, T(2π * c_vacuum) / wl
    dir = _extract_hit_direction(hit, T)
    r_hit = _extract_hit_position(hit, T)

    # Extract electric field phasor at the beam waist and local detector offset
    E0_raw = SVector{3, Complex{T}}(_hit_polarization(hit, r_hit))
    E0 = norm(E0_raw) > 0 ? E0_raw : SVector{3, Complex{T}}(transverse_polarization(dir))
    r_loc, s_poca = _local_hit_offset(d, dir, r_hit)

    # Compute spectral bandwidth σ_k = √2 / min(w01, w02) and sampling step dk_step
    w01, w02, t1, t2 = _beamlet_waist_and_axes(hit, dir, r_hit, T)
    w01, w02 = max(w01, T(1e-12)), max(w02, T(1e-12))
    sigma_k = sqrt(T(2)) / min(w01, w02)
    dk_step, M = _deduce_fourier_beamlet_sampling(sigma_k, L_box, dk)

    # Sample 2D plane-wave mode grid around the chief ray wavevector
    opl_val = safe_finite(T(_hit_opl(hit)), zero(T)) + s_poca
    return _sample_beamlet_modes!(pts, hit, dir, w01, w02, sigma_k, dk_step, M, t1, t2, E0, k0_mag, omega, r_loc, opl_val)
end

# Pass through pre-existing FourierKPoint hits with element-type conversion
function _hit_to_kpoints!(
    pts::Vector{FourierKPoint{T}}, hit::FourierKPoint, ::T,
    ::FourierBeamletPropagator, ::Union{Nothing, Real} = nothing
) where {T <: Real}
    push!(pts, FourierKPoint{T}(SVector{4, T}(hit.k4), SVector{3, Complex{T}}(hit.E), T(hit.weight)))
    return pts
end

_hit_to_kpoints!(pts::Vector{FourierKPoint{T}}, ::AbstractDetectorHit, ::T, ::FourierBeamletPropagator, ::Union{Nothing, Real} = nothing) where {T <: Real} = pts

"""
    generate_fourier_spectrum(d::FourierBeamletPropagator{T}, hits = d.hits;
                              L_box = 1.0, normal = SVector{3, T}(0, 1, 0), dk = nothing) where {T <: Real}

Generates the discrete Fourier angular spectrum `Vector{FourierKPoint{T}}` from captured hits.
"""
function generate_fourier_spectrum(
    d::FourierBeamletPropagator{T},
    hits::AbstractVector = d.hits;
    L_box::Real = 1.0,
    normal::AbstractVector = SVector{3, T}(0, 1, 0),
    dk::Union{Nothing, Real} = nothing
) where {T <: Real}
    pts = FourierKPoint{T}[]
    isempty(hits) && return pts

    # Convert every captured ray or beamlet hit into discrete FourierKPoints
    L_box_T = T(L_box)
    for hit in hits
        _hit_to_kpoints!(pts, hit, L_box_T, d, dk)
    end

    # For planar detectors, apply cosine obliquity factor sqrt(max(0, k_hat ⋅ n_inward))
    if d.is_planar
        o = orientation(d.shape)
        n_loc = SVector{3, T}(-T(o[1, 2]), -T(o[2, 2]), -T(o[3, 2]))
        @inbounds for i in eachindex(pts)
            p = pts[i]
            pts[i] = FourierKPoint{T}(p.k4, sqrt(max(zero(T), dot(direction(p), n_loc))) * p.E, p.weight)
        end
    end
    return pts
end

# ==============================================================================
# Fourier Gridding & NUFFT Field Synthesis (synthesize_field)
# ==============================================================================

"""
    AbstractGrid{T <: Real}

Abstract supertype for spatial detector evaluation grids.
"""
abstract type AbstractGrid{T <: Real} end

"""
    SpatialGrid{T <: Real, D} <: AbstractGrid{T}

Spatial evaluation grid defined over `D` coordinate `LinRange`s (1D along `x`, 2D in `x-z`, or 3D in `x-y-z`).
"""
struct SpatialGrid{T <: Real, D} <: AbstractGrid{T}
    ranges::NTuple{D, LinRange{T, Int}} # Uniform coordinate ranges along each grid axis

    @inline SpatialGrid{T, D}(ranges::NTuple{D, LinRange{T, Int}}) where {T <: Real, D} = new{T, D}(ranges)
end

# Promote arbitrary AbstractRange tuples to uniform LinRange{T, Int}
@inline function SpatialGrid(ranges::NTuple{D, AbstractRange}) where {D}
    T_range = promote_type(map(eltype, ranges)...)
    T = T_range <: Real ? T_range : Float64
    return SpatialGrid{T, D}(ranges)
end

@inline SpatialGrid(ranges::AbstractRange...) = SpatialGrid(ranges)
@inline SpatialGrid{T, D}(ranges::NTuple{D, AbstractRange}) where {T <: Real, D} =
    SpatialGrid{T, D}(map(r -> LinRange{T, Int}(T(first(r)), T(last(r)), length(r)), ranges))
@inline SpatialGrid{T}(ranges::NTuple{D, AbstractRange}) where {T <: Real, D} = SpatialGrid{T, D}(ranges)

# Extract detector-local transverse unit vectors (ex, ez), accounting for planar π z-rotation
@inline function _detector_local_axes(d::FourierBeamletPropagator{S}, ::Type{T} = S) where {S <: Real, T <: Real}
    o = orientation(d.shape)
    s = d.is_planar ? -one(T) : one(T)
    return (SVector{3, T}(s * T(o[1, 1]), s * T(o[2, 1]), s * T(o[3, 1])),
            SVector{3, T}(T(o[1, 3]), T(o[2, 3]), T(o[3, 3])))
end

# Orthonormal spatial projection axes for 1D (X), 2D (detector X-Z), and 3D (world X-Y-Z) grids
@inline _nufft_axes(::SpatialGrid{T, 1}, ::Any) where {T} = (SVector{3, T}(1, 0, 0),)
@inline _nufft_axes(::SpatialGrid{T, 2}, ::Nothing) where {T} = (SVector{3, T}(1, 0, 0), SVector{3, T}(0, 0, 1))
@inline _nufft_axes(::SpatialGrid{T, 2}, d::FourierBeamletPropagator) where {T} = _detector_local_axes(d, T)
@inline _nufft_axes(::SpatialGrid{T, 3}, ::Any) where {T} = (SVector{3, T}(1, 0, 0), SVector{3, T}(0, 1, 0), SVector{3, T}(0, 0, 1))

# Map k-points to normalized NUFFT frequencies in [0, 2π) and apply grid-center phase shift
function _nufft_points_and_phasors(pts::AbstractVector{<:FourierKPoint{T}}, grid::SpatialGrid{T, D}, d::Union{FourierBeamletPropagator, Nothing}) where {T, D}
    # Extract grid step sizes dx and FFTW fftshift center coordinates xc = first + (N ÷ 2) * dx
    dxs = map(r -> T(step(r)), grid.ranges)
    xcs = map(r -> first(r) + (length(r) ÷ 2) * T(step(r)), grid.ranges)
    ax = _nufft_axes(grid, d)
    n_pts = length(pts)
    # Allocate non-uniform frequency coordinate vectors and complex field component arrays
    coords = ntuple(_ -> Vector{T}(undef, n_pts), Val(D))
    v1, v2, v3 = ntuple(_ -> Vector{Complex{T}}(undef, n_pts), Val(3))
    @inbounds for j in 1:n_pts
        p = pts[j]
        phase = zero(T)
        # Project wavevector onto each grid axis and fold into [0, 2π) for Type-1 NUFFT
        for dim in 1:D
            kd = dot(p.k, ax[dim])
            coords[dim][j] = mod(-kd * dxs[dim], T(2π))
            phase += kd * xcs[dim]
        end
        # Multiply electric field phasor by grid-center carrier exp(i k ⋅ r_center)
        cp = cis(phase)
        v1[j], v2[j], v3[j] = p.E[1] * cp, p.E[2] * cp, p.E[3] * cp
    end
    return (coords, (v1, v2, v3))
end

# Execute vectorized 3-component Type-1 NUFFT to synthesize E(r) on the spatial grid
function _synthesize_field_nufft!(
    E_out::AbstractArray{SVector{3, Complex{T}}, D},
    pts::AbstractVector{<:FourierKPoint{T}},
    grid::SpatialGrid{T, D},
    d::Union{FourierBeamletPropagator, Nothing} = nothing
) where {T, D}
    dims = map(length, grid.ranges)
    # NonuniformFFTs spreading kernel requires at least 4 grid points per dimension
    any(<(4), dims) && throw(ArgumentError("SpatialGrid dimensions must be >= 4 for NUFFT field synthesis"))
    if isempty(pts)
        fill!(E_out, zero(SVector{3, Complex{T}}))
        return E_out
    end
    # Prepare normalized frequency coordinates and execute Type-1 NUFFT for (Ex, Ey, Ez) simultaneously
    k_coords, phasors = _nufft_points_and_phasors(pts, grid, d)
    plan = PlanNUFFT(Complex{T}, dims; ntransforms = Val(3), fftshift = true)
    set_points!(plan, k_coords)
    u1, u2, u3 = ntuple(_ -> Array{Complex{T}, D}(undef, dims), Val(3))
    exec_type1!((u1, u2, u3), plan, phasors)
    # Pack transformed Cartesian field components into SVector{3, Complex{T}} array
    @inbounds for i in eachindex(E_out)
        E_out[i] = SVector{3, Complex{T}}(u1[i], u2[i], u3[i])
    end
    return E_out
end

# Retrieve existing k-points or generate the angular spectrum on the fly
function _extract_or_generate_kpoints(d::FourierBeamletPropagator{S}, ::Type{T} = S; L_box::Real = 1.0)::Vector{FourierKPoint{T}} where {S, T <: Real}
    d.hits isa Vector{FourierKPoint{T}} && return d.hits
    pts = generate_fourier_spectrum(d; L_box = L_box)
    return pts isa Vector{FourierKPoint{T}} ? pts : FourierKPoint{T}[FourierKPoint{T}(p.k4, p.E, p.weight) for p in pts]
end

"""
    synthesize_field(d::FourierBeamletPropagator, grid::SpatialGrid{T, D}; L_box::Real = 1.0) where {T <: Real, D}
    synthesize_field(d::FourierBeamletPropagator{T}, ranges::NTuple{D, AbstractRange}; L_box::Real = 1.0) where {T <: Real, D}

Synthesizes the complex 3D electric field vector array `Array{SVector{3, Complex{T}}, D}` (in V/m)
on the spatial detector grid from captured detector hits or Fourier k-points via Type-1 NUFFT.
"""
function synthesize_field(d::FourierBeamletPropagator, grid::SpatialGrid{T, D}; L_box::Real = 1.0)::Array{SVector{3, Complex{T}}, D} where {T <: Real, D}
    # Generate frequency spectrum and evaluate spatial field on `grid` via NUFFT
    pts = _extract_or_generate_kpoints(d, T; L_box = L_box)
    E_out = zeros(SVector{3, Complex{T}}, map(length, grid.ranges))
    return _synthesize_field_nufft!(E_out, pts, grid, d)
end

synthesize_field(d::FourierBeamletPropagator{T}, ranges::NTuple{D, AbstractRange}; L_box::Real = 1.0) where {T <: Real, D} =
    synthesize_field(d, SpatialGrid{T, D}(ranges); L_box = L_box)
