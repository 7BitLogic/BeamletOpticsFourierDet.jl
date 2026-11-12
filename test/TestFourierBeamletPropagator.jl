module TestFourierBeamletPropagator

using BeamletOptics
using BeamletOpticsFourierDet
using Test
using LinearAlgebra
using StaticArrays
using GeometryBasics: Point2, Point3

const BMO = BeamletOptics
const FBD = BeamletOpticsFourierDet
const mm = 1e-3

# Helper returning ConstantRefractiveIndex(n) on BMO v0.14+ and λ -> n on BMO v0.13
_const_n(n) = isdefined(BMO, :ConstantRefractiveIndex) ? BMO.ConstantRefractiveIndex(n) : (λ -> n)

# Recursively check whether a method signature references at least one type owned by `mod`
function _type_mentions_module(@nospecialize(t), mod::Module)
    t isa UnionAll && return _type_mentions_module(t.body, mod) || _type_mentions_module(t.var.ub, mod)
    t isa Union && return _type_mentions_module(t.a, mod) || _type_mentions_module(t.b, mod)
    if t isa DataType
        parentmodule(t) === mod && return true
        return any(p -> p isa Type && _type_mentions_module(p, mod), t.parameters)
    end
    return false
end

@testset "FourierBeamletPropagator" begin

@testset "Seamless BeamletOptics Drop-In Integration & Zero Piracy" begin
    # 1. Verify zero export conflicts when loading `using BeamletOptics, BeamletOpticsFourierDet`
    for sym in names(BeamletOpticsFourierDet)
        @test isdefined(@__MODULE__, sym)
        @test getfield(@__MODULE__, sym) === getfield(BeamletOpticsFourierDet, sym)
        Base.isexported(BeamletOptics, sym) && @test getfield(BeamletOpticsFourierDet, sym) === getfield(BeamletOptics, sym)
    end

    # 2. Verify zero type piracy across all Base and BeamletOptics functions
    pirated = Method[]
    for mod in (Base, BeamletOptics), sym in names(mod; all = true)
        (Base.isdeprecated(mod, sym) || !isdefined(mod, sym)) && continue
        fn = getfield(mod, sym)
        (fn isa Function && parentmodule(fn) !== BeamletOpticsFourierDet) || continue
        for m in methods(fn)
            (m.module === BeamletOpticsFourierDet && !_type_mentions_module(m.sig, BeamletOpticsFourierDet)) && push!(pirated, m)
        end
    end
    @test isempty(pirated)

    # 3. Verify standard BMO detector inspector verbs and empty-detector guards
    λ = 632.8e-9
    det = FourierBeamletPropagator(2mm; is_planar = true)
    @test det isa BMO.AbstractDetector{Float64}
    @test length(det) == 0
    isdefined(BMO, :hit_count) && @test BMO.hit_count(det) == 0
    props = BMO.properties(det)
    @test ("Hits" => 0) in props && ("Planar" => true) in props
    @test_throws ErrorException spot_diagram(det)
    @test_throws ErrorException electric_field(det)

    # 4. Trace a Gaussian beamlet in a BMO System and test BMO readout verbs
    translate3d!(det, [0.0, 20mm, 0.0])
    @test position(det) ≈ [0.0, 20mm, 0.0]

    sys = System([det])
    solve_system!(sys, GaussianBeamlet([0.0, 0.0, 0.0], [0.0, 1.0, 0.0], λ, 100e-6; P0 = 1.5))
    @test length(det) == 1

    # Check RegularGrid dispatch on BMO v0.14+ and keyword-based readout on all versions
    if isdefined(BMO, :RegularGrid)
        xs_rg, zs_rg, E_rg = electric_field(det, RegularGrid(num_points = 32, crop_factor = 2.5))
        @test size(E_rg) == (32, 32)
        @test optical_power(det, RegularGrid(num_points = 64, crop_factor = 3.0)) ≈ 1.5 rtol = 3e-2
    end
    @test length(spot_diagram(det; num_spots = 32)) == 32

    xs, zs, E_mat = electric_field(det; n = 64, crop_factor = 2.5)
    @test E_mat isa Matrix{Point3{ComplexF64}} && size(E_mat) == (64, 64)
    xs_I, zs_I, I_mat = intensity(det; n = 64, crop_factor = 2.5)
    @test (xs_I, zs_I, size(I_mat)) == (xs, zs, (64, 64))
    @test optical_power(det; n = 64, crop_factor = 3.0) ≈ 1.5 rtol = 3e-2

    # 5. Verify ray-bundle spot_diagram and electric_field compatibility
    empty!(det)
    solve_system!(sys, UniformDiscSource([0.0, 0.0, 0.0], [0.0, 1.0, 0.0], 1mm, λ; num_rays = 25))
    @test length(spot_diagram(det)) == length(hits(det))
    @test size(electric_field(det; n = 32)[3]) == (32, 32)
end

@testset "FourierKPoint & Primitives" begin
    # Construct 4-wavevector k-point and verify virtual property accessors (.k and .omega)
    k4 = SVector{4, Float64}(1e6, 0.0, 1e7, 3e15)
    E_vec = SVector{3, ComplexF64}(1.0 + 0.0im, 0.0im, -0.1 + 0.0im)
    pt = FourierKPoint(k4, E_vec, 2.5)

    @test pt isa FourierKPoint{Float64} && pt isa BMO.AbstractDetectorHit
    @test propertynames(pt) == (:k4, :E, :weight, :k, :omega)
    @test (pt.k, pt.omega, pt.E, pt.weight) == (SVector{3, Float64}(1e6, 0.0, 1e7), 3e15, E_vec, 2.5)
    @test_throws (@isdefined(FieldError) ? FieldError : ErrorException) pt.nonexistent_field

    # Test convenience constructors with 3D k + omega and Float32 conversion
    @test FourierKPoint(SVector(1.0, 2.0, 3.0), 4.0, E_vec, 1.5).k4 == SVector(1.0, 2.0, 3.0, 4.0)
    @test FourierKPoint([1.0, 2.0, 3.0, 4.0], [1.0 + 0im, 0im, 0im], 0.5).weight == 0.5
    @test FourierKPoint{Float32}(SVector(1.0, 2.0, 3.0, 4.0), E_vec, 1.0) isa FourierKPoint{Float32}
    @test FourierKPoint{Float32}([1.0, 2.0, 3.0], 4.0, [1.0 + 0im, 0im, 0im], 1.0) isa FourierKPoint{Float32}

    # Test BMO hit interface on non-zero and zero-wavevector k-points
    @test wavenumber(pt) ≈ hypot(1e6, 0.0, 1e7)
    @test wavelength(pt) ≈ 2π / wavenumber(pt)
    @test direction(pt) ≈ normalize(SVector(1e6, 0.0, 1e7))
    @test position(pt) == SVector{3, Float64}(0.0, 0.0, 0.0)
    @test BMO.polarization(pt) == E_vec

    pt_zero = FourierKPoint(SVector(0.0, 0.0, 0.0, 0.0), E_vec, 1.0)
    @test (direction(pt_zero), wavelength(pt_zero)) == (SVector{3, Float64}(0.0, 1.0, 0.0), 1.0)

    # Test numerical safety helpers (safe_finite, safe_pos, safe_inv)
    @test (FBD.safe_finite(3.0, 1.0), FBD.safe_finite(Inf, 1.0), FBD.safe_finite(NaN, 2.0f0)) === (3.0, 1.0, 2.0)
    @test (FBD.safe_pos(5.0, 1.0), FBD.safe_pos(0.0, 1.0), FBD.safe_pos(-2.0, 1.0), FBD.safe_pos(Inf, 1.0)) === (5.0, 1.0, 1.0, 1.0)
    @test (FBD.safe_inv(4.0), FBD.safe_inv(-2.0), FBD.safe_inv(0.0, 9.0), FBD.safe_inv(Inf, 9.0)) === (0.25, -0.5, 9.0, 9.0)

    # Verify Tom Duff transverse polarization orthogonality and smoothness across |v_x| = 0.9
    for d_vec in (SVector(0.0, 1.0, 0.0), SVector(0.0, 0.0, -1.0))
        p_t = transverse_polarization(d_vec)
        @test norm(p_t) ≈ 1.0 && isapprox(dot(p_t, d_vec), 0.0; atol = 1e-15)
    end
    d_below, d_above = normalize(SVector(0.89999, 0.4359, 0.01)), normalize(SVector(0.90001, 0.4359, 0.01))
    @test norm(transverse_polarization(d_below) - transverse_polarization(d_above)) < 1e-4
end

@testset "Propagator Construction, Container API & Passive Observation" begin
    # Test 1D, 2D, 3D, and object-cloned constructors
    d_vol = FourierBeamletPropagator(10mm)
    d_plan = FourierBeamletPropagator(10mm; is_planar = true, stop = true)
    d_2arg = FourierBeamletPropagator(10mm, 20mm; stop = false)
    d_3arg = FourierBeamletPropagator(10mm, 20mm, 30mm; stop = false)
    d_from_obj = FourierBeamletPropagator(d_plan; stop = false, is_planar = true)

    @test (!d_vol.is_planar, BMO.stop(d_vol), isempty(d_vol), length(d_vol)) == (true, true, true, 0)
    @test (d_plan.is_planar, BMO.stop(d_plan), BMO.shape(d_plan) isa BMO.Mesh{Float64}) == (true, true, true)
    @test (d_2arg.is_planar, !BMO.stop(d_2arg)) == (true, true)
    @test (!d_3arg.is_planar, !BMO.stop(d_3arg)) == (true, true)
    @test (d_from_obj.is_planar, !BMO.stop(d_from_obj)) == (true, true)

    # Verify SpatialGrid construction and NUFFT minimum grid size guard (N >= 4)
    @test SpatialGrid(range(-1mm, 1mm, length = 8), range(-1mm, 1mm, length = 8)) isa SpatialGrid{Float64, 2}
    @test SpatialGrid(range(-1mm, 1mm, length = 4), range(-1mm, 1mm, length = 4), range(-1mm, 1mm, length = 4)) isa SpatialGrid{Float64, 3}
    @test_throws ArgumentError synthesize_field(d_vol, SpatialGrid((range(-1.0, 1.0, length = 3),)))
    @test_throws ArgumentError synthesize_field(d_vol, SpatialGrid(range(-1.0, 1.0, length = 8), range(-1.0, 1.0, length = 2)))
    @test_throws ArgumentError synthesize_field(d_vol, SpatialGrid(range(-1.0, 1.0, length = 4), range(-1.0, 1.0, length = 4), range(-1.0, 1.0, length = 3)))

    # Verify type narrowing on first hit and automatic container widening on heterogeneous hits
    r1 = Ray([0.0, 0.0, 0.0], [0.0, 1.0, 0.0], 632.8e-9)
    pr1 = PolarizedRay([0.0, 0.0, 0.0], [0.0, 1.0, 0.0], 632.8e-9, [1.0 + 0im, 0im, 0im])
    gb = GaussianBeamlet([0.0, 0.0, 0.0], [0.0, 1.0, 0.0], 632.8e-9, 1mm)
    agb = AstigmaticGaussianBeamlet([0.0, 0.0, 0.0], [0.0, 1.0, 0.0], 632.8e-9, 1mm, 0.5mm)

    BMO.interact3d(d_vol, r1)
    @test length(d_vol) == 1 && hits(d_vol) isa Vector{BMO.RayHit{Float64}}
    BMO.interact3d(d_vol, pr1)
    @test length(d_vol) == 2 && hits(d_vol) isa Vector{BMO.AbstractRayHit{Float64}}
    BMO.interact3d(d_vol, gb)
    @test length(d_vol) == 3 && hits(d_vol) isa Vector{BMO.AbstractDetectorHit}
    empty!(d_vol)
    @test isempty(d_vol) && isempty(hits(d_vol))

    # Verify passive observer behavior (stop = false passes beams through; stop = true returns nothing)
    d_pass = FourierBeamletPropagator(10mm; stop = false)
    d_absorb = FourierBeamletPropagator(10mm; stop = true)
    beam = Beam(r1)

    @test BMO.interact3d(d_pass, r1).ray === r1 && isnothing(BMO.intersection(r1))
    @test BMO.interact3d(d_pass, pr1).ray === pr1
    @test BMO.interact3d(d_pass, beam) === beam && BMO.interact3d(d_pass, [r1]) == [r1]
    @test BMO.interact3d(d_pass, gb) isa BMO.GaussianBeamletInteraction
    @test BMO.interact3d(d_pass, agb) isa BMO.AstigmaticGaussianBeamletInteraction

    for src_item in (r1, pr1, beam, [r1], gb, agb)
        @test isnothing(BMO.interact3d(d_absorb, src_item))
    end
end

@testset "Angular Spectrum Generation & Physics" begin
    λ = 1.0e-6
    k0 = 2π / λ

    # 1. Verify PolarizedRay k-point transversality and L_box-invariance
    d = FourierBeamletPropagator(2mm; is_planar = false)
    E0 = SVector{3, ComplexF64}(2.0 + 0im, 0im, 1.0im)
    BMO.interact3d(d, PolarizedRay([0.1mm, -5mm, -0.2mm], [0.0, 1.0, 0.0], λ, E0))

    pts_a = generate_fourier_spectrum(d; L_box = 2mm)
    pts_b = generate_fourier_spectrum(d; L_box = 8mm)
    @test length(pts_a) == 1 && pts_a[1].k ≈ SVector(0.0, k0, 0.0) && pts_a[1].weight == 1.0
    @test pts_a[1].E ≈ pts_b[1].E
    @test isapprox(dot(normalize(pts_a[1].k), pts_a[1].E), 0.0 + 0.0im; atol = 1e-14)

    # Verify WeightedRayHit scales amplitude by sqrt(power) * L_box
    empty!(d)
    P_ray = 3.5
    wh = WeightedRayHit(Ray([0.0, 0.0, 0.0], [0.0, 1.0, 0.0], λ), 0.0, P_ray)
    @test wh isa BMO.AbstractRayHit{Float64} && wavelength(wh) == λ
    push!(d, wh)
    pt_ray = generate_fourier_spectrum(d; L_box = 4mm)[1]
    @test pt_ray.weight == 1.0 && sum(abs2, pt_ray.E) ≈ P_ray * (4mm)^2

    # 2. Verify planar detector cosine obliquity scaling sqrt(cos θ) and backward-ray rejection
    d_plan = FourierBeamletPropagator(10mm; is_planar = true)
    θ = deg2rad(60.0)
    BMO.interact3d(d_plan, PolarizedRay([0.0, 0.0, 0.0], [0.0, 1.0, 0.0], λ, [1.0 + 0im, 0im, 0im]))
    E_norm = norm(generate_fourier_spectrum(d_plan)[1].E)

    empty!(d_plan)
    BMO.interact3d(d_plan, PolarizedRay([0.0, 0.0, 0.0], [sin(θ), cos(θ), 0.0], λ, [cos(θ) + 0im, -sin(θ) + 0im, 0im]))
    @test norm(generate_fourier_spectrum(d_plan)[1].E) / E_norm ≈ sqrt(cos(θ)) rtol = 1e-12

    empty!(d_plan)
    BMO.interact3d(d_plan, PolarizedRay([0.0, 0.0, 0.0], [0.0, -1.0, 0.0], λ, [1.0 + 0im, 0im, 0im]))
    @test norm(generate_fourier_spectrum(d_plan)[1].E) == 0.0

    # 3. Verify GaussianBeamlet mode transversality and rotated AstigmaticGaussianBeamlet waist axes
    empty!(d)
    BMO.interact3d(d, GaussianBeamlet([0.0, 0.0, 0.0], [0.0, 1.0, 0.0], λ, 50e-6))
    pts_gb = generate_fourier_spectrum(d; L_box = 1mm)
    @test length(pts_gb) > 1
    @test all(p -> isapprox(dot(normalize(p.k), p.E), 0.0 + 0.0im; atol = 1e-14), pts_gb)

    wx, wz = 40e-6, 100e-6
    agb = AstigmaticGaussianBeamlet([0.0, 0.0, 0.0], [0.0, 1.0, 0.0], λ, wx, wz)
    yrotate3d!(agb, deg2rad(35.0))
    empty!(d)
    BMO.interact3d(d, agb)
    w1, w2, e1, e2 = FBD._beamlet_waist_and_axes(hits(d)[1], SVector(0.0, 1.0, 0.0), SVector(0.0, 0.0, 0.0), Float64)
    @test collect(minmax(w1, w2)) ≈ [wx, wz] rtol = 1e-12
    @test isapprox(abs(dot(e1, e2)), 0.0; atol = 1e-12)

    # 4. Verify focused GaussianBeamlet after a doublet lens uses the post-lens focused waist
    lens = SphericalDoubletLens(40mm, -40mm, -150mm, 6mm, 2.5mm, 25mm, _const_n(1.515), _const_n(1.665))
    gb_in = GaussianBeamlet([0.0, -20mm, 0.0], [0.0, 1.0, 0.0], 500e-9, 2.5mm; P0 = 1.0)
    sys_lens = System([lens])
    solve_system!(sys_lens, gb_in)
    d_foc = FourierBeamletPropagator(2mm; is_planar = true)
    translate3d!(d_foc, [0.0, 75.5mm, 0.0])
    BMO.interact3d(sys_lens, d_foc, gb_in, length(rays(gb_in.chief)))
    w1_foc, _, _, _ = FBD._beamlet_waist_and_axes(hits(d_foc)[1], SVector(0.0, 1.0, 0.0), SVector(0.0, 75.5mm, 0.0), Float64)
    @test w1_foc < 50e-6
end

@testset "NUFFT Field Synthesis (1D, 2D, 3D & Rotated Frames)" begin
    # Compare Type-1 NUFFT synthesis in 1D, 2D, and 3D against exact analytical plane-wave sum
    d = FourierBeamletPropagator(2mm)
    kpts = [
        FourierKPoint(SVector(1.2e3, 6.28e6, -0.8e3, 1.88e15), SVector{3, ComplexF64}(1.5 + 0.2im, 0.0im, -0.3 + 0.1im), 1.0),
        FourierKPoint(SVector(-0.5e3, 6.28e6, 1.1e3, 1.88e15), SVector{3, ComplexF64}(0.4 - 0.5im, 0.0im, 0.8 + 0.0im), 1.0),
    ]
    foreach(pt -> push!(d, pt), kpts)
    exact_field(r) = sum(pt.E * cis(dot(pt.k, r)) for pt in kpts)

    xs_1d = range(-0.4mm, 0.4mm, length = 19)
    @test synthesize_field(d, SpatialGrid((xs_1d,))) ≈ [exact_field(SVector(x, 0.0, 0.0)) for x in xs_1d] rtol = 1e-6

    xs_2d, zs_2d = range(-0.4mm, 0.4mm, length = 17), range(-0.3mm, 0.3mm, length = 23)
    @test synthesize_field(d, SpatialGrid(xs_2d, zs_2d)) ≈ [exact_field(SVector(x, 0.0, z)) for x in xs_2d, z in zs_2d] rtol = 1e-6

    xs_3d, ys_3d, zs_3d = range(-0.2mm, 0.2mm, length = 8), range(-0.1mm, 0.1mm, length = 6), range(-0.2mm, 0.2mm, length = 8)
    @test synthesize_field(d, SpatialGrid(xs_3d, ys_3d, zs_3d)) ≈ [exact_field(SVector(x, y, z)) for x in xs_3d, y in ys_3d, z in zs_3d] rtol = 1e-6

    # Empty detector synthesizes an identically zero field
    @test all(iszero, synthesize_field(FourierBeamletPropagator(2mm), (xs_2d, zs_2d)))

    # Verify co-rotating both beam and planar detector by 30° around Z leaves local observables invariant
    λ, w0, θ_rot = 1.0e-6, 80e-6, deg2rad(30.0)
    grid_rot = SpatialGrid(range(-250e-6, 250e-6, length = 64), range(-250e-6, 250e-6, length = 64))

    d_ref = FourierBeamletPropagator(1mm; is_planar = true)
    BMO.interact3d(d_ref, GaussianBeamlet([0.0, 0.0, 0.0], [0.0, 1.0, 0.0], λ, w0; P0 = 1.0))
    obs_ref = calculate_observables(d_ref, grid_rot)

    d_tilted = FourierBeamletPropagator(1mm; is_planar = true)
    zrotate3d!(d_tilted, θ_rot)
    BMO.interact3d(d_tilted, GaussianBeamlet([0.0, 0.0, 0.0], [-sin(θ_rot), cos(θ_rot), 0.0], λ, w0; P0 = 1.0))
    obs_tilted = calculate_observables(d_tilted, grid_rot)

    @test obs_tilted.power ≈ obs_ref.power rtol = 1e-3
    @test obs_tilted.intensity ≈ obs_ref.intensity rtol = 1e-2
end

@testset "DetectorObservables, Polarimetry & Energy Conservation" begin
    # 1. Verify Stokes vectors (S0, S1, S2, S3) for canonical polarization states
    E0 = sqrt(2 * Float64(BMO.Z_vacuum))
    make_const_field(v) = fill(SVector{3, ComplexF64}(v * E0), 4, 4)
    s2 = inv(sqrt(2.0))

    stokes_cases = (
        (SVector(1.0 + 0im, 0im, 0im),      (1.0,  1.0, 0.0, 0.0)), # Horizontal (+X)
        (SVector(0im, 0im, 1.0 + 0im),      (1.0, -1.0, 0.0, 0.0)), # Vertical (+Z)
        (SVector(s2 + 0im, 0im, s2 + 0im),  (1.0,  0.0, 1.0, 0.0)), # Diagonal (+45°)
        (SVector(s2 + 0im, 0im, s2 * im),   (1.0,  0.0, 0.0, 1.0)), # Circular
        (SVector(0im, 1.0 + 0im, 0im),      (0.0,  0.0, 0.0, 0.0)), # Longitudinal (+Y)
    )
    for (pol_vec, expected_stokes) in stokes_cases
        obs = calculate_observables(make_const_field(pol_vec))
        @test all(x -> isapprox(x, 1.0; atol = 1e-12), obs.intensity)
        for k in 1:4
            @test all(x -> isapprox(x, expected_stokes[k]; atol = 1e-12), obs.stokes[k])
        end
        @test all(i -> obs.stokes[1][i]^2 >= obs.stokes[2][i]^2 + obs.stokes[3][i]^2 + obs.stokes[4][i]^2, eachindex(obs.intensity))
    end

    # 2. Verify Parseval optical power conservation in 2D and 3D grids
    λ, w0, P0 = 1.0e-6, 60e-6, 2.5
    d = FourierBeamletPropagator(1mm)
    BMO.interact3d(d, GaussianBeamlet([0.0, 0.0, 0.0], [0.0, 1.0, 0.0], λ, w0; P0 = P0))

    rs = range(-250e-6, 250e-6, length = 64)
    obs_2d = calculate_observables(d, (rs, rs))
    @test obs_2d.power ≈ P0 rtol = 2e-2

    obs_3d_5 = calculate_observables(d, SpatialGrid(rs, range(-50e-6, 50e-6, length = 5), rs))
    obs_3d_11 = calculate_observables(d, SpatialGrid(rs, range(-50e-6, 50e-6, length = 11), rs))
    @test obs_3d_5.power ≈ P0 rtol = 2e-2
    @test obs_3d_5.power ≈ obs_3d_11.power rtol = 1e-3

    # Verify peak intensity I0 = 2 P0 / (π w0²) for a macroscopic beam (w0 >> detector window)
    d_macro = FourierBeamletPropagator(1mm; is_planar = true)
    BMO.interact3d(d_macro, GaussianBeamlet([0.0, 0.0, 0.0], [0.0, 1.0, 0.0], λ, 2.5mm; P0 = 1.0))
    r_win = range(-0.5mm, 0.5mm, length = 64)
    @test maximum(calculate_observables(d_macro, (r_win, r_win)).intensity) ≈ 2.0 / (π * (2.5mm)^2) rtol = 2e-2

    # 3. Verify zero-allocation in-place calculate_observables! and DimensionMismatch check
    field_2d = synthesize_field(d, SpatialGrid(rs, rs))
    obs_pre = DetectorObservables(size(field_2d), Float64)
    calculate_observables!(obs_pre, d, field_2d; pixel_area = float(step(rs))^2)
    @test (@allocated calculate_observables!(obs_pre, d, field_2d; pixel_area = float(step(rs))^2)) == 0
    @test obs_pre.power ≈ obs_2d.power
    @test calculate_observables(field_2d, SpatialGrid(rs, rs)).power ≈ obs_2d.power
    @test calculate_observables!(obs_pre, field_2d, SpatialGrid(rs, rs)).power ≈ obs_2d.power
    @test calculate_observables!(obs_pre, d, (rs, rs)).power ≈ obs_2d.power
    @test_throws DimensionMismatch calculate_observables!(DetectorObservables((8, 8), Float64), d, field_2d)
end

@testset "Optical Breadboard Setups (Lens Focus, Cylindrical Focus & Interference)" begin
    # 1. Spherical lens focal spot: compare measured first Airy dark ring against 1.22 λ f / D
    R, n, D_beam, λ, f, y_foc = 103.36mm, 1.5168, 10mm, 632.8e-9, 200mm, 200.13mm
    src = UniformDiscSource([0.0, -10mm, 0.0], [0.0, 1.0, 0.0], D_beam, λ; num_rays = 1000)
    lens = SphericalLens(R, Inf, 1mm, 25.4mm, _const_n(n))
    det = FourierBeamletPropagator(100e-6; stop = true, is_planar = true)
    translate3d!(det, [0.0, y_foc, 0.0])
    solve_system!(System([lens, det]), src)

    xs = range(-50e-6, 50e-6, length = 129)
    obs = calculate_observables(det, (xs, xs))
    mid = 65
    @test Tuple(argmax(obs.intensity)) == (mid, mid)
    min_idx = argmin(obs.intensity[mid:(mid + 29), mid])
    @test (min_idx - 1) * step(xs) ≈ 1.22 * λ * f / D_beam rtol = 0.12

    # 2. Cylindrical lens 1D line focus (curved along Z, flat along X)
    r_cyl, ct_cyl = 25.0mm, 5.0mm
    cyl_lens = Lens(CylindricalSurface(r_cyl, 25.0mm, 25.0mm), ct_cyl, _const_n(n))
    det_cyl = FourierBeamletPropagator(20.0mm; stop = true, is_planar = true)
    translate3d!(det_cyl, [0.0, r_cyl / (n - 1.0) + ct_cyl, 0.0])
    solve_system!(System([cyl_lens, det_cyl]), UniformDiscSource([0.0, -10mm, 0.0], [0.0, 1.0, 0.0], 10mm, λ; num_rays = 1000))

    obs_cyl = calculate_observables(det_cyl, (range(-5.0mm, 5.0mm, length = 65), range(-2.0mm, 2.0mm, length = 65)))
    @test obs_cyl.intensity[33, 33] > 4 * obs_cyl.intensity[33, 1] && obs_cyl.power > 0.0

    # 3. Two-beam interference fringes from opposing tilt ±Δθ/2 in X-Y plane
    Δθ, w0_int = 10e-3, 1.0mm
    d_int = FourierBeamletPropagator(2mm; is_planar = true)
    BMO.interact3d(d_int, GaussianBeamlet([0.0, 0.0, 0.0], [sin(Δθ / 2), cos(Δθ / 2), 0.0], λ, w0_int; P0 = 1.0))
    BMO.interact3d(d_int, GaussianBeamlet([0.0, 0.0, 0.0], [-sin(Δθ / 2), cos(Δθ / 2), 0.0], λ, w0_int; P0 = 1.0))

    obs_fringe = calculate_observables(d_int, (range(-200e-6, 200e-6, length = 256), range(-50e-6, 50e-6, length = 8)))
    I_line = obs_fringe.intensity[:, 4]
    @test (maximum(I_line) - minimum(I_line)) / (maximum(I_line) + minimum(I_line)) > 0.95
end

end # @testset "FourierBeamletPropagator"

end # module TestFourierBeamletPropagator
