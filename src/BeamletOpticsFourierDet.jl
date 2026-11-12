"""
    BeamletOpticsFourierDet

Standalone 3D Fourier-optics detector package for `BeamletOptics.jl` ("BMO").

Provides [`FourierBeamletPropagator`](@ref), a drop-in [`BeamletOptics.AbstractDetector`](@ref)
that captures geometric rays, polarized rays, and Gaussian beamlets in non-sequential 3D
optical setups and synthesizes vectorial electromagnetic fields and Stokes polarimetric
observables via non-uniform fast Fourier transforms (`NonuniformFFTs.jl`).
"""
module BeamletOpticsFourierDet

using LinearAlgebra: norm, normalize, dot, cross
using StaticArrays: SVector
using GeometryBasics: Point2, Point3
using NonuniformFFTs: PlanNUFFT, set_points!, exec_type1!

import BeamletOptics
using BeamletOptics:
    AbstractObject,
    AbstractShape,
    AbstractMesh,
    AbstractSystem,
    StaticSystem,
    AbstractDetector,
    AbstractDetectorHit,
    AbstractRayHit,
    AbstractBeamletHit,
    RayHit,
    PolarizedRayHit,
    GaussianBeamletHit,
    AstigmaticGaussianBeamletHit,
    AbstractRay,
    Ray,
    PolarizedRay,
    Beam,
    GaussianBeamlet,
    AstigmaticGaussianBeamlet,
    BeamInteraction,
    GaussianBeamletInteraction,
    AstigmaticGaussianBeamletInteraction,
    QuadraticFlatMesh,
    RectangularFlatMesh,
    CubeMesh,
    CuboidMesh,
    AbstractCenterAlgorithm,
    Centroid,
    MinMax,
    c_vacuum,
    Z_vacuum,
    orientation,
    translate3d!,
    zrotate3d!,
    intersection,
    normal3d,
    rays,
    gauss_parameters,
    line_plane_distance3d,
    calc_center_point,
    vertices,
    default_properties

import Base: empty!, isempty, length, push!, getproperty, propertynames

import BeamletOptics:
    shape,
    hits,
    hits!,
    stop,
    properties,
    interact3d,
    wavenumber,
    wavelength,
    direction,
    position,
    polarization,
    optical_path_length,
    refractive_index,
    hit_point,
    electric_field,
    intensity,
    optical_power,
    spot_diagram

include("FourierBeamletPropagator.jl")
include("FourierBeamletPropagatorUtils.jl")

export AbstractFourierDetector,
       FourierBeamletPropagator,
       FourierKPoint,
       WeightedRayHit,
       AbstractGrid,
       SpatialGrid,
       DetectorObservables,
       generate_fourier_spectrum,
       synthesize_field,
       calculate_observables,
       calculate_observables!,
       transverse_polarization,
       hits,
       hits!,
       wavelength,
       wavenumber,
       refractive_index,
       electric_field,
       intensity,
       optical_power,
       spot_diagram

end # module BeamletOpticsFourierDet
