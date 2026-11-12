# BeamletOpticsFourierDet.jl

Standalone 3D Fourier-optics detector and NUFFT field synthesis package for [**BeamletOptics.jl**](https://github.com/JuliaPhysics/BeamletOptics.jl) ("BMO").

`BeamletOpticsFourierDet.jl` provides `FourierBeamletPropagator`, a drop-in subtype of `BeamletOptics.AbstractDetector` that works seamlessly alongside unmodified upstream `BeamletOptics.jl`—with full support for BMO kinematics (`translate3d!`, `rotate3d!`, `align3d!`), `System` / `StaticSystem` tracing, passive observation (`stop = false`), Makie 3D rendering (`render!`), standard BMO readout verbs (`electric_field`, `intensity`, `optical_power`, `spot_diagram`), and full vectorial Stokes polarimetry (`calculate_observables`, `calculate_observables!`, `synthesize_field`).

## Installation

```julia
using Pkg
Pkg.add("BeamletOptics")
Pkg.develop(path = "/path/to/BeamletOpticsFourierDet.jl") # or Pkg.add(url = "https://github.com/7BitLogic/BeamletOpticsFourierDet.jl")
```

## Quick Start

Load `BeamletOptics` and `BeamletOpticsFourierDet` together and place `FourierBeamletPropagator` into any `System` just like a built-in BMO `Detector`:

```julia
using BeamletOptics
using BeamletOpticsFourierDet

# 1. Optical setup: N-BK7 singlet focusing a 10 mm HeNe disc source
λ = 632.8e-9
src = UniformDiscSource([0.0, -10e-3, 0.0], [0.0, 1.0, 0.0], 10e-3, λ; num_rays = 2000)
lens = SphericalLens(103.36e-3, Inf, 1e-3, 25.4e-3, ConstantRefractiveIndex(1.5168))

# 2. Place a planar Fourier detector at the focal plane
det = FourierBeamletPropagator(100e-6; stop = true, is_planar = true)
translate3d!(det, [0.0, 200.13e-3, 0.0])

# 3. Trace rays through unmodified BeamletOptics
sys = System([lens, det])
solve_system!(sys, src)

# 4a. Rich vectorial Fourier & Stokes observables (intensity, phase, S0..S3, power)
xs = range(-50e-6, 50e-6, length = 129)
obs = calculate_observables(det, (xs, xs))

# 4b. Or use standard BeamletOptics detector verbs directly
xs_bmo, zs_bmo, I_bmo = intensity(det; n = 129, x_min = -50e-6, x_max = 50e-6, z_min = -50e-6, z_max = 50e-6)
spots = spot_diagram(det)
```

## Exported API

| Symbol | Description |
| :--- | :--- |
| `AbstractFourierDetector` | Abstract supertype (`<: BeamletOptics.AbstractDetector{T}`) for Fourier detectors |
| `FourierBeamletPropagator` | 1D/2D planar or 3D volumetric detector & angular-spectrum propagator |
| `FourierKPoint` | Discrete frequency-space plane-wave mode (`k4`, `E`, `weight`, `.k`, `.omega`) |
| `WeightedRayHit` | Geometric ray hit carrying explicit per-ray optical power (`<: AbstractRayHit{T}`) |
| `AbstractGrid`, `SpatialGrid` | 1D (`x`), 2D (`x-z`), or 3D (`x-y-z`) regular evaluation grid |
| `DetectorObservables` | Container for `intensity`, `phase`, `stokes = (S0, S1, S2, S3)`, and `power` |
| `generate_fourier_spectrum` | Decomposes captured ray/beamlet hits into `Vector{FourierKPoint{T}}` |
| `synthesize_field` | Evaluates 3D complex electric field `Array{SVector{3, Complex{T}}, D}` via Type-1 NUFFT |
| `calculate_observables` | Computes `DetectorObservables` from a detector, field array, or `SpatialGrid` |
| `calculate_observables!` | Zero-allocation in-place observables evaluation into a preallocated `DetectorObservables` |
| `transverse_polarization` | Smooth Tom Duff transverse polarization basis vector across the forward hemisphere |
| `electric_field`, `intensity`, `optical_power`, `spot_diagram` | Standard `BeamletOptics` detector readout verbs extended onto `FourierBeamletPropagator` |

## Running Tests

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```
