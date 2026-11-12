using Documenter
using BeamletOptics
using BeamletOpticsFourierDet

makedocs(
    sitename = "BeamletOpticsFourierDet.jl",
    modules = [BeamletOpticsFourierDet],
    pages = [
        "Home" => "index.md",
        "Examples" => "examples/fourier_detector.md",
    ],
    checkdocs = :exports,
)
