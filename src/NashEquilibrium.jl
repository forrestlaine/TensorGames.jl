module NashEquilibrium

using PATHSolver
using SparseArrays

include("nash.jl")

export compute_equilibrium, compute_derivatives!

end # module
