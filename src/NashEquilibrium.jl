module NashEquilibrium

using PATHSolver
using SparseArrays

include("nash.jl")

export compute_equilibrium, compute_derivatives!, expected_cost

end # module
