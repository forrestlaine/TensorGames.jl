module TensorGames

using PATHSolver
using SparseArrays
using ChainRulesCore: ChainRulesCore, NoTangent, ZeroTangent, @non_differentiable

include("nash.jl")

export compute_equilibrium, expected_cost

end # module
