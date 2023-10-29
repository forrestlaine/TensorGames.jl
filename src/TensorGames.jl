module TensorGames

using PATHSolver
using SparseArrays
using ChainRulesCore: ChainRulesCore, NoTangent, ZeroTangent, @non_differentiable
using LinearAlgebra: qr, diag, I
using ForwardDiff: ForwardDiff

include("nash.jl")

export compute_equilibrium, tensor_product

end # module
