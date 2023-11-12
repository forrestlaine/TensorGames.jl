using Test
using TensorGames
using LinearAlgebra: norm
using Zygote: gradient, forwarddiff
using Random: MersenneTwister

include("test_equilibrium.jl")
include("test_derivatives.jl")
include("test_abstract_number_array_input.jl")
