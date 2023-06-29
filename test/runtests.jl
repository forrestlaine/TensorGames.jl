using Test
using TensorGames
using LinearAlgebra: norm
using Zygote: gradient, forwarddiff
using Random: MersenneTwister

include("test_rps.jl")
include("test_chance_constr_rps.jl")
#include("test_equilibrium.jl")
#include("test_derivatives.jl")
