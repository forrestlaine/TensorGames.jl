using Test
using TensorGames
using LinearAlgebra: norm
using Zygote: gradient, forwarddiff
using Random: MersenneTwister
using FiniteDifferences: to_vec

include("test_equilibrium.jl")
include("test_derivatives.jl")
