# TensorGames.jl

[![CI](https://github.com/4estlaine/TensorGames.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/4estlaine/TensorGames.jl/actions/workflows/CI.yml)
[![codecov](https://codecov.io/gh/4estlaine/TensorGames.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/4estlaine/TensorGames.jl)
 [![License](https://img.shields.io/badge/license-MIT-blue)](https://opensource.org/licenses/MIT)

Efficient functionality for computing mixed-strategy Nash equilibrium points of a multi-player, finite action, general-sum game. Uses the PATH solver to compute, via [PATHSolver.jl](https://github.com/chkwon/PATHSolver.jl).

## Usage:
Supply a vector of cost tensors (one for each player) as input to the function ```compute_equilibrium```. 
```cost_tensors[i][j1,j2,...,jN]``` is the cost faced by player i when player 1 plays action j1, player 2 plays action j2, etc.

Additional functionality is provided via ChainRulesCore.jl to automatically differentiate solutions with respect to the elements of the cost tensors. 

## Example: 
```julia

julia> d = [3,3,3,3,3,3]; N = 6; cost_tensors = [ randn(d...) for i = 1:N];
julia> sol = compute_equilibrium(cost_tensors);
julia> sol.x
6-element Vector{Vector{Float64}}:
 [0.6147367189021904, 0.0, 0.3852632810978094]
 [0.0, 0.13423377322536922, 0.8657662267746299]
 [0.30978296032333746, 0.6902170396766623, 0.0]
 [0.0, 0.9999999999999994, 0.0]
 [0.5483759176454717, 0.20182657833950027, 0.24979750401502793]
 [0.4761196190151526, 0.38291994996153766, 0.1409604310233093]
```

Use the function ```expected_cost(sol.x, cost_tensor)``` to compute the equilibrium cost for the player whose objective is represented by cost_tensor.

Equilibrium points can also be found when minimum strategy weights are enforced. In other words, for fixed strategies of players (-i), player i's strategy is optimal among those with minimum weight specified by ϵ:
```julia
julia> d = [3,3,3,3,3,3]; N = 6; cost_tensors = [ randn(d...) for i = 1:N];
julia> sol = compute_equilibrium(cost_tensors; ϵ=0.05);
julia> sol.x
6-element Vector{Vector{Float64}}:
 [0.41301195721648803, 0.17743767597659854, 0.40955036680691337]
 [0.05, 0.05, 0.8999999999999998]
 [0.05, 0.28627171177928123, 0.6637282882207187]
 [0.07255559962614289, 0.05, 0.8774444003738571]
 [0.1925535715622543, 0.7574464284377457, 0.05]
 [0.8560862135625118, 0.05, 0.0939137864374882]
```

See additional examples of usage in the test directory, in which checks for the satisfaction of equilibrium conditions and derivative correctness are performed. 
