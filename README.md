# NashEquilibrium.jl

[![CI](https://github.com/4estlaine/NashEquilibrium.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/4estlaine/NashEquilibrium.jl/actions/workflows/CI.yml)
[![codecov](https://codecov.io/gh/4estlaine/NashEquilibrium.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/4estlaine/NashEquilibrium.jl)
 [![License](https://img.shields.io/badge/license-MIT-blue)](https://opensource.org/licenses/MIT)

Efficient functionality for computing mixed-strategy Nash equilibrium points of a multi-player, finite action, general-sum game. Uses the PATH solver to compute, via PATHSolver.jl.

## Usage:
Supply a vector of cost tensors (one for each player) as input to the function ```compute_equilibrium```. 
```cost_tensors[i][j1,j2,...,jN]``` is the cost faced by player i when player 1 plays action j1, player 2 plays action j2, etc.

Additional functionality is provided to compute derivative information of solutions with respect to the elements of the cost tensors. To compute this information,
supply the output of ```compute_equilibrium``` and a derivative tensor ```D``` to the function ```compute_derivatives!```. Here, ```D[n,i...,m] := ∂xₘ / ∂Tⁿᵢ```, where x is the concatenated vector of all player's strategy weights, and i is a cartesian index (i.e. i:=[i₁,i₂,...,iₙ]).  In other words, D[n,i...,m] is the partial derivative of xₘ with respect to the i-th element of player n's cost tensor.

## Example: 
```julia

julia> d = [3,3,3,3,3,3]; N = 6; cost_tensors = [ randn(d...) for i = 1:N]; D = zeros(N,d...,sum(d));
julia> sol = compute_equilibrium(cost_tensors);
julia> sol.x
6-element Vector{Vector{Float64}}:
 [0.6147367189021904, 0.0, 0.3852632810978094]
 [0.0, 0.13423377322536922, 0.8657662267746299]
 [0.30978296032333746, 0.6902170396766623, 0.0]
 [0.0, 0.9999999999999994, 0.0]
 [0.5483759176454717, 0.20182657833950027, 0.24979750401502793]
 [0.4761196190151526, 0.38291994996153766, 0.1409604310233093]
julia> compute_derivatives!(D, sol);  
julia> D[1, 3, 2, 1, 2, 3, 3, 1]
0.0011453641054479879

```

 See additional examples of usage in the test directory, in which checks for the satisfaction of equilibrium conditions and derivative correctness are performed. 
 


