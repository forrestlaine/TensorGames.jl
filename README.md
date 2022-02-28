# NashEquilibrium.jl

[![CI](https://github.com/4estlaine/NashEquilibrium.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/4estlaine/NashEquilibrium.jl/actions/workflows/CI.yml)
[![codecov](https://codecov.io/gh/4estlaine/NashEquilibrium.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/4estlaine/NashEquilibrium.jl)
 [![License](https://img.shields.io/badge/license-MIT-blue)](https://opensource.org/licenses/MIT)

Functionality for computing mixed-strategy Nash equilibrium points of a multi-player, finite action, general-sum game. Uses the PATH solver to compute, via PATHSolver.jl.

## Usage:
Supply a vector of cost tensors (one for each player) as input to the function ```compute_equilibrium```. 
```cost_tensors[i][j1,j2,...,jN]``` is the cost faced by player i when player 1 plays action j1, player 2 plays action j2, etc.

Additional functionality is provided to compute derivative information of solutions with respect to the elements of the cost tensors. To compute this information,
supply the output of ```compute_equilibrium``` and a derivative tensor ```D``` to the function ```compute_derivatives!```. Here, ```D[n,i...,m] := ∂xₘ / ∂Tⁿᵢ```, where x is the concatenated vector of all player's strategy weights, and i is a cartesian index (i.e. i:=[i₁,i₂,...,iₙ]).

## Example: 
```julia

julia> d = [3,3,3,3,3,3]; N = 6; cost_tensors = [ randn(d...) for i = 1:N]; D = zeros(N,d...,sum(d));
julia> sol = compute_equilibrium(cost_tensors);
julia> compute_derivatives!(D, sol);  
```

 
 


