# NashEquilibrium.jl
Functionality for computing mixed-strategy Nash equilibrium points of a multi-player, finite action, general-sum game. Uses the PATH solver to compute, via PATHSolver.jl.

## Usage:
Supply a vector of cost tensors (one for each player) as input to the function ```compute_equilibrium```. 
```cost_tensors[i][j1,j2,...,jN]``` is the cost faced by player i when player 1 plays action j1, player 2 plays action j2, etc.


## Example:

```julia

julia> d = [3,3,3,3,3,3]; N = 6; cost_tensors = [ randn(d...) for i = 1:N];
julia> compute_equilibrium(cost_tensors; silent=false)
Path 5.0.03 (Fri Jun 26 09:58:07 2020)
Written by Todd Munson, Steven Dirkse, Youngdae Kim, and Michael Ferris

Crash Log
major  func  diff  size  residual    step       prox   (label)
    0     0             2.4185e-01             0.0e+00 (f[    3])
pn_search terminated: no progress.

Major Iteration Log
major minor  func  grad  residual    step  type prox    inorm  (label)
    0     0    13     1 2.4185e-01           I 0.0e+00 1.1e-01 (f[    3])
    1     4    14     2 2.9281e-01  1.0e+00 SO 0.0e+00 1.8e-01 (f[   18])
    2     3    15     3 7.3240e-02  1.0e+00 SO 0.0e+00 3.6e-02 (f[    5])
    3     1    16     4 1.7875e-01  1.0e+00 SO 0.0e+00 8.3e-02 (f[   14])
    4     1    17     5 4.5813e-02  1.0e+00 SO 0.0e+00 1.8e-02 (f[   11])
    5     1    18     6 8.3418e-03  1.0e+00 SO 0.0e+00 4.9e-03 (f[    4])
    6     1    19     7 3.8377e-05  1.0e+00 SO 0.0e+00 2.0e-05 (f[   14])
    7     1    20     8 1.3717e-08  1.0e+00 SO 0.0e+00 5.8e-09 (f[    4])

Major Iterations. . . . 7
Minor Iterations. . . . 12
Restarts. . . . . . . . 0
Crash Iterations. . . . 0
Gradient Steps. . . . . 0
Function Evaluations. . 20
Gradient Evaluations. . 8
Basis Time. . . . . . . 0.000277
Total Time. . . . . . . 0.026862
Residual. . . . . . . . 1.371658e-08
6-element Vector{Vector{Float64}}:
 [0.0, 0.4997507402515058, 0.5002492597484942]
 [0.2911117268969821, 0.3089404902069817, 0.3999477828960362]
 [0.30284232558594837, 0.0, 0.6971576744140516]
 [0.0, 0.5027255640655567, 0.4972744359344433]
 [0.45117574225907664, 0.2990364456997047, 0.24978781204121855]
 [0.4895995548672493, 0.3763290202188116, 0.13407142491393903]
```

 
 
 


