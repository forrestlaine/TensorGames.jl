# N-Player Nash Equilibrium
A function for computing mixed-strategy Nash equilibrium points of a multi-player, finite action, general-sum game. Uses the PATH solver to compute, via PATHSolver.jl.

## Example:

```julia

N = 4                                        # number of players
d = rand(2:3, N)                             # number of actions for each player (e.g. d[n] is number of actions for player n)
cost_tensors = [randn(d...) for n in 1:N]    # Vector of cost tensors for each player (e.g. cost_tensors[n] is cost tensor for player n)
p = compute_equilibrium(cost_tensors) 
```

Result: 
``` 
Major Iterations. . . . 103
Minor Iterations. . . . 1742
Restarts. . . . . . . . 2
Crash Iterations. . . . 0
Gradient Steps. . . . . 12
Function Evaluations. . 743
Gradient Evaluations. . 106
Basis Time. . . . . . . 0.004183
Total Time. . . . . . . 0.244047
Residual. . . . . . . . 3.828238e-14
4-element Vector{Vector{Float64}}:
 [0.0, 0.0751815290826188, 0.9248184709173812]
 [0.0030312794209868166, 0.9969687205790132]
 [0.34443648257564574, 0.6555635174243543]
 [0.09433892133590749, 0.2812651913857481, 0.6243958872783444]
```

 
 
 


