#=
Chance Constrained bimatrix game: Rock, paper, scissors


min x' A y
 x
s.t. x' e = 1, x >= 0, x1 y1 >= 1/4

min x' B y
 y
s.t. y' e = 1, y >= 0, x1 y1 >= 1/4
=#
A = [0.0 -1 1
    1 0 -1
    -1 1 0]
B = -A # Zero sum game
P = [1.0 0 0
    0 0 0
    0 0 0]
Q = P # Shared constraint
confidence = [1 / 4, 1 / 4]

cost_tensors = [A, B]
constraint_tensors = [P, Q]

import TensorGames

sol = TensorGames.compute_equilibrium(cost_tensors, constraint_tensors, confidence;
    initialization=nothing,
    ϵ=0.0,
    silent=true,
    convergence_tolerance=1e-6)

#=
Expected solution:
x = y = [1/2, 1/2, 0]
=#
@test sol.x[1] == [1 / 2, 1 / 2, 0] broken = true
@test sol.x[2] == [1 / 2, 1 / 2, 0] broken = true