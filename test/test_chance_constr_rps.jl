#=
Chance Constrained bimatrix game: Rock, paper, scissors


min x' A y
 x
s.t. x' e = 1, x >= 0, x1 y1 >= 1/4

min x' B y
 y
s.t. y' e = 1, y >= 0, x1 y1 >= 1/4
=#

import TensorGames


A = [0  1 -1
    -1  0  1
     1 -1  0.0]
B = -A # Zero sum game

P = [1 0 0
     0 1 0
     0 0 1.0]

confidence = [1]

z0 = [0.25; 0.75; 0.0; 1.0; 0.0; 0.0; 0; 0; 0; 0; 0; 0; 0; 0; 0]

cost_tensors = [A, B]
constraint_tensors = [P]
constraint_ownership = [[1, 2]]

sol = TensorGames.compute_equilibrium(cost_tensors, constraint_tensors, constraint_ownership, confidence;
    initialization=z0,
    ϵ=0.0,
    silent=true,
    prob_backwards=true,
    convergence_tolerance=1e-12)

#=
Expected solution:
x = y = [1/2, 1/2, 0]
=#
@test sol.x[1] ≈ [1 / 2, 1 / 2, 0] 
@test sol.x[2] ≈ [1 / 2, 1 / 2, 0] 