#=
Bimatrix game: Rock, paper, scissors

min x' A y
 x
s.t. x' e = 1, x >= 0

min x' B y
 y
s.t. y' e = 1, y >= 0

Player 1's cost matrix (A), P1 chooses row, P2 chooses col:
            P2
        R   P   S
    R   0  -1   1
P1  P   1   0  -1
    S  -1   1   0
=#
A = [0.0 -1 1
    1 0 -1
    -1 1 0]
B = -A # Zero sum game

cost_tensors = [A, B]

sol = TensorGames.compute_equilibrium(cost_tensors, nothing, nothing;
    initialization=nothing,
    ϵ=0.0,
    silent=true,
    convergence_tolerance=1e-6)

#=
Expected solution:
x = y = [1/3, 1/3, 1/3]
=#
@test sol.x[1] == [1 / 3, 1 / 3, 1 / 3] skip = false
@test sol.x[2] == [1 / 3, 1 / 3, 1 / 3] skip = false