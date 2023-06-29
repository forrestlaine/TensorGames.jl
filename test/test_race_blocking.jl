import TensorGames

A = [1  1
     0  0.0]

B = [1 0
     1 0.0]

P1 = [0 0
      0 1.0]

# P2 = [0 0
#       0 1.0]

confidence = [0.25]

z0 = [#=x=# 0.5; 0.5; 0.5; 0.5;  #=λ=# 1; 1; #=γ=# 2]

cost_tensors = [A, B]
constraint_tensors = [P1]
constraint_ownership = [[1]]

TensorGames.compute_equilibrium(cost_tensors, constraint_tensors, constraint_ownership, confidence;
    initialization=z0,
    ϵ=0.0,
    silent=true,
    convergence_tolerance=1e-9)

#=
Expected solution:
x = y = [1/2, 1/2, 0]
=#
# @test sol.x[1] ≈ [1 / 2, 1 / 2, 0] 
# @test sol.x[2] ≈ [1 / 2, 1 / 2, 0] 