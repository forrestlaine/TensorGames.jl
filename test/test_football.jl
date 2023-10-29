import TensorGames

c = 1

A = [-1 c; 0 -1.0]

B = [1 c; 0 1.0]



P = [0 0;
     0 0.0]

# P2 = [0 0
#       0 1.0]

confidence = [1.0]


z0 = rand(11)
cost_tensors = [A, B]
constraint_tensors = [P]
constraint_ownership = [[]]

sol = TensorGames.compute_equilibrium(cost_tensors, constraint_tensors, constraint_ownership, confidence;
    initialization=z0,
    ϵ=0.0,
    prob_backwards=true,
    silent=true,
    convergence_tolerance=1e-5)


sol.x