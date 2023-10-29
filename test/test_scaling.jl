using TensorGames
using LinearAlgebra
using Plots
using Base.GC

min_n = 2
max_n = 2

min_m = 20
max_m = 20

min_c = 1
max_c = 20

max_trials = 20

result = -ones(max_n, max_m)

for n ∈ min_n:max_n

     for m ∈ min_m:max_m
          for c ∈ min_c: max_c

                    for t in 1:max_trials

                         

                         # c = m 
                         k = fill(m, n)

                         cost_tensors = []

                         function make_cost_tensor(p)
                              Ai = zeros(k...)
                              for p_d in 1:m
                                   selectdim(Ai, p, p_d) .= (rand() - 0.5)
                              end
                              Ai
                         end

                         cost_tensors = map(make_cost_tensor, 1:n)
                         cost_tensors = [i for i in cost_tensors]


                         function make_constraint_tensor(c)
                              Gi = zeros(k...)
                              inds = CartesianIndices(Gi)
                              for ind ∈ inds
                                   if c ∈ Tuple(ind)
                                        Gi[ind] = 1.0
                                   end
                              end
                              Gi
                         end

                         z0 = [1/m * ones(n*m); zeros(n+c+n*m)]

                         constraint_tensors = map(make_constraint_tensor, 1:c)
                         constraint_ownership = [1:n for _ in constraint_tensors]
                         confidence = [0.03 for _ in constraint_tensors]
                         constraint_tensors = [i for i in constraint_tensors]

                         # z0 = zeros(n*m + n + c + n*m)

                         # z0


                         function do_test()
                              sol = TensorGames.compute_equilibrium(cost_tensors, constraint_tensors, constraint_ownership, confidence;
                                   initialization=z0,
                                   ϵ=0.0,
                                   silent=true,
                                   prob_backwards=false,
                                   convergence_tolerance=1e-3
                                   )
                         end
                         @time do_test()
                         # println("===")
                         # println(sol.status)
                         # print("Simplex constraint: \t")
                         # println(abs(sum([sum(sol.x[x]) for x in 1:n])/n - 1.0) < 0.01)
                         # print("Chance constraint: \t")
                         # println(TensorGames.tensor_product(sol.vars, constraint_tensors[1]) > 0.24)

                    # println("($(n) players, $(m) tasks: $(elapsed/max_trials) seconds")
               end
               println("===")
          end
     end
end

Plots.wireframe(1:max_n, 1:max_m, result)