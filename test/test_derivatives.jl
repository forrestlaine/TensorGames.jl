function compute_value(cost_tensors, i=1; init=nothing)
    sol = compute_equilibrium(cost_tensors; initialization=init, convergence_tolerance=1e-10)
    value = expected_cost(sol.x, cost_tensors[i])
end

@testset "derivative computation" begin
    for N ∈ [2,3,4]
        for iteration ∈ 1:3
            ϵ = 1e-6
            ϵ2 = 1e-3
            d_max = 50.0 # ignore very large numerical derivatives, as they imply non-local solution
            sol_tol = 1e-10
            d = 3*ones(Int,N)
            cost_tensors = [randn(d...) for n ∈ 1:N]
            sol = compute_equilibrium(cost_tensors; convergence_tolerance=1e-10)
            x = vcat(sol.x...)
            init = [x; sol.λ]

            values = [compute_value(cost_tensors, i) for i ∈ 1:N]
            value_grads = [Zygote.gradient(compute_value, cost_tensors, i) for i ∈ 1:N]
             
            for n = 1:N
                cinds = CartesianIndices(cost_tensors[n])
                for i ∈ eachindex(cost_tensors[n])
                    ct2 = deepcopy(cost_tensors)
                    ct2[n][i] += ϵ

                    values_perturbed = [compute_value(ct2, i; init=init) for i ∈ 1:N]

                    ii = cinds[i]
                    num_derivs = (values_perturbed-values)./ϵ
                    if any(abs.(num_derivs) .≥ d_max)
                        continue
                    end
                    for j ∈ 1:length(num_derivs)
                        @test ≈(num_derivs[j], value_grads[j][1][n][ii]; atol=ϵ2)
                        #@test ≈(D[n,ii,j], num_deriv[j]; atol=ϵ2)
                    end
                end
            end
        end
    end
end

    
#@testset "derivative computation" begin
#    for N ∈ [2,3,4]
#        for iteration ∈ 1:5
#            ϵ = 1e-6
#            ϵ2 = 1e-3
#            d_max = 50.0 # ignore very large numerical derivatives, as they imply non-local solution
#            sol_tol = 1e-10
#            d = 3*ones(Int,N)
#            cost_tensors = [randn(d...) for n ∈ 1:N]
#
#            D = zeros(N, d..., sum(d))
#            sol = compute_equilibrium(cost_tensors; convergence_tolerance=sol_tol)
#            
#            x = vcat(sol.x...)
#            init = [x; sol.λ]
#            compute_derivatives!(D, sol)
#            for n = 1:N
#                cinds = CartesianIndices(cost_tensors[n])
#                for i ∈ eachindex(cost_tensors[n])
#                    ct2 = deepcopy(cost_tensors)
#                    ct2[n][i] += ϵ
#                    sol2 = compute_equilibrium(ct2; initialization=init, convergence_tolerance=sol_tol)
#                    x2 = vcat(sol2.x...)
#                    ii = cinds[i]
#                    num_deriv = (x2-x)./ϵ
#                    if any(abs.(num_deriv) .≥ d_max)
#                        continue
#                    end
#                    for j ∈ 1:length(num_deriv)
#                        @test ≈(D[n,ii,j], num_deriv[j]; atol=ϵ2)
#                    end
#                end
#            end
#        end
#    end
#end
#
