@testset "derivative computation" begin
    for N ∈ [2,3,4]
        for iteration ∈ 1:5
            ϵ = 1e-6
            ϵ2 = 1e-3
            sol_tol = 1e-10
            d = 3*ones(Int,N)
            cost_tensors = [randn(d...) for n ∈ 1:N]
            D = zeros(N, d..., sum(d))
            sol = compute_equilibrium(cost_tensors; convergence_tolerance=sol_tol)
            x = vcat(sol.x...)
            init = [x; sol.λ]
            compute_derivatives!(D, sol)
            for n = 1:N
                cinds = CartesianIndices(cost_tensors[n])
                for i ∈ eachindex(cost_tensors[n])
                    ct2 = deepcopy(cost_tensors)
                    ct2[n][i] += ϵ
                    sol2 = compute_equilibrium(ct2, init; convergence_tolerance=sol_tol)
                    x2 = vcat(sol2.x...)
                    ii = cinds[i]
                    num_deriv = (x2-x)./ϵ
                    for j ∈ 1:length(num_deriv)
                        @test ≈(D[n,ii,j], num_deriv[j]; atol=ϵ2)
                    end
                end
            end
        end
    end
end
