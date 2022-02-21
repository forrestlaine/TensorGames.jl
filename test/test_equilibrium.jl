function valid_perturbation(x)
    n = length(x)
    p = zeros(Float64, n)
    j = rand(1:n)
    for i ∈ 1:n
        if i ≠ j
            p[i] = randn()
            if x[i] < 1e-5
                p[i] = max(0, p[i])
            elseif x[i] > 1-1e-5
                p[i] = min(0, p[i])
            end
        end
    end
    sp = sum(p)
    if ≈(sp, 0; atol=1e-4) || (x[j] < 1e-5 && sp > 0) || (x[j] > 1-1e-5 && sp < 0)
        p .= 0
        return p
    end
    p[j] = -sum(p)
    p = p ./ norm(p)
    return p
end

@testset "equilibrium conditions" begin
    sol_tol = 1e-9
    ϵ = 1e-3
    for N ∈ [2,3,4]
        d = 3*ones(Int,N)
        cost_tensors = [randn(d...) for n ∈ 1:N]
        sol = compute_equilibrium(cost_tensors; convergence_tolerance=sol_tol)
        xvec = vcat(sol.x...)
        x = sol.x


        for n ∈ 1:N
            @test sum(sol.x[n]) ≈ 1.0
            @test all(sol.x[n] .≥ 0.0)
            true_cost = expected_cost(cost_tensors[n], xvec, sol.tensor_indices, sol.primal_inds)
            for j ∈ 1:100
                p = valid_perturbation(sol.x[n])
                x2 = deepcopy(sol.x)
                x2[n] += 1e-4*p
                xvec2 = vcat(x2...)
                other_cost = expected_cost(cost_tensors[n], xvec2, sol.tensor_indices, sol.primal_inds)
                @test other_cost ≥ true_cost - ϵ
            end
        end
    end
end
