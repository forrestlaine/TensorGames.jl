function primal_inds(d)
    N = length(d)
    inds = zeros(Cint, N, 2)
    inds[1, 1] = 1
    inds[1, 2] = d[1]
    for n ∈ 2:N
        inds[n, 1] = inds[n-1, 2] + 1
        inds[n, 2] = inds[n, 1] + d[n] - 1
    end
    inds
end
@non_differentiable primal_inds(d)

function prob_prod(x, ind, primal_inds, n...)
    N = size(primal_inds, 1)
    length(n) == N && return 1.0
    prod(x[primal_inds[i, 1]+ind[i]-1] for i ∈ 1:N if i ∉ n)
end

function expected_cost(CT, x::Vector{T}, indices, primal_inds)::T where {T}
    val = sum(CT[ind] * prob_prod(x, ind, primal_inds) for ind ∈ indices)
end

function expected_cost(x, cost_tensor)
    N = length(x)
    d = size(cost_tensor)
    tensor_indices = CartesianIndices(cost_tensor)
    primal_indices = primal_inds(d)
    expected_cost(cost_tensor, reduce(vcat, x), tensor_indices, primal_indices)
end

function ChainRulesCore.rrule(::typeof(expected_cost), x, cost_tensor)
    N = length(x)
    d = size(cost_tensor)
    tensor_indices = CartesianIndices(cost_tensor)
    primal_indices = primal_inds(d)
    xx = reduce(vcat, x)
    value = expected_cost(cost_tensor, xx, tensor_indices, primal_indices)

    function expected_cost_pullback(∂value)
        ∂x = zeros(sum(d))
        for n ∈ 1:N
            grad!(∂x, cost_tensor, xx, n, tensor_indices, primal_indices)
        end
        ∂x .*= ∂value
        ∂x = [∂x[primal_indices[n, 1]:primal_indices[n, 2]] for n ∈ 1:N]

        ∂ct = zero(cost_tensor)
        for ind ∈ tensor_indices
            ∂ct[ind] = ∂value * prob_prod(xx, ind, primal_indices)
        end
        NoTangent(), ∂x, ∂ct
    end
    value, expected_cost_pullback
end

function grad!(f, CT, x, n, indices, primal_inds)
    f[primal_inds[n, 1]:primal_inds[n, 2]] .= 0.0
    for ind ∈ indices
        prob = prob_prod(x, ind, primal_inds, n)
        for i ∈ 1:primal_inds[n, 2]+1-primal_inds[n, 1]
            if ind[n] == i
                f[primal_inds[n, 1]+i-1] += CT[ind] * prob
            end
        end
    end
    Cint(0)
end

function jac(J, loc, T, x, n1, n2, i, j, indices, primal_inds)
    l1 = primal_inds[n1, 2] - primal_inds[n1, 1] + 1
    l2 = primal_inds[n2, 2] - primal_inds[n2, 1] + 1
    (n1 == n2) && return zeros(l1, l2)
    for ind ∈ indices
        if ind[n1] == i && ind[n2] == j
            J[loc] += T[ind] * prob_prod(x, ind, primal_inds, n1, n2)
        end
    end
    Cint(0)
end

struct Wrapper{L} <: Function
    tensors::Vector{Array{Float64,L}}
    tensor_indices::CartesianIndices{L,NTuple{L,Base.OneTo{Int64}}}
    primal_inds::Matrix{Cint}
    dual_inds::Vector{Cint}
    N::Cint
    m::NTuple{L,Cint}
    num_primals::Cint
end

function (T::Wrapper)(n::Cint, x::Vector{Cdouble}, f::Vector{Cdouble})
    ind = 0
    for (n, tensor) ∈ enumerate(T.tensors)
        f[ind+1:ind+T.m[n]] .= 0.0
        grad!(f, tensor, x, n, T.tensor_indices, T.primal_inds)
        for j ∈ ind+1:ind+T.m[n]
            f[j] -= x[T.dual_inds[n]]
        end
        ind += T.m[n]
    end
    for n ∈ 1:T.N
        f[ind+n] = -1.0
        for i ∈ T.primal_inds[n, 1]:T.primal_inds[n, 2]
            f[ind+n] += x[i]
        end
    end
    Cint(0)
end

function (T::Wrapper)(n::Cint,
    nnz::Cint,
    x::Vector{Cdouble},
    col::Vector{Cint},
    len::Vector{Cint},
    row::Vector{Cint},
    data::Vector{Cdouble})
    @assert n == length(x) == length(col) == length(len)
    @assert nnz == length(row) == length(data)

    col[1] = 1
    ind = 2
    for n in 1:T.N
        inc = 1
        for j in (1:T.N)
            if j ≠ n
                inc = inc + T.m[j]
            end
        end
        for _ in 1:T.m[n]
            col[ind] = col[ind-1] + inc
            len[ind-1] = inc
            ind += 1
        end
    end
    for n in 1:T.N
        if n ≠ T.N
            col[ind] = col[ind-1] + T.m[n]
        end
        len[ind-1] = T.m[n]
        ind += 1
    end

    ind = 0
    for n_col in 1:T.N
        for c in 1:T.m[n_col]
            for n_row in 1:T.N
                if n_row == n_col
                    continue
                end
                for p in 1:T.m[n_row]
                    row[ind+1] = Cint(T.primal_inds[n_row, 1] + p - 1)
                    data[ind+1] = 0.0
                    jac(data, ind + 1, T.tensors[n_row], x, n_row, n_col, p, c, T.tensor_indices, T.primal_inds)
                    ind += 1
                end
            end
            row[ind+1] = Cint(T.num_primals + n_col)
            data[ind+1] = 1.0
            ind += 1
        end
    end
    for n in 1:T.N
        row[ind+1:ind+T.m[n]] = T.primal_inds[n, 1]:T.primal_inds[n, 2]
        data[ind+1:ind+T.m[n]] .= -1.0
        ind += T.m[n]
    end
    return Cint(0)
end

function compute_equilibrium(cost_tensors::AbstractVector{<:AbstractArray{<:ForwardDiff.Dual{T}}}; kwargs...) where {T}
    # strip off the duals:
    cost_tensors_v = [ForwardDiff.value.(c) for c in cost_tensors]
    cost_tensors_p = [ForwardDiff.partials.(c) for c in cost_tensors]
    # forward pass
    res = compute_equilibrium(cost_tensors_v; kwargs...)
    # backward pass
    # 1. compute jacobian
    _back = _compute_equilibrium_pullback(res)
    # 2. project input-sensitivy through jacobian to yield output sensitivy
    x_p = let
        # output sensitivities stacked for all players
        x_p_stacked = sum(zip(cost_tensors_p, _back)) do (cost_tensor_p, ∂cost_tensor)
            inflated_cost_tensor_p = reshape(cost_tensor_p, 1, size(cost_tensor_p)...)
            dims = 2:ndims(inflated_cost_tensor_p)
            reshape(sum(∂cost_tensor .* inflated_cost_tensor_p; dims), size(∂cost_tensor, 1))
        end
        # unstacking the output sensitivities
        n_actions_per_player = size(first(cost_tensors))
        x_p_stacked_it = Iterators.Stateful(x_p_stacked)
        map(n_actions_per_player) do n_actions
            Iterators.take(x_p_stacked_it, n_actions) |> collect
        end
    end

    # 3. glue primal and dual results together into a ForwardDiff.Dual-valued result
    x_d = [ForwardDiff.Dual{T}.(xi_v, xi_p) for (xi_v, xi_p) in zip(res.x, x_p)]

    (; x = x_d, res.λ, res._deriv_info)
end


function compute_equilibrium(cost_tensors;
    initialization = nothing,
    ϵ = 0.0,
    silent = true,
    convergence_tolerance = 1e-6)

    N = Cint(length(cost_tensors))
    m = Cint.(size(cost_tensors[1]))
    @assert all(m == size(tensor) for tensor ∈ cost_tensors)
    primal_indices = primal_inds(m)

    num_primals::Cint = sum(m)
    dual_inds = Vector{Cint}(num_primals+1:num_primals+N)

    tensor_indices = CartesianIndices(cost_tensors[1])
    nnz = 2 * num_primals
    for n = 1:N
        nnz += m[n] * (num_primals - m[n])
    end

    wrapper! = Wrapper(cost_tensors, tensor_indices, primal_indices, dual_inds, N, m, num_primals)

    lb = [ϵ * ones(Cdouble, num_primals); -1e20 * ones(Cdouble, N)]
    ub = 1e20 * ones(Cdouble, num_primals + N)
    z = zeros(Cdouble, num_primals + N)
    if isnothing(initialization)
        for n ∈ 1:N
            start_ind = primal_indices[n, 1]
            end_ind = primal_indices[n, 2]
            z[start_ind:end_ind] .= 1.0 / m[n]
        end
    else
        z .= initialization
    end

    n = Cint(sum(m) + N)
    x = randn(Cdouble, n)
    f = zeros(Cdouble, n)

    status, vars::Vector{Cdouble}, info = PATHSolver.solve_mcp(
        wrapper!,
        wrapper!,
        lb,
        ub,
        z;
        nnz,
        convergence_tolerance,
        silent)

    x = [vars[primal_indices[n, 1]:primal_indices[n, 2]] for n ∈ 1:N]
    λ = vars[dual_inds]

    (; x, λ, _deriv_info = (; ϵ, wrapper!, nnz = Cint(nnz), tensor_indices, primal_indices))
end

function ChainRulesCore.rrule(::typeof(compute_equilibrium),
    cost_tensors;
    initialization = nothing,
    ϵ = 0.0,
    silent = true,
    convergence_tolerance = 1e-6)
    res = compute_equilibrium(cost_tensors; initialization, ϵ, silent, convergence_tolerance)

    _back = _compute_equilibrium_pullback(res)

    function compute_equilibrium_pullback(∂res)
        ∂self = NoTangent()

        ∂cost_tensors = let
            full_sensitivities = map(∂res.x, res.x) do r, x
                r isa ZeroTangent ? zeros(x) : r
            end
            derivs = reduce(vcat, full_sensitivities)

            map(_back) do ∂cost_tensor
                dropdims(sum(∂cost_tensor .* derivs; dims = 1); dims = 1)
            end
        end
        ∂self, ∂cost_tensors
    end

    res, compute_equilibrium_pullback
end

function _compute_equilibrium_pullback(res; bound_tolerance = 1e-6, singularity_tolerance = 1e-6)
    primals = vcat(res.x...)
    vars = [primals; res.λ]
    n = Cint(length(vars))
    N = length(res.λ)
    d = [length(xi) for xi ∈ res.x]
    starts = cumsum([0; d[1:end-1]])
    num_primals = sum(d)

    lb = [res._deriv_info.ϵ * ones(num_primals); -Inf * ones(N)]
    unbound_indices = (vars .> (lb .+ bound_tolerance))
    unbound_primals = unbound_indices[1:end-N]
    nup = sum(unbound_primals)
    ubmap = zeros(Int, num_primals)
    idx = 1
    for i ∈ 1:num_primals
        if unbound_primals[i]
            ubmap[i] = idx
            idx += 1
        end
    end
    col = zeros(Cint, n)
    len = zeros(Cint, n)
    row = zeros(Cint, res._deriv_info.nnz)
    data = zeros(Cdouble, res._deriv_info.nnz)
    res._deriv_info.wrapper!(n, res._deriv_info.nnz, vars, col, len, row, data)

    colptr = zeros(Cint, n + 1)
    colptr[1:n] .= col
    colptr[n+1] = col[n] + len[n]

    ∂cost_tensors = [zeros(sum(d), d...) for _ ∈ 1:N]

    nJ = Matrix{Cdouble}((SparseMatrixCSC{Cdouble,Cint}(n, n, colptr, row, data))[unbound_indices, unbound_indices])
    factorization = qr(nJ)
    if any(abs(r) ≤ singularity_tolerance for r ∈ diag(factorization.R))
        # Artificially returning zero-derivatives if solution is non-isolated.
        return ∂cost_tensors
    end
    nJi = factorization \ (-I)

    for ind ∈ res._deriv_info.tensor_indices
        for n ∈ 1:N
            if ubmap[starts[n]+ind[n]] > 0
                ∂cost_tensors[n][unbound_primals, ind] .= (nJi[1:nup, ubmap[starts[n]+ind[n]]] *
                                                           prob_prod(primals, ind, res._deriv_info.primal_indices, n))
            end
        end
    end
    ∂cost_tensors
end
