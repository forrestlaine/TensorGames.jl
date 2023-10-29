using SparseArrays

"""
Each row corresponds to a players strategies, the first column is the index of the first strategy, and the second column is the index of the last strategy.
"""
function primal_inds(d)
    N = length(d) # number of players
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

"""
?
"""
function prob_prod(x, ind, primal_inds, n...)
    N = size(primal_inds, 1)
    length(n) == N && return 1.0
    prod(x[primal_inds[i, 1]+ind[i]-1] for i ∈ 1:N if i ∉ n)
end

"""
?
"""
function tensor_product(x::Vector{T}, CT, indices, primal_inds)::T where {T}
    sum(CT[ind] * prob_prod(x, ind, primal_inds) for ind ∈ indices)
end

"""
?
"""
function tensor_product(x, cost_tensor)
    d = size(cost_tensor)
    tensor_indices = CartesianIndices(cost_tensor)
    primal_indices = primal_inds(d)
    tensor_product(reduce(vcat, x), cost_tensor, tensor_indices, primal_indices)
end

function ChainRulesCore.rrule(::typeof(tensor_product), x, cost_tensor)
    N = length(x)
    d = size(cost_tensor)
    tensor_indices = CartesianIndices(cost_tensor)
    primal_indices = primal_inds(d)
    xx = reduce(vcat, x)
    value = tensor_product(xx, cost_tensor, tensor_indices, primal_indices)

    function expected_cost_pullback(∂value)
        ∂x = zeros(sum(d))
        ind = 1
        for p ∈ 1:N
            for p_d ∈ primal_indices[p, 1]:primal_indices[p, 2]
                tensor_gradient!(∂x, ind, cost_tensor, xx, p, p_d, tensor_indices, primal_indices, 1)
                ind += 1
            end
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

"""
Gives the gradient of a tensor product with respect to all decision variables of one player.
This statically adds the result onto an existing array in a specific location.

* `v`: The array onto which to add the result.
* `loc`: The target location in v.
* `T`: The tensor in the product.
* `x`: The vectorized list of decision variables for all players.
* `indices`: Indices of the tensor, enumerating all joint options. (i.e., CartesianIndices(CT)). 
* `primal_indices`: Two-column array which describes where the decision variables for each
    player are located in `x`; see `primal_inds`
* `scalars`: Value by which to scale the gradient by. (Often a Lagrangian multiplier.)

"""

function tensor_gradient!(v, loc, T, x, p, p_d, indices, primal_inds, scalar)
    for ind ∈ indices
        if ind[p] == p_d
            prob = prob_prod(x, ind, primal_inds, p)
            v[loc] += T[ind] * prob * scalar
        end
    end
    Cint(0)
end

function tensor_gradient(T, x, p, p_d, indices, primal_inds, scalar)
    r = 0
    for ind ∈ indices
        if ind[p] == p_d
            prob = prob_prod(x, ind, primal_inds, p)
            r += T[ind] * prob * scalar
        end
    end
    r
end

"""
Gives one element of the hessian of a tensor product, the one with respect
  to two specific players and decisions.
This statically adds the result onto an existing array in a specific location.

* v: The target array for the value to be placed in.
* loc: Target location in v.
* T: The tensor in the product.
* `p1`, `p1_d`: Player 1 index, and their decision index.
* `p2`, `p2_d`: Player 2 index, and their decision index.
* `indices`: Indices of the tensor, enumerating all joint options. (i.e., CartesianIndices(CT)). 
* `primal_indices`: Two-column array which describes where the decision variables for each
    player are located in `x`; see `primal_inds`
"""
function tensor_hessian!(v, loc, T, x, p1, p1_d, p2, p2_d, indices, primal_inds, scalar)
    (p1 == p2) && return Cint(0) # Double derivative of a tensor product is zero
    for ind ∈ indices
        if ind[p1] == p1_d && ind[p2] == p2_d
            v[loc] += T[ind] * prob_prod(x, ind, primal_inds, p1, p2) * scalar
        end
    end
    Cint(0)
end

struct Wrapper{L} <: Function
    N::Cint
    m::NTuple{L,Cint}
    C::Cint

    primal_inds::Matrix{Cint}
    dual_inds_unitude::Vector{Cint}
    dual_inds_mixture::Vector{Cint}
    dual_inds_positive::Vector{Cint}

    tensors::Vector{Array{Float64,L}}
    tensor_indices::CartesianIndices{L,NTuple{L,Base.OneTo{Int64}}}
    
    constraint_tensors
    constraint_tensor_indices
    constraint_ownership
    constraint_eps

    force_fairness

end

"""
cF
"""
function (T::Wrapper)(n::Cint, x::Vector{Cdouble}, f::Vector{Cdouble})
    @assert n == length(x) == length(f)

    ind = 1
    f[1:n] .= 0

    # Stationarity: ∇L(z)
    # = ∇_xi[T_i⊚x] - λi ∇_xi[1xi] - [γc ∇_xi[G_i⊚x] ∀c owned by i]
    for (p, tensor) ∈ enumerate(T.tensors)
        for p_d ∈ 1:T.m[p]

            # Add cost
            tensor_gradient!(f, ind, tensor, x, p, p_d, T.tensor_indices, T.primal_inds, 1)

            # Subtract mixture
            for (c, constraint_tensor) ∈ enumerate(T.constraint_tensors)
                if p ∈ T.constraint_ownership[c]
                    tensor_gradient!(f, ind, constraint_tensor, x, p, p_d, T.constraint_tensor_indices, T.primal_inds, -x[T.dual_inds_mixture[c]])
                end
            end

            # Fairness forcing - manually disable other constraints if shared constraint is active
            # Experimental! 
            q = tensor_gradient(T.constraint_tensors[1], x, p, p_d, T.constraint_tensor_indices, T.primal_inds, -x[T.dual_inds_mixture[1]])
            if T.force_fairness && !(q ≈ 0.0)
                a = 0.0
            else
                a = 1.0
            end
            


            # Subtract unitude
            f[ind] -= x[T.dual_inds_unitude[p]] * a

            # Subtract positivity
            f[ind] -= x[T.dual_inds_positive[ind]] * a

            ind += 1
        end
    end

    # Feasibility: Unitude
    # = e'x - 1
    for n ∈ 1:T.N
        f[ind] = -1.0
        for i ∈ T.primal_inds[n, 1]:(T.primal_inds[n, 2])
            f[ind] += x[i] # why not just use sum? C things?
        end
        ind += 1
    end

    # Feasibility: Mixture
    for (c, constraint_tensor) ∈ enumerate(T.constraint_tensors)
        f[ind] = tensor_product(x, constraint_tensor, T.tensor_indices, T.primal_inds) - T.constraint_eps[c]
        ind += 1
    end

    # Feasibility: Positivity
    k = 1
    for (p, tensor) ∈ enumerate(T.tensors)
        for p_d ∈ 1:T.m[p]
            f[ind] += x[(T.primal_inds[p, 1]:T.primal_inds[p, 2])[p_d]]
            ind += 1
            k += 1
        end
    end

    Cint(0)
end

"""
cJ
"""
function (T::Wrapper)(n::Cint,
    nnz::Cint,
    x::Vector{Cdouble},
    col::Vector{Cint},
    len::Vector{Cint},
    row::Vector{Cint},
    data::Vector{Cdouble})
    @assert n == length(x) == length(col) == length(len)
    @assert nnz == length(row) == length(data)

    col[begin:end]  .= 0
    len[begin:end]  .= 0
    row[begin:end]  .= 0
    data[begin:end] .= 0


    # The full Jacobian is:
    # [ d∇L/dx  d∇L/dλ  d∇L/dγ d∇L/dξ ]
    # [ dh/dx   dh/dλ   dh/dγ  dh/dξ  ]
    # [ dg/dx   dg/dλ   dg/dγ  dg/dξ  ]
    # [ dp/dx   dp/dλ   dg/dγ  dg/dξ  ]

    data_ind = 1
    col_ind = 1
    primal_ind = 1
    # Primal columns xi
    for p1 ∈ 1:T.N
        for p1_d ∈ 1:T.m[p1]

            col[col_ind] = data_ind
            row_ind = 1

            # Compute d(∇L[p2, p2_d])/dx[p1, p1_d]  
            for p2 ∈ 1:T.N
                for p2_d ∈ 1:T.m[p2]
                    if p1 != p2 # Double derivative of a tensor product is always zero
                        # Add hessian of cost
                        tensor_hessian!(data, data_ind, T.tensors[p2], x, p2, p2_d, p1, p1_d, T.tensor_indices, T.primal_inds, 1)
                        
                        # Unitude is a linear constraint so its hessian is always zero; nothing to do here

                        # Add hessian of mixture constraints
                        #   We sum up several of these for every ∇L so don't move down the column in this loop
                        for (c, constraint_tensor) ∈ enumerate(T.constraint_tensors)
                            if p2 ∈ T.constraint_ownership[c]
                                tensor_hessian!(data, data_ind, constraint_tensor, x, p2, p2_d, p1, p1_d, T.constraint_tensor_indices, T.primal_inds, -x[T.dual_inds_mixture[c]])
                            end
                        end


                        len[col_ind] += 1
                        row[data_ind] = row_ind
                        data_ind += 1
                    end
                    row_ind += 1
                end
            end

            # Compute dh[p2]/dx[p1, p1_d]
            #   It's 1 if p2 = p1, zero otherwise
            for p2 ∈ 1:T.N
                if p1 == p2
                    data[data_ind] = (1)
                    row[data_ind] = row_ind
                    len[col_ind] += 1
                    data_ind += 1
                end
                row_ind += 1
            end

            # Compute dg[c]/dx[p1]
            #   Players may affect all constraints, regardless of the ones they care about
            #   So we have to get a gradient for every one of these
            for c ∈ 1:T.C
                tensor_gradient!(data, data_ind, T.constraint_tensors[c], x, p1, p1_d, T.constraint_tensor_indices, T.primal_inds, 1)
                row[data_ind] = row_ind
                len[col_ind] += 1
                data_ind += 1

                row_ind += 1
            end

            # Compute dp[c]/dx[p1]
            #   Positivity only cares about its corresponding decision variable
            
            len[col_ind] += 1
            row[data_ind] = row_ind + primal_ind - 1
            data[data_ind] = 1
            primal_ind += 1
            data_ind += 1


            col_ind += 1
        end
    end

    # Unitude dual columns λi
    for p1 ∈ 1:T.N

        col[col_ind] = data_ind
        row_ind = 1

        # Compute d(∇L[p2, p2_d])/dλ[p1]

        for p2 ∈ 1:T.N
            for _ ∈ 1:T.m[p2]
                if p1 == p2
                    data[data_ind] = (-1)
                    row[data_ind] = row_ind
                    len[col_ind] += 1
                    data_ind += 1
                end

                row_ind += 1
            end
        end
        
        # We don't care about the jacobian for feasibilities; they don't directly involve λ
        col_ind += 1
    end

    k = 1
    # Mixture dual columns γi
    for c ∈ 1:T.C

        col[col_ind] = data_ind
        row_ind = 1
        
        # Compute d(∇L[p2, p2_d])/dγ[p1]
        # This is ∇_xi[G_c ⊚ x], but only if x cares about constraint c, otherwise it doesn't count
        for p2 ∈ 1:T.N
            for p2_d ∈ 1:T.m[p2]
                
                # unitude_active = (T.dual_inds_unitude[p2] > -1e-2)
                # positivity_active = (T.dual_inds_positive[k] < 1e-2)

                if p2 ∈ T.constraint_ownership[c]
                    tensor_gradient!(data, data_ind, T.constraint_tensors[1], x, p2, p2_d, T.constraint_tensor_indices, T.primal_inds, -1)
                    
                    # if unitude_active || positivity_active
                    #     data[data_ind] = 0
                    # end

                    row[data_ind] = row_ind
                    len[col_ind] += 1
                    data_ind += 1
                end
                row_ind += 1
                k += 1
            end
        end

        # Once again we don't care about the feasibilities; they don't directly use γ
        col_ind += 1
    end

    # Positivity columns ξi
    primal_ind = 1
    for p1 ∈ 1:T.N
        for p1_d ∈ 1:T.m[p1]
            col[col_ind] = data_ind
            len[col_ind] = 1
            row[data_ind] = primal_ind
            data[data_ind] = -1

            primal_ind += 1
            col_ind += 1
            data_ind += 1
        end
    end
    Cint(0)
end

function compute_equilibrium(cost_tensors, constraint_tensors, constraint_ownership, confidence;
    initialization=nothing,
    ϵ=0.0,
    silent=false,
    force_fairness=true,
    prob_backwards=false,
    convergence_tolerance=1e-6)

    N = Cint(length(cost_tensors))       # number of players
    m = Cint.(size(cost_tensors[1]))     # tuple containing the number of strategies corresponding to each player
    C = Cint(length(constraint_tensors)) # number of constraints
    P = sum(m)                           # number of primals 

    primal_indices = primal_inds(m)
    dual_inds_unitude = Vector{Cint}(P+1      : P+N)
    dual_inds_mixture = Vector{Cint}(P+N+1    : P+N+C)
    dual_inds_positive = Vector{Cint}(P+N+C+1  : P+N+C+P)


    cost_tensor_inds = CartesianIndices(cost_tensors[1])
    constraint_tensor_inds = CartesianIndices(constraint_tensors[1])

    nnz = (P+N+C+P)^2 + 1 # TODO: This can be tightened

    wrapper! = Wrapper(
        N, m, C,
        primal_indices, dual_inds_unitude, dual_inds_mixture, dual_inds_positive,
        cost_tensors, cost_tensor_inds,
        constraint_tensors, constraint_tensor_inds, constraint_ownership, confidence,
        force_fairness)


    if prob_backwards
        lb = [
            -1e20 * ones(Cdouble, P);
            -1e20 * ones(Cdouble, N);       
            -1e20 * ones(Cdouble, C);       
            zeros(Cdouble, P)]            

        ub = [
            1e20 * ones(Cdouble, P);  
            1e20 * ones(Cdouble, N);        
            zeros(Cdouble, C);      
            1e20    * ones(Cdouble, P)]
    else
        lb = [
            -1e20 * ones(Cdouble, P);
            -1e20 * ones(Cdouble, N);       
            zeros(Cdouble, C);       
            zeros(Cdouble, P)]            

        ub = [
            1e20 * ones(Cdouble, P);  
            1e20 * ones(Cdouble, N);        
            1e20    * ones(Cdouble, C);      
            1e20    * ones(Cdouble, P)]
    end


    z = zeros(Cdouble, P + N + C + P)


    if isnothing(initialization)
        for n ∈ 1:N # start the primals on unitude; don't do anything with the duals
            start_ind = primal_indices[n, 1]
            end_ind = primal_indices[n, 2]
            z[start_ind:end_ind] .= 1.0 / m[n]
        end
    else
        if length(initialization) != length(z)
            println("Expected initialization of size $(length(z)); got $(length(initialization))")
        end
        z .= initialization
    end

    # Debug: Print f and J for z0
    #----------------------------------------------------

    col = zeros(Cint, P + N + C + P)
    len = zeros(Cint, P + N + C + P)
    row = zeros(Cint, nnz)
    data= zeros(nnz)
    
    wrapper!(Cint(P + N + C + P), Cint(nnz), z, col, len, row, data)


    J = zeros(P + N + C + P, P + N + C + P)
    for (c, c_index) in enumerate(col)
        for r_index in c_index:(c_index+len[c]-1)
            r = row[r_index]
            J[r,c] = data[r_index]
        end
    end

    # display(J)

    f = zeros(P+N+C+P)
    wrapper!(Cint(P+N+C+P), z, f)
    #----------------------------------------------------

    status, vars::Vector{Cdouble}, info = PATHSolver.solve_mcp(
        wrapper!,
        wrapper!,
        lb,
        ub,
        z;
        # nnz,
        convergence_tolerance,
        silent,
        major_iteration_limit=10000000,
        minor_iteration_limit=10000000,
        cumulative_iteration_limit=10000000,
        max_wall_time=10,
        proximal_perterbation=0.1,
        lemke_start="first",
        nms="no",
        )

    x = [vars[primal_indices[n, 1]:primal_indices[n, 2]] for n ∈ 1:N]
    λ = vars[dual_inds_unitude]
    γ = vars[dual_inds_mixture]
    ξ = vars[dual_inds_positive]

    (; x, λ, γ, ξ, vars, status, x_raw = vars[primal_indices], _deriv_info=(; ϵ, wrapper!, nnz=Cint(nnz), cost_tensor_inds, primal_indices))
end

function ChainRulesCore.rrule(::typeof(compute_equilibrium),
    cost_tensors;
    initialization=nothing,
    ϵ=0.0,
    silent=true,
    convergence_tolerance=1e-6)
    res = compute_equilibrium(cost_tensors, constraint_tensors, confidence; initialization, ϵ, silent, convergence_tolerance)

    _back = _compute_equilibrium_pullback(res)

    function compute_equilibrium_pullback(∂res)
        ∂self = NoTangent()

        ∂cost_tensors = let
            full_sensitivities = map(∂res.x, res.x) do r, x
                r isa ZeroTangent ? zeros(x) : r
            end
            derivs = reduce(vcat, full_sensitivities)

            map(_back) do ∂cost_tensor
                dropdims(sum(∂cost_tensor .* derivs; dims=1); dims=1)
            end
        end
        ∂self, ∂cost_tensors
    end

    res, compute_equilibrium_pullback
end

function _compute_equilibrium_pullback(res; bound_tolerance=1e-6, singularity_tolerance=1e-6)
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
