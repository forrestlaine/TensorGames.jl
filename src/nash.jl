function prob_prod(x, ind, primal_inds, n...)
    N = size(primal_inds,1)
    length(n) == N && return 1.0
    prod(x[primal_inds[i,1]+ind[i]-1] for i ∈ 1:N if i ∉ n)
end

function expected_cost(T, x, indices, primal_inds)
    val = sum(T[ind]*prob_prod(x,ind,primal_inds) for ind ∈ indices)
end

function grad!(f, T, x, n, indices, primal_inds)
    f[primal_inds[n,1]:primal_inds[n,2]] .= 0.0
    for ind ∈ indices
        for i ∈ 1:primal_inds[n,2]+1-primal_inds[n,1]
            if ind[n] == i
                f[primal_inds[n,1]+i-1] += T[ind]*prob_prod(x,ind,primal_inds,n)
            end
        end
    end
    Cint(0)
end

function jac(J,loc,T, x, n1, n2, i, j, indices, primal_inds)
    l1 = primal_inds[n1,2]-primal_inds[n1,1]+1
    l2 = primal_inds[n2,2]-primal_inds[n2,1]+1
    (n1 == n2) && return zeros(l1,l2)
    for ind ∈ indices
        if ind[n1] == i && ind[n2] == j
            J[loc] += T[ind]*prob_prod(x,ind,primal_inds,n1,n2) 
        end
    end
    Cint(0)
end

struct Wrapper{L} <: Function
    tensors::Vector{Array{Float64,L}}
    tensor_indices::CartesianIndices{L, NTuple{L, Base.OneTo{Int64}}}
    primal_inds::Matrix{Cint}
    dual_inds::Vector{Cint}
    N::Cint
    m::NTuple{L, Cint}
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
        for i ∈ T.primal_inds[n,1]:T.primal_inds[n,2]
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
                    row[ind+1] = Cint(T.primal_inds[n_row,1]+p-1)
                    data[ind+1] = 0.0
                    jac(data, ind+1, T.tensors[n_row], x, n_row, n_col, p, c, T.tensor_indices, T.primal_inds)
                    ind += 1
                end
            end
            row[ind+1] = Cint(T.num_primals+n_col)
            data[ind+1] = 1.0
            ind += 1
        end
    end
    for n in 1:T.N
        row[ind+1:ind+T.m[n]] = T.primal_inds[n,1]:T.primal_inds[n,2]
        data[ind+1:ind+T.m[n]] .= -1.0
        ind += T.m[n]
    end
    return Cint(0)
end


function compute_equilibrium(cost_tensors::Vector{Array{Float64, L}},
                             initialization=nothing;
                             ϵ = 0.0, # not used atm
                             silent = true,
                             convergence_tolerance = 1e-6) where L
    N = Cint(length(cost_tensors))
    m = Cint.(size(cost_tensors[1]))
    @assert all(m == size(tensor) for tensor ∈ cost_tensors)
    primal_inds = zeros(Cint,N,2)
    primal_inds[1,1] = 1
    primal_inds[1,2] = m[1]
    for n ∈ 2:N
        primal_inds[n,1] = primal_inds[n-1,2]+1
        primal_inds[n,2] = primal_inds[n,1]+m[n]-1
    end
    
    num_primals = sum(m)
    dual_inds = Vector{Cint}(num_primals+1:num_primals+N)
    
    tensor_indices = CartesianIndices(cost_tensors[1])
    nnz = 2*num_primals
    for n = 1:N
        nnz += m[n]*(num_primals-m[n])
    end

    wrapper! = Wrapper(cost_tensors, tensor_indices, primal_inds, dual_inds, N, m, num_primals)

    lb = [zeros(Cdouble, num_primals); -1e20 * ones(Cdouble, N)]
    ub = 1e20 * ones(Cdouble, num_primals + N)
    z = zeros(Cdouble, num_primals + N)
    if isnothing(initialization)
        for n ∈ 1:N
            start_ind = primal_inds[n,1]
            end_ind = primal_inds[n,2]
            z[start_ind:end_ind] .= 1.0 / m[n]
        end
    else
        z .= initialization
    end

    n = Cint(sum(m)+N)
    x = randn(Cdouble,n)
    f = zeros(Cdouble,n)
    cnnz = Cint(nnz)
    col = zeros(Cint, n)
    len = zeros(Cint, n)
    row = zeros(Cint, cnnz)
    data = zeros(Cdouble, cnnz)
    wrapper!(n,cnnz,x,col,len,row,data)

    status, vars, info = PATHSolver.solve_mcp(
        wrapper!,
        wrapper!,
        lb,
        ub,
        z;
        nnz,
        convergence_tolerance,
        silent)

    x = [vars[primal_inds[n,1]:primal_inds[n,2]] for n ∈ 1:N]
    V = [expected_cost(cost_tensors[n], vars, tensor_indices, primal_inds) for n ∈ 1:N]
    λ = vars[dual_inds]
    
    (; x, V, λ, wrapper!, nnz=Cint(nnz), tensor_indices, primal_inds)
end

"""
    Computes derivative tensor D
    D[n,i...,m] = ∂xₘ / ∂Tⁿᵢ, where i is a cartesian index (i.e. i:=[i₁,i₂,...,iₙ])
    In other words, D[n,i...,m] is the partial derivative of xₘ with respect to 
    the i-th element of player n's cost tensor.
"""
function compute_derivatives!(D, sol; bound_tolerance = 1e-6)
    primals = vcat(sol.x...)
    vars = [primals; sol.λ]
    n = Cint(length(vars))
    N = length(sol.λ)
    d = [length(xi) for xi ∈ sol.x]
    starts = cumsum([0;d[1:end-1]])
    num_primals = sum(d)

    lb = [zeros(num_primals); -Inf*ones(N)]
    unbound_indices = ( vars .> (lb .+ bound_tolerance) )
    unbound_primals = unbound_indices[1:end-N]
    nup = sum(unbound_primals)
    ubmap = zeros(Int,num_primals)
    idx = 1
    for i ∈ 1:num_primals
        if unbound_primals[i]
            ubmap[i] = idx
            idx += 1
        end
    end
    col = zeros(Cint, n)
    len = zeros(Cint, n)
    row = zeros(Cint, sol.nnz)
    data = zeros(Cdouble, sol.nnz)
    sol.wrapper!(n,sol.nnz,vars,col,len,row,data)
        
    colptr = zeros(Cint, n+1)
    colptr[1:n] .= col
    colptr[n+1] = col[n] + len[n]
    nJi = -inv(Matrix{Cdouble}((SparseMatrixCSC{Cdouble,Cint}(n,n,colptr,row,data))[unbound_indices, unbound_indices]))

    for ind ∈ sol.tensor_indices
        if prob_prod(primals, ind, sol.primal_inds) > (bound_tolerance)^N
            for n ∈ 1:N
                D[n,ind,unbound_primals] .= nJi[1:nup,ubmap[starts[n]+ind[n]]] * prob_prod(primals,ind,sol.primal_inds,n)
            end
        end
    end
    return 
end
