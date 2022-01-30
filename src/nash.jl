using PATHSolver

function (x::Vector{Float64})(ind, primal_inds, n...)
    prod(x[primal_inds[i][1]+ind[i]-1] for i ∈ 1:length(primal_inds) if i ∉ n)
end

function (T::Array{Float64,N})(x, indices, primal_inds) where N
    val = sum(T[ind]*x(ind) for ind ∈ indices)
end

function grad!(f, T, x, n, indices, primal_inds)
    f[primal_inds[n][1]:primal_inds[n][2]] .= 0.0
    for ind ∈ indices
        for i ∈ 1:primal_inds[n][2]+1-primal_inds[n][1]
            if ind[n] == i
                f[primal_inds[n][1]+i-1] += T[ind]*x(ind,primal_inds,n)
            end
        end
    end
    Cint(0)
end

function jac(T, x, n1, n2, indices, primal_inds)
    l1 = primal_inds[n1][2]-primal_inds[n1][1]+1
    l2 = primal_inds[n2][2]-primal_inds[n2][1]+1
    (n1 == n2) && return zeros(l1,l2)
    J = [sum(T[ind]*x(ind,primal_inds,n1,n2) for ind in indices if ind[n1]==i && ind[n2] ==j) for i ∈ 1:l1, j ∈ 1:l2]
    Cint(0)
end

function jac(J,loc,T, x, n1, n2, i, j, indices, primal_inds)
    l1 = primal_inds[n1][2]-primal_inds[n1][1]+1
    l2 = primal_inds[n2][2]-primal_inds[n2][1]+1
    (n1 == n2) && return zeros(l1,l2)
    for ind ∈ indices
        if ind[n1] == i && ind[n2] == j
            J[loc] += T[ind]*x(ind,primal_inds,n1,n2) 
        end
    end
    Cint(0)
end
        

struct Wrapper{L} <: Function
    tensors::Vector{Array{Float64,L}}
    tensor_indices::CartesianIndices{L, NTuple{L, Base.OneTo{Int64}}}
    primal_inds::Vector{Vector{Cint}}
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
        f[ind+1:ind+T.m[n]] .-= x[T.dual_inds[n]]
        ind += T.m[n]
    end
    for n ∈ 1:T.N
        f[ind+n] = sum(x[T.primal_inds[n][1]:T.primal_inds[n][2]]) - 1.0
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
                    row[ind+1] = Cint(T.primal_inds[n_row][1]+p-1)
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
        row[ind+1:ind+T.m[n]] = T.primal_inds[n][1]:T.primal_inds[n][2]
        data[ind+1:ind+T.m[n]] .= -1.0
        ind += T.m[n]
    end
    return Cint(0)
end


function compute_equilibrium(cost_tensors::Vector{Array{Float64, L}};  silent=true) where L
    N = Cint(length(cost_tensors))
    m = Cint.(size(cost_tensors[1]))
    @assert all(m == size(tensor) for tensor ∈ cost_tensors)

    cs = cumsum(m)
    ls = [1; [cs[i]+1 for i ∈ 1:N-1]]
    primal_inds = [Cint.([lo,hi]) for (lo,hi) ∈ zip(ls,cs)]
    num_primals = Cint(cs[N])
    dual_inds = Cint.(num_primals+1:num_primals+N)
    
    tensor_indices = CartesianIndices(cost_tensors[1])
    nnz = 2*num_primals
    for n = 1:N
        nnz += m[n]*(num_primals-m[n])
    end

    wrapper = Wrapper(cost_tensors, tensor_indices, primal_inds, dual_inds, N, m, num_primals)

    lb = [zeros(Cdouble, num_primals); -1e20 * ones(Cdouble, N)]
    ub = 1e20 * ones(Cdouble, num_primals + N)
    z = zeros(Cdouble, num_primals + N)
    for (n, (start_ind, end_ind)) ∈ enumerate(primal_inds)
        z[start_ind:end_ind] .= 1.0 / m[n]
    end

    PATHSolver.c_api_License_SetString("2830898829&Courtesy&&&USR&45321&5_1_2021&1000&PATH&GEN&31_12_2025&0_0_0&6000&0_0")
    status, vars, info = PATHSolver.solve_mcp(
        wrapper,
        wrapper,
        lb,
        ub,
        z;
        nnz,
        silent)
    p_out = [vars[start_ind:end_ind] for (start_ind,end_ind) ∈ primal_inds]
end
