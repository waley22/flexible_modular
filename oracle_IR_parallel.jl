using JuMP, BilevelJuMP, Printf, Dates, Base.Threads
import Distributed

function oracleIR_parallel(inner_silent, opt, x_val, x_d_val, F, h, G, G_d, B2, B2_d, d, B1, B1_d, E, c2, c2_d, size_dict)
    tic = Dates.now()
    dim_y = size(c2)[1]
    dim_y_d = size(c2_d)[1]
    dim_uncset = size(F)[1]
    dim_u = size(F)[2]
    dim_cons2 = size(B2)[1]

    u_star = zeros(dim_u)
    u_star_best = zeros(dim_u)

    UB_in = Inf
    LB_in = -Inf
    # BigM = 1e8
    iter = 1
    tol = 5e-4

    T_SIZE = size_dict["T_SIZE"]
  
    dim_y_t = div(dim_y, T_SIZE)
    dim_y_d_t = div(dim_y_d, T_SIZE)
    dim_cons2_t = div(dim_cons2, T_SIZE)
    dim_uncset_t = div(dim_uncset, T_SIZE)
    dim_u_t = div(dim_u, T_SIZE)

    y_in = Dict()
    y_d_in_dict = Dict()
    theta_in = Dict()
    for t in 1:T_SIZE
        y_in[t] = Dict()
        y_d_in_dict[t] = Dict()
        y_d_in_dict[t][1] = zeros(Int64, dim_y_d_t)
        theta_in[t] = Dict()
    end
    
    while true
        UB_in_t = 0
        my_lock = Threads.ReentrantLock()
        Threads.@threads for t in 1:T_SIZE
            F_t = F[(t - 1) * dim_uncset_t + 1:t * dim_uncset_t, (t - 1) * dim_u_t + 1:t * dim_u_t]
            h_t = h[(t - 1) * dim_uncset_t + 1:t * dim_uncset_t]
            G_t = G[(t - 1) * dim_uncset_t + 1:t * dim_uncset_t, :]
            G_d_t = G_d[(t - 1) * dim_uncset_t + 1:t * dim_uncset_t, :]
            B2_t = B2[(t - 1) * dim_cons2_t + 1:t * dim_cons2_t, (t - 1) * dim_y_t + 1:t * dim_y_t]
            B2_d_t = B2_d[(t - 1) * dim_cons2_t + 1:t * dim_cons2_t, (t - 1) * dim_y_d_t + 1:t * dim_y_d_t]
            d_t = d[(t - 1) * dim_cons2_t + 1:t * dim_cons2_t]
            B1_t = B1[(t - 1) * dim_cons2_t + 1:t * dim_cons2_t, :]
            B1_d_t = B1_d[(t - 1) * dim_cons2_t + 1:t * dim_cons2_t, :]
            E_t = E[(t - 1) * dim_cons2_t + 1:t * dim_cons2_t, (t - 1) * dim_u_t + 1:t * dim_u_t]
            c2_t = c2[(t - 1) * dim_y_t + 1:t * dim_y_t]
            c2_d_t = c2_d[(t - 1) * dim_y_d_t + 1:t * dim_y_d_t]

            MP_in_t = BilevelModel(opt, mode = BilevelJuMP.IndicatorMode())
            if inner_silent
                set_silent(MP_in_t)
            end
            set_attribute(MP_in_t, "Threads", 112)
            set_attribute(MP_in_t, "TimeLimit", 3600)
            @variable(Upper(MP_in_t), theta)
            @variable(Upper(MP_in_t), u_in_t[1:dim_u_t])
            @constraint(Upper(MP_in_t), F_t * u_in_t .<= h_t + G_t * x_val + G_d_t * x_d_val)
            for i in 1:iter
                y_in[t][i] = @variable(Lower(MP_in_t), [1:dim_y_t], lower_bound = 0)
                theta_in[t][i] = @variable(Lower(MP_in_t))

                @constraint(Upper(MP_in_t), theta <= theta_in[t][i])
                @constraint(Lower(MP_in_t), theta_in[t][i] >= c2_t' * y_in[t][i] + c2_d_t' * y_d_in_dict[t][i])
                @constraint(Lower(MP_in_t), B2_t * y_in[t][i] .>= d_t - B1_t * x_val - B1_d_t * x_d_val - B2_d_t * y_d_in_dict[t][i] - E_t * u_in_t)
            end
            @objective(Upper(MP_in_t), Max, theta)
            @objective(Lower(MP_in_t), Min, sum(theta_in[t][i] for i in 1:iter))
            optimize!(MP_in_t)

            Threads.lock(my_lock) do
                try
                    UB_in_t += objective_value(MP_in_t)
                    u_star[(t - 1) * dim_u_t + 1:t * dim_u_t] = value.(u_in_t)
                catch e
                    println("Infeasible model")
                    throw(e)
                end
            end
        end
        UB_in = min(UB_in, UB_in_t)

        SP_in = Model(opt)
        if inner_silent
            set_silent(SP_in)
        end
        set_attribute(SP_in, "Threads", 112)
        set_attribute(SP_in, "TimeLimit", 200)
        @variable(SP_in, y_sp[1:dim_y], lower_bound = 0)
        @variable(SP_in, y_d_sp[1:dim_y_d], Bin)
        @constraint(SP_in, B2 * y_sp + B2_d * y_d_sp .>= d - B1 * x_val - B1_d * x_d_val - E * u_star)
        @objective(SP_in, Min, c2' * y_sp + c2_d' * y_d_sp)
        optimize!(SP_in)
        if objective_value(SP_in) > LB_in # The same problem as in the 1st level iteration: return the best u instead of the u in the current iteration
            u_star_best = u_star
        end
        LB_in = max(LB_in, objective_value(SP_in))
        @printf("UB_in: %.3f, LB_in: %.3f\n", UB_in, LB_in)

        toc = (Dates.now() - tic) / Dates.Millisecond(1) / 1000.0
        if ((UB_in - LB_in) / UB_in < tol) || (toc >= 7200)
            break
        end
        iter += 1

        for t in 1:T_SIZE
            y_d_in_dict[t][iter] = value.(y_d_sp)[(t - 1) * dim_y_d_t + 1:t * dim_y_d_t]
        end
    end
    return u_star_best, UB_in
end