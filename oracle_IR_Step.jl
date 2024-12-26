using JuMP, Printf, Dates

function oracleIR_Step(inner_silent, opt, x_val, x_d_val, F, h, G, G_d, B2, B2_d, d, B1, B1_d, E, c2, c2_d, u_nom)
    tic = Dates.now()
    dim_y = size(c2)[1]
    dim_y_d = size(c2_d)[1]
    dim_u = size(F)[2]
    dim_cons2 = size(B2)[1]

    u_star = zeros(dim_u)
    u_star_best = zeros(dim_u)
    u_last = u_nom
    y_d_in_dict = Dict()
    y_d_in_dict[1] = zeros(Int64, dim_y_d)

    UB_in = Inf
    LB_in = -Inf
    # BigM = 1e8
    iter = 1
    tol = 5e-4
    tol_3rd = 1e-4
    ub_bid_cnt = 0
    ub_pool = []
  
    while true
        ### Get starting value of 3rd-level loop with the nominal point
        obj_last = Dict(it => Inf for it in 1:iter)
        for it in 1:iter
            MP_in_start = Model(opt)
            if inner_silent
                set_silent(MP_in_start)
            end
            set_attribute(MP_in_start, "Threads", 112)
            set_attribute(MP_in_start, "TimeLimit", 1800)
            @variable(MP_in_start, y_in_start[1:dim_y], lower_bound = 0)
            @objective(MP_in_start, Min, c2' * y_in_start + c2_d' * y_d_in_dict[it])
            @constraint(MP_in_start, B2 * y_in_start .>= d - B1 * x_val - B1_d * x_d_val - B2_d * y_d_in_dict[it] - E * u_nom)
            optimize!(MP_in_start)
            obj_last[it] = objective_value(MP_in_start)
        end

        obj_cil = Inf
        u_last = u_nom
        iter_3rd = 1
        phi_cur_best = 0
        u_star_best_3rd = zeros(dim_u)
        
        while true 
            ### Fix uncertainty to the one obtained in the last iteration first
            ### Solve "lower-level" problem to get dual variables
            pi_inner = Dict()
            phi_cur_dict = Dict()
            for it in 1:iter
                MP_in_1 = Model(opt)
                if inner_silent
                    set_silent(MP_in_1)
                end
                set_attribute(MP_in_1, "Threads", 112)
                set_attribute(MP_in_1, "TimeLimit", 1800)
                @variable(MP_in_1, theta)
                @variable(MP_in_1, u_in_1[1:dim_u])
                @variable(MP_in_1, y_in_1[1:dim_y], lower_bound = 0)
                @constraint(MP_in_1, B2 * y_in_1 .>= d - B1 * x_val - B1_d * x_d_val - B2_d * y_d_in_dict[it] - E * u_in_1)
                @constraint(MP_in_1, uncertainty_fix, u_in_1 .== u_last)
                @objective(MP_in_1, Min, c2' * y_in_1 + c2_d' * y_d_in_dict[it])
                optimize!(MP_in_1)
                
                pi_inner[it] = dual.(uncertainty_fix)
                phi_cur_dict[it] = objective_value(MP_in_1)
            end
            phi_cur = minimum(convert(Array{Float64}, collect(values(phi_cur_dict))))
            @printf("3rd level iteration %d: lower level Phi = %.2f\n", iter_3rd, phi_cur)
            if phi_cur > phi_cur_best
                u_star_best_3rd = u_last
                phi_cur_best = phi_cur
            end

            if abs(obj_cil - phi_cur) / abs(obj_cil) < tol_3rd || iter_3rd >= 50
                if UB_in > phi_cur_best
                    ub_bid_cnt += 1
                    push!(ub_pool, phi_cur_best)
                    if ub_bid_cnt >= 6
                        UB_in = min(UB_in, maximum(ub_pool))
                        ub_bid_cnt = 0
                        ub_pool = []
                    end
                end
                # UB_in = phi_cur_best
                u_star = u_star_best_3rd
                # u_star = u_last
                break
            else
                obj_cil = phi_cur
                # @printf("3rd level iteration %d: lower level Phi = %.2f\n", iter_3rd, phi_cur)
            end

            ### Solve "middle-level" problem to seek the next uncertainty point
            for it in 1:iter
                obj_last[it] = phi_cur_dict[it]
            end
            MP_in_2 = Model(opt)
            if inner_silent
                set_silent(MP_in_2)
            end
            set_attribute(MP_in_2, "Threads", 112)
            set_attribute(MP_in_2, "TimeLimit", 1800)
            @variable(MP_in_2, u_in_2[1:dim_u])
            @variable(MP_in_2, u_in_2_diff[1:dim_u], lower_bound = 0)
            @variable(MP_in_2, theta)
            @constraint(MP_in_2, F * u_in_2 .<= h + G * x_val + G_d * x_d_val)
            for it in 1:iter
                @constraint(MP_in_2, theta <= obj_last[it] + pi_inner[it]' * (u_in_2 - u_last))
            end
            @constraint(MP_in_2, u_in_2_diff .>= u_in_2 - u_last)
            @constraint(MP_in_2, u_in_2_diff .>= u_last - u_in_2)
            @constraint(MP_in_2, sum(u_in_2_diff) <= 0.05 * sum(u_last))
            # @constraint(MP_in_2, u_in_2 .== u_last)
            @objective(MP_in_2, Max, theta)
            optimize!(MP_in_2)
            # @printf("3rd level iteration %d: middle level Phi = %.2f\n", iter_3rd, objective_value(MP_in_2))
 
            u_last = value.(u_in_2)
            iter_3rd += 1
        end

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
        if ((UB_in - LB_in) / UB_in < tol) || (toc >= 3600)
            break
        end
        iter += 1
        y_d_in_dict[iter] = value.(y_d_sp)
    end
    return u_star_best, LB_in
end