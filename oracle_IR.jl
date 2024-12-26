using JuMP, BilevelJuMP, Printf, Dates

function oracleIR(inner_silent, opt, x_val, x_d_val, F, h, G, G_d, B2, B2_d, d, B1, B1_d, E, c2, c2_d)
    tic = Dates.now()
    dim_y = size(c2)[1]
    dim_y_d = size(c2_d)[1]
    dim_u = size(F)[2]
    dim_cons2 = size(B2)[1]

    u_star = zeros(dim_u)
    u_star_best = zeros(dim_u)
    y_in = Dict()
    y_d_in_dict = Dict()
    y_d_in_dict[1] = zeros(Int64, dim_y_d)
    # pi_in = Dict()
    # delta1_in = Dict()
    # delta2_in = Dict()
    theta_in = Dict()

    UB_in = Inf
    LB_in = -Inf
    # BigM = 1e8
    iter = 1
    tol = 5e-4
  
    while true
        # MP_in = Model(opt)
        # if inner_silent
        #     set_silent(MP_in)
        # end
        # set_attribute(MP_in, "Threads", 112)
        # set_attribute(MP_in, "TimeLimit", 300)
        # @variable(MP_in, theta)
        # @variable(MP_in, u_in[1:dim_u])
        # @constraint(MP_in, F * u_in .<= h + G * x_val + G_d * x_d_val)
        # for i in 1:iter
        #     y_in[i] = @variable(MP_in, [1:dim_y], lower_bound = 0)
        #     pi_in[i] = @variable(MP_in, [1:dim_cons2], lower_bound = 0)
        #     delta1_in[i] = @variable(MP_in, [1:dim_y], Bin)
        #     delta2_in[i] = @variable(MP_in, [1:dim_cons2], Bin)

        #     @constraint(MP_in, theta <= c2' * y_in[i] + c2_d' * y_d_in_dict[i])
        #     @constraint(MP_in, B2 * y_in[i] .>= d - B1 * x_val - B1_d * x_d_val - B2_d * y_d_in_dict[i] - E * u_in)
        #     @constraint(MP_in, B2' * pi_in[i] .<= c2)
        #     #= @constraint(MP_in, y_in[i] .* (c2 - B2' * pi_in[i]) .== 0)
        #     @constraint(MP_in, pi_in[i] .* (B2 * y_in[i] + B2_d * y_d_in_dict[i] + E * u_in + B1 * x_val + B1_d * x_d_val - d) .== 0) =#

        #     ### Indicator formulation for KKT condition
        #     @constraint(MP_in, [k = 1:dim_y], delta1_in[i][k] => {y_in[i][k] == 0})
        #     @constraint(MP_in, [k = 1:dim_y], !delta1_in[i][k] => {(c2 - B2' * pi_in[i])[k] == 0})
        #     @constraint(MP_in, [l = 1:dim_cons2], delta2_in[i][l] => {pi_in[i][l] == 0})
        #     @constraint(MP_in, [l = 1:dim_cons2], !delta2_in[i][l] => {(B2 * y_in[i] + B2_d * y_d_in_dict[i] + E * u_in + B1 * x_val + B1_d * x_d_val - d)[l] == 0})

        #     ### Big-M formulation for KKT condition
        #     #= @constraint(MP_in, y_in[i] .<= BigM .* delta1_in[i])
        #     @constraint(MP_in, c2 - B2' * pi_in[i] .<= BigM .* (1 .- delta1_in[i]))
        #     @constraint(MP_in, pi_in[i] .<= BigM .* delta2_in[i])
        #     @constraint(MP_in, B2 * y_in[i] + B2_d * y_d_in_dict[i] + E * u_in + B1 * x_val + B1_d * x_d_val - d .<= BigM .* (1 .- delta2_in[i])) =#

        #     ### Original strong duality formulation
        #     #= @constraint(MP_in, theta <= c2_d' * y_d_in_dict[i] + (d - E * u_in - B1 * x_val - B1_d * x_d_val - B2_d * y_d_in_dict[i])' * pi_in[i])
        #     @constraint(MP_in, B2' * pi_in[i] .<= c2) =#
        # end
        # @objective(MP_in, Max, theta)

        MP_in = BilevelModel(opt, mode = BilevelJuMP.IndicatorMode())
        if inner_silent
            set_silent(MP_in)
        end
        set_attribute(MP_in, "Threads", 112)
        set_attribute(MP_in, "TimeLimit", 1800)
        @variable(Upper(MP_in), theta)
        @variable(Upper(MP_in), u_in[1:dim_u])
        @constraint(Upper(MP_in), F * u_in .<= h + G * x_val + G_d * x_d_val)
        for i in 1:iter
            y_in[i] = @variable(Lower(MP_in), [1:dim_y], lower_bound = 0)
            theta_in[i] = @variable(Lower(MP_in))
            # pi_in[i] = @variable(Lower(MP_in), [1:dim_cons2], lower_bound = 0)

            @constraint(Upper(MP_in), theta <= theta_in[i])
            @constraint(Lower(MP_in), theta_in[i] >= c2' * y_in[i] + c2_d' * y_d_in_dict[i])
            @constraint(Lower(MP_in), B2 * y_in[i] .>= d - B1 * x_val - B1_d * x_d_val - B2_d * y_d_in_dict[i] - E * u_in)
            # @constraint(Lower(MP_in), B2' * pi_in[i] .<= c2)
            #= @constraint(MP_in, y_in[i] .* (c2 - B2' * pi_in[i]) .== 0)
            @constraint(MP_in, pi_in[i] .* (B2 * y_in[i] + B2_d * y_d_in_dict[i] + E * u_in + B1 * x_val + B1_d * x_d_val - d) .== 0) =#
        end
        @objective(Upper(MP_in), Max, theta)
        @objective(Lower(MP_in), Min, sum(theta_in[i] for i in 1:iter))
        optimize!(MP_in)
        # if !is_solved_and_feasible(MP_in)
        #     return u_star, UB_in
        # end
        try
            UB_in = min(UB_in, objective_value(MP_in))
            u_star = value.(u_in)
        catch e
            println("Infeasible model")
            throw(e)
        end
        #println(u_star)
        #println(value.(y_in[iter]))

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
    return u_star_best, UB_in
end