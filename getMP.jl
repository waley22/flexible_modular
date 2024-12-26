using JuMP, BilevelJuMP

function getMP(mp_silent, opt, A, A_d, b, c1, c1_d, F, h, G, G_d, B2, B2_d, d, B1, B1_d, E, c2, c2_d, P_pi_hat)
    dim_x = size(c1)[1]
    dim_x_d = size(c1_d)[1]
    dim_y = size(c2)[1]
    dim_y_d = size(c2_d)[1]
    dim_u = size(F)[2]
    dim_uncset = size(F)[1]

    MP = Model(opt)
    @variable(MP, x[1:dim_x], lower_bound = 0)
    @variable(MP, x_d[1:dim_x_d], Int, lower_bound = 0, upper_bound = 3)
    @variable(MP, eta, lower_bound = 0)
    @constraint(MP, A * x + A_d * x_d .>= b)

    y_pi = Dict()
    y_d_pi = Dict()
    u_pi = Dict()
    lambda_pi = Dict()
    delta1_ou = Dict()
    delta2_ou = Dict()
    for i in keys(P_pi_hat)
        y_pi[i] = @variable(MP, [1:dim_y], lower_bound = 0)
        y_d_pi[i] = @variable(MP, [1:dim_y_d], Bin)
        u_pi[i] = @variable(MP, [1:dim_u], lower_bound = 0)
        lambda_pi[i] = @variable(MP, [1:dim_uncset], lower_bound = 0)
        delta1_ou[i] = @variable(MP, [1:dim_uncset], Bin)
        delta2_ou[i] = @variable(MP, [1:dim_u], Bin)

        @constraint(MP, eta >= c2' * y_pi[i] + c2_d' * y_d_pi[i])
        @constraint(MP, B2 * y_pi[i] + B2_d * y_d_pi[i] .>= d - B1 * x - B1_d * x_d - E * u_pi[i])
        @constraint(MP, F * u_pi[i] .<= h + G * x + G_d * x_d)
        @constraint(MP, F' * lambda_pi[i] .>= -E' * P_pi_hat[i])
        ### Strong duality form (MIQCP, still kind of unstable but better than the complementary slackness form)
        @constraint(MP, -(E * u_pi[i])' * P_pi_hat[i] .>= (h + G * x)' * lambda_pi[i])

        ### Complementary slackness in product mode (MIQCP, sometimes unstable due to no upper bound for lambda_pi)
        #= @constraint(MP, lambda_pi[i] .* (h + G * x + G_d * x_d - F * u_pi[i]) .== 0)
        @constraint(MP, u_pi[i] .* (F' * lambda_pi[i] + E' * P_pi_hat[i]) .== 0) =#

        ### Complementary slackness in big-M form (MILP, doesn't work very well here because bound of lambda_pi is ill-defined)
        #= @constraint(MP, [k in 1:dim_uncset], lambda_pi[i][k] <= BigM * delta1_ou[i][k])
        @constraint(MP, [k in 1:dim_uncset], (h + G * x + G_d * x_d - F * u_pi[i])[k] <= BigM * (1 - delta1_ou[i][k]))
        @constraint(MP, [l in 1:dim_u], u_pi[i][l] <= BigM * delta2_ou[i][l])
        @constraint(MP, [l in 1:dim_u], (F' * lambda_pi[i] + E' * P_pi_hat[i])[l] <= BigM * (1 - delta2_ou[i][l])) =#

        ### Complementary slackness in indicator form (MILP, too slow)
        #= @constraint(MP, [k in 1:dim_uncset], delta1_ou[i][k] => {lambda_pi[i][k] == 0})
        @constraint(MP, [k in 1:dim_uncset], !delta1_ou[i][k] => {(h + G * x + G_d * x_d - F * u_pi[i])[k] == 0})
        @constraint(MP, [l in 1:dim_u], delta2_ou[i][l] => {u_pi[i][l] == 0})
        @constraint(MP, [l in 1:dim_u], !delta2_ou[i][l] => {(F' * lambda_pi[i] + E' * P_pi_hat[i])[l] == 0}) =#
    end
    @objective(MP, Min, c1' * x + c1_d' * x_d + eta)

    if mp_silent
        set_silent(MP)
    end
    set_attribute(MP, "Threads", 112)
    set_attribute(MP, "MIPGap", 1e-3)
    set_attribute(MP, "TimeLimit", 180)

    return x, x_d, MP
end