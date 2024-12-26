using JuMP, Gurobi, Printf, ArgParse, Random, Distributions, GAMS, BilevelJuMP
import CSV, Tables, DataFrames, LinearAlgebra, Dates, Plots, XLSX
include("getMP.jl")
include("oracle_IR_Step.jl")
include("utils.jl")

cd(dirname(Base.source_path()))
GRB_ENV = Gurobi.Env()

# Run configs
write_to_csv = true
mode = 2 # 1 for new data, other for reading from CSV file
debug = false # The latter only shows the line "Set parameter Username" once, but won't work in debug mode
if debug
    opt = Gurobi.Optimizer
else
    opt = () -> Gurobi.Optimizer(GRB_ENV)
end
mp_silent = true
sp_silent = true

print_status("Write to CSV: ", write_to_csv)
print_status("Debug mode: ", debug)
print_status("MP silent: ", mp_silent)
print_status("SP silent: ", sp_silent)
println()

for k in 1:1
    for k2 in 1:1
        setName = "Linear_real"
        dataset = "$(setName)_$(k)_$(k2)"

        ### Reference: Two-Stage Robust Optimization with Decision Dependent Uncertainty, Bo Zeng and Wei Wang (2022)
        ### Dimension definition
        sets_df = CSV.read("./data/$(setName)/$(dataset)/input_size.csv", DataFrames.DataFrame)
        I_SIZE = sets_df[2, 1]      # Set of candidate and facility locations
        J_SIZE = sets_df[1, 1]      # Set of customers
        L_SIZE = sets_df[3, 1]      # Set of facility configurations
        M_SIZE = sets_df[4, 1]      # Set of types of modules
        T_SIZE = sets_df[5, 1]      # Set of time periods
        BIN_SIZE = 3                # Set of binary disaggregation for v variables

        b_param = zeros(I_SIZE, L_SIZE)
        c_param = zeros(I_SIZE, J_SIZE)
        d_param = zeros(J_SIZE, T_SIZE)
        g_param = zeros(I_SIZE, M_SIZE)
        n_param = zeros(L_SIZE, M_SIZE)
        u_param = zeros(I_SIZE, L_SIZE)
        jloc_param = zeros(J_SIZE, 2)
        iloc_param = zeros(I_SIZE, 2)
        jnearest_param = zeros(J_SIZE)

        FI_coeff_param = 100
        penalty_param = 20
        nom_weight = 0.5
        BigM = 1e6

        recover_param!(b_param, "./data/$(setName)/$dataset/input_b.csv")
        recover_param!(c_param, "./data/$(setName)/$dataset/input_c.csv")
        recover_param!(d_param, "./data/$(setName)/$dataset/input_d.csv")
        recover_param!(g_param, "./data/$(setName)/$dataset/input_g.csv")
        recover_param!(n_param, "./data/$(setName)/$dataset/input_n.csv")
        recover_param!(u_param, "./data/$(setName)/$dataset/input_u.csv")
        recover_param!(jloc_param, "./data/$(setName)/$dataset/input_jloc.csv")
        recover_param!(iloc_param, "./data/$(setName)/$dataset/input_iloc.csv")

        c_param = c_param ./ 1e3
        g_param = g_param ./ 1e3
        u_param = u_param
        # println(d_param)

        ### Model copied from Section 5.1-5.3 of Zeng and Wang (2022)
        UI = 10
        println("UI = ", UI)
        uncertain_j_list = sortperm(dropdims(sum(d_param; dims = 2); dims = 2))[end - (UI - 1):end]
        if occursin("m=4", setName)
            v_max_param = [3, 3, 1, 1]
            large_modules = [3, 4]
            large_small_counterpart = Dict(3 => 1, 4 => 2)
        elseif occursin("m=1", setName)
            v_max_param = [3]
            large_modules = []
            large_small_counterpart = Dict()
        else
            v_max_param = [3, 1]
            large_modules = [2]
            large_small_counterpart = Dict(2 => 1)
        end

        dim_x = I_SIZE * J_SIZE * L_SIZE * T_SIZE + J_SIZE * T_SIZE
        dim_x_d = I_SIZE * M_SIZE + I_SIZE * L_SIZE * T_SIZE
        dim_y = I_SIZE * J_SIZE * L_SIZE * T_SIZE + J_SIZE * T_SIZE # I_SIZE * J_SIZE * L_SIZE * T_SIZE: x; J * T: slack variables for demand; 
        dim_y_d = I_SIZE * L_SIZE * T_SIZE # I_SIZE * L_SIZE * T_SIZE: y;
        dim_u = UI * T_SIZE + UI * T_SIZE + UI * (T_SIZE - 1) # dimension of uncertainty
        # 1 - UI * T_SIZE: original u; UI * T_SIZE + 1 - 2 * UI * T_SIZE: deviation from nominal point; 2 * UI * T_SIZE + 1 - end: difference across time periods
        dim_cons1 = (I_SIZE * M_SIZE
        + I_SIZE * size(large_modules)[1]
        + J_SIZE * T_SIZE 
        + 2 * I_SIZE * L_SIZE * T_SIZE 
        + I_SIZE * T_SIZE
        + I_SIZE * M_SIZE * T_SIZE
        + 2 * I_SIZE * L_SIZE * T_SIZE
        + I_SIZE * T_SIZE * size(large_modules)[1]) # A * x + A_d * x_d >= b
        dim_uncset = T_SIZE * 2 + 2 * UI * T_SIZE + (2 * UI + 1) * T_SIZE + (2 * UI + 1) * (T_SIZE - 1) # F * u <= h + G * x + G_d * x_d
        dim_cons2 = (J_SIZE * T_SIZE 
        + 2 * I_SIZE * L_SIZE * T_SIZE 
        + I_SIZE * T_SIZE
        + I_SIZE * M_SIZE * T_SIZE
        + I_SIZE * T_SIZE * size(large_modules)[1]
        + 2 * I_SIZE * L_SIZE * T_SIZE)
        # B2 * y + B2_d * y_d >= d - B1 * x - B1_d * x_d - E * u

        ### Matrix definition (suffix _d refers to the discrete variables and their respective coefficients)
        A = zeros(dim_cons1, dim_x)
        A_d = zeros(dim_cons1, dim_x_d)
        b = zeros(dim_cons1)
        c1 = zeros(dim_x)
        c1_d = zeros(dim_x_d)
        F = zeros(dim_uncset, dim_u)
        h = zeros(dim_uncset)
        G = zeros(dim_uncset, dim_x)
        G_d = zeros(dim_uncset, dim_x_d)
        B2 = zeros(dim_cons2, dim_y)
        B2_d = zeros(dim_cons2, dim_y_d)
        d = zeros(dim_cons2)
        B1 = zeros(dim_cons2, dim_x)
        B1_d = zeros(dim_cons2, dim_x_d)
        E = zeros(dim_cons2, dim_u)
        c2 = zeros(dim_y)
        c2_d = zeros(dim_y_d)

        for mobile_mode in ["Mob", "Fix"]
            ### Fill in matrices according to model
            fill!(A, 0)
            fill!(A_d, 0)
            fill!(b, 0)
            fill!(c1, 0)
            fill!(c1_d, 0)
            fill!(F, 0)
            fill!(h, 0)
            fill!(G, 0)
            fill!(G_d, 0)
            fill!(B2, 0)
            fill!(B2_d, 0)
            fill!(d, 0)
            fill!(B1, 0)
            fill!(B1_d, 0)
            fill!(E, 0)
            fill!(c2, 0)
            fill!(c2_d, 0)
            ### First stage problem (includes a copy of the subproblem for the nominal case)

            ## First stage objective
            for i in 1:I_SIZE, j in 1:J_SIZE, l in 1:L_SIZE, t in 1:T_SIZE
                c1[get_index([i, j, l, t], [I_SIZE, J_SIZE, L_SIZE, T_SIZE])] = c_param[i, j] * nom_weight
            end
            for j in 1:J_SIZE, t in 1:T_SIZE
                c1[I_SIZE * J_SIZE * L_SIZE * T_SIZE + get_index([j, t], [J_SIZE, T_SIZE])] = penalty_param * nom_weight
            end
            for i in 1:I_SIZE, m in 1:M_SIZE
                c1_d[get_index([i, m], [I_SIZE, M_SIZE])] = g_param[i, m]
            end

            ## module_design_num_cons: I_SIZE * M_SIZE
            n_start_idx = 0
            for i in 1:I_SIZE, m in 1:M_SIZE
                idx = get_index([i, m], [I_SIZE, M_SIZE])
                A_d[n_start_idx + idx, idx] = -1
                b[n_start_idx + idx] = -v_max_param[m]
            end
            n_start_idx += I_SIZE * M_SIZE
            ## module_design_num_cons: I_SIZE
            for i in 1:I_SIZE, m_idx in 1:size(large_modules)[1]
                idx = get_index([i, m_idx], [I_SIZE, size(large_modules)[1]])
                A_d[n_start_idx + idx, get_index([i, large_modules[m_idx]], [I_SIZE, M_SIZE])] = -3
                A_d[n_start_idx + idx, get_index([i, large_small_counterpart[large_modules[m_idx]]], [I_SIZE, M_SIZE])] = -1
                b[n_start_idx + idx] = -3
            end
            n_start_idx += I_SIZE * size(large_modules)[1]
            ## demand_cons_nominal: J_SIZE * T_SIZE
            for j in 1:J_SIZE, t in 1:T_SIZE
                idx = get_index([j, t], [J_SIZE, T_SIZE])
                for i in 1:I_SIZE, l in 1:L_SIZE
                    A[n_start_idx + idx, get_index([i, j, l, t], [I_SIZE, J_SIZE, L_SIZE, T_SIZE])] = 1
                end
                A[n_start_idx + idx, I_SIZE * J_SIZE * L_SIZE * T_SIZE + idx] = 1
                b[n_start_idx + idx] = d_param[j, t]
            end
            n_start_idx += J_SIZE * T_SIZE
            ## production_cons1: I_SIZE * L_SIZE * T_SIZE
            for i in 1:I_SIZE, l in 1:L_SIZE, t in 1:T_SIZE
                idx = get_index([i, l, t], [I_SIZE, L_SIZE, T_SIZE])
                A_d[n_start_idx + idx, I_SIZE * M_SIZE + get_index([i, l, t], [I_SIZE, L_SIZE, T_SIZE])] = u_param[i, l]
                for j in 1:J_SIZE
                    A[n_start_idx + idx, get_index([i, j, l, t], [I_SIZE, J_SIZE, L_SIZE, T_SIZE])] = -1
                end
            end
            n_start_idx += I_SIZE * L_SIZE * T_SIZE
            ## production_cons2: I_SIZE * L_SIZE * T_SIZE
            for i in 1:I_SIZE, l in 1:L_SIZE, t in 1:T_SIZE
                idx = get_index([i, l, t], [I_SIZE, L_SIZE, T_SIZE])
                A_d[n_start_idx + idx, I_SIZE * M_SIZE + get_index([i, l, t], [I_SIZE, L_SIZE, T_SIZE])] = -u_param[i, l] * b_param[i, l]
                for j in 1:J_SIZE
                    A[n_start_idx + idx, get_index([i, j, l, t], [I_SIZE, J_SIZE, L_SIZE, T_SIZE])] = 1
                end
            end
            n_start_idx += I_SIZE * L_SIZE * T_SIZE
            ## config_cons: I_SIZE * T_SIZE
            for i in 1:I_SIZE, t in 1:T_SIZE
                idx = get_index([i, t], [I_SIZE, T_SIZE])
                for l in 1:L_SIZE
                    A_d[n_start_idx + idx, I_SIZE * M_SIZE + get_index([i, l, t], [I_SIZE, L_SIZE, T_SIZE])] = -1
                end
                b[n_start_idx + idx] = -1
            end
            n_start_idx += I_SIZE * T_SIZE
            if mobile_mode == "Mob"
                ## total_module_cons: M_SIZE * T_SIZE
                idx = 1
                for m in 1:M_SIZE, t in 1:T_SIZE
                    if m in large_modules
                        for i in 1:I_SIZE
                            for l in 1:L_SIZE
                                A_d[n_start_idx + idx, I_SIZE * M_SIZE + get_index([i, l, t], [I_SIZE, L_SIZE, T_SIZE])] = -n_param[l, m]
                                A_d[n_start_idx + idx, get_index([i, m], [I_SIZE, M_SIZE])] = 1
                            end
                            idx += 1
                        end
                    else
                        for i in 1:I_SIZE, l in 1:L_SIZE
                            A_d[n_start_idx + idx, I_SIZE * M_SIZE + get_index([i, l, t], [I_SIZE, L_SIZE, T_SIZE])] = -n_param[l, m]
                            A_d[n_start_idx + idx, get_index([i, m], [I_SIZE, M_SIZE])] = 1
                        end
                        idx += 1
                    end
                end
            else
                for i in 1:I_SIZE, m in 1:M_SIZE, t in 1:T_SIZE
                    idx = get_index([i, m, t], [I_SIZE, M_SIZE, T_SIZE])
                    for l in 1:L_SIZE
                        A_d[n_start_idx + idx, I_SIZE * M_SIZE + get_index([i, l, t], [I_SIZE, L_SIZE, T_SIZE])] = -n_param[l, m]
                    end
                    A_d[n_start_idx + idx, get_index([i, m], [I_SIZE, M_SIZE])] = 1
                end
            end
            n_start_idx += I_SIZE * M_SIZE * T_SIZE
            # y_bound_mmin_cons: I_SIZE * L_SIZE * T_SIZE
            for i in 1:I_SIZE, l in 1:L_SIZE, t in 1:T_SIZE
                idx = get_index([i, l, t], [I_SIZE, L_SIZE, T_SIZE])
                A_d[n_start_idx + idx, I_SIZE * M_SIZE + idx] = 1
            end
            n_start_idx += I_SIZE * L_SIZE * T_SIZE
            # y_bound_max_cons: I_SIZE * L_SIZE * T_SIZE
            for i in 1:I_SIZE, l in 1:L_SIZE, t in 1:T_SIZE
                idx = get_index([i, l, t], [I_SIZE, L_SIZE, T_SIZE])
                A_d[n_start_idx + idx, I_SIZE * M_SIZE + idx] = -1
                b[n_start_idx + idx] = -1
            end
            n_start_idx += I_SIZE * L_SIZE * T_SIZE
            # large_module_static_cons1: I_SIZE * (large_module_SIZE) * T_SIZE
            for i in 1:I_SIZE, t in 1:T_SIZE, m_idx in 1:size(large_modules)[1]
                idx = get_index([i, m_idx, t], [I_SIZE, size(large_modules)[1], T_SIZE])
                for l in 1:L_SIZE
                    A_d[n_start_idx + idx, I_SIZE * M_SIZE + get_index([i, l, t], [I_SIZE, L_SIZE, T_SIZE])] = -n_param[l, large_small_counterpart[large_modules[m_idx]]]
                end
                A_d[n_start_idx + idx, get_index([i, large_modules[m_idx]], [I_SIZE, M_SIZE])] = -3
                b[n_start_idx + idx] = -3
            end
            n_start_idx += I_SIZE * T_SIZE * size(large_modules)[1]
            # # large_module_static_cons2: I_SIZE * (L-without-large-module_SIZE) * T_SIZE
            # for i in 1:I_SIZE, l in 1:3, t in 1:T_SIZE
            #     idx = get_index([i, l, t], [I_SIZE, 3, T_SIZE])
            #     A_d[n_start_idx + idx, I_SIZE * M_SIZE + get_index([i, l, t], [I_SIZE, L_SIZE, T_SIZE])] = -1
            #     A_d[n_start_idx + idx, get_index([i, 2], [I_SIZE, M_SIZE])] = -1
            #     b[n_start_idx + idx] = -1
            # end
            # n_start_idx += I_SIZE * 3 * T_SIZE

            ### Second stage problem
            ## Second stage objective
            for i in 1:I_SIZE, j in 1:J_SIZE, l in 1:L_SIZE, t in 1:T_SIZE
                c2[get_index([i, j, l, t], [I_SIZE, J_SIZE, L_SIZE, T_SIZE])] = c_param[i, j] * (1 - nom_weight)
            end
            for j in 1:J_SIZE, t in 1:T_SIZE
                c2[I_SIZE * J_SIZE * L_SIZE * T_SIZE + get_index([j, t], [J_SIZE, T_SIZE])] = penalty_param * (1 - nom_weight)
            end

            ## demand_cons: J_SIZE * T_SIZE
            n_start_idx = 0
            for j in 1:J_SIZE, t in 1:T_SIZE
                idx = get_index([j, t], [J_SIZE, T_SIZE])
                for i in 1:I_SIZE, l in 1:L_SIZE
                    B2[n_start_idx + idx, get_index([i, j, l, t], [I_SIZE, J_SIZE, L_SIZE, T_SIZE])] = 1
                end
                B2[n_start_idx + idx, I_SIZE * J_SIZE * L_SIZE * T_SIZE + idx] = 1
                
                if j in uncertain_j_list
                    uj_idx = findall(uncertain_j_list .== j)[1]
                    E[n_start_idx + idx, get_index([uj_idx, t], [UI, T_SIZE])] = -1
                else
                    d[n_start_idx + idx] = d_param[j, t]
                end
            end
            n_start_idx += J_SIZE * T_SIZE
            ## production_cons1: I_SIZE * L_SIZE * T_SIZE
            for i in 1:I_SIZE, l in 1:L_SIZE, t in 1:T_SIZE
                idx = get_index([i, l, t], [I_SIZE, L_SIZE, T_SIZE])
                B2_d[n_start_idx + idx, get_index([i, l, t], [I_SIZE, L_SIZE, T_SIZE])] = u_param[i, l]
                for j in 1:J_SIZE
                    B2[n_start_idx + idx, get_index([i, j, l, t], [I_SIZE, J_SIZE, L_SIZE, T_SIZE])] = -1
                end
            end
            n_start_idx += I_SIZE * L_SIZE * T_SIZE
            ## production_cons2: I_SIZE * L_SIZE * T_SIZE
            for i in 1:I_SIZE, l in 1:L_SIZE, t in 1:T_SIZE
                idx = get_index([i, l, t], [I_SIZE, L_SIZE, T_SIZE])
                B2_d[n_start_idx + idx, get_index([i, l, t], [I_SIZE, L_SIZE, T_SIZE])] = -u_param[i, l] * b_param[i, l]
                for j in 1:J_SIZE
                    B2[n_start_idx + idx, get_index([i, j, l, t], [I_SIZE, J_SIZE, L_SIZE, T_SIZE])] = 1
                end
            end
            n_start_idx += I_SIZE * L_SIZE * T_SIZE
            ## config_cons: I_SIZE * T_SIZE
            for i in 1:I_SIZE, t in 1:T_SIZE
                idx = get_index([i, t], [I_SIZE, T_SIZE])
                for l in 1:L_SIZE
                    B2_d[n_start_idx + idx, get_index([i, l, t], [I_SIZE, L_SIZE, T_SIZE])] = -1
                end
                d[n_start_idx + idx] = -1
            end
            n_start_idx += I_SIZE * T_SIZE
            if mobile_mode == "Mob"
                ## total_module_cons: M_SIZE * T_SIZE
                idx = 1
                for m in 1:M_SIZE, t in 1:T_SIZE 
                    if m in large_modules
                        for i in 1:I_SIZE
                            for l in 1:L_SIZE
                                B2_d[n_start_idx + idx, get_index([i, l, t], [I_SIZE, L_SIZE, T_SIZE])] = -n_param[l, m]
                                B1_d[n_start_idx + idx, get_index([i, m], [I_SIZE, M_SIZE])] = 1
                            end
                            idx += 1
                        end
                    else
                        for i in 1:I_SIZE, l in 1:L_SIZE
                            B2_d[n_start_idx + idx, get_index([i, l, t], [I_SIZE, L_SIZE, T_SIZE])] = -n_param[l, m]
                        end
                        for i in 1:I_SIZE
                            B1_d[n_start_idx + idx, get_index([i, m], [I_SIZE, M_SIZE])] = 1
                        end
                        idx += 1
                    end
                end
            else
                ## module_fixed_cons: I_SIZE * M_SIZE * T_SIZE
                for i in 1:I_SIZE, m in 1:M_SIZE, t in 1:T_SIZE 
                    idx = get_index([i, m, t], [I_SIZE, M_SIZE, T_SIZE])
                    for l in 1:L_SIZE
                        B2_d[n_start_idx + idx, get_index([i, l, t], [I_SIZE, L_SIZE, T_SIZE])] = -n_param[l, m]
                    end
                    B1_d[n_start_idx + idx, get_index([i, m], [I_SIZE, M_SIZE])] = 1
                end
            end
            n_start_idx += I_SIZE * M_SIZE * T_SIZE
            # y_bound_mmin_cons: I_SIZE * L_SIZE * T_SIZE
            for i in 1:I_SIZE, l in 1:L_SIZE, t in 1:T_SIZE
                idx = get_index([i, l, t], [I_SIZE, L_SIZE, T_SIZE])
                B2_d[n_start_idx + idx, idx] = 1
            end
            n_start_idx += I_SIZE * L_SIZE * T_SIZE
            # y_bound_max_cons: I_SIZE * L_SIZE * T_SIZE
            for i in 1:I_SIZE, l in 1:L_SIZE, t in 1:T_SIZE
                idx = get_index([i, l, t], [I_SIZE, L_SIZE, T_SIZE])
                B2_d[n_start_idx + idx, idx] = -1
                d[n_start_idx + idx] = -1
            end
            n_start_idx += I_SIZE * L_SIZE * T_SIZE
            # large_module_static_cons1: I_SIZE * (L-with-large-module_SIZE) * T_SIZE
            for i in 1:I_SIZE, m_idx in 1:size(large_modules)[1], t in 1:T_SIZE
                idx = get_index([i, m_idx, t], [I_SIZE, size(large_modules)[1], T_SIZE])
                for l in 1:L_SIZE
                    B2_d[n_start_idx + idx, get_index([i, l, t], [I_SIZE, L_SIZE, T_SIZE])] = -n_param[l, large_small_counterpart[large_modules[m_idx]]]
                end
                B1_d[n_start_idx + idx, get_index([i, large_modules[m_idx]], [I_SIZE, M_SIZE])] = -3
                d[n_start_idx + idx] = -3
            end
            n_start_idx += I_SIZE * T_SIZE * size(large_modules)[1]
            # # large_module_static_cons2: I_SIZE * (L-without-large-module_SIZE) * T_SIZE
            # for i in 1:I_SIZE, l in 1:3, t in 1:T_SIZE
            #     idx = get_index([i, l, t], [I_SIZE, 3, T_SIZE])
            #     B2_d[n_start_idx + idx, get_index([i, l, t], [I_SIZE, L_SIZE, T_SIZE])] = -1
            #     B1_d[n_start_idx + idx, get_index([i, 2], [I_SIZE, M_SIZE])] = -1
            #     d[n_start_idx + idx] = -1
            # end
            # n_start_idx += I_SIZE * 3 * T_SIZE

            ### Distribution subproblem 1 for uncertainty set
            max_spac_dev = Dict()
            min_spac_dev = Dict()
            max_temp_dev = Dict()
            min_temp_dev = Dict()

            for t in 1:T_SIZE
                DisSP = Model(opt)
                @variable(DisSP, d_dsp[1:UI], lower_bound = 0)
                @variable(DisSP, abs_dev[1:UI], lower_bound = 0)
                @variable(DisSP, dev[1:UI])
                @variable(DisSP, dev_sign[1:UI], Bin)
                @constraint(DisSP, sum(d_dsp[j] for j in 1:UI) == sum(d_param[uncertain_j_list[j], t] for j in 1:UI))
                @constraint(DisSP, [j = 1:UI], d_dsp[j] >= 0 * d_param[uncertain_j_list[j], t])
                @constraint(DisSP, [j = 1:UI], d_dsp[j] <= 2 * d_param[uncertain_j_list[j], t])
                @constraint(DisSP, [j = 1:UI], dev[j] == sum(d_param[uncertain_j_list[j], t] for j in 1:UI) / UI - d_dsp[j])
                @constraint(DisSP, [j = 1:UI], 0 <= abs_dev[j] - dev[j])
                @constraint(DisSP, [j = 1:UI], abs_dev[j] - dev[j] <= BigM * dev_sign[j])
                @constraint(DisSP, [j = 1:UI], 0 <= abs_dev[j] + dev[j])
                @constraint(DisSP, [j = 1:UI], abs_dev[j] + dev[j] <= BigM * (1 - dev_sign[j]))
                @objective(DisSP, Max, sum(abs_dev))
                set_silent(DisSP)
                optimize!(DisSP)
                max_spac_dev[t] = objective_value(DisSP)
                # @objective(DisSP, Min, sum(abs_dev))
                # optimize!(DisSP)
                # min_spac_dev[t] = objective_value(DisSP)
                mean_dem = sum(d_param[uncertain_j_list[j], t] for j in 1:UI) / UI
                min_spac_dev[t] = sum(abs(d_param[uncertain_j_list[j], t] - mean_dem) for j in 1:UI)
            end
            for t in 1:(T_SIZE - 1)
                TempSP = Model(opt)
                @variable(TempSP, d_tsp[1:UI, 1:2], lower_bound = 0)
                @variable(TempSP, dev[1:UI])
                @variable(TempSP, abs_dev[1:UI], lower_bound = 0)
                @variable(TempSP, dev_sign[1:UI], Bin)
                @constraint(TempSP, sum(d_tsp[j] for j in 1:UI) == sum(d_param[uncertain_j_list[j], t] for j in 1:UI))
                @constraint(TempSP, [j = 1:UI, t_ = 1:2], d_tsp[j, t_] >= 0 * d_param[uncertain_j_list[j], t + t_ - 1])
                @constraint(TempSP, [j = 1:UI, t_ = 1:2], d_tsp[j, t_] <= 2 * d_param[uncertain_j_list[j], t + t_ - 1])
                @constraint(TempSP, [j = 1:UI], dev[j] == d_tsp[j, 2] - d_tsp[j, 1])
                # @constraint(TempSP, [j = 1:UI], dev[j] == d_tsp[j, 2] / sum(d_param[uncertain_j_list[j], t + 1] for j in 1:UI) - d_tsp[j, 1] / sum(d_param[uncertain_j_list[j], t] for j in 1:UI))
                @constraint(TempSP, [j = 1:UI], 0 <= abs_dev[j] - dev[j])
                @constraint(TempSP, [j = 1:UI], abs_dev[j] - dev[j] <= BigM * dev_sign[j])
                @constraint(TempSP, [j = 1:UI], 0 <= abs_dev[j] + dev[j])
                @constraint(TempSP, [j = 1:UI], abs_dev[j] + dev[j] <= BigM * (1 - dev_sign[j]))
                @objective(TempSP, Max, sum(abs_dev))
                set_silent(TempSP)
                optimize!(TempSP)
                max_temp_dev[t] = objective_value(TempSP)
                min_temp_dev[t] = sum(abs(d_param[uncertain_j_list[j], t + 1] - d_param[uncertain_j_list[j], t]) for j in 1:UI)
                # @objective(TempSP, Min, sum(abs_dev))
                # optimize!(TempSP)
                # min_temp_dev[t] = objective_value(TempSP)
            end

            ### Decision-dependent uncertainty set
            ## F * u <= h + G * x + G_d * x_d
            flex_idx1_list = []
            flex_idx2_list = []
            x_d_val_list = []
            nominal_cost_list = []
            worst_cost_list = []
            design_large_list = []
            design_small_list = []
            u_star_dict = Dict()
            u_star_final_list = []
            y_d_val_final_list = []
            for flex_idx1 in 0:0.1:1
                for flex_idx2 in 0:0.1:1
                    push!(flex_idx1_list, flex_idx1)
                    push!(flex_idx2_list, flex_idx2)

                    n_start_idx = 0
                    for t in 1:T_SIZE
                        for j in 1:UI
                            u1 = get_index([j, t], [UI, T_SIZE])
                            F[n_start_idx + t, u1] = 1
                        end
                        h[n_start_idx + t] = sum(d_param[uncertain_j_list[j], t] for j in 1:UI)
                    end
                    n_start_idx += T_SIZE
                    for t in 1:T_SIZE
                        for j in 1:UI
                            u1 = get_index([j, t], [UI, T_SIZE])
                            F[n_start_idx + t, u1] = -1
                        end
                        h[n_start_idx + t] = -sum(d_param[uncertain_j_list[j], t] for j in 1:UI)
                    end
                    n_start_idx += T_SIZE
                    for j in 1:UI, t in 1:T_SIZE
                        idx = get_index([j, t], [UI, T_SIZE])
                        F[n_start_idx + idx, idx] = -1
                        h[n_start_idx + idx] = -0 * d_param[uncertain_j_list[j], t]
                    end
                    n_start_idx += UI * T_SIZE
                    for j in 1:UI, t in 1:T_SIZE
                        idx = get_index([j, t], [UI, T_SIZE])
                        F[n_start_idx + idx, idx] = 1
                        h[n_start_idx + idx] = 2 * d_param[uncertain_j_list[j], t]
                    end
                    n_start_idx += UI * T_SIZE
                    # Spacial dispersion: flex_idx1, u[UI * T_SIZE + 1:2 * UI * T_SIZE]
                    for j in 1:UI, t in 1:T_SIZE
                        idx = get_index([j, t], [UI, T_SIZE])
                        F[n_start_idx + idx, idx] = 1
                        F[n_start_idx + idx, UI * T_SIZE + idx] = -1
                        h[n_start_idx + idx] = sum(d_param[uncertain_j_list[j_], t] for j_ in 1:UI) / UI
                    end
                    n_start_idx += UI * T_SIZE
                    for j in 1:UI, t in 1:T_SIZE
                        idx = get_index([j, t], [UI, T_SIZE])
                        F[n_start_idx + idx, idx] = -1
                        F[n_start_idx + idx, UI * T_SIZE + idx] = -1
                        h[n_start_idx + idx] = -sum(d_param[uncertain_j_list[j_], t] for j_ in 1:UI) / UI
                    end
                    n_start_idx += UI * T_SIZE
                    for t in 1:T_SIZE
                        for j in 1:UI
                            F[n_start_idx + t, UI * T_SIZE + get_index([j, t], [UI, T_SIZE])] = 1
                        end
                        h[n_start_idx + t] = min_spac_dev[t] + flex_idx1 * (max_spac_dev[t] - min_spac_dev[t])
                    end
                    n_start_idx += T_SIZE
                    # Temporal change: flex_idx2, u[2 * UI * T_SIZE + 1:end]
                    for j in 1:UI, t in 1:(T_SIZE - 1)
                        idx = get_index([j, t], [UI, T_SIZE - 1])
                        u_idx1 = get_index([j, t], [UI, T_SIZE])
                        u_idx2 = get_index([j, t + 1], [UI, T_SIZE])
                        F[n_start_idx + idx, u_idx1] = 1 # / sum(d_param[uncertain_j_list[j], t + 1] for j in 1:UI)
                        F[n_start_idx + idx, u_idx2] = -1 # / sum(d_param[uncertain_j_list[j], t] for j in 1:UI)
                        F[n_start_idx + idx, 2 * UI * T_SIZE + idx] = -1
                    end
                    n_start_idx += UI * (T_SIZE - 1)
                    for j in 1:UI, t in 1:(T_SIZE - 1)
                        idx = get_index([j, t], [UI, T_SIZE - 1])
                        u_idx1 = get_index([j, t], [UI, T_SIZE])
                        u_idx2 = get_index([j, t + 1], [UI, T_SIZE])
                        F[n_start_idx + idx, u_idx1] = -1 # / sum(d_param[uncertain_j_list[j], t + 1] for j in 1:UI)
                        F[n_start_idx + idx, u_idx2] = 1 # / sum(d_param[uncertain_j_list[j], t] for j in 1:UI)
                        F[n_start_idx + idx, 2 * UI * T_SIZE + idx] = -1
                    end
                    n_start_idx += UI * (T_SIZE - 1)
                    for t in 1:(T_SIZE - 1)
                        for j in 1:UI
                            F[n_start_idx + t, 2 * UI * T_SIZE + get_index([j, t], [UI, T_SIZE - 1])] = 1
                        end
                        h[n_start_idx + t] = min_temp_dev[t] + flex_idx2 * (max_temp_dev[t] - min_temp_dev[t])
                    end
                    n_start_idx += T_SIZE - 1

                    ### Step 1 of Variant 2
                    x_val = zeros(dim_x)
                    x_d_val = zeros(dim_x_d)
                    u_val = zeros(dim_u)
                    LB = -Inf
                    UB = Inf
                    iter = 1
                    nc = 0
                    wc = 0

                    y_val = zeros(dim_y)
                    y_d_val = zeros(dim_y_d)
                    
                    x_val_best = zeros(dim_x)
                    x_d_val_best = zeros(dim_x_d)
                    y_val_dict = Dict()
                    y_d_val_dict = Dict()
                    tic = Dates.now()

                    # Initialize the nominal u for the step method
                    Init_u_model = Model(opt)
                    set_silent(Init_u_model)
                    @variable(Init_u_model, u_init[1:dim_u])
                    @objective(Init_u_model, Min, sum(u_init))
                    @constraint(Init_u_model, F * u_init .<= h)
                    for j in 1:UI, t in 1:T_SIZE
                        @constraint(Init_u_model, u_init[get_index([j, t], [UI, T_SIZE])] == d_param[uncertain_j_list[j], t])
                    end
                    optimize!(Init_u_model)
                    u_nom = value.(u_init)

                    @printf("%s set1+_step %s: flex_idx1: %.1f, flex_idx2: %.1f\n", mobile_mode, dataset, flex_idx1, flex_idx2)
                    @printf("%s\n", Dates.format(Dates.now(), "yyyy-mm-dd HH:MM:SS"))
                    @printf("--------------------- Algorithm start ---------------------\n")
                    @printf("%-9s%-15s%-15s%-9s%-9s\n", "Iter", "Upper Bound", "Lower Bound", "Gap%", "Time (s)")
                    UB_out = Inf
                    LB_out = -Inf
                    while true
                        MP_out = Model(opt)
                        
                        set_silent(MP_out)
                        set_attribute(MP_out, "Threads", 112)
                        @variable(MP_out, x_out[1:dim_x], lower_bound = 0)
                        @variable(MP_out, x_d_out[1:dim_x_d], Int, lower_bound = 0, upper_bound = 3)
                        @variable(MP_out, eta_out, lower_bound = 0)
                        @constraint(MP_out, A * x_out + A_d * x_d_out .>= b)

                        y_out = Dict()
                        y_d_out = Dict()
                        for it in 1:(iter - 1)
                            y_out[it] = @variable(MP_out, [1:dim_y], lower_bound = 0)
                            y_d_out[it] = @variable(MP_out, [1:dim_y_d], Bin)
                            @constraint(MP_out, eta_out >= c2' * y_out[it] + c2_d' * y_d_out[it])
                            @constraint(MP_out, B2 * y_out[it] + B2_d * y_d_out[it] .>= d - B1 * x_out - B1_d * x_d_out - E * u_star_dict[it])
                        end
                        @objective(MP_out, Min, c1' * x_out + c1_d' * x_d_out + eta_out)
                        optimize!(MP_out)
                        x_val = value.(x_out)
                        x_d_val = value.(x_d_out)
                        for i in 1:(iter - 1)
                            y_val_dict[i] = value.(y_out[i])
                            y_d_val_dict[i] = value.(y_d_out[i])
                        end
                        LB_out = max(LB_out, objective_value(MP_out))
                        @printf("Iteration %d MP: %.3f\n", iter, objective_value(MP_out))

                        # Call the oracle to solve subproblem and get a new u*
                        u_star_dict[iter], second_obj = oracleIR_Step(true, opt, x_val, x_d_val, F, h, G, G_d, B2, B2_d, d, B1, B1_d, E, c2, c2_d, u_nom)
                        if UB_out > c1' * x_val + c1_d' * x_d_val + second_obj
                            x_val_best = x_val
                            x_d_val_best = x_d_val
                        end
                        UB_out = min(UB_out, c1' * x_val + c1_d' * x_d_val + second_obj)

                        toc = (Dates.now() - tic) / Dates.Millisecond(1) / 1000.0
                        @printf("Iteration %d UB_out: %.3f, LB_out: %.3f, time elapsed: %.2fs\n", iter, UB_out, LB_out, toc)

                        if ((UB_out - LB_out) / UB_out < 5e-4) || (iter >= 10)
                            break
                        end
                        iter += 1
                    end
                    @printf("---------------------- Algorithm end ----------------------\n")

                    v_val = zeros(I_SIZE, M_SIZE, T_SIZE)
                    for i in 1:I_SIZE, m in 1:M_SIZE
                        @printf("Z[i = %d, m = %d]: %.3f\n", i, m, x_d_val[get_index([i, m], [I_SIZE, M_SIZE])])
                    end
                    
                    y_origin_dict = Dict()
                    y_d_origin_dict = Dict()
                    worst_iter = 1
                    worst_cost = 0
                    for it in 1:iter
                        Original = Model(opt)
                        @variable(Original, eta_origin)
                        @variable(Original, y_origin[1:dim_y], lower_bound = 0)
                        @variable(Original, y_d_origin[1:dim_y_d], Bin)
                        @constraint(Original, B2 * y_origin + B2_d * y_d_origin .>= d - B1 * x_val_best - B1_d * x_d_val_best - E * u_star_dict[it])
                        @objective(Original, Min, c2' * y_origin + c2_d' * y_d_origin)
                        set_silent(Original)
                        set_attribute(Original, "Threads", 112)
                        optimize!(Original)

                        y_origin_dict[it] = value.(y_origin)
                        y_d_origin_dict[it] = value.(y_d_origin)
                        
                        if objective_value(Original) > worst_cost 
                            worst_iter = it
                            worst_cost = objective_value(Original)
                        end
                    end

                    y_origin_val = y_origin_dict[worst_iter]
                    y_d_origin_val = y_d_origin_dict[worst_iter]
                    y_d_str = ""
                    for i in 1:I_SIZE, t in 1:T_SIZE, l in 1:L_SIZE
                        if y_d_origin_val[get_index([i, l, t], [I_SIZE, L_SIZE, T_SIZE])] > 1e-3
                            @printf("Y[i = %d, l = %d, t = %d]: %.4f\n", i, l, t, y_d_origin_val[get_index([i, l, t], [I_SIZE, L_SIZE, T_SIZE])])
                            y_d_str = y_d_str * @sprintf("Y[i = %d, l = %d, t = %d]: %.2f; ", i, l, t, y_d_origin_val[get_index([i, l, t], [I_SIZE, L_SIZE, T_SIZE])])
                        end
                    end
                    # for i in 1:I_SIZE, j in 1:J_SIZE, l in 1:L_SIZE, t in 1:T_SIZE
                    #     if y_origin_val[get_index([i, j, l, t], [I_SIZE, J_SIZE, L_SIZE, T_SIZE])] > 1e-3
                    #         @printf("c2[i = %d, j = %d] = %.4f\n", i, j, c2[get_index([i, j, l, t], [I_SIZE, J_SIZE, L_SIZE, T_SIZE])])
                    #         @printf("X[i = %d, j = %d, l = %d, t = %d] = %.2f\n", i, j, l, t, y_origin_val[get_index([i, j, l, t], [I_SIZE, J_SIZE, L_SIZE, T_SIZE])])
                    #     end
                    # end
                    push!(y_d_val_final_list, y_d_str)
                    # Write solution and worst case to csv file
                    if write_to_csv
                        if !isdir("./data/$(setName)/Result_with_nominal")
                            mkdir("./data/$(setName)/Result_with_nominal") 
                        end
                        if !isdir("./data/$(setName)/Result_with_nominal/Variable_result")
                            mkdir("./data/$(setName)/Result_with_nominal/Variable_result")
                        end

                        x_d_out = zeros(I_SIZE, M_SIZE)
                        y_out = zeros(I_SIZE, J_SIZE, L_SIZE, T_SIZE)
                        y_d_out = zeros(I_SIZE, L_SIZE, T_SIZE)
                        uj_out = zeros(UI, T_SIZE)
                        u_out = zeros(UI, T_SIZE)

                        for i in 1:I_SIZE, m in 1:M_SIZE
                            x_d_out[i, m] = x_d_val_best[get_index([i, m], [I_SIZE, M_SIZE])]
                        end
                        for i in 1:I_SIZE, j in 1:J_SIZE, l in 1:L_SIZE, t in 1:T_SIZE
                            y_out[i, j, l, t] = y_origin_val[get_index([i, j, l, t], [I_SIZE, J_SIZE, L_SIZE, T_SIZE])]
                        end
                        for i in 1:I_SIZE, l in 1:L_SIZE, t in 1:T_SIZE
                            y_d_out[i, l, t] = y_d_origin_val[get_index([i, l, t], [I_SIZE, L_SIZE, T_SIZE])]
                        end
                        for t in 1:T_SIZE
                            uj_out[:, t] = uncertain_j_list
                        end
                        for uj in 1:UI, t in 1:T_SIZE
                            u_out[uj, t] = u_star_dict[worst_iter][get_index([uj, t], [UI, T_SIZE])]
                        end
                        CSV.write("./data/$(setName)/Result_with_nominal/Variable_result/Set1_result_x_design_$(mobile_mode)_FI1=$(flex_idx1)_FI2=$(flex_idx2).csv", Containers.rowtable(x_d_out; header = [:i, :m, :value]))
                        CSV.write("./data/$(setName)/Result_with_nominal/Variable_result/Set1_result_y_flow_$(mobile_mode)_FI1=$(flex_idx1)_FI2=$(flex_idx2).csv", Containers.rowtable(y_out; header = [:i, :j, :l, :t, :value]))
                        CSV.write("./data/$(setName)/Result_with_nominal/Variable_result/Set1_result_y_config_$(mobile_mode)_FI1=$(flex_idx1)_FI2=$(flex_idx2).csv", Containers.rowtable(y_d_out; header = [:i, :l, :t, :value]))
                        CSV.write("./data/$(setName)/Result_with_nominal/Variable_result/Set1_result_uncertain_j_list_$(mobile_mode)_FI1=$(flex_idx1)_FI2=$(flex_idx2).csv", Containers.rowtable(uj_out; header = [:uj, :t, :value]))
                        CSV.write("./data/$(setName)/Result_with_nominal/Variable_result/Set1_result_worst_case_$(mobile_mode)_FI1=$(flex_idx1)_FI2=$(flex_idx2).csv", Containers.rowtable(u_out; header = [:uj, :t, :value]))
                    end
                    wc = c2' * y_origin_val / (1 - nom_weight) + c1_d' * x_d_val_best
                    nc = c1' * x_val_best / nom_weight + c1_d' * x_d_val_best
                    @printf("Worst cost: %.2f\n", wc)
                    @printf("Nominal cost: %.2f\n", nc)
                    push!(x_d_val_list, x_d_val)
                    push!(worst_cost_list, wc)
                    push!(nominal_cost_list, nc)
                    mod_s_cnt, mod_l_cnt = 0, 0
                    for i in 1:I_SIZE
                        mod_s_cnt += x_d_val_best[get_index([i, 1], [I_SIZE, M_SIZE])]
                        mod_l_cnt += x_d_val_best[get_index([i, 2], [I_SIZE, M_SIZE])]
                    end
                    push!(design_large_list, mod_l_cnt)
                    push!(design_small_list, mod_s_cnt)

                    # u_star = u_star_dict[iter]
                    # u_star_str = ""
                    # for j in 1:J_SIZE, t in 1:T_SIZE
                    #     u_star_str = u_star_str * @sprintf("d[j = %d, t = %d]: %.2f; ", j, t, u_star[get_index([j, t], [J_SIZE, T_SIZE])])
                    # end
                    # println(u_star_str)
                    # push!(u_star_final_list, u_star_str)

                    #@printf("Worst case cost: %.4f\n", (c2' * y_val_dict[iter] + c2_d' * y_d_val_dict[iter]) / (1 - nom_weight) + c1_d' * x_d_val)
                    #@printf("Nominal case cost: %.4f\n", c1' * x_val / nom_weight + c1_d' * x_d_val)
                end
            end
            ### Output result to CSV file
            if write_to_csv
                out_df = DataFrames.DataFrame(
                    FI_spac = flex_idx1_list,
                    FI_temp = flex_idx2_list,
                    worst_case = worst_cost_list,
                    nominal_case = nominal_cost_list,
                    large_modules = design_large_list,
                    small_modules = design_small_list,
                    # worst_demand = u_star_final_list,
                    worst_decision = y_d_val_final_list
                )
                if !isdir("./data/$(setName)/Result_with_nominal")
                    mkdir("./data/$(setName)/Result_with_nominal")
                end
                CSV.write("./data/$(setName)/Result_with_nominal/$(mobile_mode)_Set1+_$(dataset).csv", out_df)
            end
        end
    end
end