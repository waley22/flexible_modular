using Distributions, LinearAlgebra, GLM, StatsBase, JuMP
import CSV, DataFrames, Plots, Tables, Plots, XLSX
include("../../utils.jl")
cd(dirname(Base.source_path()))

### Data from file (as reference)
sets_df = CSV.read("../../BMWE_MN_Simplified/input_sets.csv", DataFrames.DataFrame)
I_ORIGINAL = sets_df[2, 1]      # Set of candidate and facility locations
J_ORIGINAL = sets_df[1, 1]      # Set of customers
L_ORIGINAL = sets_df[3, 1]      # Set of facility configurations
M_ORIGINAL = sets_df[4, 1]      # Set of types of modules
T_ORIGINAL = sets_df[5, 1]      # Set of time periods

b_param_org = zeros(I_ORIGINAL, L_ORIGINAL)
c_param_org = zeros(I_ORIGINAL, J_ORIGINAL)
d_param_org = zeros(J_ORIGINAL, T_ORIGINAL)
g_param_org = zeros(I_ORIGINAL, M_ORIGINAL)
n_param_org = zeros(L_ORIGINAL, M_ORIGINAL)
u_param_org = zeros(I_ORIGINAL, L_ORIGINAL)
jloc_param_org = zeros(J_ORIGINAL, 2)
iloc_param_org = zeros(I_ORIGINAL, 2)

recover_param!(b_param_org, "../../BMWE_MN_Simplified/input_b.csv")
recover_param!(c_param_org, "../../BMWE_MN_Simplified/input_c.csv")
recover_param!(d_param_org, "../../BMWE_MN_Simplified/input_d.csv")
recover_param!(g_param_org, "../../BMWE_MN_Simplified/input_g_small.csv")
recover_param!(n_param_org, "../../BMWE_MN_Simplified/input_n.csv")
recover_param!(u_param_org, "../../BMWE_MN_Simplified/input_u_small.csv")

jloc_df = DataFrames.DataFrame(XLSX.readtable("../../BMWE_MN_Simplified/MNResidue_n_distance.xlsx", "distance_py"))
iloc_df = DataFrames.DataFrame(XLSX.readtable("../../BMWE_MN_Simplified/MNResidue_n_distance.xlsx", "cr9_py"))
for j in 1:J_ORIGINAL
    jloc_param_org[j, 1] = jloc_df[j, "Latitude"]
    jloc_param_org[j, 2] = jloc_df[j, "Longitude"]
end
for i in 1:I_ORIGINAL
    iloc_param_org[i, 1] = iloc_df[i, "Latitude"]
    iloc_param_org[i, 2] = iloc_df[i, "Longitude"]
end
dist_list = zeros(I_ORIGINAL * J_ORIGINAL)
cost_list = zeros(I_ORIGINAL * J_ORIGINAL)
for i in 1:I_ORIGINAL, j in 1:J_ORIGINAL
    dist_list[get_index([i, j], [I_ORIGINAL, J_ORIGINAL])] = norm(iloc_param_org[i, :] - jloc_param_org[j, :])
    cost_list[get_index([i, j], [I_ORIGINAL, J_ORIGINAL])] = c_param_org[i, j]
end
lg_data = DataFrames.DataFrame(X = dist_list, Y = cost_list)
ols = lm(@formula(Y ~ X), lg_data)

d_param_base = mean(d_param_org[:, 3])
tr_slope, tr_intercept = coef(ols)
Plots.scatter(dist_list, cost_list)

### New data (use original data as basis)
I_SIZE = 2
J_SIZE = 4
L_SIZE = 4
M_SIZE = 2
T_SIZE = 2

b_param = zeros(I_SIZE, L_SIZE)
c_param = zeros(I_SIZE, J_SIZE)
d_param = zeros(J_SIZE, T_SIZE)
g_param = zeros(I_SIZE, M_SIZE)
n_param = zeros(L_SIZE, M_SIZE)
u_param = zeros(I_SIZE, L_SIZE)
center_param = zeros(T_SIZE, 2)

# setName = "RadialIn"
setName = "Linear"
# setName = "Linear_t=4" # 4/8/4/2/3+
# setName = "Linear_real" # 8/30/4/2/4
for k in 0:0
    for k2 in 1:20
        local iloc_param, jloc_param, ijdist_param
        iloc_param = rand(Uniform(0, 4), (I_SIZE, 2))
        iloc_param[1, 1] = rand(Uniform(1, 3))
        iloc_param[2, 1] = rand(Uniform(1, 3))
        iloc_param[1, 2] = rand(Uniform(0, 2))
        iloc_param[2, 2] = rand(Uniform(2, 4))
        jloc_param = rand(Uniform(0, 4), (J_SIZE, 2))
        ijdist_param = zeros(I_SIZE, J_SIZE)

        for i in 1:I_SIZE
            for j in 1:J_SIZE
                ijdist_param[i, j] = norm(iloc_param[i, :] - jloc_param[j, :])
                c_param[i, j] = tr_intercept + rand(Uniform(0.9 * ijdist_param[i, j], 1.1 * ijdist_param[i, j])) * tr_slope
            end
        end

        for i in 1:I_SIZE, l in 1:L_SIZE
            global b_param
            b_param[i, l] = b_param_org[i, l]
        end
        for i in 1:I_SIZE, m in 1:M_SIZE
            global g_param
            g_param[i, m] = g_param_org[i, m]
        end
        global n_param_org, n_param
        n_param = n_param_org
        for i in 1:I_SIZE, l in 1:L_SIZE
            global u_param
            u_param[i, l] = u_param_org[i, l]
        end

        print

        ### Generate random demand
        ### 1. Uniform
        ### 2. Linear (w.r.t. x or y-coordinate)
        ### 3. Radial (demand focused in the center)
        ### 4. Radial (demand scattered more in the outskirts)

        ### Linear
        if occursin("Linear", setName)
            for j in 1:J_SIZE, t in 1:T_SIZE
                global d_param
                t_lean = k * 0.1
                d_center = (1 + (jloc_param[j, 2] - 2) / 2 * t_lean * (-1)^t) * d_param_base
                d_param[j, t] = rand(Uniform(0.8 * d_center, 1.2 * d_center))
            end
            for t in 1:T_SIZE
                d_param[:, t] = d_param[:, t] ./ sum(d_param[:, t]) * (d_param_base * J_SIZE)
            end
        end

        ### Radial
        if setName == "RadialIn"
            global d_param, center_param
            for t in 1:T_SIZE
                center = rand(Uniform(0, 4), 2)
                center_param[t, :] = center
                for j in 1:J_SIZE
                    d_center = d_param_base * 2.086 / (norm(jloc_param[j, :] .- center) + (12 - k) * 0.25)
                    d_param[j, t] = rand(Uniform(0.8 * d_center, 1.2 * d_center))
                end
                d_param[:, t] = d_param[:, t] ./ mean(d_param[:, t]) * d_param_base
            end
        end

        if !isdir("./$(setName)")
            mkdir("./$(setName)")
        end
        if !isdir("./$(setName)/$(setName)_$(k)_$(k2)")
            mkdir("./$(setName)/$(setName)_$(k)_$(k2)")
        end

        sets_out = DataFrames.DataFrame(size = [J_SIZE, I_SIZE, L_SIZE, M_SIZE, T_SIZE])
        ### Notice that i and j are reversed in the header for b, c, d, n, g and u due to how they are handled in the main file
        CSV.write("./$(setName)/$(setName)_$(k)_$(k2)/input_size.csv", sets_out)
        CSV.write("./$(setName)/$(setName)_$(k)_$(k2)/input_b.csv", Containers.rowtable(b_param; header = [:j, :l, :value]))
        CSV.write("./$(setName)/$(setName)_$(k)_$(k2)/input_c.csv", Containers.rowtable(Matrix(c_param'); header = [:j, :i, :value]))
        CSV.write("./$(setName)/$(setName)_$(k)_$(k2)/input_d.csv", Containers.rowtable(d_param; header = [:i, :t, :value]))
        CSV.write("./$(setName)/$(setName)_$(k)_$(k2)/input_g.csv", Containers.rowtable(g_param; header = [:j, :m, :value]))
        CSV.write("./$(setName)/$(setName)_$(k)_$(k2)/input_n.csv", Containers.rowtable(n_param; header = [:l, :m, :value]))
        CSV.write("./$(setName)/$(setName)_$(k)_$(k2)/input_u.csv", Containers.rowtable(u_param; header = [:j, :l, :value]))
        CSV.write("./$(setName)/$(setName)_$(k)_$(k2)/input_iloc.csv", Containers.rowtable(iloc_param; header = [:i, :coord, :value]))
        CSV.write("./$(setName)/$(setName)_$(k)_$(k2)/input_jloc.csv", Containers.rowtable(jloc_param; header = [:j, :coord, :value]))
        if setName == "RadialIn"
            CSV.write("./$(setName)/$(setName)_$(k)_$(k2)/input_center.csv", Containers.rowtable(center_param; header = [:t, :coord, :value]))
        end
    end
end
println("Generate data done")