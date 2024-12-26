using Distributions, LinearAlgebra, GLM, StatsBase, JuMP, Random
import CSV, DataFrames, Plots, Tables, Plots, XLSX
include("../../utils.jl")
cd(dirname(Base.source_path()))

base_data = "BMWE_MN_Simplified"

### Data from file (as reference)
sets_df = CSV.read("../../$(base_data)/input_sets.csv", DataFrames.DataFrame)
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

recover_param!(b_param_org, "../../$(base_data)/input_b.csv")
recover_param!(c_param_org, "../../$(base_data)/input_c.csv")
recover_param!(d_param_org, "../../$(base_data)/input_d_2.csv")
recover_param!(g_param_org, "../../$(base_data)/input_g.csv")
recover_param!(n_param_org, "../../$(base_data)/input_n.csv")
recover_param!(u_param_org, "../../$(base_data)/input_u.csv")

jloc_df = DataFrames.DataFrame(XLSX.readtable("../../$(base_data)/MNResidue_n_distance.xlsx", "distance_py"))
iloc_df = DataFrames.DataFrame(XLSX.readtable("../../$(base_data)/MNResidue_n_distance.xlsx", "cr9_py"))
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

tr_slope, tr_intercept = coef(ols)
Plots.scatter(dist_list, cost_list)

### New data (use original data as basis)
I_SIZE = 9
J_SIZE = 20
L_SIZE = 4
M_SIZE = 2
T_SIZE = 4

b_param = zeros(I_SIZE, L_SIZE)
c_param = zeros(I_SIZE, J_SIZE)
d_param = zeros(J_SIZE, T_SIZE)
g_param = zeros(I_SIZE, M_SIZE)
n_param = zeros(L_SIZE, M_SIZE)
u_param = zeros(I_SIZE, L_SIZE)

i_centers = shuffle(1:I_ORIGINAL)[1:I_SIZE]

j_nonempty_list = []
for j in 1:J_ORIGINAL
    if sum(d_param_org[j, :]) > 100
       push!(j_nonempty_list, j)
    end
end

# d_param_sum = dropdims(sum(d_param_org; dims = 2); dims = 2)
# j_centers = sortperm(d_param_sum)[end - J_SIZE + 1:end]
# j_centers = shuffle(j_nonempty_list)[1:J_SIZE]
j_centers = shuffle(collect(1:J_ORIGINAL))[1:J_SIZE]
# i_nearest_center = ones(Int, I_ORIGINAL)
j_nearest_center = ones(Int, J_ORIGINAL)
# for i in 1:I_ORIGINAL
#     for i_ in 1:I_SIZE
#         if (norm(iloc_param_org[i, :] - iloc_param_org[i_centers[i_], :]) < norm(iloc_param_org[i, :] - iloc_param_org[i_nearest_center[i], :]))
#             i_nearest_center[i] = i_
#         end
#     end
# end

for j in 1:J_ORIGINAL
    for j_ in 1:J_SIZE
        if (norm(jloc_param_org[j, :] - jloc_param_org[j_centers[j_], :]) < norm(jloc_param_org[j, :] - jloc_param_org[j_centers[j_nearest_center[j]], :]))
            j_nearest_center[j] = j_
        end
    end
end

iloc_param = zeros(I_SIZE, 2)
for i in 1:I_SIZE
    iloc_param[i, :] = iloc_param_org[i, :]
end
jloc_param = zeros(J_SIZE, 2)
for j in 1:J_SIZE
    jloc_param[j, :] = jloc_param_org[j_centers[j], :]
end

for i in 1:I_SIZE
    for j in 1:J_SIZE
        c_param[i, j] = c_param_org[i, j_centers[j]]
    end
end

for i in 1:I_SIZE, l in 1:L_SIZE
    b_param[i, l] = b_param_org[i, l]
end
for i in 1:I_SIZE, m in 1:M_SIZE
    g_param[i, m] = g_param_org[i, m]
end
for l in 1:L_SIZE, m in 1:M_SIZE
    n_param[l, m] = n_param_org[l, m]
end
for i in 1:I_SIZE, l in 1:L_SIZE
    u_param[i, l] = u_param_org[i, l]
end

## Demand parameter
for j in 1:J_ORIGINAL, t in 1:T_SIZE
    d_param[j_nearest_center[j], t] += d_param_org[j, t]
end

k = 1
k2 = 1
setName = "Linear_real" # 9/25/4/2/4
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
CSV.write("./$(setName)/$(setName)_$(k)_$(k2)/input_jcenters.csv", Containers.rowtable(j_centers))
println("Base data: ", base_data)
println("Set name: ", setName)
println("Generate real-size data done")