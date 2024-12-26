import CSV, DataFrames

function print_status(prompt::String, value::Bool) 
    print(prompt)
    if value
        printstyled("True", bold = true, color = :green)
    else
        printstyled("False", bold = true, color = :red)
    end
    println()
end

function recover_param!(param, filename)
    param_df = CSV.read(filename, DataFrames.DataFrame)
    if "i" in DataFrames.names(param_df) && "j" in DataFrames.names(param_df)
        param_df[:, :j], param_df[:, :i] = param_df[:, :i], param_df[:, :j]
    end
    for t in Tuple.(eachrow(param_df))
        index = t[1:end - 1]
        value = t[end]
        param[index...] = value
    end
    return
end

function get_index(index::Array{Int64}, dimensions::Array{Int64})
    @assert length(index) == length(dimensions)
    result = 0
    multiplier = 1
    for idx = length(index):-1:1
        if idx == length(index)
            result += index[idx]
        else
            result += (index[idx] - 1) * multiplier
        end
        multiplier *= dimensions[idx]
    end
    return result
end