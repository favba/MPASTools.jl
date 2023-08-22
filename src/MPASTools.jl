module MPASTools

export compute_errors, compute_errors_steady_state_case

using NCDatasets

"""
    compute_errors(result,exact,weight)

Given a `result` (nVertLevels, nEle) matrix, an `exact` (nVertLevels, nEle) matrix and a `weight` (nEle) vector,
computes the absolute and relative L∞ errors and the absolute and relative RMS errors.
"""
function compute_errors(result::AbstractMatrix,exact::AbstractMatrix,weight::AbstractVector)
    num = zero(eltype(result))
    den = zero(eltype(result))
    total_area = zero(eltype(result))
    l∞ = zero(eltype(result))
    L∞ = zero(eltype(result))
    @inbounds for j in eachindex(weight) 
        aj = weight[j]
        for i in axes(result,1)
            ex = exact[i,j]
            diff = result[i,j] - ex
            L∞ = max(L∞,abs(diff))
            ival = diff/ex
            ival = ifelse(isnan(ival),zero(ival),ival)
            l∞ = max(l∞,abs(ival))

            ex = ex*ex
            diff = diff*diff

            num = muladd(aj,diff,num)
            den = muladd(aj,ex,den)
            total_area += aj
        end
    end
    return (L∞, l∞, sqrt(num/total_area), sqrt(num/den))
end

"""
    get_avarage_weights_name(NCvar)

Returns the name of a MPAS mesh variable that can properly serve as weights to compute avarages for `NCvar`.

# Examples
```julia-repl
julia> get_avarage_weights_name(rho) # rho is a cell centered value on MPAS
"areaCell"
```
"""
function get_avarage_weights_name(var::NCDatasets.CFVariable)
    dnames = dimnames(var)
    wname = ""
    if "nCells" in dnames
        wname="areaCell"
    elseif "nEdges" in dnames
        wname="dvEdge"
    elseif "nVertices" in dnames
        wname="areaTriangle"
    else
        thorw(DomainError(var,"Variable doesn't seem to be neither a Cell, Edge, or Vertex value"))
    end
    return wname
end

"""
    compute_errors_steady_state_case(fields::Vector{String}, netcdf_file::String)

Computes the errors for each field name in `fields` from the NetCDF file named `netcdf_file` assuming results are from a case that should stay steady state.
The error is computed taking the initial condition as the analytical solution.
"""
function compute_errors_steady_state_case(fields::AbstractVector{T},netcdf_file::AbstractString) where T<:AbstractString
    NCDataset(netcdf_file) do nc

        #Constructing this dict to read avarageing weight only once
        dic = Dict{String,Vector}()
        err = Vector{NTuple{4,Float64}}(undef,length(fields))

        Threads.@sync for (i,field) in enumerate(fields)
            var = nc[field]
            weight_name = get_avarage_weights_name(var)
            weight_name ∉ keys(dic) && (dic[weight_name] = nc[weight_name][:])
            result = var[:,:,end]
            exact = var[:,:,1]
            let i=i, weight_name=weight_name, result=result, exact=exact
                Threads.@spawn begin
                    @inbounds err[i] = compute_errors(result, exact, dic[weight_name])
                end
            end
        end

        return err
    end
end

end #module