@inline function perturb_point(r::Number,p)
    θ = acos(rand(-1.0:0.01:1.0))
    pert = rand(0.5:0.01:1.0)*r*Vec(x=cos(θ),y=sin(θ))
    return p+pert
end

function perturb_points!(out,r::Number,points)
    @inbounds for i in eachindex(points)
        out[i] = perturb_point(r,points[i])
    end
    return out
end

function perturb_points(r::Number,points)
    out = similar(points)
    return perturb_points!(out,r,points)
end

function compute_new_circumcenters_periodic!(result, cell_pos, vert_pos, cellsOnVertex, x_period::Number, y_period::Number)

    @inbounds @inline for i in eachindex(result)
        ind_cells = cellsOnVertex[i]
        vertex_pos = vert_pos[i]
        cell1_pos = cell_pos[ind_cells[1]]
        cell_1 = closest(vertex_pos,cell1_pos,x_period,y_period)

        cell2_pos = cell_pos[ind_cells[2]]
        cell_2 = closest(vertex_pos,cell2_pos,x_period,y_period)

        cell3_pos = cell_pos[ind_cells[3]]
        cell_3 = closest(vertex_pos,cell3_pos,x_period,y_period)

        result[i] = circumcenter(cell_1,cell_2,cell_3)
    end

    return result
end

function compute_new_circumcenters_periodic(cell_pos,vert_pos,cellsOnVertex,x_period::Number,y_period::Number)
    result = similar(vert_pos)
    return compute_new_circumcenters_periodic!(result,cell_pos,vert_pos,cellsOnVertex,x_period,y_period)
end

@inline function round_to_even_Int(x::Real)
    ru = round(Int,x,RoundUp)
    rd = round(Int,x,RoundDown)
    if ru == rd
        val = ru + mod(ru,2)
    else
        val = iseven(ru) ? ru : rd
    end
    return val
end

function prepend_string_to_history(ncfile,s)
    if haskey(ncfile.attrib,"history")
        cs = ncfile.attrib["history"]
        ncfile.attrib["history"] = string(s,'\n',cs)
    else
        ncfile.attrib["history"] = s
    end
    return ncfile.attrib["history"]
end

function append_string_to_history(ncfile,s)
    if haskey(ncfile.attrib,"history")
        cs = ncfile.attrib["history"]
        ncfile.attrib["history"] = string(cs,'\n',s)
    else
        ncfile.attrib["history"] = s
    end
    return ncfile.attrib["history"]
end

function create_planar_hex_mesh(filename::AbstractString,lx::Number,ly::Number,dc::Number)
    nx = round(Int,lx/dc)
    ny = round_to_even_Int((2ly)/(√3*dc))
    ny+=mod(ny,2)
    temp_o = splitext(filename)[1]*"_hdf5.nc"
    CondaPkg.withenv() do
        run(`planar_hex --nx $nx --ny $ny --dc $dc -o $temp_o`)
        run(`nccopy -6 $temp_o $filename`)
        NCDataset(filename,"a") do f
            prepend_string_to_history(f,string("nccopy -6 ",temp_o," ",filename))
        end
        run(`rm $temp_o`)
    end
end

precompile(create_planar_hex_mesh,(String,Float64,Float64,Float64,Float64))

import CommonDataModel

const NCArrayType{T,N} = CommonDataModel.CFVariable{T, N, NCDatasets.Variable{T, N, NCDataset{Nothing, Missing}}, CommonDataModel.Attributes{NCDatasets.Variable{T, N, NCDataset{Nothing, Missing}}}, @NamedTuple{fillvalue::Nothing, missing_values::Tuple{}, scale_factor::Nothing, add_offset::Nothing, calendar::Nothing, time_origin::Nothing, time_factor::Nothing, maskingvalue::Missing}}

function distort_periodic_mesh(infile::AbstractString,pert_val::Number)

    in_file_name = splitext(infile)[1]
    outfile = in_file_name*"_distorted_hdf5.nc"
    innc = NCDataset(infile)
    outnc = NCDataset(outfile,"c")
    
    for (dimname, val) in innc.dim
        defDim(outnc,dimname,val)
    end
    
    for (attname, val) in innc.attrib
        outnc.attrib[attname] = val
    end
    
    for field in ("latCell","lonCell","xCell","yCell","zCell","indexToCellID",
                  "latEdge","lonEdge","xEdge","yEdge","zEdge","indexToEdgeID",
                  "latVertex","lonVertex","xVertex","yVertex","zVertex","indexToVertexID",
                  "nEdgesOnCell","nEdgesOnEdge","cellsOnEdge","edgesOnCell","edgesOnEdge",
                  "cellsOnCell","verticesOnCell","verticesOnEdge","edgesOnVertex",
                  "cellsOnVertex","weightsOnEdge","dvEdge","dcEdge","angleEdge","areaCell",
                  "areaTriangle","kiteAreasOnVertex","meshDensity")
        defVar(outnc,innc[field])
    end

    cells = CellBase(innc)::CellBase{false,6,Int32,Float64,Zeros.Zero}
    vertices = VertexBase(innc)::VertexBase{false,Int32,Float64,Zeros.Zero}
    edges = EdgeBase(innc)::EdgeBase{false,Int32,Float64,Zeros.Zero}
    velRecon = EdgeVelocityReconstruction(innc)::EdgeVelocityReconstruction{10,Int32,Float64}
    xp = innc.attrib["x_period"]::Float64
    yp = innc.attrib["y_period"]::Float64

    close(innc)

    new_cells_pos = perturb_points(pert_val,cells.position)
    new_vert_pos = compute_new_circumcenters_periodic(new_cells_pos,vertices.position,vertices.indices.cells,xp,yp)

    copyto!(outnc["xCell"]::NCArrayType{Float64,1},new_cells_pos.x)
    copyto!(outnc["yCell"]::NCArrayType{Float64,1},new_cells_pos.y)

    copyto!(outnc["xVertex"]::NCArrayType{Float64,1},new_vert_pos.x)
    copyto!(outnc["yVertex"]::NCArrayType{Float64,1},new_vert_pos.y)

    cells.position .= new_cells_pos
    vertices.position .= new_vert_pos

    n_obtuse_triangles = length(find_obtuse_triangles(vertices,cells,xp,yp))

    n_obtuse_triangles != 0 && @warn "Mesh distortion produced $n_obtuse_triangles obtuse triangles. Consider using a smaller perturbation value"

    at = compute_area_triangles(vertices,cells,xp,yp)
    copyto!(outnc["areaTriangle"]::NCArrayType{Float64,1},at)

    compute_edge_position!(edges.position,edges,cells,xp,yp)
    copyto!(outnc["xEdge"]::NCArrayType{Float64,1},edges.position.x)
    copyto!(outnc["yEdge"]::NCArrayType{Float64,1},edges.position.y)

    dcEdge = compute_dcEdge(edges,cells,xp,yp)
    copyto!(outnc["dcEdge"]::NCArrayType{Float64,1},dcEdge)

    dvEdge = compute_dvEdge(edges,vertices,xp,yp)
    copyto!(outnc["dvEdge"]::NCArrayType{Float64,1},dvEdge)

    compute_angleEdge!(outnc["angleEdge"]::NCArrayType{Float64,1},edges,cells,xp,yp)
    copyto!(outnc["angleEdge"]::NCArrayType{Float64,1}, compute_angleEdge(edges,cells,xp,yp))

    areaCell = compute_area_cells(cells,vertices,xp,yp)
    copyto!(outnc["areaCell"]::NCArrayType{Float64,1},areaCell)

    kite_areas = compute_kite_areas(vertices,cells,xp,yp)
    copyto!(outnc["kiteAreasOnVertex"]::NCArrayType{Float64,2}, reinterpret(reshape,Float64,kite_areas))

    fill!(velRecon.weights,zero(eltype(velRecon.weights)))
    compute_weightsOnEdge_trisk!(velRecon.weights,edges,velRecon,cells,vertices,dcEdge,dvEdge,kite_areas,areaCell)
    copyto!(outnc["weightsOnEdge"]::NCArrayType{Float64,2},CartesianIndices(axes(velRecon.weights)),velRecon.weights,CartesianIndices(axes(velRecon.weights)))

    prepend_string_to_history(outnc,"""julia -e 'using MPASTools; MPASTools.distort_periodic_mesh("$infile",$pert_val)'""")
 
    close(outnc)

    open(in_file_name*"_graph.info","w") do f
        write(f,graph_partition(cells,edges))
    end

    finalfile = in_file_name*"_distorted.nc"
    CondaPkg.withenv() do
        run(`nccopy -6 $outfile $finalfile`)
        run(`rm $outfile`)
    end

    return 0
end

precompile(distort_periodic_mesh,(String,Float64))

function create_distorted_planar_mesh(lx::Number,ly::Number,dc::Number,p::Number,o::String)
    create_planar_hex_mesh(o,lx,ly,dc)
    NCDataset(o,"a") do f
        append_string_to_history(f,string("create_distorted_planar_mesh.jl --dc ",dc," -p ",p," --lx ",lx," --ly ",ly, " -o ",o))
    end
    distort_periodic_mesh(o,p*dc)
    return 0
end

precompile(create_distorted_planar_mesh,(Float64,Float64,Float64,Float64,String))

function parse_commandline_create_planar_mesh(args)
    s = ArgParseSettings(description="This program generates two planar periodic meshes suitable for usage with MPAS, the first one has homogeneous hexagonal elements and the second one is a distorted version of the first.")

    @add_arg_table! s begin
        "--lx"
            help = "Target x direction mesh period in meters (won't be exact unless it is divisable by 'dc'.)"
            arg_type = Float64
            required = true
        "--ly"
            help = "Target y direction mesh period in meters (will not be exact)"
            arg_type = Float64
            required = true
        "--dc"
            help = "Enforced distance between cell centers in meters for the homogeneous mesh"
            arg_type = Float64
            required = true
        "--perturbation", "-p"
            help = "Maximum perturbation ratio 'p' allowed in the distorted mesh regarding 'dc'. That is, maximum allowed perturbation = p*dc"
            arg_type = Float64
            default = 0.18
        "--outFileName", "-o"
            help = "The name for the output file of the homogeneous mesh. If the file name is 'mesh.nc', the distorted mesh will be named 'mesh_distorted.nc'."
            arg_type = String
            default = "mesh.nc"
    end

    return parse_args(args,s)
end

precompile(parse_commandline_create_planar_mesh,(Vector{String},))

function create_distorted_planar_mesh_main(args)
    local_args = isempty(args) ? ["-h"] : args
    parsed_args = parse_commandline_create_planar_mesh(local_args)
    isnothing(parsed_args) && return 0
    lx::Float64 = parsed_args["lx"]
    ly::Float64 = parsed_args["ly"]
    dc::Float64 = parsed_args["dc"]
    p::Float64 = parsed_args["perturbation"]
    if p >= 0.20
        @warn "A perturbation value greater or equal to 0.20 might create a mesh with obtuse triangles" p
    end
    o::String = parsed_args["outFileName"]

    create_distorted_planar_mesh(lx,ly,dc,p,o)
    return 0
end

precompile(create_distorted_planar_mesh_main,(Vector{String},))