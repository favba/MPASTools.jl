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

function create_planar_hex_mesh(filename::AbstractString,lx::Number,ly::Number,dc::Number)
    nx = Int(lx÷dc)
    ny = Int((2ly)÷(√3*dc))
    ny+=mod(ny,2)
    temp_o = splitext(filename)[1]*"_hdf5.nc"
    CondaPkg.withenv() do
        run(`planar_hex --nx $nx --ny $ny --dc $dc -o $temp_o`)
        run(`nccopy -6 $temp_o $filename`)
        run(`rm $temp_o`)
    end
end

precompile(create_planar_hex_mesh,(String,Float64,Float64,Float64,Float64))

function distort_periodic_mesh(infile::AbstractString,pert_val::Number)

    outfile = splitext(infile)[1]*"_distorted_incomplete.nc"
    innc = NCDataset(infile)
    outnc = NCDataset(outfile,"c")
    
    for (dimname, val) in innc.dim
        defDim(outnc,dimname,val)
    end
    
    for (attname, val) in innc.attrib
        outnc.attrib[attname] = val
    end
    
    cells = CellBase(innc)
    vertices = VertexBase(innc)
    new_cells_pos = perturb_points(pert_val,cells.position)
    new_vert_pos = compute_new_circumcenters_periodic(new_cells_pos,vertices.position,vertices.indices.cells,innc.attrib["x_period"],innc.attrib["y_period"])

    new_xCell = defVar(outnc,innc["xCell"])
    copy!(new_xCell,new_cells_pos.x)
    new_yCell = defVar(outnc,innc["yCell"])
    copy!(new_yCell,new_cells_pos.y)
    defVar(outnc,innc["zCell"])

    new_xVertex = defVar(outnc,innc["xVertex"])
    copy!(new_xVertex,new_vert_pos.x)
    new_yVertex = defVar(outnc,innc["yVertex"])
    copy!(new_yVertex,new_vert_pos.y)
    defVar(outnc,innc["zVertex"])

    defVar(outnc,innc["cellsOnVertex"])
    defVar(outnc,innc["meshDensity"])

    close(innc)
    close(outnc)

    finalfilehdf5 = splitext(infile)[1]*"_distorted_hdf5.nc"
    finalfile = splitext(infile)[1]*"_distorted.nc"
    CondaPkg.withenv() do
        run(`MpasMeshConverter.x $outfile $finalfilehdf5`)
        run(`nccopy -6 $finalfilehdf5 $finalfile`)
        run(`rm $outfile $finalfilehdf5`)
    end
end

precompile(distort_periodic_mesh,(String,Float64))


function create_distorted_planar_mesh(lx::Number,ly::Number,dc::Number,p::Number,o::String)
    create_planar_hex_mesh(o,lx,ly,dc)
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