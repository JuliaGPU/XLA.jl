# layout

XLA_Layout() = XLA_Layout(0, Int64List(), TileList(), 0, 0)

function Base.convert(::Type{XLA_Layout}, layout::Layout)
    fields = layout.__protobuf_jl_internal_values
    XLA_Layout(layout.format,
        haskey(fields, :minor_to_major) ? layout.minor_to_major : Int64List(),
        haskey(fields, :tiles) ? XLA_Tile[layout.tiles...] : TileList(),
        haskey(fields, :element_size_in_bits) ? layout.element_size_in_bits : 0,
        haskey(fields, :memory_space) ? layout.memory_space : 0)
end

function Base.convert(::Type{Layout}, layout::XLA_Layout)
    fields = Pair[:format => layout.format]
    layout.minor_to_major.size != 0  && push!(fields, :minor_to_major => layout.minor_to_major)
    layout.tiles.size != 0           && push!(fields, :tiles => layout.tiles)
    layout.element_size_in_bits != 0 && push!(fields, :element_size_in_bits => layout.element_size_in_bits)
    layout.memory_space != 0         && push!(fields, :memory_space => layout.memory_space)
    Layout(; fields...)
end


# shape

XLA_Shape() = XLA_Shape(0, Int64List(), BoolList(), (), 0, XLA_Layout())

function Base.convert(::Type{XLA_Shape}, shape::Shape)
    fields = shape.__protobuf_jl_internal_values
    tuple_shapes = ()
    if haskey(fields, :tuple_shapes) && length(shape.tuple_shapes) !== 0
        tuple_shapes = tuple(XLA_Shape[shape.tuple_shapes...]...)
    end
    XLA_Shape(shape.element_type,
        haskey(fields, :dimensions) ? shape.dimensions : Int64List(),
        haskey(fields, :is_dynamic_dimension) ? shape.is_dynamic_dimension : BoolList(),
        tuple_shapes, length(tuple_shapes),
        haskey(fields, :layout) ? shape.layout : XLA_Layout()
    )
end

function Base.convert(::Type{Shape}, shape::XLA_Shape)
    fields = Pair[:element_type => shape.element_type]
    shape.dimensions.size != 0         && push!(fields, :dimensions => shape.dimensions)
    shape.dynamic_dimensions.size != 0 && push!(fields, :is_dynamic_dimension => shape.dynamic_dimensions)
    shape.tuple_shapes != ()           && push!(fields, :tuple_shapes => shape.tuple_shapes)
    shape.layout != XLA_Layout()       && push!(fields, :layout => shape.layout)
    Shape(; fields...)
end


# computation layout

function Base.convert(::Type{XLA_ComputationLayout}, pshape::ProgramShape)
    parameters = nothing
    if length(pshape.parameters) != 0
        parameters = map(pshape.parameters) do shape
            convert(XLA_Shape, shape)
        end
    end
    XLA_ComputationLayout(
        parameters === nothing ? 0 : length(parameters),
        parameters === nothing ? C_NULL : pointer(parameters),
        pshape.result,
        parameters
    )
end


# tile

function Base.convert(::Type{XLA_Tile}, tile::Tile)
    XLA_Tile(tile.dimensions)
end

function Base.convert(::Type{Tile}, tile::XLA_Tile)
    Tile(; dimensions = tile.dimensions)
end

Base.zero(::Type{XLA_Tile}) = XLA_Tile(Int64List())


# other

XLA_ShapedBuffer() = XLA_ShapedBuffer(XLA_Shape(), 0, C_NULL, 0)
