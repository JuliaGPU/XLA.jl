abstract type HloOp{Opcode}; end
function opcode(op::HloOp{Opcode}) where {Opcode}
    if Opcode == :reduceWindow
        return "reduce-window"
    end
    String(Opcode)
end

function Base.convert(::Type{Type{<:XRTArray}}, s::Shape)
    XRTArray{convert(Type, XlaType(s.element_type)),
        tuple(s.dimensions...),
        length(s.dimensions)}
end

struct HloTuple <: HloOp{:tuple}
end
fill_fields!(proto::HloInstructionProto, tup::HloTuple) = nothing

struct GenericHloOp{Opcode} <: HloOp{Opcode}
    T::Type
    shape::Tuple
end

struct HloParameter <: HloOp{:parameter}
    T::Type
    shape::Tuple
    id::Int64
end
fill_fields!(proto::HloInstructionProto, p::HloParameter) = proto.parameter_number = p.id

fill_fields!(proto::HloInstructionProto, gen::GenericHloOp) = nothing

struct HloConstant <: HloOp{:constant}
    T::Type
    shape::Tuple
    value::LiteralProto
end
HloConstant(x::XLAScalar) = HloConstant(typeof(x), (), convert(LiteralProto, x))
HloConstant(a::Array{<:XLAScalar}) = HloConstant(eltype(a), size(a), convert(LiteralProto, a))
fill_fields!(proto::HloInstructionProto, c::HloConstant) = proto.literal = c.value

# Equivalent to DotDimensionNumbers, but in a format the compiler has an
# easier time understanding
struct DimNums{N1, N2, N3, N4}
    lhs_contracting_dimensions::NTuple{N1, Int64}
    rhs_contracting_dimensions::NTuple{N2, Int64}
    lhs_batch_dimensions::NTuple{N3, Int64}
    rhs_batch_dimensions::NTuple{N4, Int64}
end

function Base.convert(::Type{DotDimensionNumbers}, dds::DimNums)
    DotDimensionNumbers(
        lhs_contracting_dimensions = collect(dds.lhs_contracting_dimensions),
        rhs_contracting_dimensions = collect(dds.rhs_contracting_dimensions),
        lhs_batch_dimensions = collect(dds.lhs_batch_dimensions),
        rhs_batch_dimensions = collect(dds.rhs_batch_dimensions),
    )
end

struct HloDot <: HloOp{:dot}
    dims::DimNums
end
function fill_fields!(proto::HloInstructionProto, d::HloDot)
    proto.dot_dimension_numbers = convert(DotDimensionNumbers, d.dims)

    # Ask for increased precision
    #proto.precision_config = xla.PrecisionConfig(operand_precision = [2 for _ in proto.operand_ids])
end

# This opcode isn't really real - there's very little point to a map
# unless we're able to compile it
struct HloMap{T} <: HloOp{:map}
end
fill_fields!(proto::HloInstructionProto, s::HloMap) = nothing

struct WindowDims
    size::Int64
    stride::Int64
    padding_low::Int64
    padding_high::Int64
    window_dilation::Int64
    base_dilation::Int64
    window_reversal::Bool
end

struct ConvDimNums{N1, N2, N3}
    input_batch_dimension::Int64
    input_feature_dimension::Int64
    input_spatial_dimensions::NTuple{N1, Int64}
    kernel_input_feature_dimension::Int64
    kernel_output_feature_dimension::Int64
    kernel_spatial_dimensions::NTuple{N2, Int64}
    output_batch_dimension::Int64
    output_feature_dimension::Int64
    output_spatial_dimensions::NTuple{N3, Int64}
end

struct HloConv <: HloOp{:convolution}
    window::NTuple{N, WindowDims} where N
    dims::ConvDimNums
end

xla.WindowDimension(w::WindowDims) = xla.WindowDimension(
    size = w.size,
    stride = w.stride,
    padding_low = w.padding_low,
    padding_high = w.padding_high,
    window_dilation = w.window_dilation,
    base_dilation = w.base_dilation,
    window_reversal = w.window_reversal
)

xla.ConvolutionDimensionNumbers(cdn::ConvDimNums) = xla.ConvolutionDimensionNumbers(
    input_batch_dimension = cdn.input_batch_dimension,
    input_feature_dimension = cdn.input_feature_dimension,
    input_spatial_dimensions = collect(cdn.input_spatial_dimensions),
    kernel_input_feature_dimension = cdn.kernel_input_feature_dimension,
    kernel_output_feature_dimension = cdn.kernel_output_feature_dimension,
    kernel_spatial_dimensions = collect(cdn.kernel_spatial_dimensions),
    output_batch_dimension = cdn.output_batch_dimension,
    output_feature_dimension = cdn.output_feature_dimension,
    output_spatial_dimensions = collect(cdn.output_spatial_dimensions),
)

function fill_fields!(proto::HloInstructionProto, d::HloConv)
    window = xla.Window(dimensions = map(xla.WindowDimension, collect(d.window)))
    proto.window = window
    proto.convolution_dimension_numbers = xla.ConvolutionDimensionNumbers(d.dims)

    # Ask for increased precision
    #proto.precision_config = xla.PrecisionConfig(operand_precision = [2 for _ in proto.operand_ids])
end

struct HloReduceWindow{fT, N} <: HloOp{Symbol("reduce-window")}
    window::NTuple{N, WindowDims}
end
(::Type{HloReduceWindow{fT}})(a::NTuple{N, WindowDims}) where {fT, N} =
    HloReduceWindow{fT, N}(a)
function fill_fields!(proto::HloInstructionProto, d::HloReduceWindow)
    window = xla.Window(dimensions = map(xla.WindowDimension, collect(d.window)))
    proto.window = window
end

struct HloReduce{fT, N} <: HloOp{:reduce}
    dims::NTuple{N, Int64}
end
(::Type{HloReduce{fT}})(dims::NTuple{N, Int64}) where {fT, N} =
    HloReduce{fT, N}(dims)
function fill_fields!(proto::HloInstructionProto, d::HloReduce)
    proto.dimensions = collect(Int64, d.dims)
end

struct HloReshape{N} <: HloOp{:reshape}
    result_shape::NTuple{N, Int}
end
fill_fields!(proto::HloInstructionProto, r::HloReshape) = nothing

struct HloBroadcast{N1, N2} <: HloOp{:broadcast}
    dim_mappings::NTuple{N1, Int}
    result_shape::NTuple{N2, Int}
end
function fill_fields!(proto::HloInstructionProto, r::HloBroadcast)
    mappings = collect(Int64, r.dim_mappings)
    proto.dimensions = mappings
end

struct HloTranspose{N} <: HloOp{:transpose}
    permutation::NTuple{N, Int}
end
function fill_fields!(proto::HloInstructionProto, r::HloTranspose)
    proto.dimensions = collect(Int64, r.permutation)
end

struct HloRev{N} <: HloOp{Symbol("reverse")}
    dims::NTuple{N, Int}
end
function fill_fields!(proto::HloInstructionProto, r::HloRev)
    proto.dimensions = collect(r.dims)
end

struct HloSelectAndScatter{T, S, N} <: HloOp{Symbol("select-and-scatter")}
    window::NTuple{N, WindowDims}
end
(::Type{HloSelectAndScatter{T, S}})(a::NTuple{N, WindowDims}) where {T, S, N} =
    HloSelectAndScatter{T, S, N}(a)

function fill_fields!(proto::HloInstructionProto, r::HloSelectAndScatter)
    window = xla.Window(dimensions = map(xla.WindowDimension, collect(r.window)))
    proto.window = window
end

struct Argument
    shape::Shape
    id::Int64
end

struct HloRng <: HloOp{:rng}
    T::Type
    shape::NTuple{N, Int} where N
    kind::Int
end
function fill_fields!(proto::HloInstructionProto, r::HloRng)
    proto.distribution = Int32(r.kind)
end

struct HloGetTupleElement <: HloOp{Symbol("get-tuple-element")}
    idx::Int
end
function fill_fields!(proto::HloInstructionProto, r::HloGetTupleElement)
    proto.tuple_index = r.idx
end

let global_id = 0
    global make_id
    make_id() = (global_id += 1; global_id)
end

struct HloDynamicUpdateSlice <: HloOp{Symbol("dynamic-update-slice")}
end
fill_fields!(proto::HloInstructionProto, r::HloDynamicUpdateSlice) = nothing

struct HloDynamicSlice{N} <: HloOp{Symbol("dynamic-slice")}
    sizes::NTuple{N, Int64}
end
fill_fields!(proto::HloInstructionProto, r::HloDynamicSlice) = proto.dynamic_slice_sizes = collect(r.sizes)


const SliceDimensions = HloInstructionProto_SliceDimensions
struct SliceDim
    start::Int64
    limit::Int64
    stride::Int64
end
Base.convert(::Type{SliceDimensions}, s::SliceDim) =
    SliceDimensions(start=s.start, limit=s.limit, stride=s.stride)

struct HloSlice{N} <: HloOp{:slice}
    dims::NTuple{N, SliceDim}
end
function fill_fields!(proto::HloInstructionProto, r::HloSlice)
    proto.slice_dimensions = map(x->convert(SliceDimensions, x), collect(r.dims))
end

struct GatherDims{N1, N2, N3}
    offset_dims::NTuple{N1, Int64}
    collapsed_slice_dims::NTuple{N2, Int64}
    start_index_map::NTuple{N3, Int64}
    index_vector_dim::Int64
end

function Base.convert(::Type{GatherDimensionNumbers}, gds::GatherDims)
    GatherDimensionNumbers(
        offset_dims = collect(gds.offset_dims),
        collapsed_slice_dims = collect(gds.collapsed_slice_dims),
        start_index_map = collect(gds.start_index_map),
        index_vector_dim = gds.index_vector_dim
    )
end

struct HloGather{N} <: HloOp{:gather}
    dims::GatherDims
    slice_sizes::NTuple{N, Int64}
end
function fill_fields!(proto::HloInstructionProto, r::HloGather)
    proto.gather_dimension_numbers = convert(GatherDimensionNumbers, r.dims)
    proto.gather_slice_sizes = collect(r.slice_sizes)
end

struct HloConcatenate <: HloOp{Symbol("concatenate")}
    dim::Int64
end
fill_fields!(proto::HloInstructionProto, r::HloConcatenate) = proto.dimensions = Int64[r.dim]

struct HloInfeed{T} <: HloOp{Symbol("infeed")}
    infeed_shape::Type{T}
end
function fill_fields!(proto::HloInstructionProto, r::HloInfeed)
    # TODO: What does this field do? TF doesn't seem to use it
    proto.infeed_config = UInt8[]
end

struct HloOutfeed <: HloOp{Symbol("outfeed")}
end
function fill_fields!(proto::HloInstructionProto, r::HloOutfeed)
    # TODO: What does this field do? TF doesn't seem to use it
    proto.outfeed_config = UInt8[]
end

struct HloAfterAll <: HloOp{Symbol("after-all")}
end
fill_fields!(proto::HloInstructionProto, r::HloAfterAll) = nothing

struct ScatterDimNums{N1, N2, N3}
    update_window_dims::NTuple{N1, Int64}
    inserted_window_dims::NTuple{N2, Int64}
    scatter_dims_to_operand_dims::NTuple{N3, Int64}
    index_vector_dim::Int64
end

function Base.convert(::Type{ScatterDimensionNumbers}, sds::ScatterDimNums)
    ScatterDimensionNumbers(
        update_window_dims = collect(sds.update_window_dims),
        inserted_window_dims = collect(sds.inserted_window_dims),
        scatter_dims_to_operand_dims = collect(sds.scatter_dims_to_operand_dims),
        index_vector_dim = sds.index_vector_dim
    )
end

struct HloScatter{fT} <: HloOp{Symbol("scatter")}
    dims::ScatterDimNums
end
function fill_fields!(proto::HloInstructionProto, r::HloScatter)
    proto.scatter_dimension_numbers = convert(ScatterDimensionNumbers, r.dims)
end

struct HloIota <: HloOp{Symbol("iota")}
    result_type::Type
    result_shape::NTuple{N, Int64} where N
    iota_dimension::Int64
end
function fill_fields!(proto::HloInstructionProto, r::HloIota)
    proto.dimensions = Int[r.iota_dimension]
end

const PaddingConfigDimension = PaddingConfig_PaddingConfigDimension
struct PaddingConfigDim
    edge_padding_low::Int64
    edge_padding_high::Int64
    interior_padding::Int64
end
Base.convert(::Type{PaddingConfigDimension}, p::PaddingConfigDim) =
    PaddingConfigDimension(
        edge_padding_low = p.edge_padding_low,
        edge_padding_high = p.edge_padding_high,
        interior_padding = p.interior_padding)

struct HloPad <: HloOp{:pad}
    padding::NTuple{N, PaddingConfigDim} where {N}
end
function fill_fields!(proto::HloInstructionProto, r::HloPad)
    proto.padding_config = PaddingConfig(dimensions=[convert(PaddingConfigDimension, p) for p in r.padding])
end

struct RepGroup{N}
    replica_ids::NTuple{N, Int64}
end
Base.convert(::Type{ReplicaGroup}, x::RepGroup) = ReplicaGroup(replica_ids=collect(x.replica_ids))

struct HloCrossReplicaSum{fT} <: HloOp{Symbol("cross-replica-sum")}
    replica_groups::NTuple{N, RepGroup} where N
    all_reduce_id::Int
    barrier::String
end
function fill_fields!(proto::HloInstructionProto, r::HloCrossReplicaSum)
    proto.replica_groups = ReplicaGroup[convert(ReplicaGroup, x) for x in r.replica_groups]
    proto.all_reduce_id = r.all_reduce_id
    proto.cross_replica_sum_barrier = r.barrier
end

function HloInstructionProto(comp::HloComputationProto, opcode::String; id=make_id(), name=nothing)
    proto = HloInstructionProto(
        opcode=opcode,
        id=id,
        name=something(name, "comp$(comp.id)_$(opcode)$id")
    )
    push!(comp.instructions, proto)
    proto
end

function shape_from_operands(op::HloOp, operands...)
    if isa(op, HloGetTupleElement)
        xshape = operands[1].shape.tuple_shapes[op.idx+1]
    elseif isa(op, HloTuple)
        xshape = Shape(
            element_type = xla.PrimitiveType.TUPLE,
            tuple_shapes = collect(op.shape for op in operands)
        )
    elseif isa(op, Union{HloMap, HloReduce, HloReduceWindow, HloScatter, HloCrossReplicaSum})
        T, shape = shape_infer(op,
            typeof(op).parameters[1],
            map(operands) do op
                convert(Type{<:XRTArray}, op.shape)
            end...
        )
        xshape = Shape(T, shape)
    elseif isa(op, Union{HloSelectAndScatter})
        T, shape = shape_infer(op,
            typeof(op).parameters[1],
            typeof(op).parameters[2],
            map(operands) do op
                convert(Type{<:XRTArray}, op.shape)
            end...
        )
        xshape = Shape(T, shape)
    elseif isa(op, GenericHloOp)
        xshape = Shape(op.T, op.shape)
    else
        T, shape = shape_infer(op,
            map(operands) do op
                convert(Type{<:XRTArray}, op.shape)
            end...
        )
        xshape = Shape(T, shape)
    end
    return xshape
end

function HloInstructionProto(op::HloOp, @nospecialize(operands::Union{Argument, HloInstructionProto}...); id=make_id(), name="$(opcode(op))$id",
        shape=shape_from_operands(op, operands...), kwargs...)
    proto = HloInstructionProto(
        opcode = opcode(op),
        shape = shape,
        id = id,
        name = name,
        operand_ids = collect(map(x->x.id, operands)),
        kwargs...
    )
    fill_fields!(proto, op)
    proto
end

function HloInstructionProto(comp::HloComputationProto, op::HloOp, @nospecialize(args...); id=make_id(), name=nothing, shape=shape_from_operands(op, args...))
    proto = HloInstructionProto(op, args...; id=id, name=something(name, "comp$(comp.id)_$(opcode(op))$id"), shape=shape)
    push!(comp.instructions, proto)
    proto
end
