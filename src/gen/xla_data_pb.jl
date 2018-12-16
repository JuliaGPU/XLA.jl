# syntax: proto3
using ProtoBuf
import ProtoBuf.meta

struct __enum_PrimitiveType <: ProtoEnum
    PRIMITIVE_TYPE_INVALID::Int32
    PRED::Int32
    S8::Int32
    S16::Int32
    S32::Int32
    S64::Int32
    U8::Int32
    U16::Int32
    U32::Int32
    U64::Int32
    F16::Int32
    F32::Int32
    BF16::Int32
    F64::Int32
    C64::Int32
    TUPLE::Int32
    OPAQUE::Int32
    TOKEN::Int32
    __enum_PrimitiveType() = new(0,1,2,3,4,5,6,7,8,9,10,11,16,12,15,13,14,17)
end #struct __enum_PrimitiveType
const PrimitiveType = __enum_PrimitiveType()

struct __enum_Format <: ProtoEnum
    INVALID_FORMAT::Int32
    DENSE::Int32
    SPARSE::Int32
    __enum_Format() = new(0,1,2)
end #struct __enum_Format
const Format = __enum_Format()

struct __enum_FftType <: ProtoEnum
    FFT::Int32
    IFFT::Int32
    RFFT::Int32
    IRFFT::Int32
    __enum_FftType() = new(0,1,2,3)
end #struct __enum_FftType
const FftType = __enum_FftType()

struct __enum_RandomDistribution <: ProtoEnum
    RNG_INVALID::Int32
    RNG_UNIFORM::Int32
    RNG_NORMAL::Int32
    __enum_RandomDistribution() = new(0,1,2)
end #struct __enum_RandomDistribution
const RandomDistribution = __enum_RandomDistribution()

mutable struct PaddingConfig_PaddingConfigDimension <: ProtoType
    edge_padding_low::Int64
    edge_padding_high::Int64
    interior_padding::Int64
    PaddingConfig_PaddingConfigDimension(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct PaddingConfig_PaddingConfigDimension

mutable struct PaddingConfig <: ProtoType
    dimensions::Base.Vector{PaddingConfig_PaddingConfigDimension}
    PaddingConfig(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct PaddingConfig

mutable struct Tile <: ProtoType
    dimensions::Base.Vector{Int64}
    Tile(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct Tile
const __pack_Tile = Symbol[:dimensions]
meta(t::Type{Tile}) = meta(t, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, true, __pack_Tile, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES, ProtoBuf.DEF_FIELD_TYPES)

mutable struct Layout <: ProtoType
    format::Int32
    minor_to_major::Base.Vector{Int64}
    max_sparse_elements::Int64
    tiles::Base.Vector{Tile}
    element_size_in_bits::Int64
    Layout(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct Layout
const __fnum_Layout = Int[4,1,5,6,7]
const __pack_Layout = Symbol[:minor_to_major]
meta(t::Type{Layout}) = meta(t, ProtoBuf.DEF_REQ, __fnum_Layout, ProtoBuf.DEF_VAL, true, __pack_Layout, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES, ProtoBuf.DEF_FIELD_TYPES)

mutable struct ShapeProto <: ProtoType
    element_type::Int32
    dimensions::Base.Vector{Int64}
    tuple_shapes::Base.Vector{ShapeProto}
    layout::Layout
    ShapeProto(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct ShapeProto
const __fnum_ShapeProto = Int[2,3,4,5]
const __pack_ShapeProto = Symbol[:dimensions]
meta(t::Type{ShapeProto}) = meta(t, ProtoBuf.DEF_REQ, __fnum_ShapeProto, ProtoBuf.DEF_VAL, true, __pack_ShapeProto, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES, ProtoBuf.DEF_FIELD_TYPES)

mutable struct ProgramShapeProto <: ProtoType
    parameters::Base.Vector{ShapeProto}
    result::ShapeProto
    parameter_names::Base.Vector{AbstractString}
    ProgramShapeProto(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct ProgramShapeProto

mutable struct ComputationStats <: ProtoType
    flop_count::Float64
    transcendental_count::Float64
    ComputationStats(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct ComputationStats

mutable struct OpMetadata <: ProtoType
    op_type::AbstractString
    op_name::AbstractString
    source_file::AbstractString
    source_line::Int32
    OpMetadata(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct OpMetadata

mutable struct ExecutionProfile <: ProtoType
    compilation_cache_hit::Bool
    compile_time_ms::Int64
    compute_cycle_count::Int64
    compute_time_ns::Int64
    compute_and_transfer_time_ns::Int64
    executable_size_in_bytes::Int64
    ExecutionProfile(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct ExecutionProfile

mutable struct ExecutionHandle <: ProtoType
    handle::Int64
    ExecutionHandle(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct ExecutionHandle

mutable struct GlobalDataHandle <: ProtoType
    handle::Int64
    GlobalDataHandle(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct GlobalDataHandle

mutable struct DeviceHandle <: ProtoType
    handle::Int64
    device_count::Int64
    DeviceHandle(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct DeviceHandle

struct __enum_ChannelHandle_ChannelType <: ProtoEnum
    CHANNEL_TYPE_INVALID::Int32
    DEVICE_TO_DEVICE::Int32
    DEVICE_TO_HOST::Int32
    HOST_TO_DEVICE::Int32
    __enum_ChannelHandle_ChannelType() = new(0,1,2,3)
end #struct __enum_ChannelHandle_ChannelType
const ChannelHandle_ChannelType = __enum_ChannelHandle_ChannelType()

mutable struct ChannelHandle <: ProtoType
    handle::Int64
    _type::Int32
    ChannelHandle(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct ChannelHandle

mutable struct DeviceAssignmentProto_ComputationDevice <: ProtoType
    replica_device_ids::Base.Vector{Int32}
    DeviceAssignmentProto_ComputationDevice(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct DeviceAssignmentProto_ComputationDevice
const __pack_DeviceAssignmentProto_ComputationDevice = Symbol[:replica_device_ids]
meta(t::Type{DeviceAssignmentProto_ComputationDevice}) = meta(t, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, true, __pack_DeviceAssignmentProto_ComputationDevice, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES, ProtoBuf.DEF_FIELD_TYPES)

mutable struct DeviceAssignmentProto <: ProtoType
    replica_count::Int32
    computation_count::Int32
    computation_devices::Base.Vector{DeviceAssignmentProto_ComputationDevice}
    DeviceAssignmentProto(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct DeviceAssignmentProto

mutable struct LiteralProto <: ProtoType
    shape::ShapeProto
    preds::Base.Vector{Bool}
    s8s::Array{UInt8,1}
    u8s::Array{UInt8,1}
    s32s::Base.Vector{Int32}
    s64s::Base.Vector{Int64}
    u32s::Base.Vector{UInt32}
    u64s::Base.Vector{UInt64}
    f32s::Base.Vector{Float32}
    f64s::Base.Vector{Float64}
    c64s::Base.Vector{Float32}
    tuple_literals::Base.Vector{LiteralProto}
    f16s::Array{UInt8,1}
    bf16s::Array{UInt8,1}
    u16s::Array{UInt8,1}
    s16s::Array{UInt8,1}
    sparse_indices::Base.Vector{Int64}
    LiteralProto(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct LiteralProto
const __fnum_LiteralProto = Int[1,2,15,3,4,5,6,7,8,9,12,10,11,13,16,17,14]
const __pack_LiteralProto = Symbol[:preds,:s32s,:s64s,:u32s,:u64s,:f32s,:f64s,:c64s,:sparse_indices]
meta(t::Type{LiteralProto}) = meta(t, ProtoBuf.DEF_REQ, __fnum_LiteralProto, ProtoBuf.DEF_VAL, true, __pack_LiteralProto, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES, ProtoBuf.DEF_FIELD_TYPES)

mutable struct WindowDimension <: ProtoType
    size::Int64
    stride::Int64
    padding_low::Int64
    padding_high::Int64
    window_dilation::Int64
    base_dilation::Int64
    window_reversal::Bool
    WindowDimension(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct WindowDimension

mutable struct Window <: ProtoType
    dimensions::Base.Vector{WindowDimension}
    Window(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct Window

mutable struct GatherDimensionNumbers <: ProtoType
    offset_dims::Base.Vector{Int64}
    collapsed_slice_dims::Base.Vector{Int64}
    start_index_map::Base.Vector{Int64}
    index_vector_dim::Int64
    GatherDimensionNumbers(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct GatherDimensionNumbers
const __pack_GatherDimensionNumbers = Symbol[:offset_dims,:collapsed_slice_dims,:start_index_map]
meta(t::Type{GatherDimensionNumbers}) = meta(t, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, true, __pack_GatherDimensionNumbers, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES, ProtoBuf.DEF_FIELD_TYPES)

mutable struct ScatterDimensionNumbers <: ProtoType
    update_window_dims::Base.Vector{Int64}
    inserted_window_dims::Base.Vector{Int64}
    scatter_dims_to_operand_dims::Base.Vector{Int64}
    index_vector_dim::Int64
    ScatterDimensionNumbers(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct ScatterDimensionNumbers
const __pack_ScatterDimensionNumbers = Symbol[:update_window_dims,:inserted_window_dims,:scatter_dims_to_operand_dims]
meta(t::Type{ScatterDimensionNumbers}) = meta(t, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, true, __pack_ScatterDimensionNumbers, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES, ProtoBuf.DEF_FIELD_TYPES)

mutable struct ConvolutionDimensionNumbers <: ProtoType
    input_batch_dimension::Int64
    input_feature_dimension::Int64
    input_spatial_dimensions::Base.Vector{Int64}
    kernel_input_feature_dimension::Int64
    kernel_output_feature_dimension::Int64
    kernel_spatial_dimensions::Base.Vector{Int64}
    output_batch_dimension::Int64
    output_feature_dimension::Int64
    output_spatial_dimensions::Base.Vector{Int64}
    ConvolutionDimensionNumbers(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct ConvolutionDimensionNumbers
const __fnum_ConvolutionDimensionNumbers = Int[7,8,11,3,4,6,9,10,12]
const __pack_ConvolutionDimensionNumbers = Symbol[:input_spatial_dimensions,:kernel_spatial_dimensions,:output_spatial_dimensions]
meta(t::Type{ConvolutionDimensionNumbers}) = meta(t, ProtoBuf.DEF_REQ, __fnum_ConvolutionDimensionNumbers, ProtoBuf.DEF_VAL, true, __pack_ConvolutionDimensionNumbers, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES, ProtoBuf.DEF_FIELD_TYPES)

mutable struct DotDimensionNumbers <: ProtoType
    lhs_contracting_dimensions::Base.Vector{Int64}
    rhs_contracting_dimensions::Base.Vector{Int64}
    lhs_batch_dimensions::Base.Vector{Int64}
    rhs_batch_dimensions::Base.Vector{Int64}
    DotDimensionNumbers(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct DotDimensionNumbers
const __pack_DotDimensionNumbers = Symbol[:lhs_contracting_dimensions,:rhs_contracting_dimensions,:lhs_batch_dimensions,:rhs_batch_dimensions]
meta(t::Type{DotDimensionNumbers}) = meta(t, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, true, __pack_DotDimensionNumbers, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES, ProtoBuf.DEF_FIELD_TYPES)

struct __enum_OpSharding_Type <: ProtoEnum
    REPLICATED::Int32
    MAXIMAL::Int32
    TUPLE::Int32
    OTHER::Int32
    __enum_OpSharding_Type() = new(0,1,2,3)
end #struct __enum_OpSharding_Type
const OpSharding_Type = __enum_OpSharding_Type()

mutable struct OpSharding <: ProtoType
    _type::Int32
    tile_shape::ShapeProto
    tile_assignment_dimensions::Base.Vector{Int64}
    tile_assignment_devices::Base.Vector{Int64}
    tuple_shardings::Base.Vector{OpSharding}
    OpSharding(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct OpSharding
const __pack_OpSharding = Symbol[:tile_assignment_dimensions,:tile_assignment_devices]
meta(t::Type{OpSharding}) = meta(t, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, true, __pack_OpSharding, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES, ProtoBuf.DEF_FIELD_TYPES)

mutable struct ReplicaGroup <: ProtoType
    replica_ids::Base.Vector{Int64}
    ReplicaGroup(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct ReplicaGroup
const __pack_ReplicaGroup = Symbol[:replica_ids]
meta(t::Type{ReplicaGroup}) = meta(t, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, true, __pack_ReplicaGroup, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES, ProtoBuf.DEF_FIELD_TYPES)

mutable struct SourceTarget <: ProtoType
    source::Int64
    target::Int64
    SourceTarget(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct SourceTarget

struct __enum_PrecisionConfig_Precision <: ProtoEnum
    DEFAULT::Int32
    HIGH::Int32
    HIGHEST::Int32
    __enum_PrecisionConfig_Precision() = new(0,1,2)
end #struct __enum_PrecisionConfig_Precision
const PrecisionConfig_Precision = __enum_PrecisionConfig_Precision()

mutable struct PrecisionConfig <: ProtoType
    operand_precision::Base.Vector{Int32}
    PrecisionConfig(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct PrecisionConfig
const __pack_PrecisionConfig = Symbol[:operand_precision]
meta(t::Type{PrecisionConfig}) = meta(t, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, true, __pack_PrecisionConfig, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES, ProtoBuf.DEF_FIELD_TYPES)

export PrimitiveType, Format, FftType, RandomDistribution, PaddingConfig_PaddingConfigDimension, PaddingConfig, Tile, Layout, ShapeProto, ProgramShapeProto, ComputationStats, OpMetadata, ExecutionProfile, ExecutionHandle, GlobalDataHandle, DeviceHandle, ChannelHandle_ChannelType, ChannelHandle, DeviceAssignmentProto_ComputationDevice, DeviceAssignmentProto, LiteralProto, WindowDimension, Window, GatherDimensionNumbers, ScatterDimensionNumbers, ConvolutionDimensionNumbers, DotDimensionNumbers, OpSharding_Type, OpSharding, ReplicaGroup, SourceTarget, PrecisionConfig_Precision, PrecisionConfig
