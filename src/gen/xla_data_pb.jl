# syntax: proto3
using Compat
using ProtoBuf
import ProtoBuf.meta
import Base: hash, isequal, ==

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
    __enum_PrimitiveType() = new(0,1,2,3,4,5,6,7,8,9,10,11,16,12,15,13,14)
end #struct __enum_PrimitiveType
const PrimitiveType = __enum_PrimitiveType()

struct __enum_PaddingValue <: ProtoEnum
    INVALID_PAD::Int32
    ZERO_PAD::Int32
    ONE_PAD::Int32
    LOWEST_PAD::Int32
    HIGHEST_PAD::Int32
    UNKNOWN_PAD::Int32
    __enum_PaddingValue() = new(0,1,2,3,4,5)
end #struct __enum_PaddingValue
const PaddingValue = __enum_PaddingValue()

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

struct __enum_UnaryOperation <: ProtoEnum
    UNOP_INVALID::Int32
    UNOP_NOT::Int32
    UNOP_EXP::Int32
    UNOP_NEGATE::Int32
    UNOP_SORT::Int32
    UNOP_TANH::Int32
    UNOP_LOG::Int32
    UNOP_FLOOR::Int32
    UNOP_CEIL::Int32
    UNOP_ABS::Int32
    UNOP_SIGN::Int32
    UNOP_IS_FINITE::Int32
    UNOP_COS::Int32
    UNOP_SIN::Int32
    UNOP_ROUND_NEAREST_AFZ::Int32
    UNOP_REAL::Int32
    UNOP_IMAG::Int32
    __enum_UnaryOperation() = new(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16)
end #struct __enum_UnaryOperation
const UnaryOperation = __enum_UnaryOperation()

struct __enum_BinaryOperation <: ProtoEnum
    BINOP_INVALID::Int32
    BINOP_ADD::Int32
    BINOP_DIV::Int32
    BINOP_MUL::Int32
    BINOP_SUB::Int32
    BINOP_EQ::Int32
    BINOP_GE::Int32
    BINOP_GT::Int32
    BINOP_LE::Int32
    BINOP_LT::Int32
    BINOP_NE::Int32
    BINOP_MAX::Int32
    BINOP_MIN::Int32
    BINOP_POW::Int32
    BINOP_REM::Int32
    BINOP_AND::Int32
    BINOP_OR::Int32
    BINOP_SHIFT_LEFT::Int32
    BINOP_SHIFT_RIGHT_ARITHMETIC::Int32
    BINOP_SHIFT_RIGHT_LOGICAL::Int32
    BINOP_COMPLEX::Int32
    BINOP_ATAN2::Int32
    __enum_BinaryOperation() = new(0,1,2,3,4,5,6,7,8,9,10,14,15,16,17,18,19,20,21,22,23,24)
end #struct __enum_BinaryOperation
const BinaryOperation = __enum_BinaryOperation()

struct __enum_RandomDistribution <: ProtoEnum
    RNG_INVALID::Int32
    RNG_UNIFORM::Int32
    RNG_NORMAL::Int32
    __enum_RandomDistribution() = new(0,1,2)
end #struct __enum_RandomDistribution
const RandomDistribution = __enum_RandomDistribution()

struct __enum_TernaryOperation <: ProtoEnum
    TRIOP_INVALID::Int32
    TRIOP_SELECT::Int32
    TRIOP_CLAMP::Int32
    __enum_TernaryOperation() = new(0,1,3)
end #struct __enum_TernaryOperation
const TernaryOperation = __enum_TernaryOperation()

struct __enum_VariadicOperation <: ProtoEnum
    VAROP_INVALID::Int32
    VAROP_TUPLE::Int32
    __enum_VariadicOperation() = new(0,1)
end #struct __enum_VariadicOperation
const VariadicOperation = __enum_VariadicOperation()

mutable struct PaddingConfig_PaddingConfigDimension <: ProtoType
    edge_padding_low::Int64
    edge_padding_high::Int64
    interior_padding::Int64
    PaddingConfig_PaddingConfigDimension(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct PaddingConfig_PaddingConfigDimension
hash(v::PaddingConfig_PaddingConfigDimension) = ProtoBuf.protohash(v)
isequal(v1::PaddingConfig_PaddingConfigDimension, v2::PaddingConfig_PaddingConfigDimension) = ProtoBuf.protoisequal(v1, v2)
==(v1::PaddingConfig_PaddingConfigDimension, v2::PaddingConfig_PaddingConfigDimension) = ProtoBuf.protoeq(v1, v2)

mutable struct PaddingConfig <: ProtoType
    dimensions::Vector{PaddingConfig_PaddingConfigDimension}
    PaddingConfig(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct PaddingConfig
hash(v::PaddingConfig) = ProtoBuf.protohash(v)
isequal(v1::PaddingConfig, v2::PaddingConfig) = ProtoBuf.protoisequal(v1, v2)
==(v1::PaddingConfig, v2::PaddingConfig) = ProtoBuf.protoeq(v1, v2)

mutable struct Layout <: ProtoType
    format::Int32
    minor_to_major::Vector{Int64}
    padded_dimensions::Vector{Int64}
    padding_value::Int32
    max_sparse_elements::Int64
    Layout(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct Layout
const __fnum_Layout = Int[4,1,2,3,5]
const __pack_Layout = Symbol[:minor_to_major,:padded_dimensions]
meta(t::Type{Layout}) = meta(t, ProtoBuf.DEF_REQ, __fnum_Layout, ProtoBuf.DEF_VAL, true, __pack_Layout, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES, ProtoBuf.DEF_FIELD_TYPES)
hash(v::Layout) = ProtoBuf.protohash(v)
isequal(v1::Layout, v2::Layout) = ProtoBuf.protoisequal(v1, v2)
==(v1::Layout, v2::Layout) = ProtoBuf.protoeq(v1, v2)

mutable struct Shape <: ProtoType
    element_type::Int32
    dimensions::Vector{Int64}
    tuple_shapes::Vector{Shape}
    layout::Layout
    Shape(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct Shape
const __fnum_Shape = Int[2,3,4,5]
const __pack_Shape = Symbol[:dimensions]
meta(t::Type{Shape}) = meta(t, ProtoBuf.DEF_REQ, __fnum_Shape, ProtoBuf.DEF_VAL, true, __pack_Shape, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES, ProtoBuf.DEF_FIELD_TYPES)
hash(v::Shape) = ProtoBuf.protohash(v)
isequal(v1::Shape, v2::Shape) = ProtoBuf.protoisequal(v1, v2)
==(v1::Shape, v2::Shape) = ProtoBuf.protoeq(v1, v2)

mutable struct ProgramShape <: ProtoType
    parameters::Vector{Shape}
    result::Shape
    parameter_names::Vector{AbstractString}
    ProgramShape(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct ProgramShape
hash(v::ProgramShape) = ProtoBuf.protohash(v)
isequal(v1::ProgramShape, v2::ProgramShape) = ProtoBuf.protoisequal(v1, v2)
==(v1::ProgramShape, v2::ProgramShape) = ProtoBuf.protoeq(v1, v2)

mutable struct ComputationStats <: ProtoType
    flop_count::Float64
    transcendental_count::Float64
    ComputationStats(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct ComputationStats
hash(v::ComputationStats) = ProtoBuf.protohash(v)
isequal(v1::ComputationStats, v2::ComputationStats) = ProtoBuf.protoisequal(v1, v2)
==(v1::ComputationStats, v2::ComputationStats) = ProtoBuf.protoeq(v1, v2)

mutable struct OpMetadata <: ProtoType
    op_type::AbstractString
    op_name::AbstractString
    source_file::AbstractString
    source_line::Int32
    OpMetadata(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct OpMetadata
hash(v::OpMetadata) = ProtoBuf.protohash(v)
isequal(v1::OpMetadata, v2::OpMetadata) = ProtoBuf.protoisequal(v1, v2)
==(v1::OpMetadata, v2::OpMetadata) = ProtoBuf.protoeq(v1, v2)

mutable struct ExecutionProfile <: ProtoType
    compilation_cache_hit::Bool
    compile_time_ms::Int64
    compute_cycle_count::Int64
    compute_time_ns::Int64
    compute_and_transfer_time_ns::Int64
    ExecutionProfile(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct ExecutionProfile
hash(v::ExecutionProfile) = ProtoBuf.protohash(v)
isequal(v1::ExecutionProfile, v2::ExecutionProfile) = ProtoBuf.protoisequal(v1, v2)
==(v1::ExecutionProfile, v2::ExecutionProfile) = ProtoBuf.protoeq(v1, v2)

mutable struct ComputationHandle <: ProtoType
    handle::Int64
    ComputationHandle(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct ComputationHandle
hash(v::ComputationHandle) = ProtoBuf.protohash(v)
isequal(v1::ComputationHandle, v2::ComputationHandle) = ProtoBuf.protoisequal(v1, v2)
==(v1::ComputationHandle, v2::ComputationHandle) = ProtoBuf.protoeq(v1, v2)

mutable struct ExecutionHandle <: ProtoType
    handle::Int64
    ExecutionHandle(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct ExecutionHandle
hash(v::ExecutionHandle) = ProtoBuf.protohash(v)
isequal(v1::ExecutionHandle, v2::ExecutionHandle) = ProtoBuf.protoisequal(v1, v2)
==(v1::ExecutionHandle, v2::ExecutionHandle) = ProtoBuf.protoeq(v1, v2)

mutable struct GlobalDataHandle <: ProtoType
    handle::Int64
    GlobalDataHandle(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct GlobalDataHandle
hash(v::GlobalDataHandle) = ProtoBuf.protohash(v)
isequal(v1::GlobalDataHandle, v2::GlobalDataHandle) = ProtoBuf.protoisequal(v1, v2)
==(v1::GlobalDataHandle, v2::GlobalDataHandle) = ProtoBuf.protoeq(v1, v2)

mutable struct ComputationDataHandle <: ProtoType
    handle::Int64
    ComputationDataHandle(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct ComputationDataHandle
hash(v::ComputationDataHandle) = ProtoBuf.protohash(v)
isequal(v1::ComputationDataHandle, v2::ComputationDataHandle) = ProtoBuf.protoisequal(v1, v2)
==(v1::ComputationDataHandle, v2::ComputationDataHandle) = ProtoBuf.protoeq(v1, v2)

mutable struct DeviceHandle <: ProtoType
    handle::Int64
    device_count::Int64
    DeviceHandle(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct DeviceHandle
hash(v::DeviceHandle) = ProtoBuf.protohash(v)
isequal(v1::DeviceHandle, v2::DeviceHandle) = ProtoBuf.protoisequal(v1, v2)
==(v1::DeviceHandle, v2::DeviceHandle) = ProtoBuf.protoeq(v1, v2)

mutable struct ChannelHandle <: ProtoType
    handle::Int64
    ChannelHandle(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct ChannelHandle
hash(v::ChannelHandle) = ProtoBuf.protohash(v)
isequal(v1::ChannelHandle, v2::ChannelHandle) = ProtoBuf.protoisequal(v1, v2)
==(v1::ChannelHandle, v2::ChannelHandle) = ProtoBuf.protoeq(v1, v2)

mutable struct DeviceAssignmentProto_ComputationDevice <: ProtoType
    replica_device_ids::Vector{Int32}
    DeviceAssignmentProto_ComputationDevice(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct DeviceAssignmentProto_ComputationDevice
const __pack_DeviceAssignmentProto_ComputationDevice = Symbol[:replica_device_ids]
meta(t::Type{DeviceAssignmentProto_ComputationDevice}) = meta(t, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, true, __pack_DeviceAssignmentProto_ComputationDevice, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES, ProtoBuf.DEF_FIELD_TYPES)
hash(v::DeviceAssignmentProto_ComputationDevice) = ProtoBuf.protohash(v)
isequal(v1::DeviceAssignmentProto_ComputationDevice, v2::DeviceAssignmentProto_ComputationDevice) = ProtoBuf.protoisequal(v1, v2)
==(v1::DeviceAssignmentProto_ComputationDevice, v2::DeviceAssignmentProto_ComputationDevice) = ProtoBuf.protoeq(v1, v2)

mutable struct DeviceAssignmentProto <: ProtoType
    replica_count::Int32
    computation_count::Int32
    computation_devices::Vector{DeviceAssignmentProto_ComputationDevice}
    DeviceAssignmentProto(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct DeviceAssignmentProto
hash(v::DeviceAssignmentProto) = ProtoBuf.protohash(v)
isequal(v1::DeviceAssignmentProto, v2::DeviceAssignmentProto) = ProtoBuf.protoisequal(v1, v2)
==(v1::DeviceAssignmentProto, v2::DeviceAssignmentProto) = ProtoBuf.protoeq(v1, v2)

mutable struct LiteralProto <: ProtoType
    shape::Shape
    preds::Vector{Bool}
    u8s::Array{UInt8,1}
    s32s::Vector{Int32}
    s64s::Vector{Int64}
    u32s::Vector{UInt32}
    u64s::Vector{UInt64}
    f32s::Vector{Float32}
    f64s::Vector{Float64}
    c64s::Vector{Float32}
    tuple_literals::Vector{LiteralProto}
    f16s::Array{UInt8,1}
    bf16s::Array{UInt8,1}
    sparse_indices::Vector{Int64}
    LiteralProto(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct LiteralProto
const __fnum_LiteralProto = Int[1,2,3,4,5,6,7,8,9,12,10,11,13,14]
const __pack_LiteralProto = Symbol[:preds,:s32s,:s64s,:u32s,:u64s,:f32s,:f64s,:c64s,:sparse_indices]
meta(t::Type{LiteralProto}) = meta(t, ProtoBuf.DEF_REQ, __fnum_LiteralProto, ProtoBuf.DEF_VAL, true, __pack_LiteralProto, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES, ProtoBuf.DEF_FIELD_TYPES)
hash(v::LiteralProto) = ProtoBuf.protohash(v)
isequal(v1::LiteralProto, v2::LiteralProto) = ProtoBuf.protoisequal(v1, v2)
==(v1::LiteralProto, v2::LiteralProto) = ProtoBuf.protoeq(v1, v2)

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
hash(v::WindowDimension) = ProtoBuf.protohash(v)
isequal(v1::WindowDimension, v2::WindowDimension) = ProtoBuf.protoisequal(v1, v2)
==(v1::WindowDimension, v2::WindowDimension) = ProtoBuf.protoeq(v1, v2)

mutable struct Window <: ProtoType
    dimensions::Vector{WindowDimension}
    Window(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct Window
hash(v::Window) = ProtoBuf.protohash(v)
isequal(v1::Window, v2::Window) = ProtoBuf.protoisequal(v1, v2)
==(v1::Window, v2::Window) = ProtoBuf.protoeq(v1, v2)

mutable struct GatherDimensionNumbers <: ProtoType
    output_window_dims::Vector{Int64}
    elided_window_dims::Vector{Int64}
    gather_dims_to_operand_dims::Vector{Int64}
    index_vector_dim::Int64
    GatherDimensionNumbers(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct GatherDimensionNumbers
const __pack_GatherDimensionNumbers = Symbol[:output_window_dims,:elided_window_dims,:gather_dims_to_operand_dims]
meta(t::Type{GatherDimensionNumbers}) = meta(t, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, true, __pack_GatherDimensionNumbers, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES, ProtoBuf.DEF_FIELD_TYPES)
hash(v::GatherDimensionNumbers) = ProtoBuf.protohash(v)
isequal(v1::GatherDimensionNumbers, v2::GatherDimensionNumbers) = ProtoBuf.protoisequal(v1, v2)
==(v1::GatherDimensionNumbers, v2::GatherDimensionNumbers) = ProtoBuf.protoeq(v1, v2)

mutable struct ConstantRequest <: ProtoType
    literal::LiteralProto
    ConstantRequest(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct ConstantRequest
const __fnum_ConstantRequest = Int[2]
meta(t::Type{ConstantRequest}) = meta(t, ProtoBuf.DEF_REQ, __fnum_ConstantRequest, ProtoBuf.DEF_VAL, true, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES, ProtoBuf.DEF_FIELD_TYPES)
hash(v::ConstantRequest) = ProtoBuf.protohash(v)
isequal(v1::ConstantRequest, v2::ConstantRequest) = ProtoBuf.protoisequal(v1, v2)
==(v1::ConstantRequest, v2::ConstantRequest) = ProtoBuf.protoeq(v1, v2)

mutable struct GetTupleElementRequest <: ProtoType
    operand::ComputationDataHandle
    index::Int64
    GetTupleElementRequest(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct GetTupleElementRequest
const __fnum_GetTupleElementRequest = Int[2,3]
meta(t::Type{GetTupleElementRequest}) = meta(t, ProtoBuf.DEF_REQ, __fnum_GetTupleElementRequest, ProtoBuf.DEF_VAL, true, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES, ProtoBuf.DEF_FIELD_TYPES)
hash(v::GetTupleElementRequest) = ProtoBuf.protohash(v)
isequal(v1::GetTupleElementRequest, v2::GetTupleElementRequest) = ProtoBuf.protoisequal(v1, v2)
==(v1::GetTupleElementRequest, v2::GetTupleElementRequest) = ProtoBuf.protoeq(v1, v2)

mutable struct SliceRequest <: ProtoType
    operand::ComputationDataHandle
    start_indices::Vector{Int64}
    limit_indices::Vector{Int64}
    strides::Vector{Int64}
    SliceRequest(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct SliceRequest
const __fnum_SliceRequest = Int[2,3,4,5]
const __pack_SliceRequest = Symbol[:start_indices,:limit_indices,:strides]
meta(t::Type{SliceRequest}) = meta(t, ProtoBuf.DEF_REQ, __fnum_SliceRequest, ProtoBuf.DEF_VAL, true, __pack_SliceRequest, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES, ProtoBuf.DEF_FIELD_TYPES)
hash(v::SliceRequest) = ProtoBuf.protohash(v)
isequal(v1::SliceRequest, v2::SliceRequest) = ProtoBuf.protoisequal(v1, v2)
==(v1::SliceRequest, v2::SliceRequest) = ProtoBuf.protoeq(v1, v2)

mutable struct DynamicSliceRequest <: ProtoType
    operand::ComputationDataHandle
    start_indices::ComputationDataHandle
    slice_sizes::Vector{Int64}
    DynamicSliceRequest(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct DynamicSliceRequest
const __fnum_DynamicSliceRequest = Int[2,3,4]
const __pack_DynamicSliceRequest = Symbol[:slice_sizes]
meta(t::Type{DynamicSliceRequest}) = meta(t, ProtoBuf.DEF_REQ, __fnum_DynamicSliceRequest, ProtoBuf.DEF_VAL, true, __pack_DynamicSliceRequest, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES, ProtoBuf.DEF_FIELD_TYPES)
hash(v::DynamicSliceRequest) = ProtoBuf.protohash(v)
isequal(v1::DynamicSliceRequest, v2::DynamicSliceRequest) = ProtoBuf.protoisequal(v1, v2)
==(v1::DynamicSliceRequest, v2::DynamicSliceRequest) = ProtoBuf.protoeq(v1, v2)

mutable struct DynamicUpdateSliceRequest <: ProtoType
    operand::ComputationDataHandle
    update::ComputationDataHandle
    start_indices::ComputationDataHandle
    DynamicUpdateSliceRequest(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct DynamicUpdateSliceRequest
const __fnum_DynamicUpdateSliceRequest = Int[2,3,4]
meta(t::Type{DynamicUpdateSliceRequest}) = meta(t, ProtoBuf.DEF_REQ, __fnum_DynamicUpdateSliceRequest, ProtoBuf.DEF_VAL, true, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES, ProtoBuf.DEF_FIELD_TYPES)
hash(v::DynamicUpdateSliceRequest) = ProtoBuf.protohash(v)
isequal(v1::DynamicUpdateSliceRequest, v2::DynamicUpdateSliceRequest) = ProtoBuf.protoisequal(v1, v2)
==(v1::DynamicUpdateSliceRequest, v2::DynamicUpdateSliceRequest) = ProtoBuf.protoeq(v1, v2)

mutable struct ConvolutionDimensionNumbers <: ProtoType
    input_batch_dimension::Int64
    input_feature_dimension::Int64
    input_spatial_dimensions::Vector{Int64}
    kernel_input_feature_dimension::Int64
    kernel_output_feature_dimension::Int64
    kernel_spatial_dimensions::Vector{Int64}
    output_batch_dimension::Int64
    output_feature_dimension::Int64
    output_spatial_dimensions::Vector{Int64}
    ConvolutionDimensionNumbers(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct ConvolutionDimensionNumbers
const __fnum_ConvolutionDimensionNumbers = Int[7,8,11,3,4,6,9,10,12]
const __pack_ConvolutionDimensionNumbers = Symbol[:input_spatial_dimensions,:kernel_spatial_dimensions,:output_spatial_dimensions]
meta(t::Type{ConvolutionDimensionNumbers}) = meta(t, ProtoBuf.DEF_REQ, __fnum_ConvolutionDimensionNumbers, ProtoBuf.DEF_VAL, true, __pack_ConvolutionDimensionNumbers, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES, ProtoBuf.DEF_FIELD_TYPES)
hash(v::ConvolutionDimensionNumbers) = ProtoBuf.protohash(v)
isequal(v1::ConvolutionDimensionNumbers, v2::ConvolutionDimensionNumbers) = ProtoBuf.protoisequal(v1, v2)
==(v1::ConvolutionDimensionNumbers, v2::ConvolutionDimensionNumbers) = ProtoBuf.protoeq(v1, v2)

mutable struct ConvolveRequest <: ProtoType
    lhs::ComputationDataHandle
    rhs::ComputationDataHandle
    window::Window
    dimension_numbers::ConvolutionDimensionNumbers
    ConvolveRequest(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct ConvolveRequest
const __fnum_ConvolveRequest = Int[2,3,4,5]
meta(t::Type{ConvolveRequest}) = meta(t, ProtoBuf.DEF_REQ, __fnum_ConvolveRequest, ProtoBuf.DEF_VAL, true, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES, ProtoBuf.DEF_FIELD_TYPES)
hash(v::ConvolveRequest) = ProtoBuf.protohash(v)
isequal(v1::ConvolveRequest, v2::ConvolveRequest) = ProtoBuf.protoisequal(v1, v2)
==(v1::ConvolveRequest, v2::ConvolveRequest) = ProtoBuf.protoeq(v1, v2)

mutable struct FftRequest <: ProtoType
    fft_type::Int32
    fft_length::Vector{Int64}
    operand::ComputationDataHandle
    FftRequest(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct FftRequest
const __pack_FftRequest = Symbol[:fft_length]
meta(t::Type{FftRequest}) = meta(t, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, true, __pack_FftRequest, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES, ProtoBuf.DEF_FIELD_TYPES)
hash(v::FftRequest) = ProtoBuf.protohash(v)
isequal(v1::FftRequest, v2::FftRequest) = ProtoBuf.protoisequal(v1, v2)
==(v1::FftRequest, v2::FftRequest) = ProtoBuf.protoeq(v1, v2)

mutable struct InfeedRequest <: ProtoType
    shape::Shape
    config::Array{UInt8,1}
    InfeedRequest(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct InfeedRequest
const __fnum_InfeedRequest = Int[2,3]
meta(t::Type{InfeedRequest}) = meta(t, ProtoBuf.DEF_REQ, __fnum_InfeedRequest, ProtoBuf.DEF_VAL, true, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES, ProtoBuf.DEF_FIELD_TYPES)
hash(v::InfeedRequest) = ProtoBuf.protohash(v)
isequal(v1::InfeedRequest, v2::InfeedRequest) = ProtoBuf.protoisequal(v1, v2)
==(v1::InfeedRequest, v2::InfeedRequest) = ProtoBuf.protoeq(v1, v2)

mutable struct OutfeedRequest <: ProtoType
    shape::Shape
    operand::ComputationDataHandle
    outfeed_config::Array{UInt8,1}
    OutfeedRequest(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct OutfeedRequest
hash(v::OutfeedRequest) = ProtoBuf.protohash(v)
isequal(v1::OutfeedRequest, v2::OutfeedRequest) = ProtoBuf.protoisequal(v1, v2)
==(v1::OutfeedRequest, v2::OutfeedRequest) = ProtoBuf.protoeq(v1, v2)

mutable struct CallRequest <: ProtoType
    to_apply::ComputationHandle
    operands::Vector{ComputationDataHandle}
    CallRequest(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct CallRequest
const __fnum_CallRequest = Int[2,3]
meta(t::Type{CallRequest}) = meta(t, ProtoBuf.DEF_REQ, __fnum_CallRequest, ProtoBuf.DEF_VAL, true, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES, ProtoBuf.DEF_FIELD_TYPES)
hash(v::CallRequest) = ProtoBuf.protohash(v)
isequal(v1::CallRequest, v2::CallRequest) = ProtoBuf.protoisequal(v1, v2)
==(v1::CallRequest, v2::CallRequest) = ProtoBuf.protoeq(v1, v2)

mutable struct CustomCallRequest <: ProtoType
    call_target_name::AbstractString
    operands::Vector{ComputationDataHandle}
    shape::Shape
    CustomCallRequest(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct CustomCallRequest
const __fnum_CustomCallRequest = Int[2,3,4]
meta(t::Type{CustomCallRequest}) = meta(t, ProtoBuf.DEF_REQ, __fnum_CustomCallRequest, ProtoBuf.DEF_VAL, true, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES, ProtoBuf.DEF_FIELD_TYPES)
hash(v::CustomCallRequest) = ProtoBuf.protohash(v)
isequal(v1::CustomCallRequest, v2::CustomCallRequest) = ProtoBuf.protoisequal(v1, v2)
==(v1::CustomCallRequest, v2::CustomCallRequest) = ProtoBuf.protoeq(v1, v2)

mutable struct HostComputeRequest <: ProtoType
    operands::Vector{ComputationDataHandle}
    channel_name::AbstractString
    cost_estimate_ns::Int64
    shape::Shape
    HostComputeRequest(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct HostComputeRequest
hash(v::HostComputeRequest) = ProtoBuf.protohash(v)
isequal(v1::HostComputeRequest, v2::HostComputeRequest) = ProtoBuf.protoisequal(v1, v2)
==(v1::HostComputeRequest, v2::HostComputeRequest) = ProtoBuf.protoeq(v1, v2)

mutable struct DotDimensionNumbers <: ProtoType
    lhs_contracting_dimensions::Vector{Int64}
    rhs_contracting_dimensions::Vector{Int64}
    lhs_batch_dimensions::Vector{Int64}
    rhs_batch_dimensions::Vector{Int64}
    DotDimensionNumbers(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct DotDimensionNumbers
const __pack_DotDimensionNumbers = Symbol[:lhs_contracting_dimensions,:rhs_contracting_dimensions,:lhs_batch_dimensions,:rhs_batch_dimensions]
meta(t::Type{DotDimensionNumbers}) = meta(t, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, true, __pack_DotDimensionNumbers, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES, ProtoBuf.DEF_FIELD_TYPES)
hash(v::DotDimensionNumbers) = ProtoBuf.protohash(v)
isequal(v1::DotDimensionNumbers, v2::DotDimensionNumbers) = ProtoBuf.protoisequal(v1, v2)
==(v1::DotDimensionNumbers, v2::DotDimensionNumbers) = ProtoBuf.protoeq(v1, v2)

mutable struct DotRequest <: ProtoType
    lhs::ComputationDataHandle
    rhs::ComputationDataHandle
    dimension_numbers::DotDimensionNumbers
    DotRequest(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct DotRequest
const __fnum_DotRequest = Int[2,3,4]
meta(t::Type{DotRequest}) = meta(t, ProtoBuf.DEF_REQ, __fnum_DotRequest, ProtoBuf.DEF_VAL, true, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES, ProtoBuf.DEF_FIELD_TYPES)
hash(v::DotRequest) = ProtoBuf.protohash(v)
isequal(v1::DotRequest, v2::DotRequest) = ProtoBuf.protoisequal(v1, v2)
==(v1::DotRequest, v2::DotRequest) = ProtoBuf.protoeq(v1, v2)

mutable struct MapRequest <: ProtoType
    operands::Vector{ComputationDataHandle}
    to_apply::ComputationHandle
    static_operands::Vector{ComputationDataHandle}
    dimensions::Vector{Int64}
    MapRequest(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct MapRequest
const __fnum_MapRequest = Int[2,3,4,5]
const __pack_MapRequest = Symbol[:dimensions]
meta(t::Type{MapRequest}) = meta(t, ProtoBuf.DEF_REQ, __fnum_MapRequest, ProtoBuf.DEF_VAL, true, __pack_MapRequest, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES, ProtoBuf.DEF_FIELD_TYPES)
hash(v::MapRequest) = ProtoBuf.protohash(v)
isequal(v1::MapRequest, v2::MapRequest) = ProtoBuf.protoisequal(v1, v2)
==(v1::MapRequest, v2::MapRequest) = ProtoBuf.protoeq(v1, v2)

mutable struct ReduceRequest <: ProtoType
    operand::ComputationDataHandle
    init_value::ComputationDataHandle
    dimensions::Vector{Int64}
    to_apply::ComputationHandle
    ReduceRequest(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct ReduceRequest
const __fnum_ReduceRequest = Int[2,3,4,5]
const __pack_ReduceRequest = Symbol[:dimensions]
meta(t::Type{ReduceRequest}) = meta(t, ProtoBuf.DEF_REQ, __fnum_ReduceRequest, ProtoBuf.DEF_VAL, true, __pack_ReduceRequest, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES, ProtoBuf.DEF_FIELD_TYPES)
hash(v::ReduceRequest) = ProtoBuf.protohash(v)
isequal(v1::ReduceRequest, v2::ReduceRequest) = ProtoBuf.protoisequal(v1, v2)
==(v1::ReduceRequest, v2::ReduceRequest) = ProtoBuf.protoeq(v1, v2)

mutable struct ReduceWindowRequest <: ProtoType
    operand::ComputationDataHandle
    init_value::ComputationDataHandle
    window::Window
    to_apply::ComputationHandle
    ReduceWindowRequest(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct ReduceWindowRequest
const __fnum_ReduceWindowRequest = Int[2,3,4,5]
meta(t::Type{ReduceWindowRequest}) = meta(t, ProtoBuf.DEF_REQ, __fnum_ReduceWindowRequest, ProtoBuf.DEF_VAL, true, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES, ProtoBuf.DEF_FIELD_TYPES)
hash(v::ReduceWindowRequest) = ProtoBuf.protohash(v)
isequal(v1::ReduceWindowRequest, v2::ReduceWindowRequest) = ProtoBuf.protoisequal(v1, v2)
==(v1::ReduceWindowRequest, v2::ReduceWindowRequest) = ProtoBuf.protoeq(v1, v2)

mutable struct BatchNormTrainingRequest <: ProtoType
    operand::ComputationDataHandle
    scale::ComputationDataHandle
    offset::ComputationDataHandle
    epsilon::Float32
    feature_index::Int64
    BatchNormTrainingRequest(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct BatchNormTrainingRequest
hash(v::BatchNormTrainingRequest) = ProtoBuf.protohash(v)
isequal(v1::BatchNormTrainingRequest, v2::BatchNormTrainingRequest) = ProtoBuf.protoisequal(v1, v2)
==(v1::BatchNormTrainingRequest, v2::BatchNormTrainingRequest) = ProtoBuf.protoeq(v1, v2)

mutable struct BatchNormInferenceRequest <: ProtoType
    operand::ComputationDataHandle
    scale::ComputationDataHandle
    offset::ComputationDataHandle
    mean::ComputationDataHandle
    variance::ComputationDataHandle
    epsilon::Float32
    feature_index::Int64
    BatchNormInferenceRequest(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct BatchNormInferenceRequest
hash(v::BatchNormInferenceRequest) = ProtoBuf.protohash(v)
isequal(v1::BatchNormInferenceRequest, v2::BatchNormInferenceRequest) = ProtoBuf.protoisequal(v1, v2)
==(v1::BatchNormInferenceRequest, v2::BatchNormInferenceRequest) = ProtoBuf.protoeq(v1, v2)

mutable struct BatchNormGradRequest <: ProtoType
    operand::ComputationDataHandle
    scale::ComputationDataHandle
    mean::ComputationDataHandle
    variance::ComputationDataHandle
    grad_output::ComputationDataHandle
    epsilon::Float32
    feature_index::Int64
    BatchNormGradRequest(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct BatchNormGradRequest
hash(v::BatchNormGradRequest) = ProtoBuf.protohash(v)
isequal(v1::BatchNormGradRequest, v2::BatchNormGradRequest) = ProtoBuf.protoisequal(v1, v2)
==(v1::BatchNormGradRequest, v2::BatchNormGradRequest) = ProtoBuf.protoeq(v1, v2)

mutable struct CrossReplicaSumRequest <: ProtoType
    operand::ComputationDataHandle
    CrossReplicaSumRequest(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct CrossReplicaSumRequest
const __fnum_CrossReplicaSumRequest = Int[2]
meta(t::Type{CrossReplicaSumRequest}) = meta(t, ProtoBuf.DEF_REQ, __fnum_CrossReplicaSumRequest, ProtoBuf.DEF_VAL, true, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES, ProtoBuf.DEF_FIELD_TYPES)
hash(v::CrossReplicaSumRequest) = ProtoBuf.protohash(v)
isequal(v1::CrossReplicaSumRequest, v2::CrossReplicaSumRequest) = ProtoBuf.protoisequal(v1, v2)
==(v1::CrossReplicaSumRequest, v2::CrossReplicaSumRequest) = ProtoBuf.protoeq(v1, v2)

mutable struct SelectAndScatterRequest <: ProtoType
    operand::ComputationDataHandle
    source::ComputationDataHandle
    init_value::ComputationDataHandle
    window::Window
    select::ComputationHandle
    scatter::ComputationHandle
    SelectAndScatterRequest(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct SelectAndScatterRequest
const __fnum_SelectAndScatterRequest = Int[2,3,4,5,6,7]
meta(t::Type{SelectAndScatterRequest}) = meta(t, ProtoBuf.DEF_REQ, __fnum_SelectAndScatterRequest, ProtoBuf.DEF_VAL, true, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES, ProtoBuf.DEF_FIELD_TYPES)
hash(v::SelectAndScatterRequest) = ProtoBuf.protohash(v)
isequal(v1::SelectAndScatterRequest, v2::SelectAndScatterRequest) = ProtoBuf.protoisequal(v1, v2)
==(v1::SelectAndScatterRequest, v2::SelectAndScatterRequest) = ProtoBuf.protoeq(v1, v2)

mutable struct ReverseRequest <: ProtoType
    operand::ComputationDataHandle
    dimensions::Vector{Int64}
    ReverseRequest(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct ReverseRequest
const __fnum_ReverseRequest = Int[2,3]
const __pack_ReverseRequest = Symbol[:dimensions]
meta(t::Type{ReverseRequest}) = meta(t, ProtoBuf.DEF_REQ, __fnum_ReverseRequest, ProtoBuf.DEF_VAL, true, __pack_ReverseRequest, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES, ProtoBuf.DEF_FIELD_TYPES)
hash(v::ReverseRequest) = ProtoBuf.protohash(v)
isequal(v1::ReverseRequest, v2::ReverseRequest) = ProtoBuf.protoisequal(v1, v2)
==(v1::ReverseRequest, v2::ReverseRequest) = ProtoBuf.protoeq(v1, v2)

mutable struct BroadcastRequest <: ProtoType
    operand::ComputationDataHandle
    broadcast_sizes::Vector{Int64}
    BroadcastRequest(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct BroadcastRequest
const __fnum_BroadcastRequest = Int[2,3]
const __pack_BroadcastRequest = Symbol[:broadcast_sizes]
meta(t::Type{BroadcastRequest}) = meta(t, ProtoBuf.DEF_REQ, __fnum_BroadcastRequest, ProtoBuf.DEF_VAL, true, __pack_BroadcastRequest, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES, ProtoBuf.DEF_FIELD_TYPES)
hash(v::BroadcastRequest) = ProtoBuf.protohash(v)
isequal(v1::BroadcastRequest, v2::BroadcastRequest) = ProtoBuf.protoisequal(v1, v2)
==(v1::BroadcastRequest, v2::BroadcastRequest) = ProtoBuf.protoeq(v1, v2)

mutable struct PadRequest <: ProtoType
    operand::ComputationDataHandle
    padding_value::ComputationDataHandle
    padding_config::PaddingConfig
    PadRequest(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct PadRequest
const __fnum_PadRequest = Int[2,3,4]
meta(t::Type{PadRequest}) = meta(t, ProtoBuf.DEF_REQ, __fnum_PadRequest, ProtoBuf.DEF_VAL, true, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES, ProtoBuf.DEF_FIELD_TYPES)
hash(v::PadRequest) = ProtoBuf.protohash(v)
isequal(v1::PadRequest, v2::PadRequest) = ProtoBuf.protoisequal(v1, v2)
==(v1::PadRequest, v2::PadRequest) = ProtoBuf.protoeq(v1, v2)

mutable struct ReshapeRequest <: ProtoType
    operand::ComputationDataHandle
    dimensions::Vector{Int64}
    new_sizes::Vector{Int64}
    ReshapeRequest(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct ReshapeRequest
const __fnum_ReshapeRequest = Int[2,3,4]
const __pack_ReshapeRequest = Symbol[:dimensions,:new_sizes]
meta(t::Type{ReshapeRequest}) = meta(t, ProtoBuf.DEF_REQ, __fnum_ReshapeRequest, ProtoBuf.DEF_VAL, true, __pack_ReshapeRequest, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES, ProtoBuf.DEF_FIELD_TYPES)
hash(v::ReshapeRequest) = ProtoBuf.protohash(v)
isequal(v1::ReshapeRequest, v2::ReshapeRequest) = ProtoBuf.protoisequal(v1, v2)
==(v1::ReshapeRequest, v2::ReshapeRequest) = ProtoBuf.protoeq(v1, v2)

mutable struct TransposeRequest <: ProtoType
    operand::ComputationDataHandle
    dimensions::Vector{Int64}
    TransposeRequest(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct TransposeRequest
const __fnum_TransposeRequest = Int[2,3]
const __pack_TransposeRequest = Symbol[:dimensions]
meta(t::Type{TransposeRequest}) = meta(t, ProtoBuf.DEF_REQ, __fnum_TransposeRequest, ProtoBuf.DEF_VAL, true, __pack_TransposeRequest, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES, ProtoBuf.DEF_FIELD_TYPES)
hash(v::TransposeRequest) = ProtoBuf.protohash(v)
isequal(v1::TransposeRequest, v2::TransposeRequest) = ProtoBuf.protoisequal(v1, v2)
==(v1::TransposeRequest, v2::TransposeRequest) = ProtoBuf.protoeq(v1, v2)

mutable struct ParameterRequest <: ProtoType
    shape::Shape
    parameter::Int64
    name::AbstractString
    ParameterRequest(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct ParameterRequest
const __fnum_ParameterRequest = Int[2,3,4]
meta(t::Type{ParameterRequest}) = meta(t, ProtoBuf.DEF_REQ, __fnum_ParameterRequest, ProtoBuf.DEF_VAL, true, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES, ProtoBuf.DEF_FIELD_TYPES)
hash(v::ParameterRequest) = ProtoBuf.protohash(v)
isequal(v1::ParameterRequest, v2::ParameterRequest) = ProtoBuf.protoisequal(v1, v2)
==(v1::ParameterRequest, v2::ParameterRequest) = ProtoBuf.protoeq(v1, v2)

mutable struct GetLocalShapeRequest <: ProtoType
    computation::ComputationHandle
    operand::ComputationDataHandle
    GetLocalShapeRequest(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct GetLocalShapeRequest
hash(v::GetLocalShapeRequest) = ProtoBuf.protohash(v)
isequal(v1::GetLocalShapeRequest, v2::GetLocalShapeRequest) = ProtoBuf.protoisequal(v1, v2)
==(v1::GetLocalShapeRequest, v2::GetLocalShapeRequest) = ProtoBuf.protoeq(v1, v2)

mutable struct GetLocalShapeResponse <: ProtoType
    shape::Shape
    GetLocalShapeResponse(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct GetLocalShapeResponse
hash(v::GetLocalShapeResponse) = ProtoBuf.protohash(v)
isequal(v1::GetLocalShapeResponse, v2::GetLocalShapeResponse) = ProtoBuf.protoisequal(v1, v2)
==(v1::GetLocalShapeResponse, v2::GetLocalShapeResponse) = ProtoBuf.protoeq(v1, v2)

mutable struct TraceRequest <: ProtoType
    tag::AbstractString
    operand::ComputationDataHandle
    TraceRequest(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct TraceRequest
const __fnum_TraceRequest = Int[2,3]
meta(t::Type{TraceRequest}) = meta(t, ProtoBuf.DEF_REQ, __fnum_TraceRequest, ProtoBuf.DEF_VAL, true, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES, ProtoBuf.DEF_FIELD_TYPES)
hash(v::TraceRequest) = ProtoBuf.protohash(v)
isequal(v1::TraceRequest, v2::TraceRequest) = ProtoBuf.protoisequal(v1, v2)
==(v1::TraceRequest, v2::TraceRequest) = ProtoBuf.protoeq(v1, v2)

mutable struct ConvertRequest <: ProtoType
    operand::ComputationDataHandle
    new_element_type::Int32
    ConvertRequest(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct ConvertRequest
const __fnum_ConvertRequest = Int[2,3]
meta(t::Type{ConvertRequest}) = meta(t, ProtoBuf.DEF_REQ, __fnum_ConvertRequest, ProtoBuf.DEF_VAL, true, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES, ProtoBuf.DEF_FIELD_TYPES)
hash(v::ConvertRequest) = ProtoBuf.protohash(v)
isequal(v1::ConvertRequest, v2::ConvertRequest) = ProtoBuf.protoisequal(v1, v2)
==(v1::ConvertRequest, v2::ConvertRequest) = ProtoBuf.protoeq(v1, v2)

mutable struct ConcatenateRequest <: ProtoType
    operands::Vector{ComputationDataHandle}
    dimension::Int64
    ConcatenateRequest(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct ConcatenateRequest
const __fnum_ConcatenateRequest = Int[2,3]
meta(t::Type{ConcatenateRequest}) = meta(t, ProtoBuf.DEF_REQ, __fnum_ConcatenateRequest, ProtoBuf.DEF_VAL, true, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES, ProtoBuf.DEF_FIELD_TYPES)
hash(v::ConcatenateRequest) = ProtoBuf.protohash(v)
isequal(v1::ConcatenateRequest, v2::ConcatenateRequest) = ProtoBuf.protoisequal(v1, v2)
==(v1::ConcatenateRequest, v2::ConcatenateRequest) = ProtoBuf.protoeq(v1, v2)

mutable struct ConditionalRequest <: ProtoType
    predicate::ComputationDataHandle
    true_operand::ComputationDataHandle
    true_computation::ComputationHandle
    false_operand::ComputationDataHandle
    false_computation::ComputationHandle
    ConditionalRequest(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct ConditionalRequest
const __fnum_ConditionalRequest = Int[2,3,4,5,6]
meta(t::Type{ConditionalRequest}) = meta(t, ProtoBuf.DEF_REQ, __fnum_ConditionalRequest, ProtoBuf.DEF_VAL, true, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES, ProtoBuf.DEF_FIELD_TYPES)
hash(v::ConditionalRequest) = ProtoBuf.protohash(v)
isequal(v1::ConditionalRequest, v2::ConditionalRequest) = ProtoBuf.protoisequal(v1, v2)
==(v1::ConditionalRequest, v2::ConditionalRequest) = ProtoBuf.protoeq(v1, v2)

mutable struct WhileRequest <: ProtoType
    condition::ComputationHandle
    body::ComputationHandle
    init::ComputationDataHandle
    WhileRequest(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct WhileRequest
const __fnum_WhileRequest = Int[2,3,4]
meta(t::Type{WhileRequest}) = meta(t, ProtoBuf.DEF_REQ, __fnum_WhileRequest, ProtoBuf.DEF_VAL, true, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES, ProtoBuf.DEF_FIELD_TYPES)
hash(v::WhileRequest) = ProtoBuf.protohash(v)
isequal(v1::WhileRequest, v2::WhileRequest) = ProtoBuf.protoisequal(v1, v2)
==(v1::WhileRequest, v2::WhileRequest) = ProtoBuf.protoeq(v1, v2)

mutable struct UnaryOpRequest <: ProtoType
    unop::Int32
    operand::ComputationDataHandle
    UnaryOpRequest(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct UnaryOpRequest
const __fnum_UnaryOpRequest = Int[2,3]
meta(t::Type{UnaryOpRequest}) = meta(t, ProtoBuf.DEF_REQ, __fnum_UnaryOpRequest, ProtoBuf.DEF_VAL, true, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES, ProtoBuf.DEF_FIELD_TYPES)
hash(v::UnaryOpRequest) = ProtoBuf.protohash(v)
isequal(v1::UnaryOpRequest, v2::UnaryOpRequest) = ProtoBuf.protoisequal(v1, v2)
==(v1::UnaryOpRequest, v2::UnaryOpRequest) = ProtoBuf.protoeq(v1, v2)

mutable struct BinaryOpRequest <: ProtoType
    binop::Int32
    lhs::ComputationDataHandle
    rhs::ComputationDataHandle
    broadcast_dimensions::Vector{Int64}
    BinaryOpRequest(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct BinaryOpRequest
const __fnum_BinaryOpRequest = Int[2,3,4,5]
const __pack_BinaryOpRequest = Symbol[:broadcast_dimensions]
meta(t::Type{BinaryOpRequest}) = meta(t, ProtoBuf.DEF_REQ, __fnum_BinaryOpRequest, ProtoBuf.DEF_VAL, true, __pack_BinaryOpRequest, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES, ProtoBuf.DEF_FIELD_TYPES)
hash(v::BinaryOpRequest) = ProtoBuf.protohash(v)
isequal(v1::BinaryOpRequest, v2::BinaryOpRequest) = ProtoBuf.protoisequal(v1, v2)
==(v1::BinaryOpRequest, v2::BinaryOpRequest) = ProtoBuf.protoeq(v1, v2)

mutable struct RngRequest <: ProtoType
    distribution::Int32
    parameter::Vector{ComputationDataHandle}
    shape::Shape
    RngRequest(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct RngRequest
const __fnum_RngRequest = Int[2,3,4]
meta(t::Type{RngRequest}) = meta(t, ProtoBuf.DEF_REQ, __fnum_RngRequest, ProtoBuf.DEF_VAL, true, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES, ProtoBuf.DEF_FIELD_TYPES)
hash(v::RngRequest) = ProtoBuf.protohash(v)
isequal(v1::RngRequest, v2::RngRequest) = ProtoBuf.protoisequal(v1, v2)
==(v1::RngRequest, v2::RngRequest) = ProtoBuf.protoeq(v1, v2)

mutable struct TernaryOpRequest <: ProtoType
    triop::Int32
    lhs::ComputationDataHandle
    rhs::ComputationDataHandle
    ehs::ComputationDataHandle
    TernaryOpRequest(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct TernaryOpRequest
const __fnum_TernaryOpRequest = Int[2,3,4,5]
meta(t::Type{TernaryOpRequest}) = meta(t, ProtoBuf.DEF_REQ, __fnum_TernaryOpRequest, ProtoBuf.DEF_VAL, true, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES, ProtoBuf.DEF_FIELD_TYPES)
hash(v::TernaryOpRequest) = ProtoBuf.protohash(v)
isequal(v1::TernaryOpRequest, v2::TernaryOpRequest) = ProtoBuf.protoisequal(v1, v2)
==(v1::TernaryOpRequest, v2::TernaryOpRequest) = ProtoBuf.protoeq(v1, v2)

mutable struct VariadicOpRequest <: ProtoType
    varop::Int32
    operands::Vector{ComputationDataHandle}
    VariadicOpRequest(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct VariadicOpRequest
const __fnum_VariadicOpRequest = Int[2,3]
meta(t::Type{VariadicOpRequest}) = meta(t, ProtoBuf.DEF_REQ, __fnum_VariadicOpRequest, ProtoBuf.DEF_VAL, true, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES, ProtoBuf.DEF_FIELD_TYPES)
hash(v::VariadicOpRequest) = ProtoBuf.protohash(v)
isequal(v1::VariadicOpRequest, v2::VariadicOpRequest) = ProtoBuf.protoisequal(v1, v2)
==(v1::VariadicOpRequest, v2::VariadicOpRequest) = ProtoBuf.protoeq(v1, v2)

mutable struct ReducePrecisionRequest <: ProtoType
    operand::ComputationDataHandle
    exponent_bits::Int32
    mantissa_bits::Int32
    ReducePrecisionRequest(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct ReducePrecisionRequest
hash(v::ReducePrecisionRequest) = ProtoBuf.protohash(v)
isequal(v1::ReducePrecisionRequest, v2::ReducePrecisionRequest) = ProtoBuf.protoisequal(v1, v2)
==(v1::ReducePrecisionRequest, v2::ReducePrecisionRequest) = ProtoBuf.protoeq(v1, v2)

mutable struct SendRequest <: ProtoType
    operand::ComputationDataHandle
    channel_handle::ChannelHandle
    SendRequest(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct SendRequest
hash(v::SendRequest) = ProtoBuf.protohash(v)
isequal(v1::SendRequest, v2::SendRequest) = ProtoBuf.protoisequal(v1, v2)
==(v1::SendRequest, v2::SendRequest) = ProtoBuf.protoeq(v1, v2)

mutable struct RecvRequest <: ProtoType
    shape::Shape
    channel_handle::ChannelHandle
    RecvRequest(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct RecvRequest
hash(v::RecvRequest) = ProtoBuf.protohash(v)
isequal(v1::RecvRequest, v2::RecvRequest) = ProtoBuf.protoisequal(v1, v2)
==(v1::RecvRequest, v2::RecvRequest) = ProtoBuf.protoeq(v1, v2)

mutable struct GatherRequest <: ProtoType
    input::ComputationDataHandle
    gather_indices::ComputationDataHandle
    dimension_numbers::GatherDimensionNumbers
    window_bounds::Vector{Int64}
    GatherRequest(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct GatherRequest
const __pack_GatherRequest = Symbol[:window_bounds]
meta(t::Type{GatherRequest}) = meta(t, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, true, __pack_GatherRequest, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES, ProtoBuf.DEF_FIELD_TYPES)
hash(v::GatherRequest) = ProtoBuf.protohash(v)
isequal(v1::GatherRequest, v2::GatherRequest) = ProtoBuf.protoisequal(v1, v2)
==(v1::GatherRequest, v2::GatherRequest) = ProtoBuf.protoeq(v1, v2)

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
    tile_shape::Shape
    tile_assignment_dimensions::Vector{Int64}
    tile_assignment_devices::Vector{Int64}
    tuple_shardings::Vector{OpSharding}
    OpSharding(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct OpSharding
const __pack_OpSharding = Symbol[:tile_assignment_dimensions,:tile_assignment_devices]
meta(t::Type{OpSharding}) = meta(t, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, true, __pack_OpSharding, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES, ProtoBuf.DEF_FIELD_TYPES)
hash(v::OpSharding) = ProtoBuf.protohash(v)
isequal(v1::OpSharding, v2::OpSharding) = ProtoBuf.protoisequal(v1, v2)
==(v1::OpSharding, v2::OpSharding) = ProtoBuf.protoeq(v1, v2)

mutable struct OpRequest <: ProtoType
    computation::ComputationHandle
    metadata::OpMetadata
    sharding::OpSharding
    binary_op_request::BinaryOpRequest
    broadcast_request::BroadcastRequest
    call_request::CallRequest
    concatenate_request::ConcatenateRequest
    constant_request::ConstantRequest
    convert_request::ConvertRequest
    convolve_request::ConvolveRequest
    cross_replica_sum_request::CrossReplicaSumRequest
    custom_call_request::CustomCallRequest
    dot_request::DotRequest
    dynamic_slice_request::DynamicSliceRequest
    dynamic_update_slice_request::DynamicUpdateSliceRequest
    get_tuple_element_request::GetTupleElementRequest
    infeed_request::InfeedRequest
    map_request::MapRequest
    pad_request::PadRequest
    parameter_request::ParameterRequest
    reduce_precision_request::ReducePrecisionRequest
    reduce_request::ReduceRequest
    reduce_window_request::ReduceWindowRequest
    reshape_request::ReshapeRequest
    reverse_request::ReverseRequest
    rng_request::RngRequest
    select_and_scatter_request::SelectAndScatterRequest
    slice_request::SliceRequest
    ternary_op_request::TernaryOpRequest
    trace_request::TraceRequest
    transpose_request::TransposeRequest
    unary_op_request::UnaryOpRequest
    variadic_op_request::VariadicOpRequest
    while_request::WhileRequest
    send_request::SendRequest
    recv_request::RecvRequest
    outfeed_request::OutfeedRequest
    batch_norm_training_request::BatchNormTrainingRequest
    batch_norm_grad_request::BatchNormGradRequest
    batch_norm_inference_request::BatchNormInferenceRequest
    fft_request::FftRequest
    bitcast_convert_request::ConvertRequest
    conditional_request::ConditionalRequest
    host_compute_request::HostComputeRequest
    gather_request::GatherRequest
    OpRequest(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct OpRequest
const __fnum_OpRequest = Int[1,33,40,2,3,4,5,6,7,8,9,10,43,11,12,13,14,15,16,17,36,18,19,20,21,22,23,24,25,26,34,27,28,29,30,31,32,35,37,38,41,42,44,45,46]
const __oneofs_OpRequest = Int[0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
const __oneof_names_OpRequest = [Symbol("op")]
meta(t::Type{OpRequest}) = meta(t, ProtoBuf.DEF_REQ, __fnum_OpRequest, ProtoBuf.DEF_VAL, true, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, __oneofs_OpRequest, __oneof_names_OpRequest, ProtoBuf.DEF_FIELD_TYPES)
hash(v::OpRequest) = ProtoBuf.protohash(v)
isequal(v1::OpRequest, v2::OpRequest) = ProtoBuf.protoisequal(v1, v2)
==(v1::OpRequest, v2::OpRequest) = ProtoBuf.protoeq(v1, v2)

mutable struct OpResponse <: ProtoType
    output::ComputationDataHandle
    OpResponse(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct OpResponse
hash(v::OpResponse) = ProtoBuf.protohash(v)
isequal(v1::OpResponse, v2::OpResponse) = ProtoBuf.protoisequal(v1, v2)
==(v1::OpResponse, v2::OpResponse) = ProtoBuf.protoeq(v1, v2)

export PrimitiveType, PaddingValue, Format, FftType, UnaryOperation, BinaryOperation, RandomDistribution, TernaryOperation, VariadicOperation, PaddingConfig_PaddingConfigDimension, PaddingConfig, Layout, Shape, ProgramShape, ComputationStats, OpMetadata, ExecutionProfile, ComputationHandle, ExecutionHandle, GlobalDataHandle, ComputationDataHandle, DeviceHandle, ChannelHandle, DeviceAssignmentProto_ComputationDevice, DeviceAssignmentProto, LiteralProto, WindowDimension, Window, GatherDimensionNumbers, ConstantRequest, GetTupleElementRequest, SliceRequest, DynamicSliceRequest, DynamicUpdateSliceRequest, ConvolutionDimensionNumbers, ConvolveRequest, FftRequest, InfeedRequest, OutfeedRequest, CallRequest, CustomCallRequest, HostComputeRequest, DotDimensionNumbers, DotRequest, MapRequest, ReduceRequest, ReduceWindowRequest, BatchNormTrainingRequest, BatchNormInferenceRequest, BatchNormGradRequest, CrossReplicaSumRequest, SelectAndScatterRequest, ReverseRequest, BroadcastRequest, PadRequest, ReshapeRequest, TransposeRequest, ParameterRequest, GetLocalShapeRequest, GetLocalShapeResponse, TraceRequest, ConvertRequest, ConcatenateRequest, ConditionalRequest, WhileRequest, UnaryOpRequest, BinaryOpRequest, RngRequest, TernaryOpRequest, VariadicOpRequest, ReducePrecisionRequest, SendRequest, RecvRequest, GatherRequest, OpSharding_Type, OpSharding, OpRequest, OpResponse
