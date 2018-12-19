# syntax: proto3
using ProtoBuf
import ProtoBuf.meta

struct __enum_HloReducePrecisionOptions_Location <: ProtoEnum
    OP_INPUTS::Int32
    OP_OUTPUTS::Int32
    UNFUSED_OP_OUTPUTS::Int32
    FUSION_INPUTS_BY_CONTENT::Int32
    FUSION_OUTPUTS_BY_CONTENT::Int32
    __enum_HloReducePrecisionOptions_Location() = new(0,1,2,3,4)
end #struct __enum_HloReducePrecisionOptions_Location
const HloReducePrecisionOptions_Location = __enum_HloReducePrecisionOptions_Location()

mutable struct HloReducePrecisionOptions <: ProtoType
    location::Int32
    exponent_bits::UInt32
    mantissa_bits::UInt32
    opcodes_to_suffix::Base.Vector{UInt32}
    opname_substrings_to_suffix::Base.Vector{AbstractString}
    HloReducePrecisionOptions(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct HloReducePrecisionOptions
const __pack_HloReducePrecisionOptions = Symbol[:opcodes_to_suffix]
meta(t::Type{HloReducePrecisionOptions}) = meta(t, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, true, __pack_HloReducePrecisionOptions, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES, ProtoBuf.DEF_FIELD_TYPES)

mutable struct DebugOptions_XlaBackendExtraOptionsEntry <: ProtoType
    key::AbstractString
    value::AbstractString
    DebugOptions_XlaBackendExtraOptionsEntry(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct DebugOptions_XlaBackendExtraOptionsEntry (mapentry)

mutable struct DebugOptions <: ProtoType
    xla_generate_hlo_graph::AbstractString
    xla_hlo_graph_addresses::Bool
    xla_hlo_graph_path::AbstractString
    xla_hlo_dump_as_graphdef::Bool
    xla_log_hlo_text::AbstractString
    xla_generate_hlo_text_to::AbstractString
    xla_dump_optimized_hlo_proto_to::AbstractString
    xla_hlo_profile::Bool
    xla_dump_computations_to::AbstractString
    xla_dump_executions_to::AbstractString
    xla_disable_hlo_passes::Base.Vector{AbstractString}
    xla_backend_optimization_level::Int32
    xla_embed_ir_in_executable::Bool
    xla_dump_ir_to::AbstractString
    xla_eliminate_hlo_implicit_broadcast::Bool
    xla_cpu_multi_thread_eigen::Bool
    xla_gpu_cuda_data_dir::AbstractString
    xla_gpu_ftz::Bool
    xla_gpu_disable_multi_streaming::Bool
    xla_llvm_enable_alias_scope_metadata::Bool
    xla_llvm_enable_noalias_metadata::Bool
    xla_llvm_enable_invariant_load_metadata::Bool
    xla_llvm_disable_expensive_passes::Bool
    hlo_reduce_precision_options::Base.Vector{HloReducePrecisionOptions}
    xla_test_all_output_layouts::Bool
    xla_test_all_input_layouts::Bool
    xla_hlo_graph_sharding_color::Bool
    xla_hlo_tfgraph_device_scopes::Bool
    xla_gpu_use_cudnn_batchnorm::Bool
    xla_dump_unoptimized_hlo_proto_to::AbstractString
    xla_dump_per_pass_hlo_proto_to::AbstractString
    xla_cpu_use_mkl_dnn::Bool
    xla_gpu_max_kernel_unroll_factor::Int32
    xla_cpu_enable_fast_math::Bool
    xla_gpu_enable_fast_min_max::Bool
    xla_gpu_crash_on_verification_failures::Bool
    xla_force_host_platform_device_count::Int32
    xla_backend_extra_options::Base.Dict{AbstractString,AbstractString} # map entry
    DebugOptions(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct DebugOptions
const __fnum_DebugOptions = Int[1,2,4,5,6,7,8,9,10,11,30,31,33,34,35,60,61,62,63,70,71,72,73,80,90,91,92,93,94,95,96,97,98,99,100,101,102,500]
meta(t::Type{DebugOptions}) = meta(t, ProtoBuf.DEF_REQ, __fnum_DebugOptions, ProtoBuf.DEF_VAL, true, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES, ProtoBuf.DEF_FIELD_TYPES)

mutable struct ExecutionOptions <: ProtoType
    shape_with_output_layout::ShapeProto
    seed::UInt64
    debug_options::DebugOptions
    device_handles::Base.Vector{DeviceHandle}
    ExecutionOptions(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct ExecutionOptions
const __fnum_ExecutionOptions = Int[2,3,4,5]
meta(t::Type{ExecutionOptions}) = meta(t, ProtoBuf.DEF_REQ, __fnum_ExecutionOptions, ProtoBuf.DEF_VAL, true, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES, ProtoBuf.DEF_FIELD_TYPES)

mutable struct GetDeviceHandlesRequest <: ProtoType
    device_count::Int64
    GetDeviceHandlesRequest(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct GetDeviceHandlesRequest

mutable struct GetDeviceHandlesResponse <: ProtoType
    device_handles::Base.Vector{DeviceHandle}
    GetDeviceHandlesResponse(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct GetDeviceHandlesResponse

mutable struct TransferToClientRequest <: ProtoType
    data::GlobalDataHandle
    shape_with_layout::ShapeProto
    TransferToClientRequest(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct TransferToClientRequest

mutable struct TransferToClientResponse <: ProtoType
    literal::LiteralProto
    TransferToClientResponse(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct TransferToClientResponse

mutable struct TransferToServerRequest <: ProtoType
    literal::LiteralProto
    device_handle::DeviceHandle
    TransferToServerRequest(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct TransferToServerRequest

mutable struct TransferToServerResponse <: ProtoType
    data::GlobalDataHandle
    TransferToServerResponse(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct TransferToServerResponse

mutable struct TransferToInfeedRequest <: ProtoType
    literal::LiteralProto
    replica_id::Int64
    device_handle::DeviceHandle
    TransferToInfeedRequest(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct TransferToInfeedRequest

mutable struct TransferToInfeedResponse <: ProtoType
    TransferToInfeedResponse(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct TransferToInfeedResponse

mutable struct TransferFromOutfeedRequest <: ProtoType
    shape_with_layout::ShapeProto
    replica_id::Int64
    device_handle::DeviceHandle
    TransferFromOutfeedRequest(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct TransferFromOutfeedRequest

mutable struct TransferFromOutfeedResponse <: ProtoType
    literal::LiteralProto
    TransferFromOutfeedResponse(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct TransferFromOutfeedResponse

mutable struct ResetDeviceRequest <: ProtoType
    device_handle::DeviceHandle
    ResetDeviceRequest(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct ResetDeviceRequest

mutable struct ResetDeviceResponse <: ProtoType
    ResetDeviceResponse(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct ResetDeviceResponse

mutable struct ComputationGraphStatsRequest <: ProtoType
    computation::HloModuleProto
    debug_options::DebugOptions
    ComputationGraphStatsRequest(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct ComputationGraphStatsRequest

mutable struct ComputationStatsResponse <: ProtoType
    stats::ComputationStats
    ComputationStatsResponse(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct ComputationStatsResponse

mutable struct CreateChannelHandleRequest <: ProtoType
    channel_type::Int32
    CreateChannelHandleRequest(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct CreateChannelHandleRequest

mutable struct CreateChannelHandleResponse <: ProtoType
    channel::ChannelHandle
    CreateChannelHandleResponse(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct CreateChannelHandleResponse

mutable struct UnregisterRequest <: ProtoType
    data::Base.Vector{GlobalDataHandle}
    UnregisterRequest(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct UnregisterRequest

mutable struct UnregisterResponse <: ProtoType
    UnregisterResponse(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct UnregisterResponse

mutable struct CompileRequest <: ProtoType
    computation::HloModuleProto
    execution_options::ExecutionOptions
    input_shape_with_layout::Base.Vector{ShapeProto}
    CompileRequest(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct CompileRequest

mutable struct CompileResponse <: ProtoType
    handle::ExecutionHandle
    CompileResponse(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct CompileResponse

mutable struct ExecuteRequest <: ProtoType
    handle::ExecutionHandle
    arguments::Base.Vector{GlobalDataHandle}
    ExecuteRequest(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct ExecuteRequest

mutable struct ExecuteGraphRequest <: ProtoType
    computation::HloModuleProto
    arguments::Base.Vector{GlobalDataHandle}
    execution_options::ExecutionOptions
    ExecuteGraphRequest(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct ExecuteGraphRequest

mutable struct ExecuteGraphParallelRequest <: ProtoType
    requests::Base.Vector{ExecuteGraphRequest}
    ExecuteGraphParallelRequest(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct ExecuteGraphParallelRequest

mutable struct ExecuteResponse <: ProtoType
    output::GlobalDataHandle
    profile::ExecutionProfile
    ExecuteResponse(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct ExecuteResponse

mutable struct ExecuteParallelResponse <: ProtoType
    responses::Base.Vector{ExecuteResponse}
    ExecuteParallelResponse(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct ExecuteParallelResponse

mutable struct WaitForExecutionRequest <: ProtoType
    execution::ExecutionHandle
    WaitForExecutionRequest(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct WaitForExecutionRequest

mutable struct WaitForExecutionResponse <: ProtoType
    output::GlobalDataHandle
    profile::ExecutionProfile
    WaitForExecutionResponse(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct WaitForExecutionResponse

mutable struct ComputeConstantGraphRequest <: ProtoType
    computation::HloModuleProto
    output_layout::Layout
    ComputeConstantGraphRequest(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct ComputeConstantGraphRequest

mutable struct ComputeConstantResponse <: ProtoType
    literal::LiteralProto
    ComputeConstantResponse(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct ComputeConstantResponse

mutable struct DeconstructTupleRequest <: ProtoType
    tuple_handle::GlobalDataHandle
    DeconstructTupleRequest(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct DeconstructTupleRequest
const __fnum_DeconstructTupleRequest = Int[2]
meta(t::Type{DeconstructTupleRequest}) = meta(t, ProtoBuf.DEF_REQ, __fnum_DeconstructTupleRequest, ProtoBuf.DEF_VAL, true, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES, ProtoBuf.DEF_FIELD_TYPES)

mutable struct DeconstructTupleResponse <: ProtoType
    element_handles::Base.Vector{GlobalDataHandle}
    DeconstructTupleResponse(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct DeconstructTupleResponse

mutable struct LoadDataRequest <: ProtoType
    columnio_tablet_path::AbstractString
    columnio_field::AbstractString
    element_shape::ShapeProto
    offset::Int64
    limit::Int64
    zip::Bool
    LoadDataRequest(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct LoadDataRequest

mutable struct LoadDataResponse <: ProtoType
    data::GlobalDataHandle
    data_shape::ShapeProto
    available_rows::Int64
    rows_loaded::Int64
    nanoseconds::Int64
    LoadDataResponse(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct LoadDataResponse

mutable struct GetShapeRequest <: ProtoType
    data::GlobalDataHandle
    GetShapeRequest(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct GetShapeRequest

mutable struct GetShapeResponse <: ProtoType
    shape::ShapeProto
    GetShapeResponse(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct GetShapeResponse

mutable struct UnpackRequest <: ProtoType
    data::GlobalDataHandle
    UnpackRequest(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct UnpackRequest

mutable struct UnpackResponse <: ProtoType
    tied_data::Base.Vector{GlobalDataHandle}
    UnpackResponse(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct UnpackResponse

export HloReducePrecisionOptions_Location, HloReducePrecisionOptions, DebugOptions_XlaBackendExtraOptionsEntry, DebugOptions, ExecutionOptions, GetDeviceHandlesRequest, GetDeviceHandlesResponse, TransferToClientRequest, TransferToClientResponse, TransferToServerRequest, TransferToServerResponse, TransferToInfeedRequest, TransferToInfeedResponse, TransferFromOutfeedRequest, TransferFromOutfeedResponse, ResetDeviceRequest, ResetDeviceResponse, ComputationGraphStatsRequest, ComputationStatsResponse, CreateChannelHandleRequest, CreateChannelHandleResponse, UnregisterRequest, UnregisterResponse, CompileRequest, CompileResponse, ExecuteRequest, ExecuteGraphRequest, ExecuteGraphParallelRequest, ExecuteResponse, ExecuteParallelResponse, WaitForExecutionRequest, WaitForExecutionResponse, ComputeConstantGraphRequest, ComputeConstantResponse, DeconstructTupleRequest, DeconstructTupleResponse, LoadDataRequest, LoadDataResponse, GetShapeRequest, GetShapeResponse, UnpackRequest, UnpackResponse
# mapentries: "DebugOptions_XlaBackendExtraOptionsEntry" => ("AbstractString", "AbstractString")
