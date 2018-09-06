# syntax: proto3
using Compat
using ProtoBuf
import ProtoBuf.meta

mutable struct HloInstructionProto_SliceDimensions <: ProtoType
    start::Int64
    limit::Int64
    stride::Int64
    HloInstructionProto_SliceDimensions(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct HloInstructionProto_SliceDimensions

mutable struct HloInstructionProto <: ProtoType
    name::AbstractString
    opcode::AbstractString
    shape::Shape
    metadata::OpMetadata
    literal::LiteralProto
    parameter_number::Int64
    fusion_kind::AbstractString
    tuple_index::Int64
    dimensions::Base.Vector{Int64}
    window::Window
    convolution_dimension_numbers::ConvolutionDimensionNumbers
    feature_group_count::Int64
    slice_dimensions::Base.Vector{HloInstructionProto_SliceDimensions}
    exponent_bits::Int32
    mantissa_bits::Int32
    dynamic_slice_sizes::Base.Vector{Int64}
    padding_config::PaddingConfig
    outfeed_config::Array{UInt8,1}
    distribution::Int32
    epsilon::Float32
    feature_index::Int64
    channel_id::Int64
    infeed_config::Array{UInt8,1}
    custom_call_target::AbstractString
    outfeed_shape::Shape
    dot_dimension_numbers::DotDimensionNumbers
    fft_type::Int32
    fft_length::Base.Vector{Int64}
    gather_dimension_numbers::GatherDimensionNumbers
    gather_slice_sizes::Base.Vector{Int64}
    channel_name::AbstractString
    cost_estimate_ns::Int64
    id::Int64
    operand_ids::Base.Vector{Int64}
    control_predecessor_ids::Base.Vector{Int64}
    called_computation_ids::Base.Vector{Int64}
    sharding::OpSharding
    backend_config::AbstractString
    replica_groups::Base.Vector{ReplicaGroup}
    all_reduce_id::Int64
    cross_replica_sum_barrier::AbstractString
    is_host_transfer::Bool
    scatter_dimension_numbers::ScatterDimensionNumbers
    precision_config::PrecisionConfigProto
    source_target_pairs::Base.Vector{SourceTarget}
    HloInstructionProto(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct HloInstructionProto
const __fnum_HloInstructionProto = Int[1,2,3,7,8,9,11,13,14,15,16,50,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,41,42,35,36,37,38,40,43,49,45,46,47,48,51,52]
const __pack_HloInstructionProto = Symbol[:dimensions,:dynamic_slice_sizes,:fft_length,:gather_slice_sizes,:operand_ids,:control_predecessor_ids,:called_computation_ids]
meta(t::Type{HloInstructionProto}) = meta(t, ProtoBuf.DEF_REQ, __fnum_HloInstructionProto, ProtoBuf.DEF_VAL, true, __pack_HloInstructionProto, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES, ProtoBuf.DEF_FIELD_TYPES)

mutable struct HloComputationProto <: ProtoType
    name::AbstractString
    instructions::Base.Vector{HloInstructionProto}
    program_shape::ProgramShape
    id::Int64
    root_id::Int64
    HloComputationProto(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct HloComputationProto
const __fnum_HloComputationProto = Int[1,2,4,5,6]
meta(t::Type{HloComputationProto}) = meta(t, ProtoBuf.DEF_REQ, __fnum_HloComputationProto, ProtoBuf.DEF_VAL, true, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES, ProtoBuf.DEF_FIELD_TYPES)

mutable struct HloModuleProto <: ProtoType
    name::AbstractString
    entry_computation_name::AbstractString
    entry_computation_id::Int64
    computations::Base.Vector{HloComputationProto}
    program_shape::ProgramShape
    id::Int64
    HloModuleProto(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct HloModuleProto
const __fnum_HloModuleProto = Int[1,2,6,3,4,5]
meta(t::Type{HloModuleProto}) = meta(t, ProtoBuf.DEF_REQ, __fnum_HloModuleProto, ProtoBuf.DEF_VAL, true, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES, ProtoBuf.DEF_FIELD_TYPES)

mutable struct HloOrderingProto_SequentialComputation <: ProtoType
    computation_name::AbstractString
    instruction_names::Base.Vector{AbstractString}
    HloOrderingProto_SequentialComputation(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct HloOrderingProto_SequentialComputation

mutable struct HloOrderingProto <: ProtoType
    sequential_computations::Base.Vector{HloOrderingProto_SequentialComputation}
    HloOrderingProto(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct HloOrderingProto

mutable struct LogicalBufferProto_Location <: ProtoType
    computation_name::AbstractString
    instruction_name::AbstractString
    shape_index::Base.Vector{Int64}
    LogicalBufferProto_Location(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct LogicalBufferProto_Location
const __pack_LogicalBufferProto_Location = Symbol[:shape_index]
meta(t::Type{LogicalBufferProto_Location}) = meta(t, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, true, __pack_LogicalBufferProto_Location, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES, ProtoBuf.DEF_FIELD_TYPES)

mutable struct LogicalBufferProto <: ProtoType
    id::Int64
    size::Int64
    defined_at::LogicalBufferProto_Location
    color::Int64
    LogicalBufferProto(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct LogicalBufferProto

mutable struct BufferAllocationProto_Assigned <: ProtoType
    logical_buffer_id::Int64
    offset::Int64
    size::Int64
    BufferAllocationProto_Assigned(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct BufferAllocationProto_Assigned

mutable struct BufferAllocationProto <: ProtoType
    index::Int64
    size::Int64
    is_thread_local::Bool
    is_tuple::Bool
    is_entry_computation_parameter::Bool
    is_constant::Bool
    parameter_number::Int64
    parameter_shape_index::Base.Vector{Int64}
    maybe_live_out::Bool
    color::Int64
    assigned::Base.Vector{BufferAllocationProto_Assigned}
    BufferAllocationProto(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct BufferAllocationProto
const __fnum_BufferAllocationProto = Int[1,2,3,11,5,12,6,10,7,8,9]
const __pack_BufferAllocationProto = Symbol[:parameter_shape_index]
meta(t::Type{BufferAllocationProto}) = meta(t, ProtoBuf.DEF_REQ, __fnum_BufferAllocationProto, ProtoBuf.DEF_VAL, true, __pack_BufferAllocationProto, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES, ProtoBuf.DEF_FIELD_TYPES)

struct __enum_HeapSimulatorTrace_Event_Kind <: ProtoEnum
    ALLOC::Int32
    FREE::Int32
    SHARE_WITH::Int32
    __enum_HeapSimulatorTrace_Event_Kind() = new(0,1,2)
end #struct __enum_HeapSimulatorTrace_Event_Kind
const HeapSimulatorTrace_Event_Kind = __enum_HeapSimulatorTrace_Event_Kind()

mutable struct HeapSimulatorTrace_Event <: ProtoType
    kind::Int32
    buffer_id::Int64
    computation_name::AbstractString
    instruction_name::AbstractString
    share_with_canonical_id::Int64
    HeapSimulatorTrace_Event(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct HeapSimulatorTrace_Event

mutable struct HeapSimulatorTrace <: ProtoType
    events::Base.Vector{HeapSimulatorTrace_Event}
    whole_module_simulation::Bool
    HeapSimulatorTrace(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct HeapSimulatorTrace

mutable struct BufferAssignmentProto_BufferAlias <: ProtoType
    source_buffer_id::Int64
    location::LogicalBufferProto_Location
    BufferAssignmentProto_BufferAlias(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct BufferAssignmentProto_BufferAlias

mutable struct BufferAssignmentProto <: ProtoType
    logical_buffers::Base.Vector{LogicalBufferProto}
    buffer_aliases::Base.Vector{BufferAssignmentProto_BufferAlias}
    buffer_allocations::Base.Vector{BufferAllocationProto}
    heap_simulator_traces::Base.Vector{HeapSimulatorTrace}
    BufferAssignmentProto(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct BufferAssignmentProto

mutable struct HloProto <: ProtoType
    hlo_module::HloModuleProto
    hlo_ordering::HloOrderingProto
    buffer_assignment::BufferAssignmentProto
    HloProto(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct HloProto

mutable struct HloSnapshot <: ProtoType
    hlo::HloProto
    arguments::Base.Vector{LiteralProto}
    result::LiteralProto
    execution_platform::AbstractString
    HloSnapshot(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct HloSnapshot

export HloInstructionProto_SliceDimensions, HloInstructionProto, HloComputationProto, HloModuleProto, HloOrderingProto_SequentialComputation, HloOrderingProto, LogicalBufferProto_Location, LogicalBufferProto, BufferAllocationProto_Assigned, BufferAllocationProto, HeapSimulatorTrace_Event_Kind, HeapSimulatorTrace_Event, HeapSimulatorTrace, BufferAssignmentProto_BufferAlias, BufferAssignmentProto, HloProto, HloSnapshot
