# syntax: proto3
using ProtoBuf
import ProtoBuf.meta
import ._ProtoBuf_Top_.tensorflow
import ._ProtoBuf_Top_.xla

mutable struct DeviceAssignment_ComputationDevice_DeviceMeshCoordinates <: ProtoType
    value::Base.Vector{Int32}
    DeviceAssignment_ComputationDevice_DeviceMeshCoordinates(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct DeviceAssignment_ComputationDevice_DeviceMeshCoordinates
const __pack_DeviceAssignment_ComputationDevice_DeviceMeshCoordinates = Symbol[:value]
meta(t::Type{DeviceAssignment_ComputationDevice_DeviceMeshCoordinates}) = meta(t, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, true, __pack_DeviceAssignment_ComputationDevice_DeviceMeshCoordinates, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES, ProtoBuf.DEF_FIELD_TYPES)

mutable struct DeviceAssignment_ComputationDevice <: ProtoType
    replica_devices::Base.Vector{DeviceAssignment_ComputationDevice_DeviceMeshCoordinates}
    DeviceAssignment_ComputationDevice(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct DeviceAssignment_ComputationDevice

mutable struct DeviceAssignment <: ProtoType
    computation_devices::Base.Vector{DeviceAssignment_ComputationDevice}
    DeviceAssignment(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct DeviceAssignment

mutable struct XLAComputationConfig <: ProtoType
    num_replicas::Int32
    num_cores_per_replica::Int32
    host_compute_metadata::tensorflow.tf2xla.HostComputeMetadata
    program_shape::xla.ProgramShapeProto
    per_core_program_shape::Base.Vector{xla.ProgramShapeProto}
    device_assignment::DeviceAssignment
    debug_options::xla.DebugOptions
    XLAComputationConfig(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct XLAComputationConfig

mutable struct XLAComputation <: ProtoType
    config::XLAComputationConfig
    hlo_snapshot::xla.HloSnapshot
    XLAComputation(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct XLAComputation

mutable struct XLAAllocation <: ProtoType
    device_ordinal::Int32
    value::xla.LiteralProto
    XLAAllocation(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct XLAAllocation

mutable struct XLATupleNode <: ProtoType
    input_index::Int32
    release_input_handle::Bool
    tuples::Base.Vector{XLATupleNode}
    XLATupleNode(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct XLATupleNode

mutable struct XRTExecutionConfig <: ProtoType
    device_ordinal::Int32
    core_index_in_replica::Int32
    execution_instance_key::AbstractString
    rng_seed::UInt32
    release_input_handles::Bool
    release_compilation_handle::Bool
    return_exploded_tuple::Bool
    XRTExecutionConfig(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct XRTExecutionConfig

export DeviceAssignment_ComputationDevice_DeviceMeshCoordinates, DeviceAssignment_ComputationDevice, DeviceAssignment, XLAComputationConfig, XLAComputation, XLAAllocation, XLATupleNode, XRTExecutionConfig
