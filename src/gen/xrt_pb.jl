# syntax: proto3
using Compat
using ProtoBuf
import ProtoBuf.meta
import ._ProtoBuf_Top_.tensorflow
import ._ProtoBuf_Top_.xla

mutable struct XLAComputationConfig <: ProtoType
    num_replicas::Int32
    num_cores_per_replica::Int32
    host_compute_metadata::tensorflow.tf2xla.HostComputeMetadata
    program_shape::xla.ProgramShape
    per_core_program_shape::Base.Vector{xla.ProgramShape}
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
    XRTExecutionConfig(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct XRTExecutionConfig

export XLAComputationConfig, XLAComputation, XLAAllocation, XLATupleNode, XRTExecutionConfig
