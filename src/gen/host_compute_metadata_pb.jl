# syntax: proto3
using Compat
using ProtoBuf
import ProtoBuf.meta
import ._ProtoBuf_Top_.tensorflow

mutable struct TensorMetadata <: ProtoType
    _type::Int32
    shape::tensorflow.TensorShapeProto
    TensorMetadata(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct TensorMetadata

mutable struct HostTransferMetadata <: ProtoType
    key::AbstractString
    metadata::Base.Vector{TensorMetadata}
    HostTransferMetadata(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct HostTransferMetadata

mutable struct HostComputeMetadata <: ProtoType
    device_to_host::Base.Vector{HostTransferMetadata}
    host_to_device::Base.Vector{HostTransferMetadata}
    HostComputeMetadata(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct HostComputeMetadata

export TensorMetadata, HostTransferMetadata, HostComputeMetadata
