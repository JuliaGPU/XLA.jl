# syntax: proto3
using ProtoBuf
import ProtoBuf.meta

mutable struct TopologyProto <: ProtoType
    mesh_shape::Base.Vector{Int32}
    num_tasks::Int32
    num_tpu_devices_per_task::Int32
    device_coordinates::Base.Vector{Int32}
    TopologyProto(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct TopologyProto
const __pack_TopologyProto = Symbol[:mesh_shape,:device_coordinates]
meta(t::Type{TopologyProto}) = meta(t, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, true, __pack_TopologyProto, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES, ProtoBuf.DEF_FIELD_TYPES)

export TopologyProto
