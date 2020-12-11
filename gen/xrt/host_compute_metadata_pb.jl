# syntax: proto3
using ProtoBuf
import ProtoBuf.meta
import ._ProtoBuf_Top_.tensorflow

mutable struct TensorMetadata <: ProtoType
    __protobuf_jl_internal_meta::ProtoMeta
    __protobuf_jl_internal_values::Dict{Symbol,Any}
    __protobuf_jl_internal_defaultset::Set{Symbol}

    function TensorMetadata(; kwargs...)
        obj = new(meta(TensorMetadata), Dict{Symbol,Any}(), Set{Symbol}())
        values = obj.__protobuf_jl_internal_values
        symdict = obj.__protobuf_jl_internal_meta.symdict
        for nv in kwargs
            fldname, fldval = nv
            fldtype = symdict[fldname].jtyp
            (fldname in keys(symdict)) || error(string(typeof(obj), " has no field with name ", fldname))
            values[fldname] = isa(fldval, fldtype) ? fldval : convert(fldtype, fldval)
        end
        obj
    end
end # mutable struct TensorMetadata
const __meta_TensorMetadata = Ref{ProtoMeta}()
function meta(::Type{TensorMetadata})
    ProtoBuf.metalock() do
        if !isassigned(__meta_TensorMetadata)
            __meta_TensorMetadata[] = target = ProtoMeta(TensorMetadata)
            allflds = Pair{Symbol,Union{Type,String}}[:_type => Int32, :shape => tensorflow.TensorShapeProto]
            meta(target, TensorMetadata, allflds, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES)
        end
        __meta_TensorMetadata[]
    end
end
function Base.getproperty(obj::TensorMetadata, name::Symbol)
    if name === :_type
        return (obj.__protobuf_jl_internal_values[name])::Int32
    elseif name === :shape
        return (obj.__protobuf_jl_internal_values[name])::tensorflow.TensorShapeProto
    else
        getfield(obj, name)
    end
end

mutable struct HostTransferMetadata <: ProtoType
    __protobuf_jl_internal_meta::ProtoMeta
    __protobuf_jl_internal_values::Dict{Symbol,Any}
    __protobuf_jl_internal_defaultset::Set{Symbol}

    function HostTransferMetadata(; kwargs...)
        obj = new(meta(HostTransferMetadata), Dict{Symbol,Any}(), Set{Symbol}())
        values = obj.__protobuf_jl_internal_values
        symdict = obj.__protobuf_jl_internal_meta.symdict
        for nv in kwargs
            fldname, fldval = nv
            fldtype = symdict[fldname].jtyp
            (fldname in keys(symdict)) || error(string(typeof(obj), " has no field with name ", fldname))
            values[fldname] = isa(fldval, fldtype) ? fldval : convert(fldtype, fldval)
        end
        obj
    end
end # mutable struct HostTransferMetadata
const __meta_HostTransferMetadata = Ref{ProtoMeta}()
function meta(::Type{HostTransferMetadata})
    ProtoBuf.metalock() do
        if !isassigned(__meta_HostTransferMetadata)
            __meta_HostTransferMetadata[] = target = ProtoMeta(HostTransferMetadata)
            allflds = Pair{Symbol,Union{Type,String}}[:key => AbstractString, :metadata => Base.Vector{TensorMetadata}]
            meta(target, HostTransferMetadata, allflds, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES)
        end
        __meta_HostTransferMetadata[]
    end
end
function Base.getproperty(obj::HostTransferMetadata, name::Symbol)
    if name === :key
        return (obj.__protobuf_jl_internal_values[name])::AbstractString
    elseif name === :metadata
        return (obj.__protobuf_jl_internal_values[name])::Base.Vector{TensorMetadata}
    else
        getfield(obj, name)
    end
end

mutable struct HostComputeMetadata <: ProtoType
    __protobuf_jl_internal_meta::ProtoMeta
    __protobuf_jl_internal_values::Dict{Symbol,Any}
    __protobuf_jl_internal_defaultset::Set{Symbol}

    function HostComputeMetadata(; kwargs...)
        obj = new(meta(HostComputeMetadata), Dict{Symbol,Any}(), Set{Symbol}())
        values = obj.__protobuf_jl_internal_values
        symdict = obj.__protobuf_jl_internal_meta.symdict
        for nv in kwargs
            fldname, fldval = nv
            fldtype = symdict[fldname].jtyp
            (fldname in keys(symdict)) || error(string(typeof(obj), " has no field with name ", fldname))
            values[fldname] = isa(fldval, fldtype) ? fldval : convert(fldtype, fldval)
        end
        obj
    end
end # mutable struct HostComputeMetadata
const __meta_HostComputeMetadata = Ref{ProtoMeta}()
function meta(::Type{HostComputeMetadata})
    ProtoBuf.metalock() do
        if !isassigned(__meta_HostComputeMetadata)
            __meta_HostComputeMetadata[] = target = ProtoMeta(HostComputeMetadata)
            allflds = Pair{Symbol,Union{Type,String}}[:device_to_host => Base.Vector{HostTransferMetadata}, :host_to_device => Base.Vector{HostTransferMetadata}]
            meta(target, HostComputeMetadata, allflds, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES)
        end
        __meta_HostComputeMetadata[]
    end
end
function Base.getproperty(obj::HostComputeMetadata, name::Symbol)
    if name === :device_to_host
        return (obj.__protobuf_jl_internal_values[name])::Base.Vector{HostTransferMetadata}
    elseif name === :host_to_device
        return (obj.__protobuf_jl_internal_values[name])::Base.Vector{HostTransferMetadata}
    else
        getfield(obj, name)
    end
end

export TensorMetadata, HostTransferMetadata, HostComputeMetadata
