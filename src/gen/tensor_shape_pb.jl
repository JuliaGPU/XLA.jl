# syntax: proto3
using ProtoBuf
import ProtoBuf.meta

mutable struct TensorShapeProto_Dim <: ProtoType
    __protobuf_jl_internal_meta::ProtoMeta
    __protobuf_jl_internal_values::Dict{Symbol,Any}
    __protobuf_jl_internal_defaultset::Set{Symbol}

    function TensorShapeProto_Dim(; kwargs...)
        obj = new(meta(TensorShapeProto_Dim), Dict{Symbol,Any}(), Set{Symbol}())
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
end # mutable struct TensorShapeProto_Dim
const __meta_TensorShapeProto_Dim = Ref{ProtoMeta}()
function meta(::Type{TensorShapeProto_Dim})
    ProtoBuf.metalock() do
        if !isassigned(__meta_TensorShapeProto_Dim)
            __meta_TensorShapeProto_Dim[] = target = ProtoMeta(TensorShapeProto_Dim)
            allflds = Pair{Symbol,Union{Type,String}}[:size => Int64, :name => AbstractString]
            meta(target, TensorShapeProto_Dim, allflds, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES)
        end
        __meta_TensorShapeProto_Dim[]
    end
end
function Base.getproperty(obj::TensorShapeProto_Dim, name::Symbol)
    if name === :size
        return (obj.__protobuf_jl_internal_values[name])::Int64
    elseif name === :name
        return (obj.__protobuf_jl_internal_values[name])::AbstractString
    else
        getfield(obj, name)
    end
end

mutable struct TensorShapeProto <: ProtoType
    __protobuf_jl_internal_meta::ProtoMeta
    __protobuf_jl_internal_values::Dict{Symbol,Any}
    __protobuf_jl_internal_defaultset::Set{Symbol}

    function TensorShapeProto(; kwargs...)
        obj = new(meta(TensorShapeProto), Dict{Symbol,Any}(), Set{Symbol}())
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
end # mutable struct TensorShapeProto
const __meta_TensorShapeProto = Ref{ProtoMeta}()
function meta(::Type{TensorShapeProto})
    ProtoBuf.metalock() do
        if !isassigned(__meta_TensorShapeProto)
            __meta_TensorShapeProto[] = target = ProtoMeta(TensorShapeProto)
            fnum = Int[2,3]
            allflds = Pair{Symbol,Union{Type,String}}[:dim => Base.Vector{TensorShapeProto_Dim}, :unknown_rank => Bool]
            meta(target, TensorShapeProto, allflds, ProtoBuf.DEF_REQ, fnum, ProtoBuf.DEF_VAL, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES)
        end
        __meta_TensorShapeProto[]
    end
end
function Base.getproperty(obj::TensorShapeProto, name::Symbol)
    if name === :dim
        return (obj.__protobuf_jl_internal_values[name])::Base.Vector{TensorShapeProto_Dim}
    elseif name === :unknown_rank
        return (obj.__protobuf_jl_internal_values[name])::Bool
    else
        getfield(obj, name)
    end
end

export TensorShapeProto_Dim, TensorShapeProto
