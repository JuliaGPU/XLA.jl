# syntax: proto3
using ProtoBuf
import ProtoBuf.meta

const PrimitiveType = (;[
    Symbol("PRIMITIVE_TYPE_INVALID") => Int32(0),
    Symbol("PRED") => Int32(1),
    Symbol("S8") => Int32(2),
    Symbol("S16") => Int32(3),
    Symbol("S32") => Int32(4),
    Symbol("S64") => Int32(5),
    Symbol("U8") => Int32(6),
    Symbol("U16") => Int32(7),
    Symbol("U32") => Int32(8),
    Symbol("U64") => Int32(9),
    Symbol("F16") => Int32(10),
    Symbol("F32") => Int32(11),
    Symbol("BF16") => Int32(16),
    Symbol("F64") => Int32(12),
    Symbol("C64") => Int32(15),
    Symbol("C128") => Int32(18),
    Symbol("TUPLE") => Int32(13),
    Symbol("OPAQUE_TYPE") => Int32(14),
    Symbol("TOKEN") => Int32(17),
]...)

const Format = (;[
    Symbol("INVALID_FORMAT") => Int32(0),
    Symbol("DENSE") => Int32(1),
]...)

const ProfileType = (;[
    Symbol("INVALID") => Int32(0),
    Symbol("WINDOW") => Int32(1),
    Symbol("FLAG") => Int32(2),
    Symbol("INTEGER") => Int32(3),
]...)

const PaddingType = (;[
    Symbol("PADDING_INVALID") => Int32(0),
    Symbol("PADDING_VALID") => Int32(1),
    Symbol("PADDING_SAME") => Int32(2),
]...)

const FftType = (;[
    Symbol("FFT") => Int32(0),
    Symbol("IFFT") => Int32(1),
    Symbol("RFFT") => Int32(2),
    Symbol("IRFFT") => Int32(3),
]...)

const RandomDistribution = (;[
    Symbol("RNG_INVALID") => Int32(0),
    Symbol("RNG_UNIFORM") => Int32(1),
    Symbol("RNG_NORMAL") => Int32(2),
]...)

const RandomAlgorithm = (;[
    Symbol("RNG_DEFAULT") => Int32(0),
    Symbol("RNG_THREE_FRY") => Int32(1),
    Symbol("RNG_PHILOX") => Int32(2),
]...)

mutable struct PaddingConfig_PaddingConfigDimension <: ProtoType
    __protobuf_jl_internal_meta::ProtoMeta
    __protobuf_jl_internal_values::Dict{Symbol,Any}
    __protobuf_jl_internal_defaultset::Set{Symbol}

    function PaddingConfig_PaddingConfigDimension(; kwargs...)
        obj = new(meta(PaddingConfig_PaddingConfigDimension), Dict{Symbol,Any}(), Set{Symbol}())
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
end # mutable struct PaddingConfig_PaddingConfigDimension
const __meta_PaddingConfig_PaddingConfigDimension = Ref{ProtoMeta}()
function meta(::Type{PaddingConfig_PaddingConfigDimension})
    ProtoBuf.metalock() do
        if !isassigned(__meta_PaddingConfig_PaddingConfigDimension)
            __meta_PaddingConfig_PaddingConfigDimension[] = target = ProtoMeta(PaddingConfig_PaddingConfigDimension)
            allflds = Pair{Symbol,Union{Type,String}}[:edge_padding_low => Int64, :edge_padding_high => Int64, :interior_padding => Int64]
            meta(target, PaddingConfig_PaddingConfigDimension, allflds, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES)
        end
        __meta_PaddingConfig_PaddingConfigDimension[]
    end
end
function Base.getproperty(obj::PaddingConfig_PaddingConfigDimension, name::Symbol)
    if name === :edge_padding_low
        return (obj.__protobuf_jl_internal_values[name])::Int64
    elseif name === :edge_padding_high
        return (obj.__protobuf_jl_internal_values[name])::Int64
    elseif name === :interior_padding
        return (obj.__protobuf_jl_internal_values[name])::Int64
    else
        getfield(obj, name)
    end
end

mutable struct PaddingConfig <: ProtoType
    __protobuf_jl_internal_meta::ProtoMeta
    __protobuf_jl_internal_values::Dict{Symbol,Any}
    __protobuf_jl_internal_defaultset::Set{Symbol}

    function PaddingConfig(; kwargs...)
        obj = new(meta(PaddingConfig), Dict{Symbol,Any}(), Set{Symbol}())
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
end # mutable struct PaddingConfig
const __meta_PaddingConfig = Ref{ProtoMeta}()
function meta(::Type{PaddingConfig})
    ProtoBuf.metalock() do
        if !isassigned(__meta_PaddingConfig)
            __meta_PaddingConfig[] = target = ProtoMeta(PaddingConfig)
            allflds = Pair{Symbol,Union{Type,String}}[:dimensions => Base.Vector{PaddingConfig_PaddingConfigDimension}]
            meta(target, PaddingConfig, allflds, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES)
        end
        __meta_PaddingConfig[]
    end
end
function Base.getproperty(obj::PaddingConfig, name::Symbol)
    if name === :dimensions
        return (obj.__protobuf_jl_internal_values[name])::Base.Vector{PaddingConfig_PaddingConfigDimension}
    else
        getfield(obj, name)
    end
end

mutable struct TileProto <: ProtoType
    __protobuf_jl_internal_meta::ProtoMeta
    __protobuf_jl_internal_values::Dict{Symbol,Any}
    __protobuf_jl_internal_defaultset::Set{Symbol}

    function TileProto(; kwargs...)
        obj = new(meta(TileProto), Dict{Symbol,Any}(), Set{Symbol}())
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
end # mutable struct TileProto
const __meta_TileProto = Ref{ProtoMeta}()
function meta(::Type{TileProto})
    ProtoBuf.metalock() do
        if !isassigned(__meta_TileProto)
            __meta_TileProto[] = target = ProtoMeta(TileProto)
            pack = Symbol[:dimensions]
            allflds = Pair{Symbol,Union{Type,String}}[:dimensions => Base.Vector{Int64}]
            meta(target, TileProto, allflds, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, pack, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES)
        end
        __meta_TileProto[]
    end
end
function Base.getproperty(obj::TileProto, name::Symbol)
    if name === :dimensions
        return (obj.__protobuf_jl_internal_values[name])::Base.Vector{Int64}
    else
        getfield(obj, name)
    end
end

mutable struct LayoutProto <: ProtoType
    __protobuf_jl_internal_meta::ProtoMeta
    __protobuf_jl_internal_values::Dict{Symbol,Any}
    __protobuf_jl_internal_defaultset::Set{Symbol}

    function LayoutProto(; kwargs...)
        obj = new(meta(LayoutProto), Dict{Symbol,Any}(), Set{Symbol}())
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
end # mutable struct LayoutProto
const __meta_LayoutProto = Ref{ProtoMeta}()
function meta(::Type{LayoutProto})
    ProtoBuf.metalock() do
        if !isassigned(__meta_LayoutProto)
            __meta_LayoutProto[] = target = ProtoMeta(LayoutProto)
            fnum = Int[4,1,6,7,8]
            pack = Symbol[:minor_to_major]
            allflds = Pair{Symbol,Union{Type,String}}[:format => Int32, :minor_to_major => Base.Vector{Int64}, :tiles => Base.Vector{TileProto}, :element_size_in_bits => Int64, :memory_space => Int64]
            meta(target, LayoutProto, allflds, ProtoBuf.DEF_REQ, fnum, ProtoBuf.DEF_VAL, pack, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES)
        end
        __meta_LayoutProto[]
    end
end
function Base.getproperty(obj::LayoutProto, name::Symbol)
    if name === :format
        return (obj.__protobuf_jl_internal_values[name])::Int32
    elseif name === :minor_to_major
        return (obj.__protobuf_jl_internal_values[name])::Base.Vector{Int64}
    elseif name === :tiles
        return (obj.__protobuf_jl_internal_values[name])::Base.Vector{TileProto}
    elseif name === :element_size_in_bits
        return (obj.__protobuf_jl_internal_values[name])::Int64
    elseif name === :memory_space
        return (obj.__protobuf_jl_internal_values[name])::Int64
    else
        getfield(obj, name)
    end
end

mutable struct ShapeProto <: ProtoType
    __protobuf_jl_internal_meta::ProtoMeta
    __protobuf_jl_internal_values::Dict{Symbol,Any}
    __protobuf_jl_internal_defaultset::Set{Symbol}

    function ShapeProto(; kwargs...)
        obj = new(meta(ShapeProto), Dict{Symbol,Any}(), Set{Symbol}())
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
end # mutable struct ShapeProto
const __meta_ShapeProto = Ref{ProtoMeta}()
function meta(::Type{ShapeProto})
    ProtoBuf.metalock() do
        if !isassigned(__meta_ShapeProto)
            __meta_ShapeProto[] = target = ProtoMeta(ShapeProto)
            fnum = Int[2,3,4,5,6]
            pack = Symbol[:dimensions,:is_dynamic_dimension]
            allflds = Pair{Symbol,Union{Type,String}}[:element_type => Int32, :dimensions => Base.Vector{Int64}, :tuple_shapes => Base.Vector{ShapeProto}, :layout => LayoutProto, :is_dynamic_dimension => Base.Vector{Bool}]
            meta(target, ShapeProto, allflds, ProtoBuf.DEF_REQ, fnum, ProtoBuf.DEF_VAL, pack, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES)
        end
        __meta_ShapeProto[]
    end
end
function Base.getproperty(obj::ShapeProto, name::Symbol)
    if name === :element_type
        return (obj.__protobuf_jl_internal_values[name])::Int32
    elseif name === :dimensions
        return (obj.__protobuf_jl_internal_values[name])::Base.Vector{Int64}
    elseif name === :tuple_shapes
        return (obj.__protobuf_jl_internal_values[name])::Base.Vector{ShapeProto}
    elseif name === :layout
        return (obj.__protobuf_jl_internal_values[name])::LayoutProto
    elseif name === :is_dynamic_dimension
        return (obj.__protobuf_jl_internal_values[name])::Base.Vector{Bool}
    else
        getfield(obj, name)
    end
end

mutable struct ProgramShapeProto <: ProtoType
    __protobuf_jl_internal_meta::ProtoMeta
    __protobuf_jl_internal_values::Dict{Symbol,Any}
    __protobuf_jl_internal_defaultset::Set{Symbol}

    function ProgramShapeProto(; kwargs...)
        obj = new(meta(ProgramShapeProto), Dict{Symbol,Any}(), Set{Symbol}())
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
end # mutable struct ProgramShapeProto
const __meta_ProgramShapeProto = Ref{ProtoMeta}()
function meta(::Type{ProgramShapeProto})
    ProtoBuf.metalock() do
        if !isassigned(__meta_ProgramShapeProto)
            __meta_ProgramShapeProto[] = target = ProtoMeta(ProgramShapeProto)
            allflds = Pair{Symbol,Union{Type,String}}[:parameters => Base.Vector{ShapeProto}, :result => ShapeProto, :parameter_names => Base.Vector{AbstractString}]
            meta(target, ProgramShapeProto, allflds, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES)
        end
        __meta_ProgramShapeProto[]
    end
end
function Base.getproperty(obj::ProgramShapeProto, name::Symbol)
    if name === :parameters
        return (obj.__protobuf_jl_internal_values[name])::Base.Vector{ShapeProto}
    elseif name === :result
        return (obj.__protobuf_jl_internal_values[name])::ShapeProto
    elseif name === :parameter_names
        return (obj.__protobuf_jl_internal_values[name])::Base.Vector{AbstractString}
    else
        getfield(obj, name)
    end
end

mutable struct ComputationStats <: ProtoType
    __protobuf_jl_internal_meta::ProtoMeta
    __protobuf_jl_internal_values::Dict{Symbol,Any}
    __protobuf_jl_internal_defaultset::Set{Symbol}

    function ComputationStats(; kwargs...)
        obj = new(meta(ComputationStats), Dict{Symbol,Any}(), Set{Symbol}())
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
end # mutable struct ComputationStats
const __meta_ComputationStats = Ref{ProtoMeta}()
function meta(::Type{ComputationStats})
    ProtoBuf.metalock() do
        if !isassigned(__meta_ComputationStats)
            __meta_ComputationStats[] = target = ProtoMeta(ComputationStats)
            allflds = Pair{Symbol,Union{Type,String}}[:flop_count => Float64, :transcendental_count => Float64]
            meta(target, ComputationStats, allflds, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES)
        end
        __meta_ComputationStats[]
    end
end
function Base.getproperty(obj::ComputationStats, name::Symbol)
    if name === :flop_count
        return (obj.__protobuf_jl_internal_values[name])::Float64
    elseif name === :transcendental_count
        return (obj.__protobuf_jl_internal_values[name])::Float64
    else
        getfield(obj, name)
    end
end

mutable struct OpMetadata <: ProtoType
    __protobuf_jl_internal_meta::ProtoMeta
    __protobuf_jl_internal_values::Dict{Symbol,Any}
    __protobuf_jl_internal_defaultset::Set{Symbol}

    function OpMetadata(; kwargs...)
        obj = new(meta(OpMetadata), Dict{Symbol,Any}(), Set{Symbol}())
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
end # mutable struct OpMetadata
const __meta_OpMetadata = Ref{ProtoMeta}()
function meta(::Type{OpMetadata})
    ProtoBuf.metalock() do
        if !isassigned(__meta_OpMetadata)
            __meta_OpMetadata[] = target = ProtoMeta(OpMetadata)
            pack = Symbol[:profile_type]
            allflds = Pair{Symbol,Union{Type,String}}[:op_type => AbstractString, :op_name => AbstractString, :source_file => AbstractString, :source_line => Int32, :profile_type => Base.Vector{Int32}]
            meta(target, OpMetadata, allflds, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, pack, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES)
        end
        __meta_OpMetadata[]
    end
end
function Base.getproperty(obj::OpMetadata, name::Symbol)
    if name === :op_type
        return (obj.__protobuf_jl_internal_values[name])::AbstractString
    elseif name === :op_name
        return (obj.__protobuf_jl_internal_values[name])::AbstractString
    elseif name === :source_file
        return (obj.__protobuf_jl_internal_values[name])::AbstractString
    elseif name === :source_line
        return (obj.__protobuf_jl_internal_values[name])::Int32
    elseif name === :profile_type
        return (obj.__protobuf_jl_internal_values[name])::Base.Vector{Int32}
    else
        getfield(obj, name)
    end
end

mutable struct ExecutionProfile <: ProtoType
    __protobuf_jl_internal_meta::ProtoMeta
    __protobuf_jl_internal_values::Dict{Symbol,Any}
    __protobuf_jl_internal_defaultset::Set{Symbol}

    function ExecutionProfile(; kwargs...)
        obj = new(meta(ExecutionProfile), Dict{Symbol,Any}(), Set{Symbol}())
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
end # mutable struct ExecutionProfile
const __meta_ExecutionProfile = Ref{ProtoMeta}()
function meta(::Type{ExecutionProfile})
    ProtoBuf.metalock() do
        if !isassigned(__meta_ExecutionProfile)
            __meta_ExecutionProfile[] = target = ProtoMeta(ExecutionProfile)
            allflds = Pair{Symbol,Union{Type,String}}[:compilation_cache_hit => Bool, :compile_time_ms => Int64, :compute_cycle_count => Int64, :compute_time_ns => Int64, :compute_and_transfer_time_ns => Int64, :executable_size_in_bytes => Int64, :profile_cache_hit => Bool]
            meta(target, ExecutionProfile, allflds, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES)
        end
        __meta_ExecutionProfile[]
    end
end
function Base.getproperty(obj::ExecutionProfile, name::Symbol)
    if name === :compilation_cache_hit
        return (obj.__protobuf_jl_internal_values[name])::Bool
    elseif name === :compile_time_ms
        return (obj.__protobuf_jl_internal_values[name])::Int64
    elseif name === :compute_cycle_count
        return (obj.__protobuf_jl_internal_values[name])::Int64
    elseif name === :compute_time_ns
        return (obj.__protobuf_jl_internal_values[name])::Int64
    elseif name === :compute_and_transfer_time_ns
        return (obj.__protobuf_jl_internal_values[name])::Int64
    elseif name === :executable_size_in_bytes
        return (obj.__protobuf_jl_internal_values[name])::Int64
    elseif name === :profile_cache_hit
        return (obj.__protobuf_jl_internal_values[name])::Bool
    else
        getfield(obj, name)
    end
end

mutable struct ExecutionHandle <: ProtoType
    __protobuf_jl_internal_meta::ProtoMeta
    __protobuf_jl_internal_values::Dict{Symbol,Any}
    __protobuf_jl_internal_defaultset::Set{Symbol}

    function ExecutionHandle(; kwargs...)
        obj = new(meta(ExecutionHandle), Dict{Symbol,Any}(), Set{Symbol}())
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
end # mutable struct ExecutionHandle
const __meta_ExecutionHandle = Ref{ProtoMeta}()
function meta(::Type{ExecutionHandle})
    ProtoBuf.metalock() do
        if !isassigned(__meta_ExecutionHandle)
            __meta_ExecutionHandle[] = target = ProtoMeta(ExecutionHandle)
            allflds = Pair{Symbol,Union{Type,String}}[:handle => Int64]
            meta(target, ExecutionHandle, allflds, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES)
        end
        __meta_ExecutionHandle[]
    end
end
function Base.getproperty(obj::ExecutionHandle, name::Symbol)
    if name === :handle
        return (obj.__protobuf_jl_internal_values[name])::Int64
    else
        getfield(obj, name)
    end
end

mutable struct GlobalDataHandle <: ProtoType
    __protobuf_jl_internal_meta::ProtoMeta
    __protobuf_jl_internal_values::Dict{Symbol,Any}
    __protobuf_jl_internal_defaultset::Set{Symbol}

    function GlobalDataHandle(; kwargs...)
        obj = new(meta(GlobalDataHandle), Dict{Symbol,Any}(), Set{Symbol}())
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
end # mutable struct GlobalDataHandle
const __meta_GlobalDataHandle = Ref{ProtoMeta}()
function meta(::Type{GlobalDataHandle})
    ProtoBuf.metalock() do
        if !isassigned(__meta_GlobalDataHandle)
            __meta_GlobalDataHandle[] = target = ProtoMeta(GlobalDataHandle)
            allflds = Pair{Symbol,Union{Type,String}}[:handle => Int64]
            meta(target, GlobalDataHandle, allflds, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES)
        end
        __meta_GlobalDataHandle[]
    end
end
function Base.getproperty(obj::GlobalDataHandle, name::Symbol)
    if name === :handle
        return (obj.__protobuf_jl_internal_values[name])::Int64
    else
        getfield(obj, name)
    end
end

mutable struct DeviceHandle <: ProtoType
    __protobuf_jl_internal_meta::ProtoMeta
    __protobuf_jl_internal_values::Dict{Symbol,Any}
    __protobuf_jl_internal_defaultset::Set{Symbol}

    function DeviceHandle(; kwargs...)
        obj = new(meta(DeviceHandle), Dict{Symbol,Any}(), Set{Symbol}())
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
end # mutable struct DeviceHandle
const __meta_DeviceHandle = Ref{ProtoMeta}()
function meta(::Type{DeviceHandle})
    ProtoBuf.metalock() do
        if !isassigned(__meta_DeviceHandle)
            __meta_DeviceHandle[] = target = ProtoMeta(DeviceHandle)
            allflds = Pair{Symbol,Union{Type,String}}[:handle => Int64, :device_count => Int64]
            meta(target, DeviceHandle, allflds, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES)
        end
        __meta_DeviceHandle[]
    end
end
function Base.getproperty(obj::DeviceHandle, name::Symbol)
    if name === :handle
        return (obj.__protobuf_jl_internal_values[name])::Int64
    elseif name === :device_count
        return (obj.__protobuf_jl_internal_values[name])::Int64
    else
        getfield(obj, name)
    end
end

const ChannelHandle_ChannelType = (;[
    Symbol("CHANNEL_TYPE_INVALID") => Int32(0),
    Symbol("DEVICE_TO_DEVICE") => Int32(1),
    Symbol("DEVICE_TO_HOST") => Int32(2),
    Symbol("HOST_TO_DEVICE") => Int32(3),
]...)

mutable struct ChannelHandle <: ProtoType
    __protobuf_jl_internal_meta::ProtoMeta
    __protobuf_jl_internal_values::Dict{Symbol,Any}
    __protobuf_jl_internal_defaultset::Set{Symbol}

    function ChannelHandle(; kwargs...)
        obj = new(meta(ChannelHandle), Dict{Symbol,Any}(), Set{Symbol}())
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
end # mutable struct ChannelHandle
const __meta_ChannelHandle = Ref{ProtoMeta}()
function meta(::Type{ChannelHandle})
    ProtoBuf.metalock() do
        if !isassigned(__meta_ChannelHandle)
            __meta_ChannelHandle[] = target = ProtoMeta(ChannelHandle)
            allflds = Pair{Symbol,Union{Type,String}}[:handle => Int64, :_type => Int32]
            meta(target, ChannelHandle, allflds, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES)
        end
        __meta_ChannelHandle[]
    end
end
function Base.getproperty(obj::ChannelHandle, name::Symbol)
    if name === :handle
        return (obj.__protobuf_jl_internal_values[name])::Int64
    elseif name === :_type
        return (obj.__protobuf_jl_internal_values[name])::Int32
    else
        getfield(obj, name)
    end
end

mutable struct DeviceAssignmentProto_ComputationDevice <: ProtoType
    __protobuf_jl_internal_meta::ProtoMeta
    __protobuf_jl_internal_values::Dict{Symbol,Any}
    __protobuf_jl_internal_defaultset::Set{Symbol}

    function DeviceAssignmentProto_ComputationDevice(; kwargs...)
        obj = new(meta(DeviceAssignmentProto_ComputationDevice), Dict{Symbol,Any}(), Set{Symbol}())
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
end # mutable struct DeviceAssignmentProto_ComputationDevice
const __meta_DeviceAssignmentProto_ComputationDevice = Ref{ProtoMeta}()
function meta(::Type{DeviceAssignmentProto_ComputationDevice})
    ProtoBuf.metalock() do
        if !isassigned(__meta_DeviceAssignmentProto_ComputationDevice)
            __meta_DeviceAssignmentProto_ComputationDevice[] = target = ProtoMeta(DeviceAssignmentProto_ComputationDevice)
            pack = Symbol[:replica_device_ids]
            allflds = Pair{Symbol,Union{Type,String}}[:replica_device_ids => Base.Vector{Int32}]
            meta(target, DeviceAssignmentProto_ComputationDevice, allflds, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, pack, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES)
        end
        __meta_DeviceAssignmentProto_ComputationDevice[]
    end
end
function Base.getproperty(obj::DeviceAssignmentProto_ComputationDevice, name::Symbol)
    if name === :replica_device_ids
        return (obj.__protobuf_jl_internal_values[name])::Base.Vector{Int32}
    else
        getfield(obj, name)
    end
end

mutable struct DeviceAssignmentProto <: ProtoType
    __protobuf_jl_internal_meta::ProtoMeta
    __protobuf_jl_internal_values::Dict{Symbol,Any}
    __protobuf_jl_internal_defaultset::Set{Symbol}

    function DeviceAssignmentProto(; kwargs...)
        obj = new(meta(DeviceAssignmentProto), Dict{Symbol,Any}(), Set{Symbol}())
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
end # mutable struct DeviceAssignmentProto
const __meta_DeviceAssignmentProto = Ref{ProtoMeta}()
function meta(::Type{DeviceAssignmentProto})
    ProtoBuf.metalock() do
        if !isassigned(__meta_DeviceAssignmentProto)
            __meta_DeviceAssignmentProto[] = target = ProtoMeta(DeviceAssignmentProto)
            allflds = Pair{Symbol,Union{Type,String}}[:replica_count => Int32, :computation_count => Int32, :computation_devices => Base.Vector{DeviceAssignmentProto_ComputationDevice}]
            meta(target, DeviceAssignmentProto, allflds, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES)
        end
        __meta_DeviceAssignmentProto[]
    end
end
function Base.getproperty(obj::DeviceAssignmentProto, name::Symbol)
    if name === :replica_count
        return (obj.__protobuf_jl_internal_values[name])::Int32
    elseif name === :computation_count
        return (obj.__protobuf_jl_internal_values[name])::Int32
    elseif name === :computation_devices
        return (obj.__protobuf_jl_internal_values[name])::Base.Vector{DeviceAssignmentProto_ComputationDevice}
    else
        getfield(obj, name)
    end
end

mutable struct LiteralProto <: ProtoType
    __protobuf_jl_internal_meta::ProtoMeta
    __protobuf_jl_internal_values::Dict{Symbol,Any}
    __protobuf_jl_internal_defaultset::Set{Symbol}

    function LiteralProto(; kwargs...)
        obj = new(meta(LiteralProto), Dict{Symbol,Any}(), Set{Symbol}())
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
end # mutable struct LiteralProto
const __meta_LiteralProto = Ref{ProtoMeta}()
function meta(::Type{LiteralProto})
    ProtoBuf.metalock() do
        if !isassigned(__meta_LiteralProto)
            __meta_LiteralProto[] = target = ProtoMeta(LiteralProto)
            fnum = Int[1,2,15,3,4,5,6,7,8,9,12,18,10,11,13,16,17,14]
            pack = Symbol[:preds,:s32s,:s64s,:u32s,:u64s,:f32s,:f64s,:c64s,:c128s,:sparse_indices]
            allflds = Pair{Symbol,Union{Type,String}}[:shape => ShapeProto, :preds => Base.Vector{Bool}, :s8s => Array{UInt8,1}, :u8s => Array{UInt8,1}, :s32s => Base.Vector{Int32}, :s64s => Base.Vector{Int64}, :u32s => Base.Vector{UInt32}, :u64s => Base.Vector{UInt64}, :f32s => Base.Vector{Float32}, :f64s => Base.Vector{Float64}, :c64s => Base.Vector{Float32}, :c128s => Base.Vector{Float64}, :tuple_literals => Base.Vector{LiteralProto}, :f16s => Array{UInt8,1}, :bf16s => Array{UInt8,1}, :u16s => Array{UInt8,1}, :s16s => Array{UInt8,1}, :sparse_indices => Base.Vector{Int64}]
            meta(target, LiteralProto, allflds, ProtoBuf.DEF_REQ, fnum, ProtoBuf.DEF_VAL, pack, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES)
        end
        __meta_LiteralProto[]
    end
end
function Base.getproperty(obj::LiteralProto, name::Symbol)
    if name === :shape
        return (obj.__protobuf_jl_internal_values[name])::ShapeProto
    elseif name === :preds
        return (obj.__protobuf_jl_internal_values[name])::Base.Vector{Bool}
    elseif name === :s8s
        return (obj.__protobuf_jl_internal_values[name])::Array{UInt8,1}
    elseif name === :u8s
        return (obj.__protobuf_jl_internal_values[name])::Array{UInt8,1}
    elseif name === :s32s
        return (obj.__protobuf_jl_internal_values[name])::Base.Vector{Int32}
    elseif name === :s64s
        return (obj.__protobuf_jl_internal_values[name])::Base.Vector{Int64}
    elseif name === :u32s
        return (obj.__protobuf_jl_internal_values[name])::Base.Vector{UInt32}
    elseif name === :u64s
        return (obj.__protobuf_jl_internal_values[name])::Base.Vector{UInt64}
    elseif name === :f32s
        return (obj.__protobuf_jl_internal_values[name])::Base.Vector{Float32}
    elseif name === :f64s
        return (obj.__protobuf_jl_internal_values[name])::Base.Vector{Float64}
    elseif name === :c64s
        return (obj.__protobuf_jl_internal_values[name])::Base.Vector{Float32}
    elseif name === :c128s
        return (obj.__protobuf_jl_internal_values[name])::Base.Vector{Float64}
    elseif name === :tuple_literals
        return (obj.__protobuf_jl_internal_values[name])::Base.Vector{LiteralProto}
    elseif name === :f16s
        return (obj.__protobuf_jl_internal_values[name])::Array{UInt8,1}
    elseif name === :bf16s
        return (obj.__protobuf_jl_internal_values[name])::Array{UInt8,1}
    elseif name === :u16s
        return (obj.__protobuf_jl_internal_values[name])::Array{UInt8,1}
    elseif name === :s16s
        return (obj.__protobuf_jl_internal_values[name])::Array{UInt8,1}
    elseif name === :sparse_indices
        return (obj.__protobuf_jl_internal_values[name])::Base.Vector{Int64}
    else
        getfield(obj, name)
    end
end

mutable struct WindowDimension <: ProtoType
    __protobuf_jl_internal_meta::ProtoMeta
    __protobuf_jl_internal_values::Dict{Symbol,Any}
    __protobuf_jl_internal_defaultset::Set{Symbol}

    function WindowDimension(; kwargs...)
        obj = new(meta(WindowDimension), Dict{Symbol,Any}(), Set{Symbol}())
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
end # mutable struct WindowDimension
const __meta_WindowDimension = Ref{ProtoMeta}()
function meta(::Type{WindowDimension})
    ProtoBuf.metalock() do
        if !isassigned(__meta_WindowDimension)
            __meta_WindowDimension[] = target = ProtoMeta(WindowDimension)
            allflds = Pair{Symbol,Union{Type,String}}[:size => Int64, :stride => Int64, :padding_low => Int64, :padding_high => Int64, :window_dilation => Int64, :base_dilation => Int64, :window_reversal => Bool]
            meta(target, WindowDimension, allflds, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES)
        end
        __meta_WindowDimension[]
    end
end
function Base.getproperty(obj::WindowDimension, name::Symbol)
    if name === :size
        return (obj.__protobuf_jl_internal_values[name])::Int64
    elseif name === :stride
        return (obj.__protobuf_jl_internal_values[name])::Int64
    elseif name === :padding_low
        return (obj.__protobuf_jl_internal_values[name])::Int64
    elseif name === :padding_high
        return (obj.__protobuf_jl_internal_values[name])::Int64
    elseif name === :window_dilation
        return (obj.__protobuf_jl_internal_values[name])::Int64
    elseif name === :base_dilation
        return (obj.__protobuf_jl_internal_values[name])::Int64
    elseif name === :window_reversal
        return (obj.__protobuf_jl_internal_values[name])::Bool
    else
        getfield(obj, name)
    end
end

mutable struct Window <: ProtoType
    __protobuf_jl_internal_meta::ProtoMeta
    __protobuf_jl_internal_values::Dict{Symbol,Any}
    __protobuf_jl_internal_defaultset::Set{Symbol}

    function Window(; kwargs...)
        obj = new(meta(Window), Dict{Symbol,Any}(), Set{Symbol}())
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
end # mutable struct Window
const __meta_Window = Ref{ProtoMeta}()
function meta(::Type{Window})
    ProtoBuf.metalock() do
        if !isassigned(__meta_Window)
            __meta_Window[] = target = ProtoMeta(Window)
            allflds = Pair{Symbol,Union{Type,String}}[:dimensions => Base.Vector{WindowDimension}]
            meta(target, Window, allflds, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES)
        end
        __meta_Window[]
    end
end
function Base.getproperty(obj::Window, name::Symbol)
    if name === :dimensions
        return (obj.__protobuf_jl_internal_values[name])::Base.Vector{WindowDimension}
    else
        getfield(obj, name)
    end
end

mutable struct GatherDimensionNumbers <: ProtoType
    __protobuf_jl_internal_meta::ProtoMeta
    __protobuf_jl_internal_values::Dict{Symbol,Any}
    __protobuf_jl_internal_defaultset::Set{Symbol}

    function GatherDimensionNumbers(; kwargs...)
        obj = new(meta(GatherDimensionNumbers), Dict{Symbol,Any}(), Set{Symbol}())
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
end # mutable struct GatherDimensionNumbers
const __meta_GatherDimensionNumbers = Ref{ProtoMeta}()
function meta(::Type{GatherDimensionNumbers})
    ProtoBuf.metalock() do
        if !isassigned(__meta_GatherDimensionNumbers)
            __meta_GatherDimensionNumbers[] = target = ProtoMeta(GatherDimensionNumbers)
            pack = Symbol[:offset_dims,:collapsed_slice_dims,:start_index_map]
            allflds = Pair{Symbol,Union{Type,String}}[:offset_dims => Base.Vector{Int64}, :collapsed_slice_dims => Base.Vector{Int64}, :start_index_map => Base.Vector{Int64}, :index_vector_dim => Int64]
            meta(target, GatherDimensionNumbers, allflds, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, pack, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES)
        end
        __meta_GatherDimensionNumbers[]
    end
end
function Base.getproperty(obj::GatherDimensionNumbers, name::Symbol)
    if name === :offset_dims
        return (obj.__protobuf_jl_internal_values[name])::Base.Vector{Int64}
    elseif name === :collapsed_slice_dims
        return (obj.__protobuf_jl_internal_values[name])::Base.Vector{Int64}
    elseif name === :start_index_map
        return (obj.__protobuf_jl_internal_values[name])::Base.Vector{Int64}
    elseif name === :index_vector_dim
        return (obj.__protobuf_jl_internal_values[name])::Int64
    else
        getfield(obj, name)
    end
end

mutable struct ScatterDimensionNumbers <: ProtoType
    __protobuf_jl_internal_meta::ProtoMeta
    __protobuf_jl_internal_values::Dict{Symbol,Any}
    __protobuf_jl_internal_defaultset::Set{Symbol}

    function ScatterDimensionNumbers(; kwargs...)
        obj = new(meta(ScatterDimensionNumbers), Dict{Symbol,Any}(), Set{Symbol}())
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
end # mutable struct ScatterDimensionNumbers
const __meta_ScatterDimensionNumbers = Ref{ProtoMeta}()
function meta(::Type{ScatterDimensionNumbers})
    ProtoBuf.metalock() do
        if !isassigned(__meta_ScatterDimensionNumbers)
            __meta_ScatterDimensionNumbers[] = target = ProtoMeta(ScatterDimensionNumbers)
            pack = Symbol[:update_window_dims,:inserted_window_dims,:scatter_dims_to_operand_dims]
            allflds = Pair{Symbol,Union{Type,String}}[:update_window_dims => Base.Vector{Int64}, :inserted_window_dims => Base.Vector{Int64}, :scatter_dims_to_operand_dims => Base.Vector{Int64}, :index_vector_dim => Int64]
            meta(target, ScatterDimensionNumbers, allflds, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, pack, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES)
        end
        __meta_ScatterDimensionNumbers[]
    end
end
function Base.getproperty(obj::ScatterDimensionNumbers, name::Symbol)
    if name === :update_window_dims
        return (obj.__protobuf_jl_internal_values[name])::Base.Vector{Int64}
    elseif name === :inserted_window_dims
        return (obj.__protobuf_jl_internal_values[name])::Base.Vector{Int64}
    elseif name === :scatter_dims_to_operand_dims
        return (obj.__protobuf_jl_internal_values[name])::Base.Vector{Int64}
    elseif name === :index_vector_dim
        return (obj.__protobuf_jl_internal_values[name])::Int64
    else
        getfield(obj, name)
    end
end

mutable struct ConvolutionDimensionNumbers <: ProtoType
    __protobuf_jl_internal_meta::ProtoMeta
    __protobuf_jl_internal_values::Dict{Symbol,Any}
    __protobuf_jl_internal_defaultset::Set{Symbol}

    function ConvolutionDimensionNumbers(; kwargs...)
        obj = new(meta(ConvolutionDimensionNumbers), Dict{Symbol,Any}(), Set{Symbol}())
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
end # mutable struct ConvolutionDimensionNumbers
const __meta_ConvolutionDimensionNumbers = Ref{ProtoMeta}()
function meta(::Type{ConvolutionDimensionNumbers})
    ProtoBuf.metalock() do
        if !isassigned(__meta_ConvolutionDimensionNumbers)
            __meta_ConvolutionDimensionNumbers[] = target = ProtoMeta(ConvolutionDimensionNumbers)
            fnum = Int[7,8,11,3,4,6,9,10,12]
            pack = Symbol[:input_spatial_dimensions,:kernel_spatial_dimensions,:output_spatial_dimensions]
            allflds = Pair{Symbol,Union{Type,String}}[:input_batch_dimension => Int64, :input_feature_dimension => Int64, :input_spatial_dimensions => Base.Vector{Int64}, :kernel_input_feature_dimension => Int64, :kernel_output_feature_dimension => Int64, :kernel_spatial_dimensions => Base.Vector{Int64}, :output_batch_dimension => Int64, :output_feature_dimension => Int64, :output_spatial_dimensions => Base.Vector{Int64}]
            meta(target, ConvolutionDimensionNumbers, allflds, ProtoBuf.DEF_REQ, fnum, ProtoBuf.DEF_VAL, pack, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES)
        end
        __meta_ConvolutionDimensionNumbers[]
    end
end
function Base.getproperty(obj::ConvolutionDimensionNumbers, name::Symbol)
    if name === :input_batch_dimension
        return (obj.__protobuf_jl_internal_values[name])::Int64
    elseif name === :input_feature_dimension
        return (obj.__protobuf_jl_internal_values[name])::Int64
    elseif name === :input_spatial_dimensions
        return (obj.__protobuf_jl_internal_values[name])::Base.Vector{Int64}
    elseif name === :kernel_input_feature_dimension
        return (obj.__protobuf_jl_internal_values[name])::Int64
    elseif name === :kernel_output_feature_dimension
        return (obj.__protobuf_jl_internal_values[name])::Int64
    elseif name === :kernel_spatial_dimensions
        return (obj.__protobuf_jl_internal_values[name])::Base.Vector{Int64}
    elseif name === :output_batch_dimension
        return (obj.__protobuf_jl_internal_values[name])::Int64
    elseif name === :output_feature_dimension
        return (obj.__protobuf_jl_internal_values[name])::Int64
    elseif name === :output_spatial_dimensions
        return (obj.__protobuf_jl_internal_values[name])::Base.Vector{Int64}
    else
        getfield(obj, name)
    end
end

mutable struct DotDimensionNumbers <: ProtoType
    __protobuf_jl_internal_meta::ProtoMeta
    __protobuf_jl_internal_values::Dict{Symbol,Any}
    __protobuf_jl_internal_defaultset::Set{Symbol}

    function DotDimensionNumbers(; kwargs...)
        obj = new(meta(DotDimensionNumbers), Dict{Symbol,Any}(), Set{Symbol}())
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
end # mutable struct DotDimensionNumbers
const __meta_DotDimensionNumbers = Ref{ProtoMeta}()
function meta(::Type{DotDimensionNumbers})
    ProtoBuf.metalock() do
        if !isassigned(__meta_DotDimensionNumbers)
            __meta_DotDimensionNumbers[] = target = ProtoMeta(DotDimensionNumbers)
            pack = Symbol[:lhs_contracting_dimensions,:rhs_contracting_dimensions,:lhs_batch_dimensions,:rhs_batch_dimensions]
            allflds = Pair{Symbol,Union{Type,String}}[:lhs_contracting_dimensions => Base.Vector{Int64}, :rhs_contracting_dimensions => Base.Vector{Int64}, :lhs_batch_dimensions => Base.Vector{Int64}, :rhs_batch_dimensions => Base.Vector{Int64}]
            meta(target, DotDimensionNumbers, allflds, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, pack, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES)
        end
        __meta_DotDimensionNumbers[]
    end
end
function Base.getproperty(obj::DotDimensionNumbers, name::Symbol)
    if name === :lhs_contracting_dimensions
        return (obj.__protobuf_jl_internal_values[name])::Base.Vector{Int64}
    elseif name === :rhs_contracting_dimensions
        return (obj.__protobuf_jl_internal_values[name])::Base.Vector{Int64}
    elseif name === :lhs_batch_dimensions
        return (obj.__protobuf_jl_internal_values[name])::Base.Vector{Int64}
    elseif name === :rhs_batch_dimensions
        return (obj.__protobuf_jl_internal_values[name])::Base.Vector{Int64}
    else
        getfield(obj, name)
    end
end

const TriangularSolveOptions_Transpose = (;[
    Symbol("TRANSPOSE_INVALID") => Int32(0),
    Symbol("NO_TRANSPOSE") => Int32(1),
    Symbol("TRANSPOSE") => Int32(2),
    Symbol("ADJOINT") => Int32(3),
]...)

mutable struct TriangularSolveOptions <: ProtoType
    __protobuf_jl_internal_meta::ProtoMeta
    __protobuf_jl_internal_values::Dict{Symbol,Any}
    __protobuf_jl_internal_defaultset::Set{Symbol}

    function TriangularSolveOptions(; kwargs...)
        obj = new(meta(TriangularSolveOptions), Dict{Symbol,Any}(), Set{Symbol}())
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
end # mutable struct TriangularSolveOptions
const __meta_TriangularSolveOptions = Ref{ProtoMeta}()
function meta(::Type{TriangularSolveOptions})
    ProtoBuf.metalock() do
        if !isassigned(__meta_TriangularSolveOptions)
            __meta_TriangularSolveOptions[] = target = ProtoMeta(TriangularSolveOptions)
            allflds = Pair{Symbol,Union{Type,String}}[:left_side => Bool, :lower => Bool, :unit_diagonal => Bool, :transpose_a => Int32]
            meta(target, TriangularSolveOptions, allflds, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES)
        end
        __meta_TriangularSolveOptions[]
    end
end
function Base.getproperty(obj::TriangularSolveOptions, name::Symbol)
    if name === :left_side
        return (obj.__protobuf_jl_internal_values[name])::Bool
    elseif name === :lower
        return (obj.__protobuf_jl_internal_values[name])::Bool
    elseif name === :unit_diagonal
        return (obj.__protobuf_jl_internal_values[name])::Bool
    elseif name === :transpose_a
        return (obj.__protobuf_jl_internal_values[name])::Int32
    else
        getfield(obj, name)
    end
end

mutable struct CholeskyOptions <: ProtoType
    __protobuf_jl_internal_meta::ProtoMeta
    __protobuf_jl_internal_values::Dict{Symbol,Any}
    __protobuf_jl_internal_defaultset::Set{Symbol}

    function CholeskyOptions(; kwargs...)
        obj = new(meta(CholeskyOptions), Dict{Symbol,Any}(), Set{Symbol}())
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
end # mutable struct CholeskyOptions
const __meta_CholeskyOptions = Ref{ProtoMeta}()
function meta(::Type{CholeskyOptions})
    ProtoBuf.metalock() do
        if !isassigned(__meta_CholeskyOptions)
            __meta_CholeskyOptions[] = target = ProtoMeta(CholeskyOptions)
            allflds = Pair{Symbol,Union{Type,String}}[:lower => Bool]
            meta(target, CholeskyOptions, allflds, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES)
        end
        __meta_CholeskyOptions[]
    end
end
function Base.getproperty(obj::CholeskyOptions, name::Symbol)
    if name === :lower
        return (obj.__protobuf_jl_internal_values[name])::Bool
    else
        getfield(obj, name)
    end
end

mutable struct FrontendAttributes_MapEntry <: ProtoType
    __protobuf_jl_internal_meta::ProtoMeta
    __protobuf_jl_internal_values::Dict{Symbol,Any}
    __protobuf_jl_internal_defaultset::Set{Symbol}

    function FrontendAttributes_MapEntry(; kwargs...)
        obj = new(meta(FrontendAttributes_MapEntry), Dict{Symbol,Any}(), Set{Symbol}())
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
end # mutable struct FrontendAttributes_MapEntry (mapentry)
const __meta_FrontendAttributes_MapEntry = Ref{ProtoMeta}()
function meta(::Type{FrontendAttributes_MapEntry})
    ProtoBuf.metalock() do
        if !isassigned(__meta_FrontendAttributes_MapEntry)
            __meta_FrontendAttributes_MapEntry[] = target = ProtoMeta(FrontendAttributes_MapEntry)
            allflds = Pair{Symbol,Union{Type,String}}[:key => AbstractString, :value => AbstractString]
            meta(target, FrontendAttributes_MapEntry, allflds, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES)
        end
        __meta_FrontendAttributes_MapEntry[]
    end
end
function Base.getproperty(obj::FrontendAttributes_MapEntry, name::Symbol)
    if name === :key
        return (obj.__protobuf_jl_internal_values[name])::AbstractString
    elseif name === :value
        return (obj.__protobuf_jl_internal_values[name])::AbstractString
    else
        getfield(obj, name)
    end
end

mutable struct FrontendAttributes <: ProtoType
    __protobuf_jl_internal_meta::ProtoMeta
    __protobuf_jl_internal_values::Dict{Symbol,Any}
    __protobuf_jl_internal_defaultset::Set{Symbol}

    function FrontendAttributes(; kwargs...)
        obj = new(meta(FrontendAttributes), Dict{Symbol,Any}(), Set{Symbol}())
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
end # mutable struct FrontendAttributes
const __meta_FrontendAttributes = Ref{ProtoMeta}()
function meta(::Type{FrontendAttributes})
    ProtoBuf.metalock() do
        if !isassigned(__meta_FrontendAttributes)
            __meta_FrontendAttributes[] = target = ProtoMeta(FrontendAttributes)
            allflds = Pair{Symbol,Union{Type,String}}[:map => Base.Dict{AbstractString,AbstractString}]
            meta(target, FrontendAttributes, allflds, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES)
        end
        __meta_FrontendAttributes[]
    end
end
function Base.getproperty(obj::FrontendAttributes, name::Symbol)
    if name === :map
        return (obj.__protobuf_jl_internal_values[name])::Base.Dict{AbstractString,AbstractString}
    else
        getfield(obj, name)
    end
end

const OpSharding_Type = (;[
    Symbol("REPLICATED") => Int32(0),
    Symbol("MAXIMAL") => Int32(1),
    Symbol("TUPLE") => Int32(2),
    Symbol("OTHER") => Int32(3),
    Symbol("MANUAL") => Int32(4),
]...)

mutable struct OpSharding <: ProtoType
    __protobuf_jl_internal_meta::ProtoMeta
    __protobuf_jl_internal_values::Dict{Symbol,Any}
    __protobuf_jl_internal_defaultset::Set{Symbol}

    function OpSharding(; kwargs...)
        obj = new(meta(OpSharding), Dict{Symbol,Any}(), Set{Symbol}())
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
end # mutable struct OpSharding
const __meta_OpSharding = Ref{ProtoMeta}()
function meta(::Type{OpSharding})
    ProtoBuf.metalock() do
        if !isassigned(__meta_OpSharding)
            __meta_OpSharding[] = target = ProtoMeta(OpSharding)
            pack = Symbol[:tile_assignment_dimensions,:tile_assignment_devices]
            allflds = Pair{Symbol,Union{Type,String}}[:_type => Int32, :tile_shape => ShapeProto, :tile_assignment_dimensions => Base.Vector{Int64}, :tile_assignment_devices => Base.Vector{Int64}, :tuple_shardings => Base.Vector{OpSharding}, :replicate_on_last_tile_dim => Bool]
            meta(target, OpSharding, allflds, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, pack, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES)
        end
        __meta_OpSharding[]
    end
end
function Base.getproperty(obj::OpSharding, name::Symbol)
    if name === :_type
        return (obj.__protobuf_jl_internal_values[name])::Int32
    elseif name === :tile_shape
        return (obj.__protobuf_jl_internal_values[name])::ShapeProto
    elseif name === :tile_assignment_dimensions
        return (obj.__protobuf_jl_internal_values[name])::Base.Vector{Int64}
    elseif name === :tile_assignment_devices
        return (obj.__protobuf_jl_internal_values[name])::Base.Vector{Int64}
    elseif name === :tuple_shardings
        return (obj.__protobuf_jl_internal_values[name])::Base.Vector{OpSharding}
    elseif name === :replicate_on_last_tile_dim
        return (obj.__protobuf_jl_internal_values[name])::Bool
    else
        getfield(obj, name)
    end
end

mutable struct ReplicaGroup <: ProtoType
    __protobuf_jl_internal_meta::ProtoMeta
    __protobuf_jl_internal_values::Dict{Symbol,Any}
    __protobuf_jl_internal_defaultset::Set{Symbol}

    function ReplicaGroup(; kwargs...)
        obj = new(meta(ReplicaGroup), Dict{Symbol,Any}(), Set{Symbol}())
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
end # mutable struct ReplicaGroup
const __meta_ReplicaGroup = Ref{ProtoMeta}()
function meta(::Type{ReplicaGroup})
    ProtoBuf.metalock() do
        if !isassigned(__meta_ReplicaGroup)
            __meta_ReplicaGroup[] = target = ProtoMeta(ReplicaGroup)
            pack = Symbol[:replica_ids]
            allflds = Pair{Symbol,Union{Type,String}}[:replica_ids => Base.Vector{Int64}]
            meta(target, ReplicaGroup, allflds, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, pack, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES)
        end
        __meta_ReplicaGroup[]
    end
end
function Base.getproperty(obj::ReplicaGroup, name::Symbol)
    if name === :replica_ids
        return (obj.__protobuf_jl_internal_values[name])::Base.Vector{Int64}
    else
        getfield(obj, name)
    end
end

mutable struct SourceTarget <: ProtoType
    __protobuf_jl_internal_meta::ProtoMeta
    __protobuf_jl_internal_values::Dict{Symbol,Any}
    __protobuf_jl_internal_defaultset::Set{Symbol}

    function SourceTarget(; kwargs...)
        obj = new(meta(SourceTarget), Dict{Symbol,Any}(), Set{Symbol}())
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
end # mutable struct SourceTarget
const __meta_SourceTarget = Ref{ProtoMeta}()
function meta(::Type{SourceTarget})
    ProtoBuf.metalock() do
        if !isassigned(__meta_SourceTarget)
            __meta_SourceTarget[] = target = ProtoMeta(SourceTarget)
            allflds = Pair{Symbol,Union{Type,String}}[:source => Int64, :target => Int64]
            meta(target, SourceTarget, allflds, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES)
        end
        __meta_SourceTarget[]
    end
end
function Base.getproperty(obj::SourceTarget, name::Symbol)
    if name === :source
        return (obj.__protobuf_jl_internal_values[name])::Int64
    elseif name === :target
        return (obj.__protobuf_jl_internal_values[name])::Int64
    else
        getfield(obj, name)
    end
end

const PrecisionConfig_Precision = (;[
    Symbol("DEFAULT") => Int32(0),
    Symbol("HIGH") => Int32(1),
    Symbol("HIGHEST") => Int32(2),
]...)

mutable struct PrecisionConfig <: ProtoType
    __protobuf_jl_internal_meta::ProtoMeta
    __protobuf_jl_internal_values::Dict{Symbol,Any}
    __protobuf_jl_internal_defaultset::Set{Symbol}

    function PrecisionConfig(; kwargs...)
        obj = new(meta(PrecisionConfig), Dict{Symbol,Any}(), Set{Symbol}())
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
end # mutable struct PrecisionConfig
const __meta_PrecisionConfig = Ref{ProtoMeta}()
function meta(::Type{PrecisionConfig})
    ProtoBuf.metalock() do
        if !isassigned(__meta_PrecisionConfig)
            __meta_PrecisionConfig[] = target = ProtoMeta(PrecisionConfig)
            pack = Symbol[:operand_precision]
            allflds = Pair{Symbol,Union{Type,String}}[:operand_precision => Base.Vector{Int32}]
            meta(target, PrecisionConfig, allflds, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, pack, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES)
        end
        __meta_PrecisionConfig[]
    end
end
function Base.getproperty(obj::PrecisionConfig, name::Symbol)
    if name === :operand_precision
        return (obj.__protobuf_jl_internal_values[name])::Base.Vector{Int32}
    else
        getfield(obj, name)
    end
end

mutable struct ParameterReplication <: ProtoType
    __protobuf_jl_internal_meta::ProtoMeta
    __protobuf_jl_internal_values::Dict{Symbol,Any}
    __protobuf_jl_internal_defaultset::Set{Symbol}

    function ParameterReplication(; kwargs...)
        obj = new(meta(ParameterReplication), Dict{Symbol,Any}(), Set{Symbol}())
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
end # mutable struct ParameterReplication
const __meta_ParameterReplication = Ref{ProtoMeta}()
function meta(::Type{ParameterReplication})
    ProtoBuf.metalock() do
        if !isassigned(__meta_ParameterReplication)
            __meta_ParameterReplication[] = target = ProtoMeta(ParameterReplication)
            pack = Symbol[:replicated_at_leaf_buffers]
            allflds = Pair{Symbol,Union{Type,String}}[:replicated_at_leaf_buffers => Base.Vector{Bool}]
            meta(target, ParameterReplication, allflds, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, pack, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES)
        end
        __meta_ParameterReplication[]
    end
end
function Base.getproperty(obj::ParameterReplication, name::Symbol)
    if name === :replicated_at_leaf_buffers
        return (obj.__protobuf_jl_internal_values[name])::Base.Vector{Bool}
    else
        getfield(obj, name)
    end
end

mutable struct WhileLoopBackendConfig_KnownTripCount <: ProtoType
    __protobuf_jl_internal_meta::ProtoMeta
    __protobuf_jl_internal_values::Dict{Symbol,Any}
    __protobuf_jl_internal_defaultset::Set{Symbol}

    function WhileLoopBackendConfig_KnownTripCount(; kwargs...)
        obj = new(meta(WhileLoopBackendConfig_KnownTripCount), Dict{Symbol,Any}(), Set{Symbol}())
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
end # mutable struct WhileLoopBackendConfig_KnownTripCount
const __meta_WhileLoopBackendConfig_KnownTripCount = Ref{ProtoMeta}()
function meta(::Type{WhileLoopBackendConfig_KnownTripCount})
    ProtoBuf.metalock() do
        if !isassigned(__meta_WhileLoopBackendConfig_KnownTripCount)
            __meta_WhileLoopBackendConfig_KnownTripCount[] = target = ProtoMeta(WhileLoopBackendConfig_KnownTripCount)
            allflds = Pair{Symbol,Union{Type,String}}[:n => Int64]
            meta(target, WhileLoopBackendConfig_KnownTripCount, allflds, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES)
        end
        __meta_WhileLoopBackendConfig_KnownTripCount[]
    end
end
function Base.getproperty(obj::WhileLoopBackendConfig_KnownTripCount, name::Symbol)
    if name === :n
        return (obj.__protobuf_jl_internal_values[name])::Int64
    else
        getfield(obj, name)
    end
end

mutable struct WhileLoopBackendConfig <: ProtoType
    __protobuf_jl_internal_meta::ProtoMeta
    __protobuf_jl_internal_values::Dict{Symbol,Any}
    __protobuf_jl_internal_defaultset::Set{Symbol}

    function WhileLoopBackendConfig(; kwargs...)
        obj = new(meta(WhileLoopBackendConfig), Dict{Symbol,Any}(), Set{Symbol}())
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
end # mutable struct WhileLoopBackendConfig
const __meta_WhileLoopBackendConfig = Ref{ProtoMeta}()
function meta(::Type{WhileLoopBackendConfig})
    ProtoBuf.metalock() do
        if !isassigned(__meta_WhileLoopBackendConfig)
            __meta_WhileLoopBackendConfig[] = target = ProtoMeta(WhileLoopBackendConfig)
            allflds = Pair{Symbol,Union{Type,String}}[:known_trip_count => WhileLoopBackendConfig_KnownTripCount]
            meta(target, WhileLoopBackendConfig, allflds, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES)
        end
        __meta_WhileLoopBackendConfig[]
    end
end
function Base.getproperty(obj::WhileLoopBackendConfig, name::Symbol)
    if name === :known_trip_count
        return (obj.__protobuf_jl_internal_values[name])::WhileLoopBackendConfig_KnownTripCount
    else
        getfield(obj, name)
    end
end

mutable struct CustomCallOutputOperandAliasing <: ProtoType
    __protobuf_jl_internal_meta::ProtoMeta
    __protobuf_jl_internal_values::Dict{Symbol,Any}
    __protobuf_jl_internal_defaultset::Set{Symbol}

    function CustomCallOutputOperandAliasing(; kwargs...)
        obj = new(meta(CustomCallOutputOperandAliasing), Dict{Symbol,Any}(), Set{Symbol}())
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
end # mutable struct CustomCallOutputOperandAliasing
const __meta_CustomCallOutputOperandAliasing = Ref{ProtoMeta}()
function meta(::Type{CustomCallOutputOperandAliasing})
    ProtoBuf.metalock() do
        if !isassigned(__meta_CustomCallOutputOperandAliasing)
            __meta_CustomCallOutputOperandAliasing[] = target = ProtoMeta(CustomCallOutputOperandAliasing)
            pack = Symbol[:output_shape_index,:operand_shape_index]
            allflds = Pair{Symbol,Union{Type,String}}[:output_shape_index => Base.Vector{Int64}, :operand_index => Int64, :operand_shape_index => Base.Vector{Int64}]
            meta(target, CustomCallOutputOperandAliasing, allflds, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, pack, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES)
        end
        __meta_CustomCallOutputOperandAliasing[]
    end
end
function Base.getproperty(obj::CustomCallOutputOperandAliasing, name::Symbol)
    if name === :output_shape_index
        return (obj.__protobuf_jl_internal_values[name])::Base.Vector{Int64}
    elseif name === :operand_index
        return (obj.__protobuf_jl_internal_values[name])::Int64
    elseif name === :operand_shape_index
        return (obj.__protobuf_jl_internal_values[name])::Base.Vector{Int64}
    else
        getfield(obj, name)
    end
end

export PrimitiveType, Format, ProfileType, PaddingType, FftType, RandomDistribution, RandomAlgorithm, PaddingConfig_PaddingConfigDimension, PaddingConfig, TileProto, LayoutProto, ShapeProto, ProgramShapeProto, ComputationStats, OpMetadata, ExecutionProfile, ExecutionHandle, GlobalDataHandle, DeviceHandle, ChannelHandle_ChannelType, ChannelHandle, DeviceAssignmentProto_ComputationDevice, DeviceAssignmentProto, LiteralProto, WindowDimension, Window, GatherDimensionNumbers, ScatterDimensionNumbers, ConvolutionDimensionNumbers, DotDimensionNumbers, TriangularSolveOptions_Transpose, TriangularSolveOptions, CholeskyOptions, FrontendAttributes_MapEntry, FrontendAttributes, OpSharding_Type, OpSharding, ReplicaGroup, SourceTarget, PrecisionConfig_Precision, PrecisionConfig, ParameterReplication, WhileLoopBackendConfig_KnownTripCount, WhileLoopBackendConfig, CustomCallOutputOperandAliasing
# mapentries: "FrontendAttributes_MapEntry" => ("AbstractString", "AbstractString")
