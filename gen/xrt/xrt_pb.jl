# syntax: proto3
using ProtoBuf
import ProtoBuf.meta
import ._ProtoBuf_Top_.tensorflow
import ._ProtoBuf_Top_.xla

mutable struct DeviceAssignment_ComputationDevice_DeviceMeshCoordinates <: ProtoType
    __protobuf_jl_internal_meta::ProtoMeta
    __protobuf_jl_internal_values::Dict{Symbol,Any}
    __protobuf_jl_internal_defaultset::Set{Symbol}

    function DeviceAssignment_ComputationDevice_DeviceMeshCoordinates(; kwargs...)
        obj = new(meta(DeviceAssignment_ComputationDevice_DeviceMeshCoordinates), Dict{Symbol,Any}(), Set{Symbol}())
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
end # mutable struct DeviceAssignment_ComputationDevice_DeviceMeshCoordinates
const __meta_DeviceAssignment_ComputationDevice_DeviceMeshCoordinates = Ref{ProtoMeta}()
function meta(::Type{DeviceAssignment_ComputationDevice_DeviceMeshCoordinates})
    ProtoBuf.metalock() do
        if !isassigned(__meta_DeviceAssignment_ComputationDevice_DeviceMeshCoordinates)
            __meta_DeviceAssignment_ComputationDevice_DeviceMeshCoordinates[] = target = ProtoMeta(DeviceAssignment_ComputationDevice_DeviceMeshCoordinates)
            pack = Symbol[:value]
            allflds = Pair{Symbol,Union{Type,String}}[:value => Base.Vector{Int32}]
            meta(target, DeviceAssignment_ComputationDevice_DeviceMeshCoordinates, allflds, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, pack, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES)
        end
        __meta_DeviceAssignment_ComputationDevice_DeviceMeshCoordinates[]
    end
end
function Base.getproperty(obj::DeviceAssignment_ComputationDevice_DeviceMeshCoordinates, name::Symbol)
    if name === :value
        return (obj.__protobuf_jl_internal_values[name])::Base.Vector{Int32}
    else
        getfield(obj, name)
    end
end

mutable struct DeviceAssignment_ComputationDevice <: ProtoType
    __protobuf_jl_internal_meta::ProtoMeta
    __protobuf_jl_internal_values::Dict{Symbol,Any}
    __protobuf_jl_internal_defaultset::Set{Symbol}

    function DeviceAssignment_ComputationDevice(; kwargs...)
        obj = new(meta(DeviceAssignment_ComputationDevice), Dict{Symbol,Any}(), Set{Symbol}())
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
end # mutable struct DeviceAssignment_ComputationDevice
const __meta_DeviceAssignment_ComputationDevice = Ref{ProtoMeta}()
function meta(::Type{DeviceAssignment_ComputationDevice})
    ProtoBuf.metalock() do
        if !isassigned(__meta_DeviceAssignment_ComputationDevice)
            __meta_DeviceAssignment_ComputationDevice[] = target = ProtoMeta(DeviceAssignment_ComputationDevice)
            allflds = Pair{Symbol,Union{Type,String}}[:replica_devices => Base.Vector{DeviceAssignment_ComputationDevice_DeviceMeshCoordinates}]
            meta(target, DeviceAssignment_ComputationDevice, allflds, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES)
        end
        __meta_DeviceAssignment_ComputationDevice[]
    end
end
function Base.getproperty(obj::DeviceAssignment_ComputationDevice, name::Symbol)
    if name === :replica_devices
        return (obj.__protobuf_jl_internal_values[name])::Base.Vector{DeviceAssignment_ComputationDevice_DeviceMeshCoordinates}
    else
        getfield(obj, name)
    end
end

mutable struct DeviceAssignment <: ProtoType
    __protobuf_jl_internal_meta::ProtoMeta
    __protobuf_jl_internal_values::Dict{Symbol,Any}
    __protobuf_jl_internal_defaultset::Set{Symbol}

    function DeviceAssignment(; kwargs...)
        obj = new(meta(DeviceAssignment), Dict{Symbol,Any}(), Set{Symbol}())
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
end # mutable struct DeviceAssignment
const __meta_DeviceAssignment = Ref{ProtoMeta}()
function meta(::Type{DeviceAssignment})
    ProtoBuf.metalock() do
        if !isassigned(__meta_DeviceAssignment)
            __meta_DeviceAssignment[] = target = ProtoMeta(DeviceAssignment)
            allflds = Pair{Symbol,Union{Type,String}}[:computation_devices => Base.Vector{DeviceAssignment_ComputationDevice}]
            meta(target, DeviceAssignment, allflds, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES)
        end
        __meta_DeviceAssignment[]
    end
end
function Base.getproperty(obj::DeviceAssignment, name::Symbol)
    if name === :computation_devices
        return (obj.__protobuf_jl_internal_values[name])::Base.Vector{DeviceAssignment_ComputationDevice}
    else
        getfield(obj, name)
    end
end

mutable struct XLAComputationConfig_Experimental_UpdateIndexPair <: ProtoType
    __protobuf_jl_internal_meta::ProtoMeta
    __protobuf_jl_internal_values::Dict{Symbol,Any}
    __protobuf_jl_internal_defaultset::Set{Symbol}

    function XLAComputationConfig_Experimental_UpdateIndexPair(; kwargs...)
        obj = new(meta(XLAComputationConfig_Experimental_UpdateIndexPair), Dict{Symbol,Any}(), Set{Symbol}())
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
end # mutable struct XLAComputationConfig_Experimental_UpdateIndexPair
const __meta_XLAComputationConfig_Experimental_UpdateIndexPair = Ref{ProtoMeta}()
function meta(::Type{XLAComputationConfig_Experimental_UpdateIndexPair})
    ProtoBuf.metalock() do
        if !isassigned(__meta_XLAComputationConfig_Experimental_UpdateIndexPair)
            __meta_XLAComputationConfig_Experimental_UpdateIndexPair[] = target = ProtoMeta(XLAComputationConfig_Experimental_UpdateIndexPair)
            allflds = Pair{Symbol,Union{Type,String}}[:index => Int32, :updated => Bool]
            meta(target, XLAComputationConfig_Experimental_UpdateIndexPair, allflds, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES)
        end
        __meta_XLAComputationConfig_Experimental_UpdateIndexPair[]
    end
end
function Base.getproperty(obj::XLAComputationConfig_Experimental_UpdateIndexPair, name::Symbol)
    if name === :index
        return (obj.__protobuf_jl_internal_values[name])::Int32
    elseif name === :updated
        return (obj.__protobuf_jl_internal_values[name])::Bool
    else
        getfield(obj, name)
    end
end

mutable struct XLAComputationConfig_Experimental <: ProtoType
    __protobuf_jl_internal_meta::ProtoMeta
    __protobuf_jl_internal_values::Dict{Symbol,Any}
    __protobuf_jl_internal_defaultset::Set{Symbol}

    function XLAComputationConfig_Experimental(; kwargs...)
        obj = new(meta(XLAComputationConfig_Experimental), Dict{Symbol,Any}(), Set{Symbol}())
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
end # mutable struct XLAComputationConfig_Experimental
const __meta_XLAComputationConfig_Experimental = Ref{ProtoMeta}()
function meta(::Type{XLAComputationConfig_Experimental})
    ProtoBuf.metalock() do
        if !isassigned(__meta_XLAComputationConfig_Experimental)
            __meta_XLAComputationConfig_Experimental[] = target = ProtoMeta(XLAComputationConfig_Experimental)
            allflds = Pair{Symbol,Union{Type,String}}[:stateful_input_indices => Base.Vector{XLAComputationConfig_Experimental_UpdateIndexPair}]
            meta(target, XLAComputationConfig_Experimental, allflds, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES)
        end
        __meta_XLAComputationConfig_Experimental[]
    end
end
function Base.getproperty(obj::XLAComputationConfig_Experimental, name::Symbol)
    if name === :stateful_input_indices
        return (obj.__protobuf_jl_internal_values[name])::Base.Vector{XLAComputationConfig_Experimental_UpdateIndexPair}
    else
        getfield(obj, name)
    end
end

mutable struct XLAComputationConfig <: ProtoType
    __protobuf_jl_internal_meta::ProtoMeta
    __protobuf_jl_internal_values::Dict{Symbol,Any}
    __protobuf_jl_internal_defaultset::Set{Symbol}

    function XLAComputationConfig(; kwargs...)
        obj = new(meta(XLAComputationConfig), Dict{Symbol,Any}(), Set{Symbol}())
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
end # mutable struct XLAComputationConfig
const __meta_XLAComputationConfig = Ref{ProtoMeta}()
function meta(::Type{XLAComputationConfig})
    ProtoBuf.metalock() do
        if !isassigned(__meta_XLAComputationConfig)
            __meta_XLAComputationConfig[] = target = ProtoMeta(XLAComputationConfig)
            allflds = Pair{Symbol,Union{Type,String}}[:num_replicas => Int32, :num_cores_per_replica => Int32, :host_compute_metadata => tensorflow.tf2xla.HostComputeMetadata, :program_shape => xla.ProgramShapeProto, :per_core_program_shape => Base.Vector{xla.ProgramShapeProto}, :device_assignment => DeviceAssignment, :debug_options => xla.DebugOptions, :experimental => XLAComputationConfig_Experimental]
            meta(target, XLAComputationConfig, allflds, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES)
        end
        __meta_XLAComputationConfig[]
    end
end
function Base.getproperty(obj::XLAComputationConfig, name::Symbol)
    if name === :num_replicas
        return (obj.__protobuf_jl_internal_values[name])::Int32
    elseif name === :num_cores_per_replica
        return (obj.__protobuf_jl_internal_values[name])::Int32
    elseif name === :host_compute_metadata
        return (obj.__protobuf_jl_internal_values[name])::tensorflow.tf2xla.HostComputeMetadata
    elseif name === :program_shape
        return (obj.__protobuf_jl_internal_values[name])::xla.ProgramShapeProto
    elseif name === :per_core_program_shape
        return (obj.__protobuf_jl_internal_values[name])::Base.Vector{xla.ProgramShapeProto}
    elseif name === :device_assignment
        return (obj.__protobuf_jl_internal_values[name])::DeviceAssignment
    elseif name === :debug_options
        return (obj.__protobuf_jl_internal_values[name])::xla.DebugOptions
    elseif name === :experimental
        return (obj.__protobuf_jl_internal_values[name])::XLAComputationConfig_Experimental
    else
        getfield(obj, name)
    end
end

mutable struct XLAComputation <: ProtoType
    __protobuf_jl_internal_meta::ProtoMeta
    __protobuf_jl_internal_values::Dict{Symbol,Any}
    __protobuf_jl_internal_defaultset::Set{Symbol}

    function XLAComputation(; kwargs...)
        obj = new(meta(XLAComputation), Dict{Symbol,Any}(), Set{Symbol}())
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
end # mutable struct XLAComputation
const __meta_XLAComputation = Ref{ProtoMeta}()
function meta(::Type{XLAComputation})
    ProtoBuf.metalock() do
        if !isassigned(__meta_XLAComputation)
            __meta_XLAComputation[] = target = ProtoMeta(XLAComputation)
            allflds = Pair{Symbol,Union{Type,String}}[:config => XLAComputationConfig, :hlo_snapshot => xla.HloSnapshot]
            meta(target, XLAComputation, allflds, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES)
        end
        __meta_XLAComputation[]
    end
end
function Base.getproperty(obj::XLAComputation, name::Symbol)
    if name === :config
        return (obj.__protobuf_jl_internal_values[name])::XLAComputationConfig
    elseif name === :hlo_snapshot
        return (obj.__protobuf_jl_internal_values[name])::xla.HloSnapshot
    else
        getfield(obj, name)
    end
end

mutable struct XLAAllocation <: ProtoType
    __protobuf_jl_internal_meta::ProtoMeta
    __protobuf_jl_internal_values::Dict{Symbol,Any}
    __protobuf_jl_internal_defaultset::Set{Symbol}

    function XLAAllocation(; kwargs...)
        obj = new(meta(XLAAllocation), Dict{Symbol,Any}(), Set{Symbol}())
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
end # mutable struct XLAAllocation
const __meta_XLAAllocation = Ref{ProtoMeta}()
function meta(::Type{XLAAllocation})
    ProtoBuf.metalock() do
        if !isassigned(__meta_XLAAllocation)
            __meta_XLAAllocation[] = target = ProtoMeta(XLAAllocation)
            fnum = Int[2]
            allflds = Pair{Symbol,Union{Type,String}}[:value => xla.LiteralProto]
            meta(target, XLAAllocation, allflds, ProtoBuf.DEF_REQ, fnum, ProtoBuf.DEF_VAL, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES)
        end
        __meta_XLAAllocation[]
    end
end
function Base.getproperty(obj::XLAAllocation, name::Symbol)
    if name === :value
        return (obj.__protobuf_jl_internal_values[name])::xla.LiteralProto
    else
        getfield(obj, name)
    end
end

mutable struct XLATupleNode <: ProtoType
    __protobuf_jl_internal_meta::ProtoMeta
    __protobuf_jl_internal_values::Dict{Symbol,Any}
    __protobuf_jl_internal_defaultset::Set{Symbol}

    function XLATupleNode(; kwargs...)
        obj = new(meta(XLATupleNode), Dict{Symbol,Any}(), Set{Symbol}())
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
end # mutable struct XLATupleNode
const __meta_XLATupleNode = Ref{ProtoMeta}()
function meta(::Type{XLATupleNode})
    ProtoBuf.metalock() do
        if !isassigned(__meta_XLATupleNode)
            __meta_XLATupleNode[] = target = ProtoMeta(XLATupleNode)
            allflds = Pair{Symbol,Union{Type,String}}[:input_index => Int32, :release_input_handle => Bool, :tuples => Base.Vector{XLATupleNode}]
            meta(target, XLATupleNode, allflds, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES)
        end
        __meta_XLATupleNode[]
    end
end
function Base.getproperty(obj::XLATupleNode, name::Symbol)
    if name === :input_index
        return (obj.__protobuf_jl_internal_values[name])::Int32
    elseif name === :release_input_handle
        return (obj.__protobuf_jl_internal_values[name])::Bool
    elseif name === :tuples
        return (obj.__protobuf_jl_internal_values[name])::Base.Vector{XLATupleNode}
    else
        getfield(obj, name)
    end
end

mutable struct CommonExecutionConfig <: ProtoType
    __protobuf_jl_internal_meta::ProtoMeta
    __protobuf_jl_internal_values::Dict{Symbol,Any}
    __protobuf_jl_internal_defaultset::Set{Symbol}

    function CommonExecutionConfig(; kwargs...)
        obj = new(meta(CommonExecutionConfig), Dict{Symbol,Any}(), Set{Symbol}())
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
end # mutable struct CommonExecutionConfig
const __meta_CommonExecutionConfig = Ref{ProtoMeta}()
function meta(::Type{CommonExecutionConfig})
    ProtoBuf.metalock() do
        if !isassigned(__meta_CommonExecutionConfig)
            __meta_CommonExecutionConfig[] = target = ProtoMeta(CommonExecutionConfig)
            pack = Symbol[:local_replica_mapping]
            allflds = Pair{Symbol,Union{Type,String}}[:replica_id => Int32, :local_replica_mapping => Base.Vector{Int32}, :run_id => Int64]
            meta(target, CommonExecutionConfig, allflds, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, pack, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES)
        end
        __meta_CommonExecutionConfig[]
    end
end
function Base.getproperty(obj::CommonExecutionConfig, name::Symbol)
    if name === :replica_id
        return (obj.__protobuf_jl_internal_values[name])::Int32
    elseif name === :local_replica_mapping
        return (obj.__protobuf_jl_internal_values[name])::Base.Vector{Int32}
    elseif name === :run_id
        return (obj.__protobuf_jl_internal_values[name])::Int64
    else
        getfield(obj, name)
    end
end

mutable struct XRTExecutionConfig <: ProtoType
    __protobuf_jl_internal_meta::ProtoMeta
    __protobuf_jl_internal_values::Dict{Symbol,Any}
    __protobuf_jl_internal_defaultset::Set{Symbol}

    function XRTExecutionConfig(; kwargs...)
        obj = new(meta(XRTExecutionConfig), Dict{Symbol,Any}(), Set{Symbol}())
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
end # mutable struct XRTExecutionConfig
const __meta_XRTExecutionConfig = Ref{ProtoMeta}()
function meta(::Type{XRTExecutionConfig})
    ProtoBuf.metalock() do
        if !isassigned(__meta_XRTExecutionConfig)
            __meta_XRTExecutionConfig[] = target = ProtoMeta(XRTExecutionConfig)
            fnum = Int[1,2,3,4,5,6,7,9]
            allflds = Pair{Symbol,Union{Type,String}}[:device_ordinal => Int32, :core_index_in_replica => Int32, :execution_instance_key => AbstractString, :rng_seed => UInt32, :release_input_handles => Bool, :release_compilation_handle => Bool, :return_exploded_tuple => Bool, :common_config => CommonExecutionConfig]
            meta(target, XRTExecutionConfig, allflds, ProtoBuf.DEF_REQ, fnum, ProtoBuf.DEF_VAL, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES)
        end
        __meta_XRTExecutionConfig[]
    end
end
function Base.getproperty(obj::XRTExecutionConfig, name::Symbol)
    if name === :device_ordinal
        return (obj.__protobuf_jl_internal_values[name])::Int32
    elseif name === :core_index_in_replica
        return (obj.__protobuf_jl_internal_values[name])::Int32
    elseif name === :execution_instance_key
        return (obj.__protobuf_jl_internal_values[name])::AbstractString
    elseif name === :rng_seed
        return (obj.__protobuf_jl_internal_values[name])::UInt32
    elseif name === :release_input_handles
        return (obj.__protobuf_jl_internal_values[name])::Bool
    elseif name === :release_compilation_handle
        return (obj.__protobuf_jl_internal_values[name])::Bool
    elseif name === :return_exploded_tuple
        return (obj.__protobuf_jl_internal_values[name])::Bool
    elseif name === :common_config
        return (obj.__protobuf_jl_internal_values[name])::CommonExecutionConfig
    else
        getfield(obj, name)
    end
end

mutable struct XRTChainedExecuteConfig <: ProtoType
    __protobuf_jl_internal_meta::ProtoMeta
    __protobuf_jl_internal_values::Dict{Symbol,Any}
    __protobuf_jl_internal_defaultset::Set{Symbol}

    function XRTChainedExecuteConfig(; kwargs...)
        obj = new(meta(XRTChainedExecuteConfig), Dict{Symbol,Any}(), Set{Symbol}())
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
end # mutable struct XRTChainedExecuteConfig
const __meta_XRTChainedExecuteConfig = Ref{ProtoMeta}()
function meta(::Type{XRTChainedExecuteConfig})
    ProtoBuf.metalock() do
        if !isassigned(__meta_XRTChainedExecuteConfig)
            __meta_XRTChainedExecuteConfig[] = target = ProtoMeta(XRTChainedExecuteConfig)
            fnum = Int[1,2,3,5]
            allflds = Pair{Symbol,Union{Type,String}}[:rng_seed => UInt32, :core_index_in_replica => Int32, :execution_instance_key => AbstractString, :common_config => CommonExecutionConfig]
            meta(target, XRTChainedExecuteConfig, allflds, ProtoBuf.DEF_REQ, fnum, ProtoBuf.DEF_VAL, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES)
        end
        __meta_XRTChainedExecuteConfig[]
    end
end
function Base.getproperty(obj::XRTChainedExecuteConfig, name::Symbol)
    if name === :rng_seed
        return (obj.__protobuf_jl_internal_values[name])::UInt32
    elseif name === :core_index_in_replica
        return (obj.__protobuf_jl_internal_values[name])::Int32
    elseif name === :execution_instance_key
        return (obj.__protobuf_jl_internal_values[name])::AbstractString
    elseif name === :common_config
        return (obj.__protobuf_jl_internal_values[name])::CommonExecutionConfig
    else
        getfield(obj, name)
    end
end

mutable struct XRTChainedExecuteOp_Input <: ProtoType
    __protobuf_jl_internal_meta::ProtoMeta
    __protobuf_jl_internal_values::Dict{Symbol,Any}
    __protobuf_jl_internal_defaultset::Set{Symbol}

    function XRTChainedExecuteOp_Input(; kwargs...)
        obj = new(meta(XRTChainedExecuteOp_Input), Dict{Symbol,Any}(), Set{Symbol}())
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
end # mutable struct XRTChainedExecuteOp_Input
const __meta_XRTChainedExecuteOp_Input = Ref{ProtoMeta}()
function meta(::Type{XRTChainedExecuteOp_Input})
    ProtoBuf.metalock() do
        if !isassigned(__meta_XRTChainedExecuteOp_Input)
            __meta_XRTChainedExecuteOp_Input[] = target = ProtoMeta(XRTChainedExecuteOp_Input)
            allflds = Pair{Symbol,Union{Type,String}}[:op_index => Int64, :output_index => Int64]
            meta(target, XRTChainedExecuteOp_Input, allflds, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES)
        end
        __meta_XRTChainedExecuteOp_Input[]
    end
end
function Base.getproperty(obj::XRTChainedExecuteOp_Input, name::Symbol)
    if name === :op_index
        return (obj.__protobuf_jl_internal_values[name])::Int64
    elseif name === :output_index
        return (obj.__protobuf_jl_internal_values[name])::Int64
    else
        getfield(obj, name)
    end
end

mutable struct XRTChainedExecuteOp_Output <: ProtoType
    __protobuf_jl_internal_meta::ProtoMeta
    __protobuf_jl_internal_values::Dict{Symbol,Any}
    __protobuf_jl_internal_defaultset::Set{Symbol}

    function XRTChainedExecuteOp_Output(; kwargs...)
        obj = new(meta(XRTChainedExecuteOp_Output), Dict{Symbol,Any}(), Set{Symbol}())
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
end # mutable struct XRTChainedExecuteOp_Output
const __meta_XRTChainedExecuteOp_Output = Ref{ProtoMeta}()
function meta(::Type{XRTChainedExecuteOp_Output})
    ProtoBuf.metalock() do
        if !isassigned(__meta_XRTChainedExecuteOp_Output)
            __meta_XRTChainedExecuteOp_Output[] = target = ProtoMeta(XRTChainedExecuteOp_Output)
            allflds = Pair{Symbol,Union{Type,String}}[:output_index => Int64, :result_index => Int64]
            meta(target, XRTChainedExecuteOp_Output, allflds, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES)
        end
        __meta_XRTChainedExecuteOp_Output[]
    end
end
function Base.getproperty(obj::XRTChainedExecuteOp_Output, name::Symbol)
    if name === :output_index
        return (obj.__protobuf_jl_internal_values[name])::Int64
    elseif name === :result_index
        return (obj.__protobuf_jl_internal_values[name])::Int64
    else
        getfield(obj, name)
    end
end

mutable struct XRTChainedExecuteOp <: ProtoType
    __protobuf_jl_internal_meta::ProtoMeta
    __protobuf_jl_internal_values::Dict{Symbol,Any}
    __protobuf_jl_internal_defaultset::Set{Symbol}

    function XRTChainedExecuteOp(; kwargs...)
        obj = new(meta(XRTChainedExecuteOp), Dict{Symbol,Any}(), Set{Symbol}())
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
end # mutable struct XRTChainedExecuteOp
const __meta_XRTChainedExecuteOp = Ref{ProtoMeta}()
function meta(::Type{XRTChainedExecuteOp})
    ProtoBuf.metalock() do
        if !isassigned(__meta_XRTChainedExecuteOp)
            __meta_XRTChainedExecuteOp[] = target = ProtoMeta(XRTChainedExecuteOp)
            allflds = Pair{Symbol,Union{Type,String}}[:data_handle => Int64, :computation_handle => Int64, :outputs => Base.Vector{XRTChainedExecuteOp_Output}, :inputs => Base.Vector{XRTChainedExecuteOp_Input}]
            oneofs = Int[1,1,0,0]
            oneof_names = Symbol[Symbol("op_oneof")]
            meta(target, XRTChainedExecuteOp, allflds, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, oneofs, oneof_names)
        end
        __meta_XRTChainedExecuteOp[]
    end
end
function Base.getproperty(obj::XRTChainedExecuteOp, name::Symbol)
    if name === :data_handle
        return (obj.__protobuf_jl_internal_values[name])::Int64
    elseif name === :computation_handle
        return (obj.__protobuf_jl_internal_values[name])::Int64
    elseif name === :outputs
        return (obj.__protobuf_jl_internal_values[name])::Base.Vector{XRTChainedExecuteOp_Output}
    elseif name === :inputs
        return (obj.__protobuf_jl_internal_values[name])::Base.Vector{XRTChainedExecuteOp_Input}
    else
        getfield(obj, name)
    end
end

mutable struct XRTChainedExecutePlan <: ProtoType
    __protobuf_jl_internal_meta::ProtoMeta
    __protobuf_jl_internal_values::Dict{Symbol,Any}
    __protobuf_jl_internal_defaultset::Set{Symbol}

    function XRTChainedExecutePlan(; kwargs...)
        obj = new(meta(XRTChainedExecutePlan), Dict{Symbol,Any}(), Set{Symbol}())
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
end # mutable struct XRTChainedExecutePlan
const __meta_XRTChainedExecutePlan = Ref{ProtoMeta}()
function meta(::Type{XRTChainedExecutePlan})
    ProtoBuf.metalock() do
        if !isassigned(__meta_XRTChainedExecutePlan)
            __meta_XRTChainedExecutePlan[] = target = ProtoMeta(XRTChainedExecutePlan)
            allflds = Pair{Symbol,Union{Type,String}}[:ops => Base.Vector{XRTChainedExecuteOp}]
            meta(target, XRTChainedExecutePlan, allflds, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES)
        end
        __meta_XRTChainedExecutePlan[]
    end
end
function Base.getproperty(obj::XRTChainedExecutePlan, name::Symbol)
    if name === :ops
        return (obj.__protobuf_jl_internal_values[name])::Base.Vector{XRTChainedExecuteOp}
    else
        getfield(obj, name)
    end
end

mutable struct XRTMetricsCollect <: ProtoType
    __protobuf_jl_internal_meta::ProtoMeta
    __protobuf_jl_internal_values::Dict{Symbol,Any}
    __protobuf_jl_internal_defaultset::Set{Symbol}

    function XRTMetricsCollect(; kwargs...)
        obj = new(meta(XRTMetricsCollect), Dict{Symbol,Any}(), Set{Symbol}())
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
end # mutable struct XRTMetricsCollect
const __meta_XRTMetricsCollect = Ref{ProtoMeta}()
function meta(::Type{XRTMetricsCollect})
    ProtoBuf.metalock() do
        if !isassigned(__meta_XRTMetricsCollect)
            __meta_XRTMetricsCollect[] = target = ProtoMeta(XRTMetricsCollect)
            allflds = Pair{Symbol,Union{Type,String}}[:metrics_regex => Base.Vector{AbstractString}]
            meta(target, XRTMetricsCollect, allflds, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES)
        end
        __meta_XRTMetricsCollect[]
    end
end
function Base.getproperty(obj::XRTMetricsCollect, name::Symbol)
    if name === :metrics_regex
        return (obj.__protobuf_jl_internal_values[name])::Base.Vector{AbstractString}
    else
        getfield(obj, name)
    end
end

mutable struct Percentiles_Point <: ProtoType
    __protobuf_jl_internal_meta::ProtoMeta
    __protobuf_jl_internal_values::Dict{Symbol,Any}
    __protobuf_jl_internal_defaultset::Set{Symbol}

    function Percentiles_Point(; kwargs...)
        obj = new(meta(Percentiles_Point), Dict{Symbol,Any}(), Set{Symbol}())
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
end # mutable struct Percentiles_Point
const __meta_Percentiles_Point = Ref{ProtoMeta}()
function meta(::Type{Percentiles_Point})
    ProtoBuf.metalock() do
        if !isassigned(__meta_Percentiles_Point)
            __meta_Percentiles_Point[] = target = ProtoMeta(Percentiles_Point)
            allflds = Pair{Symbol,Union{Type,String}}[:percentile => Float64, :value => Float64]
            meta(target, Percentiles_Point, allflds, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES)
        end
        __meta_Percentiles_Point[]
    end
end
function Base.getproperty(obj::Percentiles_Point, name::Symbol)
    if name === :percentile
        return (obj.__protobuf_jl_internal_values[name])::Float64
    elseif name === :value
        return (obj.__protobuf_jl_internal_values[name])::Float64
    else
        getfield(obj, name)
    end
end

mutable struct Percentiles <: ProtoType
    __protobuf_jl_internal_meta::ProtoMeta
    __protobuf_jl_internal_values::Dict{Symbol,Any}
    __protobuf_jl_internal_defaultset::Set{Symbol}

    function Percentiles(; kwargs...)
        obj = new(meta(Percentiles), Dict{Symbol,Any}(), Set{Symbol}())
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
end # mutable struct Percentiles
const __meta_Percentiles = Ref{ProtoMeta}()
function meta(::Type{Percentiles})
    ProtoBuf.metalock() do
        if !isassigned(__meta_Percentiles)
            __meta_Percentiles[] = target = ProtoMeta(Percentiles)
            allflds = Pair{Symbol,Union{Type,String}}[:start_nstime => UInt64, :end_nstime => UInt64, :min_value => Float64, :max_value => Float64, :mean => Float64, :stddev => Float64, :num_samples => UInt64, :total_samples => UInt64, :accumulator => Float64, :points => Base.Vector{Percentiles_Point}]
            meta(target, Percentiles, allflds, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES)
        end
        __meta_Percentiles[]
    end
end
function Base.getproperty(obj::Percentiles, name::Symbol)
    if name === :start_nstime
        return (obj.__protobuf_jl_internal_values[name])::UInt64
    elseif name === :end_nstime
        return (obj.__protobuf_jl_internal_values[name])::UInt64
    elseif name === :min_value
        return (obj.__protobuf_jl_internal_values[name])::Float64
    elseif name === :max_value
        return (obj.__protobuf_jl_internal_values[name])::Float64
    elseif name === :mean
        return (obj.__protobuf_jl_internal_values[name])::Float64
    elseif name === :stddev
        return (obj.__protobuf_jl_internal_values[name])::Float64
    elseif name === :num_samples
        return (obj.__protobuf_jl_internal_values[name])::UInt64
    elseif name === :total_samples
        return (obj.__protobuf_jl_internal_values[name])::UInt64
    elseif name === :accumulator
        return (obj.__protobuf_jl_internal_values[name])::Float64
    elseif name === :points
        return (obj.__protobuf_jl_internal_values[name])::Base.Vector{Percentiles_Point}
    else
        getfield(obj, name)
    end
end

const MetricValues_UnitOfMeasure = (;[
    Symbol("INVALID") => Int32(0),
    Symbol("NUMBER") => Int32(1),
    Symbol("TIME") => Int32(2),
    Symbol("BYTES") => Int32(3),
]...)

mutable struct MetricValues <: ProtoType
    __protobuf_jl_internal_meta::ProtoMeta
    __protobuf_jl_internal_values::Dict{Symbol,Any}
    __protobuf_jl_internal_defaultset::Set{Symbol}

    function MetricValues(; kwargs...)
        obj = new(meta(MetricValues), Dict{Symbol,Any}(), Set{Symbol}())
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
end # mutable struct MetricValues
const __meta_MetricValues = Ref{ProtoMeta}()
function meta(::Type{MetricValues})
    ProtoBuf.metalock() do
        if !isassigned(__meta_MetricValues)
            __meta_MetricValues[] = target = ProtoMeta(MetricValues)
            allflds = Pair{Symbol,Union{Type,String}}[:name => AbstractString, :percentiles_value => Percentiles, :int64_value => Int64, :unit_of_measure => Int32]
            oneofs = Int[0,1,1,0]
            oneof_names = Symbol[Symbol("values_oneof")]
            meta(target, MetricValues, allflds, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, oneofs, oneof_names)
        end
        __meta_MetricValues[]
    end
end
function Base.getproperty(obj::MetricValues, name::Symbol)
    if name === :name
        return (obj.__protobuf_jl_internal_values[name])::AbstractString
    elseif name === :percentiles_value
        return (obj.__protobuf_jl_internal_values[name])::Percentiles
    elseif name === :int64_value
        return (obj.__protobuf_jl_internal_values[name])::Int64
    elseif name === :unit_of_measure
        return (obj.__protobuf_jl_internal_values[name])::Int32
    else
        getfield(obj, name)
    end
end

mutable struct MetricsReport <: ProtoType
    __protobuf_jl_internal_meta::ProtoMeta
    __protobuf_jl_internal_values::Dict{Symbol,Any}
    __protobuf_jl_internal_defaultset::Set{Symbol}

    function MetricsReport(; kwargs...)
        obj = new(meta(MetricsReport), Dict{Symbol,Any}(), Set{Symbol}())
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
end # mutable struct MetricsReport
const __meta_MetricsReport = Ref{ProtoMeta}()
function meta(::Type{MetricsReport})
    ProtoBuf.metalock() do
        if !isassigned(__meta_MetricsReport)
            __meta_MetricsReport[] = target = ProtoMeta(MetricsReport)
            allflds = Pair{Symbol,Union{Type,String}}[:metrics => Base.Vector{MetricValues}]
            meta(target, MetricsReport, allflds, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES)
        end
        __meta_MetricsReport[]
    end
end
function Base.getproperty(obj::MetricsReport, name::Symbol)
    if name === :metrics
        return (obj.__protobuf_jl_internal_values[name])::Base.Vector{MetricValues}
    else
        getfield(obj, name)
    end
end

mutable struct MemoryInfo <: ProtoType
    __protobuf_jl_internal_meta::ProtoMeta
    __protobuf_jl_internal_values::Dict{Symbol,Any}
    __protobuf_jl_internal_defaultset::Set{Symbol}

    function MemoryInfo(; kwargs...)
        obj = new(meta(MemoryInfo), Dict{Symbol,Any}(), Set{Symbol}())
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
end # mutable struct MemoryInfo
const __meta_MemoryInfo = Ref{ProtoMeta}()
function meta(::Type{MemoryInfo})
    ProtoBuf.metalock() do
        if !isassigned(__meta_MemoryInfo)
            __meta_MemoryInfo[] = target = ProtoMeta(MemoryInfo)
            allflds = Pair{Symbol,Union{Type,String}}[:kb_total => Int64, :kb_free => Int64]
            meta(target, MemoryInfo, allflds, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES)
        end
        __meta_MemoryInfo[]
    end
end
function Base.getproperty(obj::MemoryInfo, name::Symbol)
    if name === :kb_total
        return (obj.__protobuf_jl_internal_values[name])::Int64
    elseif name === :kb_free
        return (obj.__protobuf_jl_internal_values[name])::Int64
    else
        getfield(obj, name)
    end
end

export DeviceAssignment_ComputationDevice_DeviceMeshCoordinates, DeviceAssignment_ComputationDevice, DeviceAssignment, XLAComputationConfig_Experimental_UpdateIndexPair, XLAComputationConfig_Experimental, XLAComputationConfig, XLAComputation, XLAAllocation, XLATupleNode, CommonExecutionConfig, XRTExecutionConfig, XRTChainedExecuteConfig, XRTChainedExecuteOp_Input, XRTChainedExecuteOp_Output, XRTChainedExecuteOp, XRTChainedExecutePlan, XRTMetricsCollect, Percentiles_Point, Percentiles, MetricValues_UnitOfMeasure, MetricValues, MetricsReport, MemoryInfo
