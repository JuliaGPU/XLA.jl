# syntax: proto3
using ProtoBuf
import ProtoBuf.meta

const DebugOptions_StepMarkerLocation = (;[
    Symbol("STEP_MARK_AT_ENTRY") => Int32(0),
    Symbol("STEP_MARK_AT_TOP_LEVEL_WHILE_LOOP") => Int32(1),
    Symbol("STEP_MARK_AT_SECOND_LEVEL_WHILE_LOOP") => Int32(3),
    Symbol("STEP_MARK_NONE") => Int32(2),
]...)

mutable struct DebugOptions_XlaBackendExtraOptionsEntry <: ProtoType
    __protobuf_jl_internal_meta::ProtoMeta
    __protobuf_jl_internal_values::Dict{Symbol,Any}
    __protobuf_jl_internal_defaultset::Set{Symbol}

    function DebugOptions_XlaBackendExtraOptionsEntry(; kwargs...)
        obj = new(meta(DebugOptions_XlaBackendExtraOptionsEntry), Dict{Symbol,Any}(), Set{Symbol}())
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
end # mutable struct DebugOptions_XlaBackendExtraOptionsEntry (mapentry)
const __meta_DebugOptions_XlaBackendExtraOptionsEntry = Ref{ProtoMeta}()
function meta(::Type{DebugOptions_XlaBackendExtraOptionsEntry})
    ProtoBuf.metalock() do
        if !isassigned(__meta_DebugOptions_XlaBackendExtraOptionsEntry)
            __meta_DebugOptions_XlaBackendExtraOptionsEntry[] = target = ProtoMeta(DebugOptions_XlaBackendExtraOptionsEntry)
            allflds = Pair{Symbol,Union{Type,String}}[:key => AbstractString, :value => AbstractString]
            meta(target, DebugOptions_XlaBackendExtraOptionsEntry, allflds, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES)
        end
        __meta_DebugOptions_XlaBackendExtraOptionsEntry[]
    end
end
function Base.getproperty(obj::DebugOptions_XlaBackendExtraOptionsEntry, name::Symbol)
    if name === :key
        return (obj.__protobuf_jl_internal_values[name])::AbstractString
    elseif name === :value
        return (obj.__protobuf_jl_internal_values[name])::AbstractString
    else
        getfield(obj, name)
    end
end

mutable struct DebugOptions <: ProtoType
    __protobuf_jl_internal_meta::ProtoMeta
    __protobuf_jl_internal_values::Dict{Symbol,Any}
    __protobuf_jl_internal_defaultset::Set{Symbol}

    function DebugOptions(; kwargs...)
        obj = new(meta(DebugOptions), Dict{Symbol,Any}(), Set{Symbol}())
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
end # mutable struct DebugOptions
const __meta_DebugOptions = Ref{ProtoMeta}()
function meta(::Type{DebugOptions})
    ProtoBuf.metalock() do
        if !isassigned(__meta_DebugOptions)
            __meta_DebugOptions[] = target = ProtoMeta(DebugOptions)
            fnum = Int[2,9,30,124,104,31,33,35,60,61,62,63,134,70,71,72,73,90,91,92,94,97,98,99,120,121,126,129,140,100,122,101,123,102,103,106,107,108,109,110,111,112,113,114,115,116,118,131,132,144,125,146,127,128,130,135,136,137,138,141,142,143,500]
            allflds = Pair{Symbol,Union{Type,String}}[:xla_hlo_graph_addresses => Bool, :xla_hlo_profile => Bool, :xla_disable_hlo_passes => Base.Vector{AbstractString}, :xla_enable_hlo_passes_only => Base.Vector{AbstractString}, :xla_disable_all_hlo_passes => Bool, :xla_backend_optimization_level => Int32, :xla_embed_ir_in_executable => Bool, :xla_eliminate_hlo_implicit_broadcast => Bool, :xla_cpu_multi_thread_eigen => Bool, :xla_gpu_cuda_data_dir => AbstractString, :xla_gpu_ftz => Bool, :xla_gpu_disable_multi_streaming => Bool, :xla_gpu_use_random_streams => Bool, :xla_llvm_enable_alias_scope_metadata => Bool, :xla_llvm_enable_noalias_metadata => Bool, :xla_llvm_enable_invariant_load_metadata => Bool, :xla_llvm_disable_expensive_passes => Bool, :xla_test_all_output_layouts => Bool, :xla_test_all_input_layouts => Bool, :xla_hlo_graph_sharding_color => Bool, :xla_gpu_use_cudnn_batchnorm => Bool, :xla_cpu_use_mkl_dnn => Bool, :xla_gpu_max_kernel_unroll_factor => Int32, :xla_cpu_enable_fast_math => Bool, :xla_cpu_fast_math_honor_nans => Bool, :xla_cpu_fast_math_honor_infs => Bool, :xla_cpu_fast_math_honor_division => Bool, :xla_cpu_fast_math_honor_functions => Bool, :xla_cpu_enable_fast_min_max => Bool, :xla_gpu_enable_fast_min_max => Bool, :xla_allow_excess_precision => Bool, :xla_gpu_crash_on_verification_failures => Bool, :xla_gpu_autotune_level => Int32, :xla_force_host_platform_device_count => Int32, :xla_gpu_disable_gpuasm_optimizations => Bool, :xla_hlo_evaluator_use_fast_path => Bool, :xla_allow_scalar_index_dynamic_ops => Bool, :xla_step_marker_location => Int32, :xla_dump_to => AbstractString, :xla_dump_hlo_module_re => AbstractString, :xla_dump_hlo_pass_re => AbstractString, :xla_dump_hlo_as_text => Bool, :xla_dump_hlo_as_proto => Bool, :xla_dump_hlo_as_dot => Bool, :xla_dump_hlo_as_url => Bool, :xla_dump_hlo_as_html => Bool, :xla_dump_hlo_snapshots => Bool, :xla_dump_include_timestamp => Bool, :xla_dump_max_hlo_modules => Int32, :xla_dump_module_metadata => Bool, :xla_gpu_force_conv_nchw => Bool, :xla_gpu_force_conv_nhwc => Bool, :xla_gpu_ptx_file => Base.Vector{AbstractString}, :xla_gpu_algorithm_denylist_path => AbstractString, :xla_gpu_deterministic_reductions => Bool, :xla_tpu_detect_nan => Bool, :xla_tpu_detect_inf => Bool, :xla_cpu_enable_xprof_traceme => Bool, :xla_gpu_unsafe_fallback_to_driver_on_ptxas_not_found => Bool, :xla_gpu_asm_extra_flags => AbstractString, :xla_multiheap_size_constraint_per_heap => Int32, :xla_detailed_logging => Bool, :xla_backend_extra_options => Base.Dict{AbstractString,AbstractString}]
            meta(target, DebugOptions, allflds, ProtoBuf.DEF_REQ, fnum, ProtoBuf.DEF_VAL, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES)
        end
        __meta_DebugOptions[]
    end
end
function Base.getproperty(obj::DebugOptions, name::Symbol)
    if name === :xla_hlo_graph_addresses
        return (obj.__protobuf_jl_internal_values[name])::Bool
    elseif name === :xla_hlo_profile
        return (obj.__protobuf_jl_internal_values[name])::Bool
    elseif name === :xla_disable_hlo_passes
        return (obj.__protobuf_jl_internal_values[name])::Base.Vector{AbstractString}
    elseif name === :xla_enable_hlo_passes_only
        return (obj.__protobuf_jl_internal_values[name])::Base.Vector{AbstractString}
    elseif name === :xla_disable_all_hlo_passes
        return (obj.__protobuf_jl_internal_values[name])::Bool
    elseif name === :xla_backend_optimization_level
        return (obj.__protobuf_jl_internal_values[name])::Int32
    elseif name === :xla_embed_ir_in_executable
        return (obj.__protobuf_jl_internal_values[name])::Bool
    elseif name === :xla_eliminate_hlo_implicit_broadcast
        return (obj.__protobuf_jl_internal_values[name])::Bool
    elseif name === :xla_cpu_multi_thread_eigen
        return (obj.__protobuf_jl_internal_values[name])::Bool
    elseif name === :xla_gpu_cuda_data_dir
        return (obj.__protobuf_jl_internal_values[name])::AbstractString
    elseif name === :xla_gpu_ftz
        return (obj.__protobuf_jl_internal_values[name])::Bool
    elseif name === :xla_gpu_disable_multi_streaming
        return (obj.__protobuf_jl_internal_values[name])::Bool
    elseif name === :xla_gpu_use_random_streams
        return (obj.__protobuf_jl_internal_values[name])::Bool
    elseif name === :xla_llvm_enable_alias_scope_metadata
        return (obj.__protobuf_jl_internal_values[name])::Bool
    elseif name === :xla_llvm_enable_noalias_metadata
        return (obj.__protobuf_jl_internal_values[name])::Bool
    elseif name === :xla_llvm_enable_invariant_load_metadata
        return (obj.__protobuf_jl_internal_values[name])::Bool
    elseif name === :xla_llvm_disable_expensive_passes
        return (obj.__protobuf_jl_internal_values[name])::Bool
    elseif name === :xla_test_all_output_layouts
        return (obj.__protobuf_jl_internal_values[name])::Bool
    elseif name === :xla_test_all_input_layouts
        return (obj.__protobuf_jl_internal_values[name])::Bool
    elseif name === :xla_hlo_graph_sharding_color
        return (obj.__protobuf_jl_internal_values[name])::Bool
    elseif name === :xla_gpu_use_cudnn_batchnorm
        return (obj.__protobuf_jl_internal_values[name])::Bool
    elseif name === :xla_cpu_use_mkl_dnn
        return (obj.__protobuf_jl_internal_values[name])::Bool
    elseif name === :xla_gpu_max_kernel_unroll_factor
        return (obj.__protobuf_jl_internal_values[name])::Int32
    elseif name === :xla_cpu_enable_fast_math
        return (obj.__protobuf_jl_internal_values[name])::Bool
    elseif name === :xla_cpu_fast_math_honor_nans
        return (obj.__protobuf_jl_internal_values[name])::Bool
    elseif name === :xla_cpu_fast_math_honor_infs
        return (obj.__protobuf_jl_internal_values[name])::Bool
    elseif name === :xla_cpu_fast_math_honor_division
        return (obj.__protobuf_jl_internal_values[name])::Bool
    elseif name === :xla_cpu_fast_math_honor_functions
        return (obj.__protobuf_jl_internal_values[name])::Bool
    elseif name === :xla_cpu_enable_fast_min_max
        return (obj.__protobuf_jl_internal_values[name])::Bool
    elseif name === :xla_gpu_enable_fast_min_max
        return (obj.__protobuf_jl_internal_values[name])::Bool
    elseif name === :xla_allow_excess_precision
        return (obj.__protobuf_jl_internal_values[name])::Bool
    elseif name === :xla_gpu_crash_on_verification_failures
        return (obj.__protobuf_jl_internal_values[name])::Bool
    elseif name === :xla_gpu_autotune_level
        return (obj.__protobuf_jl_internal_values[name])::Int32
    elseif name === :xla_force_host_platform_device_count
        return (obj.__protobuf_jl_internal_values[name])::Int32
    elseif name === :xla_gpu_disable_gpuasm_optimizations
        return (obj.__protobuf_jl_internal_values[name])::Bool
    elseif name === :xla_hlo_evaluator_use_fast_path
        return (obj.__protobuf_jl_internal_values[name])::Bool
    elseif name === :xla_allow_scalar_index_dynamic_ops
        return (obj.__protobuf_jl_internal_values[name])::Bool
    elseif name === :xla_step_marker_location
        return (obj.__protobuf_jl_internal_values[name])::Int32
    elseif name === :xla_dump_to
        return (obj.__protobuf_jl_internal_values[name])::AbstractString
    elseif name === :xla_dump_hlo_module_re
        return (obj.__protobuf_jl_internal_values[name])::AbstractString
    elseif name === :xla_dump_hlo_pass_re
        return (obj.__protobuf_jl_internal_values[name])::AbstractString
    elseif name === :xla_dump_hlo_as_text
        return (obj.__protobuf_jl_internal_values[name])::Bool
    elseif name === :xla_dump_hlo_as_proto
        return (obj.__protobuf_jl_internal_values[name])::Bool
    elseif name === :xla_dump_hlo_as_dot
        return (obj.__protobuf_jl_internal_values[name])::Bool
    elseif name === :xla_dump_hlo_as_url
        return (obj.__protobuf_jl_internal_values[name])::Bool
    elseif name === :xla_dump_hlo_as_html
        return (obj.__protobuf_jl_internal_values[name])::Bool
    elseif name === :xla_dump_hlo_snapshots
        return (obj.__protobuf_jl_internal_values[name])::Bool
    elseif name === :xla_dump_include_timestamp
        return (obj.__protobuf_jl_internal_values[name])::Bool
    elseif name === :xla_dump_max_hlo_modules
        return (obj.__protobuf_jl_internal_values[name])::Int32
    elseif name === :xla_dump_module_metadata
        return (obj.__protobuf_jl_internal_values[name])::Bool
    elseif name === :xla_gpu_force_conv_nchw
        return (obj.__protobuf_jl_internal_values[name])::Bool
    elseif name === :xla_gpu_force_conv_nhwc
        return (obj.__protobuf_jl_internal_values[name])::Bool
    elseif name === :xla_gpu_ptx_file
        return (obj.__protobuf_jl_internal_values[name])::Base.Vector{AbstractString}
    elseif name === :xla_gpu_algorithm_denylist_path
        return (obj.__protobuf_jl_internal_values[name])::AbstractString
    elseif name === :xla_gpu_deterministic_reductions
        return (obj.__protobuf_jl_internal_values[name])::Bool
    elseif name === :xla_tpu_detect_nan
        return (obj.__protobuf_jl_internal_values[name])::Bool
    elseif name === :xla_tpu_detect_inf
        return (obj.__protobuf_jl_internal_values[name])::Bool
    elseif name === :xla_cpu_enable_xprof_traceme
        return (obj.__protobuf_jl_internal_values[name])::Bool
    elseif name === :xla_gpu_unsafe_fallback_to_driver_on_ptxas_not_found
        return (obj.__protobuf_jl_internal_values[name])::Bool
    elseif name === :xla_gpu_asm_extra_flags
        return (obj.__protobuf_jl_internal_values[name])::AbstractString
    elseif name === :xla_multiheap_size_constraint_per_heap
        return (obj.__protobuf_jl_internal_values[name])::Int32
    elseif name === :xla_detailed_logging
        return (obj.__protobuf_jl_internal_values[name])::Bool
    elseif name === :xla_backend_extra_options
        return (obj.__protobuf_jl_internal_values[name])::Base.Dict{AbstractString,AbstractString}
    else
        getfield(obj, name)
    end
end

mutable struct ExecutionOptions <: ProtoType
    __protobuf_jl_internal_meta::ProtoMeta
    __protobuf_jl_internal_values::Dict{Symbol,Any}
    __protobuf_jl_internal_defaultset::Set{Symbol}

    function ExecutionOptions(; kwargs...)
        obj = new(meta(ExecutionOptions), Dict{Symbol,Any}(), Set{Symbol}())
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
end # mutable struct ExecutionOptions
const __meta_ExecutionOptions = Ref{ProtoMeta}()
function meta(::Type{ExecutionOptions})
    ProtoBuf.metalock() do
        if !isassigned(__meta_ExecutionOptions)
            __meta_ExecutionOptions[] = target = ProtoMeta(ExecutionOptions)
            fnum = Int[2,3,4,5,6,7,8,9,10,11,12]
            allflds = Pair{Symbol,Union{Type,String}}[:shape_with_output_layout => ShapeProto, :seed => UInt64, :debug_options => DebugOptions, :device_handles => Base.Vector{DeviceHandle}, :num_replicas => Int32, :device_assignment => DeviceAssignmentProto, :alias_passthrough_params => Bool, :num_partitions => Int32, :launch_id => Int32, :use_spmd_partitioning => Bool, :deduplicate_hlo => Bool]
            meta(target, ExecutionOptions, allflds, ProtoBuf.DEF_REQ, fnum, ProtoBuf.DEF_VAL, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES)
        end
        __meta_ExecutionOptions[]
    end
end
function Base.getproperty(obj::ExecutionOptions, name::Symbol)
    if name === :shape_with_output_layout
        return (obj.__protobuf_jl_internal_values[name])::ShapeProto
    elseif name === :seed
        return (obj.__protobuf_jl_internal_values[name])::UInt64
    elseif name === :debug_options
        return (obj.__protobuf_jl_internal_values[name])::DebugOptions
    elseif name === :device_handles
        return (obj.__protobuf_jl_internal_values[name])::Base.Vector{DeviceHandle}
    elseif name === :num_replicas
        return (obj.__protobuf_jl_internal_values[name])::Int32
    elseif name === :device_assignment
        return (obj.__protobuf_jl_internal_values[name])::DeviceAssignmentProto
    elseif name === :alias_passthrough_params
        return (obj.__protobuf_jl_internal_values[name])::Bool
    elseif name === :num_partitions
        return (obj.__protobuf_jl_internal_values[name])::Int32
    elseif name === :launch_id
        return (obj.__protobuf_jl_internal_values[name])::Int32
    elseif name === :use_spmd_partitioning
        return (obj.__protobuf_jl_internal_values[name])::Bool
    elseif name === :deduplicate_hlo
        return (obj.__protobuf_jl_internal_values[name])::Bool
    else
        getfield(obj, name)
    end
end

mutable struct GetDeviceHandlesRequest <: ProtoType
    __protobuf_jl_internal_meta::ProtoMeta
    __protobuf_jl_internal_values::Dict{Symbol,Any}
    __protobuf_jl_internal_defaultset::Set{Symbol}

    function GetDeviceHandlesRequest(; kwargs...)
        obj = new(meta(GetDeviceHandlesRequest), Dict{Symbol,Any}(), Set{Symbol}())
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
end # mutable struct GetDeviceHandlesRequest
const __meta_GetDeviceHandlesRequest = Ref{ProtoMeta}()
function meta(::Type{GetDeviceHandlesRequest})
    ProtoBuf.metalock() do
        if !isassigned(__meta_GetDeviceHandlesRequest)
            __meta_GetDeviceHandlesRequest[] = target = ProtoMeta(GetDeviceHandlesRequest)
            allflds = Pair{Symbol,Union{Type,String}}[:device_count => Int64]
            meta(target, GetDeviceHandlesRequest, allflds, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES)
        end
        __meta_GetDeviceHandlesRequest[]
    end
end
function Base.getproperty(obj::GetDeviceHandlesRequest, name::Symbol)
    if name === :device_count
        return (obj.__protobuf_jl_internal_values[name])::Int64
    else
        getfield(obj, name)
    end
end

mutable struct GetDeviceHandlesResponse <: ProtoType
    __protobuf_jl_internal_meta::ProtoMeta
    __protobuf_jl_internal_values::Dict{Symbol,Any}
    __protobuf_jl_internal_defaultset::Set{Symbol}

    function GetDeviceHandlesResponse(; kwargs...)
        obj = new(meta(GetDeviceHandlesResponse), Dict{Symbol,Any}(), Set{Symbol}())
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
end # mutable struct GetDeviceHandlesResponse
const __meta_GetDeviceHandlesResponse = Ref{ProtoMeta}()
function meta(::Type{GetDeviceHandlesResponse})
    ProtoBuf.metalock() do
        if !isassigned(__meta_GetDeviceHandlesResponse)
            __meta_GetDeviceHandlesResponse[] = target = ProtoMeta(GetDeviceHandlesResponse)
            allflds = Pair{Symbol,Union{Type,String}}[:device_handles => Base.Vector{DeviceHandle}]
            meta(target, GetDeviceHandlesResponse, allflds, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES)
        end
        __meta_GetDeviceHandlesResponse[]
    end
end
function Base.getproperty(obj::GetDeviceHandlesResponse, name::Symbol)
    if name === :device_handles
        return (obj.__protobuf_jl_internal_values[name])::Base.Vector{DeviceHandle}
    else
        getfield(obj, name)
    end
end

mutable struct TransferToClientRequest <: ProtoType
    __protobuf_jl_internal_meta::ProtoMeta
    __protobuf_jl_internal_values::Dict{Symbol,Any}
    __protobuf_jl_internal_defaultset::Set{Symbol}

    function TransferToClientRequest(; kwargs...)
        obj = new(meta(TransferToClientRequest), Dict{Symbol,Any}(), Set{Symbol}())
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
end # mutable struct TransferToClientRequest
const __meta_TransferToClientRequest = Ref{ProtoMeta}()
function meta(::Type{TransferToClientRequest})
    ProtoBuf.metalock() do
        if !isassigned(__meta_TransferToClientRequest)
            __meta_TransferToClientRequest[] = target = ProtoMeta(TransferToClientRequest)
            allflds = Pair{Symbol,Union{Type,String}}[:data => GlobalDataHandle, :shape_with_layout => ShapeProto]
            meta(target, TransferToClientRequest, allflds, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES)
        end
        __meta_TransferToClientRequest[]
    end
end
function Base.getproperty(obj::TransferToClientRequest, name::Symbol)
    if name === :data
        return (obj.__protobuf_jl_internal_values[name])::GlobalDataHandle
    elseif name === :shape_with_layout
        return (obj.__protobuf_jl_internal_values[name])::ShapeProto
    else
        getfield(obj, name)
    end
end

mutable struct TransferToClientResponse <: ProtoType
    __protobuf_jl_internal_meta::ProtoMeta
    __protobuf_jl_internal_values::Dict{Symbol,Any}
    __protobuf_jl_internal_defaultset::Set{Symbol}

    function TransferToClientResponse(; kwargs...)
        obj = new(meta(TransferToClientResponse), Dict{Symbol,Any}(), Set{Symbol}())
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
end # mutable struct TransferToClientResponse
const __meta_TransferToClientResponse = Ref{ProtoMeta}()
function meta(::Type{TransferToClientResponse})
    ProtoBuf.metalock() do
        if !isassigned(__meta_TransferToClientResponse)
            __meta_TransferToClientResponse[] = target = ProtoMeta(TransferToClientResponse)
            allflds = Pair{Symbol,Union{Type,String}}[:literal => LiteralProto]
            meta(target, TransferToClientResponse, allflds, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES)
        end
        __meta_TransferToClientResponse[]
    end
end
function Base.getproperty(obj::TransferToClientResponse, name::Symbol)
    if name === :literal
        return (obj.__protobuf_jl_internal_values[name])::LiteralProto
    else
        getfield(obj, name)
    end
end

mutable struct TransferToServerRequest <: ProtoType
    __protobuf_jl_internal_meta::ProtoMeta
    __protobuf_jl_internal_values::Dict{Symbol,Any}
    __protobuf_jl_internal_defaultset::Set{Symbol}

    function TransferToServerRequest(; kwargs...)
        obj = new(meta(TransferToServerRequest), Dict{Symbol,Any}(), Set{Symbol}())
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
end # mutable struct TransferToServerRequest
const __meta_TransferToServerRequest = Ref{ProtoMeta}()
function meta(::Type{TransferToServerRequest})
    ProtoBuf.metalock() do
        if !isassigned(__meta_TransferToServerRequest)
            __meta_TransferToServerRequest[] = target = ProtoMeta(TransferToServerRequest)
            allflds = Pair{Symbol,Union{Type,String}}[:literal => LiteralProto, :device_handle => DeviceHandle]
            meta(target, TransferToServerRequest, allflds, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES)
        end
        __meta_TransferToServerRequest[]
    end
end
function Base.getproperty(obj::TransferToServerRequest, name::Symbol)
    if name === :literal
        return (obj.__protobuf_jl_internal_values[name])::LiteralProto
    elseif name === :device_handle
        return (obj.__protobuf_jl_internal_values[name])::DeviceHandle
    else
        getfield(obj, name)
    end
end

mutable struct TransferToServerResponse <: ProtoType
    __protobuf_jl_internal_meta::ProtoMeta
    __protobuf_jl_internal_values::Dict{Symbol,Any}
    __protobuf_jl_internal_defaultset::Set{Symbol}

    function TransferToServerResponse(; kwargs...)
        obj = new(meta(TransferToServerResponse), Dict{Symbol,Any}(), Set{Symbol}())
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
end # mutable struct TransferToServerResponse
const __meta_TransferToServerResponse = Ref{ProtoMeta}()
function meta(::Type{TransferToServerResponse})
    ProtoBuf.metalock() do
        if !isassigned(__meta_TransferToServerResponse)
            __meta_TransferToServerResponse[] = target = ProtoMeta(TransferToServerResponse)
            allflds = Pair{Symbol,Union{Type,String}}[:data => GlobalDataHandle]
            meta(target, TransferToServerResponse, allflds, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES)
        end
        __meta_TransferToServerResponse[]
    end
end
function Base.getproperty(obj::TransferToServerResponse, name::Symbol)
    if name === :data
        return (obj.__protobuf_jl_internal_values[name])::GlobalDataHandle
    else
        getfield(obj, name)
    end
end

mutable struct TransferToInfeedRequest <: ProtoType
    __protobuf_jl_internal_meta::ProtoMeta
    __protobuf_jl_internal_values::Dict{Symbol,Any}
    __protobuf_jl_internal_defaultset::Set{Symbol}

    function TransferToInfeedRequest(; kwargs...)
        obj = new(meta(TransferToInfeedRequest), Dict{Symbol,Any}(), Set{Symbol}())
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
end # mutable struct TransferToInfeedRequest
const __meta_TransferToInfeedRequest = Ref{ProtoMeta}()
function meta(::Type{TransferToInfeedRequest})
    ProtoBuf.metalock() do
        if !isassigned(__meta_TransferToInfeedRequest)
            __meta_TransferToInfeedRequest[] = target = ProtoMeta(TransferToInfeedRequest)
            allflds = Pair{Symbol,Union{Type,String}}[:literal => LiteralProto, :replica_id => Int64, :device_handle => DeviceHandle]
            meta(target, TransferToInfeedRequest, allflds, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES)
        end
        __meta_TransferToInfeedRequest[]
    end
end
function Base.getproperty(obj::TransferToInfeedRequest, name::Symbol)
    if name === :literal
        return (obj.__protobuf_jl_internal_values[name])::LiteralProto
    elseif name === :replica_id
        return (obj.__protobuf_jl_internal_values[name])::Int64
    elseif name === :device_handle
        return (obj.__protobuf_jl_internal_values[name])::DeviceHandle
    else
        getfield(obj, name)
    end
end

mutable struct TransferToInfeedResponse <: ProtoType
    __protobuf_jl_internal_meta::ProtoMeta
    __protobuf_jl_internal_values::Dict{Symbol,Any}
    __protobuf_jl_internal_defaultset::Set{Symbol}

    function TransferToInfeedResponse(; kwargs...)
        obj = new(meta(TransferToInfeedResponse), Dict{Symbol,Any}(), Set{Symbol}())
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
end # mutable struct TransferToInfeedResponse
const __meta_TransferToInfeedResponse = Ref{ProtoMeta}()
function meta(::Type{TransferToInfeedResponse})
    ProtoBuf.metalock() do
        if !isassigned(__meta_TransferToInfeedResponse)
            __meta_TransferToInfeedResponse[] = target = ProtoMeta(TransferToInfeedResponse)
            allflds = Pair{Symbol,Union{Type,String}}[]
            meta(target, TransferToInfeedResponse, allflds, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES)
        end
        __meta_TransferToInfeedResponse[]
    end
end

mutable struct TransferFromOutfeedRequest <: ProtoType
    __protobuf_jl_internal_meta::ProtoMeta
    __protobuf_jl_internal_values::Dict{Symbol,Any}
    __protobuf_jl_internal_defaultset::Set{Symbol}

    function TransferFromOutfeedRequest(; kwargs...)
        obj = new(meta(TransferFromOutfeedRequest), Dict{Symbol,Any}(), Set{Symbol}())
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
end # mutable struct TransferFromOutfeedRequest
const __meta_TransferFromOutfeedRequest = Ref{ProtoMeta}()
function meta(::Type{TransferFromOutfeedRequest})
    ProtoBuf.metalock() do
        if !isassigned(__meta_TransferFromOutfeedRequest)
            __meta_TransferFromOutfeedRequest[] = target = ProtoMeta(TransferFromOutfeedRequest)
            allflds = Pair{Symbol,Union{Type,String}}[:shape_with_layout => ShapeProto, :replica_id => Int64, :device_handle => DeviceHandle]
            meta(target, TransferFromOutfeedRequest, allflds, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES)
        end
        __meta_TransferFromOutfeedRequest[]
    end
end
function Base.getproperty(obj::TransferFromOutfeedRequest, name::Symbol)
    if name === :shape_with_layout
        return (obj.__protobuf_jl_internal_values[name])::ShapeProto
    elseif name === :replica_id
        return (obj.__protobuf_jl_internal_values[name])::Int64
    elseif name === :device_handle
        return (obj.__protobuf_jl_internal_values[name])::DeviceHandle
    else
        getfield(obj, name)
    end
end

mutable struct TransferFromOutfeedResponse <: ProtoType
    __protobuf_jl_internal_meta::ProtoMeta
    __protobuf_jl_internal_values::Dict{Symbol,Any}
    __protobuf_jl_internal_defaultset::Set{Symbol}

    function TransferFromOutfeedResponse(; kwargs...)
        obj = new(meta(TransferFromOutfeedResponse), Dict{Symbol,Any}(), Set{Symbol}())
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
end # mutable struct TransferFromOutfeedResponse
const __meta_TransferFromOutfeedResponse = Ref{ProtoMeta}()
function meta(::Type{TransferFromOutfeedResponse})
    ProtoBuf.metalock() do
        if !isassigned(__meta_TransferFromOutfeedResponse)
            __meta_TransferFromOutfeedResponse[] = target = ProtoMeta(TransferFromOutfeedResponse)
            allflds = Pair{Symbol,Union{Type,String}}[:literal => LiteralProto]
            meta(target, TransferFromOutfeedResponse, allflds, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES)
        end
        __meta_TransferFromOutfeedResponse[]
    end
end
function Base.getproperty(obj::TransferFromOutfeedResponse, name::Symbol)
    if name === :literal
        return (obj.__protobuf_jl_internal_values[name])::LiteralProto
    else
        getfield(obj, name)
    end
end

mutable struct ResetDeviceRequest <: ProtoType
    __protobuf_jl_internal_meta::ProtoMeta
    __protobuf_jl_internal_values::Dict{Symbol,Any}
    __protobuf_jl_internal_defaultset::Set{Symbol}

    function ResetDeviceRequest(; kwargs...)
        obj = new(meta(ResetDeviceRequest), Dict{Symbol,Any}(), Set{Symbol}())
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
end # mutable struct ResetDeviceRequest
const __meta_ResetDeviceRequest = Ref{ProtoMeta}()
function meta(::Type{ResetDeviceRequest})
    ProtoBuf.metalock() do
        if !isassigned(__meta_ResetDeviceRequest)
            __meta_ResetDeviceRequest[] = target = ProtoMeta(ResetDeviceRequest)
            allflds = Pair{Symbol,Union{Type,String}}[:device_handle => DeviceHandle]
            meta(target, ResetDeviceRequest, allflds, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES)
        end
        __meta_ResetDeviceRequest[]
    end
end
function Base.getproperty(obj::ResetDeviceRequest, name::Symbol)
    if name === :device_handle
        return (obj.__protobuf_jl_internal_values[name])::DeviceHandle
    else
        getfield(obj, name)
    end
end

mutable struct ResetDeviceResponse <: ProtoType
    __protobuf_jl_internal_meta::ProtoMeta
    __protobuf_jl_internal_values::Dict{Symbol,Any}
    __protobuf_jl_internal_defaultset::Set{Symbol}

    function ResetDeviceResponse(; kwargs...)
        obj = new(meta(ResetDeviceResponse), Dict{Symbol,Any}(), Set{Symbol}())
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
end # mutable struct ResetDeviceResponse
const __meta_ResetDeviceResponse = Ref{ProtoMeta}()
function meta(::Type{ResetDeviceResponse})
    ProtoBuf.metalock() do
        if !isassigned(__meta_ResetDeviceResponse)
            __meta_ResetDeviceResponse[] = target = ProtoMeta(ResetDeviceResponse)
            allflds = Pair{Symbol,Union{Type,String}}[]
            meta(target, ResetDeviceResponse, allflds, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES)
        end
        __meta_ResetDeviceResponse[]
    end
end

mutable struct ComputationGraphStatsRequest <: ProtoType
    __protobuf_jl_internal_meta::ProtoMeta
    __protobuf_jl_internal_values::Dict{Symbol,Any}
    __protobuf_jl_internal_defaultset::Set{Symbol}

    function ComputationGraphStatsRequest(; kwargs...)
        obj = new(meta(ComputationGraphStatsRequest), Dict{Symbol,Any}(), Set{Symbol}())
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
end # mutable struct ComputationGraphStatsRequest
const __meta_ComputationGraphStatsRequest = Ref{ProtoMeta}()
function meta(::Type{ComputationGraphStatsRequest})
    ProtoBuf.metalock() do
        if !isassigned(__meta_ComputationGraphStatsRequest)
            __meta_ComputationGraphStatsRequest[] = target = ProtoMeta(ComputationGraphStatsRequest)
            allflds = Pair{Symbol,Union{Type,String}}[:computation => HloModuleProto, :debug_options => DebugOptions]
            meta(target, ComputationGraphStatsRequest, allflds, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES)
        end
        __meta_ComputationGraphStatsRequest[]
    end
end
function Base.getproperty(obj::ComputationGraphStatsRequest, name::Symbol)
    if name === :computation
        return (obj.__protobuf_jl_internal_values[name])::HloModuleProto
    elseif name === :debug_options
        return (obj.__protobuf_jl_internal_values[name])::DebugOptions
    else
        getfield(obj, name)
    end
end

mutable struct ComputationStatsResponse <: ProtoType
    __protobuf_jl_internal_meta::ProtoMeta
    __protobuf_jl_internal_values::Dict{Symbol,Any}
    __protobuf_jl_internal_defaultset::Set{Symbol}

    function ComputationStatsResponse(; kwargs...)
        obj = new(meta(ComputationStatsResponse), Dict{Symbol,Any}(), Set{Symbol}())
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
end # mutable struct ComputationStatsResponse
const __meta_ComputationStatsResponse = Ref{ProtoMeta}()
function meta(::Type{ComputationStatsResponse})
    ProtoBuf.metalock() do
        if !isassigned(__meta_ComputationStatsResponse)
            __meta_ComputationStatsResponse[] = target = ProtoMeta(ComputationStatsResponse)
            allflds = Pair{Symbol,Union{Type,String}}[:stats => ComputationStats]
            meta(target, ComputationStatsResponse, allflds, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES)
        end
        __meta_ComputationStatsResponse[]
    end
end
function Base.getproperty(obj::ComputationStatsResponse, name::Symbol)
    if name === :stats
        return (obj.__protobuf_jl_internal_values[name])::ComputationStats
    else
        getfield(obj, name)
    end
end

mutable struct CreateChannelHandleRequest <: ProtoType
    __protobuf_jl_internal_meta::ProtoMeta
    __protobuf_jl_internal_values::Dict{Symbol,Any}
    __protobuf_jl_internal_defaultset::Set{Symbol}

    function CreateChannelHandleRequest(; kwargs...)
        obj = new(meta(CreateChannelHandleRequest), Dict{Symbol,Any}(), Set{Symbol}())
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
end # mutable struct CreateChannelHandleRequest
const __meta_CreateChannelHandleRequest = Ref{ProtoMeta}()
function meta(::Type{CreateChannelHandleRequest})
    ProtoBuf.metalock() do
        if !isassigned(__meta_CreateChannelHandleRequest)
            __meta_CreateChannelHandleRequest[] = target = ProtoMeta(CreateChannelHandleRequest)
            allflds = Pair{Symbol,Union{Type,String}}[:channel_type => Int32]
            meta(target, CreateChannelHandleRequest, allflds, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES)
        end
        __meta_CreateChannelHandleRequest[]
    end
end
function Base.getproperty(obj::CreateChannelHandleRequest, name::Symbol)
    if name === :channel_type
        return (obj.__protobuf_jl_internal_values[name])::Int32
    else
        getfield(obj, name)
    end
end

mutable struct CreateChannelHandleResponse <: ProtoType
    __protobuf_jl_internal_meta::ProtoMeta
    __protobuf_jl_internal_values::Dict{Symbol,Any}
    __protobuf_jl_internal_defaultset::Set{Symbol}

    function CreateChannelHandleResponse(; kwargs...)
        obj = new(meta(CreateChannelHandleResponse), Dict{Symbol,Any}(), Set{Symbol}())
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
end # mutable struct CreateChannelHandleResponse
const __meta_CreateChannelHandleResponse = Ref{ProtoMeta}()
function meta(::Type{CreateChannelHandleResponse})
    ProtoBuf.metalock() do
        if !isassigned(__meta_CreateChannelHandleResponse)
            __meta_CreateChannelHandleResponse[] = target = ProtoMeta(CreateChannelHandleResponse)
            allflds = Pair{Symbol,Union{Type,String}}[:channel => ChannelHandle]
            meta(target, CreateChannelHandleResponse, allflds, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES)
        end
        __meta_CreateChannelHandleResponse[]
    end
end
function Base.getproperty(obj::CreateChannelHandleResponse, name::Symbol)
    if name === :channel
        return (obj.__protobuf_jl_internal_values[name])::ChannelHandle
    else
        getfield(obj, name)
    end
end

mutable struct UnregisterRequest <: ProtoType
    __protobuf_jl_internal_meta::ProtoMeta
    __protobuf_jl_internal_values::Dict{Symbol,Any}
    __protobuf_jl_internal_defaultset::Set{Symbol}

    function UnregisterRequest(; kwargs...)
        obj = new(meta(UnregisterRequest), Dict{Symbol,Any}(), Set{Symbol}())
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
end # mutable struct UnregisterRequest
const __meta_UnregisterRequest = Ref{ProtoMeta}()
function meta(::Type{UnregisterRequest})
    ProtoBuf.metalock() do
        if !isassigned(__meta_UnregisterRequest)
            __meta_UnregisterRequest[] = target = ProtoMeta(UnregisterRequest)
            allflds = Pair{Symbol,Union{Type,String}}[:data => Base.Vector{GlobalDataHandle}]
            meta(target, UnregisterRequest, allflds, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES)
        end
        __meta_UnregisterRequest[]
    end
end
function Base.getproperty(obj::UnregisterRequest, name::Symbol)
    if name === :data
        return (obj.__protobuf_jl_internal_values[name])::Base.Vector{GlobalDataHandle}
    else
        getfield(obj, name)
    end
end

mutable struct UnregisterResponse <: ProtoType
    __protobuf_jl_internal_meta::ProtoMeta
    __protobuf_jl_internal_values::Dict{Symbol,Any}
    __protobuf_jl_internal_defaultset::Set{Symbol}

    function UnregisterResponse(; kwargs...)
        obj = new(meta(UnregisterResponse), Dict{Symbol,Any}(), Set{Symbol}())
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
end # mutable struct UnregisterResponse
const __meta_UnregisterResponse = Ref{ProtoMeta}()
function meta(::Type{UnregisterResponse})
    ProtoBuf.metalock() do
        if !isassigned(__meta_UnregisterResponse)
            __meta_UnregisterResponse[] = target = ProtoMeta(UnregisterResponse)
            allflds = Pair{Symbol,Union{Type,String}}[]
            meta(target, UnregisterResponse, allflds, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES)
        end
        __meta_UnregisterResponse[]
    end
end

mutable struct CompileRequest <: ProtoType
    __protobuf_jl_internal_meta::ProtoMeta
    __protobuf_jl_internal_values::Dict{Symbol,Any}
    __protobuf_jl_internal_defaultset::Set{Symbol}

    function CompileRequest(; kwargs...)
        obj = new(meta(CompileRequest), Dict{Symbol,Any}(), Set{Symbol}())
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
end # mutable struct CompileRequest
const __meta_CompileRequest = Ref{ProtoMeta}()
function meta(::Type{CompileRequest})
    ProtoBuf.metalock() do
        if !isassigned(__meta_CompileRequest)
            __meta_CompileRequest[] = target = ProtoMeta(CompileRequest)
            allflds = Pair{Symbol,Union{Type,String}}[:computation => HloModuleProto, :execution_options => ExecutionOptions, :input_shape_with_layout => Base.Vector{ShapeProto}]
            meta(target, CompileRequest, allflds, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES)
        end
        __meta_CompileRequest[]
    end
end
function Base.getproperty(obj::CompileRequest, name::Symbol)
    if name === :computation
        return (obj.__protobuf_jl_internal_values[name])::HloModuleProto
    elseif name === :execution_options
        return (obj.__protobuf_jl_internal_values[name])::ExecutionOptions
    elseif name === :input_shape_with_layout
        return (obj.__protobuf_jl_internal_values[name])::Base.Vector{ShapeProto}
    else
        getfield(obj, name)
    end
end

mutable struct CompileResponse <: ProtoType
    __protobuf_jl_internal_meta::ProtoMeta
    __protobuf_jl_internal_values::Dict{Symbol,Any}
    __protobuf_jl_internal_defaultset::Set{Symbol}

    function CompileResponse(; kwargs...)
        obj = new(meta(CompileResponse), Dict{Symbol,Any}(), Set{Symbol}())
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
end # mutable struct CompileResponse
const __meta_CompileResponse = Ref{ProtoMeta}()
function meta(::Type{CompileResponse})
    ProtoBuf.metalock() do
        if !isassigned(__meta_CompileResponse)
            __meta_CompileResponse[] = target = ProtoMeta(CompileResponse)
            allflds = Pair{Symbol,Union{Type,String}}[:handle => ExecutionHandle]
            meta(target, CompileResponse, allflds, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES)
        end
        __meta_CompileResponse[]
    end
end
function Base.getproperty(obj::CompileResponse, name::Symbol)
    if name === :handle
        return (obj.__protobuf_jl_internal_values[name])::ExecutionHandle
    else
        getfield(obj, name)
    end
end

mutable struct ExecuteRequest <: ProtoType
    __protobuf_jl_internal_meta::ProtoMeta
    __protobuf_jl_internal_values::Dict{Symbol,Any}
    __protobuf_jl_internal_defaultset::Set{Symbol}

    function ExecuteRequest(; kwargs...)
        obj = new(meta(ExecuteRequest), Dict{Symbol,Any}(), Set{Symbol}())
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
end # mutable struct ExecuteRequest
const __meta_ExecuteRequest = Ref{ProtoMeta}()
function meta(::Type{ExecuteRequest})
    ProtoBuf.metalock() do
        if !isassigned(__meta_ExecuteRequest)
            __meta_ExecuteRequest[] = target = ProtoMeta(ExecuteRequest)
            allflds = Pair{Symbol,Union{Type,String}}[:handle => ExecutionHandle, :arguments => Base.Vector{GlobalDataHandle}]
            meta(target, ExecuteRequest, allflds, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES)
        end
        __meta_ExecuteRequest[]
    end
end
function Base.getproperty(obj::ExecuteRequest, name::Symbol)
    if name === :handle
        return (obj.__protobuf_jl_internal_values[name])::ExecutionHandle
    elseif name === :arguments
        return (obj.__protobuf_jl_internal_values[name])::Base.Vector{GlobalDataHandle}
    else
        getfield(obj, name)
    end
end

mutable struct ExecuteGraphRequest <: ProtoType
    __protobuf_jl_internal_meta::ProtoMeta
    __protobuf_jl_internal_values::Dict{Symbol,Any}
    __protobuf_jl_internal_defaultset::Set{Symbol}

    function ExecuteGraphRequest(; kwargs...)
        obj = new(meta(ExecuteGraphRequest), Dict{Symbol,Any}(), Set{Symbol}())
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
end # mutable struct ExecuteGraphRequest
const __meta_ExecuteGraphRequest = Ref{ProtoMeta}()
function meta(::Type{ExecuteGraphRequest})
    ProtoBuf.metalock() do
        if !isassigned(__meta_ExecuteGraphRequest)
            __meta_ExecuteGraphRequest[] = target = ProtoMeta(ExecuteGraphRequest)
            allflds = Pair{Symbol,Union{Type,String}}[:computation => HloModuleProto, :arguments => Base.Vector{GlobalDataHandle}, :execution_options => ExecutionOptions]
            meta(target, ExecuteGraphRequest, allflds, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES)
        end
        __meta_ExecuteGraphRequest[]
    end
end
function Base.getproperty(obj::ExecuteGraphRequest, name::Symbol)
    if name === :computation
        return (obj.__protobuf_jl_internal_values[name])::HloModuleProto
    elseif name === :arguments
        return (obj.__protobuf_jl_internal_values[name])::Base.Vector{GlobalDataHandle}
    elseif name === :execution_options
        return (obj.__protobuf_jl_internal_values[name])::ExecutionOptions
    else
        getfield(obj, name)
    end
end

mutable struct ExecuteGraphParallelRequest <: ProtoType
    __protobuf_jl_internal_meta::ProtoMeta
    __protobuf_jl_internal_values::Dict{Symbol,Any}
    __protobuf_jl_internal_defaultset::Set{Symbol}

    function ExecuteGraphParallelRequest(; kwargs...)
        obj = new(meta(ExecuteGraphParallelRequest), Dict{Symbol,Any}(), Set{Symbol}())
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
end # mutable struct ExecuteGraphParallelRequest
const __meta_ExecuteGraphParallelRequest = Ref{ProtoMeta}()
function meta(::Type{ExecuteGraphParallelRequest})
    ProtoBuf.metalock() do
        if !isassigned(__meta_ExecuteGraphParallelRequest)
            __meta_ExecuteGraphParallelRequest[] = target = ProtoMeta(ExecuteGraphParallelRequest)
            allflds = Pair{Symbol,Union{Type,String}}[:requests => Base.Vector{ExecuteGraphRequest}]
            meta(target, ExecuteGraphParallelRequest, allflds, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES)
        end
        __meta_ExecuteGraphParallelRequest[]
    end
end
function Base.getproperty(obj::ExecuteGraphParallelRequest, name::Symbol)
    if name === :requests
        return (obj.__protobuf_jl_internal_values[name])::Base.Vector{ExecuteGraphRequest}
    else
        getfield(obj, name)
    end
end

mutable struct ExecuteResponse <: ProtoType
    __protobuf_jl_internal_meta::ProtoMeta
    __protobuf_jl_internal_values::Dict{Symbol,Any}
    __protobuf_jl_internal_defaultset::Set{Symbol}

    function ExecuteResponse(; kwargs...)
        obj = new(meta(ExecuteResponse), Dict{Symbol,Any}(), Set{Symbol}())
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
end # mutable struct ExecuteResponse
const __meta_ExecuteResponse = Ref{ProtoMeta}()
function meta(::Type{ExecuteResponse})
    ProtoBuf.metalock() do
        if !isassigned(__meta_ExecuteResponse)
            __meta_ExecuteResponse[] = target = ProtoMeta(ExecuteResponse)
            allflds = Pair{Symbol,Union{Type,String}}[:output => GlobalDataHandle, :profile => ExecutionProfile]
            meta(target, ExecuteResponse, allflds, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES)
        end
        __meta_ExecuteResponse[]
    end
end
function Base.getproperty(obj::ExecuteResponse, name::Symbol)
    if name === :output
        return (obj.__protobuf_jl_internal_values[name])::GlobalDataHandle
    elseif name === :profile
        return (obj.__protobuf_jl_internal_values[name])::ExecutionProfile
    else
        getfield(obj, name)
    end
end

mutable struct ExecuteParallelResponse <: ProtoType
    __protobuf_jl_internal_meta::ProtoMeta
    __protobuf_jl_internal_values::Dict{Symbol,Any}
    __protobuf_jl_internal_defaultset::Set{Symbol}

    function ExecuteParallelResponse(; kwargs...)
        obj = new(meta(ExecuteParallelResponse), Dict{Symbol,Any}(), Set{Symbol}())
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
end # mutable struct ExecuteParallelResponse
const __meta_ExecuteParallelResponse = Ref{ProtoMeta}()
function meta(::Type{ExecuteParallelResponse})
    ProtoBuf.metalock() do
        if !isassigned(__meta_ExecuteParallelResponse)
            __meta_ExecuteParallelResponse[] = target = ProtoMeta(ExecuteParallelResponse)
            allflds = Pair{Symbol,Union{Type,String}}[:responses => Base.Vector{ExecuteResponse}]
            meta(target, ExecuteParallelResponse, allflds, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES)
        end
        __meta_ExecuteParallelResponse[]
    end
end
function Base.getproperty(obj::ExecuteParallelResponse, name::Symbol)
    if name === :responses
        return (obj.__protobuf_jl_internal_values[name])::Base.Vector{ExecuteResponse}
    else
        getfield(obj, name)
    end
end

mutable struct WaitForExecutionRequest <: ProtoType
    __protobuf_jl_internal_meta::ProtoMeta
    __protobuf_jl_internal_values::Dict{Symbol,Any}
    __protobuf_jl_internal_defaultset::Set{Symbol}

    function WaitForExecutionRequest(; kwargs...)
        obj = new(meta(WaitForExecutionRequest), Dict{Symbol,Any}(), Set{Symbol}())
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
end # mutable struct WaitForExecutionRequest
const __meta_WaitForExecutionRequest = Ref{ProtoMeta}()
function meta(::Type{WaitForExecutionRequest})
    ProtoBuf.metalock() do
        if !isassigned(__meta_WaitForExecutionRequest)
            __meta_WaitForExecutionRequest[] = target = ProtoMeta(WaitForExecutionRequest)
            allflds = Pair{Symbol,Union{Type,String}}[:execution => ExecutionHandle]
            meta(target, WaitForExecutionRequest, allflds, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES)
        end
        __meta_WaitForExecutionRequest[]
    end
end
function Base.getproperty(obj::WaitForExecutionRequest, name::Symbol)
    if name === :execution
        return (obj.__protobuf_jl_internal_values[name])::ExecutionHandle
    else
        getfield(obj, name)
    end
end

mutable struct WaitForExecutionResponse <: ProtoType
    __protobuf_jl_internal_meta::ProtoMeta
    __protobuf_jl_internal_values::Dict{Symbol,Any}
    __protobuf_jl_internal_defaultset::Set{Symbol}

    function WaitForExecutionResponse(; kwargs...)
        obj = new(meta(WaitForExecutionResponse), Dict{Symbol,Any}(), Set{Symbol}())
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
end # mutable struct WaitForExecutionResponse
const __meta_WaitForExecutionResponse = Ref{ProtoMeta}()
function meta(::Type{WaitForExecutionResponse})
    ProtoBuf.metalock() do
        if !isassigned(__meta_WaitForExecutionResponse)
            __meta_WaitForExecutionResponse[] = target = ProtoMeta(WaitForExecutionResponse)
            allflds = Pair{Symbol,Union{Type,String}}[:output => GlobalDataHandle, :profile => ExecutionProfile]
            meta(target, WaitForExecutionResponse, allflds, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES)
        end
        __meta_WaitForExecutionResponse[]
    end
end
function Base.getproperty(obj::WaitForExecutionResponse, name::Symbol)
    if name === :output
        return (obj.__protobuf_jl_internal_values[name])::GlobalDataHandle
    elseif name === :profile
        return (obj.__protobuf_jl_internal_values[name])::ExecutionProfile
    else
        getfield(obj, name)
    end
end

mutable struct ComputeConstantGraphRequest <: ProtoType
    __protobuf_jl_internal_meta::ProtoMeta
    __protobuf_jl_internal_values::Dict{Symbol,Any}
    __protobuf_jl_internal_defaultset::Set{Symbol}

    function ComputeConstantGraphRequest(; kwargs...)
        obj = new(meta(ComputeConstantGraphRequest), Dict{Symbol,Any}(), Set{Symbol}())
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
end # mutable struct ComputeConstantGraphRequest
const __meta_ComputeConstantGraphRequest = Ref{ProtoMeta}()
function meta(::Type{ComputeConstantGraphRequest})
    ProtoBuf.metalock() do
        if !isassigned(__meta_ComputeConstantGraphRequest)
            __meta_ComputeConstantGraphRequest[] = target = ProtoMeta(ComputeConstantGraphRequest)
            allflds = Pair{Symbol,Union{Type,String}}[:computation => HloModuleProto, :output_layout => LayoutProto]
            meta(target, ComputeConstantGraphRequest, allflds, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES)
        end
        __meta_ComputeConstantGraphRequest[]
    end
end
function Base.getproperty(obj::ComputeConstantGraphRequest, name::Symbol)
    if name === :computation
        return (obj.__protobuf_jl_internal_values[name])::HloModuleProto
    elseif name === :output_layout
        return (obj.__protobuf_jl_internal_values[name])::LayoutProto
    else
        getfield(obj, name)
    end
end

mutable struct ComputeConstantResponse <: ProtoType
    __protobuf_jl_internal_meta::ProtoMeta
    __protobuf_jl_internal_values::Dict{Symbol,Any}
    __protobuf_jl_internal_defaultset::Set{Symbol}

    function ComputeConstantResponse(; kwargs...)
        obj = new(meta(ComputeConstantResponse), Dict{Symbol,Any}(), Set{Symbol}())
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
end # mutable struct ComputeConstantResponse
const __meta_ComputeConstantResponse = Ref{ProtoMeta}()
function meta(::Type{ComputeConstantResponse})
    ProtoBuf.metalock() do
        if !isassigned(__meta_ComputeConstantResponse)
            __meta_ComputeConstantResponse[] = target = ProtoMeta(ComputeConstantResponse)
            allflds = Pair{Symbol,Union{Type,String}}[:literal => LiteralProto]
            meta(target, ComputeConstantResponse, allflds, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES)
        end
        __meta_ComputeConstantResponse[]
    end
end
function Base.getproperty(obj::ComputeConstantResponse, name::Symbol)
    if name === :literal
        return (obj.__protobuf_jl_internal_values[name])::LiteralProto
    else
        getfield(obj, name)
    end
end

mutable struct DeconstructTupleRequest <: ProtoType
    __protobuf_jl_internal_meta::ProtoMeta
    __protobuf_jl_internal_values::Dict{Symbol,Any}
    __protobuf_jl_internal_defaultset::Set{Symbol}

    function DeconstructTupleRequest(; kwargs...)
        obj = new(meta(DeconstructTupleRequest), Dict{Symbol,Any}(), Set{Symbol}())
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
end # mutable struct DeconstructTupleRequest
const __meta_DeconstructTupleRequest = Ref{ProtoMeta}()
function meta(::Type{DeconstructTupleRequest})
    ProtoBuf.metalock() do
        if !isassigned(__meta_DeconstructTupleRequest)
            __meta_DeconstructTupleRequest[] = target = ProtoMeta(DeconstructTupleRequest)
            fnum = Int[2]
            allflds = Pair{Symbol,Union{Type,String}}[:tuple_handle => GlobalDataHandle]
            meta(target, DeconstructTupleRequest, allflds, ProtoBuf.DEF_REQ, fnum, ProtoBuf.DEF_VAL, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES)
        end
        __meta_DeconstructTupleRequest[]
    end
end
function Base.getproperty(obj::DeconstructTupleRequest, name::Symbol)
    if name === :tuple_handle
        return (obj.__protobuf_jl_internal_values[name])::GlobalDataHandle
    else
        getfield(obj, name)
    end
end

mutable struct DeconstructTupleResponse <: ProtoType
    __protobuf_jl_internal_meta::ProtoMeta
    __protobuf_jl_internal_values::Dict{Symbol,Any}
    __protobuf_jl_internal_defaultset::Set{Symbol}

    function DeconstructTupleResponse(; kwargs...)
        obj = new(meta(DeconstructTupleResponse), Dict{Symbol,Any}(), Set{Symbol}())
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
end # mutable struct DeconstructTupleResponse
const __meta_DeconstructTupleResponse = Ref{ProtoMeta}()
function meta(::Type{DeconstructTupleResponse})
    ProtoBuf.metalock() do
        if !isassigned(__meta_DeconstructTupleResponse)
            __meta_DeconstructTupleResponse[] = target = ProtoMeta(DeconstructTupleResponse)
            allflds = Pair{Symbol,Union{Type,String}}[:element_handles => Base.Vector{GlobalDataHandle}]
            meta(target, DeconstructTupleResponse, allflds, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES)
        end
        __meta_DeconstructTupleResponse[]
    end
end
function Base.getproperty(obj::DeconstructTupleResponse, name::Symbol)
    if name === :element_handles
        return (obj.__protobuf_jl_internal_values[name])::Base.Vector{GlobalDataHandle}
    else
        getfield(obj, name)
    end
end

mutable struct LoadDataRequest <: ProtoType
    __protobuf_jl_internal_meta::ProtoMeta
    __protobuf_jl_internal_values::Dict{Symbol,Any}
    __protobuf_jl_internal_defaultset::Set{Symbol}

    function LoadDataRequest(; kwargs...)
        obj = new(meta(LoadDataRequest), Dict{Symbol,Any}(), Set{Symbol}())
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
end # mutable struct LoadDataRequest
const __meta_LoadDataRequest = Ref{ProtoMeta}()
function meta(::Type{LoadDataRequest})
    ProtoBuf.metalock() do
        if !isassigned(__meta_LoadDataRequest)
            __meta_LoadDataRequest[] = target = ProtoMeta(LoadDataRequest)
            allflds = Pair{Symbol,Union{Type,String}}[:columnio_tablet_path => AbstractString, :columnio_field => AbstractString, :element_shape => ShapeProto, :offset => Int64, :limit => Int64, :zip => Bool]
            meta(target, LoadDataRequest, allflds, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES)
        end
        __meta_LoadDataRequest[]
    end
end
function Base.getproperty(obj::LoadDataRequest, name::Symbol)
    if name === :columnio_tablet_path
        return (obj.__protobuf_jl_internal_values[name])::AbstractString
    elseif name === :columnio_field
        return (obj.__protobuf_jl_internal_values[name])::AbstractString
    elseif name === :element_shape
        return (obj.__protobuf_jl_internal_values[name])::ShapeProto
    elseif name === :offset
        return (obj.__protobuf_jl_internal_values[name])::Int64
    elseif name === :limit
        return (obj.__protobuf_jl_internal_values[name])::Int64
    elseif name === :zip
        return (obj.__protobuf_jl_internal_values[name])::Bool
    else
        getfield(obj, name)
    end
end

mutable struct LoadDataResponse <: ProtoType
    __protobuf_jl_internal_meta::ProtoMeta
    __protobuf_jl_internal_values::Dict{Symbol,Any}
    __protobuf_jl_internal_defaultset::Set{Symbol}

    function LoadDataResponse(; kwargs...)
        obj = new(meta(LoadDataResponse), Dict{Symbol,Any}(), Set{Symbol}())
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
end # mutable struct LoadDataResponse
const __meta_LoadDataResponse = Ref{ProtoMeta}()
function meta(::Type{LoadDataResponse})
    ProtoBuf.metalock() do
        if !isassigned(__meta_LoadDataResponse)
            __meta_LoadDataResponse[] = target = ProtoMeta(LoadDataResponse)
            allflds = Pair{Symbol,Union{Type,String}}[:data => GlobalDataHandle, :data_shape => ShapeProto, :available_rows => Int64, :rows_loaded => Int64, :nanoseconds => Int64]
            meta(target, LoadDataResponse, allflds, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES)
        end
        __meta_LoadDataResponse[]
    end
end
function Base.getproperty(obj::LoadDataResponse, name::Symbol)
    if name === :data
        return (obj.__protobuf_jl_internal_values[name])::GlobalDataHandle
    elseif name === :data_shape
        return (obj.__protobuf_jl_internal_values[name])::ShapeProto
    elseif name === :available_rows
        return (obj.__protobuf_jl_internal_values[name])::Int64
    elseif name === :rows_loaded
        return (obj.__protobuf_jl_internal_values[name])::Int64
    elseif name === :nanoseconds
        return (obj.__protobuf_jl_internal_values[name])::Int64
    else
        getfield(obj, name)
    end
end

mutable struct GetShapeRequest <: ProtoType
    __protobuf_jl_internal_meta::ProtoMeta
    __protobuf_jl_internal_values::Dict{Symbol,Any}
    __protobuf_jl_internal_defaultset::Set{Symbol}

    function GetShapeRequest(; kwargs...)
        obj = new(meta(GetShapeRequest), Dict{Symbol,Any}(), Set{Symbol}())
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
end # mutable struct GetShapeRequest
const __meta_GetShapeRequest = Ref{ProtoMeta}()
function meta(::Type{GetShapeRequest})
    ProtoBuf.metalock() do
        if !isassigned(__meta_GetShapeRequest)
            __meta_GetShapeRequest[] = target = ProtoMeta(GetShapeRequest)
            allflds = Pair{Symbol,Union{Type,String}}[:data => GlobalDataHandle]
            meta(target, GetShapeRequest, allflds, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES)
        end
        __meta_GetShapeRequest[]
    end
end
function Base.getproperty(obj::GetShapeRequest, name::Symbol)
    if name === :data
        return (obj.__protobuf_jl_internal_values[name])::GlobalDataHandle
    else
        getfield(obj, name)
    end
end

mutable struct GetShapeResponse <: ProtoType
    __protobuf_jl_internal_meta::ProtoMeta
    __protobuf_jl_internal_values::Dict{Symbol,Any}
    __protobuf_jl_internal_defaultset::Set{Symbol}

    function GetShapeResponse(; kwargs...)
        obj = new(meta(GetShapeResponse), Dict{Symbol,Any}(), Set{Symbol}())
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
end # mutable struct GetShapeResponse
const __meta_GetShapeResponse = Ref{ProtoMeta}()
function meta(::Type{GetShapeResponse})
    ProtoBuf.metalock() do
        if !isassigned(__meta_GetShapeResponse)
            __meta_GetShapeResponse[] = target = ProtoMeta(GetShapeResponse)
            allflds = Pair{Symbol,Union{Type,String}}[:shape => ShapeProto]
            meta(target, GetShapeResponse, allflds, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES)
        end
        __meta_GetShapeResponse[]
    end
end
function Base.getproperty(obj::GetShapeResponse, name::Symbol)
    if name === :shape
        return (obj.__protobuf_jl_internal_values[name])::ShapeProto
    else
        getfield(obj, name)
    end
end

mutable struct UnpackRequest <: ProtoType
    __protobuf_jl_internal_meta::ProtoMeta
    __protobuf_jl_internal_values::Dict{Symbol,Any}
    __protobuf_jl_internal_defaultset::Set{Symbol}

    function UnpackRequest(; kwargs...)
        obj = new(meta(UnpackRequest), Dict{Symbol,Any}(), Set{Symbol}())
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
end # mutable struct UnpackRequest
const __meta_UnpackRequest = Ref{ProtoMeta}()
function meta(::Type{UnpackRequest})
    ProtoBuf.metalock() do
        if !isassigned(__meta_UnpackRequest)
            __meta_UnpackRequest[] = target = ProtoMeta(UnpackRequest)
            allflds = Pair{Symbol,Union{Type,String}}[:data => GlobalDataHandle]
            meta(target, UnpackRequest, allflds, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES)
        end
        __meta_UnpackRequest[]
    end
end
function Base.getproperty(obj::UnpackRequest, name::Symbol)
    if name === :data
        return (obj.__protobuf_jl_internal_values[name])::GlobalDataHandle
    else
        getfield(obj, name)
    end
end

mutable struct UnpackResponse <: ProtoType
    __protobuf_jl_internal_meta::ProtoMeta
    __protobuf_jl_internal_values::Dict{Symbol,Any}
    __protobuf_jl_internal_defaultset::Set{Symbol}

    function UnpackResponse(; kwargs...)
        obj = new(meta(UnpackResponse), Dict{Symbol,Any}(), Set{Symbol}())
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
end # mutable struct UnpackResponse
const __meta_UnpackResponse = Ref{ProtoMeta}()
function meta(::Type{UnpackResponse})
    ProtoBuf.metalock() do
        if !isassigned(__meta_UnpackResponse)
            __meta_UnpackResponse[] = target = ProtoMeta(UnpackResponse)
            allflds = Pair{Symbol,Union{Type,String}}[:tied_data => Base.Vector{GlobalDataHandle}]
            meta(target, UnpackResponse, allflds, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES)
        end
        __meta_UnpackResponse[]
    end
end
function Base.getproperty(obj::UnpackResponse, name::Symbol)
    if name === :tied_data
        return (obj.__protobuf_jl_internal_values[name])::Base.Vector{GlobalDataHandle}
    else
        getfield(obj, name)
    end
end

export DebugOptions_StepMarkerLocation, DebugOptions_XlaBackendExtraOptionsEntry, DebugOptions, ExecutionOptions, GetDeviceHandlesRequest, GetDeviceHandlesResponse, TransferToClientRequest, TransferToClientResponse, TransferToServerRequest, TransferToServerResponse, TransferToInfeedRequest, TransferToInfeedResponse, TransferFromOutfeedRequest, TransferFromOutfeedResponse, ResetDeviceRequest, ResetDeviceResponse, ComputationGraphStatsRequest, ComputationStatsResponse, CreateChannelHandleRequest, CreateChannelHandleResponse, UnregisterRequest, UnregisterResponse, CompileRequest, CompileResponse, ExecuteRequest, ExecuteGraphRequest, ExecuteGraphParallelRequest, ExecuteResponse, ExecuteParallelResponse, WaitForExecutionRequest, WaitForExecutionResponse, ComputeConstantGraphRequest, ComputeConstantResponse, DeconstructTupleRequest, DeconstructTupleResponse, LoadDataRequest, LoadDataResponse, GetShapeRequest, GetShapeResponse, UnpackRequest, UnpackResponse
# mapentries: "DebugOptions_XlaBackendExtraOptionsEntry" => ("AbstractString", "AbstractString")
