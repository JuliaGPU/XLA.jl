# syntax: proto3
using ProtoBuf
import ProtoBuf.meta

const Kind = (;[
    Symbol("UNDEFINED_ALIAS") => Int32(0),
    Symbol("MAY_ALIAS") => Int32(1),
    Symbol("MUST_ALIAS") => Int32(2),
]...)

mutable struct HloInstructionProto_SliceDimensions <: ProtoType
    __protobuf_jl_internal_meta::ProtoMeta
    __protobuf_jl_internal_values::Dict{Symbol,Any}
    __protobuf_jl_internal_defaultset::Set{Symbol}

    function HloInstructionProto_SliceDimensions(; kwargs...)
        obj = new(meta(HloInstructionProto_SliceDimensions), Dict{Symbol,Any}(), Set{Symbol}())
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
end # mutable struct HloInstructionProto_SliceDimensions
const __meta_HloInstructionProto_SliceDimensions = Ref{ProtoMeta}()
function meta(::Type{HloInstructionProto_SliceDimensions})
    ProtoBuf.metalock() do
        if !isassigned(__meta_HloInstructionProto_SliceDimensions)
            __meta_HloInstructionProto_SliceDimensions[] = target = ProtoMeta(HloInstructionProto_SliceDimensions)
            allflds = Pair{Symbol,Union{Type,String}}[:start => Int64, :limit => Int64, :stride => Int64]
            meta(target, HloInstructionProto_SliceDimensions, allflds, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES)
        end
        __meta_HloInstructionProto_SliceDimensions[]
    end
end
function Base.getproperty(obj::HloInstructionProto_SliceDimensions, name::Symbol)
    if name === :start
        return (obj.__protobuf_jl_internal_values[name])::Int64
    elseif name === :limit
        return (obj.__protobuf_jl_internal_values[name])::Int64
    elseif name === :stride
        return (obj.__protobuf_jl_internal_values[name])::Int64
    else
        getfield(obj, name)
    end
end

mutable struct HloInstructionProto <: ProtoType
    __protobuf_jl_internal_meta::ProtoMeta
    __protobuf_jl_internal_values::Dict{Symbol,Any}
    __protobuf_jl_internal_defaultset::Set{Symbol}

    function HloInstructionProto(; kwargs...)
        obj = new(meta(HloInstructionProto), Dict{Symbol,Any}(), Set{Symbol}())
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
end # mutable struct HloInstructionProto
const __meta_HloInstructionProto = Ref{ProtoMeta}()
function meta(::Type{HloInstructionProto})
    ProtoBuf.metalock() do
        if !isassigned(__meta_HloInstructionProto)
            __meta_HloInstructionProto[] = target = ProtoMeta(HloInstructionProto)
            fnum = Int[1,2,3,7,8,9,11,13,14,15,16,50,58,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,63,33,34,41,42,35,36,37,38,40,43,49,45,71,47,60,48,51,52,54,55,56,57,59,62,61,64,65,74,66,67,68,69,70,72,73,75]
            pack = Symbol[:dimensions,:dynamic_slice_sizes,:fft_length,:gather_slice_sizes,:operand_ids,:control_predecessor_ids,:called_computation_ids,:outer_dimension_partitions]
            allflds = Pair{Symbol,Union{Type,String}}[:name => AbstractString, :opcode => AbstractString, :shape => ShapeProto, :metadata => OpMetadata, :literal => LiteralProto, :parameter_number => Int64, :fusion_kind => AbstractString, :tuple_index => Int64, :dimensions => Base.Vector{Int64}, :window => Window, :convolution_dimension_numbers => ConvolutionDimensionNumbers, :feature_group_count => Int64, :batch_group_count => Int64, :slice_dimensions => Base.Vector{HloInstructionProto_SliceDimensions}, :exponent_bits => Int32, :mantissa_bits => Int32, :dynamic_slice_sizes => Base.Vector{Int64}, :padding_config => PaddingConfig, :outfeed_config => Vector{UInt8}, :distribution => Int32, :epsilon => Float32, :feature_index => Int64, :channel_id => Int64, :infeed_config => Vector{UInt8}, :custom_call_target => AbstractString, :outfeed_shape => ShapeProto, :dot_dimension_numbers => DotDimensionNumbers, :fft_type => Int32, :fft_length => Base.Vector{Int64}, :comparison_direction => AbstractString, :gather_dimension_numbers => GatherDimensionNumbers, :gather_slice_sizes => Base.Vector{Int64}, :channel_name => AbstractString, :cost_estimate_ns => Int64, :id => Int64, :operand_ids => Base.Vector{Int64}, :control_predecessor_ids => Base.Vector{Int64}, :called_computation_ids => Base.Vector{Int64}, :sharding => OpSharding, :backend_config => Vector{UInt8}, :replica_groups => Base.Vector{ReplicaGroup}, :all_reduce_id => Int64, :use_global_device_ids => Bool, :is_host_transfer => Bool, :is_stable => Bool, :scatter_dimension_numbers => ScatterDimensionNumbers, :precision_config => PrecisionConfig, :source_target_pairs => Base.Vector{SourceTarget}, :domain_entry_sharding => OpSharding, :domain_exit_sharding => OpSharding, :constrain_layout => Bool, :operand_shapes_with_layout => Base.Vector{ShapeProto}, :triangular_solve_options => TriangularSolveOptions, :cholesky_options => CholeskyOptions, :parameter_replication => ParameterReplication, :outer_dimension_partitions => Base.Vector{Int64}, :custom_call_has_side_effect => Bool, :custom_call_output_operand_aliasing => Base.Vector{CustomCallOutputOperandAliasing}, :delta => Int64, :indices_are_sorted => Bool, :frontend_attributes => FrontendAttributes, :unique_indices => Bool, :rng_algorithm => Int32, :comparison_type => AbstractString, :is_cross_program_prefetch => Bool, :padding_type => Int32]
            meta(target, HloInstructionProto, allflds, ProtoBuf.DEF_REQ, fnum, ProtoBuf.DEF_VAL, pack, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES)
        end
        __meta_HloInstructionProto[]
    end
end
function Base.getproperty(obj::HloInstructionProto, name::Symbol)
    if name === :name
        return (obj.__protobuf_jl_internal_values[name])::AbstractString
    elseif name === :opcode
        return (obj.__protobuf_jl_internal_values[name])::AbstractString
    elseif name === :shape
        return (obj.__protobuf_jl_internal_values[name])::ShapeProto
    elseif name === :metadata
        return (obj.__protobuf_jl_internal_values[name])::OpMetadata
    elseif name === :literal
        return (obj.__protobuf_jl_internal_values[name])::LiteralProto
    elseif name === :parameter_number
        return (obj.__protobuf_jl_internal_values[name])::Int64
    elseif name === :fusion_kind
        return (obj.__protobuf_jl_internal_values[name])::AbstractString
    elseif name === :tuple_index
        return (obj.__protobuf_jl_internal_values[name])::Int64
    elseif name === :dimensions
        return (obj.__protobuf_jl_internal_values[name])::Base.Vector{Int64}
    elseif name === :window
        return (obj.__protobuf_jl_internal_values[name])::Window
    elseif name === :convolution_dimension_numbers
        return (obj.__protobuf_jl_internal_values[name])::ConvolutionDimensionNumbers
    elseif name === :feature_group_count
        return (obj.__protobuf_jl_internal_values[name])::Int64
    elseif name === :batch_group_count
        return (obj.__protobuf_jl_internal_values[name])::Int64
    elseif name === :slice_dimensions
        return (obj.__protobuf_jl_internal_values[name])::Base.Vector{HloInstructionProto_SliceDimensions}
    elseif name === :exponent_bits
        return (obj.__protobuf_jl_internal_values[name])::Int32
    elseif name === :mantissa_bits
        return (obj.__protobuf_jl_internal_values[name])::Int32
    elseif name === :dynamic_slice_sizes
        return (obj.__protobuf_jl_internal_values[name])::Base.Vector{Int64}
    elseif name === :padding_config
        return (obj.__protobuf_jl_internal_values[name])::PaddingConfig
    elseif name === :outfeed_config
        return (obj.__protobuf_jl_internal_values[name])::Vector{UInt8}
    elseif name === :distribution
        return (obj.__protobuf_jl_internal_values[name])::Int32
    elseif name === :epsilon
        return (obj.__protobuf_jl_internal_values[name])::Float32
    elseif name === :feature_index
        return (obj.__protobuf_jl_internal_values[name])::Int64
    elseif name === :channel_id
        return (obj.__protobuf_jl_internal_values[name])::Int64
    elseif name === :infeed_config
        return (obj.__protobuf_jl_internal_values[name])::Vector{UInt8}
    elseif name === :custom_call_target
        return (obj.__protobuf_jl_internal_values[name])::AbstractString
    elseif name === :outfeed_shape
        return (obj.__protobuf_jl_internal_values[name])::ShapeProto
    elseif name === :dot_dimension_numbers
        return (obj.__protobuf_jl_internal_values[name])::DotDimensionNumbers
    elseif name === :fft_type
        return (obj.__protobuf_jl_internal_values[name])::Int32
    elseif name === :fft_length
        return (obj.__protobuf_jl_internal_values[name])::Base.Vector{Int64}
    elseif name === :comparison_direction
        return (obj.__protobuf_jl_internal_values[name])::AbstractString
    elseif name === :gather_dimension_numbers
        return (obj.__protobuf_jl_internal_values[name])::GatherDimensionNumbers
    elseif name === :gather_slice_sizes
        return (obj.__protobuf_jl_internal_values[name])::Base.Vector{Int64}
    elseif name === :channel_name
        return (obj.__protobuf_jl_internal_values[name])::AbstractString
    elseif name === :cost_estimate_ns
        return (obj.__protobuf_jl_internal_values[name])::Int64
    elseif name === :id
        return (obj.__protobuf_jl_internal_values[name])::Int64
    elseif name === :operand_ids
        return (obj.__protobuf_jl_internal_values[name])::Base.Vector{Int64}
    elseif name === :control_predecessor_ids
        return (obj.__protobuf_jl_internal_values[name])::Base.Vector{Int64}
    elseif name === :called_computation_ids
        return (obj.__protobuf_jl_internal_values[name])::Base.Vector{Int64}
    elseif name === :sharding
        return (obj.__protobuf_jl_internal_values[name])::OpSharding
    elseif name === :backend_config
        return (obj.__protobuf_jl_internal_values[name])::Vector{UInt8}
    elseif name === :replica_groups
        return (obj.__protobuf_jl_internal_values[name])::Base.Vector{ReplicaGroup}
    elseif name === :all_reduce_id
        return (obj.__protobuf_jl_internal_values[name])::Int64
    elseif name === :use_global_device_ids
        return (obj.__protobuf_jl_internal_values[name])::Bool
    elseif name === :is_host_transfer
        return (obj.__protobuf_jl_internal_values[name])::Bool
    elseif name === :is_stable
        return (obj.__protobuf_jl_internal_values[name])::Bool
    elseif name === :scatter_dimension_numbers
        return (obj.__protobuf_jl_internal_values[name])::ScatterDimensionNumbers
    elseif name === :precision_config
        return (obj.__protobuf_jl_internal_values[name])::PrecisionConfig
    elseif name === :source_target_pairs
        return (obj.__protobuf_jl_internal_values[name])::Base.Vector{SourceTarget}
    elseif name === :domain_entry_sharding
        return (obj.__protobuf_jl_internal_values[name])::OpSharding
    elseif name === :domain_exit_sharding
        return (obj.__protobuf_jl_internal_values[name])::OpSharding
    elseif name === :constrain_layout
        return (obj.__protobuf_jl_internal_values[name])::Bool
    elseif name === :operand_shapes_with_layout
        return (obj.__protobuf_jl_internal_values[name])::Base.Vector{ShapeProto}
    elseif name === :triangular_solve_options
        return (obj.__protobuf_jl_internal_values[name])::TriangularSolveOptions
    elseif name === :cholesky_options
        return (obj.__protobuf_jl_internal_values[name])::CholeskyOptions
    elseif name === :parameter_replication
        return (obj.__protobuf_jl_internal_values[name])::ParameterReplication
    elseif name === :outer_dimension_partitions
        return (obj.__protobuf_jl_internal_values[name])::Base.Vector{Int64}
    elseif name === :custom_call_has_side_effect
        return (obj.__protobuf_jl_internal_values[name])::Bool
    elseif name === :custom_call_output_operand_aliasing
        return (obj.__protobuf_jl_internal_values[name])::Base.Vector{CustomCallOutputOperandAliasing}
    elseif name === :delta
        return (obj.__protobuf_jl_internal_values[name])::Int64
    elseif name === :indices_are_sorted
        return (obj.__protobuf_jl_internal_values[name])::Bool
    elseif name === :frontend_attributes
        return (obj.__protobuf_jl_internal_values[name])::FrontendAttributes
    elseif name === :unique_indices
        return (obj.__protobuf_jl_internal_values[name])::Bool
    elseif name === :rng_algorithm
        return (obj.__protobuf_jl_internal_values[name])::Int32
    elseif name === :comparison_type
        return (obj.__protobuf_jl_internal_values[name])::AbstractString
    elseif name === :is_cross_program_prefetch
        return (obj.__protobuf_jl_internal_values[name])::Bool
    elseif name === :padding_type
        return (obj.__protobuf_jl_internal_values[name])::Int32
    else
        getfield(obj, name)
    end
end

mutable struct HloComputationProto <: ProtoType
    __protobuf_jl_internal_meta::ProtoMeta
    __protobuf_jl_internal_values::Dict{Symbol,Any}
    __protobuf_jl_internal_defaultset::Set{Symbol}

    function HloComputationProto(; kwargs...)
        obj = new(meta(HloComputationProto), Dict{Symbol,Any}(), Set{Symbol}())
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
end # mutable struct HloComputationProto
const __meta_HloComputationProto = Ref{ProtoMeta}()
function meta(::Type{HloComputationProto})
    ProtoBuf.metalock() do
        if !isassigned(__meta_HloComputationProto)
            __meta_HloComputationProto[] = target = ProtoMeta(HloComputationProto)
            fnum = Int[1,2,4,5,6]
            allflds = Pair{Symbol,Union{Type,String}}[:name => AbstractString, :instructions => Base.Vector{HloInstructionProto}, :program_shape => ProgramShapeProto, :id => Int64, :root_id => Int64]
            meta(target, HloComputationProto, allflds, ProtoBuf.DEF_REQ, fnum, ProtoBuf.DEF_VAL, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES)
        end
        __meta_HloComputationProto[]
    end
end
function Base.getproperty(obj::HloComputationProto, name::Symbol)
    if name === :name
        return (obj.__protobuf_jl_internal_values[name])::AbstractString
    elseif name === :instructions
        return (obj.__protobuf_jl_internal_values[name])::Base.Vector{HloInstructionProto}
    elseif name === :program_shape
        return (obj.__protobuf_jl_internal_values[name])::ProgramShapeProto
    elseif name === :id
        return (obj.__protobuf_jl_internal_values[name])::Int64
    elseif name === :root_id
        return (obj.__protobuf_jl_internal_values[name])::Int64
    else
        getfield(obj, name)
    end
end

mutable struct HloScheduleProto_InstructionSequence <: ProtoType
    __protobuf_jl_internal_meta::ProtoMeta
    __protobuf_jl_internal_values::Dict{Symbol,Any}
    __protobuf_jl_internal_defaultset::Set{Symbol}

    function HloScheduleProto_InstructionSequence(; kwargs...)
        obj = new(meta(HloScheduleProto_InstructionSequence), Dict{Symbol,Any}(), Set{Symbol}())
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
end # mutable struct HloScheduleProto_InstructionSequence
const __meta_HloScheduleProto_InstructionSequence = Ref{ProtoMeta}()
function meta(::Type{HloScheduleProto_InstructionSequence})
    ProtoBuf.metalock() do
        if !isassigned(__meta_HloScheduleProto_InstructionSequence)
            __meta_HloScheduleProto_InstructionSequence[] = target = ProtoMeta(HloScheduleProto_InstructionSequence)
            pack = Symbol[:instruction_ids]
            allflds = Pair{Symbol,Union{Type,String}}[:instruction_ids => Base.Vector{Int64}]
            meta(target, HloScheduleProto_InstructionSequence, allflds, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, pack, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES)
        end
        __meta_HloScheduleProto_InstructionSequence[]
    end
end
function Base.getproperty(obj::HloScheduleProto_InstructionSequence, name::Symbol)
    if name === :instruction_ids
        return (obj.__protobuf_jl_internal_values[name])::Base.Vector{Int64}
    else
        getfield(obj, name)
    end
end

mutable struct HloScheduleProto_SequencesEntry <: ProtoType
    __protobuf_jl_internal_meta::ProtoMeta
    __protobuf_jl_internal_values::Dict{Symbol,Any}
    __protobuf_jl_internal_defaultset::Set{Symbol}

    function HloScheduleProto_SequencesEntry(; kwargs...)
        obj = new(meta(HloScheduleProto_SequencesEntry), Dict{Symbol,Any}(), Set{Symbol}())
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
end # mutable struct HloScheduleProto_SequencesEntry (mapentry)
const __meta_HloScheduleProto_SequencesEntry = Ref{ProtoMeta}()
function meta(::Type{HloScheduleProto_SequencesEntry})
    ProtoBuf.metalock() do
        if !isassigned(__meta_HloScheduleProto_SequencesEntry)
            __meta_HloScheduleProto_SequencesEntry[] = target = ProtoMeta(HloScheduleProto_SequencesEntry)
            allflds = Pair{Symbol,Union{Type,String}}[:key => Int64, :value => HloScheduleProto_InstructionSequence]
            meta(target, HloScheduleProto_SequencesEntry, allflds, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES)
        end
        __meta_HloScheduleProto_SequencesEntry[]
    end
end
function Base.getproperty(obj::HloScheduleProto_SequencesEntry, name::Symbol)
    if name === :key
        return (obj.__protobuf_jl_internal_values[name])::Int64
    elseif name === :value
        return (obj.__protobuf_jl_internal_values[name])::HloScheduleProto_InstructionSequence
    else
        getfield(obj, name)
    end
end

mutable struct HloScheduleProto <: ProtoType
    __protobuf_jl_internal_meta::ProtoMeta
    __protobuf_jl_internal_values::Dict{Symbol,Any}
    __protobuf_jl_internal_defaultset::Set{Symbol}

    function HloScheduleProto(; kwargs...)
        obj = new(meta(HloScheduleProto), Dict{Symbol,Any}(), Set{Symbol}())
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
end # mutable struct HloScheduleProto
const __meta_HloScheduleProto = Ref{ProtoMeta}()
function meta(::Type{HloScheduleProto})
    ProtoBuf.metalock() do
        if !isassigned(__meta_HloScheduleProto)
            __meta_HloScheduleProto[] = target = ProtoMeta(HloScheduleProto)
            allflds = Pair{Symbol,Union{Type,String}}[:sequences => Base.Dict{Int64,HloScheduleProto_InstructionSequence}]
            meta(target, HloScheduleProto, allflds, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES)
        end
        __meta_HloScheduleProto[]
    end
end
function Base.getproperty(obj::HloScheduleProto, name::Symbol)
    if name === :sequences
        return (obj.__protobuf_jl_internal_values[name])::Base.Dict{Int64,HloScheduleProto_InstructionSequence}
    else
        getfield(obj, name)
    end
end

mutable struct HloInputOutputAliasProto_AliasEntryProto <: ProtoType
    __protobuf_jl_internal_meta::ProtoMeta
    __protobuf_jl_internal_values::Dict{Symbol,Any}
    __protobuf_jl_internal_defaultset::Set{Symbol}

    function HloInputOutputAliasProto_AliasEntryProto(; kwargs...)
        obj = new(meta(HloInputOutputAliasProto_AliasEntryProto), Dict{Symbol,Any}(), Set{Symbol}())
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
end # mutable struct HloInputOutputAliasProto_AliasEntryProto
const __meta_HloInputOutputAliasProto_AliasEntryProto = Ref{ProtoMeta}()
function meta(::Type{HloInputOutputAliasProto_AliasEntryProto})
    ProtoBuf.metalock() do
        if !isassigned(__meta_HloInputOutputAliasProto_AliasEntryProto)
            __meta_HloInputOutputAliasProto_AliasEntryProto[] = target = ProtoMeta(HloInputOutputAliasProto_AliasEntryProto)
            pack = Symbol[:output_shape_index,:parameter_shape_index]
            allflds = Pair{Symbol,Union{Type,String}}[:output_shape_index => Base.Vector{Int64}, :parameter_number => Int64, :parameter_shape_index => Base.Vector{Int64}, :kind => Int32]
            meta(target, HloInputOutputAliasProto_AliasEntryProto, allflds, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, pack, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES)
        end
        __meta_HloInputOutputAliasProto_AliasEntryProto[]
    end
end
function Base.getproperty(obj::HloInputOutputAliasProto_AliasEntryProto, name::Symbol)
    if name === :output_shape_index
        return (obj.__protobuf_jl_internal_values[name])::Base.Vector{Int64}
    elseif name === :parameter_number
        return (obj.__protobuf_jl_internal_values[name])::Int64
    elseif name === :parameter_shape_index
        return (obj.__protobuf_jl_internal_values[name])::Base.Vector{Int64}
    elseif name === :kind
        return (obj.__protobuf_jl_internal_values[name])::Int32
    else
        getfield(obj, name)
    end
end

mutable struct HloInputOutputAliasProto <: ProtoType
    __protobuf_jl_internal_meta::ProtoMeta
    __protobuf_jl_internal_values::Dict{Symbol,Any}
    __protobuf_jl_internal_defaultset::Set{Symbol}

    function HloInputOutputAliasProto(; kwargs...)
        obj = new(meta(HloInputOutputAliasProto), Dict{Symbol,Any}(), Set{Symbol}())
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
end # mutable struct HloInputOutputAliasProto
const __meta_HloInputOutputAliasProto = Ref{ProtoMeta}()
function meta(::Type{HloInputOutputAliasProto})
    ProtoBuf.metalock() do
        if !isassigned(__meta_HloInputOutputAliasProto)
            __meta_HloInputOutputAliasProto[] = target = ProtoMeta(HloInputOutputAliasProto)
            allflds = Pair{Symbol,Union{Type,String}}[:entries => Base.Vector{HloInputOutputAliasProto_AliasEntryProto}]
            meta(target, HloInputOutputAliasProto, allflds, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES)
        end
        __meta_HloInputOutputAliasProto[]
    end
end
function Base.getproperty(obj::HloInputOutputAliasProto, name::Symbol)
    if name === :entries
        return (obj.__protobuf_jl_internal_values[name])::Base.Vector{HloInputOutputAliasProto_AliasEntryProto}
    else
        getfield(obj, name)
    end
end

mutable struct DynamicParameterBindingProto_Binding <: ProtoType
    __protobuf_jl_internal_meta::ProtoMeta
    __protobuf_jl_internal_values::Dict{Symbol,Any}
    __protobuf_jl_internal_defaultset::Set{Symbol}

    function DynamicParameterBindingProto_Binding(; kwargs...)
        obj = new(meta(DynamicParameterBindingProto_Binding), Dict{Symbol,Any}(), Set{Symbol}())
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
end # mutable struct DynamicParameterBindingProto_Binding
const __meta_DynamicParameterBindingProto_Binding = Ref{ProtoMeta}()
function meta(::Type{DynamicParameterBindingProto_Binding})
    ProtoBuf.metalock() do
        if !isassigned(__meta_DynamicParameterBindingProto_Binding)
            __meta_DynamicParameterBindingProto_Binding[] = target = ProtoMeta(DynamicParameterBindingProto_Binding)
            pack = Symbol[:dynamic_param_index,:target_param_index]
            allflds = Pair{Symbol,Union{Type,String}}[:dynamic_param_num => Int64, :dynamic_param_index => Base.Vector{Int64}, :target_param_num => Int64, :target_param_index => Base.Vector{Int64}, :target_param_dim_num => Int64]
            meta(target, DynamicParameterBindingProto_Binding, allflds, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, pack, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES)
        end
        __meta_DynamicParameterBindingProto_Binding[]
    end
end
function Base.getproperty(obj::DynamicParameterBindingProto_Binding, name::Symbol)
    if name === :dynamic_param_num
        return (obj.__protobuf_jl_internal_values[name])::Int64
    elseif name === :dynamic_param_index
        return (obj.__protobuf_jl_internal_values[name])::Base.Vector{Int64}
    elseif name === :target_param_num
        return (obj.__protobuf_jl_internal_values[name])::Int64
    elseif name === :target_param_index
        return (obj.__protobuf_jl_internal_values[name])::Base.Vector{Int64}
    elseif name === :target_param_dim_num
        return (obj.__protobuf_jl_internal_values[name])::Int64
    else
        getfield(obj, name)
    end
end

mutable struct DynamicParameterBindingProto <: ProtoType
    __protobuf_jl_internal_meta::ProtoMeta
    __protobuf_jl_internal_values::Dict{Symbol,Any}
    __protobuf_jl_internal_defaultset::Set{Symbol}

    function DynamicParameterBindingProto(; kwargs...)
        obj = new(meta(DynamicParameterBindingProto), Dict{Symbol,Any}(), Set{Symbol}())
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
end # mutable struct DynamicParameterBindingProto
const __meta_DynamicParameterBindingProto = Ref{ProtoMeta}()
function meta(::Type{DynamicParameterBindingProto})
    ProtoBuf.metalock() do
        if !isassigned(__meta_DynamicParameterBindingProto)
            __meta_DynamicParameterBindingProto[] = target = ProtoMeta(DynamicParameterBindingProto)
            allflds = Pair{Symbol,Union{Type,String}}[:entries => Base.Vector{DynamicParameterBindingProto_Binding}]
            meta(target, DynamicParameterBindingProto, allflds, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES)
        end
        __meta_DynamicParameterBindingProto[]
    end
end
function Base.getproperty(obj::DynamicParameterBindingProto, name::Symbol)
    if name === :entries
        return (obj.__protobuf_jl_internal_values[name])::Base.Vector{DynamicParameterBindingProto_Binding}
    else
        getfield(obj, name)
    end
end

mutable struct CrossProgramPrefetch <: ProtoType
    __protobuf_jl_internal_meta::ProtoMeta
    __protobuf_jl_internal_values::Dict{Symbol,Any}
    __protobuf_jl_internal_defaultset::Set{Symbol}

    function CrossProgramPrefetch(; kwargs...)
        obj = new(meta(CrossProgramPrefetch), Dict{Symbol,Any}(), Set{Symbol}())
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
end # mutable struct CrossProgramPrefetch
const __meta_CrossProgramPrefetch = Ref{ProtoMeta}()
function meta(::Type{CrossProgramPrefetch})
    ProtoBuf.metalock() do
        if !isassigned(__meta_CrossProgramPrefetch)
            __meta_CrossProgramPrefetch[] = target = ProtoMeta(CrossProgramPrefetch)
            pack = Symbol[:index]
            allflds = Pair{Symbol,Union{Type,String}}[:parameter => Int64, :index => Base.Vector{Int64}]
            meta(target, CrossProgramPrefetch, allflds, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, pack, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES)
        end
        __meta_CrossProgramPrefetch[]
    end
end
function Base.getproperty(obj::CrossProgramPrefetch, name::Symbol)
    if name === :parameter
        return (obj.__protobuf_jl_internal_values[name])::Int64
    elseif name === :index
        return (obj.__protobuf_jl_internal_values[name])::Base.Vector{Int64}
    else
        getfield(obj, name)
    end
end

mutable struct HloModuleProto <: ProtoType
    __protobuf_jl_internal_meta::ProtoMeta
    __protobuf_jl_internal_values::Dict{Symbol,Any}
    __protobuf_jl_internal_defaultset::Set{Symbol}

    function HloModuleProto(; kwargs...)
        obj = new(meta(HloModuleProto), Dict{Symbol,Any}(), Set{Symbol}())
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
end # mutable struct HloModuleProto
const __meta_HloModuleProto = Ref{ProtoMeta}()
function meta(::Type{HloModuleProto})
    ProtoBuf.metalock() do
        if !isassigned(__meta_HloModuleProto)
            __meta_HloModuleProto[] = target = ProtoMeta(HloModuleProto)
            fnum = Int[1,2,6,3,4,5,7,8,9,10]
            allflds = Pair{Symbol,Union{Type,String}}[:name => AbstractString, :entry_computation_name => AbstractString, :entry_computation_id => Int64, :computations => Base.Vector{HloComputationProto}, :host_program_shape => ProgramShapeProto, :id => Int64, :schedule => HloScheduleProto, :input_output_alias => HloInputOutputAliasProto, :dynamic_parameter_binding => DynamicParameterBindingProto, :cross_program_prefetches => Base.Vector{CrossProgramPrefetch}]
            meta(target, HloModuleProto, allflds, ProtoBuf.DEF_REQ, fnum, ProtoBuf.DEF_VAL, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES)
        end
        __meta_HloModuleProto[]
    end
end
function Base.getproperty(obj::HloModuleProto, name::Symbol)
    if name === :name
        return (obj.__protobuf_jl_internal_values[name])::AbstractString
    elseif name === :entry_computation_name
        return (obj.__protobuf_jl_internal_values[name])::AbstractString
    elseif name === :entry_computation_id
        return (obj.__protobuf_jl_internal_values[name])::Int64
    elseif name === :computations
        return (obj.__protobuf_jl_internal_values[name])::Base.Vector{HloComputationProto}
    elseif name === :host_program_shape
        return (obj.__protobuf_jl_internal_values[name])::ProgramShapeProto
    elseif name === :id
        return (obj.__protobuf_jl_internal_values[name])::Int64
    elseif name === :schedule
        return (obj.__protobuf_jl_internal_values[name])::HloScheduleProto
    elseif name === :input_output_alias
        return (obj.__protobuf_jl_internal_values[name])::HloInputOutputAliasProto
    elseif name === :dynamic_parameter_binding
        return (obj.__protobuf_jl_internal_values[name])::DynamicParameterBindingProto
    elseif name === :cross_program_prefetches
        return (obj.__protobuf_jl_internal_values[name])::Base.Vector{CrossProgramPrefetch}
    else
        getfield(obj, name)
    end
end

mutable struct LogicalBufferProto_Location <: ProtoType
    __protobuf_jl_internal_meta::ProtoMeta
    __protobuf_jl_internal_values::Dict{Symbol,Any}
    __protobuf_jl_internal_defaultset::Set{Symbol}

    function LogicalBufferProto_Location(; kwargs...)
        obj = new(meta(LogicalBufferProto_Location), Dict{Symbol,Any}(), Set{Symbol}())
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
end # mutable struct LogicalBufferProto_Location
const __meta_LogicalBufferProto_Location = Ref{ProtoMeta}()
function meta(::Type{LogicalBufferProto_Location})
    ProtoBuf.metalock() do
        if !isassigned(__meta_LogicalBufferProto_Location)
            __meta_LogicalBufferProto_Location[] = target = ProtoMeta(LogicalBufferProto_Location)
            pack = Symbol[:shape_index]
            allflds = Pair{Symbol,Union{Type,String}}[:computation_name => AbstractString, :instruction_name => AbstractString, :shape_index => Base.Vector{Int64}]
            meta(target, LogicalBufferProto_Location, allflds, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, pack, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES)
        end
        __meta_LogicalBufferProto_Location[]
    end
end
function Base.getproperty(obj::LogicalBufferProto_Location, name::Symbol)
    if name === :computation_name
        return (obj.__protobuf_jl_internal_values[name])::AbstractString
    elseif name === :instruction_name
        return (obj.__protobuf_jl_internal_values[name])::AbstractString
    elseif name === :shape_index
        return (obj.__protobuf_jl_internal_values[name])::Base.Vector{Int64}
    else
        getfield(obj, name)
    end
end

mutable struct LogicalBufferProto <: ProtoType
    __protobuf_jl_internal_meta::ProtoMeta
    __protobuf_jl_internal_values::Dict{Symbol,Any}
    __protobuf_jl_internal_defaultset::Set{Symbol}

    function LogicalBufferProto(; kwargs...)
        obj = new(meta(LogicalBufferProto), Dict{Symbol,Any}(), Set{Symbol}())
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
end # mutable struct LogicalBufferProto
const __meta_LogicalBufferProto = Ref{ProtoMeta}()
function meta(::Type{LogicalBufferProto})
    ProtoBuf.metalock() do
        if !isassigned(__meta_LogicalBufferProto)
            __meta_LogicalBufferProto[] = target = ProtoMeta(LogicalBufferProto)
            allflds = Pair{Symbol,Union{Type,String}}[:id => Int64, :size => Int64, :defined_at => LogicalBufferProto_Location, :color => Int64]
            meta(target, LogicalBufferProto, allflds, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES)
        end
        __meta_LogicalBufferProto[]
    end
end
function Base.getproperty(obj::LogicalBufferProto, name::Symbol)
    if name === :id
        return (obj.__protobuf_jl_internal_values[name])::Int64
    elseif name === :size
        return (obj.__protobuf_jl_internal_values[name])::Int64
    elseif name === :defined_at
        return (obj.__protobuf_jl_internal_values[name])::LogicalBufferProto_Location
    elseif name === :color
        return (obj.__protobuf_jl_internal_values[name])::Int64
    else
        getfield(obj, name)
    end
end

mutable struct BufferAllocationProto_Assigned <: ProtoType
    __protobuf_jl_internal_meta::ProtoMeta
    __protobuf_jl_internal_values::Dict{Symbol,Any}
    __protobuf_jl_internal_defaultset::Set{Symbol}

    function BufferAllocationProto_Assigned(; kwargs...)
        obj = new(meta(BufferAllocationProto_Assigned), Dict{Symbol,Any}(), Set{Symbol}())
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
end # mutable struct BufferAllocationProto_Assigned
const __meta_BufferAllocationProto_Assigned = Ref{ProtoMeta}()
function meta(::Type{BufferAllocationProto_Assigned})
    ProtoBuf.metalock() do
        if !isassigned(__meta_BufferAllocationProto_Assigned)
            __meta_BufferAllocationProto_Assigned[] = target = ProtoMeta(BufferAllocationProto_Assigned)
            allflds = Pair{Symbol,Union{Type,String}}[:logical_buffer_id => Int64, :offset => Int64, :size => Int64]
            meta(target, BufferAllocationProto_Assigned, allflds, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES)
        end
        __meta_BufferAllocationProto_Assigned[]
    end
end
function Base.getproperty(obj::BufferAllocationProto_Assigned, name::Symbol)
    if name === :logical_buffer_id
        return (obj.__protobuf_jl_internal_values[name])::Int64
    elseif name === :offset
        return (obj.__protobuf_jl_internal_values[name])::Int64
    elseif name === :size
        return (obj.__protobuf_jl_internal_values[name])::Int64
    else
        getfield(obj, name)
    end
end

mutable struct BufferAllocationProto <: ProtoType
    __protobuf_jl_internal_meta::ProtoMeta
    __protobuf_jl_internal_values::Dict{Symbol,Any}
    __protobuf_jl_internal_defaultset::Set{Symbol}

    function BufferAllocationProto(; kwargs...)
        obj = new(meta(BufferAllocationProto), Dict{Symbol,Any}(), Set{Symbol}())
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
end # mutable struct BufferAllocationProto
const __meta_BufferAllocationProto = Ref{ProtoMeta}()
function meta(::Type{BufferAllocationProto})
    ProtoBuf.metalock() do
        if !isassigned(__meta_BufferAllocationProto)
            __meta_BufferAllocationProto[] = target = ProtoMeta(BufferAllocationProto)
            fnum = Int[1,2,3,11,5,12,6,10,7,8,9]
            pack = Symbol[:parameter_shape_index]
            allflds = Pair{Symbol,Union{Type,String}}[:index => Int64, :size => Int64, :is_thread_local => Bool, :is_tuple => Bool, :is_entry_computation_parameter => Bool, :is_constant => Bool, :parameter_number => Int64, :parameter_shape_index => Base.Vector{Int64}, :maybe_live_out => Bool, :color => Int64, :assigned => Base.Vector{BufferAllocationProto_Assigned}]
            meta(target, BufferAllocationProto, allflds, ProtoBuf.DEF_REQ, fnum, ProtoBuf.DEF_VAL, pack, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES)
        end
        __meta_BufferAllocationProto[]
    end
end
function Base.getproperty(obj::BufferAllocationProto, name::Symbol)
    if name === :index
        return (obj.__protobuf_jl_internal_values[name])::Int64
    elseif name === :size
        return (obj.__protobuf_jl_internal_values[name])::Int64
    elseif name === :is_thread_local
        return (obj.__protobuf_jl_internal_values[name])::Bool
    elseif name === :is_tuple
        return (obj.__protobuf_jl_internal_values[name])::Bool
    elseif name === :is_entry_computation_parameter
        return (obj.__protobuf_jl_internal_values[name])::Bool
    elseif name === :is_constant
        return (obj.__protobuf_jl_internal_values[name])::Bool
    elseif name === :parameter_number
        return (obj.__protobuf_jl_internal_values[name])::Int64
    elseif name === :parameter_shape_index
        return (obj.__protobuf_jl_internal_values[name])::Base.Vector{Int64}
    elseif name === :maybe_live_out
        return (obj.__protobuf_jl_internal_values[name])::Bool
    elseif name === :color
        return (obj.__protobuf_jl_internal_values[name])::Int64
    elseif name === :assigned
        return (obj.__protobuf_jl_internal_values[name])::Base.Vector{BufferAllocationProto_Assigned}
    else
        getfield(obj, name)
    end
end

const HeapSimulatorTrace_Event_Kind = (;[
    Symbol("ALLOC") => Int32(0),
    Symbol("FREE") => Int32(1),
    Symbol("SHARE_WITH") => Int32(2),
]...)

mutable struct HeapSimulatorTrace_Event <: ProtoType
    __protobuf_jl_internal_meta::ProtoMeta
    __protobuf_jl_internal_values::Dict{Symbol,Any}
    __protobuf_jl_internal_defaultset::Set{Symbol}

    function HeapSimulatorTrace_Event(; kwargs...)
        obj = new(meta(HeapSimulatorTrace_Event), Dict{Symbol,Any}(), Set{Symbol}())
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
end # mutable struct HeapSimulatorTrace_Event
const __meta_HeapSimulatorTrace_Event = Ref{ProtoMeta}()
function meta(::Type{HeapSimulatorTrace_Event})
    ProtoBuf.metalock() do
        if !isassigned(__meta_HeapSimulatorTrace_Event)
            __meta_HeapSimulatorTrace_Event[] = target = ProtoMeta(HeapSimulatorTrace_Event)
            allflds = Pair{Symbol,Union{Type,String}}[:kind => Int32, :buffer_id => Int64, :computation_name => AbstractString, :instruction_name => AbstractString, :share_with_canonical_id => Int64]
            meta(target, HeapSimulatorTrace_Event, allflds, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES)
        end
        __meta_HeapSimulatorTrace_Event[]
    end
end
function Base.getproperty(obj::HeapSimulatorTrace_Event, name::Symbol)
    if name === :kind
        return (obj.__protobuf_jl_internal_values[name])::Int32
    elseif name === :buffer_id
        return (obj.__protobuf_jl_internal_values[name])::Int64
    elseif name === :computation_name
        return (obj.__protobuf_jl_internal_values[name])::AbstractString
    elseif name === :instruction_name
        return (obj.__protobuf_jl_internal_values[name])::AbstractString
    elseif name === :share_with_canonical_id
        return (obj.__protobuf_jl_internal_values[name])::Int64
    else
        getfield(obj, name)
    end
end

mutable struct HeapSimulatorTrace <: ProtoType
    __protobuf_jl_internal_meta::ProtoMeta
    __protobuf_jl_internal_values::Dict{Symbol,Any}
    __protobuf_jl_internal_defaultset::Set{Symbol}

    function HeapSimulatorTrace(; kwargs...)
        obj = new(meta(HeapSimulatorTrace), Dict{Symbol,Any}(), Set{Symbol}())
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
end # mutable struct HeapSimulatorTrace
const __meta_HeapSimulatorTrace = Ref{ProtoMeta}()
function meta(::Type{HeapSimulatorTrace})
    ProtoBuf.metalock() do
        if !isassigned(__meta_HeapSimulatorTrace)
            __meta_HeapSimulatorTrace[] = target = ProtoMeta(HeapSimulatorTrace)
            allflds = Pair{Symbol,Union{Type,String}}[:events => Base.Vector{HeapSimulatorTrace_Event}, :whole_module_simulation => Bool]
            meta(target, HeapSimulatorTrace, allflds, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES)
        end
        __meta_HeapSimulatorTrace[]
    end
end
function Base.getproperty(obj::HeapSimulatorTrace, name::Symbol)
    if name === :events
        return (obj.__protobuf_jl_internal_values[name])::Base.Vector{HeapSimulatorTrace_Event}
    elseif name === :whole_module_simulation
        return (obj.__protobuf_jl_internal_values[name])::Bool
    else
        getfield(obj, name)
    end
end

mutable struct HloModuleGroupProto <: ProtoType
    __protobuf_jl_internal_meta::ProtoMeta
    __protobuf_jl_internal_values::Dict{Symbol,Any}
    __protobuf_jl_internal_defaultset::Set{Symbol}

    function HloModuleGroupProto(; kwargs...)
        obj = new(meta(HloModuleGroupProto), Dict{Symbol,Any}(), Set{Symbol}())
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
end # mutable struct HloModuleGroupProto
const __meta_HloModuleGroupProto = Ref{ProtoMeta}()
function meta(::Type{HloModuleGroupProto})
    ProtoBuf.metalock() do
        if !isassigned(__meta_HloModuleGroupProto)
            __meta_HloModuleGroupProto[] = target = ProtoMeta(HloModuleGroupProto)
            allflds = Pair{Symbol,Union{Type,String}}[:name => AbstractString, :hlo_modules => Base.Vector{HloModuleProto}]
            meta(target, HloModuleGroupProto, allflds, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES)
        end
        __meta_HloModuleGroupProto[]
    end
end
function Base.getproperty(obj::HloModuleGroupProto, name::Symbol)
    if name === :name
        return (obj.__protobuf_jl_internal_values[name])::AbstractString
    elseif name === :hlo_modules
        return (obj.__protobuf_jl_internal_values[name])::Base.Vector{HloModuleProto}
    else
        getfield(obj, name)
    end
end

mutable struct BufferAssignmentProto_BufferAlias <: ProtoType
    __protobuf_jl_internal_meta::ProtoMeta
    __protobuf_jl_internal_values::Dict{Symbol,Any}
    __protobuf_jl_internal_defaultset::Set{Symbol}

    function BufferAssignmentProto_BufferAlias(; kwargs...)
        obj = new(meta(BufferAssignmentProto_BufferAlias), Dict{Symbol,Any}(), Set{Symbol}())
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
end # mutable struct BufferAssignmentProto_BufferAlias
const __meta_BufferAssignmentProto_BufferAlias = Ref{ProtoMeta}()
function meta(::Type{BufferAssignmentProto_BufferAlias})
    ProtoBuf.metalock() do
        if !isassigned(__meta_BufferAssignmentProto_BufferAlias)
            __meta_BufferAssignmentProto_BufferAlias[] = target = ProtoMeta(BufferAssignmentProto_BufferAlias)
            allflds = Pair{Symbol,Union{Type,String}}[:source_buffer_id => Int64, :location => LogicalBufferProto_Location]
            meta(target, BufferAssignmentProto_BufferAlias, allflds, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES)
        end
        __meta_BufferAssignmentProto_BufferAlias[]
    end
end
function Base.getproperty(obj::BufferAssignmentProto_BufferAlias, name::Symbol)
    if name === :source_buffer_id
        return (obj.__protobuf_jl_internal_values[name])::Int64
    elseif name === :location
        return (obj.__protobuf_jl_internal_values[name])::LogicalBufferProto_Location
    else
        getfield(obj, name)
    end
end

mutable struct BufferAssignmentProto <: ProtoType
    __protobuf_jl_internal_meta::ProtoMeta
    __protobuf_jl_internal_values::Dict{Symbol,Any}
    __protobuf_jl_internal_defaultset::Set{Symbol}

    function BufferAssignmentProto(; kwargs...)
        obj = new(meta(BufferAssignmentProto), Dict{Symbol,Any}(), Set{Symbol}())
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
end # mutable struct BufferAssignmentProto
const __meta_BufferAssignmentProto = Ref{ProtoMeta}()
function meta(::Type{BufferAssignmentProto})
    ProtoBuf.metalock() do
        if !isassigned(__meta_BufferAssignmentProto)
            __meta_BufferAssignmentProto[] = target = ProtoMeta(BufferAssignmentProto)
            allflds = Pair{Symbol,Union{Type,String}}[:logical_buffers => Base.Vector{LogicalBufferProto}, :buffer_aliases => Base.Vector{BufferAssignmentProto_BufferAlias}, :buffer_allocations => Base.Vector{BufferAllocationProto}, :heap_simulator_traces => Base.Vector{HeapSimulatorTrace}]
            meta(target, BufferAssignmentProto, allflds, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES)
        end
        __meta_BufferAssignmentProto[]
    end
end
function Base.getproperty(obj::BufferAssignmentProto, name::Symbol)
    if name === :logical_buffers
        return (obj.__protobuf_jl_internal_values[name])::Base.Vector{LogicalBufferProto}
    elseif name === :buffer_aliases
        return (obj.__protobuf_jl_internal_values[name])::Base.Vector{BufferAssignmentProto_BufferAlias}
    elseif name === :buffer_allocations
        return (obj.__protobuf_jl_internal_values[name])::Base.Vector{BufferAllocationProto}
    elseif name === :heap_simulator_traces
        return (obj.__protobuf_jl_internal_values[name])::Base.Vector{HeapSimulatorTrace}
    else
        getfield(obj, name)
    end
end

mutable struct HloProto <: ProtoType
    __protobuf_jl_internal_meta::ProtoMeta
    __protobuf_jl_internal_values::Dict{Symbol,Any}
    __protobuf_jl_internal_defaultset::Set{Symbol}

    function HloProto(; kwargs...)
        obj = new(meta(HloProto), Dict{Symbol,Any}(), Set{Symbol}())
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
end # mutable struct HloProto
const __meta_HloProto = Ref{ProtoMeta}()
function meta(::Type{HloProto})
    ProtoBuf.metalock() do
        if !isassigned(__meta_HloProto)
            __meta_HloProto[] = target = ProtoMeta(HloProto)
            fnum = Int[1,3]
            allflds = Pair{Symbol,Union{Type,String}}[:hlo_module => HloModuleProto, :buffer_assignment => BufferAssignmentProto]
            meta(target, HloProto, allflds, ProtoBuf.DEF_REQ, fnum, ProtoBuf.DEF_VAL, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES)
        end
        __meta_HloProto[]
    end
end
function Base.getproperty(obj::HloProto, name::Symbol)
    if name === :hlo_module
        return (obj.__protobuf_jl_internal_values[name])::HloModuleProto
    elseif name === :buffer_assignment
        return (obj.__protobuf_jl_internal_values[name])::BufferAssignmentProto
    else
        getfield(obj, name)
    end
end

mutable struct HloSnapshot <: ProtoType
    __protobuf_jl_internal_meta::ProtoMeta
    __protobuf_jl_internal_values::Dict{Symbol,Any}
    __protobuf_jl_internal_defaultset::Set{Symbol}

    function HloSnapshot(; kwargs...)
        obj = new(meta(HloSnapshot), Dict{Symbol,Any}(), Set{Symbol}())
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
end # mutable struct HloSnapshot
const __meta_HloSnapshot = Ref{ProtoMeta}()
function meta(::Type{HloSnapshot})
    ProtoBuf.metalock() do
        if !isassigned(__meta_HloSnapshot)
            __meta_HloSnapshot[] = target = ProtoMeta(HloSnapshot)
            allflds = Pair{Symbol,Union{Type,String}}[:hlo => HloProto, :arguments => Base.Vector{LiteralProto}, :result => LiteralProto, :execution_platform => AbstractString]
            meta(target, HloSnapshot, allflds, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES)
        end
        __meta_HloSnapshot[]
    end
end
function Base.getproperty(obj::HloSnapshot, name::Symbol)
    if name === :hlo
        return (obj.__protobuf_jl_internal_values[name])::HloProto
    elseif name === :arguments
        return (obj.__protobuf_jl_internal_values[name])::Base.Vector{LiteralProto}
    elseif name === :result
        return (obj.__protobuf_jl_internal_values[name])::LiteralProto
    elseif name === :execution_platform
        return (obj.__protobuf_jl_internal_values[name])::AbstractString
    else
        getfield(obj, name)
    end
end

mutable struct HloPassMetadata <: ProtoType
    __protobuf_jl_internal_meta::ProtoMeta
    __protobuf_jl_internal_values::Dict{Symbol,Any}
    __protobuf_jl_internal_defaultset::Set{Symbol}

    function HloPassMetadata(; kwargs...)
        obj = new(meta(HloPassMetadata), Dict{Symbol,Any}(), Set{Symbol}())
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
end # mutable struct HloPassMetadata
const __meta_HloPassMetadata = Ref{ProtoMeta}()
function meta(::Type{HloPassMetadata})
    ProtoBuf.metalock() do
        if !isassigned(__meta_HloPassMetadata)
            __meta_HloPassMetadata[] = target = ProtoMeta(HloPassMetadata)
            pack = Symbol[:module_group_module_ids]
            allflds = Pair{Symbol,Union{Type,String}}[:pass_id => Int64, :pass_name => AbstractString, :pipeline_name => AbstractString, :dump_filenames => Base.Vector{AbstractString}, :module_changed => Bool, :module_id => Int64, :module_group_module_ids => Base.Vector{Int64}, :start_timestamp_usec => Int64, :end_timestamp_usec => Int64]
            meta(target, HloPassMetadata, allflds, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, pack, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES)
        end
        __meta_HloPassMetadata[]
    end
end
function Base.getproperty(obj::HloPassMetadata, name::Symbol)
    if name === :pass_id
        return (obj.__protobuf_jl_internal_values[name])::Int64
    elseif name === :pass_name
        return (obj.__protobuf_jl_internal_values[name])::AbstractString
    elseif name === :pipeline_name
        return (obj.__protobuf_jl_internal_values[name])::AbstractString
    elseif name === :dump_filenames
        return (obj.__protobuf_jl_internal_values[name])::Base.Vector{AbstractString}
    elseif name === :module_changed
        return (obj.__protobuf_jl_internal_values[name])::Bool
    elseif name === :module_id
        return (obj.__protobuf_jl_internal_values[name])::Int64
    elseif name === :module_group_module_ids
        return (obj.__protobuf_jl_internal_values[name])::Base.Vector{Int64}
    elseif name === :start_timestamp_usec
        return (obj.__protobuf_jl_internal_values[name])::Int64
    elseif name === :end_timestamp_usec
        return (obj.__protobuf_jl_internal_values[name])::Int64
    else
        getfield(obj, name)
    end
end

mutable struct HloModuleMetadataProto <: ProtoType
    __protobuf_jl_internal_meta::ProtoMeta
    __protobuf_jl_internal_values::Dict{Symbol,Any}
    __protobuf_jl_internal_defaultset::Set{Symbol}

    function HloModuleMetadataProto(; kwargs...)
        obj = new(meta(HloModuleMetadataProto), Dict{Symbol,Any}(), Set{Symbol}())
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
end # mutable struct HloModuleMetadataProto
const __meta_HloModuleMetadataProto = Ref{ProtoMeta}()
function meta(::Type{HloModuleMetadataProto})
    ProtoBuf.metalock() do
        if !isassigned(__meta_HloModuleMetadataProto)
            __meta_HloModuleMetadataProto[] = target = ProtoMeta(HloModuleMetadataProto)
            pack = Symbol[:partitioned_module_ids]
            allflds = Pair{Symbol,Union{Type,String}}[:canonical_module_id => Int64, :module_group_name => AbstractString, :original_module_id => Int64, :partitioned_module_ids => Base.Vector{Int64}, :pass_metadata => Base.Vector{HloPassMetadata}]
            meta(target, HloModuleMetadataProto, allflds, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, pack, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES)
        end
        __meta_HloModuleMetadataProto[]
    end
end
function Base.getproperty(obj::HloModuleMetadataProto, name::Symbol)
    if name === :canonical_module_id
        return (obj.__protobuf_jl_internal_values[name])::Int64
    elseif name === :module_group_name
        return (obj.__protobuf_jl_internal_values[name])::AbstractString
    elseif name === :original_module_id
        return (obj.__protobuf_jl_internal_values[name])::Int64
    elseif name === :partitioned_module_ids
        return (obj.__protobuf_jl_internal_values[name])::Base.Vector{Int64}
    elseif name === :pass_metadata
        return (obj.__protobuf_jl_internal_values[name])::Base.Vector{HloPassMetadata}
    else
        getfield(obj, name)
    end
end

export Kind, HloInstructionProto_SliceDimensions, HloInstructionProto, HloComputationProto, HloScheduleProto_InstructionSequence, HloScheduleProto_SequencesEntry, HloScheduleProto, HloInputOutputAliasProto_AliasEntryProto, HloInputOutputAliasProto, DynamicParameterBindingProto_Binding, DynamicParameterBindingProto, CrossProgramPrefetch, HloModuleProto, LogicalBufferProto_Location, LogicalBufferProto, BufferAllocationProto_Assigned, BufferAllocationProto, HeapSimulatorTrace_Event_Kind, HeapSimulatorTrace_Event, HeapSimulatorTrace, HloModuleGroupProto, BufferAssignmentProto_BufferAlias, BufferAssignmentProto, HloProto, HloSnapshot, HloModuleMetadataProto, HloPassMetadata
# mapentries: "HloScheduleProto_SequencesEntry" => ("Int64", "HloScheduleProto_InstructionSequence")
