abstract type HloOp{Opcode, T, Shape}; end
opcode(op::HloOp{Opcode}) where {Opcode} = String(Opcode)
shape(op::HloOp{<:Any, T, SHP}) where {T, SHP} = Shape(
        element_type = convert(XlaType, T).which,
        dimensions = [SHP...],
        layout = Layout(
            format = Format.DENSE,
            minor_to_major=collect(0:(length(SHP)-1)),
            max_sparse_elements = 0
        )
    )

struct HloParameter{T, Shape} <: HloOp{:parameter, T, Shape}
    id::Int64
end
fill_fields!(proto::HloInstructionProto, p::HloParameter) = proto.parameter_number = p.id

struct GenericHloOp{Opcode, T, Shape} <: HloOp{Opcode, T, Shape}; end
fill_fields!(proto::HloInstructionProto, gen::GenericHloOp) = nothing

struct HloConstant{T, Shape} <: HloOp{:constant, T, Shape}
    value::LiteralProto
end
HloConstant(x::XLAScalar) = HloConstant{typeof(x), ()}(convert(LiteralProto, x))
HloConstant(a::Array{<:XLAScalar}) = HloConstant{eltype(a), size(a)}(convert(LiteralProto, a))
fill_fields!(proto::HloInstructionProto, c::HloConstant) = proto.literal = c.value

# Equivalent to DotDimensionNumbers, but in a format the compiler has an
# easier time understanding
struct DimNums{N1, N2, N3, N4}
    lhs_contracting_dimensions::NTuple{N1, Int64}
    rhs_contracting_dimensions::NTuple{N2, Int64}
    lhs_batch_dimensions::NTuple{N3, Int64}
    rhs_batch_dimensions::NTuple{N4, Int64}
end

function Base.convert(::Type{DotDimensionNumbers}, dds::DimNums)
    DotDimensionNumbers(
        lhs_contracting_dimensions = collect(dds.lhs_contracting_dimensions),
        rhs_contracting_dimensions = collect(dds.rhs_contracting_dimensions),
        lhs_batch_dimensions = collect(dds.lhs_batch_dimensions),
        rhs_batch_dimensions = collect(dds.rhs_batch_dimensions),
    )
end

struct HloDot{T, Shape} <: HloOp{:dot, T, Shape}
    dims::DimNums
end
fill_fields!(proto::HloInstructionProto, d::HloDot) = proto.dot_dimension_numbers = convert(DotDimensionNumbers, d.dims)

const SliceDimensions = HloInstructionProto_SliceDimensions
struct HloSlice{T, Shape} <: HloOp{:slice, T, Shape}
    slice_dimensions::Vector{SliceDimensions}
end
fill_fields!(proto::HloInstructionProto, s::HloSlice) = proto.slice_dimensions = s.slice_dimensions

# This opcode isn't really real - there's very little point to a map
# unless we're able to compile it
struct HloMap{T, Shape} <: HloOp{:map, T, Shape}
    f
end
fill_fields!(proto::HloInstructionProto, s::HloMap) = nothing
function (m::HloMap)(args::XRTArray...)
    XRTArray(args[1].storage.sess, map(m.f, map(x->convert(Array, x), args)...))::typeof(args[1])
end

struct Argument
    shape::Shape
    id::Int64
end

let global_id = 0
    global make_id
    make_id() = (global_id += 1; global_id)
end

function HloInstructionProto(op::HloOp, operands::Union{Argument, HloInstructionProto}...; id=make_id(), name="$(opcode(op))$id")
    proto = HloInstructionProto(
        opcode = opcode(op),
        shape = shape(op),
        id = id,
        name = name,
        operand_ids = collect(map(x->x.id, operands))
    )
    fill_fields!(proto, op)
    proto
end

function HloInstructionProto(comp::HloComputationProto, op::HloOp, args...; id=length(comp.instructions), name=nothing)
    proto = HloInstructionProto(op, args...; id=id, name=something(name, "comp$(comp.id)_$(opcode(op))$id"))
    push!(comp.instructions, proto)
    proto
end
