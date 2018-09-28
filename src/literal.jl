const XLAScalar = Union{Bool, Int8, Int16, Int32, Int64,
                        UInt8, UInt16, UInt32, UInt64,
                        Float16, Float32, Float64, Complex{Float32}}

struct XlaType
    which::Int32
end

const xla_type_mapping = Dict{Int32, Type}(
    xla.PrimitiveType.PRED => Bool,
    xla.PrimitiveType.S8   => Int8,
    xla.PrimitiveType.S16  => Int16,
    xla.PrimitiveType.S32  => Int32,
    xla.PrimitiveType.S64  => Int64,
    xla.PrimitiveType.U8   => UInt8,
    xla.PrimitiveType.U16  => UInt16,
    xla.PrimitiveType.U32  => UInt32,
    xla.PrimitiveType.U64  => UInt64,
    xla.PrimitiveType.F16  => Float16,
    xla.PrimitiveType.F32  => Float32,
    xla.PrimitiveType.F64  => Float64,
    xla.PrimitiveType.C64  => Complex{Float32},
)
const reverse_xla_type_mapping = Dict(v=>k for (k,v) in xla_type_mapping)

function Base.convert(::Type{Type}, t::XlaType)
    xla_type_mapping[t.which]
end

function Base.convert(::Type{XlaType}, t::Type)
    XlaType(reverse_xla_type_mapping[t])
end

function assign_data!(lp::LiteralProto, a::Vector{<:XLAScalar})
    if eltype(a) == Bool
        lp.preds = vec(a)
    elseif eltype(a) == Int8
        lp.u8s = vec(a)
    elseif eltype(a) == Int32
        lp.s32s = vec(a)
    elseif eltype(a) == Int64
        lp.s64s = vec(a)
    elseif eltype(a) == UInt32
        lp.u32s = vec(a)
    elseif eltype(a) == UInt64
        lp.u64s = vec(a)
    elseif eltype(a) == Float16
        lp.f16s = vec(a)
    elseif eltype(a) == Float32
        lp.f32s = vec(a)
    elseif eltype(a) == Float64
        lp.f64s = vec(a)
    elseif eltype(a) == Complex{Float32}
        lp.c64s = collect(reinterpret(Float32, a))
    end
    nothing
end

function Base.convert(::Type{LiteralProto}, a::XLAScalar)
    xlat = convert(XlaType, typeof(a))
    shape = Shape(
        element_type = xlat.which,
        dimensions = Int64[],
        layout = Layout(
            format = Format.DENSE,
        )
    )
    lp = LiteralProto()
    lp.shape = shape
    assign_data!(lp, typeof(a)[a])
    return lp
end


function Base.convert(::Type{LiteralProto}, a::Array{<:XLAScalar})
    # For performance on TPUs, we need to permute the dimensions to
    # have smaller dimensions be more major
    xlat = convert(XlaType, eltype(a))
    perm = sortperm(collect(size(a)); rev=true)
    shape = Shape(
        element_type = xlat.which,
        dimensions = Int64[size(a)...],
        layout = Layout(
            format = Format.DENSE,
            minor_to_major=collect(0:(ndims(a)-1))[perm],
            max_sparse_elements = 0
        )
    )
    lp = LiteralProto()
    lp.shape = shape
    assign_data!(lp, perm == [] ? vec(a) : vec(permutedims(a, perm)))
    return lp
end

function to_data(lp::LiteralProto)
    elt = lp.shape.element_type
    if elt == xla.PrimitiveType.PRED
        return lp.preds
    elseif elt == xla.PrimitiveType.U8
        return lp.u8s
    elseif elt == xla.PrimitiveType.S32
        return lp.s32s
    elseif elt == xla.PrimitiveType.S64
        return lp.s64s
    elseif elt == xla.PrimitiveType.U32
        return lp.u32s
    elseif elt == xla.PrimitiveType.U64
        return lp.u64s
    elseif elt == xla.PrimitiveType.F16
        return lp.f16s
    elseif elt == xla.PrimitiveType.F32
        return lp.f32s
    elseif elt == xla.PrimitiveType.F64
        return lp.f64s
    elseif elt == xla.PrimitiveType.C64
        return collect(reinterpret(Compplex{Float32}, lp.c64s))
    end
end

function Base.convert(::Type{Array}, proto::LiteralProto)
    s = proto.shape
    if s.layout.minor_to_major != 0:(length(s.dimensions)-1)
        b = to_data(proto)
        perm = tuple((x+1 for x in s.layout.minor_to_major)...)
        r = reshape(b, tuple(collect(s.dimensions)[collect(perm)]...))
        @Base.show size(r)
        return convert(Array, permutedims(r, Base.invperm(perm)))
    else
        b = to_data(proto)
        return reshape(b, tuple(s.dimensions...))
    end
end
