mutable struct XRTCompilation
    sess
    shape::ProgramShape
    h::Int64
    global compile
    function compile(sess, comp::XLAComputation)
        buf = IOBuffer()
        writeproto(buf, comp)
        res = new(sess, comp.hlo_snapshot.hlo.hlo_module.program_shape,
            run(sess, TensorFlow.Ops.xrt_compile(String(take!(buf)))))
        finalizer(close, res)
        res
    end
end
function Base.setproperty!(c::XRTCompilation, args...)
    error("XRTCompilation may not be modified")
end

function Base.close(c::XRTCompilation)
    run(c.sess, TensorFlow.Ops.xrt_release_compilation_handle(c.h))
    Core.setfield!(c, :h, -1)
end

mutable struct XRTAllocation
    sess
    h::Int64
    function XRTAllocation(sess, literal::LiteralProto)
        buf = IOBuffer()
        writeproto(buf, XLAAllocation(
            device_ordinal = Int32(0),
            value = literal))
        res = new(sess, run(sess, TensorFlow.Ops.xrt_allocate(String(take!(buf)))))
        finalizer(close, res)
        res
    end
    global run
    function run(com::XRTCompilation, inputs::XRTAllocation...; config=XRTExecutionConfig())
        iob = PipeBuffer();
        writeproto(iob, config)
        str = String(take!(iob))

        res = new(com.sess, run(com.sess, TensorFlow.Ops.xrt_execute(com.h,
            str, collect(map(x->x.h, inputs)))))
        finalizer(close, res)
        T = convert(Type, XlaType(com.shape.result.element_type))
        dims = (com.shape.result.dimensions...,)
        XRTArray{T, dims, length(dims)}(res)
    end
end
function Base.setproperty!(c::XRTAllocation, args...)
    error("XRTAllocation may not be modified")
end

function Base.close(c::XRTAllocation)
    run(c.sess, TensorFlow.Ops.xrt_release_allocation_handle(c.h))
    Core.setfield!(c, :h, -1)
end

struct XRTArray{T, Dims, N} <: AbstractArray{T, N}
    storage::XRTAllocation
    function XRTArray{T, Dims, N}(sess, a::Array{T, N}) where {T, Dims, N}
        @assert size(a) == Dims
        @assert length(Dims) == N
        new{T, Dims, N}(XRTAllocation(sess, convert(LiteralProto, a)))
    end
    function XRTArray{T, Dims, N}(h::XRTAllocation) where {T, Dims, N}
        @assert length(Dims) == N
        new{T, Dims, N}(h)
    end
end
const XRTMatrix{T, Dims} = XRTArray{T, Dims, 2} where {T, Dims}
const XRTVector{T, Dims} = XRTArray{T, Dims, 1} where {T, Dims}
XRTArray(sess, A::AbstractArray) = XRTArray(sess, collect(A)::Array)
Base.eltype(A::XRTArray{T}) where {T} = T
Base.size(A::XRTArray{T, Dims}) where {T, Dims} = Dims
Base.size(A::XRTArray{T, Dims}, i) where {T, Dims} = Dims[i]
@inline function Base.axes(A::XRTArray{<:Any, Dims}, d) where {Dims}
    d <= length(Dims) ? axes(A)[d] : Base.OneTo(1)
end

import .xla: Shape
function Shape(::Type{XRTArray{T, Dims, N}} where N) where {T, Dims}
    Shape(
        element_type = convert(XlaType, T).which,
        dimensions = Int64[Dims...],
        layout = Layout(
            format = Format.DENSE,
            minor_to_major=collect(0:(length(Dims)-1)),
            max_sparse_elements = 0
        )
    )
end

shape(A::XRTArray) = Shape(typeof(A))

Base.isempty(A::XRTArray{T, Dims}) where {T, Dims} = prod(Dims) == 0

function XRTArray(sess, a::Array{T}) where {T}
    XRTArray{T, size(a), ndims(a)}(sess, a)
end

function Base.convert(::Type{Array}, A::XRTArray)
    literal = run(A.storage.sess, TensorFlow.Ops.xrt_read_literal(A.storage.h))::String
    convert(Array, readproto(IOBuffer(literal), LiteralProto()))
end

Base.print_array(io::IO, A::XRTArray) = Base.print_array(io, convert(Array, A))
Base.show_vector(io::IO, A::XRTArray, opn='[', cls=']') =
    Base.show_vector(io, convert(Array, A), opn, cls)
Base._show_nonempty(io::IO, A::XRTArray, prefix::String) =
    Base._show_nonempty(io, convert(Array, A), prefix)
Base._show_nonempty(io::IO, A::XRTMatrix, prefix::String) =
    Base._show_nonempty(io, convert(Array, A), prefix)
