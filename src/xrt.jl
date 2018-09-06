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
        XRTArray{T, dims}(res)
    end
end
function Base.setproperty!(c::XRTAllocation, args...)
    error("XRTAllocation may not be modified")
end

function Base.close(c::XRTAllocation)
    run(c.sess, TensorFlow.Ops.xrt_release_allocation_handle(c.h))
    Core.setfield!(c, :h, -1)
end

struct XRTArray{T, Dims}
    storage::XRTAllocation
    function XRTArray{T, Dims}(sess, a::Array) where {T, Dims}
        @assert size(a) == Dims
        new{T, Dims}(XRTAllocation(sess, convert(LiteralProto, a)))
    end
    function XRTArray{T, Dims}(h::XRTAllocation) where {T, Dims}
        new{T, Dims}(h)
    end
end
Base.eltype(A::XRTArray{T}) where {T} = T
Base.size(A::XRTArray{T, Dims}) where {T, Dims} = Dims
Base.size(A::XRTArray{T, Dims}, i) where {T, Dims} = Dims[i]
@inline function Base.axes(A::XRTArray{<:Any, Dims}, d) where {Dims}
    d <= length(Dims) ? axes(A)[d] : Base.OneTo(1)
end

function shape(A::XRTArray{T, Dims}) where {T, Dims}
    shape = Shape(
        element_type = convert(XlaType, T).which,
        dimensions = Int64[Dims...],
        layout = Layout(
            format = Format.DENSE,
            minor_to_major=collect(0:(length(Dims)-1)),
            max_sparse_elements = 0
        )
    )
end

Base.isempty(A::XRTArray{T, Dims}) where {T, Dims} = prod(Dims) == 0

function XRTArray(sess, a::Array{T}) where {T}
    XRTArray{T, size(a)}(sess, a)
end

function Base.convert(::Type{Array}, A::XRTArray)
    literal = run(A.storage.sess, TensorFlow.Ops.xrt_read_literal(A.storage.h))::String
    convert(Array, readproto(IOBuffer(literal), LiteralProto()))
end

Base.summary(io::IO, X::XRTArray) = Base.summary(io, X, size(X))

function Base.show(io::IO, ::MIME"text/plain", X::XRTArray)
    # 0) show summary before setting :compact
    summary(io, X)
    isempty(X) && return
    print(io, ":")

    # 1) compute new IOContext
    if !haskey(io, :compact) && length(axes(X, 2)) > 1
        io = IOContext(io, :compact => true)
    end
    if get(io, :limit, false) && eltype(X) === Method
        # override usual show method for Vector{Method}: don't abbreviate long lists
        io = IOContext(io, :limit => false)
    end

    if get(io, :limit, false) && displaysize(io)[1]-4 <= 0
        return print(io, " …")
    else
        println(io)
    end

    # 2) update typeinfo
    #
    # it must come after printing the summary, which can exploit :typeinfo itself
    # (e.g. views)
    # we assume this function is always called from top-level, i.e. that it's not nested
    # within another "show" method; hence we always print the summary, without
    # checking for current :typeinfo (this could be changed in the future)
    io = IOContext(io, :typeinfo => eltype(X))

    # 2) show actual content
    Base.print_array(io, convert(Array, X))
end
