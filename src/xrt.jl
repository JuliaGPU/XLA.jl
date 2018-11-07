function current_device(sess)
    dvs = sess.graph.op_context.devices
    isempty(dvs) ? nothing : dvs[end]
end

mutable struct XRTCompilation
    sess
    device::Union{Nothing, TensorFlow.Device}
    # XLA Program Shape
    shape::ProgramShape
    # Julia Return Type
    rt::Type
    h::Int64
    global compile
    function compile(sess, comp::XLAComputation, rt::Type)
        buf = IOBuffer()
        writeproto(buf, comp)
        local alloc
        op = as_default(sess.graph) do
            alloc = placeholder(String)
            TensorFlow.Ops.xrt_compile(alloc)
        end
        res = new(sess,
            current_device(sess),
            comp.hlo_snapshot.hlo.hlo_module.program_shape, rt,
            run(sess, op, Dict(alloc => String(take!(buf))))[1])
        finalizer(close, res)
        res
    end
end
function Base.setproperty!(c::XRTCompilation, args...)
    error("XRTCompilation may not be modified")
end

function Base.close(c::XRTCompilation)
    f() = as_default(c.sess.graph) do
        try
            run(c.sess, TensorFlow.Ops.release_compilation_handle(c.h))
        catch err
            # This runs as a finalizer. The error gets printed using the
            # C printer rather than the julia one. Transform the error
            # message to make sure it's readable
            isa(err, TensorFlow.TFException) || rethrow(err)
            error(string("TensorFlow error: ", string(err.status)))
        end
    end
    if c.device === nothing
        f()
    else
        with_device(f, c.device)
    end
    Core.setfield!(c, :h, -1)
end

const tf = TensorFlow
using Libdl
const run_func = Ref{Ptr{Cvoid}}(C_NULL)
const the_ptr = Ref{Ptr{Cvoid}}(C_NULL)
mutable struct XRTAllocation
    sess
    device::Union{Nothing, TensorFlow.Device}
    h::Int64
    function XRTAllocation(sess, literal::LiteralProto)
        buf = IOBuffer()
        writeproto(buf, XLAAllocation(
            device_ordinal = Int32(0),
            value = literal))
        local alloc
        op = as_default(sess.graph) do
            alloc = placeholder(String)
            TensorFlow.Ops.xrt_allocate(alloc)
        end
        h = run(sess, op, Dict(alloc=>String(take!(buf))))
        #ccall(:jl_, Cvoid, (Any,), "Got allocation $(h)")
        res = new(sess,
            current_device(sess),
            h)
        finalizer(close, res)
        res
    end
    global run
    global _run
    function _run(com::XRTCompilation, inputs::XRTAllocation...; config=XRTExecutionConfig())
        iob = PipeBuffer();
        writeproto(iob, config)
        str = String(take!(iob))
        local alloc
        op = as_default(com.sess.graph) do
            alloc = placeholder(String)
            # Make sure everything is on the same device
            @assert all(input->input.device == com.device, inputs)
            _op() = TensorFlow.Ops.xrt_execute(com.h, alloc, collect(map(x->x.h, inputs)))
            return com.device === nothing ? _op() : with_device(_op, com.device)
        end
        h = run(com.sess, op, Dict(alloc => str))
        #ccall(:jl_, Cvoid, (Any,), "Got allocation $(h)")
        res = @GC.preserve inputs new(com.sess, com.device, h)
        finalizer(close, res)
        res
    end

    global _run_async
    function _run_async(com::XRTCompilation, inputs::XRTAllocation...; config=XRTExecutionConfig())
        iob = PipeBuffer();
        writeproto(iob, config)
        str = String(take!(iob))
        local alloc
        op = as_default(com.sess.graph) do
            # Make sure everything is on the same device
            @assert all(input->input.device == com.device, inputs)
            _op() = TensorFlow.Ops.xrt_execute(com.h, str, collect(map(x->x.h, inputs)))
            return com.device === nothing ? _op() : with_device(_op, com.device)
        end
        h = run(com.sess, op, Dict(); async=true)
        ccall(:jl_, Cvoid, (Any,), "Got allocation $(h)")
        res = @GC.preserve inputs new(com.sess, com.device, h)
        finalizer(close, res)
        res
    end

    global run_async
    function run_async(com::XRTCompilation, inputs::XRTAllocation...; config=XRTExecutionConfig())
        res = _run_async(com, inputs...; config=config)
        T = convert(Type, XlaType(com.shape.result.element_type))
        dims = (com.shape.result.dimensions...,)
        XRTArray{T, dims, length(dims)}(res)
    end

    function run(com::XRTCompilation, inputs::XRTAllocation...; config=XRTExecutionConfig())
        res = _run(com, inputs...; config=config)
        T = convert(Type, XlaType(com.shape.result.element_type))
        dims = (com.shape.result.dimensions...,)
        XRTArray{T, dims, length(dims)}(res)
    end
end
function Base.setproperty!(c::XRTAllocation, args...)
    error("XRTAllocation may not be modified")
end

const tf = TensorFlow
global counter = 0
function Base.close(c::XRTAllocation)
    #ccall(:jl_, Cvoid, (Any,), "Start Releasing allocation $(c.h)")
    global counter
    counter += 1
    desc = tf.NodeDescription(c.sess.graph, "Const", "ReleaseAllocationHandle$(counter)/Const")
    desc["value"] = TensorFlow.RawTensor(c.h)
    desc["dtype"] = typeof(c.h)
    cc = tf.Tensor(tf.Operation(desc))
    desc2 = tf.NodeDescription(c.sess.graph, "XRTReleaseAllocationHandle", "ReleaseAllocationHandle$(counter)")
    tf.add_input(desc2, cc)
    c.device !== nothing && tf.set_device(desc2, c.device)
    #ccall(:jl_, Cvoid, (Any,), "Releasing allocation $(c.h)")
    try
        run(c.sess, tf.Tensor(tf.Operation(desc2)))
    catch err
        isa(err, TensorFlow.TFException) || rethrow(err)
        error(string("TensorFlow error: ", string(err.status)))
    end
    #ccall(:jl_, Cvoid, (Any,), "Done releasing allocation $(c.h)")
    Core.setfield!(c, :h, -1)
end

mutable struct XRTStorage{T, N}
    localstorage::Union{Array{T, N}, Nothing}
    remotestorage::Union{XRTAllocation, Nothing}
    function XRTStorage{T, N}(localstorage::Union{Array{T, N}, Nothing},
                              remotestorage::Union{XRTAllocation, Nothing}) where {T, N}
        @assert localstorage !== nothing || remotestorage !== nothing
        new{T, N}(localstorage, remotestorage)
    end
    function XRTStorage{T, N}(a::Array{T, N}) where {T, N}
        new{T, N}(a, nothing)
    end
    function XRTStorage{T, N}(a::XRTAllocation) where {T, N}
        new{T, N}(nothing, a)
    end
end

struct XRTArray{T, Dims, N} <: AbstractArray{T, N}
    storage::XRTStorage{T, N}
    @noinline function XRTArray{T, Dims, N}(a::Array{T, N}) where {T, Dims, N}
        @assert size(a) == Dims
        @assert length(Dims) == N
        new{T, Dims, N}(XRTStorage{T, N}(a))
    end
    @noinline function XRTArray{T, (), 0}(a::T) where {T<:XLAScalar}
        XRTArray{T, (), 0}(fill(a))
    end
    @noinline function XRTArray{T, (), 0}(a::T) where {T<:XRTArray}
        # Can share storage
        storage = a.storage.remotestorage !== nothing ?
            XRTStorage{T, 0}(a.storage.remotestorage) :
            XRTStorage{T, 0}(fill(a))
        new{T, (), 0}(storage)
    end
    @noinline function XRTArray{T, Dims, N}(h::XRTAllocation) where {T, Dims, N}
        @assert length(Dims) == N
        new{T, Dims, N}(XRTStorage{T, N}(h))
    end
end

function Base.convert(::Type{Array}, A::XRTArray)
    if A.storage.localstorage !== nothing
        return copy(A.storage.localstorage)
    end
    rs = A.storage.remotestorage
    sess, dev, h = rs.sess, rs.device, rs.h
    op() = as_default(sess.graph) do
        run(sess, TensorFlow.Ops.xrt_read_literal(h))::String
    end
    literal = dev === nothing ? op() : with_device(op, dev)
    A.storage.localstorage = convert(Array, readproto(IOBuffer(literal), LiteralProto()))
    return copy(A.storage.localstorage)
end

@noinline function Base.convert(::Type{T}, A::XRTArray{T, (), 0}) where {T}
    convert(Array, A)[]::T
end

function gethandle!(sess, A::XRTArray)
    if A.storage.remotestorage !== nothing
        @assert A.storage.remotestorage.sess === sess
        return A.storage.remotestorage
    end
    A.storage.remotestorage = XRTAllocation(sess, convert(LiteralProto, A.storage.localstorage))
    A.storage.remotestorage
end

const XRTMatrix{T, Dims} = XRTArray{T, Dims, 2} where {T, Dims}
const XRTVector{T, Dims} = XRTArray{T, Dims, 1} where {T, Dims}
XRTArray(A::AbstractArray) = XRTArray(collect(A)::Array)
function XRTArray(a::Array{T}) where {T}
    XRTArray{T, size(a), ndims(a)}(a)
end
function XRTArray(a::XLAScalar)
    XRTArray{typeof(a), (), 0}(a)
end
function XRTArray(a::Tuple{XLAScalar})
    XRTArray{typeof(a[1]), (), 0}(a[1])
end
function XRTArray(a::Tuple{XRTArray})
    XRTArray{typeof(a[1]), (), 0}(a[1])
end
XRTArray(sess, A::AbstractArray) = XRTArray(sess, collect(A)::Array)
XRTArray(sess, a::XLAScalar) = XRTArray(sess, fill(a))
function XRTArray(sess, a::Array{T}) where {T}
    ret = XRTArray{T, size(a), ndims(a)}(a)
    gethandle!(sess, ret)
    ret
end

Base.eltype(A::Type{<:XRTArray{T}}) where {T} = T
Base.size(A::Type{<:XRTArray{T, Dims}} where T) where {Dims} = Dims
Base.size(A::Type{<:XRTArray{T, Dims}} where T, i) where {Dims} = i <= length(Dims) ? Dims[i] : 1

Base.eltype(A::XRTArray{T}) where {T} = T
Base.size(A::XRTArray{T, Dims}) where {T, Dims} = Dims
Base.size(A::XRTArray{T, Dims}, i) where {T, Dims} = i <= length(Dims) ? Dims[i] : 1
@inline function Base.axes(A::XRTArray{<:Any, Dims}, d) where {Dims}
    d <= length(Dims) ? axes(A)[d] : Base.OneTo(1)
end

import .xla: Shape
function Shape(::Type{XRTArray{T, Dims, N}} where N) where {T, Dims}
    Shape(T, Dims)
end
Shape(T::Type, SHP::Tuple) = Shape(convert(XlaType, T), SHP)
function Shape(::Type{<:XRTArray{T, SHP1}}, SHP2::Tuple) where {T, SHP1}
    # Map arrays of arrays to higher rank arrays
    Shape(T, (SHP1..., SHP2...))
end
function Shape(XLAT::XlaType, SHP::Tuple)
    perm = sortperm(collect(SHP); rev=true)
    Shape(
        element_type = XLAT.which,
        dimensions = Int64[SHP...],
        layout = Layout(
            format = Format.DENSE,
            minor_to_major = perm .- 1,
            max_sparse_elements = 0
        )
    )
end

Base.convert(::Type{Shape}, AT::Type{XRTArray{T, Dims, N}} where N) where {T, Dims} =
    Shape(AT)

shape(A::XRTArray) = Shape(typeof(A))

Base.isempty(A::XRTArray{T, Dims}) where {T, Dims} = prod(Dims) == 0

Base.print_array(io::IO, A::XRTArray) = Base.print_array(io, convert(Array, A))
Base.show_vector(io::IO, A::XRTArray, opn='[', cls=']') =
    Base.show_vector(io, convert(Array, A), opn, cls)
Base._show_nonempty(io::IO, A::XRTArray, prefix::String) =
    Base._show_nonempty(io, convert(Array, A), prefix)
Base._show_nonempty(io::IO, A::XRTMatrix, prefix::String) =
    Base._show_nonempty(io, convert(Array, A), prefix)
