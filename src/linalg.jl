function Base.:*(A::XRTArray, B::XRTArray)
    ddots = DimNums((1,), (0,), (), ())
    HloDot(ddots)(A, B)
end
function Base.:*(A::XRTArray, B::XRTVector)
    ddots = DimNums((1,), (0,), (), ())
    HloDot(ddots)(A, B)
end
Base.:*(A::XRTArray{<:Any, (), 0}, B::XRTVector) = broadcast(*, A, B)

using LinearAlgebra
function Base.:*(A::Adjoint{<:Any, <:XRTArray}, B::XRTArray)
    ddots = DimNums((1,), (0,), (), ())
    HloDot(ddots)(HloTranspose((1,0))(A.parent), B)
end

function Base.:*(A::Adjoint{<:Any, <:XRTVector}, B::XRTVector)
    ddots = DimNums((0,), (0,), (), ())
    HloDot(ddots)(A.parent, B)
end

function Base.:*(A::Transpose{<:Any, <:XRTArray}, B::XRTArray)
    ddots = DimNums((1,), (0,), (), ())
    HloDot(ddots)(HloTranspose((1,0))(A.parent), B)
end

function Base.:*(A::XRTArray, B::Transpose{<:Any, <:XRTArray})
    ddots = DimNums((1,), (0,), (), ())
    HloDot(ddots)(A, HloTranspose((1,0))(B.parent))
end

function Base.:*(A::Transpose{<:Any, <:XRTVector}, B::XRTVector)
    ddots = DimNums((0,), (0,), (), ())
    HloDot(ddots)(A.parent, B)
end

Base.convert(::Type{XRTArray}, x::XLAScalar) = XRTArray(x)

const XRTScalar{T} = XRTArray{T, (), 0}
@noinline Base.convert(::Type{Bool}, A::XRTScalar{Bool}) = convert(Array, A)[]
function Base.promote_rule(A::Type{XRTScalar{T}}, B::Type{XRTScalar{S}}) where {T<:XLAScalar, S<:XLAScalar}
    XRTArray{promote_type(T, S), (), 0}
end

# Scalar conversions - Happens via HloConvert. In the future, we may
# want to make sure to more closely match julia semantics.
Base.convert(::Type{XRTScalar{T}}, x::XRTScalar{S}) where {T,S} = GenericHloOp{:convert}(T, ())(x)

# A couple of scalar embeddings (could use casette in the future)
using Base.Meta
for (binop, xlaop) in
        ((:+, :add),
         (:-, :subtract),
         (:/, :divide),
         (:*, :multiply))
    @eval Base.$(binop)(A::XRTScalar{T},
                        B::XRTScalar{T}) where {T<:XLAScalar} =
        GenericHloOp{$(quot(xlaop))}(T, ())(A, B)
    @eval Base.$(binop)(A::XRTScalar, B::XRTScalar, args::XRTScalar{<:XLAScalar}...) =
        $(binop)(promote(A, B, args...)...)
end


# XRT Scalars are not numbers so import some functions manually
import Base: /, \, *
for f in (:/, :\, :*)
    if f != :/
        @eval ($f)(A::XRTArray{<:Any, (), 0}, B::XRTArray) = broadcast($f, A, B)
    end
    if f != :\
        @eval ($f)(A::XRTArray, B::XRTArray{<:Any, (), 0}) = broadcast($f, A, B)
    end
end


embed(x::XLAScalar) = XRTArray(x)
embed(x) = x

concat_dims(::Type{<:XRTArray{T, Sz1}}, Sz2::Tuple) where {T, Sz1} = (Sz1..., Sz2...)
concat_dims(::Type, Sz2::Tuple) where {T, Sz1} = Sz2

function Base.zeros(::Type{XRTArray{T, Sz, N}}) where {T, Sz, N}
    TT = T
    HloBroadcast(ntuple(i->i-1, ndims(T)), Sz)(XRTArray((zero(TT),)))
end
Base.zero(::Type{XRTArray{T, Sz, N}}) where {T, Sz, N} = HloBroadcast(ntuple(i->i-1, ndims(T)), Sz)(XRTArray((zero(T),)))

Base.zero(A::Type{XRTArray{T, (), 0}}) where T = XRTArray(zero(T))
Base.zero(A::XRTArray{<:Any, (), 0}) = zero(typeof(A))
Base.one(A::Type{XRTArray{T, (), 0}}) where T = XRTArray(one(T))
Base.one(A::XRTArray{<:Any, (), 0}) = one(typeof(A))
Base.max(A::XRTArray{T, (), 0}, B::XRTArray{T, (), 0}) where {T<:XLAScalar} =
    GenericHloOp{:maximum}(T, ())(A, B)
Base.exp(A::XRTArray{T, (), 0}) where {T} = GenericHloOp{:exponential}(T, ())(A)
Base.log(A::XRTArray{T, (), 0}) where {T} = GenericHloOp{:log}(T, ())(A)
Base.sqrt(A::XRTArray{T, (), 0}) where {T} = GenericHloOp{:power}(T, ())(A, XRTArray(convert(T, 0.5)))
@eval Base.isless(A::XRTArray{<:Any, (), 0}, B::XRTArray{<:Any, (), 0}) =
    convert(Bool, GenericHloOp{$(QuoteNode(Symbol("less-than")))}(Bool, ())(A, B))
@eval Base.:<=(A::XRTArray{<:Any, (), 0}, B::XRTArray{<:Any, (), 0}) =
    GenericHloOp{$(QuoteNode(Symbol("less-than-or-equal-to")))}(Bool, ())(A, B)
Base.inv(x::XRTScalar{T}) where {T} = XRTArray(one(T))/x

Base.:-(x::XRTScalar{T}) where {T} = -(XRTArray(zero(T)), x)

# XRTArrays are immutable. Copy is identity.
Base.copy(A::XRTArray) = A
#Base.transpose(A::XRTArray) = HloTranspose((1,0))(A)
#Base.adjoint(A::XRTArray{<:Real}) = HloTranspose((1,0))(A)
Base.permutedims(A::XRTArray, perm) = HloTranspose(map(x->x-1, perm))(A)


# Recursive version of promote shape
# TODO: This shouldn't be necessary if inference did better
# purity/nothrow modeling.
check_shape_match(a::Tuple{}, b::Tuple{}) = true
function check_shape_match(a::Tuple, b::Tuple{})
    first(a) == 1:1 || return false
    check_shape_match(Base.tail(a), ())
end
function check_shape_match(a::Tuple, b::Tuple)
    first(a) == first(b) || return false
    check_shape_match(Base.tail(a), Base.tail(b))
end
function promote_shape_rec(a, b)
    if length(a) < length(b)
        return promote_shape(b, a)
    end
    check_shape_match(a, b) || error("Dimensions must match")
    a
end

function Base.promote_shape(a::XRTArray, b::XRTArray)
    promote_shape_rec(axes(a), axes(b))
end

import Base.Broadcast

Base.similar(x::XRTArray) = error()

struct XRTArrayStyle{N} <: Broadcast.AbstractArrayStyle{N} end
(::Type{<:XRTArrayStyle})(::Val{N}) where {N} = XRTArrayStyle{N}()
Broadcast.BroadcastStyle(::Type{<:XRTArray{<:Any,Dims,N}}) where {Dims, N} =
    XRTArrayStyle{N}()

@noinline _ccdims(s) = tuple(findall(==(1), s)...)
@noinline _ncdims(s) = tuple(findall(x->x!=(1), s)...)

@Base.pure ccdims(s) = _ccdims(s)
@Base.pure ncdims(s) = _ncdims(s)

@Base.pure getindex_tuple(s::Tuple, k::Tuple) = s[collect(k)]

@inline function broadcast_to_size(arg, rsize)
    if size(arg) != rsize
        collapse_dims = ccdims(size(arg))
        non_collapse_dims = ncdims(size(arg))
        if !isa(arg, XRTArray)
            arg = convert(XRTArray, arg)
        end
        if collapse_dims != ()
            arg = HloReshape(
                getindex_tuple(size(arg), non_collapse_dims)
            )(arg)
        end
        return HloBroadcast(map(x->x - 1, non_collapse_dims), rsize)(
            arg
        )
    else
        return arg
    end
end

# TODO: This should probably assert that the sessions are equivalent
any_sess(a::XRTArray, args::AnyXLA...) = a.storage.sess
any_sess(a::XLAScalar, args::AnyXLA...) = any_sess(args...)

function Broadcast.copy(bc::Broadcast.Broadcasted{<:XRTArrayStyle})
    ElType = Broadcast.combine_eltypes(bc.f, bc.args)
    bc′ = Broadcast.flatten(bc)
    if Base.isconcretetype(ElType)
        args = bc′.args
        # This could be axes(bc′) if we had better constant prop
        rsize = map(length, Broadcast.combine_axes(bc′.args...))
        args = map(arg->convert(XRTArray, broadcast_to_size(arg, rsize)), bc′.args)
        return HloMap{typeof(bc′.f)}()(bc′.f, args...)
    end
    # TODO: Pull back CPU, do this there
    error("No hope")
end

using NNlib

@Base.pure function conv_windows(sz_, pad_, stride_, dilation_)
    ntuple(length(pad_)) do i
        (sz, p, s, d) = sz_[i], pad_[i], stride_[i], dilation_[i]
        WindowDims(sz, s, p, p, d, 1, true)
    end
end

function (::NNlib._conv{pad, stride, dilation})(input::XRTArray{T}, kernel::XRTArray{T}) where {T, pad, stride, dilation}
    pad_, stride_ = NNlib.padtuple(input, pad), NNlib.padtuple(input, stride)
    dilation_ = NNlib.padtuple(kernel, dilation)
    sz_ = size(kernel)
    windows = conv_windows(sz_, pad_, stride_, dilation_)
    convdims = ConvDimNums(
        3, 2, (0, 1),
        2, 3, (0, 1),
        3, 2, (0, 1)
    )
    HloConv(windows, convdims)(input, kernel)
end

function (::NNlib._∇conv_data{pad, stride, dilation, 0})(dy::XRTArray{T}, input::XRTArray{T}, kernel::XRTArray) where {T, pad, stride, dilation}
    pad_, stride_ = NNlib.padtuple(input, pad), NNlib.padtuple(input, stride)
    dilation_ = NNlib.padtuple(kernel, dilation)
    mirrored = kernel #permutedims(kernel, (2, 1, 3, 4)) #HloRev((0,1))(kernel)
    sz_ = size(kernel); isz = size(input); osz = size(dy)
    windows = ntuple(length(pad_)) do i
        (sz, p, s, d) = sz_[i], pad_[i], stride_[i], dilation_[i]
        @assert s == 1
        @assert d == 1
        padded_out_size = isz[i] + sz - 1
        pad_before = sz - 1 - p
        pad_after = padded_out_size - osz[i] - pad_before
        # N.B.: The window reversal flag is flipped here
        WindowDims(sz, s, pad_before, pad_after, d, 1, false)
    end
    convdims = ConvDimNums(
        3, 2, (0, 1),
        # N.B. The input and output dimensions are exchanged here from the
        # standard notion of convolution
        3, 2, (0, 1),
        3, 2, (0, 1)
    )
    HloConv(windows, convdims)(dy, mirrored)
end

function (::NNlib._∇conv_filter{pad, stride, dilation, 0})(dy::XRTArray{T}, input::XRTArray{T}, kernel::XRTArray) where {T, pad, stride, dilation}
    # TODO: Validate that this is correct. Unfortunately, the NNlib
    # implementation itself is broken, so we need to fix that first
    pad_, stride_ = NNlib.padtuple(input, pad), NNlib.padtuple(input, stride)
    dilation_ = NNlib.padtuple(kernel, dilation)
    sz_ = size(kernel); isz = size(input); osz = size(dy)
    input = HloRev((0,1))(input)
    windows = ntuple(length(pad_)) do i
        (sz, p, s, d) = sz_[i], pad_[i], stride_[i], dilation_[i]
        @assert s == 1
        @assert d == 1
        padded_in_size = osz[i] + sz - 1
        pad_total = padded_in_size - isz[i]
        pad_before = max(div(pad_total, 2), 0)
        pad_after = pad_total - pad_before
        WindowDims(osz[i], s, pad_before, pad_after, d, 1, false)
    end
    convdims = ConvDimNums(
        # N.B. The input and output dimensions are exchanged here from the
        # standard notion of convolution
        2, 3, (0, 1),
        3, 2, (0, 1),
        2, 3, (0, 1)
    )
    r = HloConv(windows, convdims)(input, dy)
    @assert size(r) == size(kernel)
    r
end


function make_pooling_windows(x, k, pad, stride)
    k_, pad_, stride_ = NNlib.padtuple(x, k),
                        NNlib.padtuple(x, pad),
                        NNlib.padtuple(x, stride)
    windows = ntuple(ndims(x)) do i
        (sz, p, s) = i <= length(k) ? (k[i], pad[i], stride[i]) : (1, 0, 1)
        WindowDims(sz, s, p, p, 1, 1, false)
    end
end

function NNlib.maxpool(x::XRTArray, k; pad = map(_->0,k), stride = k)
    HloReduceWindow{typeof(max)}(
        make_pooling_windows(x, k, pad, stride)
    )(max, x, XRTArray(typemin(eltype(x))))
end

function NNlib.meanpool(x::XRTArray, k; pad = map(_->0,k), stride = k)
    HloReduceWindow{typeof(+)}(
        make_pooling_windows(x, k, pad, stride)
    )(+, x, XRTArray(zero(eltype(x))))
end

function NNlib.∇maxpool(dy::XRTArray, y::XRTArray, x::XRTArray, k; pad = map(_->0,k), stride = k)
    HloSelectAndScatter{typeof(>=), typeof(+)}(
        make_pooling_windows(x, k, pad, stride)
    )(>=, +, x, dy, XRTArray(zero(eltype(x))))
end

function NNlib.softmax(xs::XRTArray)
    ys = xs .- maximum(xs)
    exp.(ys) ./ sum(exp.(ys))
end
function NNlib.∇softmax(Δ, xs::XRTArray)
    s = sum(exp, xs, dims=1)
    exp.(xs)./s.*(Δ .- sum(Δ .* exp.(xs), dims=1)./s)
end

@Base.pure dims_tuple(n) = tuple((0:n-1)...)
@Base.pure rev_dims_tuple(n) = tuple((n-1:-1:0)...)
dims_tuple(A, ::Colon) = dims_tuple(ndims(A))
dims_tuple(A, t::Tuple) = map(x->x-1, t)
dims_tuple(A, n::Int) = (n-1,)

@inline function Base.reshape(A::XRTArray, dims::Tuple{Vararg{Union{Int,Colon}}})
    reshape(A, Base._reshape_uncolon(A, dims))
end

@inline function Base.reshape(A::XRTArray, dims::Tuple{Vararg{Int}})
    prod(dims) == prod(size(A)) || Base._throw_dmrsa(dims, prod(size(A)))
    HloReshape(dims)(
        # HLO reshape semantics collapse the opposite way
        HloTranspose(rev_dims_tuple(ndims(A)))(A)
    )
end

@Base.pure reduced_dimensions_collapes(sz, dims) = ntuple(i->i in dims ? 1 : sz[i], length(sz))
function Base.mapreduce(f, op, A::XRTArray; dims=:)
    dt = dims_tuple(A, dims)
    res = HloReduce{typeof(op)}(dt)(op,
        HloMap{typeof(f)}()(f, A),
        XRTArray(zero(eltype(A)))
    )
    if dims != (:)
        # Put back the dimensions that HloReduce dropped;
        # Julia semantics require this.
        res = HloReshape(reduced_dimensions_collapes(size(A), dims))(res)
    end
    return res
end

using Flux

function Flux.rand_similar(x::XRTArray{T, Shape}) where {T, Shape}
    HloRng(T, Shape, xla.RandomDistribution.RNG_UNIFORM)(
        XRTArray(zero(T)),
        XRTArray(one(T)))
end

using LinearAlgebra
# TODO: Base scales by maxabs if necessary. Do we need that?
function LinearAlgebra.dot(A::XRTVector, B::XRTVector)
    ddots = DimNums((0,), (0,), (), ())
    HloDot(ddots)(A, B)
end
LinearAlgebra.norm2(A::XRTArray, p) = sqrt(dot(A, A))
function LinearAlgebra.norm(A::XRTArray, p::Real)
    if p == 2
        return LinearAlgebra.norm2(A, p)
    end
    error("Not implemented")
end

function Base.setindex(A::XRTVector{T}, v::T, i::XRTArray{<:Integer, (), 0}) where {T}
    Base.setindex(A, XRTArray{T, (), 0}(v), i)
end
function Base.setindex(A::XRTVector{T}, v::XRTArray{T, (), 0}, i::XRTArray{<:Integer, (), 0}) where {T}
    # TODO: It would be better to handle the dim mapping internally,
    # but that requires access to the julia element type, which we don't currently
    # expose
    vb = HloBroadcast(ntuple(i->i-1, ndims(T)), ntuple(_->1, ndims(A)))(v)
    inds = HloBroadcast((), (1,))(i)
    if ndims(T) != 0
        ones = HloBroadcast((), (ndims(T),))(XRTArray(1))
        inds = HloConcatenate(0)(ones, inds)
    end
    HloDynamicUpdateSlice()(A, vb, inds)
end
Base.setindex(A::XRTVector, v, i::Int64) = Base.setindex(A, v, XRTArray{Int64, (), 0}(i))

function Base.getindex(A::XRTVector{T}, i::XRTArray{<:Integer, (), 0}) where {T}
    # TODO: It would be better to handle the dim mapping internally,
    # but that requires access to the julia element type, which we don't currently
    # expose
    inds = HloBroadcast((), (1,))(i)
    sizes = ntuple(_->1, ndims(A))
    if ndims(T) != 0
        ones = HloBroadcast((), (ndims(T),))(XRTArray(1))
        inds = HloConcatenate(0)(ones, inds)
        sizes = (size(T)..., sizes...)
    end
    ret = HloDynamicSlice(sizes)(A, inds)
    ret = HloReshape(size(T))(ret)
    convert(T, ret)
end
Base.getindex(A::XRTVector, i::Int64) = Base.getindex(A, XRTArray{Int64, (), 0}(i))
