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

make_hlo_transpose(A::XRTArray) = HloTranspose((1,0))(A)
make_hlo_transpose(A::XRTVector) = make_hlo_transpose(HloReshape((length(A), 1))(A))

function Base.:*(A::Adjoint{<:Any, <:XRTArray}, B::XRTArray)
    ddots = DimNums((1,), (0,), (), ())
    HloDot(ddots)(make_hlo_transpose(A.parent), B)
end

function Base.:*(A::Adjoint{<:Any, <:XRTVector}, B::XRTVector)
    ddots = DimNums((0,), (0,), (), ())
    HloDot(ddots)(A.parent, B)
end

function Base.:*(A::Transpose{<:Any, <:XRTArray}, B::XRTArray)
    ddots = DimNums((1,), (0,), (), ())
    HloDot(ddots)(make_hlo_transpose(A.parent), B)
end

function Base.:*(A::XRTArray{T}, B::Transpose{S}) where {T, S}
    ddots = DimNums((1,), (0,), (), ())
    HloDot(ddots)(convert(XRTArray{promote_type(T, S)}, A), make_hlo_transpose(convert(XRTArray{promote_type(T, S)}, B.parent)))
end

function Base.:*(A::XRTVector{T}, B::Transpose{S, <:XRTVector}) where {T, S}
    ddots = DimNums((1,), (0,), (), ())
    HloDot(ddots)(reshape(convert(XRTArray{promote_type(T, S)}, A), (size(A, 1), 1)),
        make_hlo_transpose(convert(XRTArray{promote_type(T, S)}, B.parent)))
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

function Base.promote_rule(A::Type{T}, B::Type{XRTScalar{S}}) where {T<:XLAScalar, S<:XLAScalar}
    XRTArray{promote_type(T, S), (), 0}
end


# Scalar conversions - Happens via HloConvert. In the future, we may
# want to make sure to more closely match julia semantics.
Base.convert(::Type{XRTScalar{T}}, x::XRTScalar{S}) where {T,S} = GenericHloOp{:convert}(T, ())(x)
Base.convert(::Type{XRTScalar{T}}, x::T) where T = XRTScalar{T}(x)
Base.convert(::Type{XRTScalar{T}}, x::XRTScalar{T}) where T = x
Base.convert(::Type{XRTScalar{T}}, x::XLAScalar) where {T<:XLAScalar} = XRTScalar{T}(convert(T, x))

# A couple of scalar embeddings (could use casette in the future)
using Base.Meta
for (binop, xlaop) in (
        (:+, :add),
        (:-, :subtract),
        (:/, :divide),
        (:*, :multiply),
        (:%, :remainder),
        (:&, :and),
        (:<<, Symbol("shift-left")),
        (:>>>, Symbol("shift-right-logical")),
        (:^, :power),
    )
    @eval Base.$(binop)(A::XRTScalar{T},
                        B::XRTScalar{T}) where {T<:XLAScalar} =
        GenericHloOp{$(quot(xlaop))}(T, ())(A, B)
    @eval Base.$(binop)(A::XRTScalar, B::XRTScalar, args::XRTScalar{<:XLAScalar}...) =
        $(binop)(promote(A, B, args...)...)
    @eval Base.$(binop)(A::XRTScalar, B::Number) = $(binop)(promote(A, B)...)
    @eval Base.$(binop)(A::Number, B::XRTScalar) = $(binop)(promote(A, B)...)
    @eval Base.$(binop)(A::XRTScalar{T}, B::XRTScalar{T}) where {T<:XRTArray} =
        Base.$(binop)(convert(T, A), convert(T, B))
end
Base.:>>(A::XRTScalar{<:Unsigned}, B::XRTScalar) = A >>> B
Base.:>>(A::XRTScalar{T}, B::XRTScalar) where {T <: Signed} = GenericHloOp{Symbol("shift-right-arithmetic")}(T, ())(A, B)

Base.:^(A::XRTScalar{T}, B::Integer) where {T} = A ^ XRTArray(Float32(B))

Base.literal_pow(::typeof(^), x::XRTScalar, ::Val{0}) = one(x)
Base.literal_pow(::typeof(^), x::XRTScalar, ::Val{1}) = x
Base.literal_pow(::typeof(^), x::XRTScalar, ::Val{2}) = x*x
Base.literal_pow(::typeof(^), x::XRTScalar, ::Val{3}) = x*x*x

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

@noinline make_zeros_dims_tuple(::Type{T}) where {T} = ntuple(i->i-1, ndims(T))

function Base.zeros(::Type{XRTArray{T, Sz, N}}) where {T, Sz, N}
    # TODO: This assignment being necessary is probably a compiler bug
    TT = T
    HloBroadcast(make_zeros_dims_tuple(TT), Sz)(XRTArray((zero(TT),)))::XRTArray{T, Sz, N}
end
Base.zero(X::Type{XRTArray{T, Sz, N}}) where {T, Sz, N} = zeros(X)

Base.zero(A::Type{XRTArray{T, (), 0}}) where T = XRTArray(zero(T))
Base.zero(A::XRTArray) = zero(typeof(A))
Base.one(A::Type{XRTArray{T, (), 0}}) where T = XRTArray(one(T))
Base.one(A::XRTArray{<:Any, (), 0}) = one(typeof(A))
Base.max(A::XRTArray{T, (), 0}, B::XRTArray{T, (), 0}) where {T<:XLAScalar} =
    GenericHloOp{:maximum}(T, ())(A, B)
    
for (fname, hlo) in (
        (:exp, :exponential),
        (:sin, :sine),
        (:cos, :cosine),
        (:tanh, :tanh),
        (:expm1, Symbol("exponential-minus-one")),
        (:log, :log),
        (:log1p, Symbol("log-plus-one"))
    )
    @eval Base.$(fname)(x::XRTArray{T, (), 0}) where {T} = GenericHloOp{$(quot(hlo))}(T, ())(x)
end

Base.sqrt(A::XRTArray{T, (), 0}) where {T} = GenericHloOp{:power}(T, ())(A, XRTArray(convert(T, 0.5)))

@eval Base.isless(A::XRTArray{<:Any, (), 0}, B::XRTArray{<:Any, (), 0}) =
    convert(Bool, GenericHloOp{$(QuoteNode(Symbol("less-than")))}(Bool, ())(A, B))
@eval Base.:(<=)(A::XRTArray{<:Any, (), 0}, B::XRTArray{<:Any, (), 0}) =
    GenericHloOp{$(QuoteNode(Symbol("less-than-or-equal-to")))}(Bool, ())(A, B)
@eval Base.:(==)(A::XRTArray{<:Any, (), 0}, B::XRTArray{<:Any, (), 0}) =
    GenericHloOp{$(QuoteNode(Symbol("equal-to")))}(Bool, ())(A, B)
@eval Base.:(!=)(A::XRTArray{<:Any, (), 0}, B::XRTArray{<:Any, (), 0}) =
    GenericHloOp{$(QuoteNode(Symbol("not-equal-to")))}(Bool, ())(A, B)
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
        return HloMap{Core.Typeof(bc′.f)}()(bc′.f, args...)
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
        padded_out_size = isz[i] + sz - 1
        pad_before = sz - 1 - p
        expanded_osz = (osz[i]-1)*s + 1
        pad_after = padded_out_size - expanded_osz - pad_before
        # N.B.: The window reversal flag is flipped here
        WindowDims(sz, 1, pad_before, pad_after, d, s, false)
    end
    convdims = ConvDimNums(
        3, 2, (0, 1),
        # N.B. The input and output dimensions are exchanged here from the
        # standard notion of convolution
        3, 2, (0, 1),
        3, 2, (0, 1)
    )
    r = HloConv(windows, convdims)(dy, mirrored)
    @assert size(r) == size(input)
    r
end

function (::NNlib._∇conv_filter{pad, stride, dilation, 0})(dy::XRTArray{T}, input::XRTArray{T}, kernel::XRTArray) where {T, pad, stride, dilation}
    # TODO: Validate that this is correct. Unfortunately, the NNlib
    # implementation itself is broken, so we need to fix that first
    pad_, stride_ = NNlib.padtuple(input, pad), NNlib.padtuple(input, stride)
    dilation_ = NNlib.padtuple(kernel, dilation)
    sz_ = size(kernel); isz = size(input); osz = size(dy)
    windows = ntuple(length(pad_)) do i
        (sz, p, s, d) = sz_[i], pad_[i], stride_[i], dilation_[i]
        expanded_osz = (osz[i]-1)*s + 1
        padded_in_size = expanded_osz + (sz - 1) * d
        pad_total = padded_in_size - isz[i]
        pad_before = max(ceil(Int, pad_total/2), 0)
        pad_after = pad_total - pad_before
        WindowDims(osz[i], d, pad_before, pad_after, s, 1, false)
    end
    convdims = ConvDimNums(
        # N.B. The input and output dimensions are exchanged here from the
        # standard notion of convolution
        2, 3, (0, 1),
        3, 2, (0, 1),
        2, 3, (0, 1)
    )
    r = HloConv(windows, convdims)(input, dy)
    r = HloRev((0,1))(r)
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
    s = HloReduceWindow{typeof(+)}(
        make_pooling_windows(x, k, pad, stride)
    )(+, x, XRTArray(zero(eltype(x))))
    wsize = convert(eltype(x), prod(k))
    s ./ XRTArray(wsize)
end

function NNlib.∇maxpool(dy::XRTArray, y::XRTArray, x::XRTArray, k; pad = map(_->0,k), stride = k)
    HloSelectAndScatter{typeof(>=), typeof(+)}(
        make_pooling_windows(x, k, pad, stride)
    )(>=, +, x, dy, XRTArray(zero(eltype(x))))
end

function make_pad_config(x, k, pad, stride)
    k_, pad_, stride_ = NNlib.padtuple(x, k),
                        NNlib.padtuple(x, pad),
                        NNlib.padtuple(x, stride)
    windows = ntuple(ndims(x)) do i
        (sz, p, s) = i <= length(k) ? (k[i], pad[i], stride[i]) : (1, 0, 1)
        #@assert p == 0 # TODO: What does this do? Probably reduce the edge padding
        PaddingConfigDim(sz - 1, sz - 1, s - 1)
    end
end

function NNlib.∇meanpool(dy::XRTArray, y::XRTArray, x::XRTArray, k; pad = map(_->0,k), stride = k)
    # General algorithm for ∇meanpool.
    # The idea here is relatively simple. Without loss of generality, we can ignore the
    # division by a constant - i.e. we're looking at the gradient of a (potentially,
    # padded and strided sum of a window). These windows can be overlapping, so in general,
    # we once again need to do a summed reduction of all the gradients. We do so as follows:
    # Suppose we have a 3x3 array:
    #
    #   123
    #   456
    #   789
    #
    # And we sum in a 2x2 window, i.e.
    # our gradients look like:
    #
    #   ab
    #   cd
    #
    # with a=1+2+4+5, c=4+5+7+8, etc. To compute, the gradient, we pad up to
    #
    #  0000
    #  0ab0
    #  0cd0
    #  0000
    #
    # and once again sum with a 2x2 window. It is easy to check that this procedure gives
    # the correct answer. If we have a stride, we need to add interior padding in addition.
    # Another way to think of this and arrive at the same result is to see
    # the summation as a convolution with an all-ones kernel.
    #
    # TODO: Does it make sense to special case to the case of non-overlapping windows
    # or does XLA take care of that?
    wsize = convert(eltype(x), prod(k))
    dy = dy ./ XRTArray(wsize)
    pdy = HloPad(make_pad_config(x, k, pad, stride))(dy, XRTArray(zero(eltype(dy))))
    HloReduceWindow{typeof(+)}(
        make_pooling_windows(x, k, pad, map(_->1, k))
    )(+, pdy, XRTArray(zero(eltype(x))))
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

@Base.pure rev_tuple(n) = reverse(n)

@inline function Base.reshape(A::XRTArray, dims::Tuple{Vararg{Int}})
    prod(dims) == prod(size(A)) || Base._throw_dmrsa(dims, prod(size(A)))
    HloTranspose(rev_dims_tuple(length(dims)))(
       HloReshape(rev_tuple(dims))(
            # HLO reshape semantics collapse the opposite way
            HloTranspose(rev_dims_tuple(ndims(A)))(A)
       )
    )
end

@Base.pure reduced_dimensions_collapes(sz, dims) = ntuple(i->i in dims ? 1 : sz[i], length(sz))
function Base.mapreduce(f, op, A::XRTArray; dims=:)
    dt = dims_tuple(A, dims)
    res = HloReduce{Core.Typeof(op)}(dt)(op,
        HloMap{Core.Typeof(f)}()(f, A),
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
    inds = HloBroadcast((), (1,))(i - 1)
    if ndims(T) != 0
        ones = HloBroadcast((), (ndims(T),))(XRTArray(1))
        inds = HloConcatenate(0)(ones, inds)
    end
    HloDynamicUpdateSlice()(A, vb, inds)
end
Base.setindex(A::XRTVector, v, i::Int64) = Base.setindex(A, v, XRTArray{Int64, (), 0}(i))

# TODO: Support trailing singleton 
@Base.pure start_idxs_tuple(rs) = ntuple(i->isa(rs[i], Colon) ? 0 : first(rs[i]) - 1, Val(length(rs)))
function Base.setindex(A::XRTArray{T,<:Any,N}, up::XRTArray{T,<:Any,N}, rs::Vararg{Union{UnitRange{<:Integer}, Colon}, N}) where {T, N}
    inds = map(x->HloBroadcast((), (1,))(XRTArray(x)), start_idxs_tuple(rs))
    HloDynamicUpdateSlice()(A, up, HloConcatenate(0)(inds...))
end

@Base.pure function compute_result_size(tA, tInds) 
    map(i->size(tA, i),
        tuple(Iterators.filter(
            i->tInds.parameters[i] <: Colon, 1:length(tInds.parameters))...))
end

function Base.getindex(A::XRTArray{T, <:Any, N}, inds::Vararg{Union{Colon, XRTArray{<:Integer, (), 0}}, N}) where {T, N}
    # TODO: It would be better to handle the dim mapping internally,
    # but that requires access to the julia element type, which we don't currently
    # expose
    ind_vector = HloConcatenate(0)(ntuple(Val(length(inds))) do i
        if isa(inds[i], Colon)
            HloBroadcast((), (1,))(XRTArray(0))
        elseif isa(inds[i], XRTArray)
            HloBroadcast((), (1,))(inds[i] - 1)
        end
    end...)
    sizes = ntuple(Val(length(inds))) do i
        isa(inds[i], Colon) ? size(A, i) : 1
    end
    if ndims(T) != 0
        ones = HloBroadcast((), (ndims(T),))(XRTArray(1))
        ind_vector = HloConcatenate(0)(ones, ind_vector)
        sizes = (size(T)..., sizes...)
    end
    result_sizes = compute_result_size(typeof(A), typeof(inds))
    ret = HloDynamicSlice(sizes)(A, ind_vector)
    ret = HloReshape(tuple((ndims(T) == 0 ? () : size(T))..., result_sizes...))(ret)
    if result_sizes === () && !(T <: XLAScalar)
        return convert(T, ret)
    end
    return ret
end
Base.getindex(A::XRTVector, i::Int64) = Base.getindex(A, XRTArray{Int64, (), 0}(i))
# Override for scalars
Base.getindex(x::XRTArray{T,(),0}) where {T} = x

@Base.pure function compute_gather_tuples(A, idxs)
    array_idxs = tuple(Iterators.filter(i->!(idxs.parameters[i] <: Colon), 1:length(idxs.parameters))...)
    offset_dims = tuple(Iterators.filter(i->idxs.parameters[i+1] <: Colon, 0:length(idxs.parameters)-1)...)
    start_index_map = tuple(Iterators.filter(i->idxs.parameters[i+1] <: XRTArray, 0:length(idxs.parameters)-1)...)
    slice_sizes = map(i->size(A, i), tuple(filter(i->idxs.parameters[i] <: Colon, 1:length(idxs.parameters))...))
    slice_sizes = tuple(slice_sizes..., (1 for _ in 1:length(array_idxs))...)
    (array_idxs, offset_dims, start_index_map, slice_sizes)
end

function Base.getindex(A::XRTArray{<:Any, <:Any, N}, idxs::Vararg{Union{Colon, XRTArray{<:Integer, <:Any, 1}}, N}) where {N}
    (array_idxs_idxs, offset_dims, start_index_map, slice_sizes) = compute_gather_tuples(typeof(A), typeof(idxs))
    array_idxs = ntuple(i->idxs[array_idxs_idxs[i]], Val{length(array_idxs_idxs)}())
    M = length(array_idxs)
    if length(array_idxs) == 1
        gather_idxs = array_idxs[1] .- XRTArray(one(eltype(array_idxs[1])))
    else
        gather_shape = map(length, array_idxs)
        gather_idxs = HloConcatenate(M)(ntuple(Val{M}) do i
            HloBroadcast((i,), (gather_shape..., 1))(array_idxs[i] .- XRTArray(one(eltype(array_idxs[i]))))
        end...)
    end
    dims = GatherDims(
        offset_dims,
        start_index_map, #= collapsed_slice_dims =#
        start_index_map,
        M #= index_vector_dim =#
    )
    HloGather(dims, slice_sizes)(A, gather_idxs)
end

@noinline function compute_slice_dim(rs, sz, i)
    r = rs[i]
    isa(r, UnitRange) ?
        SliceDim(first(r)-1, last(r), step(r)) :
        SliceDim(0, sz[i], 1)
end

@Base.pure slices_tuple(rs, sz) = ntuple(i->compute_slice_dim(rs, sz, i), Val(length(sz)))

function Base.getindex(A::XRTArray{<:Any, <:Any, N}, rs::Vararg{Union{UnitRange{<:Integer}, Colon}, N}) where {N}
    HloSlice(slices_tuple(rs, size(typeof(A))))(A)
end

using DiffRules
DiffRules.select(p::XRTArray{Bool, (), 0}, x::XRTArray{T, (), 0}, y::XRTArray{T, (), 0}) where {T} =
    GenericHloOp{:select}(T, ())(p, x, y)
DiffRules.select(p::Bool, x::XRTArray{T, (), 0}, y::XRTArray{T, (), 0}) where {T} =
    GenericHloOp{:select}(T, ())(XRTArray{Bool, (), 0}(p), x, y)

function XRTArray(a::UnitRange{T}) where {T}
    HloIota(T, (length(a),), 0)() .+ first(a)
end

function XRTArray(a::Transpose)
    make_hlo_transpose(XRTArray(a.parent))
end

function change_eltype(T::Type, x::XRTArray)
    GenericHloOp{:convert}(T, size(x))(x)
end
change_eltype(::Type{T}, x::XRTArray{T}) where {T} = x
Base.convert(::Type{XRTArray{T}}, x::XRTArray{S}) where {T,S} = change_eltype(T, x)
Base.convert(::Type{XRTArray{T}}, x::XRTArray{T}) where {T} = x
Base.convert(::Type{XRTArray}, x::XRTArray) = x

for T in Base.uniontypes(XLA.XLAScalar)
    @eval (::Type{$T})(x::XRTScalar{S}) where {S<:XLAScalar} = convert(XRTArray{$T}, x)
    @eval (::Type{$T})(x::XRTScalar{$T}) where {S<:XLAScalar} = x
end

@Base.pure @noinline function padding_tuple(aiT::Type{<:XRTArray}, dims::Int)
    ntuple(_->1, Val(dims))
end

struct ntupler{dims, T}
    a::T
end
function (nt::ntupler{dims, T})(i::Int) where {dims, T}
    ai = nt.a[i]
    if ndims(ai) < dims
        pt = ntuple(_->1, dims-ndims(ai))
        HloBroadcast(ntuple(i->i-1, ndims(ai)), (size(ai)...,
            pt...))(ai)
    else
        ai
    end
end

function _cat(dims::Int, a::XRTArray{T}...) where {T}
    b = ntuple(ntupler{dims, typeof(a)}(a), length(a))
    HloConcatenate(dims-1)(b...)
end
function _cat(dims::Int, a::XRTArray...)
    let eT = Base.promote_eltype(a...)
        _cat(dims, map(a->change_eltype(eT, a), a)...)
    end
end
Base.hcat(a::XRTArray...) = _cat(2, a...)
Base.vcat(a::XRTArray...) = _cat(1, a...)
Base.cat(a::XRTArray...; dims) = _cat(dims, a...)

using Statistics
@Base.pure @noinline count_summands(T::Type{<:XRTArray}, dims) = prod(size(T)[[dims...]])
dimsum(A, dims) = sum(A; dims=dims)
function Statistics.mean(A::XRTArray; dims=:)
    summed = dimsum(A, dims)
    x = convert(eltype(A), count_summands(typeof(A), dims))
    nsummands = XRTArray(x)
    summed ./ nsummands
end

Base.@pure @noinline function compute_dropdims_d(A, dims)
    for i in 1:length(dims)
        1 <= dims[i] <= ndims(A) || throw(ArgumentError("dropped dims must be in range 1:ndims(A)"))
        length(axes(A, dims[i])) == 1 || throw(ArgumentError("dropped dims must all be size 1"))
        for j = 1:i-1
            dims[j] == dims[i] && throw(ArgumentError("dropped dims must be unique"))
        end
    end
    d = ()
    for i = 1:ndims(A)
        if !in(i, dims)
            d = tuple(d..., axes(A, i))
        end
    end
    return d
end
function Base._dropdims(A::XRTArray, dims::Base.Dims)
    reshape(A, compute_dropdims_d(typeof(A), dims))
end
