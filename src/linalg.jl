function Base.:*(A::HLOArray, B::HLOArray)
    ddots = DimNums((1,), (0,), (), ())
    HloDot(ddots)(A, B)
end
function Base.:*(A::HLOVector, B::HLOArray)
    ddots = DimNums((1,), (0,), (), ())
    HloDot(ddots)(HloReshape((length(A), 1))(A), B)
end
Base.:*(A::HLOArray{<:Any, (), 0}, B::HLOVector) = broadcast(*, A, B)

using LinearAlgebra

make_hlo_transpose(A::HLOArray) = HloTranspose((1,0))(A)
make_hlo_transpose(A::HLOVector) = make_hlo_transpose(HloReshape((length(A), 1))(A))

function Base.:*(A::Adjoint{<:Any, <:HLOArray}, B::HLOArray)
    ddots = DimNums((1,), (0,), (), ())
    HloDot(ddots)(make_hlo_transpose(A.parent), B)
end

function Base.:*(A::Adjoint{<:Any, <:HLOVector}, B::HLOVector)
    ddots = DimNums((0,), (0,), (), ())
    HloDot(ddots)(A.parent, B)
end

function Base.:*(A::Transpose{<:Any, <:HLOArray}, B::HLOArray)
    ddots = DimNums((1,), (0,), (), ())
    HloDot(ddots)(make_hlo_transpose(A.parent), B)
end

function Base.:*(A::HLOArray{T}, B::Transpose{S}) where {T, S}
    ddots = DimNums((1,), (0,), (), ())
    HloDot(ddots)(convert(HLOArray{promote_type(T, S)}, A), make_hlo_transpose(convert(HLOArray{promote_type(T, S)}, B.parent)))
end

function Base.:*(A::HLOVector{T}, B::Transpose{S, <:HLOVector}) where {T, S}
    ddots = DimNums((1,), (0,), (), ())
    HloDot(ddots)(reshape(convert(HLOArray{promote_type(T, S)}, A), (size(A, 1), 1)),
        make_hlo_transpose(convert(HLOArray{promote_type(T, S)}, B.parent)))
end

function Base.:*(A::Transpose{<:Any, <:HLOVector}, B::HLOVector)
    ddots = DimNums((0,), (0,), (), ())
    HloDot(ddots)(A.parent, B)
end

Base.convert(::Type{HLOArray}, x::XLAScalar) = HLOArray(x)

const HLOScalar{T} = HLOArray{T, (), 0}
@noinline Base.convert(::Type{Bool}, A::HLOScalar{Bool}) = convert(Array, A)[]
function Base.promote_rule(A::Type{HLOScalar{T}}, B::Type{HLOScalar{S}}) where {T<:XLAScalar, S<:XLAScalar}
    HLOArray{promote_type(T, S), (), 0}
end

function Base.promote_rule(A::Type{T}, B::Type{HLOScalar{S}}) where {T<:XLAScalar, S<:XLAScalar}
    HLOArray{promote_type(T, S), (), 0}
end


# Scalar conversions - Happens via HloConvert. In the future, we may
# want to make sure to more closely match julia semantics.
Base.convert(::Type{HLOScalar{T}}, x::HLOScalar{S}) where {T,S} = GenericHloOp{:convert}(T, ())(x)
Base.convert(::Type{HLOScalar{T}}, x::T) where T = HLOScalar{T}(x)
Base.convert(::Type{HLOScalar{T}}, x::HLOScalar{T}) where T = x
Base.convert(::Type{HLOScalar{T}}, x::XLAScalar) where {T<:XLAScalar} = HLOScalar{T}(convert(T, x))

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
    @eval Base.$(binop)(A::HLOScalar{T},
                        B::HLOScalar{T}) where {T<:XLAScalar} =
        GenericHloOp{$(quot(xlaop))}(T, ())(A, B)
    @eval Base.$(binop)(A::HLOScalar, B::HLOScalar, args::HLOScalar{<:XLAScalar}...) =
        $(binop)(promote(A, B, args...)...)
    @eval Base.$(binop)(A::HLOScalar, B::Number) = $(binop)(promote(A, B)...)
    @eval Base.$(binop)(A::Number, B::HLOScalar) = $(binop)(promote(A, B)...)
    @eval Base.$(binop)(A::HLOScalar{T}, B::HLOScalar{T}) where {T<:HLOArray} =
        Base.$(binop)(convert(T, A), convert(T, B))
end
Base.:>>(A::HLOScalar{<:Unsigned}, B::HLOScalar) = A >>> B
Base.:>>(A::HLOScalar{T}, B::HLOScalar) where {T <: Signed} = GenericHloOp{Symbol("shift-right-arithmetic")}(T, ())(A, B)

Base.:^(A::HLOScalar{T}, B::Integer) where {T} = A ^ HLOArray(Float32(B))

Base.literal_pow(::typeof(^), x::HLOScalar, ::Val{0}) = one(x)
Base.literal_pow(::typeof(^), x::HLOScalar, ::Val{1}) = x
Base.literal_pow(::typeof(^), x::HLOScalar, ::Val{2}) = x*x
Base.literal_pow(::typeof(^), x::HLOScalar, ::Val{3}) = x*x*x

# XRT Scalars are not numbers so import some functions manually
import Base: /, \, *
for f in (:/, :\, :*)
    if f != :/
        @eval ($f)(A::HLOArray{<:Any, (), 0}, B::HLOArray) = broadcast($f, A, B)
    end
    if f != :\
        @eval ($f)(A::HLOArray, B::HLOArray{<:Any, (), 0}) = broadcast($f, A, B)
    end
end


embed(x::XLAScalar) = HLOArray(x)
embed(x) = x

concat_dims(::Type{<:HLOArray{T, Sz1}}, Sz2::Tuple) where {T, Sz1} = (Sz1..., Sz2...)
concat_dims(::Type, Sz2::Tuple) where {T, Sz1} = Sz2

@noinline make_zeros_dims_tuple(::Type{T}) where {T} = ntuple(i->i-1, ndims(T))

function Base.zeros(::Type{HLOArray{T, Sz, N}}) where {T, Sz, N}
    # TODO: This assignment being necessary is probably a compiler bug
    TT = T
    HloBroadcast(make_zeros_dims_tuple(TT), Sz)(HLOArray((zero(TT),)))::HLOArray{T, Sz, N}
end
Base.zero(X::Type{HLOArray{T, Sz, N}}) where {T, Sz, N} = zeros(X)

Base.zero(A::Type{HLOArray{T, (), 0}}) where T = HLOArray(zero(T))
Base.zero(A::HLOArray) = zero(typeof(A))
Base.one(A::Type{HLOArray{T, (), 0}}) where T = HLOArray(one(T))
Base.one(A::HLOArray{<:Any, (), 0}) = one(typeof(A))
Base.max(A::HLOArray{T, (), 0}, B::HLOArray{T, (), 0}) where {T<:XLAScalar} =
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
    @eval Base.$(fname)(x::HLOArray{T, (), 0}) where {T} = GenericHloOp{$(quot(hlo))}(T, ())(x)
end

Base.sqrt(A::HLOArray{T, (), 0}) where {T} = GenericHloOp{:power}(T, ())(A, HLOArray(convert(T, 0.5)))

@eval Base.isless(A::HLOArray{<:Any, (), 0}, B::HLOArray{<:Any, (), 0}) =
    convert(Bool, GenericHloOp{$(QuoteNode(Symbol("less-than")))}(Bool, ())(A, B))
@eval Base.:(<=)(A::HLOArray{<:Any, (), 0}, B::HLOArray{<:Any, (), 0}) =
    GenericHloOp{$(QuoteNode(Symbol("less-than-or-equal-to")))}(Bool, ())(A, B)
@eval Base.:(==)(A::HLOArray{<:Any, (), 0}, B::HLOArray{<:Any, (), 0}) =
    GenericHloOp{$(QuoteNode(Symbol("equal-to")))}(Bool, ())(A, B)
@eval Base.:(!=)(A::HLOArray{<:Any, (), 0}, B::HLOArray{<:Any, (), 0}) =
    GenericHloOp{$(QuoteNode(Symbol("not-equal-to")))}(Bool, ())(A, B)
Base.inv(x::HLOScalar{T}) where {T} = HLOArray(one(T))/x

Base.:-(x::HLOScalar{T}) where {T} = -(HLOArray(zero(T)), x)

# HLOArrays are immutable. Copy is identity.
Base.copy(A::HLOArray) = A
#Base.transpose(A::HLOArray) = HloTranspose((1,0))(A)
#Base.adjoint(A::HLOArray{<:Real}) = HloTranspose((1,0))(A)
Base.permutedims(A::HLOArray, perm) = HloTranspose(map(x->x-1, perm))(A)


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

function Base.promote_shape(a::HLOArray, b::HLOArray)
    promote_shape_rec(axes(a), axes(b))
end

import Base.Broadcast

Base.similar(x::HLOArray) = error()

struct HLOArrayStyle{N} <: Broadcast.AbstractArrayStyle{N} end
(::Type{<:HLOArrayStyle})(::Val{N}) where {N} = HLOArrayStyle{N}()
Broadcast.BroadcastStyle(::Type{<:HLOArray{<:Any,Dims,N}}) where {Dims, N} =
    HLOArrayStyle{N}()

@noinline _ccdims(s) = tuple(findall(==(1), s)...)
@noinline _ncdims(s) = tuple(findall(x->x!=(1), s)...)

@Base.pure ccdims(s) = _ccdims(s)
@Base.pure ncdims(s) = _ncdims(s)

@Base.pure getindex_tuple(s::Tuple, k::Tuple) = s[collect(k)]

@inline function broadcast_to_size(arg, rsize)
    if size(arg) != rsize
        collapse_dims = ccdims(size(arg))
        non_collapse_dims = ncdims(size(arg))
        if !isa(arg, HLOArray)
            arg = convert(HLOArray, arg)
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
any_sess(a::HLOArray, args::AnyXLA...) = a.storage.sess
any_sess(a::XLAScalar, args::AnyXLA...) = any_sess(args...)

function Broadcast.copy(bc::Broadcast.Broadcasted{<:HLOArrayStyle})
    ElType = Broadcast.combine_eltypes(bc.f, bc.args)
    bc′ = Broadcast.flatten(bc)
    if Base.isconcretetype(ElType)
        args = bc′.args
        # This could be axes(bc′) if we had better constant prop
        rsize = map(length, Broadcast.combine_axes(bc′.args...))
        args = map(arg->convert(HLOArray, broadcast_to_size(arg, rsize)), bc′.args)
        return HloMap{Core.Typeof(bc′.f)}()(bc′.f, args...)
    end
    # TODO: Pull back CPU, do this there
    error("No hope $(bc.f) $(typeof(bc.args)) $(ElType)")
end

#=
using NNlib

@Base.pure function conv_windows(sz_, pad_, stride_, dilation_)
    ntuple(length(pad_)) do i
        (sz, p, s, d) = sz_[i], pad_[i], stride_[i], dilation_[i]
        WindowDims(sz, s, p, p, d, 1, true)
    end
end

function (::NNlib._conv{pad, stride, dilation})(input::HLOArray{T}, kernel::HLOArray{T}) where {T, pad, stride, dilation}
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

function (::NNlib._∇conv_data{pad, stride, dilation, 0})(dy::HLOArray{T}, input::HLOArray{T}, kernel::HLOArray) where {T, pad, stride, dilation}
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

function (::NNlib._∇conv_filter{pad, stride, dilation, 0})(dy::HLOArray{T}, input::HLOArray{T}, kernel::HLOArray) where {T, pad, stride, dilation}
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

function NNlib.maxpool(x::HLOArray, k; pad = map(_->0,k), stride = k)
    HloReduceWindow{typeof(max)}(
        make_pooling_windows(x, k, pad, stride)
    )(max, x, HLOArray(typemin(eltype(x))))
end

function NNlib.meanpool(x::HLOArray, k; pad = map(_->0,k), stride = k)
    s = HloReduceWindow{typeof(+)}(
        make_pooling_windows(x, k, pad, stride)
    )(+, x, HLOArray(zero(eltype(x))))
    wsize = convert(eltype(x), prod(k))
    s ./ HLOArray(wsize)
end

function NNlib.∇maxpool(dy::HLOArray, y::HLOArray, x::HLOArray, k; pad = map(_->0,k), stride = k)
    HloSelectAndScatter{typeof(>=), typeof(+)}(
        make_pooling_windows(x, k, pad, stride)
    )(>=, +, x, dy, HLOArray(zero(eltype(x))))
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

function NNlib.∇meanpool(dy::HLOArray, y::HLOArray, x::HLOArray, k; pad = map(_->0,k), stride = k)
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
    dy = dy ./ HLOArray(wsize)
    pdy = HloPad(make_pad_config(x, k, pad, stride))(dy, HLOArray(zero(eltype(dy))))
    HloReduceWindow{typeof(+)}(
        make_pooling_windows(x, k, pad, map(_->1, k))
    )(+, pdy, HLOArray(zero(eltype(x))))
end

@Base.pure dims_tuple(n) = tuple((0:n-1)...)
@Base.pure rev_dims_tuple(n) = tuple((n-1:-1:0)...)
dims_tuple(A, ::Colon) = dims_tuple(ndims(A))
dims_tuple(A, t::Tuple) = map(x->x-1, t)
dims_tuple(A, n::Int) = (n-1,)

@inline function Base.reshape(A::HLOArray, dims::Tuple{Vararg{Union{Int,Colon}}})
    reshape(A, Base._reshape_uncolon(A, dims))
end

@Base.pure rev_tuple(n) = reverse(n)

@inline function Base.reshape(A::HLOArray, dims::Tuple{Vararg{Int}})
    prod(dims) == prod(size(A)) || Base._throw_dmrsa(dims, prod(size(A)))
    HloTranspose(rev_dims_tuple(length(dims)))(
       HloReshape(rev_tuple(dims))(
            # HLO reshape semantics collapse the opposite way
            HloTranspose(rev_dims_tuple(ndims(A)))(A)
       )
    )
end

@Base.pure reduced_dimensions_collapes(sz, dims) = ntuple(i->i in dims ? 1 : sz[i], length(sz))
function Base.mapreduce(f, op, A::HLOArray; dims=:)
    dt = dims_tuple(A, dims)
    res = HloReduce{Core.Typeof(op)}(dt)(op,
        HloMap{Core.Typeof(f)}()(f, A),
        HLOArray(zero(eltype(A)))
    )
    if dims != (:)
        # Put back the dimensions that HloReduce dropped;
        # Julia semantics require this.
        res = HloReshape(reduced_dimensions_collapes(size(A), dims))(res)
    end
    return res
end

using Flux

function Flux.rand_similar(x::HLOArray{T, Shape}) where {T, Shape}
    HloRng(T, Shape, xla.RandomDistribution.RNG_UNIFORM)(
        HLOArray(zero(T)),
        HLOArray(one(T)))
end

using LinearAlgebra
# TODO: Base scales by maxabs if necessary. Do we need that?
function LinearAlgebra.dot(A::HLOVector, B::HLOVector)
    ddots = DimNums((0,), (0,), (), ())
    HloDot(ddots)(A, B)
end
LinearAlgebra.norm2(A::HLOArray, p) = sqrt(dot(A, A))
function LinearAlgebra.norm(A::HLOArray, p::Real)
    if p == 2
        return LinearAlgebra.norm2(A, p)
    end
    error("Not implemented")
end

function Base.setindex(A::HLOVector{T}, v::T, i::HLOArray{<:Integer, (), 0}) where {T}
    Base.setindex(A, HLOArray{T, (), 0}(v), i)
end
function Base.setindex(A::HLOVector{T}, v::HLOArray{T, (), 0}, i::HLOArray{<:Integer, (), 0}) where {T}
    # TODO: It would be better to handle the dim mapping internally,
    # but that requires access to the julia element type, which we don't currently
    # expose
    vb = HloBroadcast(ntuple(i->i-1, ndims(T)), ntuple(_->1, ndims(A)))(v)
    inds = HloBroadcast((), (1,))(i - 1)
    if ndims(T) != 0
        ones = HloBroadcast((), (ndims(T),))(HLOArray(1))
        inds = HloConcatenate(0)(ones, inds)
    end
    HloDynamicUpdateSlice()(A, vb, inds)
end
Base.setindex(A::HLOVector, v, i::Int64) = Base.setindex(A, v, HLOArray{Int64, (), 0}(i))

# TODO: Support trailing singleton
@Base.pure start_idxs_tuple(rs) = ntuple(i->isa(rs[i], Colon) ? 0 : first(rs[i]) - 1, Val(length(rs)))
function Base.setindex(A::HLOArray{T,<:Any,N}, up::HLOArray{T,<:Any,N}, rs::Vararg{Union{UnitRange{<:Integer}, Colon}, N}) where {T, N}
    inds = map(x->HloBroadcast((), (1,))(HLOArray(x)), start_idxs_tuple(rs))
    HloDynamicUpdateSlice()(A, up, HloConcatenate(0)(inds...))
end

@Base.pure function compute_result_size(tA, tInds)
    map(i->size(tA, i),
        tuple(Iterators.filter(
            i->tInds.parameters[i] <: Colon, 1:length(tInds.parameters))...))
end

function Base.getindex(A::HLOArray{T, <:Any, N}, inds::Vararg{Union{Colon, HLOArray{<:Integer, (), 0}}, N}) where {T, N}
    # TODO: It would be better to handle the dim mapping internally,
    # but that requires access to the julia element type, which we don't currently
    # expose
    ind_vector = HloConcatenate(0)(ntuple(Val(length(inds))) do i
        if isa(inds[i], Colon)
            HloBroadcast((), (1,))(HLOArray(0))
        elseif isa(inds[i], HLOArray)
            HloBroadcast((), (1,))(inds[i] - 1)
        end
    end...)
    sizes = ntuple(Val(length(inds))) do i
        isa(inds[i], Colon) ? size(A, i) : 1
    end
    if ndims(T) != 0
        ones = HloBroadcast((), (ndims(T),))(HLOArray(1))
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
Base.getindex(A::HLOVector, i::Int64) = Base.getindex(A, HLOArray{Int64, (), 0}(i))
# Override for scalars
Base.getindex(x::HLOArray{T,(),0}) where {T} = x

@Base.pure function compute_gather_tuples(A, idxs)
    array_idxs = tuple(Iterators.filter(i->!(idxs.parameters[i] <: Colon), 1:length(idxs.parameters))...)
    offset_dims = tuple(Iterators.filter(i->idxs.parameters[i+1] <: Colon, 0:length(idxs.parameters)-1)...)
    start_index_map = tuple(Iterators.filter(i->idxs.parameters[i+1] <: HLOArray, 0:length(idxs.parameters)-1)...)
    slice_sizes = map(i->size(A, i), tuple(filter(i->idxs.parameters[i] <: Colon, 1:length(idxs.parameters))...))
    slice_sizes = tuple(slice_sizes..., (1 for _ in 1:length(array_idxs))...)
    (array_idxs, offset_dims, start_index_map, slice_sizes)
end

function Base.getindex(A::HLOArray{<:Any, <:Any, N}, idxs::Vararg{Union{Colon, HLOArray{<:Integer, <:Any, 1}}, N}) where {N}
    (array_idxs_idxs, offset_dims, start_index_map, slice_sizes) = compute_gather_tuples(typeof(A), typeof(idxs))
    array_idxs = ntuple(i->idxs[array_idxs_idxs[i]], Val{length(array_idxs_idxs)}())
    M = length(array_idxs)
    if length(array_idxs) == 1
        gather_idxs = array_idxs[1] .- HLOArray(one(eltype(array_idxs[1])))
    else
        gather_shape = map(length, array_idxs)
        gather_idxs = HloConcatenate(M)(ntuple(Val{M}) do i
            HloBroadcast((i,), (gather_shape..., 1))(array_idxs[i] .- HLOArray(one(eltype(array_idxs[i]))))
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

function Base.getindex(A::HLOArray{<:Any, <:Any, N}, rs::Vararg{Union{UnitRange{<:Integer}, Colon}, N}) where {N}
    HloSlice(slices_tuple(rs, size(typeof(A))))(A)
end

using DiffRules
DiffRules.select(p::HLOArray{Bool, (), 0}, x::HLOArray{T, (), 0}, y::HLOArray{T, (), 0}) where {T} =
    GenericHloOp{:select}(T, ())(p, x, y)
DiffRules.select(p::Bool, x::HLOArray{T, (), 0}, y::HLOArray{T, (), 0}) where {T} =
    GenericHloOp{:select}(T, ())(HLOArray{Bool, (), 0}(p), x, y)

function HLOArray(a::UnitRange{T}) where {T}
    HloIota(T, (length(a),), 0)() .+ first(a)
end

function HLOArray(a::Transpose)
    make_hlo_transpose(HLOArray(a.parent))
end

function change_eltype(T::Type, x::HLOArray)
    GenericHloOp{:convert}(T, size(x))(x)
end
change_eltype(::Type{T}, x::HLOArray{T}) where {T} = x
Base.convert(::Type{HLOArray{T}}, x::HLOArray{S}) where {T,S} = change_eltype(T, x)
Base.convert(::Type{HLOArray{T}}, x::HLOArray{T}) where {T} = x
Base.convert(::Type{HLOArray}, x::HLOArray) = x

for T in Base.uniontypes(XLA.XLAScalar)
    @eval (::Type{$T})(x::HLOScalar{S}) where {S<:XLAScalar} = convert(HLOArray{$T}, x)
    @eval (::Type{$T})(x::HLOScalar{$T}) where {S<:XLAScalar} = x
end

@Base.pure @noinline function padding_tuple(aiT::Type{<:HLOArray}, dims::Int)
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

function _cat(dims::Int, a::HLOArray{T}...) where {T}
    b = ntuple(ntupler{dims, typeof(a)}(a), length(a))
    HloConcatenate(dims-1)(b...)
end
function _cat(dims::Int, a::HLOArray...)
    let eT = Base.promote_eltype(a...)
        _cat(dims, map(a->change_eltype(eT, a), a)...)
    end
end
Base.hcat(a::HLOArray...) = _cat(2, a...)
Base.vcat(a::HLOArray...) = _cat(1, a...)
Base.cat(a::HLOArray...; dims) = _cat(dims, a...)

using Statistics
@Base.pure @noinline count_summands(T::Type{<:HLOArray}, dims) = prod(size(T)[[dims...]])
@Base.pure @noinline count_summands(T::Type{<:HLOArray}, dims::Colon) = prod(size(T))
dimsum(A, dims) = sum(A; dims=dims)
function Statistics.mean(A::HLOArray; dims=:)
    summed = dimsum(A, dims)
    x = convert(eltype(A), count_summands(typeof(A), dims))
    nsummands = HLOArray(x)
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
function Base._dropdims(A::HLOArray, dims::Base.Dims)
    reshape(A, compute_dropdims_d(typeof(A), dims))
end

@Base.pure function make_repeat_dims(sz, rep_dims)
    padded_sz = ntuple(i->i <= length(sz) ? sz[i] : 1, length(rep_dims))
    dim_mapping = tuple((2*i+1 for i = 0:length(sz)-1)...)
    result_szs = tuple(Iterators.flatten(zip(rep_dims, padded_sz))...)
    final_sz = tuple((a*b for (a,b) in zip(rep_dims, padded_sz))...)
    dim_mapping, result_szs, final_sz
end

function Base.repeat(A::HLOArray, dims::Integer...)
    dims_mapping, result_szs, final_sz = make_repeat_dims(size(A), dims)
    XLA.HloReshape(final_sz)(XLA.HloBroadcast(dims_mapping, result_szs)(A))
end
=#
