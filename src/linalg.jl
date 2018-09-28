function Base.:*(A::XRTArray, B::XRTArray)
    ddots = DimNums((1,), (0,), (), ())
    HloDot(ddots)(A, B)
end

function Base.:*(A::XRTArray, B::XRTVector)
    ddots = DimNums((1,), (0,), (), ())
    HloDot(ddots)(A, B)
end

# A couple of scalar embeddings (could use casette in the future)
Base.:+(A::XRTArray{T, (), 0}, B::XRTArray{T, (), 0}) where {T<:XLAScalar} =
    GenericHloOp{:add}(T, ())(A, B)
Base.:/(A::XRTArray{T, (), 0}, B::XRTArray{T, (), 0}) where {T<:XLAScalar} =
    GenericHloOp{:divide}(T, ())(A, B)
Base.zero(A::Type{XRTArray{T, (), 0}}) where T = XRTArray(zero(T))
Base.zero(A::XRTArray{<:Any, (), 0}) = zero(typeof(A))
Base.max(A::XRTArray{T, (), 0}, B::XRTArray{T, (), 0}) where {T<:XLAScalar} =
    GenericHloOp{:maximum}(T, ())(A, B)
Base.exp(A::XRTArray{T, (), 0}) where {T} = GenericHloOp{:exponential}(T, ())(A)

Base.transpose(A::XRTArray) = HloTranspose((1,0))(A)
Base.permutedims(A::XRTArray, perm) = HloTranspose(map(x->x-1, perm))(A)

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
            arg = XRTArray(fill(arg))
        end
        if collapse_dims != ()
            arg = HloReshape(
                dims_tuple(ndims(arg)),
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
        args = map(arg->broadcast_to_size(arg, rsize), bc′.args)
        return HloMap(bc′.f)(args...)
    end
    # TODO: Pull back CPU, do this there
    error("No hope")
end

using NNlib

function NNlib.conv(input::XRTArray, kernel::XRTArray; pad = 0, stride = 1, dilation = 1)
    pad_, stride_ = NNlib.padtuple(input, pad), NNlib.padtuple(input, stride)
    dilation_ = NNlib.padtuple(kernel, dilation)
    sz_ = size(kernel)
    windows = ntuple(length(pad_)) do i
        (sz, p, s, d) = sz_[i], pad_[i], stride_[i], dilation_[i]
        WindowDims(sz, s, p, p, d, 1, false)
    end
    convdims = ConvDimNums(
        3, 2, (0, 1),
        2, 3, (0, 1),
        3, 2, (0, 1)
    )
    HloConv(windows, convdims)(input, kernel)
end

function make_maxpool_windows(x, k, pad, stride)
    k_, pad_, stride_ = NNlib.padtuple(x, k),
                        NNlib.padtuple(x, pad),
                        NNlib.padtuple(x, stride)
    windows = ntuple(ndims(x)) do i
        (sz, p, s) = i <= length(k) ? (k[i], pad[i], stride[i]) : (1, 0, 1)
        WindowDims(sz, s, p, p, 1, 1, false)
    end
end

function NNlib.maxpool(x::XRTArray, k; pad = map(_->0,k), stride = k)
    HloReduceWindow(
        max,
        make_maxpool_windows(x, k, pad, stride)
    )(x, XRTArray(typemin(eltype(x))))
end

function NNlib.∇maxpool(dy::XRTArray, y::XRTArray, x::XRTArray, k; pad = map(_->0,k), stride = k)
    HloSelectAndScatter2(
        >=, +,
        make_maxpool_windows(x, k, pad, stride)
    )(x, dy, XRTArray(zero(eltype(x))))
end

NNlib.softmax(xs::XRTArray) = exp.(xs) ./ sum(exp.(xs))
function NNlib.∇softmax(Δ, xs::XRTArray)
    s = sum(exp, xs, dims=1)
    exp.(xs)./s.*(Δ .- sum(Δ .* exp.(xs), dims=1)./s)
end

@Base.pure dims_tuple(n) = tuple((0:n-1)...)
dims_tuple(A, ::Colon) = dims_tuple(ndims(A))
dims_tuple(A, t::Tuple) = t
dims_tuple(A, n::Int) = (n,)

@inline function Base.reshape(A::XRTArray, dims::Tuple{Vararg{Union{Int,Colon}}})
    dims = Base._reshape_uncolon(A, dims)
    HloReshape(
        dims_tuple(ndims(A)), dims
    )(A)
end

@inline function Base.reshape(A::XRTArray, dims::Tuple{Vararg{Int}})
    HloReshape(
        dims_tuple(ndims(A)), dims
    )(A)
end

function Base.mapreduce(f, op, A::XRTArray; dims=:)
    HloReduce(op, dims_tuple(A, dims))(
        HloMap(f)(A),
        XRTArray(zero(eltype(A)))
    )
end

using Flux

function Flux.rand_similar(x::XRTArray{T, Shape}) where {T, Shape}
    HloRng(T, Shape, xla.RandomDistribution.RNG_UNIFORM)(
        XRTArray(zero(T)),
        XRTArray(one(T)))
end
