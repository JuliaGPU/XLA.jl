function Base.:*(A::XRTArray, B::XRTArray)
    ddots = DimNums((1,), (0,), (), ())
    HloDot{eltype(A), (size(A, 1), size(B, 2))}(ddots)(A, B)
end

function Base.:*(A::XRTArray, B::XRTVector)
    ddots = DimNums((1,), (0,), (), ())
    HloDot{eltype(A), (size(A, 1),)}(ddots)(A, B)
end

import Base.Broadcast

struct XRTArrayStyle{N} <: Broadcast.AbstractArrayStyle{N} end
(::Type{<:XRTArrayStyle})(::Val{N}) where {N} = XRTArrayStyle{N}()
Broadcast.BroadcastStyle(::Type{<:XRTArray{<:Any,Dims,N}}) where {Dims, N} =
    XRTArrayStyle{N}()
global cheat

@noinline _ccdims(s) = tuple(findall(==(1), s)...)
@noinline _ncdims(s) = tuple(findall(==(1), s)...)

@Base.pure ccdims(s) = _ccdims(s)
@Base.pure ncdims(s) = _ncdims(s)

@inline function broadcast_to_size(arg, rsize)
    if size(arg) != rsize
        collapse_dims = ccdims(size(arg))
        non_collapse_dims = ncdims(size(arg))
        if !isa(arg, XRTArray)
            arg = XRTArray(cheat::TensorFlow.Session, fill(arg))
        end
        if collapse_dims != ()
            let s = size(arg), nc = non_collapse_dims
                arg = HloCollapse{eltype(arg), ntuple(i->s[nc[i]], length(nc))}(
                    map(x->x - 1, collapse_dims))(arg)
            end
        end
        return HloBroadcast{eltype(arg), rsize}(map(x->x - 1, non_collapse_dims))(
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
        global cheat
        cheat = any_sess(args...)::TensorFlow.Session
        # This could be axes(bc′) if we had better constant prop
        rsize = map(length, Broadcast.combine_axes(bc′.args...))
        args = map(arg->broadcast_to_size(arg, rsize), bc′.args)
        return HloMap{ElType, rsize}(bc′.f)(args...)
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
    HloConv{eltype(input), NNlib.cdims(size(input), NNlib.dilation_dims(kernel, dilation), pad_, stride_)}(
        windows, convdims)(input, kernel)
end

function NNlib.maxpool(x::XRTArray, k; pad = map(_->0,k), stride = k)
    k_, pad_, stride_ = NNlib.padtuple(x, k),
                        NNlib.padtuple(x, pad),
                        NNlib.padtuple(x, stride)
    windows = ntuple(length(k_)) do i
        (sz, p, s) = k[i], pad[i], stride[i]
        WindowDims(sz, s, p, p, 1, 1, false)
    end
    rdims = NNlib.pdims(size(x), k, NNlib.expand(Val{length(k)}, pad),
                NNlib.expand(Val{length(k)}, stride))
    HloReduceWindow{eltype(x), rdims}(
        max,
        windows
    )(x)
end

NNlib.softmax(xs::XRTArray) = exp.(xs) ./ sum(exp.(xs))

@Base.pure dims_tuple(n) = tuple((0:n-1)...)

@inline function Base.reshape(A::XRTArray, dims::Tuple{Vararg{Union{Int,Colon}}})
    dims = Base._reshape_uncolon(A, dims)
    HloReshape{eltype(A), dims}(
        dims_tuple(ndims(A)), dims
    )(A)
end

function Base.mapreduce(f, op, A::XRTArray; dims = :)
    HloReduce{eltype(A), ()}(op, tuple((0:ndims(A)-1)...))(
        HloMap{eltype(A), size(A)}(f)(A),
        XRTArray(A.storage.sess, fill(zero(eltype(A))))
    )
end

using Flux

function Flux.rand_similar(x::XRTArray{T, Shape}) where {T, Shape}
    HloRng{T, Shape}(xla.RandomDistribution.RNG_UNIFORM)(
        XRTArray(x.storage.sess, fill(zero(T))),
        XRTArray(x.storage.sess, fill(one(T))))
end
