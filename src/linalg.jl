const XRTMatrix{T, Dims} = XRTArray{T, Dims, 2} where {T, Dims}
const XRTVector{T, Dims} = XRTArray{T, Dims, 1} where {T, Dims}
function Base.:*(A::XRTArray, B::XRTArray)
    ddots = DimNums((1,), (0,), (), ())
    HloDot{eltype(A), (size(A, 1), size(B, 2))}(ddots)(A, B)
end

function Base.:*(A::XRTArray, B::XRTVector)
    ddots = DimNums((1,), (0,), (), ())
    @Base.show size(A, 1)
    HloDot{eltype(A), (size(A, 1),)}(ddots)(A, B)
end

import Base.Broadcast

struct XRTArrayStyle{N} <: Broadcast.AbstractArrayStyle{N} end
(::Type{<:XRTArrayStyle})(::Val{N}) where {N} = XRTArrayStyle{N}()
Broadcast.BroadcastStyle(::Type{<:XRTArray{<:Any,Dims,N}}) where {Dims, N} =
    XRTArrayStyle{N}()

function Broadcast.copy(bc::Broadcast.Broadcasted{<:XRTArrayStyle})
    ElType = Broadcast.combine_eltypes(bc.f, bc.args)
    bc′ = Broadcast.flatten(bc)
    if Base.isconcretetype(ElType)
        rsize = map(length, Broadcast.broadcast_axes(bc′))
        args = map(bc′.args) do arg
            if size(arg) != rsize
                collapse_dims = tuple(findall(==(1), size(arg))...)
                non_collapse_dims = tuple(findall(x->x != 1, size(arg))...)
                if collapse_dims != ()
                    arg = HloCollapse{eltype(arg), size(arg)[collect(non_collapse_dims)]}(
                        map(x->x - 1, collapse_dims))(arg)
                end
                return HloBroadcast{eltype(arg), rsize}(map(x->x - 1, non_collapse_dims))(
                    arg
                )
            else
                return arg
            end
        end
        return HloMap{ElType, rsize}(bc′.f)(args...)
    end
    # TODO: Pull back CPU, do this there
    error("No hope")
end

using NNlib

function NNlib.conv(input::XRTArray, kernel::XRTArray; pad = 0, stride = 1, dilation = 1)
    pad_, stride_ = NNlib.padtuple(input, pad), NNlib.padtuple(input, stride)
    dialation_ = NNlib.padtuple(kernel, dilation)
    windows = tuple(map(zip(size(kernel), pad_, stride_, dialation_)) do (sz, p, s, d)
        WindowDims(sz, s, p, p, d, 1, false)
    end...)
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
    windows = tuple(map(zip(k_, pad_, stride_)) do (sz, p, s)
        WindowDims(sz, s, p, p, 1, 1, false)
    end...)
    rdims = NNlib.pdims(size(x), k, NNlib.expand(Val{length(k)}, pad),
                NNlib.expand(Val{length(k)}, stride))
    HloReduceWindow{eltype(x), rdims}(
        max,
        windows
    )(x)
end

function Base.reshape(A::XRTArray, dims::Tuple{Vararg{Union{Int,Colon}}})
    dims = Base._reshape_uncolon(A, dims)
    HloReshape{eltype(A), dims}(
        tuple((0:ndims(A)-1)...), dims
    )(A)
end
