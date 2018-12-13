using Revise

using XLA
using TensorFlow
using Metalhead
using Flux
using Zygote
using Statistics

include("resnet.jl")
include("tpu_batch_norm.jl")

# Model zero initialization
zero_init(::Type{ImmutableChain{T}}) where {T} = ImmutableChain(zero_init(T))
@Base.pure @noinline fc(T) = fieldcount(T)
@generated function zero_init(::Type{T}) where {T<:Tuple}
    Expr(:tuple, (:(zero_init($(fieldtype(T, i)))) for i = 1:fieldcount(T))...)
end
zero_init(::Type{Conv{F,A,V,Stride,Pad,Dilation}}) where {F,A,V,Stride,Pad,Dilation} =
    Conv{F,A,V,Stride,Pad,Dilation}(zero_init(F), zero_init(A), zero_init(V))
zero_init(T::Type{<:XRTArray}) = zero(T)
@noinline not_defined_error(T) = error("Not defined for $T")
function zero_init(::Type{T}) where {T}
    isdefined(T, :instance) || not_defined_error(T)
    T.instance
end
zero_init(::Type{ResidualBlock{L, S}}) where {L, S} = ResidualBlock(zero_init(L), zero_init(S))
zero_init(::Type{ConvNorm{C, N}}) where {C, N} = ConvNorm(zero_init(C), zero_init(N))
zero_init(::Type{Dense{F, S, T}}) where {F, S, T} = Dense(zero_init(S), zero_init(T), zero_init(F))

zero_init(::Type{TPUBatchNorm{F,V,W}}) where {F,V,W} = TPUBatchNorm(
    zero_init(F), zero_init(V), zero_init(V),
    zero_init(W), zero_init(W),
    zero(XRTArray{Float32, (), 0}),
    zero(XRTArray{Float32, (), 0}))

map_to_tpu(x::Bool) = x
map_to_tpu(x::Real) = XRTArray(convert(Float32, x))
map_to_tpu(x::Chain) = ImmutableChain(map(map_to_tpu, x.layers)...)
map_to_tpu(x::AbstractArray) = XRTArray(Float32.(Flux.data(x)))
map_to_tpu(x) = Flux.mapchildren(map_to_tpu, x)
resnet_host = Flux.mapleaves(x->isa(x, AbstractArray) ? Float32.(Flux.data(x)) : x, resnet50())
resnet = map_to_tpu(resnet_host)

function my_derivative_with_loss(f, args...)
  y, back = Zygote._forward(Zygote.Context{Nothing}(nothing), f, args...)
  J = Δ -> Zygote.tailmemaybe(back(Δ))
  return y, J(1f0)[1]
end

function crossentropy(ŷ::AbstractVecOrMat, y::XRTArray{<:Any, Sz}) where {Sz}
    weight = 1f0 # one(eltype(y))
    z = sum(y .* log.(ŷ) .* weight)
    return z / XRTArray(1f0)
end

update_params(model::XRTArray, updates::XRTArray, η) = model + updates * η
update_params(model, updates::Nothing, η) = model
function update_params(model, updates, η)
    new_fields = ntuple(Val(nfields(model))) do i
        update_params(getfield(model, i), getfield(updates, i), η)
    end
    isa(model, Tuple) ?
        tuple(new_fields...) :
        typeof(model)(new_fields...)
end

# Flatten and unflatten tuples to sum everything in one cross-replica sum operation
#=
# TODO: It would be nice to write it like this, but that currently doesn't work
flatten_tuple(a::Nothing, args...) = flatten_tuple(args...)
flatten_tuple(a::Tuple, args...) = tuple(flatten_tuple(a...)..., flatten_tuple(args...)...)
flatten_tuple(a::NamedTuple, args...) = tuple(flatten_tuple(a...)..., flatten_tuple(args...)...)
flatten_tuple(a, args...) = tuple(a, flatten_tuple(args...)...)
flatten_tuple() = tuple()

unflatten_tuple(pattern::Nothing, tup) = (nothing, tup)
unflatten_tuple(pattern::Tuple{}, tup) = ((), tup)
function unflatten_tuple(pattern::Tuple, tup)
    head, tup = unflatten_tuple(first(pattern), tup)
    tail, tup = unflatten_tuple(Base.tail(pattern), tup)
    ((head, tail...), tup)
end
function unflatten_tuple(pattern::NamedTuple, tup)
    head, tup = unflatten_tuple(first(pattern), tup)
    tail, tup = unflatten_tuple(Base.tail(pattern), tup)
    ((;first(keys(pattern))=>head, tail...), tup)
end
unflatten_tuple(a, tup) = (first(tup), Base.tail(tup))
=#

@generated function flatten_tuple(t)
    stmts = Any[]
    tuple_args = Any[]
    function collect_args(t, sym)
        for i = 1:fieldcount(t)
            fT = fieldtype(t, i)
            if fT === Nothing
                continue
            end
            g = gensym()
            push!(stmts, :($g = getfield($sym, $i)))
            if fT <: Tuple || fT <: NamedTuple
                collect_args(fT, g)
            else
                push!(tuple_args, g)
            end
        end
    end
    collect_args(t, :t)
    quote
        $(stmts...)
        tuple($(tuple_args...))
    end
end

@generated function unflatten_tuple(pattern, t)
    function expand_arg(pattern, idx)
        fields = Any[]
        for i = 1:fieldcount(pattern)
            fT = fieldtype(pattern, i)
            if fT === Nothing
                push!(fields, nothing)
                continue
            end
            if fT <: Tuple || fT <: NamedTuple
                arg, idx = expand_arg(fT, idx)
                push!(fields, arg)
            else
                push!(fields, Expr(:call, :getfield, :t, idx))
                idx += 1
            end
        end
        Expr(:new, pattern, fields...), idx
    end
    res = expand_arg(pattern, 1)[1]
    res
end

function make_one_hot(data)
    operand = zero(XRTArray{Float32, (1000, length(data)), 2})
    updates = XLA.HloBroadcast((), (1, length(data)))(XRTArray(1f0))
    update_computation = (x,y)->y
    scatter_indices = hcat(XRTArray(0:(length(data)-1)), data)
    sdims = XLA.ScatterDimNums(
        #= update_window_dims =# (0,),
        #= inserted_window_dims =# (0,),
        #= scatter_dims_to_operand_dims =# (1, 0),
        #= index_vector_dim =# 1
    )
    XLA.HloScatter{typeof(update_computation)}(sdims)(
            update_computation,
            operand,
            scatter_indices,
            updates
        )
end

function epoch_loop(::Val{batch_size}, ::Type{xrticT}) where {batch_size, xrticT}
    nbatches = XRTArray(2000)
    η = XRTArray(0.1f0)
    xrtic = zero_init(xrticT)
    while nbatches > XRTArray(0)
        # N.B: Ideally we'd have the labels be UInt16, but that doesn't work yet on TPUs
        (x,), _ = XLA.HloInfeed(Tuple{XRTArray{Float32, (224*224*3*batch_size,), 1}})(XLA.HloAfterAll()())
        labels = zero(XRTArray{Int32, (batch_size,), 1})
        loss, updates = let x = reshape(x, (224, 224, 3, batch_size)),
                            y = make_one_hot(labels)
            my_derivative_with_loss(xrtic->crossentropy(xrtic(x), y), xrtic)
        end
        updates = unflatten_tuple(updates,
            XLA.HloCrossReplicaSum{typeof(+)}((), 0, "")(+, flatten_tuple(updates)...))
        xrtic = update_params(xrtic, updates, η)
        XLA.HloOutfeed()((loss,), XLA.HloAfterAll()())
        nbatches -= XRTArray(1)
    end
    return xrtic
end

batch_size = 128

(ir, sv) = @time XLA.code_typed_xla(epoch_loop, Tuple{Val{batch_size}, Type{typeof(resnet)}})
@time XLA.compile_to_xla(ir, sv)
(ir, sv) = @time XLA.code_typed_xla(epoch_loop, Tuple{Val{batch_size}, Type{typeof(resnet)}})
@time XLA.compile_to_xla(ir, sv)
