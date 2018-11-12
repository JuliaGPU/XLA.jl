using Revise

using XLA
using TensorFlow
using Metalhead
using Flux
using Zygote
using Statistics

include("resnet.jl")
include("tpu_batch_norm.jl")

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

function epoch_loop(::Val{batch_size}, xrtic, nbatches, η) where {batch_size}
    while nbatches > XRTArray(0)
        # N.B: Ideally we'd have the labels be UInt16, but that doesn't work yet on TPUs
        (x, labels), _ = XLA.HloInfeed(Tuple{XRTArray{Float32, (224, 224, 3, batch_size), 4},
                                        XRTArray{UInt32, (batch_size,), 1}})(XLA.HloAfterAll()())
        y = make_one_hot(labels)
        loss, updates = my_derivative_with_loss(xrtic->crossentropy(xrtic(x), y), xrtic)
        updates = unflatten_tuple(updates,
            XLA.HloCrossReplicaSum{typeof(+)}((), 0, "")(+, flatten_tuple(updates)...))
        xrtic = update_params(xrtic, updates, η)
        XLA.HloOutfeed()((loss,), XLA.HloAfterAll()())
        nbatches -= XRTArray(1)
    end
    return xrtic
end

sess = Session(Graph(); target="grpc://localhost:8470")

batch_size = 128
nbatches = 10_000
compld = @tpu_compile epoch_loop(Val(batch_size), resnet, XRTArray(nbatches), XRTArray(0.1f0))


models = Any[]
for i = 0:7
    with_device("/job:tpu_worker/replica:1/task:1/device:TPU:$(i+1)") do
        ic_alloc = XRTRemoteStruct(sess, resnet)
        push!(models, ic_alloc)
    end
end

using ProtoBuf
all_ops = Any[]
all_allocs = Any[]
feed = Dict()
for i = 0:7
    with_device("/job:tpu_worker/replica:1/task:1/device:TPU:$(i+1)") do
        ic_alloc = models[i+1]
        global compld
        config=XLA.XRTExecutionConfig(device_ordinal=i)
        iob = PipeBuffer();
        writeproto(iob, config)
        str = String(take!(iob))
        pc = placeholder(String)
        feed[pc] = str
        params = (XRTArray(nbatches), XRTArray(0.1f0))
        append!(all_allocs, params)
        push!(all_ops, TensorFlow.Ops.xrt_execute(compld.h, pc, Int64[
            XLA.gethandle!(sess, ic_alloc).h,
            XLA.gethandle!(sess, params[1]).h,
            XLA.gethandle!(sess, params[2]).h
        ]))
    end
end
t = @async run(sess, all_ops, feed; async=true)

let infeed_ops = map(0:7) do i
        with_device("/job:tpu_worker/replica:1/task:1/device:TPU:$(i+1)") do
            XLA._make_infeed_op(sess,
                (Float32, UInt32), ((224, 224, 3, batch_size), (1000, batch_size)),
                [TensorFlow.Ops.zeros(Tensor{Float32}, (224, 224, 3, batch_size)),
                 TensorFlow.Ops.zeros(Tensor{UInt32}, (1000, batch_size))];
                device = TensorFlow.Device("/job:tpu_worker/replica:1/task:1/device:TPU:$(i+1)"),
                device_ordinal = i)
        end
    end, outfeed_ops = map(0:7) do i
        with_device("/job:tpu_worker/replica:1/task:1/device:TPU:$(i+1)") do
            XLA.make_outfeed_op(sess,
                Tuple{XRTArray{Float32, (), 0}};
                device = TensorFlow.Device("/job:tpu_worker/replica:1/task:1/device:TPU:$(i+1)"),
                device_ordinal = i)
        end
    end
    global infeed_zeros
    global outfeed_losses
    global sess
    function infeed_zeros()
        run(sess, infeed_ops; async=true)
    end
    function outfeed_losses()
        run(sess, outfeed_ops)
    end
end

for i = 1:nbatches
    @info "Feeding batch $i"
    infeed_zeros()
    @show outfeed_losses()
end
