using Revise

using XLA
using Metalhead
using Flux
using TensorFlow
using Zygote

# For now. Media.jl has a bug
pop!(Base.Multimedia.displays)

# Connect to the TensorFlow distributed session (either xrt_server or TPU)
sess = Session(Graph(); target="grpc://localhost:8470")

# Filter Dropout layers for now (the data-dependent condition is slightly
# harder to compile).
ic = ImmutableChain(filter(x->!isa(x, Flux.Dropout), VGG19().layers.layers)...)
xrt(ic) = Flux.mapleaves(x->isa(x, AbstractArray) ? XRTArray(x) : x, ic)
xrtic = xrt(ic)

# Note: Most of this boiler plate will go away. Also we'll have better handling
# of sclaras so the loop below will look prettier. For now, just something that works.
function my_derivative(f, args...)
  y, back = Zygote._forward(Zygote.Context{Nothing}(nothing), f, args...)
  J = Δ -> Zygote.tailmemaybe(back(Δ))
  return J(1f0)[1]
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

#update_params(xrtic, updates, η) = xrtic # TODO: Here, return the model with updates applied
function epoch_loop(::Val{batch_size}, xrtic, nbatches, η) where {batch_size}
    while nbatches > XRTArray(0)
        (x, y), _ = XLA.HloInfeed(Tuple{XRTArray{Float32, (224, 224, 3, batch_size), 4},
                                   XRTArray{Float32, (1000, batch_size), 2}})(XLA.HloAfterAll()())
        updates = my_derivative(xrtic->crossentropy(xrtic(x), y), xrtic)
        xrtic = update_params(xrtic, updates, η)
        nbatches -= XRTArray(1)
    end
    return xrtic
end

#@tpu_compile epoch_loop(Val(20), xrtic, XRTArray(20), XRTArray(0.1f0))

compld = @tpu_compile epoch_loop(Val(20), xrtic, XRTArray(1), XRTArray(0.1f0))

ic_alloc = XRTAllocation(sess, XLA.struct_to_literal(xrtic))

nbatches = 25
GC.gc()
t = @async XLA.run_async(compld, ic_alloc, XLA.gethandle!(sess, XRTArray(nbatches)), XLA.gethandle!(sess, XRTArray(0.1f0)))
GC.gc()

const tf = TensorFlow
let n = 0
    global infeed
    function infeed(tup::NTuple{N, AbstractArray} where N)
        placeholders = [tf.placeholder(eltype(el), shape=size(el)) for el in tup]
        n += 1
        desc = tf.NodeDescription(sess.graph, "InfeedEnqueueTuple", "InfeedEnqueueTuple$(n)")
        tf.set_attr_shape_list(desc, "shapes", Vector{Int64}[collect(size(x)) for x in tup])
        tf.set_attr_list(desc, "dtypes", DataType[eltype(x) for x in tup])
        desc["device_ordinal"] = 0
        tf.add_input(desc, placeholders)
        eq = tf.Tensor(tf.Operation(desc));
        arr = rand(Float32, 10, 10)
        run(sess, eq, Dict((x=>y for (x, y) in zip(placeholders, tup))...))
    end
end

t

for i = 1:nbatches
    @infeed "Feeding batch $i"
    infeed((rand(Float32, 224, 224, 3, 20), rand(Float32, 1000, 20)))
end
