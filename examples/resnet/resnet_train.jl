using Revise

using XLA
using TensorFlow
using Metalhead
using Flux
using Zygote
using Statistics

include("resnet.jl")

# We modify (the implementation of) batchnorm to be more ammenable to TPUs.
struct TPUBatchNorm{F,V,W}
   λ::F  # activation function
   β::V  # bias
   γ::V  # scale
   μ::W  # moving mean
   σ::W  # moving std
   ϵ::XRTArray{Float32, (), 0}
   momentum::XRTArray{Float32, (), 0}
   active::Bool
end

Flux.children(BN::TPUBatchNorm) =
  (BN.λ, BN.β, BN.γ, BN.μ, BN.σ, BN.ϵ, BN.momentum, BN.active)

Flux.mapchildren(f, BN::TPUBatchNorm) =  # e.g. mapchildren(cu, BN)
  TPUBatchNorm(BN.λ, f(BN.β), f(BN.γ), f(BN.μ), f(BN.σ), BN.ϵ, BN.momentum, BN.active)

function compute_affine_shape(x)
  dims = length(size(x))
  channels = size(x, dims-1)
  affine_shape = let dims=dims, channels=channels
   ntuple(i->i == dims - 1 ? channels : 1, dims)
  end
  #=
  m = let sz = size(x)
    prod(ntuple(i->sz[i], dims-2)) * sz[end]
  end
  =#
  affine_shape
end
@Zygote.grad compute_affine_shape(x) = compute_affine_shape(x), Δ->(nothing,)

# This is a bit of a trick. We use Zygote's backward pass infrastructure
# to backpropagate the batchnorm statistics to the parameter update
function batchnorm_statistics(active::Bool, bn_μ, bn_σ, bn_ε, x)
  @assert !BN.active
  affine_shape = compute_affine_shape(x)
  μ = reshape(BN.μ, affine_shape...)
  σ = reshape(BN.σ, affine_shape...)
  (μ, σ)
end

@Zygote.grad function batchnorm_statistics(active::Bool, bn_μ, bn_σ, bn_ϵ, x)
  ϵ = convert(eltype(x), bn_ϵ)
  axes = (1, 2, 4)
  μ = mean(x, dims = axes)
  meansub = (x .- μ)
  sq = meansub .* meansub
  σ = sqrt.(mean(sq, dims = axes) .+ ϵ)
  m = let sz = size(x), dims = length(size(x))
    prod(ntuple(i->sz[i], dims-2)) * sz[end]
  end
  (μ, σ), let μ=dropdims(μ, dims = axes),
              σ=dropdims(σ, dims = axes) .* XRTArray(convert(eltype(x), m) / convert(eltype(x), m - 1))
    function(Δ)
      (nothing, μ, σ, nothing, nothing)
    end
  end
end


function (BN::TPUBatchNorm)(x)
  γ = BN.γ
  β = BN.β
  affine_shape = compute_affine_shape(x)

  tup = batchnorm_statistics(BN.active, BN.μ, BN.σ, BN.ϵ, x)
  μ = first(tup)
  σ = first(Base.tail(tup))

  let λ = BN.λ
    λ.(reshape(γ, affine_shape...) .* ((x .- μ) ./ σ) .+ reshape(β, affine_shape...))
  end
end

map_to_tpu(x::Bool) = x
map_to_tpu(x::Real) = XRTArray(convert(Float32, x))
map_to_tpu(x::Chain) = ImmutableChain(map(map_to_tpu, x.layers)...)
map_to_tpu(x::AbstractArray) = XRTArray(Float32.(Flux.data(x)))
map_to_tpu(x::Flux.BatchNorm) = TPUBatchNorm(
  map(map_to_tpu, Flux.children(x))...)
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

function update_params(BN::TPUBatchNorm{F,V,W}, updates, η) where {F,V,W}
    mtm = BN.momentum
    TPUBatchNorm{F,V,W}(
      update_params(BN.λ, updates.λ, η),
      update_params(BN.β, updates.β, η),
      update_params(BN.γ, updates.γ, η),
      (XRTArray(1f0) - mtm) .* BN.μ + mtm .* updates.μ,
      (XRTArray(1f0) - mtm) .* BN.σ + mtm .* updates.σ,
      BN.ϵ, mtm, BN.active
    )
end


function make_one_hot(data)
    operand = zero(XRTArray{Float32, (1000, length(data), 2})
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
        xrtic = update_params(xrtic, updates, η)
        XLA.HloOutfeed()((loss,), XLA.HloAfterAll()())
        nbatches -= XRTArray(1)
    end
    return xrtic
end

sess = Session(Graph(); target="grpc://localhost:8470")
compld = @tpu_compile epoch_loop(Val(20), resnet, XRTArray(1), XRTArray(0.1f0))


function evaluate2(xrtic, x)
    loss, updates = my_derivative_with_loss(xrtic->sum(xrtic(x)), xrtic)
end

function bisect(n)
    start = ImmutableChain((resnet_host.layers[1:n]...,))
    y = start(rand(Float32, 224, 224, 3, 1))
    remainder = ImmutableChain((resnet.layers[n+1:end]...,))
    XLA.code_typed_xla(evaluate2, Tuple{typeof(remainder), typeof(XRTArray(y))})[1]
end

function bisect_one(n)
    start = ImmutableChain((resnet_host.layers[1:n]...,))
    y = start(rand(Float32, 224, 224, 3, 1))
    remainder = resnet.layers[n+1]
    XLA.code_typed_xla(evaluate2, Tuple{typeof(remainder), typeof(XRTArray(y))})[1]
end


function bisect_one_weak(n)
    start = ImmutableChain((resnet_host.layers[1:n]...,))
    y = start(rand(Float32, 224, 224, 3, 1))
    remainder = resnet.layers[n+1]
    code_typed(evaluate2, Tuple{typeof(remainder), typeof(XRTArray(y))}; params = Core.Compiler.CustomParams(typemax(UInt); aggressive_constant_propagation=true, ignore_all_inlining_heuristics=true))
end


function bisect_forward(n)
    start = ImmutableChain((resnet_host.layers[1:n]...,))
    y = start(rand(Float32, 224, 224, 3, 1))
    remainder = ImmutableChain((resnet.layers[n+1:end]...,))
    XLA.code_typed_xla(remainder, Tuple{typeof(XRTArray(y))})[1]
end


function bisect_residual(b, n)
    start = ImmutableChain((resnet_host.layers[1:b-1]...,))
    y = start(rand(Float32, 224, 224, 3, 1))
    host_block = resnet_host.layers[b]
    host_start = ImmutableChain((host_block.layers[1:n]...,))
    y = host_start(y)
    remainder = ImmutableChain((resnet.layers[b].layers[n+1:end]...,))
    XLA.code_typed_xla(evaluate2, Tuple{typeof(remainder), typeof(XRTArray(y))})[1]
end

bisect_residual(18, 2)
