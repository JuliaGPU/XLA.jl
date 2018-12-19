using Zygote
using Statistics

# We modify (the implementation of) batchnorm to be more ammenable to TPUs.
struct TPUBatchNorm{F,V,W}
   λ::F  # activation function
   β::V  # bias
   γ::V  # scale
   μ::W  # moving mean
   σ::W  # moving std
   ϵ::XRTArray{Float32, (), 0}
   momentum::XRTArray{Float32, (), 0}
   #active::Bool
end

Flux.children(BN::TPUBatchNorm) =
  (BN.λ, BN.β, BN.γ, BN.μ, BN.σ, BN.ϵ, BN.momentum)

Flux.mapchildren(f, BN::TPUBatchNorm) =  # e.g. mapchildren(cu, BN)
  TPUBatchNorm(BN.λ, f(BN.β), f(BN.γ), f(BN.μ), f(BN.σ), BN.ϵ, BN.momentum #=, BN.active=#)

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
  #@assert !BN.active
  affine_shape = compute_affine_shape(x)
  μ = reshape(bn_μ, affine_shape...)
  σ = reshape(bn_σ, affine_shape...)
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

  tup = batchnorm_statistics(true, BN.μ, BN.σ, BN.ϵ, x)
  μ = first(tup)
  σ = first(Base.tail(tup))

  let λ = BN.λ
    λ.(reshape(γ, affine_shape...) .* ((x .- μ) ./ σ) .+ reshape(β, affine_shape...))
  end
end

function update_params(BN::TPUBatchNorm{F,V,W}, updates, η) where {F,V,W}
    mtm = BN.momentum
    TPUBatchNorm{F,V,W}(
      update_params(BN.λ, updates.λ, η),
      update_params(BN.β, updates.β, η),
      update_params(BN.γ, updates.γ, η),
      (XRTArray(1f0) - mtm) .* BN.μ + mtm .* updates.μ,
      (XRTArray(1f0) - mtm) .* BN.σ + mtm .* updates.σ,
      BN.ϵ, mtm #, BN.active
    )
end

map_to_tpu(x::Flux.BatchNorm) = TPUBatchNorm(map(map_to_tpu, Flux.children(x))[1:end-1]...)
