using Zygote
using Statistics

# We modify (the implementation of) batchnorm to be more ammenable to TPUs
struct TPUBatchNorm{F,V,W}
   λ::F  # activation function
   β::V  # bias
   γ::V  # scale
   μ::W  # moving mean
   σ::W  # moving std
   ϵ::HLOArray{Float32, (), 0}
   momentum::HLOArray{Float32, (), 0}
   #active::Bool
end

#Flux.children(BN::TPUBatchNorm) =
#  (BN.λ, BN.β, BN.γ, BN.μ, BN.σ, BN.ϵ, BN.momentum)

#Flux.mapchildren(f, BN::TPUBatchNorm) =  # e.g. mapchildren(cu, BN)
#  TPUBatchNorm(BN.λ, f(BN.β), f(BN.γ), f(BN.μ), f(BN.σ), BN.ϵ, BN.momentum #=, BN.active=#)

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
@Zygote.adjoint compute_affine_shape(x) = compute_affine_shape(x), Δ->(nothing,)

# This is a bit of a trick. We use Zygote's backward pass infrastructure
# to backpropagate the batchnorm statistics to the parameter update
function tpu_batchnorm_statistics(active::Bool, bn_μ, bn_σ, bn_ε, x)
  #@assert !BN.active
  affine_shape = compute_affine_shape(x)
  μ = reshape(bn_μ, affine_shape...)
  σ = reshape(bn_σ, affine_shape...)
  return (x .- μ)./σ
end

@Zygote.adjoint function tpu_batchnorm_statistics(active::Bool, bn_μ, bn_σ, bn_ϵ, x)
  ϵ = convert(eltype(x), bn_ϵ)
  axes = (1, 2, 4)
  m = Int32(size(x, 1) * size(x, 2) * size(x, 4))

  # Calculate μ and σ for the "forward pass"
  μ = mean(x, dims = axes)
  meansub = (x .- μ)
  sq = meansub .* meansub
  σ = sqrt.(mean(sq, dims = axes) .+ ϵ)

  x̂ = (x .- μ)./σ
  # Return "forward pass" calculation and closure that returns our "gradients"
  x̂, let μ_dropped=dropdims(μ, dims = axes),
         σ_dropped=dropdims(σ, dims = axes) .* convert(eltype(x), m) / convert(eltype(x), m - 1)
    function(Δ)
      let axes = (1, 2, 4), x̂ = x̂
        # https://kratzert.github.io/2016/02/12/understanding-the-gradient-flow-through-the-batch-normalization-layer.html
        Δ_x = (Δ .- mean(Δ, dims=axes) .- x̂ .* mean(Δ .* x̂, dims=axes))./σ
        return (nothing, μ_dropped, σ_dropped, nothing, Δ_x)
      end
    end
  end
end

function (BN::TPUBatchNorm)(x)
  γ = BN.γ
  β = BN.β
  affine_shape = compute_affine_shape(x)

  x̂ = tpu_batchnorm_statistics(true, BN.μ, BN.σ, BN.ϵ, x)

  let λ = BN.λ
    λ.(reshape(γ, affine_shape...) .* x̂ .+ reshape(β, affine_shape...))
  end
end

function update_params(BN::TPUBatchNorm{F,V,W}, updates, state, η, β) where {F,V,W}
    mtm = BN.momentum
    new_λ, new_λ_state = update_params(BN.λ, updates.λ, state[1], η, β)
    new_β, new_β_state = update_params(BN.β, updates.β, state[2], η, β)
    new_γ, new_γ_state = update_params(BN.γ, updates.γ, state[3], η, β)

    new_BN = TPUBatchNorm{F,V,W}(
      new_λ,
      new_β,
      new_γ,
      (HLOArray(1f0) - mtm) .* BN.μ + mtm .* updates.μ,
      (HLOArray(1f0) - mtm) .* BN.σ + mtm .* updates.σ,
      BN.ϵ, mtm #, BN.active
    )
    new_state = tuple(
      new_λ_state,
      new_β_state,
      new_γ_state,
      state[4],
      state[5],
      state[6],
      state[7],
    )
    return (new_BN, new_state)
end
