using Zygote
using Statistics

# We modify (the implementation of) batchnorm to be more ammenable to CPUs pretending to be TPUs.
struct CPUBatchNorm{F,V,W}
   λ::F  # activation function
   β::V  # bias
   γ::V  # scale
   μ::W  # moving mean
   σ::W  # moving std
   ϵ::Float32
   momentum::Float32
   #active::Bool
end

Flux.children(BN::CPUBatchNorm) =
  (BN.λ, BN.β, BN.γ, BN.μ, BN.σ, BN.ϵ, BN.momentum)

Flux.mapchildren(f, BN::CPUBatchNorm) =  # e.g. mapchildren(cu, BN)
  CPUBatchNorm(BN.λ, f(BN.β), f(BN.γ), f(BN.μ), f(BN.σ), BN.ϵ, BN.momentum #=, BN.active=#)

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
function batchnorm_statistics(active::Bool, bn_μ, bn_σ, bn_ε, x)
  #@assert !BN.active
  affine_shape = compute_affine_shape(x)
  μ = reshape(bn_μ, affine_shape...)
  σ = reshape(bn_σ, affine_shape...)
  return (x .- μ)./σ
end

@Zygote.adjoint function batchnorm_statistics(active::Bool, bn_μ, bn_σ, bn_ϵ, x)
  ϵ = convert(eltype(x), bn_ϵ)
  axes = (1, 2, 4)
  m = prod(size.(Ref(x), axes))

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
      # calculate sensitivities on x, first as contributions from μ, then σ:
      return (nothing, μ_dropped, σ_dropped, nothing, 1 ./ σ .*(Δ .- mean(Δ, dims=axes) .- x̂ .* mean(Δ .* x̂, dims=axes)))
    end
  end
end

function (BN::CPUBatchNorm)(x)
  γ = BN.γ
  β = BN.β
  affine_shape = compute_affine_shape(x)

  x̂ = batchnorm_statistics(true, BN.μ, BN.σ, BN.ϵ, x)

  let λ = BN.λ
    λ.(reshape(γ, affine_shape...) .* x̂ .+ reshape(β, affine_shape...))
  end
end

function update_params(BN::CPUBatchNorm{F,V,W}, updates, η) where {F,V,W}
    mtm = BN.momentum
    CPUBatchNorm{F,V,W}(
      update_params(BN.λ, updates.λ, η),
      update_params(BN.β, updates.β, η),
      update_params(BN.γ, updates.γ, η),
      (1f0 - mtm) .* BN.μ + mtm .* updates.μ,
      (1f0 - mtm) .* BN.σ + mtm .* updates.σ,
      BN.ϵ, mtm #, BN.active
    )
end
