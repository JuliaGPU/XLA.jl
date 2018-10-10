using Flux

struct ImmutableChain{T<:Tuple}
    layers::T
end


ImmutableChain(x, y, xs...) = ImmutableChain((x, y, xs...))

Flux.children(c::ImmutableChain) = c.layers
Flux.mapchildren(f, c::ImmutableChain) = ImmutableChain(f.(c.layers))

applyall(::Tuple{}, x) = x
applyall(fs::Tuple, x) = applyall(Base.tail(fs), first(fs)(x))

(c::ImmutableChain)(x) = applyall(c.layers, x)
