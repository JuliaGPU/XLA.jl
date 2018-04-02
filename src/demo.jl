# Load dataset
using Flux, Flux.Data.MNIST
using Flux: onehotbatch, argmax, crossentropy, throttle
using Base.Iterators: repeated, partition
# using CuArrays

# Classify MNIST digits with a convolutional network

imgs = MNIST.images()

labels = onehotbatch(MNIST.labels(), 0:9)

using Flux

struct Chain2{T, S}
   x::T
   y::S
end
(c::Chain2)(arg) = c.y(c.x(arg))

m = Chain2(
  Dense(28^2, 32, relu),
  Dense(32, 10))

xf = Float32.(reshape(imgs[1],:))

m(xf)

@info "Running MLP inference via XLA"
@xla m(xf)
