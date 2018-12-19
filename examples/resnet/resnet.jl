using Flux
using Flux: @treelike

struct ResidualBlock{L,S}
  layers::L
  shortcut::S
end

struct ConvNorm{C, N}
  conv::C
  norm::N
end
(cn::ConvNorm)(value) = cn.norm(cn.conv(value))

@treelike ResidualBlock
@treelike ConvNorm

function ResidualBlock(filters, kernels::Array{Tuple{Int,Int}}, pads::Array{Tuple{Int,Int}}, strides::Array{Tuple{Int,Int}}, shortcut = identity)
  local layers = []
  for i in 2:length(filters)
    push!(layers, ConvNorm(
        Conv(kernels[i-1], filters[i-1]=>filters[i], pad = pads[i-1], stride = strides[i-1]),
        BatchNorm(filters[i])))
  end
  ResidualBlock(Tuple(layers),shortcut)
end

function ResidualBlock(filters, kernels::Array{Int}, pads::Array{Int}, strides::Array{Int}, shortcut = identity)
  @assert length(filters) == 4
  ResidualBlock(filters, [(i,i) for i in kernels], [(i,i) for i in pads], [(i,i) for i in strides], shortcut)
end

# This is a hack, to avoid Zygote's trouble with destructuring
rb_apply(t::Tuple{Any}, value, shortcut) = relu.(first(t)(value) + shortcut)
rb_apply(t::Tuple, value, shortcut) = rb_apply(Base.tail(t), relu.(first(t)(value)), shortcut)
function (block::ResidualBlock)(input)
  rb_apply(block.layers, copy(input), block.shortcut(input))
end

function Bottleneck(filters::Int, downsample::Bool = false, res_top::Bool = false)
  if(!downsample && !res_top)
    return ResidualBlock([4 * filters, filters, filters, 4 * filters], [1,3,1], [0,1,0], [1,1,1])
  elseif(downsample && res_top)
    return ResidualBlock([filters, filters, filters, 4 * filters], [1,3,1], [0,1,0], [1,1,1], Chain(Conv((1,1), filters=>4 * filters, pad = (0,0), stride = (1,1)), BatchNorm(4 * filters)))
  else
    shortcut = Chain(Conv((1,1), 2 * filters=>4 * filters, pad = (0,0), stride = (2,2)), BatchNorm(4 * filters))
    return ResidualBlock([2 * filters, filters, filters, 4 * filters], [1,3,1], [0,1,0], [1,1,2], shortcut)
  end
end

function resnet50()
  local layers = [3, 4, 6, 3]
  local layer_arr = []

  push!(layer_arr, Conv((7,7), 3=>64, pad = (3,3), stride = (2,2)))
  push!(layer_arr, x -> maxpool(x, (3,3), pad = (1,1), stride = (2,2)))

  initial_filters = 64
  for i in 1:length(layers)
    push!(layer_arr, Bottleneck(initial_filters, true, i==1))
    for j in 2:layers[i]
      push!(layer_arr, Bottleneck(initial_filters))
    end
    initial_filters *= 2
  end

  push!(layer_arr, x -> meanpool(x, (7,7)))
  push!(layer_arr, x -> reshape(x, :, size(x,4)))
  push!(layer_arr, (Dense(2048, 1000)))

  # Disable softmax, use logitcrossentropy instead
  #push!(layer_arr, softmax)

  Chain(layer_arr...)
end
