# This file contains shape transfer functions for HLO ops.
# It is lifted into the type domain by a type assertion in execute.jl,
# as well as being used in the compiler.

# HLO ops with specified type and dimensions
function shape_infer(op::Union{GenericHloOp, HloParameter, HloRng, HloConstant},
                     args::Type{<:XRTArray}...)
    return (op.T, op.shape)
end

eltypes_match(_::Type{<:XRTArray}) = true
eltypes_match(a1::Type{<:XRTArray}, a2::Type{<:XRTArray}, args::Type{<:XRTArray}...) =
    eltype(a1) == eltype(a2) && eltypes_match(a2, args...)

shapes_match(_::Type{<:XRTArray}) = true
shapes_match(a1::Type{<:XRTArray}, a2::Type{<:XRTArray}, args::Type{<:XRTArray}...) =
    size(a1) == size(a2) && eltypes_match(a2, args...)

function shape_infer(op::Union{HloBroadcast, HloReshape},
                     args::Type{<:XRTArray}...)
    eltypes_match(args...) || error("eltype mismatch in $(typeof(op))")
    (eltype(args[1]), op.result_shape)
end

@Base.pure drop_reduce_dims(shape, dims) = tuple((shape[i] for i in 1:length(shape) if !(i-1 in dims))...)

function shape_infer(op::HloReduce{fT}, ::Type{<:fT},
                     args::Type{<:XRTArray}...) where {fT}
    # For now, restrict to a single reduction array
    @assert length(args) == 2
    shapes_match(args[1:div(length(args), 2)]...) || error("shape mismatch in $(typeof(op))")
    # For now, assume reductions preserve type
    (eltype(args[1]), drop_reduce_dims(size(args[1]), op.dims))
end

dilated_bound(bound, dilation) = (bound - 1) * dilation + 1
function strided_bound(bound, window_size, stride)
    (bound == 0 || window_size > bound) && return 0
    return div(bound - window_size, stride) + 1
end

function infer_window_output_shape(base_shape, windows)
    ntuple(length(windows)) do i
        window = windows[i]
        dilated_base = dilated_bound(base_shape[i], window.base_dilation)
        padded_dilated_base = window.padding_low + dilated_base + window.padding_high
        dilated_window = dilated_bound(window.size, window.window_dilation)
        return strided_bound(padded_dilated_base, dilated_window, window.stride)
    end
end

function shape_infer(op::HloReduceWindow{fT}, ::Type{<:fT},
                     args::Type{<:XRTArray}...) where {fT}
    # For now, assume reductions preserve type
    (eltype(args[1]), infer_window_output_shape(size(args[1]), op.window))
end

function updims(s)
    dims = collect(map(x->x+1, s))
    ndims(dims) == 0 ? dims[] : dims
end

@Base.pure function shape_infer_conv(op::HloConv, lhs::Type{<:XRTArray}, kernel::Type{<:XRTArray})
    input_spatial_dims = size(lhs)[updims(op.dims.input_spatial_dimensions)]
    window_output_shape = infer_window_output_shape(input_spatial_dims, op.window)
    dimensions = Vector(undef, length(window_output_shape) + 2)
    dimensions[updims(op.dims.output_spatial_dimensions)] .= window_output_shape
    dimensions[updims(op.dims.output_batch_dimension)] =
        size(lhs)[updims(op.dims.input_batch_dimension)]
    dimensions[updims(op.dims.output_feature_dimension)] =
        size(kernel)[updims(op.dims.kernel_output_feature_dimension)]
    tuple(dimensions...)
end

function shape_infer(op::HloConv,
                     lhs::Type{<:XRTArray}, kernel::Type{<:XRTArray})
    shape = shape_infer_conv(op, lhs, kernel)
    # For now, assume convolution preserve type
    (eltype(lhs), shape)
end

@Base.pure function compute_dot_dimensions(op::HloDot, lhs::Type{<:XRTArray}, rhs::Type{<:XRTArray})
    lcd = op.dims.lhs_contracting_dimensions[1]
    rcd = op.dims.rhs_contracting_dimensions[1]
    rbatch = op.dims.rhs_batch_dimensions
    dimensions = Int[]
    append!(dimensions, size(lhs)[filter(x->x != lcd+1, 1:ndims(lhs))])
    append!(dimensions, size(rhs)[filter(x->x != rcd+1 && !(x-1 ∈ rbatch), 1:ndims(rhs))])
    tuple(dimensions...)
end


function shape_infer(op::HloDot, lhs::Type{<:XRTArray}, rhs::Type{<:XRTArray})
    (eltype(lhs), compute_dot_dimensions(op, lhs, rhs))
end

function shape_infer(op::HloMap{fT}, ::Type{<:fT}, args::Type{<:XRTArray}...) where fT
    shapes_match(args...) || error("Shape mismatch in HloMap")
    (eltype(args[1]), size(args[1]))
end

@Base.pure transpose_dims(dims, permutation) = map(x->(x + 1) > length(dims) ? 1 : dims[x + 1], permutation)

function shape_infer(op::HloTranspose, args::Type{<:XRTArray}...)
    shapes_match(args...) || error("Shape mismatch in HloMap")
    (eltype(args[1]), transpose_dims(size(args[1]), op.permutation))
end

function shape_infer(op::HloSelectAndScatter2, args::Type{<:XRTArray}...)
    (eltype(args[1]), size(args[1]))
end

function shape_infer(op::HloRev, A::Type{<:XRTArray})
    (eltype(A), size(A))
end

function infer_rt(op::HloOp, args::Type...)
    T, shape = shape_infer(op, args...)
    XRTArray{T, shape, length(shape)}
end


