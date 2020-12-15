# This file contains shape transfer functions for HLO ops.
# It is lifted into the type domain by a type assertion in execute.jl,
# as well as being used in the compiler.

# HLO ops with specified type and dimensions
function shape_infer(op::Union{GenericHloOp, HloParameter, HloRng, HloConstant},
                     args::Type{<:HLOArray}...)
    return (op.T, op.shape)
end

eltypes_match(_::Type{<:HLOArray}) = true
eltypes_match(a1::Type{<:HLOArray}, a2::Type{<:HLOArray}, args::Type{<:HLOArray}...) =
    eltype(a1) == eltype(a2) && eltypes_match(a2, args...)

ndims_match(_::Type{<:HLOArray}) = true
ndims_match(a1::Type{<:HLOArray}, a2::Type{<:HLOArray}, args::Type{<:HLOArray}...) =
    ndims(a1) == ndims(a2) && eltypes_match(a2, args...)

shapes_match(_::Type{<:HLOArray}) = true
shapes_match(a1::Type{<:HLOArray}, a2::Type{<:HLOArray}, args::Type{<:HLOArray}...) =
    size(a1) == size(a2) && shapes_match(a2, args...)

@Base.pure non_elt_args(T, shape) = shape[1+ndims(T):end]

function shape_infer(op::HloBroadcast,
                     arg::Type{<:HLOArray}) where {T}
    (eltype(arg), op.result_shape)
end

function shape_infer(op::HloReshape,
                     arg::Type{<:HLOArray{T}}, args::Type{<:HLOArray}...) where {T}
    eltypes_match(arg, args...) || error("eltype mismatch in $(typeof(op))")
    (eltype(arg), non_elt_args(T, op.result_shape))
end

@Base.pure drop_reduce_dims(shape, dims) = tuple((shape[i] for i in 1:length(shape) if !(i-1 in dims))...)

function shape_infer(op::HloReduce{fT}, ::Type{<:fT},
                     args::Type{<:HLOArray}...) where {fT}
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

@Base.pure function infer_window_output_shape(base_shape, windows)
    ntuple(length(windows)) do i
        window = windows[i]
        dilated_base = dilated_bound(base_shape[i], window.base_dilation)
        padded_dilated_base = window.padding_low + dilated_base + window.padding_high
        dilated_window = dilated_bound(window.size, window.window_dilation)
        return strided_bound(padded_dilated_base, dilated_window, window.stride)
    end
end

function shape_infer(op::HloReduceWindow{fT}, ::Type{<:fT},
                     args::Type{<:HLOArray}...) where {fT}
    # For now, assume reductions preserve type
    (eltype(args[1]), infer_window_output_shape(size(args[1]), op.window))
end

@Base.pure function pad_output_size(op::HloPad, arg::Type{<:HLOArray})
    sz = size(arg)
    ntuple(length(op.padding)) do i
        pad = op.padding[i]
        pad.edge_padding_low + pad.edge_padding_low + (sz[i]-1)*pad.interior_padding + sz[i]
    end
end

function shape_infer(op::HloPad, arg::Type{<:HLOArray}, val::Type{<:HLOArray})
    (eltype(arg), pad_output_size(op, arg))
end

function updims(s)
    dims = collect(map(x->x+1, s))
    ndims(dims) == 0 ? dims[] : dims
end

@Base.pure function shape_infer_conv(op::HloConv, lhs::Type{<:HLOArray}, kernel::Type{<:HLOArray})
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
                     lhs::Type{<:HLOArray}, kernel::Type{<:HLOArray})
    shape = shape_infer_conv(op, lhs, kernel)
    # For now, assume convolution preserve type
    (eltype(lhs), shape)
end

function shape_infer(op::HloDynamicUpdateSlice, lhs::Type{<:HLOArray{T, Shp}}, args...) where {T, Shp}
    (T, Shp)
end

function shape_infer(op::HloDynamicSlice, args...) where {T, Shp}
    (eltype(args[1]), non_elt_args(eltype(args[1]), op.sizes))
end

@Base.pure function shape_infer(op::HloGather, A::Type{<:HLOArray}, idxs::Type{<:HLOArray})
    start_idxs = length(size(idxs)) == op.dims.index_vector_dim ?
        (size(idxs)..., 1) : size(idxs)
    result_rank = length(op.dims.offset_dims) + length(start_idxs) - 1
    slice_idx = gather_idx = 1
    shape = ntuple(result_rank) do i
        is_window_index = (i - 1) in op.dims.offset_dims
        if is_window_index
            while (slice_idx-1) in op.dims.collapsed_slice_dims
                slice_idx += 1
            end
            r = op.slice_sizes[slice_idx]; slice_idx += 1
            return r
        else
            if (gather_idx - 1) == op.dims.index_vector_dim
                gather_idx += 1
            end
            r = start_idxs[gather_idx]; gather_idx += 1
            return r
        end
    end
    (eltype(A), shape)
end

@Base.pure slice_dims(dims) = map(dim->div(dim.limit-dim.start, dim.stride), dims)
function shape_infer(op::HloSlice, args...) where {T,Shp}
    (eltype(args[1]), slice_dims(op.dims))
end

@Base.pure function compute_concat_shape(dim::Int64, args...)
    rank = ndims(args[1])
    all(x->ndims(x) == rank, args) || error("All arrays must have the same rank")
    dim <= rank - 1 || error("Concatenation dimension out of bounds")
    ntuple(ndims(args[1])) do i
        if i - 1 == dim
            return sum(j->size(args[j])[i], 1:length(args))
        else
            sz = size(args[1])[i]
            all(j->sz == size(args[j])[i], 1:length(args)) || error("Non-concatenated dimensions must match")
            return sz
        end
    end
end

function shape_infer(op::HloConcatenate, args::Type{<:HLOArray}...)
    eltypes_match(args...) || error("eltype mismatch in $(typeof(op))")
    ndims_match(args...) || error("ndims mismatch in $(typeof(op))")
    shp = compute_concat_shape(op.dim, args...)
    (eltype(args[1]), shp)
end


@Base.pure function compute_dot_dimensions(op::HloDot, lhs::Type{<:HLOArray}, rhs::Type{<:HLOArray})
    lcd = op.dims.lhs_contracting_dimensions[1]
    rcd = op.dims.rhs_contracting_dimensions[1]
    rbatch = op.dims.rhs_batch_dimensions
    dimensions = Int[]
    append!(dimensions, size(lhs)[filter(x->x != lcd+1, 1:ndims(lhs))])
    append!(dimensions, size(rhs)[filter(x->x != rcd+1 && !(x-1 ∈ rbatch), 1:ndims(rhs))])
    tuple(dimensions...)
end


function shape_infer(op::HloDot, lhs::Type{<:HLOArray}, rhs::Type{<:HLOArray})
    (eltype(lhs), compute_dot_dimensions(op, lhs, rhs))
end

function shape_infer(op::HloMap{fT}, ::Type{<:fT}, args::Type{<:HLOArray}...) where fT
    shapes_match(args...) || error("Shape mismatch in HloMap")
    (eltype(args[1]), size(args[1]))
end

@Base.pure transpose_dims(dims, permutation) = map(x->(x + 1) > length(dims) ? 1 : dims[x + 1], permutation)

function shape_infer(op::HloTranspose, args::Type{<:HLOArray}...)
    shapes_match(args...) || error("Shape mismatch in HloMap")
    (eltype(args[1]), transpose_dims(size(args[1]), op.permutation))
end

function shape_infer(op::HloSelectAndScatter{T, S}, ::Type{T}, ::Type{S}, args::Type{<:HLOArray}...) where {T,S}
    (eltype(args[1]), size(args[1]))
end

function shape_infer(op::HloScatter, T::Type, args...)
    (eltype(args[1]), size(args[1]))
end

function shape_infer(op::HloRev, A::Type{<:HLOArray})
    (eltype(A), size(A))
end

shape_infer(op::HloInfeed) = (op.infeed_shape.parameters[1],
                              op.infeed_shape.parameters[2])

shape_infer(op::HloIota) = (op.result_type, op.result_shape)

function infer_rt(op::HloOp, args::Type...)
    T, shape = shape_infer(op, args...)
    HLOArray{T, shape, length(shape)}
end

infer_rt(op::HloInfeed, arg::Type) = Tuple{op.infeed_shape, HloToken}
infer_rt(op::HloAfterAll, args::Type...) = HloToken
infer_rt(op::HloOutfeed, args::Type...) = HloToken

function infer_rt(op::HloCrossReplicaSum, fT::Type, args::Type...)
    Tuple{args...}
end

function infer_rt(op::HloMap, f, args::Type...)
    T, shape = shape_infer(op, typeof(f), args...)
    HLOArray{T, shape, length(shape)}
#=  TODO:
    T = Core.Compiler.return_type(f, Tuple{map(x->HLOArray{eltype(x), (), 0}, args)...})
    if isa(T, UnionAll) || T.abstract
        HLOArray{<:T,
            size(args[1]), length(size(args[1]))}
    else
        HLOArray{T,
            size(args[1]), length(size(args[1]))}
    end
=#
end

infer_rt(op::HloSort, f::Type, args::Type...) = Tuple{args...}
