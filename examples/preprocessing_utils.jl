function on_host_preprocessing(x)
    # Convert from UInt8 to Float32 by performing channel normalization
    μ = reshape(Float32[0.485, 0.456, 0.406], (1, 1, 3, 1)).*255
    σ = reshape(Float32[0.229, 0.224, 0.225], (1, 1, 3, 1)).*255
    return (x .- μ)./σ
end

function make_onehot(labels)
    batch_size = length(labels)
    operand = zeros(XRTArray{Float32, (1000, batch_size), 2})
    updates = XLA.HloBroadcast((), (1, batch_size))(XRTArray(1f0))
    update_computation = (x, y) -> y
    scatter_indices = hcat(XRTArray(0:(batch_size-1)), labels)
    sdims = XLA.ScatterDimNums(
        #= update_window_dims =# (0,),
        #= inserted_window_dims =# (0,),
        #= scatter_dims_to_operand_dims =# (1, 0),
        #= index_vector_dim =# 1
    )
    return XLA.HloScatter{typeof(update_computation)}(sdims)(
        update_computation,
        operand,
        scatter_indices,
        updates
    )
end

# Unpack a UInt32 array of size (width, height, batch) to a Float32 array of size (width, height, 3, batch)
# Also perform channel normalization at the same time because, you know, why not?
function unpack_pixels(pixels::XRTArray)
    r_plane = convert(XRTArray{Float32}, (pixels .& XRTArray(0xff000000)) .>> XRTArray(0x00000018))
    g_plane = convert(XRTArray{Float32}, (pixels .& XRTArray(0x00ff0000)) .>> XRTArray(0x00000010))
    b_plane = convert(XRTArray{Float32}, (pixels .& XRTArray(0x0000ff00)) .>> XRTArray(0x00000008))

    # Channel normalization
    r_plane = (r_plane .- XRTArray(123.675f0))./XRTArray(58.395f0)
    g_plane = (g_plane .- XRTArray(116.280f0))./XRTArray(57.120f0)
    b_plane = (b_plane .- XRTArray(103.530f0))./XRTArray(57.375f0)

    # Up-index the planes to four dimensions, then swap the 4th and 3rd dimensions:
    r_plane = permutedims(reshape(r_plane, (size(r_plane)..., 1)), (1, 2, 4, 3))
    g_plane = permutedims(reshape(g_plane, (size(g_plane)..., 1)), (1, 2, 4, 3))
    b_plane = permutedims(reshape(b_plane, (size(b_plane)..., 1)), (1, 2, 4, 3))

    # Concatenate along the 3rd dimension
    return cat(r_plane, g_plane, b_plane; dims=3)
end

function unpack_pixels_host(pixels)
    r_plane = convert(Array{Float32}, (pixels .& 0xff000000) .>> 0x00000018)
    g_plane = convert(Array{Float32}, (pixels .& 0x00ff0000) .>> 0x00000010)
    b_plane = convert(Array{Float32}, (pixels .& 0x0000ff00) .>> 0x00000008)

    # Channel normalization
    r_plane = (r_plane .- 123.675f0)./58.395f0
    g_plane = (g_plane .- 116.280f0)./57.120f0
    b_plane = (b_plane .- 103.530f0)./57.375f0

    # Up-index the planes to four dimensions, then swap the 4th and 3rd dimensions:
    r_plane = permutedims(reshape(r_plane, (size(r_plane)..., 1)), (1, 2, 4, 3))
    g_plane = permutedims(reshape(g_plane, (size(g_plane)..., 1)), (1, 2, 4, 3))
    b_plane = permutedims(reshape(b_plane, (size(b_plane)..., 1)), (1, 2, 4, 3))

    # Concatenate along the 3rd dimension
    return cat(r_plane, g_plane, b_plane; dims=3)
end

function pixel_pack(pixels)
    packed_pixels = zeros(UInt32, (size(pixels,1), size(pixels,2), size(pixels,4)))
    for x in 1:size(pixels, 1)
        for y in 1:size(pixels, 2)
            for batch_idx in 1:size(pixels, 4)
                # Pack those pixels in good boy
                packed_pixels[x, y, batch_idx] = (
                    UInt32(pixels[x, y, 1, batch_idx]) << 24 +
                    UInt32(pixels[x, y, 2, batch_idx]) << 16 +
                    UInt32(pixels[x, y, 3, batch_idx]) << 8
                )
            end
        end
    end
    return packed_pixels
end


