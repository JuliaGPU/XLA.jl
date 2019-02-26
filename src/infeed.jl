const tf = TensorFlow
function _make_infeed_op(sess, eltypes, sizes, inputs; device=nothing, device_ordinal=0)
    desc = tf.NodeDescription(tf_graph(sess), "InfeedEnqueueTuple", tf.get_name("InfeedEnqueueTuple"))
    tf.set_attr_shape_list(desc, "shapes", Vector{Int64}[collect(x) for x in sizes])
    tf.set_attr_list(desc, "dtypes", DataType[eltypes...])
    desc["device_ordinal"] = device_ordinal
    tf.add_input(desc, inputs)
    if device !== nothing
        tf.set_device(desc, device)
    end
    eq = tf.Tensor(tf.Operation(desc));
    eq
end
function make_infeed_op(sess, tup::NTuple{N, AbstractArray} where N; device=nothing, device_ordinal=0)
    placeholders = [tf.placeholder(eltype(el), shape=size(el)) for el in tup]
    eq = _make_infeed_op(sess, map(eltype, tup), map(size, tup), placeholders; device=device, device_ordinal=device_ordinal)
    feeds = Dict((x=>y for (x, y) in zip(placeholders, tup))...)
    eq, feeds
end
function make_outfeed_op(sess, tup::Type{<:NTuple}; device=nothing, device_ordinal=0)
    desc = tf.NodeDescription(tf_graph(sess), "OutfeedDequeueTuple", tf.get_name("OutfeedDequeueTuple"))
    tf.set_attr_shape_list(desc, "shapes", Vector{Int64}[collect(size(x)) for x in tup.parameters])
    tf.set_attr_list(desc, "dtypes", DataType[eltype(x) for x in tup.parameters])
    desc["device_ordinal"] = device_ordinal
    if device !== nothing
        tf.set_device(desc, device)
    end
    eq = tf.Tensor(tf.Operation(desc));
    eq
end
function infeed(sess, tup::NTuple{N, AbstractArray} where N; device=nothing)
    eq, feeds = make_infeed_op(sess, tup; device=device)
    run(sess, eq, feeds)
end
function outfeed(sess, tup::Type{<:NTuple}; device=nothing)
    eq = make_outfeed_op(sess, tup; device=device)
    run(sess, eq)
end
function infeed_and_outfeed(sess, infeed_tup::NTuple{N, AbstractArray} where N,
        outfeed_tup::Type{<:NTuple}; device=nothing)
    eq_infeed, feeds = make_infeed_op(sess, infeed_tup; device=device)
    eq_outfeed = make_outfeed_op(sess, outfeed_tup; device=device)
    run(sess, [eq_infeed, eq_outfeed], feeds)[2]
end
