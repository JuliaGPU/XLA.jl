module tensorflow
  using Compat
  const _ProtoBuf_Top_ = @static isdefined(parentmodule(@__MODULE__), :_ProtoBuf_Top_) ? (parentmodule(@__MODULE__))._ProtoBuf_Top_ : parentmodule(@__MODULE__)
  include("tensor_shape_pb.jl")
  include("types_pb.jl")
  module tf2xla
    using Compat
    const _ProtoBuf_Top_ = @static isdefined(parentmodule(@__MODULE__), :_ProtoBuf_Top_) ? (parentmodule(@__MODULE__))._ProtoBuf_Top_ : parentmodule(@__MODULE__)
    include("host_compute_metadata_pb.jl")
  end
end
