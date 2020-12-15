module LibTPU

using ProtoBuf
using Artifacts
using CEnum

import ..XLA: Layout, Shape, ProgramShape, XLA

const libtpu = artifact"libtpu/libtpu.so"

include("libtpu_common.jl")
include("libtpu_h.jl")

include("status.jl")
include("platform.jl")
include("executor.jl")
include("stream.jl")

include("highlevel.jl")

function __init__()
    args = ["--install_signal_handlers=0"; split(get(ENV, "JULIA_LIBTPU_ARGS", ""))]
    @ccall libtpu.TfTpu_Initialize(true::Bool, length(args)::Cint, args::Ptr{Cstring})::Cvoid
end

end
