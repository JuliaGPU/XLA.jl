module LibTPU

using ProtoBuf
using Artifacts
using CEnum

import ..XLA: Layout, Shape, ProgramShape, XLA

const libtpu = artifact"libtpu/libtpu.so"

include("libtpu_common.jl")
include("libtpu_h.jl")

include("highlevel.jl")

export SE_DeviceMemoryBase, TpuExecutor

end
