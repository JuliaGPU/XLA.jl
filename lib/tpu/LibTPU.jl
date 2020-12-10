module LibTPU

using ProtoBuf

import ..XLA: Layout, Shape, ProgramShape, XLA

include("libtpu_h.jl")

export SE_DeviceMemoryBase, TpuExecutor

end
