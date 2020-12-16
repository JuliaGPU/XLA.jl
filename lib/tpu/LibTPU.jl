module LibTPU

using ProtoBuf
using LazyArtifacts
using CEnum

import ..XLA: XLA, Layout, Shape, ProgramShape, Tile

let library = Ref{String}() # TODO: plain string after JuliaLang/julia#38907
    global libtpu
    function libtpu()
        if !isassigned(library)
            library[] = artifact"libtpu/libtpu.so"

            args = ["--install_signal_handlers=0"; split(get(ENV, "JULIA_LIBTPU_ARGS", ""))]
            @ccall library[].TfTpu_Initialize(true::Bool, length(args)::Cint, args::Ptr{Cstring})::Cvoid
        end
        library[]::String
    end
end

include("libtpu_common.jl")
include("libtpu_h.jl")

include("status.jl")
include("platform.jl")
include("executor.jl")
include("device.jl")
include("stream.jl")
include("memory.jl")
include("shape.jl")
include("compiler.jl")

end
