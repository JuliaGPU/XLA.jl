ENV["JULIA_CXX_RTTI"]="1"

if !haskey(ENV, "LIBXLA")
    error("LIBXLA environment variable not set. Set it to a .so that has xla symbols exposed.")
end

# Load generated xla protobuf definitions
include(joinpath(dirname(@__FILE__), "gen","xla.jl"))
using .xla
using ProtoBuf

# Load Flux and linear algebra
using Flux
using LinearAlgebra

# Uncomment this if you also want to play with
# tensorflow itself
# ENV["LIBTENSORFLOW"] = ENV["LIBXLA"]
# using TensorFlow

# Workarounds some bugs
code_typed(*, Tuple{Array{Float64, 2}, Array{Float32,1}})
Base.keys(s::Core.SimpleVector) = 1:length(s)


# Hacky version to get current beta compiler
#using NotInferenceDontLookHere
#const Compiler = NI
const Compiler = Core.Compiler
Base.getindex(x::Compiler.IRCode, i) = Compiler.getindex(x, i)
Base.setindex!(x::Compiler.IRCode, v, i) = Compiler.setindex!(x, v, i)

using Core.IR
using Base.Meta
module IRShow
    using Core.Compiler: CFG, IRCode, scan_ssa_use!, ReturnNode, GotoIfNot, Argument
    using Core.IR
    using Base: IdSet, peek
    using Base.Meta: isexpr
    Core.Compiler.push!(x::IdSet, y) = Base.push!(x, y)
    include(Base.joinpath(Base.Sys.BINDIR, Base.DATAROOTDIR, "julia", "base", "compiler/ssair/show.jl"))
end

# Load XLA C++ library
using Cxx
Cxx.addHeaderDir(joinpath(dirname(@__FILE__), "cxx_headers/proto"), kind=C_System)
Cxx.addHeaderDir(joinpath(dirname(@__FILE__), "cxx_headers/genfiles"), kind=C_System)
Cxx.addHeaderDir(joinpath(dirname(@__FILE__), "cxx_headers/tensorflow"), kind=C_System)
Cxx.addHeaderDir(joinpath(dirname(@__FILE__), "cxx_headers/eigen"), kind=C_System)
Libdl.dlopen(ENV["LIBXLA"], Libdl.RTLD_GLOBAL)

cxx"""
#include "tensorflow/compiler/xla/client/client_library.h"
#include "tensorflow/compiler/xla/client/computation_builder.h"
"""

# Initialize client
client = icxx"xla::ClientLibrary::LocalClientOrDie();"

include("xla_compiler.jl")

@info "XLA Compiler loaded. See demo.jl for examples."
