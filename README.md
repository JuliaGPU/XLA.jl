# XLAHacks.jl - An XLA compiler prototype for julia

WARNING: This is extremely ugly code. Use for demo only ;).

# Instructions

First, get a copy of the appropriate branch of julia:
```
git clone https://github.com/JuliaLang/julia.git
cd julia
git checkout kf/xlademo
cat > Make.user << END
LLVM_ASSERTIONS=1
BUILD_LLVM_CLANG=1
END

make -j20
```

Then, set up a .so that exposes xla C++ symbols (in particular ComputationBuilder) and set the LIBXLA environment variable to it:
```
export LIBXLA=<PATH_TO_SO>
```

It's possible to build one for OSS tensor flow by applying the 
`libjlxla.patch` in this directory and running:

```
bazel build --config=opt //tensorflow/compiler/xla/julia:libjlxla.so
```

You'll have to figure out how to get such a library from the internal
code base.

Then,

```
git clone git clone git@github.com:Keno/XLAHacks.jl.git
cd XLAHacks.jl.git # N.B.: It's important you're in this directory when starting julia - it'll pick up its environment from there
~/julia/julia --depwarn=no # Depwarn disables deprecation warnings - Strongly recommended - this is pre-alpha code so tons of warning
] up --manifest --fixed # Include the ']', that's the hotkey for the pkg manager
] build Cxx
julia> include("src/XLA.jl")
```

That should add everything properly. Alternatively,
dependencies can be added manually:

```
] add Cxx#7186a67559fac50f22f25b0ba59ed48f7367948d
] add Flux#9ebab5cb0410db48c7baf520ed77a6465806c12f
] add ProtoBuf
] build Cxx
```

