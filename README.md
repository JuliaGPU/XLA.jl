# XLA.jl - Compiling Julia to XLA

NOTE: We're in the process of adding better instructions. Check back in a bit.

# Quick start quide
- Grab julia on branch [kf/tpu3](https://github.com/JuliaLang/julia/tree/kf/tpu3) (Prebuilt Linux x86_64 binaries with TPU support are available [here](https://storage.googleapis.com/julia-tpu-binaries/julia.v1.1.0-kf.tpu3.x86_64-linux-gnu.tar.gz))
- Instantiate this repo
- `julia> using TensorFlow`
- Get yourself an `xrt_server` (either running it locally via ``run(`$(joinpath(dirname(pathof(TensorFlow)),"..","deps","downloads","bin","xrt_server"))`)`` or by spinning up a Google Cloud TPU and starting an SSH tunnel to expose its port to the world) and connect it to localhost:8470
- Run the script in `examples/vgg_forward.jl`
