# XLAHacks.jl - An XLA compiler prototype for julia

WARNING: This is extremely ugly code. Use for demo only ;).

# Quick start quide
- Grab julia on branch [kf/tpu3](https://github.com/JuliaLang/julia/tree/kf/tpu3)
- Instantiate this repo
- Get yourself an `xrt_server` (either running it locally via ``run(`$(joinpath(dirname(pathof(TensorFlow)),"..","deps","downloads","bin","xrt_server"))`)`` or by spinning up a Google Cloud TPU and starting an SSH tunnel to expose its port to the world) and connect it to localhost:8470
- Run the script in examples/vgg_forward.jl
