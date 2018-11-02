# XLA.jl - Compiling Julia to XLA

NOTE: We're in the process of adding better instructions. Check back in a bit.

# Getting started on TPUs

The process for setting up this repository to run against TPUs is much 
the same as the process for setting up the repository locally. However, since
there is additional steps involved in launching the actual TPU, we are providing
a tutorial to walk you through all the steps. It is recommended for those new to
Julia and/or TPUs. If you're already familiar with both, you may skip the tutorial
and just use the setup guide below. The tutorial will open in Google Cloud Shell,
by clicking the button below:

<p align="center">
<a href="https://console.cloud.google.com/cloudshell/open?cloudshell_git_repo=https%3A%2F%2Fgithub.com%2FJuliaTPU%2FXLA.jl&cloudshell_tutorial=docs%2Ftpu_tutorial.md">
<img alt="Open in Cloud Shell" src ="http://gstatic.com/cloudssh/images/open-btn.svg"></a>
</p>

# Getting started (CPU/GPU backend)
- Grab julia on branch [kf/tpu3](https://github.com/JuliaLang/julia/tree/kf/tpu3) (Prebuilt Linux x86_64 binaries with TPU support are available [here](https://storage.googleapis.com/julia-tpu-binaries/julia.v1.1.0-kf.tpu3.x86_64-linux-gnu.tar.gz))
- Instantiate this repo
- `julia> using TensorFlow`
- Get yourself an `xrt_server` (either running it locally via ``run(`$(joinpath(dirname(pathof(TensorFlow)),"..","deps","downloads","bin","xrt_server"))`)`` or by spinning up a Google Cloud TPU and starting an SSH tunnel to expose its port to the world) and connect it to localhost:8470
- Run the script in `examples/vgg_forward.jl`
