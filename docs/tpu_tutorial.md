# Julia on TPUs #

## Introduction ##

Welcome! In this tutorial you will learn how to run julia on a TPU.
For convenience we will be using the [`ctpu`](https://github.com/tensorflow/tpu/tree/master/tools/ctpu)
utility that comes with the Google Cloud Shell to setup the TPUs,
though you can of course use the GUI or any other command line utility
as well.

## Set up a GCP Project  ##

In order to use a TPU, you will need to select a GCP project to create the TPU resources in. This project will need to have billing enabled. If you are viewing this tutorial in the Cloud Shell, below this text you should see a menu for selecting which project to use:

<walkthrough-project-billing-setup permissions="tpu.nodes.create,compute.instances.create">
</walkthrough-project-billing-setup>

## Enable TPUs for your GCP Project ##
Next you need to enable your project for using TPUs if you have not already done so. Clicking, the button below should take care of this for you.
Do not proceed until you see the "The necessary APIs have been enabled" message below the button.

<walkthrough-enable-apis apis="tpu.googleapis.com">
</walkthrough-enable-apis>


## Check that everything is set up ##

First we neet to tell `ctpu` about the project we just selected. We do so via environment variable (click the little shell button to automatically paste into the cloud shell):
```bash
export DEVSHELL_PROJECT_ID={{project-id}}
```
Next we can verify that everything is set up correctly:
```bash
ctpu print-config
```


If everything worked, you should see output similar to the following

```
ctpu configuration:
        name: julia
        project: {{project-id}}
        zone: us-central1-b
```

## Launch a VM and a Cloud TPU ##

In this step we will spin up a GCP VM and a corresponding Cloud TPU. **NOTE: Google will start charging you as soon as you complete this step. Make sure to shut down all resources when you're done.**

### Launch your VM and TPU

Run the following command to launch both a VM and a TPU. You will be logged into the created VM once the command complete:
```bash
ctpu up
```

### For TFRC users

If you have access to free TPUs as part of the TensorFlow Research Cloud (TFRC), you need to make sure to launch all your TPUs in zone *us-central1-f* in order to take advantage of your free quota. **Note: The created GCP VM will still be charged**

```bash
ctpu up -zone=us-central1-f
```

## Install Julia and XLA.jl

Start by downloading and extracting custom version of julia required by XLA.jl:
```bash
wget https://storage.googleapis.com/julia-tpu-binaries/julia.v1.1.0-kf.tpu3.x86_64-linux-gnu.tar.gz && mkdir julia-tpu && tar -C julia-tpu -xzf julia.v1.1.0-kf.tpu3.x86_64-linux-gnu.tar.gz
```

Then, clone a copy of XLA.jl to the VM:
```bash
git clone https://github.com/JuliaTPU/XLA.jl
```

and launch julia with the XLA.jl project set:
```bash
cd XLA.jl && ~/julia-tpu/bin/julia --project=.
```

Then set up XLA.jl as you would any julia package (hit `]` to get to the package REPL):

```julia
pkg> instantiate
```

## Let's compute something

Start by loading the relevant packages
```julia
using XLA
using TensorFlow
```

Set up the TPU session. Replace 10.240.1.2 with your TPU IP if it is different.
```julia
sess = Session(Graph(); target="grpc://10.240.1.2:8470")
run(sess, TensorFlow.Ops.configure_distributed_tpu())
```

Code that has to be run on the TPU needs to be compiled for the TPU, along with its example input arguments (for the type and shape information).
```julia
f(x) = x*x
data = XRTArray(rand(Float32,5,5))
f_compiled = @tpu_compile f(data)
```

Run the compiled function on the data you want to.
```julia
run(f_compiled, data)
```

Let's see how fast a TPU core really is. We will time our matrix multiplication example on a larger matrix and divide the time by 2n^3. We sum up the result in order to not bring it back to the CPU. The time also includes the time for data transfer to the TPU, and hence the number of teraflops reported is conservative.
```julia
f(x) = sum(x*x)
data = XRTArray(rand(Float32,10^4,10^4))
f_compiled = @tpu_compile f(data)
run(f_compiled, data)
teraflops = @time run(f_compiled, data) / (2*(1e4)^3) / 1e12
```

## You're now ready to use TPUs

Congrats! We will add additional tutorial material here in the near future.

## Clean up

Once you're done, quit the vm

```bash
exit
```

and then clean up any resources:
```bash
ctpu delete
```

**Note: If you used a zone setting to create the resources, you will need to specify it here again, e.e.**
```bash
ctpu delete -zone=us-central1-f
```

If you would like to revisit your setup later, you may instead pause the VM/TPU:
```bash
ctpu pause
```
This will keep any data you may have added to the VM for your future use, but does incur a (small) storage cost for the disk of the VM.

## You're done ##

Please file any issues you encounter in the issue tracker at [https://github.com/JuliaTPU/XLA.jl](https://github.com/JuliaTPU/XLA.jl)

If you have a commerical interest in using TPUs, please contact Julia Computing at [info@juliacomputing.com](mailto:info@juliacomputing.com) for commercial support and assistance.

<walkthrough-conclusion-trophy />
