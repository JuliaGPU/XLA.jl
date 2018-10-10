# XLAHacks.jl - An XLA compiler prototype for julia

WARNING: This is extremely ugly code. Use for demo only ;).

# Quick start quide
- Grab julia on branch kf/tpu2
- Instantiate this repo
- Get a version of libtensorflow.so that supports xrt and grpc (
 add  "//tensorflow/core/distributed_runtime/rpc:grpc_session" and
 "//tensorflow/compiler/xrt:xrt_server" to the libtensorflow build
 rule and copy both libtensorflow_framework.so and libtensorflow.so
 into the TensorFlow.jl deps/usr/bin directory).
- Get yourself an xrt_server (either using my patch from https://github.com/tensorflow/tensorflow/pull/22302
 or a Cloud TPU) and connect it to localhost:8470
- Run the script in examples/vgg_forward.jl
