# XLAHacks.jl - An XLA compiler prototype for julia

WARNING: This is extremely ugly code. Use for demo only ;).

# Quick start quide
- Grab julia on branch kf/tpu2
- Instantiate this repo
- Put TensorFlow into development mode:
```
pkg> dev TensorFlow
```
- Follow the instructions at https://www.tensorflow.org/install/source up until (but not including) the `bazel build step` (tensorflow version is up to you, but master often fails - `eaebeb1d4d939fb9fd0b75e32a76151cb517bfb6` seems ok at the moment)
- Apply the following patch to tensorflow (the repo you cloned in the previous step, not the
julia wrapper):
```
diff --git a/tensorflow/BUILD b/tensorflow/BUILD
index 9b62a50452..63eed1fb5e 100644
--- a/tensorflow/BUILD
+++ b/tensorflow/BUILD
@@ -442,6 +442,8 @@ tf_cc_shared_object(
         "//tensorflow/c:exported_symbols.lds",
         "//tensorflow/c:version_script.lds",
         "//tensorflow/c/eager:c_api",
+        "//tensorflow/compiler/xrt:xrt_server",
+        "//tensorflow/core/distributed_runtime/rpc:grpc_session",
         "//tensorflow/core:tensorflow",
     ],
 )
```
- Do the following:
```
    bazel build -c opt //tensorflow:libtensorflow.so
    mkdir -p ~/.julia/dev/TensorFlow/deps/usr/bin
    cp bazel-bin/tensorflow/libtensorflow.so ~/.julia/dev/TensorFlow/deps/usr/bin/
    cp bazel-bin/tensorflow/libtensorflow_framework.so ~/.julia/dev/TensorFlow/deps/usr/bin/
```
- Get yourself an xrt_server (either using my patch from https://github.com/tensorflow/tensorflow/pull/22302
 or a Cloud TPU) and connect it to localhost:8470
- Run the script in examples/vgg_forward.jl
