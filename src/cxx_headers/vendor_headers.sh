#!/bin/sh
rsync -avm --include='*.h' -f 'hide,! */' ~/tensorflow/ tensorflow
rsync -avm --include='*' -f 'hide,! */' ~/tensorflow/third_party/eigen3/ tensorflow/third_party/eigen3
rsync -avm --include='*.h' -f 'hide,! */' ~/.cache/bazel/_bazel_keno/194f131dbf4e1f471c25958d40967b3c/external/protobuf_archive/src/ proto/
rsync -avm --include='*.h' -f 'hide,! */' ~/.cache/bazel/_bazel_keno/194f131dbf4e1f471c25958d40967b3c/execroot/org_tensorflow/bazel-out/host/genfiles/ genfiles/
rsync -avm --include='*.h' -f 'hide,! */' ~/.cache/bazel/_bazel_keno/194f131dbf4e1f471c25958d40967b3c/execroot/org_tensorflow/bazel-out/k8-py3-opt/genfiles/ genfiles/
rsync -avm --include='*' -f 'hide,! */' ~/.cache/bazel/_bazel_keno/194f131dbf4e1f471c25958d40967b3c/external/eigen_archive/ eigen/
