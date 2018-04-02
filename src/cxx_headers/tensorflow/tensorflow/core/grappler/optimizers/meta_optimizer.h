/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TENSORFLOW_GRAPPLER_OPTIMIZERS_META_OPTIMIZER_H_
#define TENSORFLOW_GRAPPLER_OPTIMIZERS_META_OPTIMIZER_H_

#include "tensorflow/core/framework/device_base.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/optimizers/graph_optimizer.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/protobuf/rewriter_config.pb.h"

namespace tensorflow {
namespace grappler {

// Run the other grappler optimizers based on the specified rewriter config.
class MetaOptimizer : public GraphOptimizer {
 public:
  MetaOptimizer(DeviceBase* cpu_device, const RewriterConfig& cfg)
      : cpu_device_(cpu_device), cfg_(cfg) {}
  ~MetaOptimizer() override {}

  string name() const override { return "meta_optimizer"; };

  Status Optimize(Cluster* cluster, const GrapplerItem& item,
                  GraphDef* optimized_graph) override;

  void PrintResult();

  void Feedback(Cluster* cluster, const GrapplerItem& item,
                const GraphDef& optimized_graph, double result) override;

 private:
  std::unique_ptr<GraphOptimizer> NewOptimizer(const string& optimizer);
  DeviceBase* const cpu_device_;  // may be NULL
  RewriterConfig cfg_;
  std::vector<std::pair<string, string>> result_;
};

bool MetaOptimizerEnabled(const RewriterConfig& cfg);

// Run the meta optimizer.
//
// If <cpu_device> is non-null, it is the device to be used for executing ops
// during constant folding; if NULL, a new device is created for doing constant
// folding. For performance, it is recommended to pass in an existing cpu_device
// when possible.
Status RunMetaOptimizer(const GrapplerItem& item, const RewriterConfig& cfg,
                        DeviceBase* cpu_device, Cluster* cluster,
                        GraphDef* optimized_graph);

}  // namespace grappler
}  // namespace tensorflow

#endif  // TENSORFLOW_GRAPPLER_OPTIMIZERS_META_OPTIMIZER_H_
