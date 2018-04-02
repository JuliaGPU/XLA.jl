if !isdefined(@__MODULE__, :sess)
    sess = TensorFlow.Session(;target="grpc://localhost:2222", enable_xla=true)
end

cxx"""
#include "tensorflow/compiler/xla/service/hlo_tfgraph_builder.h"
"""

function GenerateGraphDef(client, builder, comp, args)
    String(icxx"""
        xla::hlo_graph_dumper::HloTfGraphBuilder builder;
        xla::Service *service = (xla::Service *)$client->stub();
        xla::Computation comp = $builder->Build().ValueOrDie();
        xla::UserComputation *C = service->computation_tracker().Resolve(comp.handle()).ValueOrDie();
        xla::HloModuleConfig config;
        auto module = service->computation_tracker().BuildHloModule(
            C->GetVersionedHandle(),
            config).ValueOrDie();
        bool ok = builder.AddComputation(*module->entry_computation()).ok();
        assert(ok);
        builder.GetGraphDef().SerializeAsString();
    """)
end

macro generate_graph(expr)
    call = gen_call_with_extracted_types_and_shapes(__module__, Expr(:quote, generate_xla_for), Expr(:quote, TransferArgs), expr)
    quote
        builder = icxx"""new xla::ComputationBuilder($client, "test");"""
        (comp, args) = $call
        GenerateGraphDef(client, builder, comp, args)
    end
end

a, b = rand(Float32, 10, 10), rand(Float32, 10, 10)
f(a, b) = a .+ b

import_graph_def(sess.graph, Vector{UInt8}(@generate_graph f(a, b)))

