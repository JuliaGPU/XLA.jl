using XLA
using TensorFlow
using Test

using Flux
using Zygote

struct Relu{T}
    m::T
end
(r::Relu)(x) = relu.(r.m(x))

xrt_server_process = run(`$(joinpath(dirname(pathof(TensorFlow)),"..","deps","downloads","bin","xrt_server"))`; wait=false)
sess = Session(Graph(); target="grpc://localhost:8470")
try
    # Issue #5
    f() = XLA.HloRng(Float32, (5,5), 1)(XRTArray(0f0),XRTArray(1f0))
    @test isa(begin
        run(@tpu_compile f())
    end, XRTArray)

    # Test convolutions with respect to the weights
    model = Relu(Conv(XRTArray(fill(-1f0, 1, 1, 1, 1)), XRTArray(Float32[0f0])))
    function compute_updates(xrtic, x)
      loss, back =  let xrtic=xrtic, x=x
        Zygote._forward(
          Zygote.Context{Nothing}(nothing),
          xrtic -> sum(xrtic(x)),
          xrtic,
        )
      end
      Zygote.tailmemaybe(back(1f0))
    end
    x = reshape(Float32[1.0 -2.0; -3.0 -4.0], (2, 2, 1, 1))
    let compld = @tpu_compile compute_updates(model, XRTArray(x))
        res = run(compld, XRTRemoteStruct(sess, model), XRTArray(x))
        res = convert(typeof(res).parameters[1], res)
        @test convert(Array, res[1].m.weight)[] == -9f0
    end

    # Bringing back allocations
    f(a, b) = (a, b)
    A, B = XRTArray(sess, rand(Float32, 10, 10)), XRTArray(sess, rand(Float32, 10, 10))
    result = run((@tpu_compile f(A, B)), A, B)
    @test isa(result, XLA.XRTRemoteStruct)
    @test isa(fetch(result), Tuple{XRTArray{Float32, (10, 10), 2}, XRTArray{Float32, (10, 10), 2}})
finally
    GC.gc() # Try running finalizers, before the process exits
    close(sess)
    kill(xrt_server_process)
end
