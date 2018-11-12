using XLA
using TensorFlow
using Test

xrt_server_process = run(`$(joinpath(dirname(pathof(TensorFlow)),"..","deps","downloads","bin","xrt_server"))`; wait=false)
sess = Session(Graph(); target="grpc://localhost:8470")
try
    # Issue #5
    f() = XLA.HloRng(Float32, (5,5), 1)(XRTArray(0f0),XRTArray(1f0))
    @test isa(begin
        run(@tpu_compile f())
    end, XRTArray)

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
