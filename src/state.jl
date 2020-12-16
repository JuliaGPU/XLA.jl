# global state

export platform, executor, default_stream, synchronize, transfer_manager

function platform()
    get!(task_local_storage(), :TpuPlatform) do
        p = TpuPlatform()
        initialize!(p)
        p
    end
end

function executor(device::Int=0)
    get!(task_local_storage(), (:TpuExecutor, device)) do
        exec = TpuExecutor(platform())
        initialize!(exec, device)
        exec
    end
end

function default_stream(exec::TpuExecutor=executor())
    get!(task_local_storage(), (:TpuStream, exec)) do
        TpuStream(executor())
    end
end

function synchronize(exec::TpuExecutor=executor())
    LibTPU.TpuExecutor_SynchronizeAllActivity(exec)
end

function transfer_manager()
    get!(task_local_storage(), :TpuTransferManager) do
        p = TpuTransferManager()
        p
    end
end
