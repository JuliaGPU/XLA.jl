# global state

export platform, executor, global_stream, synchronize

function platform()
    get!(task_local_storage(), :TpuPlatform) do
        p = TpuPlatform()
        initialize!(p)
        p
    end
end

function executor()
    get!(task_local_storage(), :TpuExecutor) do
        exec = TpuExecutor(platform())
        initialize!(exec)
        exec
    end
end

function global_stream()
    get!(task_local_storage(), :TpuStream) do
        TpuStream(executor())
    end
end

function synchronize()
    # FIXME: we only synchronize the default stream here.
    #        LibTPU.TpuExecutor_SynchronizeAllActivity doesn't seem to work
    #        https://github.com/tensorflow/tensorflow/blob/6433b89b4969af4d2dd6eb0531e0f0507dfaffb4/tensorflow/compiler/xla/pjrt/local_device_state.cc#L74-L78
    #        There's also TpuExecutor_BlockUntilDoneOrFailed, but which streams
    #        does that sync exactly?
    with_status() do status
        LibTPU.TpuExecutor_BlockHostUntilDone(executor(), global_stream(), status)
    end

end
