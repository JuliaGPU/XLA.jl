# Julia wrapper for header: tpu_executor_c_api.h
# Automatically generated using Clang.jl


function TF_NewStatus()
    ccall((:TF_NewStatus, libtpu()), Ptr{TF_Status}, ())
end

function TF_DeleteStatus(arg1)
    ccall((:TF_DeleteStatus, libtpu()), Cvoid, (Ptr{TF_Status},), arg1)
end

function TF_SetStatus(s, code, msg)
    ccall((:TF_SetStatus, libtpu()), Cvoid, (Ptr{TF_Status}, TF_Code, Cstring), s, code, msg)
end

function TF_SetStatusFromIOError(s, error_code, context)
    ccall((:TF_SetStatusFromIOError, libtpu()), Cvoid, (Ptr{TF_Status}, Cint, Cstring), s, error_code, context)
end

function TF_GetCode(s)
    ccall((:TF_GetCode, libtpu()), TF_Code, (Ptr{TF_Status},), s)
end

function TF_Message(s)
    ccall((:TF_Message, libtpu()), Cstring, (Ptr{TF_Status},), s)
end

function TpuPlatform_New()
    ccall((:TpuPlatform_New, libtpu()), Ptr{SE_Platform}, ())
end

function TpuPlatform_Free(platform)
    ccall((:TpuPlatform_Free, libtpu()), Cvoid, (Ptr{SE_Platform},), platform)
end

function TpuPlatform_Initialize(platform, options_size, options_key, options_value, status)
    ccall((:TpuPlatform_Initialize, libtpu()), Cvoid, (Ptr{SE_Platform}, Csize_t, Ptr{Cstring}, Ptr{Cstring}, Ptr{TF_Status}), platform, options_size, options_key, options_value, status)
end

function TpuPlatform_Initialized(platform)
    ccall((:TpuPlatform_Initialized, libtpu()), Bool, (Ptr{SE_Platform},), platform)
end

function TpuPlatform_GetExecutor(platform, config, status)
    ccall((:TpuPlatform_GetExecutor, libtpu()), Ptr{SE_StreamExecutor}, (Ptr{SE_Platform}, Ptr{SE_StreamExecutorConfig}, Ptr{TF_Status}), platform, config, status)
end

function TpuPlatform_Id(platform)
    ccall((:TpuPlatform_Id, libtpu()), SE_PlatformId, (Ptr{SE_Platform},), platform)
end

function TpuPlatform_VisibleDeviceCount(platform)
    ccall((:TpuPlatform_VisibleDeviceCount, libtpu()), Int64, (Ptr{SE_Platform},), platform)
end

function TpuPlatform_TpuMemoryLimit(platform)
    ccall((:TpuPlatform_TpuMemoryLimit, libtpu()), Int64, (Ptr{SE_Platform},), platform)
end

function TpuPlatform_ShouldRegisterTpuDeviceToDeviceCopy(platform)
    ccall((:TpuPlatform_ShouldRegisterTpuDeviceToDeviceCopy, libtpu()), Bool, (Ptr{SE_Platform},), platform)
end

function TpuPlatform_GetTopologyPtr(platform)
    ccall((:TpuPlatform_GetTopologyPtr, libtpu()), Ptr{SE_TpuTopology}, (Ptr{SE_Platform},), platform)
end

function TpuPlatform_GetHostLocation(platform)
    ccall((:TpuPlatform_GetHostLocation, libtpu()), Ptr{SE_TpuTopology_Host}, (Ptr{SE_Platform},), platform)
end

function TpuExecutor_Init(executor, device_ordinal, device_options, status)
    ccall((:TpuExecutor_Init, libtpu()), Cvoid, (Ptr{SE_StreamExecutor}, Cint, Ptr{SE_DeviceOptions}, Ptr{TF_Status}), executor, device_ordinal, device_options, status)
end

function TpuExecutor_Free(executor)
    ccall((:TpuExecutor_Free, libtpu()), Cvoid, (Ptr{SE_StreamExecutor},), executor)
end

function TpuExecutor_PlatformDeviceCount(executor)
    ccall((:TpuExecutor_PlatformDeviceCount, libtpu()), Cint, (Ptr{SE_StreamExecutor},), executor)
end

function TpuExecutor_Allocate(executor, size, memory_space)
    ccall((:TpuExecutor_Allocate, libtpu()), SE_DeviceMemoryBase, (Ptr{SE_StreamExecutor}, UInt64, Int64), executor, size, memory_space)
end

function TpuExecutor_Deallocate(executor, memory)
    ccall((:TpuExecutor_Deallocate, libtpu()), Cvoid, (Ptr{SE_StreamExecutor}, Ptr{SE_DeviceMemoryBase}), executor, memory)
end

function TpuExecutor_GetAllocatorStats(executor, stats)
    ccall((:TpuExecutor_GetAllocatorStats, libtpu()), Bool, (Ptr{SE_StreamExecutor}, Ptr{SE_AllocatorStats}), executor, stats)
end

function TpuExecutor_DeviceMemoryUsage(executor, free, total)
    ccall((:TpuExecutor_DeviceMemoryUsage, libtpu()), Bool, (Ptr{SE_StreamExecutor}, Ptr{Int64}, Ptr{Int64}), executor, free, total)
end

function TpuExecutor_AllocateStream(executor, stream)
    ccall((:TpuExecutor_AllocateStream, libtpu()), Bool, (Ptr{SE_StreamExecutor}, Ptr{SE_Stream}), executor, stream)
end

function TpuExecutor_DeallocateStream(executor, stream)
    ccall((:TpuExecutor_DeallocateStream, libtpu()), Cvoid, (Ptr{SE_StreamExecutor}, Ptr{SE_Stream}), executor, stream)
end

function TpuExecutor_CreateStreamDependency(executor, dependent, other)
    ccall((:TpuExecutor_CreateStreamDependency, libtpu()), Bool, (Ptr{SE_StreamExecutor}, Ptr{SE_Stream}, Ptr{SE_Stream}), executor, dependent, other)
end

function TpuExecutor_GetStatus(executor, stream, status)
    ccall((:TpuExecutor_GetStatus, libtpu()), Cvoid, (Ptr{SE_StreamExecutor}, Ptr{SE_Stream}, Ptr{TF_Status}), executor, stream, status)
end

function TpuExecutor_GetCoreLocation(executor)
    ccall((:TpuExecutor_GetCoreLocation, libtpu()), Ptr{SE_TpuTopology_Core}, (Ptr{SE_StreamExecutor},), executor)
end

function TpuExecutor_AllocateEvent(executor, event, status)
    ccall((:TpuExecutor_AllocateEvent, libtpu()), Cvoid, (Ptr{SE_StreamExecutor}, Ptr{SE_Event}, Ptr{TF_Status}), executor, event, status)
end

function TpuExecutor_DeallocateEvent(executor, event, status)
    ccall((:TpuExecutor_DeallocateEvent, libtpu()), Cvoid, (Ptr{SE_StreamExecutor}, Ptr{SE_Event}, Ptr{TF_Status}), executor, event, status)
end

function TpuExecutor_PollForEventStatus(executor, event)
    ccall((:TpuExecutor_PollForEventStatus, libtpu()), Cint, (Ptr{SE_StreamExecutor}, Ptr{SE_Event}), executor, event)
end

function TpuExecutor_RecordEvent(executor, stream, event, status)
    ccall((:TpuExecutor_RecordEvent, libtpu()), Cvoid, (Ptr{SE_StreamExecutor}, Ptr{SE_Stream}, Ptr{SE_Event}, Ptr{TF_Status}), executor, stream, event, status)
end

function TpuExecutor_WaitForEvent(executor, stream, event, status)
    ccall((:TpuExecutor_WaitForEvent, libtpu()), Cvoid, (Ptr{SE_StreamExecutor}, Ptr{SE_Stream}, Ptr{SE_Event}, Ptr{TF_Status}), executor, stream, event, status)
end

function TpuExecutor_AllocateTimer(executor, timer)
    ccall((:TpuExecutor_AllocateTimer, libtpu()), Bool, (Ptr{SE_StreamExecutor}, Ptr{SE_Timer}), executor, timer)
end

function TpuExecutor_DeallocateTimer(executor, timer)
    ccall((:TpuExecutor_DeallocateTimer, libtpu()), Cvoid, (Ptr{SE_StreamExecutor}, Ptr{SE_Timer}), executor, timer)
end

function TpuExecutor_StartTimer(executor, stream, timer)
    ccall((:TpuExecutor_StartTimer, libtpu()), Bool, (Ptr{SE_StreamExecutor}, Ptr{SE_Stream}, Ptr{SE_Timer}), executor, stream, timer)
end

function TpuExecutor_StopTimer(executor, stream, timer)
    ccall((:TpuExecutor_StopTimer, libtpu()), Bool, (Ptr{SE_StreamExecutor}, Ptr{SE_Stream}, Ptr{SE_Timer}), executor, stream, timer)
end

function TpuExecutor_SynchronousMemcpyToHost(executor, host_dst, device_src, size, status)
    ccall((:TpuExecutor_SynchronousMemcpyToHost, libtpu()), Cvoid, (Ptr{SE_StreamExecutor}, Ptr{Cvoid}, Ptr{SE_DeviceMemoryBase}, UInt64, Ptr{TF_Status}), executor, host_dst, device_src, size, status)
end

function TpuExecutor_SynchronousMemcpyFromHost(executor, device_dst, host_src, size, status)
    ccall((:TpuExecutor_SynchronousMemcpyFromHost, libtpu()), Cvoid, (Ptr{SE_StreamExecutor}, Ptr{SE_DeviceMemoryBase}, Ptr{Cvoid}, UInt64, Ptr{TF_Status}), executor, device_dst, host_src, size, status)
end

function TpuExecutor_MemcpyToHost(executor, stream, host_dst, device_src, size)
    ccall((:TpuExecutor_MemcpyToHost, libtpu()), Bool, (Ptr{SE_StreamExecutor}, Ptr{SE_Stream}, Ptr{Cvoid}, Ptr{SE_DeviceMemoryBase}, UInt64), executor, stream, host_dst, device_src, size)
end

function TpuExecutor_MemcpyFromHost(executor, stream, device_dst, host_src, size)
    ccall((:TpuExecutor_MemcpyFromHost, libtpu()), Bool, (Ptr{SE_StreamExecutor}, Ptr{SE_Stream}, Ptr{SE_DeviceMemoryBase}, Ptr{Cvoid}, UInt64), executor, stream, device_dst, host_src, size)
end

function TpuExecutor_EnqueueInfeed(executor, infeed_queue_index, data, size, status)
    ccall((:TpuExecutor_EnqueueInfeed, libtpu()), Cvoid, (Ptr{SE_StreamExecutor}, Int32, Ptr{UInt8}, Int64, Ptr{TF_Status}), executor, infeed_queue_index, data, size, status)
end

function TpuExecutor_DequeueOutfeed(executor, outfeed_queue_index, data, size, status)
    ccall((:TpuExecutor_DequeueOutfeed, libtpu()), Cvoid, (Ptr{SE_StreamExecutor}, Int32, Ptr{UInt8}, Int64, Ptr{TF_Status}), executor, outfeed_queue_index, data, size, status)
end

function TpuExecutor_WaitForInfeedReady(executor, infeed_queue_index, status)
    ccall((:TpuExecutor_WaitForInfeedReady, libtpu()), Cvoid, (Ptr{SE_StreamExecutor}, Int32, Ptr{TF_Status}), executor, infeed_queue_index, status)
end

function TpuExecutor_WaitForOutfeedReady(executor, outfeed_queue_index, status)
    ccall((:TpuExecutor_WaitForOutfeedReady, libtpu()), Cvoid, (Ptr{SE_StreamExecutor}, Int32, Ptr{TF_Status}), executor, outfeed_queue_index, status)
end

function TpuExecutor_BlockHostUntilDone(executor, stream, status)
    ccall((:TpuExecutor_BlockHostUntilDone, libtpu()), Cvoid, (Ptr{SE_StreamExecutor}, Ptr{SE_Stream}, Ptr{TF_Status}), executor, stream, status)
end

function TpuExecutor_BlockUntilDoneOrFailed(executor, status)
    ccall((:TpuExecutor_BlockUntilDoneOrFailed, libtpu()), Cvoid, (Ptr{SE_StreamExecutor}, Ptr{TF_Status}), executor, status)
end

function TpuExecutor_SyncAndForgetFailedStreams(executor)
    ccall((:TpuExecutor_SyncAndForgetFailedStreams, libtpu()), Cvoid, (Ptr{SE_StreamExecutor},), executor)
end

function TpuExecutor_SynchronizeAllActivity(executor)
    ccall((:TpuExecutor_SynchronizeAllActivity, libtpu()), Bool, (Ptr{SE_StreamExecutor},), executor)
end

function TpuStream_New(parent)
    ccall((:TpuStream_New, libtpu()), Ptr{SE_Stream}, (Ptr{SE_StreamExecutor},), parent)
end

function TpuStream_Free(arg1)
    ccall((:TpuStream_Free, libtpu()), Cvoid, (Ptr{SE_Stream},), arg1)
end

function TpuStream_Stream(arg1)
    ccall((:TpuStream_Stream, libtpu()), Ptr{Cvoid}, (Ptr{SE_Stream},), arg1)
end

function TpuStream_Status(arg1)
    ccall((:TpuStream_Status, libtpu()), Bool, (Ptr{SE_Stream},), arg1)
end

function TpuStream_IsSameSharedMemoryLocation(arg1, arg2)
    ccall((:TpuStream_IsSameSharedMemoryLocation, libtpu()), Bool, (Ptr{SE_Stream}, Ptr{SE_Stream}), arg1, arg2)
end

function TpuStream_EnqueueTransferHostToDevice(stream, device_dst, host_src, size, status)
    ccall((:TpuStream_EnqueueTransferHostToDevice, libtpu()), Cvoid, (Ptr{SE_Stream}, SE_DeviceMemoryBase, Ptr{Cvoid}, UInt64, Ptr{TF_Status}), stream, device_dst, host_src, size, status)
end

function TpuStream_EnqueueTransferDeviceToHost(stream, device_src, host_dst, size, status)
    ccall((:TpuStream_EnqueueTransferDeviceToHost, libtpu()), Cvoid, (Ptr{SE_Stream}, SE_DeviceMemoryBase, Ptr{Cvoid}, UInt64, Ptr{TF_Status}), stream, device_src, host_dst, size, status)
end

function TpuStream_TpuEnqueueOnDeviceSendRecvLocal(stream, send_buffer, recv_buffer, status)
    ccall((:TpuStream_TpuEnqueueOnDeviceSendRecvLocal, libtpu()), Cvoid, (Ptr{SE_Stream}, SE_DeviceMemoryBase, SE_DeviceMemoryBase, Ptr{TF_Status}), stream, send_buffer, recv_buffer, status)
end

function TpuEvent_New(parent)
    ccall((:TpuEvent_New, libtpu()), Ptr{SE_Event}, (Ptr{SE_StreamExecutor},), parent)
end

function TpuEvent_Free(arg1)
    ccall((:TpuEvent_Free, libtpu()), Cvoid, (Ptr{SE_Event},), arg1)
end

function TpuTimer_New(parent)
    ccall((:TpuTimer_New, libtpu()), Ptr{SE_Timer}, (Ptr{SE_StreamExecutor},), parent)
end

function TpuTimer_Free(arg1)
    ccall((:TpuTimer_Free, libtpu()), Cvoid, (Ptr{SE_Timer},), arg1)
end

function TpuTimer_Nanoseconds(arg1)
    ccall((:TpuTimer_Nanoseconds, libtpu()), Int64, (Ptr{SE_Timer},), arg1)
end

function TpuTimer_Microseconds(arg1)
    ccall((:TpuTimer_Microseconds, libtpu()), Int64, (Ptr{SE_Timer},), arg1)
end

function TpuStatus_New()
    ccall((:TpuStatus_New, libtpu()), Ptr{TF_Status}, ())
end

function TpuStatus_Create(code, msg)
    ccall((:TpuStatus_Create, libtpu()), Ptr{TF_Status}, (Int32, Cstring), code, msg)
end

function TpuStatus_Set(status, code, msg, len)
    ccall((:TpuStatus_Set, libtpu()), Cvoid, (Ptr{TF_Status}, Int32, Cstring, Int32), status, code, msg, len)
end

function TpuStatus_Free(status)
    ccall((:TpuStatus_Free, libtpu()), Cvoid, (Ptr{TF_Status},), status)
end

function TpuStatus_Message(status)
    ccall((:TpuStatus_Message, libtpu()), Cstring, (Ptr{TF_Status},), status)
end

function TpuStatus_Code(status)
    ccall((:TpuStatus_Code, libtpu()), Cint, (Ptr{TF_Status},), status)
end

function TpuStatus_Ok(status)
    ccall((:TpuStatus_Ok, libtpu()), Bool, (Ptr{TF_Status},), status)
end

function TpuStreamExecutorConfig_Default()
    ccall((:TpuStreamExecutorConfig_Default, libtpu()), Ptr{SE_StreamExecutorConfig}, ())
end

function TpuStreamExecutorConfig_SetOrdinal(arg1, ordinal)
    ccall((:TpuStreamExecutorConfig_SetOrdinal, libtpu()), Cvoid, (Ptr{SE_StreamExecutorConfig}, Cint), arg1, ordinal)
end

function TpuStreamExecutorConfig_Free(arg1)
    ccall((:TpuStreamExecutorConfig_Free, libtpu()), Cvoid, (Ptr{SE_StreamExecutorConfig},), arg1)
end

function TpuDeviceDescription_New()
    ccall((:TpuDeviceDescription_New, libtpu()), Ptr{SE_DeviceDescription}, ())
end

function TpuDeviceDescription_Free(description)
    ccall((:TpuDeviceDescription_Free, libtpu()), Cvoid, (Ptr{SE_DeviceDescription},), description)
end

function TpuExecutor_CreateDeviceDescription(executor, description, status)
    ccall((:TpuExecutor_CreateDeviceDescription, libtpu()), Cvoid, (Ptr{SE_StreamExecutor}, Ptr{SE_DeviceDescription}, Ptr{TF_Status}), executor, description, status)
end

function TpuExecutor_NewDeviceOptions(flags)
    ccall((:TpuExecutor_NewDeviceOptions, libtpu()), Ptr{SE_DeviceOptions}, (UInt32,), flags)
end

function TpuExecutor_FreeDeviceOptions(options)
    ccall((:TpuExecutor_FreeDeviceOptions, libtpu()), Cvoid, (Ptr{SE_DeviceOptions},), options)
end

function TpuExecutor_HostCallback(executor, stream, callback_fn, ctx)
    ccall((:TpuExecutor_HostCallback, libtpu()), Bool, (Ptr{SE_StreamExecutor}, Ptr{SE_Stream}, SE_StatusCallbackFn, Ptr{Cvoid}), executor, stream, callback_fn, ctx)
end

function TpuTransferManager_New()
    ccall((:TpuTransferManager_New, libtpu()), Ptr{XLA_TransferManager}, ())
end

function TpuTransferManager_Free(manager)
    ccall((:TpuTransferManager_Free, libtpu()), Cvoid, (Ptr{XLA_TransferManager},), manager)
end

function TpuTransferManager_PlatformId(manager)
    ccall((:TpuTransferManager_PlatformId, libtpu()), SE_PlatformId, (Ptr{XLA_TransferManager},), manager)
end

function TpuTransferManager_HostShapeToDeviceShape(manager, host_shape, device_shape)
    ccall((:TpuTransferManager_HostShapeToDeviceShape, libtpu()), Cvoid, (Ptr{XLA_TransferManager}, Ptr{XLA_Shape}, Ptr{XLA_Shape}), manager, host_shape, device_shape)
end

function TpuTransferManager_TransferLiteralToDeviceAsync(manager, stream, literal, device_buffer, status)
    ccall((:TpuTransferManager_TransferLiteralToDeviceAsync, libtpu()), Cvoid, (Ptr{XLA_TransferManager}, Ptr{SE_Stream}, Ptr{XLA_Literal}, Ptr{XLA_ShapedBuffer}, Ptr{TF_Status}), manager, stream, literal, device_buffer, status)
end

function TpuTransferManager_TransferLiteralFromDevice(manager, stream, device_buffer, literal, callback, ctx)
    ccall((:TpuTransferManager_TransferLiteralFromDevice, libtpu()), Cvoid, (Ptr{XLA_TransferManager}, Ptr{SE_Stream}, Ptr{XLA_ShapedBuffer}, Ptr{XLA_Literal}, XLA_StatusCallbackFn, Ptr{Cvoid}), manager, stream, device_buffer, literal, callback, ctx)
end

function TpuTransferManager_GetByteSizeRequirement(manager, shape)
    ccall((:TpuTransferManager_GetByteSizeRequirement, libtpu()), Int64, (Ptr{XLA_TransferManager}, Ptr{XLA_Shape}), manager, shape)
end

function TpuTransferManager_ChooseCompactLayoutForShape(manager, host_shape, output, status)
    ccall((:TpuTransferManager_ChooseCompactLayoutForShape, libtpu()), Cvoid, (Ptr{XLA_TransferManager}, Ptr{XLA_Shape}, Ptr{XLA_Shape}, Ptr{TF_Status}), manager, host_shape, output, status)
end

function TpuTransferManager_CanShapedBufferBeAccessedNow(manager, executor, device_buffer)
    ccall((:TpuTransferManager_CanShapedBufferBeAccessedNow, libtpu()), Bool, (Ptr{XLA_TransferManager}, Ptr{SE_StreamExecutor}, Ptr{XLA_ShapedBuffer}), manager, executor, device_buffer)
end

function TpuTransferManager_CanBufferBeAccessedNow(manager, executor, device_buffer)
    ccall((:TpuTransferManager_CanBufferBeAccessedNow, libtpu()), Bool, (Ptr{XLA_TransferManager}, Ptr{SE_StreamExecutor}, Ptr{SE_DeviceMemoryBase}), manager, executor, device_buffer)
end

function TpuTransferManager_WriteSingleTupleIndexTable(manager, stream, elements, elements_len, shape, region, status)
    ccall((:TpuTransferManager_WriteSingleTupleIndexTable, libtpu()), Cvoid, (Ptr{XLA_TransferManager}, Ptr{SE_Stream}, Ptr{SE_DeviceMemoryBase}, Csize_t, Ptr{XLA_Shape}, Ptr{SE_DeviceMemoryBase}, Ptr{TF_Status}), manager, stream, elements, elements_len, shape, region, status)
end

function TpuTransferManager_GetInfeedLayout(shape, infeed_shape)
    ccall((:TpuTransferManager_GetInfeedLayout, libtpu()), Cvoid, (Ptr{XLA_Shape}, Ptr{XLA_Shape}), shape, infeed_shape)
end

function TpuTransferManager_LinearizeToBuffers(manager, c_literal, buffers_array, buffers_size, buffers_array_size, status)
    ccall((:TpuTransferManager_LinearizeToBuffers, libtpu()), Cvoid, (Ptr{XLA_TransferManager}, Ptr{XLA_Literal}, Ptr{Ptr{Cstring}}, Ptr{Ptr{Int64}}, Ptr{Int64}, Ptr{TF_Status}), manager, c_literal, buffers_array, buffers_size, buffers_array_size, status)
end

function TpuTransferManager_FreeBuffers(buffers_array, buffers_size, buffers_array_size)
    ccall((:TpuTransferManager_FreeBuffers, libtpu()), Cvoid, (Ptr{Cstring}, Ptr{Int64}, Int64), buffers_array, buffers_size, buffers_array_size)
end

function TpuTransferManager_TransferLiteralToInfeed(manager, executor, c_literal, status)
    ccall((:TpuTransferManager_TransferLiteralToInfeed, libtpu()), Cvoid, (Ptr{XLA_TransferManager}, Ptr{SE_StreamExecutor}, Ptr{XLA_Literal}, Ptr{TF_Status}), manager, executor, c_literal, status)
end

function TpuTransferManager_TransferBuffersToInfeed(manager, executor, buffers_array, buffers_size_in_uint32, buffers_array_size, status)
    ccall((:TpuTransferManager_TransferBuffersToInfeed, libtpu()), Cvoid, (Ptr{XLA_TransferManager}, Ptr{SE_StreamExecutor}, Ptr{Ptr{UInt32}}, Ptr{Int64}, Int64, Ptr{TF_Status}), manager, executor, buffers_array, buffers_size_in_uint32, buffers_array_size, status)
end

function TpuTransferManager_TransferLiteralFromOutfeed(manager, executor, shape, c_literal, status)
    ccall((:TpuTransferManager_TransferLiteralFromOutfeed, libtpu()), Cvoid, (Ptr{XLA_TransferManager}, Ptr{SE_StreamExecutor}, Ptr{XLA_Shape}, Ptr{XLA_Literal}, Ptr{TF_Status}), manager, executor, shape, c_literal, status)
end

function TpuTransferManager_ResetDevices(manager, executors, num_executors, status)
    ccall((:TpuTransferManager_ResetDevices, libtpu()), Cvoid, (Ptr{XLA_TransferManager}, Ptr{Ptr{SE_StreamExecutor}}, Int64, Ptr{TF_Status}), manager, executors, num_executors, status)
end

function TpuComputationPlacer_New()
    ccall((:TpuComputationPlacer_New, libtpu()), Ptr{XLA_ComputationPlacer}, ())
end

function TpuComputationPlacer_Free(placer)
    ccall((:TpuComputationPlacer_Free, libtpu()), Cvoid, (Ptr{XLA_ComputationPlacer},), placer)
end

function TpuComputationPlacer_AssignDevices(placer, replica_count, computation_count, assignment, status)
    ccall((:TpuComputationPlacer_AssignDevices, libtpu()), Cvoid, (Ptr{XLA_ComputationPlacer}, Cint, Cint, Ptr{Cint}, Ptr{TF_Status}), placer, replica_count, computation_count, assignment, status)
end

function TpuComputationPlacer_AssignLocalDevices(host, replica_count, computation_count, assignment, status)
    ccall((:TpuComputationPlacer_AssignLocalDevices, libtpu()), Cvoid, (Ptr{SE_TpuTopology_Host}, Cint, Cint, Ptr{Cint}, Ptr{TF_Status}), host, replica_count, computation_count, assignment, status)
end

function TpuTopology_LogicalDevicesPerHost(tpu_topology, tpu_core_type)
    ccall((:TpuTopology_LogicalDevicesPerHost, libtpu()), Cint, (Ptr{SE_TpuTopology}, TpuCoreTypeEnum), tpu_topology, tpu_core_type)
end

function TpuTopology_LogicalDevicesPerChip(tpu_topology, tpu_core_type)
    ccall((:TpuTopology_LogicalDevicesPerChip, libtpu()), Cint, (Ptr{SE_TpuTopology}, TpuCoreTypeEnum), tpu_topology, tpu_core_type)
end

function TpuTopology_HostCount(tpu_topology)
    ccall((:TpuTopology_HostCount, libtpu()), Cint, (Ptr{SE_TpuTopology},), tpu_topology)
end

function TpuTopology_ChipsPerHost(tpu_topology)
    ccall((:TpuTopology_ChipsPerHost, libtpu()), Cint, (Ptr{SE_TpuTopology},), tpu_topology)
end

function TpuTopology_ChipBounds_X(tpu_topology)
    ccall((:TpuTopology_ChipBounds_X, libtpu()), Cint, (Ptr{SE_TpuTopology},), tpu_topology)
end

function TpuTopology_ChipBounds_Y(tpu_topology)
    ccall((:TpuTopology_ChipBounds_Y, libtpu()), Cint, (Ptr{SE_TpuTopology},), tpu_topology)
end

function TpuTopology_ChipBounds_Z(tpu_topology)
    ccall((:TpuTopology_ChipBounds_Z, libtpu()), Cint, (Ptr{SE_TpuTopology},), tpu_topology)
end

function TpuTopology_HasChip(tpu_topology, x, y, z)
    ccall((:TpuTopology_HasChip, libtpu()), Bool, (Ptr{SE_TpuTopology}, Cint, Cint, Cint), tpu_topology, x, y, z)
end

function TpuTopology_Core(tpu_topology, x, y, z, tpu_core_type, index)
    ccall((:TpuTopology_Core, libtpu()), Ptr{SE_TpuTopology_Core}, (Ptr{SE_TpuTopology}, Cint, Cint, Cint, TpuCoreTypeEnum, Cint), tpu_topology, x, y, z, tpu_core_type, index)
end

function TpuTopology_NumCores(tpu_topology, tpu_core_type)
    ccall((:TpuTopology_NumCores, libtpu()), Cint, (Ptr{SE_TpuTopology}, TpuCoreTypeEnum), tpu_topology, tpu_core_type)
end

function TpuTopology_Cores(tpu_topology, tpu_core_type, cores)
    ccall((:TpuTopology_Cores, libtpu()), Cvoid, (Ptr{SE_TpuTopology}, TpuCoreTypeEnum, Ptr{Ptr{SE_TpuTopology_Core}}), tpu_topology, tpu_core_type, cores)
end

function TpuTopology_IdForHost(tpu_topology, x, y, z)
    ccall((:TpuTopology_IdForHost, libtpu()), Cint, (Ptr{SE_TpuTopology}, Cint, Cint, Cint), tpu_topology, x, y, z)
end

function TpuTopology_Version(tpu_topology)
    ccall((:TpuTopology_Version, libtpu()), TpuVersionEnum, (Ptr{SE_TpuTopology},), tpu_topology)
end

function TpuCoreLocation_ChipCoordinates(tpu_core_location, x, y, z)
    ccall((:TpuCoreLocation_ChipCoordinates, libtpu()), Cvoid, (Ptr{SE_TpuTopology_Core}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}), tpu_core_location, x, y, z)
end

function TpuCoreLocation_HostCoordinates(tpu_core_location, x, y, z)
    ccall((:TpuCoreLocation_HostCoordinates, libtpu()), Cvoid, (Ptr{SE_TpuTopology_Core}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}), tpu_core_location, x, y, z)
end

function TpuCoreLocation_Index(tpu_core_location)
    ccall((:TpuCoreLocation_Index, libtpu()), Cint, (Ptr{SE_TpuTopology_Core},), tpu_core_location)
end

function TpuCoreLocation_Id(tpu_core_location)
    ccall((:TpuCoreLocation_Id, libtpu()), Cint, (Ptr{SE_TpuTopology_Core},), tpu_core_location)
end

function TpuHostLocation_Id(tpu_host_location)
    ccall((:TpuHostLocation_Id, libtpu()), Cint, (Ptr{SE_TpuTopology_Host},), tpu_host_location)
end

function TpuHostLocation_NumCores(tpu_host_location, tpu_core_type)
    ccall((:TpuHostLocation_NumCores, libtpu()), Cint, (Ptr{SE_TpuTopology_Host}, TpuCoreTypeEnum), tpu_host_location, tpu_core_type)
end

function TpuHostLocation_Cores(tpu_host_location, tpu_core_type, cores)
    ccall((:TpuHostLocation_Cores, libtpu()), Cvoid, (Ptr{SE_TpuTopology_Host}, TpuCoreTypeEnum, Ptr{Ptr{SE_TpuTopology_Core}}), tpu_host_location, tpu_core_type, cores)
end

function TpuCompiler_New()
    ccall((:TpuCompiler_New, libtpu()), Ptr{Tpu_Compiler}, ())
end

function TpuCompiler_Free(compiler)
    ccall((:TpuCompiler_Free, libtpu()), Cvoid, (Ptr{Tpu_Compiler},), compiler)
end

function TpuCompiler_RunHloPasses(compiler, se_hlo_module, stream_executor, allocator, result, status)
    ccall((:TpuCompiler_RunHloPasses, libtpu()), Cvoid, (Ptr{Tpu_Compiler}, Ptr{XLA_HloModule}, Ptr{SE_StreamExecutor}, Ptr{SE_DeviceMemoryAllocator}, Ptr{XLA_HloModule}, Ptr{TF_Status}), compiler, se_hlo_module, stream_executor, allocator, result, status)
end

function TpuCompiler_RunBackend(compiler, se_hlo_module, stream_executor, allocator, result, status)
    ccall((:TpuCompiler_RunBackend, libtpu()), Cvoid, (Ptr{Tpu_Compiler}, Ptr{XLA_HloModule}, Ptr{SE_StreamExecutor}, Ptr{SE_DeviceMemoryAllocator}, Ptr{Ptr{SE_Executable}}, Ptr{TF_Status}), compiler, se_hlo_module, stream_executor, allocator, result, status)
end

function TpuCompiler_Compile(compiler, se_hlo_module_group, stream_exec_lists, num_lists, allocator, executables, status)
    ccall((:TpuCompiler_Compile, libtpu()), Cvoid, (Ptr{Tpu_Compiler}, Ptr{XLA_HloModuleGroup}, Ptr{SE_StreamExecutorList}, Cint, Ptr{SE_DeviceMemoryAllocator}, Ptr{Ptr{SE_Executable}}, Ptr{TF_Status}), compiler, se_hlo_module_group, stream_exec_lists, num_lists, allocator, executables, status)
end

function TpuCompiler_ShapeSize(compiler, c_shape)
    ccall((:TpuCompiler_ShapeSize, libtpu()), Int64, (Ptr{Tpu_Compiler}, Ptr{XLA_Shape}), compiler, c_shape)
end

function TpuExecutable_ExecuteAsyncOnStream(executable, se_options, se_arguments, se_arguments_size, hlo_execution_profile, se_output, status)
    ccall((:TpuExecutable_ExecuteAsyncOnStream, libtpu()), Cvoid, (Ptr{SE_Executable}, Ptr{SE_ExecutableRunOptions}, Ptr{Ptr{SE_ExecutionInput}}, Cint, Ptr{SE_HloExecutionProfile}, Ptr{SE_ExecutionOutput}, Ptr{TF_Status}), executable, se_options, se_arguments, se_arguments_size, hlo_execution_profile, se_output, status)
end

function TpuExecutable_Fingerprint(executable, fingerprint, size)
    ccall((:TpuExecutable_Fingerprint, libtpu()), Cvoid, (Ptr{SE_Executable}, Ptr{Cstring}, Ptr{Csize_t}), executable, fingerprint, size)
end

function TpuExecutable_HloModule(executable)
    ccall((:TpuExecutable_HloModule, libtpu()), XLA_HloModule, (Ptr{SE_Executable},), executable)
end

function TpuExecutable_Free(arg1)
    ccall((:TpuExecutable_Free, libtpu()), Cvoid, (Ptr{SE_Executable},), arg1)
end

function XlaShapeToTpuShapeRepresentation(serialized_xla_shape, data_type, use_fast_memory, serialized_tpu_shape, status)
    ccall((:XlaShapeToTpuShapeRepresentation, libtpu()), Cvoid, (Ptr{XLA_Shape}, Cint, Bool, Ptr{XLA_Shape}, Ptr{TF_Status}), serialized_xla_shape, data_type, use_fast_memory, serialized_tpu_shape, status)
end

function XlaShapeToTpuPaddedShape(serialized_xla_shape, padded_shape, status)
    ccall((:XlaShapeToTpuPaddedShape, libtpu()), Cvoid, (Ptr{XLA_Shape}, Ptr{XLA_Shape}, Ptr{TF_Status}), serialized_xla_shape, padded_shape, status)
end
