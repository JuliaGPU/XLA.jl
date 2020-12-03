#=
00000000026c0000 T TpuCompile_CompileAndBuild
00000000080443a0 T TpuCompile_CreateCompilationCacheKey
0000000008044360 T TpuCompile_CreateGuaranteedConstFingerprint
0000000008045380 T TpuCompile_DestroyCompilationCacheKey
00000000080441c0 T TpuCompile_IsTpuCompilationEnabled
00000000080441e0 T TpuCompile_ShouldTpuCompileOpIgnoreCancellation
000000000803ae00 T TpuCompiler_Compile
000000000803adc0 T TpuCompiler_Free
000000000803ad80 T TpuCompiler_New
000000000803bb40 T TpuCompiler_RunBackend
000000000803b960 T TpuCompiler_RunHloPasses
000000000803bcc0 T TpuCompiler_ShapeSize
000000000803a3a0 T TpuComputationPlacer_AssignDevices
000000000803a5c0 T TpuComputationPlacer_AssignLocalDevices
000000000803a380 T TpuComputationPlacer_Free
000000000803a360 T TpuComputationPlacer_New
0000000007f4b9a0 T TpuConfigurationApi_CompilationCacheServerAddressFromConfig
0000000007f4b780 T TpuConfigurationApi_FreeCharArray
0000000007f4b7a0 T TpuConfigurationApi_FreeInt32Array
0000000007f4bbe0 T TpuConfigurationApi_GetServerAddressAndPort
0000000007f4b880 T TpuConfigurationApi_HasTPUPodState
0000000007f4b8a0 T TpuConfigurationApi_RemoteCompilationCacheSizeInBytes
0000000007f4b820 T TpuConfigurationApi_TpuMemoryLimit
0000000007f4b7c0 T TpuConfigurationApi_TpusPerHost
000000000803aac0 T TpuCoreLocation_ChipCoordinates
000000000803ab20 T TpuCoreLocation_HostCoordinates
000000000803ab60 T TpuCoreLocation_Id
000000000803ab40 T TpuCoreLocation_Index
0000000008038940 T TpuDeviceDescription_Free
0000000008038900 T TpuDeviceDescription_New
0000000008052be0 T TpuDriver_Allocate
0000000008052cc0 T TpuDriver_AllocateShape
0000000008052ac0 T TpuDriver_AllocateTuple
00000000080521a0 T TpuDriver_Close
00000000080524a0 T TpuDriver_CompileProgram
0000000008052560 T TpuDriver_CompileProgramFromText
00000000080522e0 T TpuDriver_ComputeLinearizedBytesFromShape
0000000008052de0 T TpuDriver_Deallocate
0000000008052400 T TpuDriver_DelinearizeShape
00000000080531e0 T TpuDriver_EventAddCallback
0000000008053240 T TpuDriver_EventAwait
0000000008052940 T TpuDriver_ExecuteProgram
0000000008052780 T TpuDriver_FreeCompiledProgramHandle
0000000008053140 T TpuDriver_FreeCompiledProgramShape
00000000080532e0 T TpuDriver_FreeEvent
00000000080531a0 T TpuDriver_FreeStatus
00000000080522a0 T TpuDriver_FreeSystemInfo
0000000008053080 T TpuDriver_GetCompiledProgramShape
0000000008053360 T TpuDriver_Initialize
0000000008052360 T TpuDriver_LinearizeShape
00000000080527c0 T TpuDriver_LoadProgram
0000000008051fe0 T TpuDriver_Open
0000000008052220 T TpuDriver_QuerySystemInfo
00000000080521e0 T TpuDriver_Reset
0000000008052f40 T TpuDriver_TransferFromDevice
0000000008052fe0 T TpuDriver_TransferFromDeviceToDevice
0000000008052ea0 T TpuDriver_TransferToDevice
0000000008052880 T TpuDriver_UnloadProgram
0000000008053340 T TpuDriver_Version
0000000008038800 T TpuEvent_Free
00000000080387c0 T TpuEvent_New
000000000803bd40 T TpuExecutable_ExecuteAsyncOnStream
000000000803cf40 T TpuExecutable_Fingerprint
000000000803cfa0 T TpuExecutable_Free
000000000803cf80 T TpuExecutable_HloModule
00000000080352e0 T TpuExecutable_LoadProgramAndEnqueueToStream
0000000008036880 T TpuExecute_RuntimeInputToPaddedData
00000000080378c0 T TpuExecutor_Allocate
0000000008037b80 T TpuExecutor_AllocateEvent
0000000008037a40 T TpuExecutor_AllocateStream
0000000008037d80 T TpuExecutor_AllocateTimer
0000000008038320 T TpuExecutor_BlockHostUntilDone
00000000080383a0 T TpuExecutor_BlockUntilDoneOrFailed
0000000008038bc0 T TpuExecutor_CreateDeviceDescription
0000000008037aa0 T TpuExecutor_CreateStreamDependency
0000000008037920 T TpuExecutor_Deallocate
0000000008037bc0 T TpuExecutor_DeallocateEvent
0000000008037a60 T TpuExecutor_DeallocateStream
0000000008037dc0 T TpuExecutor_DeallocateTimer
0000000008038120 T TpuExecutor_DequeueOutfeed
00000000080379e0 T TpuExecutor_DeviceMemoryUsage
0000000008038080 T TpuExecutor_EnqueueInfeed
0000000008037880 T TpuExecutor_Free
0000000008038ae0 T TpuExecutor_FreeDeviceOptions
0000000008037960 T TpuExecutor_GetAllocatorStats
0000000008037b60 T TpuExecutor_GetCoreLocation
0000000008037ae0 T TpuExecutor_GetStatus
0000000008038b40 T TpuExecutor_HostCallback
0000000008037740 T TpuExecutor_Init
0000000008038020 T TpuExecutor_MemcpyFromHost
0000000008037fc0 T TpuExecutor_MemcpyToHost
0000000008038a60 T TpuExecutor_NewDeviceOptions
00000000080378a0 T TpuExecutor_PlatformDeviceCount
0000000008037c40 T TpuExecutor_PollForEventStatus
0000000008037c80 T TpuExecutor_RecordEvent
0000000008037e00 T TpuExecutor_StartTimer
0000000008037e40 T TpuExecutor_StopTimer
0000000008038420 T TpuExecutor_SyncAndForgetFailedStreams
0000000008038440 T TpuExecutor_SynchronizeAllActivity
0000000008037f20 T TpuExecutor_SynchronousMemcpyFromHost
0000000008037e80 T TpuExecutor_SynchronousMemcpyToHost
0000000008037d00 T TpuExecutor_WaitForEvent
00000000080381a0 T TpuExecutor_WaitForInfeedReady
0000000008038260 T TpuExecutor_WaitForOutfeedReady
0000000008dfbfa0 W TpuHalMetadataImplLinked
000000000803abc0 T TpuHostLocation_Cores
000000000803ab80 T TpuHostLocation_Id
000000000803aba0 T TpuHostLocation_NumCores
000000000804ac60 T TpuMeshState_Create
000000000804acc0 T TpuMeshState_Free
000000000804ad00 T TpuMeshState_MeshCommonState
0000000008044340 T TpuNetUtil_RecycleUnusedPort
000000000803f520 T TpuNodeContext_CloseTpuHost
000000000803f340 T TpuNodeContext_Create
000000000803f3e0 T TpuNodeContext_Free
000000000803f580 T TpuNodeContext_Initialize
000000000803f4c0 T TpuNodeContext_StopChipHeartbeats
00000000080372e0 T TpuPlatform_Free
00000000080375c0 T TpuPlatform_GetExecutor
0000000008037720 T TpuPlatform_GetHostLocation
0000000008037700 T TpuPlatform_GetTopologyPtr
0000000008037680 T TpuPlatform_Id
0000000008037300 T TpuPlatform_Initialize
00000000080375a0 T TpuPlatform_Initialized
00000000080372a0 T TpuPlatform_New
00000000080376e0 T TpuPlatform_ShouldRegisterTpuDeviceToDeviceCopy
00000000080376c0 T TpuPlatform_TpuMemoryLimit
00000000080376a0 T TpuPlatform_VisibleDeviceCount
00000000026c2c40 T TpuProgram_DeserializeFromGetTpuProgramResponseProto
00000000026c19a0 T TpuProgram_Free
00000000026c1ae0 T TpuProgram_FreeArray
00000000026c1e20 T TpuProgram_GetExecutableInfo
00000000026c2220 T TpuProgram_GetHloMetadata
00000000026c1fc0 T TpuProgram_GetHostTransferInfo
00000000026c2640 T TpuProgram_GetMayModifyVariables
00000000026c1ca0 T TpuProgram_GetProgramSize
00000000026c2720 T TpuProgram_GetTpuProgram
00000000026c26a0 T TpuProgram_HasSharding
00000000026c1d00 T TpuProgram_LogProgramMemorySummary
00000000026c1920 T TpuProgram_New
00000000026c1a60 T TpuProgram_NewArray
00000000026c29c0 T TpuProgram_SerializeCompilerMetadata
00000000026c2b60 T TpuProgram_SerializeHostComputeMetadata
00000000026c2820 T TpuProgram_SerializeTpuExecutable
00000000026c1b00 T TpuProgram_UnloadAndDestroy
00000000080465a0 T TpuStatus_Code
0000000008046420 T TpuStatus_Create
00000000080464c0 T TpuStatus_Free
0000000008046500 T TpuStatus_Message
0000000008046400 T TpuStatus_New
00000000080465c0 T TpuStatus_Ok
0000000008046540 T TpuStatus_Set
00000000080389c0 T TpuStreamExecutorConfig_Default
0000000008038a20 T TpuStreamExecutorConfig_Free
0000000008038a00 T TpuStreamExecutorConfig_SetOrdinal
0000000008038620 T TpuStream_EnqueueTransferDeviceToHost
0000000008038560 T TpuStream_EnqueueTransferHostToDevice
00000000080384a0 T TpuStream_Free
0000000008038540 T TpuStream_IsSameSharedMemoryLocation
0000000008038460 T TpuStream_New
0000000008038500 T TpuStream_Status
00000000080384e0 T TpuStream_Stream
00000000080386e0 T TpuStream_TpuEnqueueOnDeviceSendRecvLocal
0000000008038880 T TpuTimer_Free
00000000080388e0 T TpuTimer_Microseconds
00000000080388c0 T TpuTimer_Nanoseconds
0000000008038840 T TpuTimer_New
0000000008044320 T TpuTopology_AvailableCoreCount
000000000803a7e0 T TpuTopology_ChipBounds_X
000000000803a800 T TpuTopology_ChipBounds_Y
000000000803a820 T TpuTopology_ChipBounds_Z
000000000803a860 T TpuTopology_Core
000000000803a8c0 T TpuTopology_Cores
000000000803a840 T TpuTopology_HasChip
000000000803aa80 T TpuTopology_IdForHost
000000000803a7c0 T TpuTopology_LogicalDevicesPerChip
000000000803a7a0 T TpuTopology_LogicalDevicesPerHost
000000000803a8a0 T TpuTopology_NumCores
000000000803aaa0 T TpuTopology_Version
00000000080394e0 T TpuTransferManager_CanBufferBeAccessedNow
0000000008039460 T TpuTransferManager_CanShapedBufferBeAccessedNow
0000000008039380 T TpuTransferManager_ChooseCompactLayoutForShape
0000000008038fe0 T TpuTransferManager_Free
0000000008039b60 T TpuTransferManager_FreeBuffers
0000000008039300 T TpuTransferManager_GetByteSizeRequirement
0000000008044160 T TpuTransferManager_GetInfeedLayout
0000000008039040 T TpuTransferManager_HostShapeToDeviceShape
0000000008039860 T TpuTransferManager_LinearizeToBuffers
0000000008038fc0 T TpuTransferManager_New
0000000008039000 T TpuTransferManager_PlatformId
000000000803a100 T TpuTransferManager_ResetDevices
0000000008039cc0 T TpuTransferManager_TransferBuffersToInfeed
00000000080391e0 T TpuTransferManager_TransferLiteralFromDevice
000000000803a020 T TpuTransferManager_TransferLiteralFromOutfeed
00000000080390e0 T TpuTransferManager_TransferLiteralToDeviceAsync
0000000008039be0 T TpuTransferManager_TransferLiteralToInfeed
0000000008039540 T TpuTransferManager_WriteSingleTupleIndexTable
0000000007f4b140 T WaitForDistributedTpuOp_DoWork
000000000803d2c0 T XlaShapeToTpuPaddedShape
000000000803cfe0 T XlaShapeToTpuShapeRepresentation
=#

using Base.Meta

macro new_with_finalizer(new_sym)
    if isa(new_sym, Symbol)
        call = :(ccall(
            ($(quot(new_sym)), libtpu),
            Ptr{Cvoid}, ()))
    else
        call = new_sym
    end
    esc(quote
        handle = $call
        ret = new(handle)
        finalizer(close, ret)
        ret
    end)
end

const libtpu = "/lib/libtpu.so"

let initialized = false
    global initialize_tpu_runtime
    function initialize_tpu_runtime()
        if !initialized
            ccall((:TfTpu_Initialize, libtpu), Cvoid, (Bool,), true)
            initialized = true
        end
    end
end

mutable struct TpuStatus
    handle::Ptr{Cvoid}
    function TpuStatus()
        @new_with_finalizer(TpuStatus_New)
    end
end
is_ok(s::TpuStatus) = @ccall libtpu.TpuStatus_Ok(s::Ptr{Cvoid})::Bool

function with_status(f)
    s = TpuStatus()
    r = f(s)
    @assert is_ok(s)
    close(s)
    r
end

mutable struct TpuPlatform
    handle::Ptr{Cvoid}
    function TpuPlatform()
        initialize_tpu_runtime()
        @new_with_finalizer(TpuPlatform_New)
    end
end

function initialize!(p::TpuPlatform)
    status = Ref{Ptr{Cvoid}}(C_NULL)
    with_status() do s
        @ccall libtpu.TpuPlatform_Initialize(p::Ptr{Cvoid}, 0::Csize_t,
            C_NULL::Ptr{Ptr{UInt8}}, C_NULL::Ptr{Ptr{UInt8}},
            s::Ptr{Cvoid})::Cvoid
    end
end

mutable struct TpuStreamExecutorConfig
    handle::Ptr{Cvoid}
    function TpuStreamExecutorConfig()
        @new_with_finalizer(TpuStreamExecutorConfig_Default)
    end
end
set_ordinal!(sec::TpuStreamExecutorConfig, o::Integer) =
    @ccall libtpu.TpuStreamExecutorConfig_SetOrdinal(sec::Ptr{Cvoid}, o::Cint)::Cvoid

mutable struct TpuExecutor
    handle::Ptr{Cvoid}
    function TpuExecutor(platform::TpuPlatform; config = TpuStreamExecutorConfig())
        @new_with_finalizer with_status() do s
            @ccall libtpu.TpuPlatform_GetExecutor(platform::Ptr{Cvoid},
                config::Ptr{Cvoid}, s::Ptr{Cvoid})::Ptr{Cvoid}
        end
    end
end

mutable struct SEDeviceOptions
    handle::Ptr{Cvoid}
    function SEDeviceOptions(flags::Cuint)
        @new_with_finalizer @ccall libtpu.TpuExecutor_NewDeviceOptions(flags::Cuint)::Ptr{Cvoid}
    end
end

function initialize!(e::TpuExecutor, ordinal = 0, options = SEDeviceOptions(UInt32(0)))
    with_status() do s
        @ccall libtpu.TpuExecutor_Init(e::Ptr{Cvoid}, ordinal::Cint, options::Ptr{Cvoid}, s::Ptr{Cvoid})::Cvoid
    end
end

mutable struct TpuCompiler
    handle::Ptr{Cvoid}
    function TpuCompiler()
        @new_with_finalizer(TpuCompiler_New)
    end
end

struct SEStreamExecutorList
    lists::Ptr{Ptr{Cvoid}}
    count::Cint
end

struct SE_DeviceMemoryBase
    opaque::Ptr{Cvoid}
    size::UInt64
    payload::UInt64
end

struct SE_ScopedDeviceMemory
    wrapped::SE_DeviceMemoryBase
    device_ordinal::Cint
end

function device_allocate(ctx::Ptr{Cvoid}, device_ordinal::Cint, size::UInt64,
    retry_on_failure::Bool, memory_space::Int64, result::Ptr{SE_ScopedDeviceMemory},
    status::Ptr{Cvoid})
    error("Unimplemented")
end

function device_deallocate(ctx::Ptr{Cvoid}, base::Ptr{SE_ScopedDeviceMemory},
    device_ordinal::Cint, status::Ptr{Cvoid})
    error("Unimplemented")
end

mutable struct SEDeviceMemoryAllocator
    platform::Ptr{Cvoid}
    ctx::Ptr{Cvoid}
    AllocateFn::Ptr{Cvoid}
    DeallocateFn::Ptr{Cvoid}

    # roots
    _platform::TpuPlatform

    function SEDeviceMemoryAllocator(platform::TpuPlatform)
        new(
            Base.unsafe_convert(Ptr{Cvoid}, platform),
            C_NULL,
            @cfunction(device_allocate, Cvoid, (Ptr{Cvoid}, Cint, UInt64, Bool, Int64, Ptr{SE_ScopedDeviceMemory}, Ptr{Cvoid})),
            @cfunction(device_deallocate, Cvoid, (Ptr{Cvoid}, Cint, UInt64, Bool, Int64, Ptr{SE_ScopedDeviceMemory}, Ptr{Cvoid})),
        )
    end
end
Base.unsafe_convert(::Type{Ptr{SEDeviceMemoryAllocator}}, x::SEDeviceMemoryAllocator) =
    Ptr{SEDeviceMemoryAllocator}(pointer_from_objref(x))

mutable struct TpuExecutable
    handle::Ptr{Cvoid}
    function TpuExecutable(handle::Ptr{Cvoid})
        @new_with_finalizer(begin
            handle
        end)
    end
end

struct TpuSerializedProto
    bytes::Ptr{UInt8}
    size::Csize_t
end
TpuSerializedProto() = TpuSerializedProto(C_NULL, 0)

struct LibTPUList{T}
    inlined::NTuple{6, T}
    size::Int64
end
LibTPUList{T}() where {T} = LibTPUList{T}(ntuple(_->zero(T),6), 0)

const Int64List = LibTPUList{Int64}
const BoolList = LibTPUList{Bool}

struct XLA_Tile
    dimensions::Int64List
end
Base.zero(::Type{XLA_Tile}) = XLA_Tile(Int64List())

const TileList = LibTPUList{XLA_Tile}

struct XLA_Layout
    format::Cint
    minor_to_major::Int64List
    tiles::TileList
    element_size_in_bits::Int64
    memory_space::Int64
    XLA_Layout() = new(0, Int64List(), TileList(), 0, 0)
end

struct XLA_Shape
    element_type::Cint
    dimensions::Int64List
    dynamic_dimensions::BoolList
    tuple_shapes::Ptr{XLA_Shape}
    ntuple_shapes::Cint
    layout::XLA_Layout
end

struct XLA_ComputationLayout
    parameter_count::Cint
    parameter_layouts::Ptr{XLA_Shape}
    result_layout::XLA_Shape
end

struct XLA_HloModuleConfig
    seed::UInt64
    launch_id::Int32
    replica_count::Int64
    num_partitions::Int64
    use_spmd_partitioning::Bool
    has_static_device_assignment::Bool
    static_device_assignment::TpuSerializedProto
    has_entry_computation_layout::Bool
    entry_computation_layout::XLA_ComputationLayout
end

struct XLA_HloModule
    proto::TpuSerializedProto
    module_config::XLA_HloModuleConfig
end

struct XLA_HloModuleGroup
    proto::TpuSerializedProto
    module_config::Ptr{XLA_HloModuleConfig}
end

function run_backend!(compiler::TpuCompiler, mod::XLA.HloModuleProto,
                  exec::TpuExecutor,
                  allocator::SEDeviceMemoryAllocator)
    buf = IOBuffer()
    writeproto(buf, mod)
    module_serialized = take!(buf)
    @GC.preserve exec module_serialized begin
        modr = Ref(XLA_HloModule(
            TpuSerializedProto(Base.unsafe_convert(Ptr{UInt8}, module_serialized), sizeof(module_serialized)),
            XLA_HloModuleConfig(
            0, 1, 1, 0, true, false, TpuSerializedProto(), true,
            XLA_ComputationLayout(0, C_NULL, XLA_Shape(0, Int64List(), BoolList(),
            C_NULL, 0, XLA_Layout())))
        ))
        result = Ref{Ptr{Cvoid}}()
        with_status() do s
            @ccall libtpu.TpuCompiler_RunBackend(compiler::Ptr{Cvoid},
                modr::Ptr{XLA_HloModule}, exec::Ptr{Cvoid},
                allocator::Ptr{SEDeviceMemoryAllocator},
                result::Ptr{Ptr{Cvoid}}, s::Ptr{Cvoid})::Cvoid
        end
        return TpuExecutable(result[])
    end
end

mutable struct SE_ExecutableRunOptions
    allocator::SEDeviceMemoryAllocator
    device_ordinal::Cint
    stream::Ptr{Cvoid}
    host_to_device_stream::Ptr{Cvoid}
    device_assignment::TpuSerializedProto
    rng_seed::Cint
    run_id::Int64
    launch_id::Cint

    # roots
end

struct SE_ExecutionInput
    #= stuff =#
end

struct XLA_ShapedBuffer
    on_device_shape::XLA_Shape
    device_ordinal::Cint

    bases::Ptr{Cvoid}
    count::Csize_t
end

struct XLA_ShapeIndex
    indices::NTuple{8, Int64}
    count::Int64
end

struct SE_ExecutionOutput
    result::XLA_ShapedBuffer
    to_be_released::Ptr{Ptr{Cvoid}}
    to_be_released_size::Csize_t
    aliased_indices::Ptr{XLA_ShapeIndex}
    aliased_indices_size::Cint
end

function execute_async!(executable::TpuExecutable, options::SE_ExecutableRunOptions,
    arguments::Vector{SE_ExecutionInput}, hlo_execution_profile)
    output = Ref{SE_ExecutionOutput}()
    with_status() do s
        @ccall libtpu.TpuExecutable_ExecuteAsyncOnStreamFn(executable::Ptr{TpuExecutable}, options::Ptr{SE_ExecutableRunOptions},
            arguments::Ptr{SE_ExecutionInput}, length(arguments)::Csize_t,
            output::Ptr{SE_ExecutionOutput}, s::Ptr{Cvoid})::Cvoid
    end
    output[]
end

for (T, sym) in
    ((TpuStatus, :TpuStatus_Free),
     (TpuPlatform, :TpuPlatform_Free),
     (TpuExecutor, :TpuExecutor_Free),
     (TpuStreamExecutorConfig, :TpuStreamExecutorConfig_Free),
     (SEDeviceOptions, :TpuExecutor_FreeDeviceOptions),
     (TpuCompiler, :TpuCompiler_Free),
     (TpuExecutable, :TpuExecutable_Free))
    Base.unsafe_convert(::Type{Ptr{Cvoid}}, h::T) = h.handle
    @eval function Base.close(h::$T)
        ccall(($(quot(sym)), libtpu), Cvoid, (Ptr{Cvoid},), h)
        h.handle = C_NULL
    end
end

#=
using Revise
using XLA, ProtoBuf

includet("/home/keno/.julia/dev/XLA/src/libtpu.jl")

p = TpuPlatform()
initialize!(p)
exec = TpuExecutor(p)
initialize!(exec)
compiler = TpuCompiler()

using XLA: HloModuleGroupProto, Shape, HloInstructionProto,
    HloComputationProto, HloModuleProto, HloModuleGroupProto,
    XRTScalar, ProgramShapeProto

c = HloConstant(Float32(0))

inst = HloInstructionProto()
XLA.fill_fields!(inst, c)

computations = HloComputationProto[]
comp = HloComputationProto(
    name = "comp",
    instructions = [inst],
    id = 0
)
hlo_module = HloModuleProto(
    name = "test",
    computations = computations,
    entry_computation_name = "comp",
    entry_computation_id = 0,
    id = 0,
    host_program_shape = ProgramShapeProto(result=Shape(Float32, ())),
)

executable = run_backend!(compiler, hlo_module, exec, SEDeviceMemoryAllocator(p))[]
execute_async!(executable, SE_ExecutableRunOptions(stream))
=#
