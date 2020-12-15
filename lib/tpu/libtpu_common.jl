# Automatically generated using Clang.jl


# Skipping MacroDefinition: TF_CAPI_EXPORT __attribute__ ( ( visibility ( "default" ) ) )

const TPU_C_API_MAX_INLINED = 6

@cenum TF_AttrType::UInt32 begin
    TF_ATTR_STRING = 0
    TF_ATTR_INT = 1
    TF_ATTR_FLOAT = 2
    TF_ATTR_BOOL = 3
    TF_ATTR_TYPE = 4
    TF_ATTR_SHAPE = 5
    TF_ATTR_TENSOR = 6
    TF_ATTR_PLACEHOLDER = 7
    TF_ATTR_FUNC = 8
end


struct TF_Status end

@cenum TF_Code::UInt32 begin
    TF_OK = 0
    TF_CANCELLED = 1
    TF_UNKNOWN = 2
    TF_INVALID_ARGUMENT = 3
    TF_DEADLINE_EXCEEDED = 4
    TF_NOT_FOUND = 5
    TF_ALREADY_EXISTS = 6
    TF_PERMISSION_DENIED = 7
    TF_UNAUTHENTICATED = 16
    TF_RESOURCE_EXHAUSTED = 8
    TF_FAILED_PRECONDITION = 9
    TF_ABORTED = 10
    TF_OUT_OF_RANGE = 11
    TF_UNIMPLEMENTED = 12
    TF_INTERNAL = 13
    TF_UNAVAILABLE = 14
    TF_DATA_LOSS = 15
end

@cenum TpuCoreTypeEnum::UInt32 begin
    kTensorCore = 0
    kEmbeddingV1 = 1
    kEmbeddingV2 = 2
end

@cenum TpuVersionEnum::UInt32 begin
    kUnknownTpuVersion = 0
    kTpuV2 = 1
    kTpuV3 = 2
    kTpuV4 = 3
end


struct SE_Platform end
struct SE_StreamExecutor end
struct SE_Stream end
struct SE_Event end
struct SE_Timer end

struct TpuSerializedProto
    bytes::Cstring
    size::Csize_t
end

struct SE_PlatformId
    id::Ptr{Cvoid}
end

struct SE_StreamExecutorConfig end
struct SE_DeviceOptions end
const SE_StatusCallbackFn = Ptr{Cvoid}

struct SE_DeviceMemoryBase
    opaque::Ptr{Cvoid}
    size::UInt64
    payload::UInt64
end

struct SE_ScopedDeviceMemory
    wrapped::SE_DeviceMemoryBase
    device_ordinal::Cint
end

struct SE_AllocatorStats
    num_allocs::Int64
    bytes_in_use::Int64
    peak_bytes_in_use::Int64
    largest_alloc_size::Int64
    has_bytes_limit::Bool
    bytes_limit::Int64
    bytes_reserved::Int64
    peak_bytes_reserved::Int64
    has_bytes_reservable_limit::Bool
    bytes_reservable_limit::Int64
    largest_free_block_bytes::Int64
end

const SE_AllocateFn = Ptr{Cvoid}
const SE_DeallocateFn = Ptr{Cvoid}

struct SE_DeviceMemoryAllocator
    platform::Ptr{SE_Platform}
    ctx::Ptr{Cvoid}
    allocate::SE_AllocateFn
    deallocate::SE_DeallocateFn
end

struct SE_DeviceDescription
    device_vendor::Cstring
    platform_version::Cstring
    driver_version::Cstring
    runtime_version::Cstring
    pci_bus_id::Cstring
    name::Cstring
    thread_dim_limit_x::Int64
    thread_dim_limit_y::Int64
    thread_dim_limit_z::Int64
    block_dim_limit_x::Int64
    block_dim_limit_y::Int64
    block_dim_limit_z::Int64
    threads_per_core_limit::Int64
    threads_per_block_limit::Int64
    threads_per_warp::Int64
    registers_per_core_limit::Int64
    registers_per_block_limit::Int64
    device_address_bits::Int64
    device_memory_size::Int64
    memory_bandwidth::Int64
    shared_memory_per_core::Int64
    shared_memory_per_block::Int64
    clock_rate_ghz::Cfloat
    cuda_compute_capability_major::Cint
    cuda_compute_capability_minor::Cint
    rocm_amdgpu_isa_version::Cint
    numa_node::Cint
    core_count::Cint
    ecc_enabled::Bool
end

struct Tpu_Compiler end
struct SE_Executable end

struct SE_ExecutableRunOptions
    allocator::SE_DeviceMemoryAllocator
    device_ordinal::Cint
    stream::Ptr{SE_Stream}
    host_to_device_stream::Ptr{SE_Stream}
    device_assignment::TpuSerializedProto
    rng_seed::Cint
    run_id::Int64
    launch_id::Cint
end

struct SE_MaybeOwningDeviceMemory
    memory::SE_DeviceMemoryBase
    owned::Bool
    device_ordinal::Cint
    allocator::SE_DeviceMemoryAllocator
end

struct LibTPUList{T}
    inlined::NTuple{6, T}
    size::Int64
end

LibTPUList{T}() where {T} = LibTPUList{T}(ntuple(_->zero(T),6), 0)

function Base.convert(::Type{LibTPUList{T}}, x::Vector{T}) where {T}
    @assert length(x) <= 6
    LibTPUList{T}(ntuple(6) do i
        i > length(x) ? zero(T) : x[i]
    end, length(x))
end

Base.convert(::Type{Vector{T}}, x::LibTPUList) where {T} =
    T[x.inlined[1:x.size]...]

const Int64List = LibTPUList{Int64}
const BoolList = LibTPUList{Bool}

struct XLA_Tile
    dimensions::Int64List
end

const TileList = LibTPUList{XLA_Tile}

struct XLA_Layout
    format::Cint
    minor_to_major::Int64List
    tiles::TileList
    element_size_in_bits::Int64
    memory_space::Int64
end

struct XLA_Shape
    element_type::Cint
    dimensions::Int64List
    dynamic_dimensions::BoolList
    tuple_shapes::NTuple{N, XLA_Shape} where N      # HACK
    ntuple_shapes::Cint
    layout::XLA_Layout
end

struct XLA_ShapedBuffer
    on_device_shape::XLA_Shape
    device_ordinal::Cint
    bases::Ptr{SE_DeviceMemoryBase}
    count::Csize_t
end

struct XLA_Literal
    buffers::Ptr{Cstring}
    sizes::Ptr{Csize_t}
    count::Csize_t
    shape::XLA_Shape
end

struct XLA_MaybeOwningDeviceMemoryShapeTree
    shape::XLA_Shape
    buffers::Ptr{SE_MaybeOwningDeviceMemory}
end

struct XLA_ShapeIndex
    indices::NTuple{8, Int64}
    count::Int64
end

struct SE_ExecutionInput
    shape_tree::XLA_MaybeOwningDeviceMemoryShapeTree
    unowned_indices::Ptr{XLA_ShapeIndex}
    unowned_indices_size::Cint
    dynamic_shape::XLA_Shape
end

struct SE_ExecutionOutput
    result::XLA_ShapedBuffer
    to_be_released::Ptr{SE_MaybeOwningDeviceMemory}
    to_be_released_size::Cint
    aliased_indices::Ptr{XLA_ShapeIndex}
    aliased_indices_size::Cint
end

struct XLA_ComputationLayout
    parameter_count::Cint
    parameter_layouts::Ptr{XLA_Shape}
    result_layout::XLA_Shape

    _parameter_layouts::Union{Nothing, Vector{XLA_Shape}}   # HACK
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

struct SE_HloExecutionProfile end

struct SE_StreamExecutorList
    exec::Ptr{Ptr{SE_StreamExecutor}}
    count::Cint
end

struct XLA_HloModuleGroup
    proto::TpuSerializedProto
    module_config::Ptr{XLA_HloModuleConfig}
end

struct XLA_HloModule
    proto::TpuSerializedProto
    module_config::XLA_HloModuleConfig
end

struct XLA_TransferManager end
struct XLA_ComputationPlacer end
const XLA_CallbackFn = Ptr{Cvoid}
const XLA_StatusCallbackFn = Ptr{Cvoid}
struct SE_TpuTopology end
struct SE_TpuTopology_Core end
const SE_TpuTopology_Host = SE_TpuTopology_Core

struct TfTpu_ExecutorApiFn
    decltype::Cvoid
end
