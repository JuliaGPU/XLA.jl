const Compiler = Core.Compiler
using .Compiler: ReturnNode, argextype, SSAValue, ⊑, widenconst, userefs, LineInfoNode, GotoNode,
    IRCode, GotoIfNot, PiNode, PhiNode, foreachssa, block_for_inst, insert_node!,
    dominates, dominated, DominatedBlocks
using InteractiveUtils
using ..XLA: HloParameter, HloOp, XLAComputationConfig, HloModuleProto, HloProto,
    HloSnapshot, XLAComputation, HloMap, HloGetTupleElement, XLAScalar,
    HloReduceWindow, HloReduce, HloTuple

Base.iterate(d::Compiler.DominatedBlocks, state...) = Compiler.iterate(d, state...)

function grab_ir(f, argtypes)
    method = @which f(W, x, b)
    sig = typeof((f, W, x, b))
    mi = Compiler.code_for_method(method, sig, Core.svec(), params.world)
    sv = Compiler.OptimizationState(mi, params)
    ir = Compiler.inflate_ir(ci, mi)
end

macro check_ir(cond, args...)
    @assert length(args) <= 1
    reason = length(args) == 0 ? nothing : esc(args[1])
    quote
        if !$(esc(cond))
            throw(NotOffloadableError($(esc(:ir)), $(esc(:sv)), $(reason)))
        end
    end
end

include("outlining.jl")

function process_function_argument(argvals, sig, idx, ir, stmt, sparams)
    sz = Compiler.isconstType(sig.parameters[idx]) ? 0 : sizeof(sig.parameters[idx])
    if sz != 0
        # If the function being called has non-zero information, we need to make sure it's a constant
        # In the future, XLA will allow this.
        fty = argextype(stmt.args[2+idx], ir, sparams)
        if !isa(fty, Const)
            error("At the moment applied computations must not capture non-constant values")
        end
        argvals = argvals === nothing ? Vector{Any}(undef, 1) : argvals
        argvals[1] = fty.val
    end
    argvals
end

function is_known_call(e::Expr, @nospecialize(func), src, spvals::Core.SimpleVector, slottypes::Vector{Any} = Compiler.empty_slottypes)
    if e.head !== :call
        return false
    end
    f = Compiler.argextype(e.args[1], src, spvals, slottypes)
    return (isa(f, Const) && f.val === func) ||
           (isa(f, DataType) && isdefined(f, :instance) && f.instance === func)
end

function _compile_to_xla!(computations, comp, ir, sv)
    ir = outline_control_flow!(ir, sv)
    arg_instrs = Vector{HloInstructionProto}(undef, length(ir.argtypes))
    xla_args = Type[]
    sparams = sv === nothing ? Core.svec() : sv.sp
    for i = 1:length(ir.argtypes)
        # TODO: We could represent this as a kConstant
        isa(ir.argtypes[i], Const) && continue
        if sv !== nothing && sv.linfo.def.isva && i == length(ir.argtypes)
            # To XLA, we express varargs as separate parameters, so gather all of
            # them and then put them in a tuple
            vargs = HloInstructionProto[]
            for T in ir.argtypes[i].parameters
                instr = HloInstructionProto(comp, "parameter")
                instr.shape = dtype_to_shape(T)
                instr.parameter_number = length(xla_args)
                push!(vargs, instr)
                push!(xla_args, T)
            end
            arg_instrs[i] = HloInstructionProto(comp, HloTuple(), vargs...)
        elseif ir.argtypes[i] ⊑ XRTArray
            AT = widenconst(ir.argtypes[i])
            eltype = AT.parameters[1]
            dims = AT.parameters[2]
            arg_instrs[i] = HloInstructionProto(comp, HloParameter(eltype, dims, length(xla_args)))
            push!(xla_args, ir.argtypes[i])
        elseif representable(ir.argtypes[i])
            T = widenconst(ir.argtypes[i])
            instr = HloInstructionProto(comp, "parameter")
            instr.shape = dtype_to_shape(T)
            instr.parameter_number = length(xla_args)
            arg_instrs[i] = instr
            push!(xla_args, T)
        end
    end
    ssa_vals = Vector{HloInstructionProto}(undef, length(ir.stmts))
    function hlo_eval(arg)
        if isa(arg, Compiler.Argument)
            return arg_instrs[arg.n]
        elseif isa(arg, SSAValue)
            @check_ir isassigned(ssa_vals, arg.id) "SSA value $(arg) not mapped to HLO"
            return ssa_vals[arg.id]
        else
            @check_ir false "Tried to evaluate $(arg) as an HLO argument"
        end
    end
    for (stmt_idx, stmt) in pairs(ir.stmts)
        isa(stmt, Nothing) && continue
        isa(stmt, ReturnNode) && continue
        @check_ir !isa(stmt, PhiNode) "Unsupported control flow found in function"
        if isa(stmt, PiNode)
            if !(widenconst(argextype(stmt.val, ir, sparams)) <: XLAScalar)
                ssa_vals[stmt_idx] = hlo_eval(stmt.val)
            end
            continue
        end
        if isexpr(stmt, :new) || (isexpr(stmt, :call) && is_known_call(stmt, tuple, ir, sparams))
            args = Any[]
            for arg in stmt.args[2:end]
                ty = argextype(arg, ir, sparams)
                representable(arg) || continue
                push!(args, hlo_eval(arg))
            end
            ssa_vals[stmt_idx] = HloInstructionProto(comp, HloTuple(), args...)
            continue
        end
        if isexpr(stmt, :call) && is_known_call(stmt, getfield, ir, sparams)
            structT = widenconst(argextype(stmt.args[2], ir, sparams))
            elt = argextype(stmt.args[3], ir, sparams)
            isa(elt, Const) || error("non-constant structure indexing (1)")
            idx = Compiler.try_compute_fieldidx(structT, elt.val)
            idx !== nothing || error("non-constant structure indexing (2)")
            if !representable(fieldtype(structT, idx))
                continue
            end
            tuple_idx = count(i->representable(fieldtype(structT, i)), 1:idx-1) + 1
            proto = HloInstructionProto(comp, HloGetTupleElement(tuple_idx - 1),
                hlo_eval(stmt.args[2]))
            ssa_vals[stmt_idx] = proto
            continue
        end
        if isa(stmt, GlobalRef) && isa(argextype(stmt, ir, sparams), Const)
            continue
        end
        if !isexpr(stmt, :invoke) && !(isexpr(stmt, :call) && (isa(stmt.args[1], _HloIf) || isa(stmt.args[1], _HloWhile)))
            @Base.show stmt
            Base.display(ir)
            error("Unrecognized expr")
        end
        hlo_inst = isexpr(stmt, :invoke) ? stmt.args[2] : stmt.args[1]
        hlo_inst = argextype(hlo_inst, ir, sparams)
        hlo_inst = hlo_inst.val
        if isa(hlo_inst, HloOp)
            if isa(hlo_inst, HloMap) || isa(hlo_inst, HloReduceWindow) || isa(hlo_inst, HloReduce)
                args = map(hlo_eval, stmt.args[4:end])
                proto = HloInstructionProto(comp, hlo_inst, args...)
                sig = Tuple{typeof(hlo_inst).parameters[1], (XRTArray{eltype(argextype(stmt.args[i], ir, sparams)), (), 0} for i = 4:length(stmt.args))...}
                argvals = process_function_argument(nothing, sig, 1, ir, stmt, sparams)
                res = code_typed_xla(sig; argvals=argvals)
                if res === nothing
                    throw(NotOffloadableError(ir, sv, sprint(io->showerror(io, MethodError(
                        Compiler.isconstType(sig.parameters[1]) ? sig.parameters[1].parameters[1] : sig.parameters[1].instance, Tuple{sig.parameters[2:end]...}
                    )))))
                end
                ir′, sv′ = res
                if !Compiler.isconstType(sig.parameters[1]) && sizeof(sig.parameters[1]) != 0
                    ir′.argtypes[1] = Const(argvals[1])
                end
                comp′ = HloComputationProto(
                    name = "comp$(length(computations))",
                    instructions = HloInstructionProto[ ],
                    id = length(computations)
                )
                pushfirst!(computations, comp′)
                _compile_to_xla!(computations, comp′, ir′, sv′)
                proto.called_computation_ids = [comp′.id]
            elseif isa(hlo_inst, Union{HloScatter, HloCrossReplicaSum})
                args = map(hlo_eval, stmt.args[4:end])
                shape = dtype_to_shape(infer_rt(hlo_inst, map(arg->widenconst(argextype(arg, ir, sparams)), stmt.args[3:end])...))
                proto = HloInstructionProto(comp, hlo_inst, args...; shape=shape)
                sig = Tuple{typeof(hlo_inst).parameters[1], (XRTArray{eltype(argextype(stmt.args[4], ir, sparams)), (), 0} for i = 1:2)...}
                argvals = process_function_argument(nothing, sig, 1, ir, stmt, sparams)
                ir′, sv′ = code_typed_xla(sig; argvals=argvals)
                if sizeof(sig.parameters[1]) != 0
                    ir′.argtypes[1] = Const(argvals[1])
                end
                comp′ = HloComputationProto(
                    name = "comp$(length(computations))",
                    instructions = HloInstructionProto[ ],
                    id = length(computations)
                )
                pushfirst!(computations, comp′)
                _compile_to_xla!(computations, comp′, ir′, sv′)
                proto.called_computation_ids = [comp′.id]
            elseif isa(hlo_inst, HloSelectAndScatter)
                args = map(hlo_eval, stmt.args[5:end])
                proto = HloInstructionProto(comp, hlo_inst, args...)
                proto.called_computation_ids = Int64[]
                # Compile comparison function
                let
                    select_sig = Tuple{typeof(hlo_inst).parameters[1], (XRTArray{eltype(argextype(stmt.args[5], ir, sparams)), (), 0} for i = 1:2)...}
                    select_argvals = process_function_argument(nothing, select_sig, 1, ir, stmt, sparams)
                    ir′, sv′ = code_typed_xla(select_sig; argvals=select_argvals)
                    if sizeof(select_sig.parameters[1]) != 0
                        ir′.argtypes[1] = Const(select_argvals[1])
                    end
                    comp′ = HloComputationProto(
                        name = "comp$(length(computations))",
                        instructions = HloInstructionProto[ ],
                        id = length(computations)
                    )
                    pushfirst!(computations, comp′)
                    _compile_to_xla!(computations, comp′, ir′, sv′)
                    push!(proto.called_computation_ids, comp′.id)
                end
                # Compile Scatter function
                let
                    scatter_sig = Tuple{typeof(hlo_inst).parameters[2], (XRTArray{eltype(argextype(stmt.args[5], ir, sparams)), (), 0} for i = 1:2)...}
                    scatter_argvals = process_function_argument(nothing, scatter_sig, 2, ir, stmt, sparams)
                    ir′, sv′ = code_typed_xla(scatter_sig; argvals=scatter_argvals)
                    if sizeof(scatter_sig.parameters[1]) != 0
                        ir′.argtypes[1] = Const(scatter_argvals[1])
                    end
                    comp′ = HloComputationProto(
                        name = "comp$(length(computations))",
                        instructions = HloInstructionProto[ ],
                        id = length(computations)
                    )
                    pushfirst!(computations, comp′)
                    _compile_to_xla!(computations, comp′, ir′, sv′)
                    push!(proto.called_computation_ids, comp′.id)
                end
            elseif isa(hlo_inst, _HloIf)
                @assert isexpr(stmt, :call)
                args = map(hlo_eval, stmt.args[2:end])
                if_type = ir.types[stmt_idx]
                proto = HloInstructionProto(comp,
                    GenericHloOp{:conditional}(if_type.parameters[1], if_type.parameters[2]), args...)
                proto.called_computation_ids = Int64[]
                for ir′ in (hlo_inst.if_func, hlo_inst.else_func)
                    comp′ = HloComputationProto(
                        name = "comp$(length(computations))",
                        instructions = HloInstructionProto[ ],
                        id = length(computations)
                    )
                    # Having a paremeter instruction is required, so if we don't
                    # otherwise have it, add it manually
                    if !representable(widenconst(ir′.argtypes[2]))
                        instr = HloInstructionProto(comp′, "parameter")
                        instr.shape = dtype_to_shape(Tuple{})
                        instr.parameter_number = 0
                    end
                    pushfirst!(computations, comp′)
                    _compile_to_xla!(computations, comp′, ir′, nothing)
                    push!(proto.called_computation_ids, comp′.id)
                end
            elseif isa(hlo_inst, _HloWhile)
                initial = hlo_eval(stmt.args[2])
                while_type = ir.types[stmt_idx]
                tuple_T = hlo_inst.condition_func.argtypes[2]
                shape = dtype_to_shape(tuple_T)
                proto = HloInstructionProto(comp,
                    GenericHloOp{:while}(Float32, ()), initial,
                    shape = shape)
                proto.called_computation_ids = Int64[]
                for ir′ in (hlo_inst.body_func, hlo_inst.condition_func)
                    comp′ = HloComputationProto(
                        name = "comp$(length(computations))",
                        instructions = HloInstructionProto[ ],
                        id = length(computations)
                    )
                    if !representable(widenconst(ir′.argtypes[2]))
                        instr = HloInstructionProto(comp′, "parameter")
                        instr.shape = dtype_to_shape(Tuple{})
                        instr.parameter_number = 0
                    end
                    pushfirst!(computations, comp′)
                    _compile_to_xla!(computations, comp′, ir′, nothing)
                    push!(proto.called_computation_ids, comp′.id)
                end
            elseif isa(hlo_inst, Union{HloInfeed, HloAfterAll})
                args = map(hlo_eval, stmt.args[3:end])
                shape = dtype_to_shape(infer_rt(hlo_inst, map(arg->argextype(arg, ir, sparams), stmt.args[3:end])...); tensorflow_order=isa(hlo_inst, HloInfeed))
                proto = HloInstructionProto(comp,
                    hlo_inst, args...,
                    shape = shape)
            elseif isa(hlo_inst, HloOutfeed)
                args = map(hlo_eval, stmt.args[3:end])
                shape = dtype_to_shape(infer_rt(hlo_inst, map(arg->argextype(arg, ir, sparams), stmt.args[3:end])...))
                proto = HloInstructionProto(comp,
                    hlo_inst, args...,
                    shape = shape)
                proto.outfeed_shape = dtype_to_shape(widenconst(argextype(stmt.args[3], ir, sparams)); tensorflow_order=true)
            else
                args = map(hlo_eval, stmt.args[3:end])
                shape = Shape(shape_infer(hlo_inst, map(arg->argextype(arg, ir, sparams), stmt.args[3:end])...)...)
                proto = HloInstructionProto(comp, hlo_inst, args...; shape=shape)
            end
            ssa_vals[stmt_idx] = proto
        elseif isa(hlo_inst, Type) && hlo_inst <: XRTArray
            ty = argextype(stmt.args[3], ir, sparams)
            if isa(ty, Const) && isa(ty.val, XLAScalar)
                ssa_vals[stmt_idx] = HloInstructionProto(comp, HloConstant(ty.val))
            else
                # Allow treating an XRTArray as a scalar inside another XRTArray
                # This is a no-op at the HLO level
                ty = widenconst(ty)
                @assert (hlo_inst <: XRTArray{<:Any, (), 0})
                ssa_vals[stmt_idx] = hlo_eval(stmt.args[3])
            end
        elseif hlo_inst == Base.convert
            ty = argextype(stmt.args[3], ir, sparams)
            arg = argextype(stmt.args[4], ir, sparams)
            if arg <: XRTArray{<:Any, (), 0} && isa(ty, Const) && ty.val == eltype(arg)
                # Allow conversions from a scalar array to its element
                # type and codegen it as a no-op.
                ssa_vals[stmt_idx] = hlo_eval(stmt.args[4])
            else
                @show stmt
                error("Unrecognized convert")
            end
        else
            @show stmt
            error("Unrecognized")
        end
    end
    ret = ir.stmts[end]
    @assert isa(ret, ReturnNode)
    comp.root_id = hlo_eval(ret.val).id
    xla_args
end

const ComputationDevice = DeviceAssignment_ComputationDevice
const DeviceMeshCoordinates = DeviceAssignment_ComputationDevice_DeviceMeshCoordinates
function compile_to_xla(ir, sv; replica_device_coords = nothing)
    # TODO: Since all HLO operations are essentially effect free, we could just have
    # a compilation result that throws this error.
    @check_ir isa(ir.stmts[end], ReturnNode) && isdefined(ir.stmts[end], :val) analyze_unreachable
    rt = widenconst(argextype(ir.stmts[end].val, ir, sv.sp))
    computations = HloComputationProto[]
    comp = HloComputationProto(
        name = "comp",
        instructions = HloInstructionProto[ ],
        id = 0
    )
    push!(computations, comp)
    xla_args = _compile_to_xla!(computations, comp, ir, sv)

    pshape = ProgramShape(
        parameters = collect(map(dtype_to_shape, xla_args)),
        result = dtype_to_shape(rt)
    )
    if replica_device_coords !== nothing
        coordinates = vec([ DeviceMeshCoordinates(value=[dev.x, dev.y, dev.core]) for dev in replica_device_coords ])
        cd = ComputationDevice(
            replica_devices = coordinates
        )
        da = DeviceAssignment(
            computation_devices = [cd]
        )
        config = XLAComputationConfig(
            program_shape = pshape,
            device_assignment = da,
            num_replicas = length(coordinates)
        )
    else
        config = XLAComputationConfig(
            program_shape = pshape
        )
    end
    hlo_module = HloModuleProto(
        name = "test",
        computations = computations,
        entry_computation_name = "comp",
        entry_computation_id = 0,
        id = 0,
        host_program_shape = pshape,
    )
    hlo = HloProto(
        hlo_module = hlo_module
    )
    hlo_snap = HloSnapshot(
        hlo = hlo
    )
    xlac = XLAComputation(
        config=config,
        hlo_snapshot = hlo_snap
    )
    xlac, rt
end

function representable(T)
    return sizeof(T) != 0
end

function dtype_to_shape(T::Type{<:XRTArray}; tensorflow_order=false)
    shp = convert(Shape, T)
    if tensorflow_order
        shp.layout.minor_to_major = collect((length(shp.layout.minor_to_major) - 1):-1:0)
    end
    shp
end
dtype_to_shape(T::Type{<:XLA.XLAScalar}; tensorflow_order=false) = dtype_to_shape(XRTArray{T, (), 0})
function dtype_to_shape(T::Type{HloToken}; tensorflow_order=false)
    Shape(
        element_type = xla.PrimitiveType.TOKEN
    )
end
function dtype_to_shape(T::DataType; tensorflow_order=false)
    Shape(
        element_type = xla.PrimitiveType.TUPLE,
        tuple_shapes = collect(dtype_to_shape(fieldtype(T, i); tensorflow_order=tensorflow_order) for i = 1:fieldcount(T) if representable(fieldtype(T, i)))
    )
end
