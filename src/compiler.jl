using .Compiler: ReturnNode, argextype, Argument, SSAValue, ⊑, widenconst
using XLA: HloParameter, HloOp, XLAComputationConfig, HloModuleProto, HloProto,
    HloSnapshot, XLAComputation, HloMap

function grab_ir(f, argtypes)
    method = @which f(W, x, b)
    sig = typeof((f, W, x, b))
    mi = Compiler.code_for_method(method, sig, Core.svec(), params.world)
    sv = Compiler.OptimizationState(mi, params)
    ir = Compiler.inflate_ir(ci, mi)
end

function compile_to_xla(ir, sv)
    @assert length(ir.cfg.blocks) == 1
    rt = argextype(ir.stmts[end].val, ir, sv.sp)
    arg_instrs = Vector{HloInstructionProto}(undef, length(ir.argtypes))
    xla_args = Type[]
    computations = HloComputationProto[]
    comp = HloComputationProto(
        name = "comp",
        instructions = HloInstructionProto[ ],
        id = 0
    )
    push!(computations, comp)
    for i = 1:length(ir.argtypes)
        if ir.argtypes[i] ⊑ XRTArray
            AT = widenconst(ir.argtypes[i])
            eltype = AT.parameters[1]
            dims = AT.parameters[2]
            arg_instrs[i] = HloInstructionProto(comp, HloParameter{eltype, dims}(length(xla_args)))
            push!(xla_args, ir.argtypes[i])
        end
    end
    ssa_vals = Vector{HloInstructionProto}(undef, length(ir.stmts))
    function hlo_eval(arg)
        if isa(arg, Argument)
            return arg_instrs[arg.n]
        elseif isa(arg, SSAValue)
            return ssa_vals[arg.id]
        else
            error()
        end
    end
    for (idx, stmt) in pairs(ir.stmts)
        isexpr(stmt, :new) && continue
        isa(stmt, ReturnNode) && continue
        isexpr(stmt, :invoke) || error("Unrecognized expr")
        if isa(stmt.args[2], HloOp)
            args = map(hlo_eval, stmt.args[3:end])
            hlo_inst = stmt.args[2]
            proto = HloInstructionProto(comp, hlo_inst, args...)
            if isa(hlo_inst, HloMap)
                # We need to compute the mapped scalar function
                if hlo_inst.f == +
                    comp′ = HloComputationProto(
                        name = "plus",
                        instructions = HloInstructionProto[ ],
                        id = 1
                    )
                    pushfirst!(computations, comp′)
                    args = [ HloInstructionProto(comp′, HloParameter{Float32, ()}(i)) for i = 0:1 ]
                    plus = HloInstructionProto(comp′, GenericHloOp{:add, Float32, ()}(), args...)
                    comp′.root_id = plus.id
                    proto.called_computation_ids = [comp′.id]
                end
            end
            ssa_vals[idx] = proto
        end
    end
    ret = ir.stmts[end]
    @assert isa(ret, ReturnNode)
    comp.root_id = hlo_eval(ret.val).id

    pshape = ProgramShape(
        parameters = collect(map(Shape, xla_args)),
        result = Shape(rt)
    )

    config = XLAComputationConfig(
        program_shape = pshape
    )
    hlo_module = HloModuleProto(
        name = "test",
        computations = computations,
        entry_computation_name = "comp",
        entry_computation_id = 0,
        id = 0,
        program_shape = pshape,
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

end
