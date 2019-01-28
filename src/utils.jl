using Flux

struct ImmutableChain{T<:Tuple}
    layers::T
end


ImmutableChain(x, y, xs...) = ImmutableChain((x, y, xs...))

Flux.children(c::ImmutableChain) = c.layers
Flux.mapchildren(f, c::ImmutableChain) = ImmutableChain(f.(c.layers))

applyall(::Tuple{}, x) = x
applyall(fs::Tuple, x) = applyall(Base.tail(fs), first(fs)(x))

(c::ImmutableChain)(x) = applyall(c.layers, x)

# Helper function to take a recursive data structure (Such as an ImmutableChain)
# and flatten it out into a tuple
@generated function flatten_tuple(t)
    stmts = Any[]
    tuple_args = Any[]
    function collect_args(t, sym)
        for i = 1:fieldcount(t)
            fT = fieldtype(t, i)
            if fT === Nothing
                continue
            end
            g = gensym()
            push!(stmts, :($g = getfield($sym, $i)))
            if !(fT <: AbstractArray)
                collect_args(fT, g)
            else
                push!(tuple_args, g)
            end
        end
    end
    collect_args(t, :t)
    return quote
        $(stmts...)
        tuple($(tuple_args...))
    end
end

# The inverse of flatten_tuple()
@generated function unflatten_tuple(pattern, t)
    function expand_arg(pattern, idx)
        fields = Any[]
        for i = 1:fieldcount(pattern)
            fT = fieldtype(pattern, i)
            if fT === Nothing
                push!(fields, nothing)
                continue
            end
            if fT <: Tuple || fT <: NamedTuple
                arg, idx = expand_arg(fT, idx)
                push!(fields, arg)
            else
                push!(fields, Expr(:call, :getfield, :t, idx))
                idx += 1
            end
        end
        Expr(:new, pattern, fields...), idx
    end
    res = expand_arg(pattern, 1)[1]
    return res
end
