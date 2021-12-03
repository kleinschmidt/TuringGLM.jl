module TuringGLM

using Statistics: mean, std, var
using DataAPI: DataAPI
using LinearAlgebra: Diagonal, I, LowerTriangular
using LinearAlgebra: normalize, normalize!
using REPL: levenshtein
using SparseArrays: SparseMatrixCSC
using StatsBase: StatsBase
using Tables: Tables
using TableOperations: TableOperations

using Reexport: @reexport

@reexport begin
    using Turing
end

include("utils.jl")
include("contrasts.jl")
include("formula.jl")
include("terms.jl")
include("remat.jl")
include("error_messages.jl")
include("schema.jl")
include("data_constructors.jl")

export @formula

end # module
