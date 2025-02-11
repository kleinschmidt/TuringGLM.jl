using TuringGLM
using Test
using CSV
using DataFrames
using CategoricalArrays: CategoricalValue
using CategoricalArrays: categorical, levels
using Statistics: mean, std
using Random: seed!

const T = TuringGLM

x_float = [1.1, 2.3, 3.14, 3.65]
x_int = [1, 2, 3, 4]
y_float = [2.3, 3.4, 4.5, 5.4]
y_int = [2, 3, 4, 5]

nt_str = (; x_float, x_int=[1, 2, 3, 4], x_cat=string.(x_int), y_float, y_int)

nt_cat = (;
    x_float,
    x_int,
    x_cat=categorical(x_int),
    x_cat_ordered=categorical(x_int; ordered=true),
    y_float,
    y_int,
)

df_str = DataFrame(nt_str)

df_cat = DataFrame(nt_cat)

@testset "TuringGLM.jl" begin
    include("data_constructors.jl")
    include("utils.jl")
    include("priors.jl")
    include("turing_model.jl")
end
