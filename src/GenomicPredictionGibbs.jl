module GenomicPredictionGibbs

using LinearAlgebra, Distributions
include("Bayes_B.jl")

export Bayes_B_Gibbs_run, Bayes_B_Gibbs_step
end
