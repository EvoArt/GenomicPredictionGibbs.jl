
using GLMakie
using LinearAlgebra, Distributions
include("simulate_gene_effects.jl")
"""
    Bayes_B_Gibbs(X, y, β, σ²ᵦ, σ²ₑ, π; b=1, c=1, S²ᵦ=var(y)*0.1, S²ₑ=var(y), νᵦ=3, νₑ=3)

Performs Gibbs sampling for the Bayesian variable selection model (BayesB) in a linear regression context.

# Arguments
- `X::Matrix{T}`: The design matrix of size `n × p`, where `n` is the number of observations and `p` is the number of predictors.
- `y::Vector{T}`: The response vector of length `n`.
- `β::Vector{T}`: The vector of regression coefficients of length `p`.
- `σ²ᵦ::T`: The prior variance of the regression coefficients.
- `σ²ₑ::T`: The residual variance.
- `π::T`: The prior inclusion probability for each predictor.
- `b::T=1`: Hyperparameter for the Beta prior on `π`.
- `c::T=1`: Hyperparameter for the Beta prior on `π`.
- `S²ᵦ::T=var(y)*0.1`: Prior scale for the variance of regression coefficients.
- `S²ₑ::T=var(y)`: Prior scale for the residual variance.
- `νᵦ::T=3`: Degrees of freedom for the prior on the variance of regression coefficients.
- `νₑ::T=3`: Degrees of freedom for the prior on the residual variance.

# Returns
- `β::Vector{T}`: Updated vector of regression coefficients.
- `σ²ᵦ::T`: Updated prior variance of the regression coefficients.
- `σ²ₑ::T`: Updated residual variance.
- `π::T`: Updated prior inclusion probability.

# Description
This function implements the BayesB algorithm using Gibbs sampling. It iteratively updates the regression coefficients (`β`), the prior variance of the coefficients (`σ²ᵦ`), the residual variance (`σ²ₑ`), and the prior inclusion probability (`π`). The algorithm assumes a spike-and-slab prior for the regression coefficients, where each coefficient is either zero (spike) or drawn from a normal distribution (slab).

# Notes
- The function assumes that the input data `X` and `y` are preprocessed appropriately.
- The function uses the `Normal`, `Bernoulli`, and `InverseGamma` distributions for sampling.
- The `clamp` function is used to ensure numerical stability of probabilities.

# Example
function Bayes_B_Gibbs(X,y,β,σ²ᵦ,σ²ₑ,π;b=1,c=1,S²ᵦ = var(y) *0.1,S²ₑ = var(y),νᵦ=3,νₑ=3)
    n,p = size(X)
    Xβ = X .* transpose(β)
    predicted = vec(sum(Xβ,dims = 2))
    k = 0
    for j in 1:p
        β_prev = β[j]
        xⱼ = X[:,j]
        xⱼ² = dot(xⱼ, xⱼ) 
        X₍₋ⱼ₎β₍₋ⱼ₎ = predicted .- Xβ[:,j]
        xⱼ² = dot(xⱼ, xⱼ)
        rⱼ = dot(xⱼ, (y - X₍₋ⱼ₎β₍₋ⱼ₎))
                xj2   = dot(xⱼ, xⱼ)
        βhat  = rⱼ / xj2   

        σ2_0  = σ²ₑ / xj2
        σ2_1  = σ²ᵦ + σ2_0

        p₀ = log(π)     + logpdf(Normal(0.0, sqrt(σ2_0)), βhat)
        p₁ = log1p(-π)  + logpdf(Normal(0.0, sqrt(σ2_1)), βhat)

        m = max(p₀, p₁)
        prob = exp(p₁ - m) / (exp(p₀ - m) + exp(p₁ - m))
        prob = clamp(prob, 1e-12, 1-1e-12)
        if isnan(prob)
            prob = 0.0
        end
        δ = rand(Bernoulli(prob))
        if δ
            k += 1
            μⱼ = (σ²ᵦ/σ²ₑ) * rⱼ / (1 + σ²ᵦ * xⱼ² / σ²ₑ)
 
            σ²ⱼ = σ²ᵦ*σ²ₑ / (σ²ₑ + σ²ᵦ*xⱼ²)
            β[j] = rand(Normal(μⱼ,√σ²ⱼ))
        else
            β[j] = 0.0
        end
        if β[j] != β_prev
            Xβ[:,j] .= β[j] .* X[:,j]
           predicted .+= (Xβ[:,j] .- X[:,j] .* β_prev) 
        end
        if j <= 5 || j == 144 || j == 261 || j == 444 || j == 665 || j == 812
            println("SNP $j: rⱼ=$rⱼ, xⱼ²=$xⱼ², p₀=$p₀, p₁=$p₁, prob=$prob")
            if j == 144 || j == 261 || j == 444 || j == 665 || j == 812
                println("  ^^^ TRUE CAUSAL SNP")
            end
        end
    end
    e = y - predicted    
    aᵦ = (νᵦ + k)/2
    bᵦ = (νᵦ*S²ᵦ + sum(β .^2))/2
    aₑ = (n + νₑ)/2
    bₑ = (νₑ*S²ₑ + e'e)/2

    σ²ᵦ = rand(InverseGamma(aᵦ, bᵦ))
    σ²ₑ = rand(InverseGamma(aₑ, bₑ))
    π = rand(Beta(b + (p - k), c + k))

    return β,σ²ᵦ,σ²ₑ,π
end

#zorp = generate_bayesb_data(20000,2000,σ²_α =4.0,σ²_e =1.0)
zorp =generate_bayesb_data_h2(20000, 2000, h²_target=0.5, π=0.99)
X = zorp[1]
y = zorp[2]
beta = zeros(2000)
beta = randn(2000)
σ²ᵦ,σ²ₑ,pip = 0.5,0.5,0.5

ps = []
bs=[]
es =[]
betas = zeros(2000)
for i in 1:1000
    println(i)
    beta,σ²ᵦ,σ²ₑ,pip = Bayes_B_Gibbs(X,y,beta,σ²ᵦ,σ²ₑ,pip) 
    push!(ps,pip)
    push!(bs,σ²ᵦ)
    push!(es,σ²ₑ)
    betas .+= (beta .!=0)
end

lines(bs)

lines(betas[zorp[3].β .!=0])

lines(betas)


X, y, params = generate_bayesb_data_h2(5000, 1000, h²_target=0.5, π=0.99)

# Now test your Gibbs sampler
beta = zeros(1000)
σ²ᵦ, σ²ₑ, pip = 1.0, 1.0, 0.99

# Use appropriate hyperparameters based on generated data
S²ᵦ = params.σ²_α  # Use the actual effect variance
S²ₑ = params.σ²_e   # Use the actual residual variance

for i in 1:100
    beta, σ²ᵦ, σ²ₑ, pip = Bayes_B_Gibbs(X, y, beta, σ²ᵦ, σ²ₑ, pip, 
                                        S²ᵦ=S²ᵦ, S²ₑ=S²ₑ)
end

# Check variable selection performance
println("True causal SNPs: ", findall(params.δ .== 1))
println("Estimated causal SNPs: ", findall(abs.(beta) .> 0.01))
