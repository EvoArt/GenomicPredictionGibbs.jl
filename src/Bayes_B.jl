function Bayes_B_Gibbs_step(X,y,β,σ²ᵦ,σ²ₑ,π;b=1,c=1,S²ᵦ = var(y) *0.1,S²ₑ = var(y),νᵦ=3,νₑ=3)
    n,p = size(X)
    Xβ = X .* transpose(β)
    predicted = vec(sum(Xβ,dims = 2))
    k = 0 # keep track of non-zero β coefficients
    n_nan = 0 

    # update β
    for j in 1:p
        # precompute/ store values
        β_prev = β[j]
        xⱼ = X[:,j]
        xⱼ² = dot(xⱼ, xⱼ)
        X₍₋ⱼ₎β₍₋ⱼ₎ = predicted .- Xβ[:,j]
        rⱼ = dot(xⱼ, (y - X₍₋ⱼ₎β₍₋ⱼ₎))
        β̂  = rⱼ / xⱼ²  
        σ²₀  = σ²ₑ / xⱼ² 
        σ²₁  = σ²ᵦ + σ²₀

        # calculate non-zero probability
        # work with logs to avoid underflow
        p₀ = log(π)     + logpdf(Normal(0.0, √σ²₀), β̂)
        p₁ = log1p(-π)  + logpdf(Normal(0.0, √σ²₁), β̂)
        # logsumexp
        m = max(p₀, p₁)
        prob = exp(p₁ - m) / (exp(p₀ - m) + exp(p₁ - m))
        prob = clamp(prob, 1e-12, 1-1e-12)
        if isnan(prob) 
            prob = 0.0
           n_nan += 1
        end
        δ = rand(Bernoulli(prob))
        if δ
            # draw non-zero coefficient value
            k += 1
            μⱼ = (σ²ᵦ/σ²ₑ) * rⱼ / (1 + σ²ᵦ / σ²₀)
            σ²ⱼ = σ²ᵦ*σ²ₑ / (σ²ₑ + σ²ᵦ*xⱼ²)
            β[j] = rand(Normal(μⱼ,√σ²ⱼ))
        else
            β[j] = 0.0
        end
        if β[j] != β_prev
            # update `predicted` incrementally 
            Xβ[:,j] .= β[j] .* X[:,j]
           predicted .+= (Xβ[:,j] .- X[:,j] .* β_prev) 
        end
    end

    # draw variance and prob-zero params
    e = y - predicted
    aᵦ = (νᵦ + k)/2
    bᵦ = (νᵦ*S²ᵦ + sum(β .^2))/2
    aₑ = (n + νₑ)/2
    bₑ = (νₑ*S²ₑ + e'e)/2

    σ²ᵦ = rand(InverseGamma(aᵦ, bᵦ))
    σ²ₑ = rand(InverseGamma(aₑ, bₑ))
    π = rand(Beta(b + (p - k), c + k))

    if n_nan >0
        println(println("Warning: $(n_nan) numerical errors during SNP inclusion probability calculations."))
    end

    return β,σ²ᵦ,σ²ₑ,π
end


function Bayes_B_Gibbs_run(X,y,n_iter;b=1,c=1,S²ᵦ = var(y) *0.1,S²ₑ = var(y),νᵦ=3,νₑ=3)

    n,p = size(X)
    β_res = Array{Float64}(undef,p,n_iter)
    σ²ᵦ_res = Vector{Float64}(undef,n_iter)
    σ²ₑ_res = Vector{Float64}(undef,n_iter)
    π_res = Vector{Float64}(undef,n_iter)

    β = zeros(p)
    σ²ᵦ = rand(InverseGamma(νᵦ, νᵦ*S²ᵦ))
    σ²ₑ = rand(InverseGamma(νₑ, νₑ*S²ₑ))
    π = rand(Beta(b,c))

    for iter in 1:n_iter
        β,σ²ᵦ,σ²ₑ,π = Bayes_B_Gibbs_step(X,y,β,σ²ᵦ,σ²ₑ,π;b=b,c=c,S²ᵦ = S²ᵦ,S²ₑ = S²ₑ,νᵦ=νᵦ,νₑ=νₑ)
        β_res[:,iter] .= β
        σ²ᵦ_res[iter] = σ²ᵦ
        σ²ₑ_res[iter] = σ²ₑ
        π_res[iter] = π
    end
    return (β = β_res ,σ²ᵦ = σ²ᵦ_res ,σ²ₑ = σ²ₑ_res,π = π_res)
end
