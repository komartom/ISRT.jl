module ISRT

using Random, StatsBase, IterTools, LinearAlgebra

export Model, miSVM, maxinst


struct Options

    n_trees::Int
    n_subfeat::Int
    n_thresholds::Int
    max_depth::Int
    min_samples_leaf::Int
    min_samples_split::Int
    bagging::Float64
    lambdas::Vector{Float32}
    sparse::Bool
    epochs::Int

end


struct Split

    cost::Float64
    feature::Int
    threshold::Float32
    selector::Vector{Float32}

    n_left::Int
    n_right::Int

end

Split() = Split(Inf, 0, 0.0f0, [0.0f0], 0, 0)

function maxinst(bag::Matrix{Float32}, w::Vector{Float32})
   
    max_ii, max_val = 0, -Inf32
    for ii in 1:size(bag, 1)
        val = 0.0f0
        for ff in 1:size(bag, 2)
            val += bag[ii, ff] * w[ff]
        end
        if val > max_val
            max_val = val
            max_ii = ii
        end
    end
    
    return max_ii, max_val
    
end

function (split::Split)(bag::Matrix{Float32})

    instance, _ = maxinst(bag, split.selector)
    return bag[instance, split.feature] < split.threshold

end


mutable struct Node

    depth::Int
    is_leaf::Bool
    probability::Float64

    left::Node
    right::Node
    split::Split
    parent::Node
    samples::Vector{Int}

    Node(depth, samples) = (
        node = new();
        node.depth = depth;
        node.samples = samples;
        node)

end


function entropy_loss(V, y, n_pos_samples, n_neg_samples, n_samples, feature, threshold, selector)

    n_left_pos = 0
    n_left_neg = 0

    for ii in 1:n_samples
        if V[ii] < threshold
            if y[ii]
                n_left_pos += 1
            else
                n_left_neg += 1
            end
        end
    end

    n_left = n_left_pos + n_left_neg
    w_left = n_left / n_samples
    p_left_pos = n_left_pos / n_left
    p_left_neg = 1.0 - p_left_pos
    entropy_left = (p_left_neg == 0.0) ? 0.0 : -p_left_neg*log2(p_left_neg)
    entropy_left += (p_left_pos == 0.0) ? 0.0 : -p_left_pos*log2(p_left_pos)

    n_right_pos = n_pos_samples - n_left_pos
    n_right_neg = n_neg_samples - n_left_neg

    n_right = n_right_pos + n_right_neg
    w_right = n_right / n_samples
    p_right_pos = n_right_pos / n_right
    p_right_neg = 1.0 - p_right_pos
    entropy_right = (p_right_neg == 0.0) ? 0.0 : -p_right_neg*log2(p_right_neg)
    entropy_right += (p_right_pos == 0.0) ? 0.0 : -p_right_pos*log2(p_right_pos)

    cost = (w_left * entropy_left) + (w_right * entropy_right)

    return Split(cost, feature, threshold, selector, n_left, n_right)

end


function miSVM_update_wb!(w, b, t, λ, bag, label, ind)
    
    maxi, maxv = maxinst(bag, w)
    
    η = 1.0f0 / (λ * t)
    α = 1.0f0 - η * λ
    if (label * (maxv + b)) < 1.0f0
        β = label * η
        for ii in ind
            w[ii] = α * w[ii] + β * bag[maxi, ii]
        end
        return α * b + β
    else
        for ii in ind
            w[ii] = α * w[ii]
        end
        return α * b
    end
    
end


function miSVM(rng::AbstractRNG, X::Vector{Matrix{Float32}}, Y::AbstractArray{Bool}, λ::Float32=1.0f0, epochs=100, sparse=true)
   
    labels = [yy ? 1.0f0 : -1.0f0 for yy in Y]

    n_features = size(X[1], 2)
    w, b = zeros(Float32, n_features), randn(rng, Float32)
    ind = sample(rng, 1:n_features, sparse ? round(Int, sqrt(n_features)) : n_features, replace=false, ordered=true)
    for ii in ind
        w[ii] = randn(rng, Float32)
    end

    t = 1
    randbag() = rand(rng, 1:length(X))
    for _ in 1:epochs, bb in repeatedly(randbag, length(X))        
        b = miSVM_update_wb!(w, b, t, λ, X[bb], labels[bb], ind)
        t += 1
    end
    
    return w, b
    
end


function split!(rng, node, X, Y, features, opt)

    Y_node = Y[node.samples]
    n_samples = length(node.samples)
    n_pos_samples = sum(Y_node)
    n_neg_samples = n_samples - n_pos_samples
    node.probability = n_pos_samples / n_samples

    if node.depth == opt.max_depth ||
        node.probability == 1.0f0 ||
        node.probability == 0.0f0 ||
        n_samples < opt.min_samples_split ||
        n_samples == opt.min_samples_leaf
        node.is_leaf = true
        return
    end

    best_split = Split()

    X_node = X[node.samples]
    I = Vector{Int}(undef, n_samples)
    V = Vector{Float32}(undef, n_samples)
    selectors = Vector{Vector{Float32}}(undef, length(opt.lambdas) + (isdefined(node, :parent) ? 1 : 0))

    for (ll, lambda) in enumerate(opt.lambdas)
        selectors[ll] = miSVM(rng, X_node, Y_node, lambda, opt.epochs, opt.sparse)[1]
    end

    if isdefined(node, :parent)
        selectors[end] = node.parent.split.selector
    end

    for selector in selectors

        for (ii, bag) in enumerate(X_node)
            I[ii], _ = maxinst(bag, selector)
        end
        
        mtry = 1
        for feature in shuffle!(rng, features)

            minv, maxv = Inf32, -Inf32
            for (ii, bag) in enumerate(X_node)
                val = bag[I[ii], feature]
                if val < minv; minv = val; end
                if val > maxv; maxv = val; end
                V[ii] = val
            end

            if minv == maxv; continue; end
            scale = maxv - minv

            for ii in 1:opt.n_thresholds
                threshold = rand(rng, Float32) * scale + minv
                split = entropy_loss(V, Y_node, n_pos_samples, n_neg_samples, n_samples, feature, threshold, selector)
                if split.cost < best_split.cost
                    best_split = split
                end
            end

            if mtry == opt.n_subfeat
                break
            else
                mtry += 1
            end

        end

    end

    if best_split.feature == 0 ||
        best_split.n_left < opt.min_samples_leaf ||
        best_split.n_right < opt.min_samples_leaf
        node.is_leaf = true
        return
    end

    node.is_leaf = false
    node.split = best_split

    ll, rr = 1, 1
    left_samples = Vector{Int}(undef, best_split.n_left)
    right_samples = Vector{Int}(undef, best_split.n_right)
    for (ss, bag) in zip(node.samples, X_node)
        if best_split(bag)
            left_samples[ll] = ss
            ll += 1
        else
            right_samples[rr] = ss
            rr += 1
        end
    end

    node.left = Node(node.depth + 1, left_samples)
    node.left.parent = node

    node.right = Node(node.depth + 1, right_samples)
    node.right.parent = node

    return

end


function tree_builder(X::Vector{Matrix{Float32}}, Y::AbstractArray{Bool}, opt::Options, seed::Int=1234)
    rng = MersenneTwister(seed)

    n_samples, n_features = length(X), size(X[1], 2)

    features = collect(1:n_features)

    samples = (opt.bagging > 0.0 
        ? sort(rand(rng, 1:n_samples, round(Int, opt.bagging * n_samples))) 
        : collect(1:n_samples))

    root = Node(1, samples)
    stack = Node[root]
    while length(stack) > 0
        node = pop!(stack)
        split!(rng, node, X, Y, features, opt)
        if !node.is_leaf
            push!(stack, node.left, node.right)
        end
    end

    return root

end


struct Model

    trees::Vector{Node}

    function Model(
                    X                   ::Vector{Matrix{Float32}}, 
                    Y                   ::AbstractArray{Bool}; 
                    n_trees             ::Int=100, 
                    n_subfeat           ::Int=0, 
                    n_thresholds        ::Int=1, 
                    max_depth           ::Int=-1, 
                    min_samples_leaf    ::Int=1,
                    min_samples_split   ::Int=2,
                    bagging             ::Float64=0.0,
                    lambdas             ::Vector{Float32}=[1.0f0],
                    sparse              ::Bool=true,
                    epochs              ::Int=10,
                    seed                ::Int=1234)

        @assert length(X) == length(Y)
        @assert n_trees >= 1
        @assert n_thresholds >= 1
        @assert min_samples_leaf >= 1
        @assert min_samples_split >= 1
        @assert length(lambdas) >= 1
        @assert bagging >= 0.0
        @assert epochs >= 1
        @assert seed > 0

        if !(1 <= n_subfeat <= size(X[1], 2))
            n_subfeat = round(Int, sqrt(size(X[1], 2)))
        end

        opt = Options(n_trees, n_subfeat, n_thresholds, max_depth, min_samples_leaf, min_samples_split, bagging, lambdas, sparse, epochs)

        trees = Vector{Node}(undef, n_trees)
        seeds = abs.(rand(MersenneTwister(seed), Int, n_trees))
        Threads.@threads for tt in 1:n_trees
            trees[tt] = tree_builder(X, Y, opt, seeds[tt])
        end

        return new(trees)

    end

end


function (node::Node)(sample::Matrix{Float32})

    while !node.is_leaf
        node = node.split(sample) ? node.left : node.right
    end

    return node.probability

end


function (model::Model)(sample::Matrix{Float32})

    probability = 0.0
    for tree in model.trees
        probability += tree(sample)
    end

    return probability / length(model.trees)

end


function (model::Model)(X::Vector{Matrix{Float32}})

    scores = Vector{Float64}(undef, length(X))
    Threads.@threads for bb in 1:length(X)
        scores[bb] = model(X[bb])
    end

    return scores

end


# predictions with instance level distributions

function (node::Node)(sample::Matrix{Float32}, instdist::Bool)

    distribution = zeros(Float64, size(sample, 1))

    while !node.is_leaf
        instance, _ = maxinst(sample, node.split.selector)
        distribution[instance] += 1.0
        node = (sample[instance, node.split.feature] < node.split.threshold) ? node.left : node.right
    end

    return node.probability, distribution ./ sum(distribution)

end


function (model::Model)(sample::Matrix{Float32}, instdist::Bool)

    probability = 0.0
    distribution = zeros(Float64, size(sample, 1))

    for tree in model.trees
        tree_probability, tree_distribution = tree(sample, true)
        distribution .+= tree_distribution
        probability += tree_probability
    end

    return probability / length(model.trees), distribution ./ length(model.trees)

end


function (model::Model)(X::Vector{Matrix{Float32}}, instdist::Bool)

    scores = Vector{Float64}(undef, length(X))
    distributions = Vector{Vector{Float64}}(undef, length(X))

    Threads.@threads for bb in 1:length(X)
        score, distribution = model(X[bb], true)
        scores[bb] = score
        distributions[bb] = distribution
    end

    return scores, distributions

end

end #module