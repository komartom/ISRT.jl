# Instance Selection Randomized Trees (ISRT.jl)
Explainable Multiple Instance Learning with Instance Selection Randomized Trees

## Evaluation protocol
5-times repeated 10-fold Cross-Validation
```julia
include("./src/ISRT.jl")
using MIDatasets, ROCAnalysis, Statistics, ProgressMeter

function evaluate(dataset)
    
    X, Y, folds = midataset(dataset, folds=true)
    
    AUCs = zeros(10, 5)
    @showprogress 1 "Computing..." for rr in 1:5, ff in 1:10
        
        Xtrain = X[folds[rr][ff]]
        Ytrain = Y[folds[rr][ff]]

        Xtest = X[.!folds[rr][ff]]
        Ytest = Y[.!folds[rr][ff]]

        model = ISRT.Model(Xtrain, Ytrain, n_trees=500, n_thresholds=8, epochs=1, seed=1234)
        scores = model(Xtest)
        
        AUCs[ff, rr] = auc(roc(scores[.!Ytest], scores[Ytest]))

    end
    
    return (dataset=dataset, mean=mean(mean(AUCs, dims=1))*100, std=std(mean(AUCs, dims=1))*100, AUCs=AUCs)
   
end
```

## Evaluate ISRT on Musk1
```julia
evaluate("Musk1") # Should return 97.2 (1.3)
```