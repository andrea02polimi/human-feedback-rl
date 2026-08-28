"""The pieces HybridAlgorithm is assembled from, one concern per file.

    feedback_collection     asking the oracle, counting what comes back
    reward_model_training   fitting the reward to the feedback so far
    gradient_fusion         two gradients into one optimizer step
    reliability_weight      estimating alpha once per iteration
    alpha_estimation        the maths behind alpha
    demonstration_losses    the two demonstration IRL losses
    reward_training         gradient norms, reward normalization
    imitation_metrics       agent-versus-expert error
    reward_diagnostics      what is only logged, in five files
"""
