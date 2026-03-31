```text


Input: 
    env, 
    agent (PPO, DQN, A2C, ...)


Initialize:

    reward_model ← Ensemble_Reward_Model, Reward_Net, ...

    wrapped_env ← Reward_wrapper(env, reward_model)

    fragment ← Active fragmenter, Random fragmenter

Train:

    1) policy worker
    
        wrapped_env is updated with the new reward_model from preference worker

        train(agent, wrapped_env) 


    2) preference worker

        trajectories collected during training from policy worker

        pairs ← fragment(trajectories, num_pairs)

        preferences ← label(pairs)

        dataset ← dataset ∪ (pairs, preferences)

        train(reward_model, dataset)


    end


Notes:

    



```