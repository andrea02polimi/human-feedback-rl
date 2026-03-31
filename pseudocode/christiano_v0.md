```text

Input:
    env
    agent  // PPO, DQN, A2C, ...

Initialize:
    reward_model ← EnsembleRewardModel
    wrapped_env ← RewardWrapper(env, reward_model)
    fragmenter ← ActiveFragmenter or RandomFragmenter
    dataset ← ∅

Training:

    for iteration = 1 ... N:

        trajectories ← collect_rollouts(env, agent)

        segments ← fragmenter.split(trajectories)
        selected_segments ← fragmenter.select(segments, num_pairs * 2)

        pairs ← pair_randomly(selected_segments)
        preferences ← query_labels(pairs)

        dataset ← dataset ∪ (pairs, preferences)

        train(reward_model, dataset)
        train(agent, wrapped_env)

    end


```

## Notes

- **Pairs number schedule** 
  - The number of queried pairs follows an inverse decay over training progress: value(p) = final + (initial - final) / (1 + k * p)
  - > At the beginning of training we compare a number of trajectory segments drawn from rollouts of an
untrained (randomly initialized) policy. In the Atari domain we also pretrain the reward predictor
for 200 epochs before beginning RL training, to reduce the likelihood of irreversibly learning a bad
policy based on an untrained predictor. For the rest of training, labels are fed in at a rate decaying
inversely with the number of timesteps; after twice as many timesteps have elapsed, we answer about
half as many queries per unit time.


- **Fragmentation**
  - Split each trajectory into fixed-length segments
  - Compute reward variance across ensembles for each segment `σ`
  - Select the top `2 × num_pairs` most uncertain segments
  - Randomly pair them to form queries

- **Dataset**
  - Split into training and validation sets

- **Reward model**
  - Trained with cross-entropy loss on preference labels


