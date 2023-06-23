import ray
from ray import tune
from ray.rllib.algorithms import td3
from ray.rllib.algorithms.td3.td3 import TD3Trainer
from ray.rllib.execution.replay_buffer import LocalReplayBuffer
from ray.rllib.execution.rollout_ops import ParallelRollouts
from ray.rllib.execution.train_ops import TrainOneStep, UpdateTargetNetwork
from ray.rllib.execution.metric_ops import StandardMetricsReporting

if __name__ == '__main__':
    # Initialize Ray
    ray.init()

    # Define the environment configuration
    env_config = {"env": "Pendulum-v0"}

    # Define the TD3 agent configuration
    agent_config = td3.TD3_DEFAULT_CONFIG.copy()
    agent_config.update({
        "twin_q": True,
        "policy_delay": 2,
        "smooth_target_policy": True,
        "target_noise_clip": 0.5,
        "actor_hiddens": [400, 300],
        "critic_hiddens": [400, 300],
        "n_step": 3,
        "buffer_size": 50000,
        "learning_starts": 1000,
        "rollout_fragment_length": 100,
        "train_batch_size": 256,
        "target_network_update_freq": 0,
        "tau": 0.005,
        "prioritized_replay": True,
        "prioritized_replay_alpha": 0.6,
        "prioritized_replay_beta": 0.4,
        "final_prioritized_replay_beta": 0.4,
        "prioritized_replay_beta_annealing_timesteps": 20000,
        "prioritized_replay_eps": 1e-6,
    })


    # Define the training pipeline
    def execution_plan(workers, config):
        rollouts = ParallelRollouts(workers, mode="bulk_sync")
        replay_buffer = LocalReplayBuffer(
            num_shards=1,
            learning_starts=config["learning_starts"],
            buffer_size=config["buffer_size"],
            replay_batch_size=config["rollout_fragment_length"])

        store_op = rollouts \
            .for_each(lambda batch: replay_buffer.add_batch(batch)) \
            .for_each(lambda x: replay_buffer.replay())

        train_op = store_op \
            .combine(rollouts) \
            .for_each(TrainOneStep(workers)) \
            .for_each(UpdateTargetNetwork(
            workers, config["tau"], config["target_network_update_freq"]))

        return StandardMetricsReporting(train_op, workers, config)


    # Register the custom execution plan
    TD3Trainer.execution_plan = execution_plan

    # Train the agent
    analysis = tune.run(
        TD3Trainer,
        config=agent_config,
        stop={"training_iteration": 100},
        checkpoint_at_end=True
    )

    # Get the best trained agent
    best_agent = analysis.get_best_trial(metric="episode_reward_mean", mode="max")

    # Print the best agent's hyperparameters
    print(best_agent.config)

