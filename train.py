import os
import random
import shutil
import time
from collections import deque

import cv2
import gym
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from IPython.display import clear_output, display
from skimage import transform

from ddpg import DDPG
from RoadFollowingEnv.car_racing import RoadFollowingEnv
from utils import (FrameStack, ReplayBuffer, VideoRecorder, do_random_exploration,
                   preprocess_frame)
from replay_buffers import PrioritizedReplayBuffer
from stable_baselines.ddpg.noise import OrnsteinUhlenbeckActionNoise

def reward1(state):
    # -10 for driving off-track
    if state.num_contacts == 0: return -10
    # + 1 x throttle
    reward = state.velocity
    reward -= 0.01
    return reward

def make_env(title=None, frame_skip=0):
    env = RoadFollowingEnv(title=title,
                           reward_fn=reward1,
                           preprocess_frame_fn=preprocess_frame,
                           throttle_scale=0.1,
                           steer_scale=0.25,
                           frame_skip=frame_skip)
    env.seed(0)
    return env

def test_agent(test_env, model, n=10, record=False):
    rwds = []
    for j in range(n):
        o, d, ep_ret = model.encode([test_env.reset()])[0], False, 0
        rendered_frame = test_env.render(mode="rgb_array")
        video_recorder = VideoRecorder(os.path.join(model.video_dir, "episode{}.avi".format(model.get_episode_counter())),
                                       frame_size=rendered_frame.shape) if j == 0 and record_eval else None
        while not d:
            # Take deterministic actions at test time (noise_scale=0)
            action = model.predict([o])[0][0]
            o, reward, d, _ = test_env.step(action)
            o = model.encode([o])[0]
            rendered_frame = test_env.render(mode="rgb_array")
            if video_recorder: video_recorder.add_frame(rendered_frame)
            ep_ret += reward
        if video_recorder: video_recorder.release()
        rwds.append(ep_ret)
    return np.mean(rwds)

def train(params, model_name, save_interval=10, eval_interval=10, record_eval=True, restart=False):
    # Create env
    print("Creating environment")
    env      = make_env(model_name, frame_skip=0)#, frame_skip=1)
    test_env = make_env(model_name + " (Test)")

    # Traning parameters
    actor_lr        = params["actor_lr"]
    critic_lr       = params["critic_lr"]
    discount_factor = params["discount_factor"]
    polyak          = params["polyak"]
    initial_std     = params["initial_std"]
    replay_size     = params["replay_size"]
    start_steps     = params["start_steps"]
    batch_size      = params["batch_size"]

    print("")
    print("Training parameters:")
    for k, v, in params.items(): print(f"  {k}: {v}")
    print("")

    # Environment constants
    vae_input_shape  = (84, 84, 1)
    input_shape      = (10,)
    num_actions      = env.action_space.shape[0]
    action_min       = env.action_space.low
    action_max       = env.action_space.high

    # Create model
    print("Creating model")
    model = DDPG(vae_input_shape,
                 input_shape,
                 env.action_space,
                 initial_actor_lr=actor_lr,
                 initial_critic_lr=critic_lr,
                 discount_factor=discount_factor,
                 initial_std=initial_std,
                 lr_decay=1.0,
                 output_dir=os.path.join("models", model_name))

    # Prompt to load existing model if any
    if not restart:
        if os.path.isdir(model.log_dir) and len(os.listdir(model.log_dir)) > 0:
            answer = input("Model \"{}\" already exists. Do you wish to continue (C) or restart training (R)? ".format(model_name))
            if answer.upper() == "C":
                model.load_latest_checkpoint()
            elif answer.upper() == "R":
                shutil.rmtree(model.output_dir)
                for d in model.dirs:
                    os.makedirs(d)
            else:
                raise Exception("There are already log files for model \"{}\". Please delete it or change model_name and try again".format(model_name))
    else:
        shutil.rmtree(model.output_dir)
        for d in model.dirs:
            os.makedirs(d)
    model.init_logging()
    model.write_dict_to_summary("hyperparameters", params, 0)

    # Create replay buffer
    #replay_buffer = ReplayBuffer(input_shapes, num_actions, size=replay_size)
    replay_buffer = PrioritizedReplayBuffer(replay_size, alpha=0.6)

    # Fill replay buffer with data from random exploration
    """do_random_exploration(env, replay_buffer, start_steps)

    # Train for one epoch over replay data
    print("Training inital parameters")
    for i in range(start_steps):
        # Sample mini-batch randomly
        states, taken_actions, rewards, states_next, terminals, w, eid = replay_buffer.sample(batch_size, beta=0.4)

        states = np.expand_dims(np.squeeze(states, axis=1), axis=0)
        states_next = np.expand_dims(np.squeeze(states_next, axis=1), axis=0)

        #assert len(states) == len(input_shapes)
        #for j in range(len(states)):
        #    assert states[j].shape == (batch_size, *input_shapes[j])
        assert taken_actions.shape == (batch_size, num_actions)
        assert rewards.shape == (batch_size,)
        assert terminals.shape == (batch_size,)

        # Optimize network
        deltas = model.train(states, taken_actions, rewards, states_next, terminals, np.squeeze(w))

        replay_buffer.update_priorities(eid, np.abs(deltas) + 1e-6)"""
    noise = OrnsteinUhlenbeckActionNoise(
            mean=np.zeros((2,)),
            sigma=initial_std)
    ep_counter = 0
    # For every episode
    while True:
        episode_counter = model.get_episode_counter()
        print(f"Episode {episode_counter} (Step {model.get_predict_step_counter()})")

        # Save model periodically
        if episode_counter % save_interval == 0:
            model.save()
        
        # Run evaluation periodically
        if episode_counter % eval_interval == 0:
            eval_reward = test_agent(test_env, model, n=1, record=record_eval)
            model.write_value_to_summary("eval/episode_reward", eval_reward, episode_counter)

        # Reset environment
        state, terminal_state, total_reward, total_q, episode_len = model.encode([env.reset()])[0], False, 0, 0, 0
        noise.reset()

        # While episode not done
        while not terminal_state:
            # Sample action given state
            action, q_value = model.predict([state], greedy=True, write_to_summary=True)
            action, q_value = action[0], q_value[0]
            action = np.clip(action+noise(), env.action_space.low, env.action_space.high)
            total_q += q_value

            # Perform action
            new_state, reward, terminal_state, _ = env.step(action)
            new_state = model.encode([new_state])[0]
            env.render()
            total_reward += reward
            episode_len  += 1

            # Store tranisition
            replay_buffer.add(state, action, reward, new_state, terminal_state)
            state = new_state

        # Train for one epoch over replay data
        print("Training parameters")
        if ep_counter > 10:
            for i in range(500):
                if i % 1000 == 0:
                    print("Iteration {}/{}".format(i, 3000))

                # Sample mini-batch randomly
                states, taken_actions, rewards, states_next, terminals, w, eid = replay_buffer.sample(batch_size, beta=0.4)

                assert states.shape == (batch_size, *input_shape)
                assert states_next.shape == (batch_size, *input_shape)
                assert taken_actions.shape == (batch_size, num_actions)
                assert rewards.shape == (batch_size,)
                assert terminals.shape == (batch_size,)

                # Optimize network
                deltas = model.train(states, taken_actions, rewards, states_next, terminals, np.squeeze(w))

                replay_buffer.update_priorities(eid, np.abs(deltas) + 1e-6)

        ep_counter += 1
        # Write episodic values
        model.write_value_to_summary("train/episode_reward", total_reward, episode_counter)
        model.write_value_to_summary("train/episodic/q_value", total_q, episode_counter)
        model.write_episodic_summaries()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Trains an agent in a the RoadFollowing environment")

    # Hyper parameters
    parser.add_argument("--actor_lr", type=float, default=1e-4)
    parser.add_argument("--critic_lr", type=float, default=1e-3)
    parser.add_argument("--discount_factor", type=float, default=0.9)
    parser.add_argument("--polyak", type=float, default=0.999)
    parser.add_argument("--initial_std", type=float, default=0.4)
    parser.add_argument("--replay_size", type=int, default=int(1e4))
    parser.add_argument("--start_steps", type=int, default=int(1e4))
    parser.add_argument("--batch_size", type=int, default=64)

    # Training vars
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save_interval", type=int, default=10)
    parser.add_argument("--eval_interval", type=int, default=10)
    parser.add_argument("--record_eval", type=bool, default=True)
    parser.add_argument("-restart", action="store_true")

    params = vars(parser.parse_args())

    # Remove non-hyperparameters
    model_name = params["model_name"]; del params["model_name"]
    seed = params["seed"]; del params["seed"]
    save_interval = params["save_interval"]; del params["save_interval"]
    eval_interval = params["eval_interval"]; del params["eval_interval"]
    record_eval = params["record_eval"]; del params["record_eval"]
    restart = params["restart"]; del params["restart"]

    # Reset tf and set seed
    tf.reset_default_graph()
    if isinstance(seed, int):
        tf.random.set_random_seed(seed)
        np.random.seed(seed)

    # Call main func
    train(params, model_name,
            save_interval=save_interval,
            eval_interval=eval_interval,
            record_eval=record_eval,
            restart=restart)
