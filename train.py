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
from skimage import transform
from stable_baselines.ddpg.noise import OrnsteinUhlenbeckActionNoise

from ddpg import DDPG
from replay_buffers import PrioritizedReplayBuffer
from RoadFollowingEnv.car_racing import RoadFollowingEnv
from utils import VideoRecorder, preprocess_frame

def reward1(state):
    # -10 for driving off-track
    if state.num_contacts == 0: return -10
    # + 1 x throttle
    reward = state.velocity
    reward -= 0.01
    return reward

def create_encode_state_fn(model, with_measurements=False, stack=None):
    def encode_state(state):
        frame = preprocess_frame(state.frame)
        encoded_state = model.encode([frame])[0]
        if with_measurements:
            encoded_state = np.append(encoded_state, [state.throttle, state.steering, state.velocity / 30.0])
        if isinstance(stack, int):
            s1 = np.array(encoded_state)
            if not hasattr(state, "stack"):
                state.stack = [np.zeros_like(encoded_state) for _ in range(stack)]
                state.stack_idx = 0
            state.stack[state.stack_idx % stack] = s1
            state.stack_idx += 1
            concat_state = np.concatenate(state.stack)
            return concat_state
        return np.array(encoded_state)
    return encode_state

def make_env(title=None, frame_skip=0):
    env = RoadFollowingEnv(title=title,
                           reward_fn=reward1,
                           throttle_scale=0.1,
                           #steer_scale=0.25,
                           max_speed=30,
                           frame_skip=frame_skip)
    env.seed(0)
    return env

def test_agent(test_env, model, record=False):
    # Init test env
    state, terminal, total_reward = test_env.reset(), False, 0
    rendered_frame = test_env.render(mode="rgb_array")

    # Init video recording
    video_filename = os.path.join(model.video_dir, "episode{}.avi".format(model.get_episode_counter()))
    video_recorder = VideoRecorder(video_filename, frame_size=rendered_frame.shape)
    video_recorder.add_frame(rendered_frame)

    # While non-terminal state
    while not terminal:
        # Take deterministic actions at test time (noise_scale=0)
        action = model.predict([state], greedy=True)[0][0]
        state, reward, terminal, _ = test_env.step(action)

        # Add frame
        rendered_frame = test_env.render(mode="rgb_array")
        if video_recorder: video_recorder.add_frame(rendered_frame)
        total_reward += reward

    # Release video
    if video_recorder:
        video_recorder.release()
    
    return total_reward, test_env.reward

def train(params, model_name, save_interval=10, eval_interval=10, record_eval=True, restart=False):
    # Create env
    print("Creating environment")
    env      = make_env(model_name, frame_skip=0)
    test_env = make_env(model_name + " (Test)")

    # Traning parameters
    actor_lr                 = params["actor_lr"]
    critic_lr                = params["critic_lr"]
    discount_factor          = params["discount_factor"]
    polyak                   = params["polyak"]
    initial_std              = params["initial_std"]
    grad_norm                = params["grad_norm"]
    replay_size              = params["replay_size"]
    replay_alpha             = params["replay_alpha"]
    replay_beta              = params["replay_beta"]
    start_steps              = params["start_steps"]
    batch_size               = params["batch_size"]
    num_episodes             = params["num_episodes"]
    train_steps_per_episode  = params["train_steps_per_episode"]
    num_exploration_episodes = params["num_exploration_episodes"]

    print("")
    print("Training parameters:")
    for k, v, in params.items(): print(f"  {k}: {v}")
    print("")

    with_measurements = False#True
    stack = None

    # Environment constants
    vae_input_shape  = (84, 84, 1)
    input_shape      = np.array([10])
    if with_measurements: input_shape[0] += 3
    if stack is not None: input_shape[0] *= stack
    num_actions      = env.action_space.shape[0]
    action_min       = env.action_space.low
    action_max       = env.action_space.high
    action_noise     = OrnsteinUhlenbeckActionNoise(mean=np.array([0.0, 0.5]), sigma=initial_std)#, dt=0.2)
    #action_noise = NormalActionNoise(mean=np.zeros((2,)), sigma=initial_std)

    # Create model
    print("Creating model")
    model = DDPG(vae_input_shape,
                 input_shape,
                 env.action_space,
                 action_noise,
                 initial_actor_lr=actor_lr,
                 initial_critic_lr=critic_lr,
                 discount_factor=discount_factor,
                 polyak=polyak,
                 lr_decay=1.0,
                 grad_norm=grad_norm,
                 output_dir=os.path.join("models", model_name))

    # Set state encoding fn
    encode_state_fn = create_encode_state_fn(model, with_measurements=with_measurements, stack=stack)
    env.encode_state_fn = encode_state_fn
    test_env.encode_state_fn = encode_state_fn

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
    replay_buffer = PrioritizedReplayBuffer(replay_size, alpha=replay_alpha)
    #replay_buffer = ReplayBuffer(replay_size)
    episode_counter = -num_exploration_episodes

    # For every episode
    while episode_counter < num_episodes:
        print(f"Episode {episode_counter} (Step {model.get_predict_step_counter()})")

        # Save model periodically
        if episode_counter % save_interval == 0:
            model.save()
        
        # Run evaluation periodically
        if episode_counter % eval_interval == 0:
            eval_reward, eval_score = test_agent(test_env, model, record=record_eval)
            model.write_value_to_summary("eval/episodic/score", eval_score, episode_counter)
            model.write_value_to_summary("eval/episodic/reward", eval_reward, episode_counter)

        # Reset environment
        state, terminal_state, total_reward, total_q = env.reset(), False, 0, 0
        
        # While episode not done
        while not terminal_state:
            if episode_counter < 0:
                action = np.array([-1.0+np.random.rand()*2.0, 0.7+np.random.rand()*0.3])
                q_value = 0
            else:
                # Sample action given state
                action, q_value = model.predict([state], greedy=False, write_to_summary=True)
                action, q_value = action[0], q_value[0]
            total_q += q_value

            env.value_label.text = "Q(s, a)={:.2f}".format(model.sess.run(model.Q_value_of_actor, feed_dict={model.input_states: [state]})[0])

            # Perform action
            new_state, reward, terminal_state, _ = env.step(action)
            env.render()
            total_reward += reward

            # Store tranisition
            replay_buffer.add(state, action, reward, new_state, terminal_state)
            state = new_state

        # Train for one epoch over replay data
        print("Training...")
        n = train_steps_per_episode
        for i in range(n):
            if i % (train_steps_per_episode // 10) == 0:
                print("{}%".format(i / train_steps_per_episode * 100))

            # Sample mini-batch randomly
            states, taken_actions, rewards, states_next, terminals, w, eid = replay_buffer.sample(batch_size, beta=replay_beta)

            assert states.shape == (batch_size, *input_shape)
            assert states_next.shape == (batch_size, *input_shape)
            assert taken_actions.shape == (batch_size, num_actions)
            assert rewards.shape == (batch_size,)
            assert terminals.shape == (batch_size,)

            # Optimize network
            deltas = model.train(states, taken_actions, rewards, states_next, terminals, w)

            replay_buffer.update_priorities(eid, np.abs(deltas) + 1e-6)

        # Write episodic values
        model.write_value_to_summary("train/episodic/score", env.reward, episode_counter)
        model.write_value_to_summary("train/episodic/reward", total_reward, episode_counter)
        model.write_value_to_summary("train/episodic/q_value", total_q, episode_counter)
        model.write_episodic_summaries(episode_counter)
        episode_counter += 1

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Trains an agent in a the RoadFollowing environment")

    # Hyper parameters
    parser.add_argument("--actor_lr", type=float, default=1e-4)
    parser.add_argument("--critic_lr", type=float, default=1e-3)
    parser.add_argument("--discount_factor", type=float, default=0.9)
    parser.add_argument("--polyak", type=float, default=0.995)
    parser.add_argument("--initial_std", type=float, default=0.4)
    parser.add_argument("--grad_norm", type=float, default=5e-3)
    parser.add_argument("--replay_size", type=int, default=int(1e4))
    parser.add_argument("--replay_alpha", type=float, default=0.6)
    parser.add_argument("--replay_beta", type=float, default=0.4)
    parser.add_argument("--start_steps", type=int, default=int(1e4))
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_episodes", type=int, default=200)
    parser.add_argument("--num_exploration_episodes", type=int, default=10)
    parser.add_argument("--train_steps_per_episode", type=int, default=500)

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
        random.seed(0)

    # Call main func
    train(params, model_name,
          save_interval=save_interval,
          eval_interval=eval_interval,
          record_eval=record_eval,
          restart=restart)
