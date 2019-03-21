import os
import re
import shutil

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from utils import build_mlp, create_polyak_update_ops, create_counter_variable, create_mean_metrics_from_dict, clip_grad
from vae import VAE


class DDPG():
    """
        Deep deterministic policy gradient implementation
    """

    def __init__(self, vae_input_shape, input_shape, action_space, action_noise,
                 initial_actor_lr=3e-4, initial_critic_lr=3e-4, lr_decay=0.998,
                 discount_factor=0.99, polyak=0.995, grad_norm=5e-3,
                 output_dir="./"):
        """
            input_shapes (list):
                List of the input shapes.
                E.g. input_shapes[0] = (width, height, depth)
            action_space (gym.Box):
                A class where:
                    action_space.low   = Array of minimum possible value for each action
                    action_space.high  = Array of maximum possible value for each action
                    action_space.shape = Number of actions in the action space
            output_dir (string):
                Name of the model
        """

        # Verify action space
        assert len(action_space.shape) == 1, "The implementation supports only 1D action, continous spaces"
        num_actions = action_space.shape[0]
        
        self.action_noise = action_noise
        self.action_space = action_space

        # Create counters
        self.train_step_counter   = create_counter_variable(name="train_step_counter")
        self.predict_step_counter = create_counter_variable(name="predict_step_counter")
        self.episode_counter      = create_counter_variable(name="episode_counter")

        # Create placeholders
        self.input_states      = tf.placeholder(shape=(None, *input_shape), dtype=tf.float32, name="input_state_placeholder")      # s
        self.input_states_next = tf.placeholder(shape=(None, *input_shape), dtype=tf.float32, name="input_next_state_placeholder") # s'
        self.taken_actions     = tf.placeholder(shape=(None, num_actions), dtype=tf.float32, name="taken_action_placeholder")      # a
        self.rewards           = tf.placeholder(shape=(None,), dtype=tf.float32, name="rewards_placeholder")                       # r
        self.terminals         = tf.placeholder(shape=(None,), dtype=tf.float32, name="terminals_placeholder")                     # d
        self.is_weights        = tf.placeholder(shape=(None,), dtype=tf.float32, name="is_weights_placeholder")                    # w

        # Load pre-trained variational autoencoder
        z_dim = 10
        self.vae = VAE(input_shape=vae_input_shape,
                       z_dim=z_dim, model_type="mlp",
                       model_name="bce_mlp_zdim10_beta4_data10k_v2",
                       model_dir="train_vae/", training=False)
        self.vae_mean_squashed = tf.nn.sigmoid(self.vae.mean)

        states = self.input_states
        states_next = self.input_states_next
        hidden_sizes=(500,300)

        #with tf.variable_scope("rdn_target"):
        #    target = build_mlp(states_next, [500, 300], tf.nn.leaky_relu, tf.nn.sigmoid, trainable=False)
        #with tf.variable_scope("rdn"):
        #    prediction = build_mlp(states_next, [10, 300], tf.nn.leaky_relu, tf.nn.sigmoid, trainable=True)
        #self.rdn_diff = tf.reduce_mean((target - prediction)**2, axis=-1)
        #self.rdn_diff_norm = tf.clip_by_value(self.rdn_diff / 1e-4, 0, 1.0)
        #intrinsic_reward = self.rdn_diff_norm
        #self.rdn_loss = tf.reduce_mean(self.rdn_diff)

        with tf.variable_scope("main"):
            # μ(s; θ), Q(s, a; ϕ), Q(s, μ(s; θ); ϕ)
            self.actor_mean, self.Q_value, self.Q_value_of_actor = self.build_mlp_actor_critic(states, self.taken_actions, action_space, hidden_sizes=hidden_sizes)

        with tf.variable_scope("target"):
            # μ(s'; θ_{targ}), _, Q(s', μ(s'; θ_{targ}); ϕ_{targ})
            self.target_actor_mean, _, self.Q_target_value = self.build_mlp_actor_critic(states_next, self.taken_actions, action_space, hidden_sizes=hidden_sizes)

        # Create polyak update ops
        self.update_target_params_op, init_target_params_op  = create_polyak_update_ops("main/", "target/", polyak=polyak)

        # Critic (MSBE) loss = min_θ mse(Q(s, a), r + gamma * Q(s', μ(s'; θ_{targ}); ϕ_{targ}))
        self.total_reward = self.rewards + intrinsic_reward
        self.Q_target     = tf.stop_gradient(self.total_reward + discount_factor * (1.0 - self.terminals) * self.Q_target_value)
        self.Q_delta      = self.Q_value - self.Q_target
        self.critic_loss  = tf.reduce_mean((self.Q_delta)**2 * self.is_weights)

        # Policy loss = max_θ Σ Q(s, μ(s; θ); ϕ)
        self.actor_loss = -tf.reduce_mean(self.Q_value_of_actor * self.is_weights)

        # Minimize loss
        actor_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="main/pi/")
        critic_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="main/q/")
        #rdn_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="rdn/")
        actor_lr = tf.train.exponential_decay(initial_actor_lr, self.episode_counter.var, 1, lr_decay, staircase=True)
        critic_lr = tf.train.exponential_decay(initial_critic_lr, self.episode_counter.var, 1, lr_decay, staircase=True)
        self.train_actor_op = clip_grad(tf.train.AdamOptimizer(learning_rate=actor_lr, epsilon=1e-5), actor_params, self.actor_loss, grad_norm)
        self.train_critic_op = clip_grad(tf.train.AdamOptimizer(learning_rate=critic_lr, epsilon=1e-5), critic_params, self.critic_loss, grad_norm)
        #self.train_rdn_op = clip_grad(tf.train.AdamOptimizer(learning_rate=0.0001), rdn_params, self.rdn_loss, grad_norm)

        # Create session
        self.sess = tf.Session()

        # Set up critic metrics
        metrics = {}
        metrics["losses/episodic/critic"] = tf.metrics.mean(self.critic_loss)
        #metrics["losses/episodic/rdn"] = tf.metrics.mean(self.rdn_loss)
        for i in range(num_actions):
            metrics["actor.train/episodic/action_{}/taken_actions".format(i)] = tf.metrics.mean(tf.reduce_mean(self.taken_actions[:, i]))
            metrics["actor.train/episodic/action_{}/mean".format(i)] = tf.metrics.mean(tf.reduce_mean(self.actor_mean[:, i]))
        metrics["critic.train/episodic/Q_value"] = tf.metrics.mean(self.Q_value)
        metrics["critic.train/episodic/Q_target"] = tf.metrics.mean(self.Q_target)
        metrics["critic.train/episodic/Q_delta"] = tf.metrics.mean(self.Q_delta)
        self.episodic_critic_summaries, self.update_critic_metrics_op = create_mean_metrics_from_dict(metrics)

        # Set up actor metrics
        metrics = {}
        metrics["losses/episodic/actor"] = tf.metrics.mean(self.actor_loss)
        self.episodic_actor_summaries, self.update_actor_metrics_op = create_mean_metrics_from_dict(metrics)

        # Set up stepwise summaries
        summaries = []
        summaries.append(tf.summary.scalar("train/critic_lr", critic_lr))
        summaries.append(tf.summary.histogram("train/input_states", states))
        summaries.append(tf.summary.histogram("train/input_states_next", states_next))
        #summaries.append(tf.summary.histogram("train/rdn_diff_norm", self.rdn_diff_norm))
        #summaries.append(tf.summary.histogram("train/rdn_diff", self.rdn_diff))
        self.critic_stepwise_summaries_op = tf.summary.merge(summaries)
        summaries = []
        summaries.append(tf.summary.scalar("train/actor_lr", actor_lr))
        self.actor_stepwise_summaries_op = tf.summary.merge(summaries)

        summaries = []
        for i in range(num_actions):
            summaries.append(tf.summary.scalar("actor.predict/action_{}/mean".format(i), self.actor_mean[0, i]))
        self.prediction_summaries = tf.summary.merge(summaries)

        # Run variable initializers
        self.sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

        # Initialize θ_{targ} <- θ and ϕ_{targ} <- ϕ
        self.sess.run(init_target_params_op)

        # Set up model saver and dirs
        self.output_dir = output_dir
        self.saver = tf.train.Saver()
        self.model_dir = "{}/checkpoints/".format(self.output_dir)
        self.log_dir   = "{}/logs/".format(self.output_dir)
        self.video_dir = "{}/videos/".format(self.output_dir)
        self.dirs = [self.model_dir, self.log_dir, self.video_dir]
        for d in self.dirs: os.makedirs(d, exist_ok=True)
    
    def build_mlp_actor_critic(self, input_states, taken_actions, action_space,
                               hidden_sizes=(300,), activation=tf.nn.leaky_relu, 
                               output_activation=tf.sigmoid):
        # Actor branch
        with tf.variable_scope("pi"):
            # μ(s; θ) = mlp(s)
            pi_normalized = build_mlp(input_states, list(hidden_sizes) + list(action_space.shape), activation, output_activation)
            pi = action_space.low + pi_normalized * (action_space.high - action_space.low) # Scale μ to action space
        
        # Critic brach
        with tf.variable_scope("q"):
            # Q(s, a; ϕ) = mlp(concat(s, a))
            q = tf.squeeze(build_mlp(tf.concat([input_states, taken_actions], axis=-1), list(hidden_sizes) + [1], activation, None), axis=1)

        # Critic on actor branch
        with tf.variable_scope("q", reuse=True):
            # Q(s, μ(s; θ); ϕ) = mlp(concat(s, μ(s; θ)))
            q_pi = tf.squeeze(build_mlp(tf.concat([input_states, pi], axis=-1), list(hidden_sizes) + [1], activation, None), axis=1)
            
        return pi, q, q_pi

    def init_logging(self):
        self.train_writer = tf.summary.FileWriter(self.log_dir, self.sess.graph)
        
    def load_latest_checkpoint(self):
        model_checkpoint = tf.train.latest_checkpoint(self.model_dir)
        if model_checkpoint:
            try:
                self.saver.restore(self.sess, model_checkpoint)
                print("Model checkpoint restored from {}".format(model_checkpoint))
                return True
            except:
                return False

    def save(self):
        model_checkpoint = os.path.join(self.model_dir, "model.ckpt")
        self.saver.save(self.sess, model_checkpoint, global_step=self.episode_counter.var)
        print("Model checkpoint saved to {}".format(model_checkpoint))
        
    def train(self, input_states, taken_actions, rewards, input_states_next, terminals, w):
        feed_dict = {
            self.input_states: input_states,
            self.input_states_next: input_states_next,
            self.taken_actions: taken_actions,
            self.rewards: rewards,
            self.terminals: terminals,
            self.is_weights: w
        }

        # Train critic
        _, _, critic_summary, deltas = self.sess.run([self.train_critic_op, self.update_critic_metrics_op,
                                                      self.critic_stepwise_summaries_op, self.Q_delta],
                                                      feed_dict=feed_dict)

        # Train actor
        actor_summary = self.sess.run([self.train_actor_op, self.update_actor_metrics_op, self.actor_stepwise_summaries_op],
                                      feed_dict=feed_dict)[-1]

        # Train rnd pred
        #self.sess.run(self.train_rdn_op, feed_dict=feed_dict)

        # Update target networks
        self.sess.run(self.update_target_params_op)

        # Write to summary
        step_idx = self.sess.run(self.train_step_counter.var)
        self.train_writer.add_summary(critic_summary, step_idx)
        self.train_writer.add_summary(actor_summary, step_idx)
        self.sess.run(self.train_step_counter.inc_op) # Inc step counter

        return deltas

    def predict(self, input_states, greedy=False, write_to_summary=False):
        # Return μ(s; θ) if greedy, else return action sampled from N(μ(s; θ), σ)
        action, summary, Q_value = self.sess.run([self.actor_mean, self.prediction_summaries, self.Q_value_of_actor],
                                        feed_dict={
                                            self.input_states: input_states
                                        })
        if not greedy:
            action = np.clip(action + self.action_noise(), self.action_space.low, self.action_space.high)
        if write_to_summary:
            self.train_writer.add_summary(summary, self.get_predict_step_counter())
            self.sess.run(self.predict_step_counter.inc_op)
        return action, Q_value

    def encode(self, vae_input):
        return self.sess.run(self.vae_mean_squashed, feed_dict={
            self.vae.input_states: vae_input
        })

    def get_episode_counter(self):
        return self.sess.run(self.episode_counter.var)

    def get_predict_step_counter(self):
        return self.sess.run(self.predict_step_counter.var)

    def write_value_to_summary(self, summary_name, value, step):
        summary = tf.Summary()
        summary.value.add(tag=summary_name, simple_value=value)
        self.train_writer.add_summary(summary, step)

    def write_dict_to_summary(self, summary_name, params, step):
        summary_op = tf.summary.text(summary_name, tf.stack([tf.convert_to_tensor([k, str(v)]) for k, v in params.items()]))
        self.train_writer.add_summary(self.sess.run(summary_op), step)

    def write_episodic_summaries(self, episode_idx):
        self.train_writer.add_summary(self.sess.run(self.episodic_critic_summaries), episode_idx)
        self.train_writer.add_summary(self.sess.run(self.episodic_actor_summaries), episode_idx)
        self.sess.run([self.episode_counter.inc_op, tf.local_variables_initializer()])
        self.action_noise.reset()
