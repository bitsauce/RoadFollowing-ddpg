import numpy as np
import tensorflow as tf
import os

class VAE():
    def __init__(self, input_shape, input_tensor=None,
                 z_dim=32, beta=1,
                 learning_rate=3e-4, training=True,
                 loss_type="bce", model_type="cnn",
                 model_name="vae", model_dir="",
                 reuse=tf.AUTO_REUSE):

        with tf.variable_scope("vae", reuse=reuse):
            if input_tensor is None:
                # Get and verify input
                self.input_states = tf.placeholder(shape=(None, *input_shape), dtype=tf.float32, name="input_state_placeholder")
                self.input_states = tf.check_numerics(self.input_states, "Invalid value for self.input_states")
            else:
                self.input_states = input_tensor
            verify_input_op = tf.Assert(tf.reduce_all(tf.logical_and(self.input_states >= 0, self.input_states <= 1)),
                                    ["min=", tf.reduce_min(self.input_states),
                                    "max=", tf.reduce_max(self.input_states)],
                                    name="verify_input")
            with tf.control_dependencies([verify_input_op]):
                if training:
                    self.input_states = tf.image.random_flip_left_right(self.input_states)
                else:
                    self.input_states = tf.multiply(self.input_states, 1, name="input_state_identity")

            # Encoder
            def encoder_mlp(images, z_dim):
                with tf.variable_scope("encoder", reuse=tf.AUTO_REUSE):
                    x         = tf.layers.dense(images, 512, activation=tf.nn.leaky_relu, name="dense1")
                    x         = tf.layers.dense(x, 256, activation=tf.nn.leaky_relu, name="dense2")
                    mean      = tf.layers.dense(x, z_dim, activation=None, name="mean")
                    logstd_sq = tf.layers.dense(x, z_dim, activation=None, name="logstd_sqare")
                    return x, mean, logstd_sq

            # Decoder
            def decoder_mlp(z):
                with tf.variable_scope("decoder", reuse=tf.AUTO_REUSE):
                    x = tf.layers.dense(z, 256, activation=tf.nn.leaky_relu, name="dense1")
                    x = tf.layers.dense(x, 512, activation=tf.nn.leaky_relu, name="dense2")
                    x = tf.layers.dense(x, np.prod(input_shape), activation=None, name="dense3")
                    return x

            def encoder_cnn(input_states, z_dim):
                with tf.variable_scope("encoder", reuse=tf.AUTO_REUSE):
                    print("Encoder")
                    print(input_states.shape)
                    x = tf.layers.conv2d(input_states, filters=16, kernel_size=3, strides=2, activation=tf.nn.leaky_relu, padding="valid", name="conv1")
                    print(x.shape)
                    x = tf.layers.conv2d(x, filters=16, kernel_size=3, strides=2, activation=tf.nn.leaky_relu, padding="valid", name="conv2")
                    print(x.shape)
                    x = tf.layers.conv2d(x, filters=16, kernel_size=3, strides=2, activation=tf.nn.leaky_relu, padding="valid", name="conv3")
                    print(x.shape)
                    x = tf.layers.conv2d(x, filters=16, kernel_size=3, strides=2, activation=tf.nn.leaky_relu, padding="valid", name="conv4")
                    print(x.shape)
                    self.x_shape = x.shape
                    self.x_shape2 = tf.shape(x)
                    x = tf.layers.flatten(x, name="flatten")
                    mean      = tf.layers.dense(x, z_dim, activation=None, name="mean")
                    logstd_sq = tf.layers.dense(x, z_dim, activation=None, name="logstd_sqare")
                    return x, mean, logstd_sq

            def decoder_cnn(z):
                with tf.variable_scope("decoder", reuse=tf.AUTO_REUSE):
                    print("Decoder")
                    x = tf.layers.dense(z, self.x_shape[1] * self.x_shape[2] * self.x_shape[3], activation=None, name="dense1")
                    print(x.shape)
                    x = tf.reshape(x, (-1, self.x_shape[1], self.x_shape[2], self.x_shape[3]))
                    print(x.shape)
                    x = tf.layers.conv2d_transpose(x, filters=16, kernel_size=3, strides=2, activation=tf.nn.leaky_relu, padding="valid", name="deconv1")
                    print(x.shape)
                    x = tf.layers.conv2d_transpose(x, filters=16, kernel_size=4, strides=2, activation=tf.nn.leaky_relu, padding="valid", name="deconv2")
                    print(x.shape)
                    x = tf.layers.conv2d_transpose(x, filters=16, kernel_size=3, strides=2, activation=tf.nn.leaky_relu, padding="valid", name="deconv3")
                    print(x.shape)
                    x = tf.layers.conv2d_transpose(x, filters=1, kernel_size=4, strides=2, activation=None, padding="valid", name="deconv4")
                    print(x.shape)
                    return x

            if model_type == "mlp":
                encoder = encoder_mlp
                decoder = decoder_mlp
            elif model_type == "cnn":
                encoder = encoder_cnn
                decoder = decoder_cnn
            else:
                raise Exception("Please select a valid VAE model")

            # Flatten input
            self.flattened_input = tf.layers.flatten(self.input_states, name="flattened_input")

            # Get encoded mean and std for input
            _, self.mean, self.logstd_sq = encoder(self.flattened_input if model_type == "mlp" else self.input_states, z_dim)
            self.mean      = tf.check_numerics(self.mean,      "Invalid value for self.mean")
            self.logstd_sq = tf.check_numerics(self.logstd_sq, "Invalid value for self.logstd_sq")

            # Sample normal distribution
            self.normal = tf.distributions.Normal(self.mean, tf.exp(0.5 * self.logstd_sq), validate_args=True)
            if training:
                self.sample = tf.squeeze(self.normal.sample(1), axis=0)
                self.sample = tf.check_numerics(self.sample, "Invalid value for self.sample")
            else:
                self.sample = self.mean

            # Decode random sample
            self.reconstructed_logits = tf.layers.flatten(decoder(self.sample),  name="reconstructed_logits")
            self.reconstructed_logits = tf.check_numerics(self.reconstructed_logits, "Invalid value for self.reconstructed_logits")
            self.reconstructed_states = tf.nn.sigmoid(self.reconstructed_logits, name="reconstructed_states")

            # Generative graph
            self.input_z = tf.placeholder(tf.float32, (None, z_dim), name="input_z_placeholder")
            self.generated_state = tf.nn.sigmoid(decoder(self.input_z), name="generated_state")

            def bce(t, y, name="bce"):
                epsilon = 1e-10
                with tf.variable_scope(name):
                    return -(t * tf.log(epsilon + y) + (1 - t) * tf.log(epsilon + 1 - y))

            def kl_divergence(mean, logstd_sq, name="kl_divergence"):
                with tf.variable_scope(name):
                    return -0.5 * tf.reduce_sum(1 + logstd_sq - tf.square(mean) - tf.exp(logstd_sq), axis=1)

            # Reconstruction loss
            if loss_type == "bce":
                self.reconstruction_loss = tf.reduce_mean(
                        tf.reduce_sum(
                            tf.nn.sigmoid_cross_entropy_with_logits(
                                labels=self.flattened_input,
                                logits=self.reconstructed_logits,
                                name="sigmoid_cross_entropy_with_logits"
                            ),
                            axis=1
                        )
                    )
            elif loss_type == "bce_v2":
                self.reconstruction_loss = tf.reduce_mean(
                        tf.reduce_sum(
                            bce(
                                self.flattened_input,
                                self.reconstructed_states,
                                name="bce_v2"
                            ),
                            axis=1
                        )
                    )
            elif loss_type == "mse":
                self.reconstruction_loss = tf.reduce_mean(
                        tf.reduce_sum(
                            tf.math.squared_difference(
                                self.flattened_input,
                                self.reconstructed_logits,
                                name="mse_loss"
                            ),
                            axis=1
                        )
                    )
            else:
                raise Exception("Please select a valid VAE loss type")

            self.kl_loss = tf.reduce_mean(kl_divergence(self.mean, self.logstd_sq, name="kl_divergence"))
            self.loss = self.reconstruction_loss + beta * self.kl_loss

            # Set model dirs
            self.model_name = model_name
            self.model_dir = "./{}models/{}".format(model_dir, self.model_name)
            self.log_dir   = "./{}logs/{}".format(model_dir, self.model_name)
            self.dirs = [self.model_dir, self.log_dir]

            # Epoch variable
            self.step_idx = tf.Variable(0, name="step_idx", trainable=False)
            self.inc_step_idx = tf.assign(self.step_idx, self.step_idx + 1)

            # Create optimizer
            self.saver = tf.train.Saver()
            if training:
                # Summary
                self.mean_kl_loss, self.update_mean_kl_loss = tf.metrics.mean(self.kl_loss)
                self.mean_reconstruction_loss, self.update_mean_reconstruction_loss = tf.metrics.mean(self.reconstruction_loss)
                self.merge_summary = tf.summary.merge([
                    tf.summary.scalar("kl_loss", self.mean_kl_loss),
                    tf.summary.scalar("reconstruction_loss", self.mean_reconstruction_loss)
                ])

                # Create optimizer
                self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
                self.train_step = self.optimizer.minimize(self.loss)
                for d in self.dirs: os.makedirs(d, exist_ok=True)

    def init_session(self, sess=None, init_logging=True):
        if sess is None:
            # Create session
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config)

            # Run global initializer
            self.sess.run(tf.global_variables_initializer())
        else:
            self.sess = sess

        if init_logging:
            self.train_writer = tf.summary.FileWriter(os.path.join(self.log_dir, "train"), self.sess.graph)
            self.val_writer = tf.summary.FileWriter(os.path.join(self.log_dir, "val"), self.sess.graph)

    def generate_from_latent(self, z):
        return self.sess.run(self.generated_state, feed_dict={
                self.input_z: z
            })

    def reconstruct(self, input_states):
        return self.sess.run(self.reconstructed_states, feed_dict={
                self.input_states: input_states
            })

    def save(self):
        model_checkpoint = os.path.join(self.model_dir, "model.ckpt")
        self.saver.save(self.sess, model_checkpoint, global_step=self.step_idx)
        print("Model checkpoint saved to {}".format(model_checkpoint))
    
    def load_latest_checkpoint(self):
        model_checkpoint = tf.train.latest_checkpoint(self.model_dir)
        if model_checkpoint:
            try:
                self.saver.restore(self.sess, model_checkpoint)
                print("Model checkpoint restored from {}".format(model_checkpoint))
                return True
            except Exception as e:
                print(e)
                return False

    def get_step_idx(self):
        return tf.train.global_step(self.sess, self.step_idx)

    def train_one_epoch(self, train_images, batch_size):
        np.random.shuffle(train_images)
        self.sess.run(tf.local_variables_initializer())
        for i in range(train_images.shape[0] // batch_size):
            self.sess.run([self.train_step, self.update_mean_kl_loss, self.update_mean_reconstruction_loss], feed_dict={
                self.input_states: train_images[i*batch_size:(i+1)*batch_size]
            })
        self.train_writer.add_summary(self.sess.run(self.merge_summary), self.get_step_idx())
        self.sess.run(self.inc_step_idx)

    def evaluate(self, val_images, batch_size):
        self.sess.run(tf.local_variables_initializer())
        for i in range(val_images.shape[0] // batch_size):
            self.sess.run([self.update_mean_kl_loss, self.update_mean_reconstruction_loss], feed_dict={
                self.input_states: val_images[i*batch_size:(i+1)*batch_size]
            })
        self.val_writer.add_summary(self.sess.run(self.merge_summary), self.get_step_idx())
        return self.sess.run([self.mean_reconstruction_loss, self.mean_kl_loss])
