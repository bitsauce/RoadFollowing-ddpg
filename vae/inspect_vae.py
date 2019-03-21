from tkinter import *
from tkinter.ttk import *

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import argparse
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from models import MlpVAE, ConvVAE

parser = argparse.ArgumentParser(description="Visualizes the features learned by the VAE")
parser.add_argument("-reconstruct", action="store_true")
parser.add_argument("--model_name", type=str, required=True)
parser.add_argument("--models_dir", type=str, default=".")
parser.add_argument("--model_type", type=str, default="mlp")
parser.add_argument("--z_dim", type=int, default=10)
args = parser.parse_args()

input_shape = (84, 84, 1)

if args.model_type == "cnn": VAEClass = ConvVAE
elif args.model_type == "mlp": VAEClass = MlpVAE    
else: raise Exception("No model type \"{}\"".format(args.model_type))

vae = VAEClass(input_shape, z_dim=args.z_dim, model_name=args.model_name, models_dir=args.models_dir, training=False)
vae.init_session(init_logging=False)
if not vae.load_latest_checkpoint():
    print("Failed to load latest checkpoint for model \"{}\"".format(args.model_name))

if args.reconstruct:
    from RoadFollowingEnv.car_racing import RoadFollowingEnv
    from utils import preprocess_frame
    from pyglet.window import key

    def make_env(title=None, frame_skip=0):
        env = RoadFollowingEnv(title=title,
                            encode_state_fn=lambda x: preprocess_frame(x.frame),
                            throttle_scale=0.1,
                            max_speed=30.0,
                            frame_skip=frame_skip)
        return env

    env = make_env()
    action = np.zeros(env.action_space.shape[0])
    restart = False
    def key_press(k, mod):
        global restart
        if k==0xff0d: restart = True
        if k==key.LEFT:  action[0] = -1.0
        if k==key.RIGHT: action[0] = +1.0
        if k==key.UP:    action[1] = +1.0
        if k==key.DOWN:  action[1] = -1.0
    def key_release(k, mod):
        if k==key.LEFT  and action[0]==-1.0: action[0] = 0
        if k==key.RIGHT and action[0]==+1.0: action[0] = 0
        if k==key.UP:    action[1] = 0
        if k==key.DOWN:  action[1] = 0
    env.render()
    env.viewer.window.on_key_press = key_press
    env.viewer.window.on_key_release = key_release

    done = True
    def update(*args):
        global done, restart, action, env, im1, im2
        if done or restart:
            env.reset()
        s, r, done, info = env.step(action)
        env.render(mode="human")
        reconstruted_state = vae.reconstruct([s])
        im1.set_array(s[:, :, 0])
        im2.set_array(reconstruted_state[0].reshape(84, 84))
        return im1, im2

    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    im1 = ax1.imshow(np.zeros((84,84)), cmap="gray", animated=True, vmin=0, vmax=1)
    im2 = ax2.imshow(np.zeros((84,84)), cmap="gray", animated=True, vmin=0, vmax=1)
    ani = animation.FuncAnimation(fig, update, interval=16, blit=True)
    plt.show()

    env.close()
else:
    class UI():
        def __init__(self, z_dim, generate_fn, slider_range=3, image_scale=4):
            # Setup tkinter window
            self.window = Tk()
            self.window.title("VAE Inspector")
            self.window.style = Style()
            self.window.style.theme_use("clam") # ('clam', 'alt', 'default', 'classic')

            # Setup image
            self.image = Label(self.window)
            self.image.pack(side=LEFT, padx=50, pady=20)

            self.image_scale = image_scale
            self.update_image(np.ones((84, 84)) * 127)

            self.generate_fn = generate_fn

            # Setup sliders for latent vector z
            slider_frames = []
            self.z_vars = [DoubleVar() for _ in range(z_dim)]
            for i in range(z_dim):
                # On slider change event
                def create_slider_event(i, z_i, label):
                    def event(_=None, generate=True):
                        label.configure(text="z[{}]={}{:.2f}".format(i, "" if z_i.get() < 0 else " ", z_i.get()))
                        if generate: self.generate_fn(np.array([z_i.get() for z_i in self.z_vars]))
                    return event

                if i % 20 == 0:
                    sliders_frame = Frame(self.window)
                    slider_frames.append(sliders_frame)

                # Create widgets
                inner_frame = Frame(sliders_frame) # Frame for side-by-side label and slider layout
                label = Label(inner_frame, font="TkFixedFont")

                # Create event function
                on_value_changed = create_slider_event(i, self.z_vars[i], label)
                on_value_changed(generate=False) # Call once to set label text

                # Create slider
                slider = Scale(inner_frame, value=0.0, variable=self.z_vars[i], orient=HORIZONTAL, length=200,
                            from_=-slider_range, to=slider_range, command=on_value_changed)

                # Pack
                slider.pack(side=RIGHT, pady=10)
                label.pack(side=LEFT, padx=10)
                inner_frame.pack(side=TOP)
            for f in reversed(slider_frames): f.pack(side=RIGHT, padx=20, pady=20)

        def update_image(self, image_array):
            image_size = (84 * self.image_scale, 84 * self.image_scale)
            pil_image = Image.fromarray(image_array).resize(image_size, resample=Image.NEAREST)
            self.tkimage = ImageTk.PhotoImage(image=pil_image)
            self.image.configure(image=self.tkimage)

        def mainloop(self):
            self.generate_fn(np.array([z_i.get() for z_i in self.z_vars]))
            self.window.mainloop()

    def generate(z):
        generated_image = vae.generate_from_latent([z])[0] * 255
        ui.update_image(generated_image.reshape(input_shape)[:, :, 0])

    ui = UI(vae.sample.shape[1], generate)
    ui.mainloop()
    