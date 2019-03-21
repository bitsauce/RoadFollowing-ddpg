import argparse
import gzip
import pickle
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from RoadFollowingEnv.car_racing import RoadFollowingEnv
from utils import preprocess_frame

parser = argparse.ArgumentParser(description="Script to collect data for VAE training")
parser.add_argument("--output", type=str, required=True)
parser.add_argument("--num_images", type=int, default=10000)
parser.add_argument("-append", action="store_true", help="Append data onto existing dataset")
args = parser.parse_args()

if args.append:
    with gzip.open(args.output, "rb") as f:
        images = [x for x in pickle.load(f)]
else:
    images = []

def make_env(title=None, frame_skip=0):
    env = RoadFollowingEnv(title=title,
                        encode_state_fn=lambda x: preprocess_frame(x.frame),
                        throttle_scale=0.1,
                        max_speed=30.0,
                        frame_skip=frame_skip)
    return env

if __name__ == "__main__":
    from pyglet.window import key
    env = make_env()
    a = np.zeros(env.action_space.shape[0])
    def key_press(k, mod):
        global restart
        if k==0xff0d: restart = True
        if k==key.LEFT:  a[0] = -1.0
        if k==key.RIGHT: a[0] = +1.0
        if k==key.UP:    a[1] = +1.0
        if k==key.DOWN:  a[1] = -1.0
    def key_release(k, mod):
        if k==key.LEFT  and a[0]==-1.0: a[0] = 0
        if k==key.RIGHT and a[0]==+1.0: a[0] = 0
        if k==key.UP:    a[1] = 0
        if k==key.DOWN:  a[1] = 0
    env.render()
    env.viewer.window.on_key_press = key_press
    env.viewer.window.on_key_release = key_release
    while len(images) < args.num_images:
        env.reset()
        restart = False
        while len(images) < args.num_images:
            if len(images) % (args.num_images // 100) == 0:
                print("{}%".format(int(len(images) / args.num_images * 100)))
            s, r, done, info = env.step(a)
            env.render(mode="human")
            images.append(s)
            if done or restart: break
    env.close()

with gzip.open(args.output, "wb") as f:
    pickle.dump(np.stack(images, axis=0), f)
