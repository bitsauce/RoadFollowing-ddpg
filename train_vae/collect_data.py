import argparse
import gzip
import pickle

import numpy as np

from gym.envs.box2d.car_racing import CarRacing

parser = argparse.ArgumentParser(description="Script to collect data for VAE training")
parser.add_argument("--num_images", type=int, default=10000)
parser.add_argument("-append", action="store_true", help="Append data onto existing dataset")
args = parser.parse_args()

def preprocess_frame(frame):
    frame = frame[:-12, 6:-6] # Crop to 84x84
    frame = np.dot(frame[..., 0:3], [0.299, 0.587, 0.114])
    frame = frame / 255.0
    return np.expand_dims(frame, axis=-1)

with gzip.open("data10k.pklz", "rb") as f:
    images = np.expand_dims(pickle.load(f), axis=-1)

with gzip.open("data10k.pklz", "wb") as f:
    pickle.dump(images, f)

exit()

if args.append:
    with gzip.open("data.pklz", "rb") as f:
        images = [x for x in pickle.load(f)]
else:
    images = []

if __name__ == "__main__":
    from pyglet.window import key
    a = np.array([0.0, 0.0, 0.0])
    def key_press(k, mod):
        global restart
        if k==0xff0d: restart = True
        if k==key.LEFT:  a[0] = -1.0
        if k==key.RIGHT: a[0] = +1.0
        if k==key.UP:    a[1] = +1.0
        if k==key.DOWN:  a[2] = +0.8   # set 1.0 for wheels to block to zero rotation
    def key_release(k, mod):
        if k==key.LEFT  and a[0]==-1.0: a[0] = 0
        if k==key.RIGHT and a[0]==+1.0: a[0] = 0
        if k==key.UP:    a[1] = 0
        if k==key.DOWN:  a[2] = 0
    env = CarRacing()
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
            images.append(preprocess_frame(env.state))
            if done or restart: break
    env.close()

with gzip.open("data.pklz", "wb") as f:
    pickle.dump(np.stack(images, axis=0), f)
