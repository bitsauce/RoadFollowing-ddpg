if __name__ == "__main__":
    import sys
    sys.path.append("..")
    from vae import VAE
    import numpy as np
    import tensorflow as tf
    import numpy as np
    import gzip
    import pickle
    import shutil
    import os
    import argparse

    parser = argparse.ArgumentParser(description="Trains a VAE on input data")
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--dataset", type=str, default="data20k.pklz")
    parser.add_argument("--loss_type", type=str, default="bce")
    parser.add_argument("--model_type", type=str, default="mlp")
    parser.add_argument("--beta", type=int, default=4)
    parser.add_argument("--z_dim", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    with gzip.open(args.dataset, "rb") as f:
        images = pickle.load(f)
    
    np.random.seed(0)
    np.random.shuffle(images)

    val_split = int(images.shape[0] * 0.1)
    train_images = images[val_split:]
    val_images = images[:val_split]

    w, h = images.shape[1:3]
    print("train_images.shape", train_images.shape)
    print("val_images.shape", val_images.shape)

    tf.reset_default_graph()

    if args.model_name is None:
        args.model_name = "{}_{}_zdim{}_beta{}_{}".format(args.loss_type, args.model_type, args.z_dim,
                                                          args.beta, os.path.splitext(os.path.basename(args.dataset))[0])

    vae = VAE(input_shape=(w, h, 1),
              z_dim=args.z_dim, beta=args.beta,
              learning_rate=args.learning_rate,
              loss_type=args.loss_type,
              model_type=args.model_type,
              model_name=args.model_name)
    vae.init_session()

    min_val_loss = float("inf")
    counter = 0
    print("Training")
    while True:
        epoch = vae.get_step_idx()
        if (epoch + 1) % 10 == 0: print(f"Epoch {epoch + 1}")
        
        # Calculate evaluation metrics
        val_loss, _ = vae.evaluate(val_images, args.batch_size)
        
        # Train one epoch
        vae.train_one_epoch(train_images, args.batch_size)
        
        # Early stopping
        if val_loss < min_val_loss:
            counter = 0
            min_val_loss = val_loss
        else:
            counter += 1
            if counter >= 10:
                print("No improvement in last 10 epochs, stopping")
                vae.save()
                break
    vae.train_writer.flush()
    vae.val_writer.flush()
