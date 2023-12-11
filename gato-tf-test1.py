#!/usr/bin/env python3

# This code does the following:
# 1. Load training images and perform encoding and embedding using Gato classes
# 2. Load training values for continuous and discrete values and encode them using Gato classes
# 3. Create the right observation and mask tensors for Gato transformer
# 4. Train the Gato model
# 5. Perform inference using the trained Gato model and verify the results


import sys
import json
import timeit
import argparse
import logging
import tensorflow as tf
from gato import Gato, GatoConfig

# Set logging level
logging.basicConfig(level=logging.INFO)


def image_to_patches(image_file, x_size, y_size, num_patches, input_dim) -> tf.Tensor:
    """Load and extract image patches"""

    # Read PNG file
    logging.info("Creating image tokens, read image: %s", image_file)
    image = tf.io.read_file(image_file)

    image = tf.image.decode_png(image, channels=3)  # decode PNG
    image = tf.cast(image, dtype=tf.float32)  # cast to float32
    image = image / 255.0  # normalize to [0, 1]

    # Resize image to x_size, y_size
    image = tf.image.resize(image, (x_size, y_size))
    logging.debug("  image shape: %s", image.shape)

    # Split image into num_patches of size 16x16
    image = tf.image.extract_patches(
        images=tf.expand_dims(image, axis=0),
        sizes=[1, 16, 16, 1],
        strides=[1, 16, 16, 1],
        rates=[1, 1, 1, 1],
        padding="VALID",
    )
    logging.debug("  patch extracted image: %s", image.shape)

    # Reshape to (1, num_patches, input_dim)
    return tf.reshape(image, (1, num_patches, input_dim))


def create_encoding(num_observations, num_patches):
    """Create encoding tensor"""
    arr = []

    # 0 - image patch embedding
    # 1 - continuous value embedding
    # 2 - discrete embedding (actions, texts)
    for i in range(num_observations):
        arr.extend([0] * num_patches + [1, 2])
    # logging.debug("encoding: %s", arr)

    logging.info("Creating encoding..")
    return tf.constant([arr])


def encode_continuous_value(value, input_dim=512):
    """Encode continuous value"""

    logging.info("Creating continuous value..")
    # resize value to input_dim
    value = value + [0.0] * (input_dim - len(value))
    value = tf.cast(value, dtype=tf.float32)
    logging.debug("  value: %s", value)
    return tf.reshape(value, (1, 1, input_dim))


def encode_discrete_value(value: int, input_dim=512):
    """Encode discrete value"""
    logging.info("Creating discrete value..")
    arr = [0.0] * input_dim
    arr[0] = value
    logging.debug("  arr: %s", arr)
    return tf.reshape(arr, (1, 1, input_dim))


def encode_row_pos(num_observations, x_scale, y_scale):
    """Create row_pos tensor"""

    logging.info("Creating row_pos..")
    # row_pos from
    arr1 = [i / y_scale for i in range(y_scale)]
    arr1 *= x_scale
    arr1.extend([0, 0])
    # Repeat this array num_observations times
    arr1 *= num_observations
    logging.debug(" arr1: %s", arr1)

    # row_pos to
    arr2 = [(i + 1) / y_scale for i in range(y_scale)]
    arr2 *= x_scale
    arr2.extend([0, 0])
    # Repeat this array num_observationss times
    arr2 *= num_observations
    logging.debug(" arr2: %s", arr2)

    return (
        tf.constant([arr1]),  # pos_from
        tf.constant([arr2]),  # pos_to
    )


def encode_col_pos(num_observations, x_scale, y_scale):
    """Create col_pos tensor"""
    logging.info("Creating col_pos..")
    arr1 = []
    arr2 = []

    # col_pos from
    for i in range(x_scale):
        arr1.extend([i / x_scale] * y_scale)

    arr1.extend([0, 0])
    # Repeat this array num_observations times
    arr1 *= num_observations
    logging.debug(" arr1: %s", arr1)

    for i in range(x_scale):
        arr2.extend([(i + 1) / x_scale] * y_scale)

    arr2.extend([0, 0])
    # Repeat this array num_observations times
    arr2 *= num_observations
    logging.debug(" arr2: %s", arr2)

    return (
        tf.constant([arr1]),  # pos_from
        tf.constant([arr2]),  # pos_to
    )


def encode_obs(num_observations, num_patches):
    """Create obs tensor"""

    logging.info("Creating obs..")
    # obs token
    arr1 = [i for i in range(num_patches + 2)]
    arr1 = arr1 * num_observations
    logging.debug("arr1: %s", arr1)

    arr2 = [1] * (num_patches + 1) + [0]
    arr2 = arr2 * num_observations
    logging.debug("arr2: %s", arr2)

    return (
        tf.constant([arr1]),  # obs token
        tf.constant([arr2]),  # obs token masking (for action tokens)
    )


def process_episode(episode_config_file, input_dim, x_size, y_size, num_patches):
    """Process episode config file and create input_ids tensor"""

    input_ids = None
    input_array = []

    # Load episode config
    with open(episode_config_file) as f:
        episode_config = json.load(f)
        logging.debug("episode_config: %s", episode_config)

        # For each step in episode config, create input_ids
        for key in episode_config["steps"]:
            logging.debug("  jointAngles: %s", key["jointAngles"])
            logging.debug("   action: %s", key["action"])
            logging.debug("    snapshot: %s", key["snapshot"])

            img_file = prefix + key["snapshot"]
            logging.debug("    img_file: %s", img_file)

            image = image_to_patches(img_file, x_size, y_size, num_patches, input_dim)

            continuous_value = encode_continuous_value(key["jointAngles"], input_dim)
            discrete_value = encode_discrete_value(key["action"], input_dim)

            input_array.append(image)
            input_array.append(continuous_value)
            input_array.append(discrete_value)

        logging.debug("input_array size: %s", len(input_array))
        input_ids = tf.concat(
            input_array,  # repeat num_observations times
            axis=1,
        )
        logging.debug("input_ids shape: %s", input_ids.shape)
        return input_ids


if __name__ == "__main__":
    # Command line arguments
    parser = argparse.ArgumentParser(description="Test Gato model")
    # Argument for experiment name
    parser.add_argument(
        "--experiment_name",
        "-e",
        type=str,
        default="exp1",
        help="Experiment name",
    )

    args = parser.parse_args()

    prefix = "experiments/{}/".format(args.experiment_name)
    json_file = prefix + args.experiment_name + ".json"
    logging.info("Loading experiment config from: %s", json_file)

    # Create model instance
    config = GatoConfig.small()
    gato = Gato(config, trainable=True)

    input_dim = config.input_dim
    logging.debug("input_dim: %s", input_dim)

    x_size = 80  # Hardcoded for now
    y_size = 64  # Hardcoded for now
    num_observations = None

    num_patches = (x_size // 16) * (y_size // 16)
    logging.debug("num_patches: %s", num_patches)

    x_scale = x_size // 16  # Hardcoded for now
    y_scale = y_size // 16  # Hardcoded for now
    logging.debug("  x_scale: %s, y_scale: %s", x_scale, y_scale)

    # Load experiment config
    with open(json_file) as f:
        config = json.load(f)

        exp_info = config["experiment"]
        num_observations = exp_info["stepsPerEpisode"]
        logging.info("num_observations: %s", num_observations)

        # List out episode config files
        episode_config_files = config["episodes"]
        logging.info("episode_config_files: %s", episode_config_files)

        # Loop through each episode config file
        for episode_config_file in episode_config_files:
            episode_config_file = prefix + episode_config_file
            logging.info("episode_config_file: %s", episode_config_file)

            input_ids = process_episode(
                episode_config_file, input_dim, x_size, y_size, num_patches
            )

            encoding = create_encoding(num_observations, num_patches)
            logging.debug(" encoding shape: %s", encoding)
            logging.debug("  encoding shape: %s", encoding.shape)

            row_pos = encode_row_pos(num_observations, x_scale, y_scale)
            logging.debug("  row_pos shape: %s, %s", row_pos[0].shape, row_pos[1].shape)

            col_pos = encode_col_pos(num_observations, x_scale, y_scale)
            logging.debug("  col_pos shape: %s, %s", col_pos[0].shape, col_pos[1].shape)

            obs = encode_obs(num_observations, num_patches)
            logging.debug("  obs shape: %s, %s", obs[0].shape, obs[1].shape)

            logging.info("Running GATO..")

            start = timeit.default_timer()
            hidden_states = gato((input_ids, (encoding, row_pos, col_pos), obs))
            stop = timeit.default_timer()

            logging.info("  Success! hidden_states: %s", hidden_states.shape)
            logging.info("  Time taken: %s seconds", round(stop - start, 2))

    sys.exit(0)
