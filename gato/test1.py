import sys
import tensorflow as tf
from gato import Gato, GatoConfig

# Create model instance
config = GatoConfig.small()
gato = Gato(config)

# Fake inputs for Gato
input_dim = config.input_dim
print("input_dim: ", input_dim)

# Read PNG file
image = tf.io.read_file("images/snap_512x512_2023-12-05_19-54-06-438.png")
image = tf.image.decode_png(image, channels=3)
# Convert to float32
image = tf.cast(image, dtype=tf.float32)
# Scale to [0, 1]
image = image / 255.0
# Convert to tensor
image = tf.expand_dims(image, axis=0)
print("image: ", type(image), image.shape)

sys.exit(0)
input_ids = tf.concat([
  # ...
  # observation 1
  tf.random.uniform((1, 1, input_dim)),  # image patch 0
  tf.random.uniform((1, 1, input_dim)),  # image patch 1
  tf.random.uniform((1, 1, input_dim)),  # image patch 2
  # ...
  tf.random.uniform((1, 1, input_dim)),  # image patch 19
  tf.fill((1, 1, input_dim), value=0.25),  # continuous value
  tf.fill((1, 1, input_dim), value=624.0),  # discrete (actions, texts)

  # observation 2
  tf.random.uniform((1, 1, input_dim)),  # image patch 0
  tf.random.uniform((1, 1, input_dim)),  # image patch 1
  tf.random.uniform((1, 1, input_dim)),  # image patch 2
  # ...
  tf.random.uniform((1, 1, input_dim)),  # image patch 19
  tf.fill((1, 1, input_dim), value=0.12),  # continuous value
  tf.fill((1, 1, input_dim), value=295.0)  # discrete (actions, texts)
  # ...
], axis=1)
encoding = tf.constant([
  # 0 - image patch embedding
  # 1 - continuous value embedding
  # 2 - discrete embedding (actions, texts)
  [0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 1, 2]
])
row_pos = (
  tf.constant([[0.00, 0.25, 0.50, 0.75, 0, 0, 0.00, 0.25, 0.50, 0.75, 0, 0]]),  # pos_from
  tf.constant([[0.25, 0.50, 0.75, 1.00, 0, 0, 0.25, 0.50, 0.75, 1.00, 0, 0]])   # pos_to
)
col_pos = (
  tf.constant([[0.00, 0.00, 0.00, 0.80, 0, 0, 0.00, 0.00, 0.00, 0.80, 0, 0]]),  # pos_from
  tf.constant([[0.20, 0.20, 0.20, 1.00, 0, 0, 0.20, 0.20, 0.20, 1.00, 0, 0]])   # pos_to
)
obs = (
  tf.constant([[ 0,  1,  2, 19, 20, 21,  0,  1,  2, 19, 20, 21]]),  # obs token
  tf.constant([[ 1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  0]])   # obs token masking (for action tokens)
)

print("Running GATO..")
hidden_states = gato((input_ids, (encoding, row_pos, col_pos), obs))

# Print hidden states
print("Hidden states:")
print(hidden_states)