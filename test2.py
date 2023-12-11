import tensorflow as tf
from gato import Gato, GatoConfig

# Create model instance
config = GatoConfig.small()
gato = Gato(config)

# Fake inputs for Gato
input_dim = config.input_dim

print("input_dim: ", input_dim)

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
print ("input_ids shape: ", input_ids.shape)

encoding = tf.constant([
  # 0 - image patch embedding
  # 1 - continuous value embedding
  # 2 - discrete embedding (actions, texts)
  [0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 1, 2]
])
print ("encoding shape: ", encoding.shape)

row_pos = (
  tf.constant([[0.00, 0.25, 0.50, 0.75, 0, 0, 0.00, 0.25, 0.50, 0.75, 0, 0]]),  # pos_from
  tf.constant([[0.25, 0.50, 0.75, 1.00, 0, 0, 0.25, 0.50, 0.75, 1.00, 0, 0]])   # pos_to
)

print ("row pos shape: ", row_pos[0].shape, row_pos[1].shape)

col_pos = (
  tf.constant([[0.00, 0.00, 0.00, 0.80, 0, 0, 0.00, 0.00, 0.00, 0.80, 0, 0]]),  # pos_from
  tf.constant([[0.20, 0.20, 0.20, 1.00, 0, 0, 0.20, 0.20, 0.20, 1.00, 0, 0]])   # pos_to
)
print ("col pos shape: ", col_pos[0].shape, col_pos[1].shape)

obs = (
  tf.constant([[ 0,  1,  2, 19, 20, 21,  0,  1,  2, 19, 20, 21]]),  # obs token
  tf.constant([[ 1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  0]])   # obs token masking (for action tokens)
)

print ("obs shape: ", obs[0].shape, obs[1].shape)

hidden_states = gato((input_ids, (encoding, row_pos, col_pos), obs))