import sys
import json
import timeit
import argparse
import logging
import tensorflow as tf
from gato import Gato, GatoConfig, DataLoader


# Set logging level
logging.basicConfig(level=logging.INFO)

file = "experiments/Cuboid100Episodes/MasterJsonForEpisodes2023-12-08_22-24-37-645.json"

data = DataLoader(file)


(all_ids, all_encoding, all_row_pos, all_col_pos, all_obs) = data.load()

gato = Gato(GatoConfig.small(), trainable=True)

logging.info("Running GATO..")
hidden_states = gato((all_ids, (all_encoding, all_row_pos, all_col_pos), all_obs))
