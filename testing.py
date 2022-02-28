from src.game_environment import Game
import src.mcts as mcts
import math
from src.tfmodel import ActiveInferenceModel
import os
import time
import argparse
import cv2
import numpy as np
import tensorflow as tf
import src.util as u
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

# Set Tensorflow's verbosity
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Create parser for the command line arguments
parser = argparse.ArgumentParser(description='Training script.')

# Demo arguments
parser.add_argument('-n', '--network', type=str, default='', required=True, help='The path of a checkpoint to be loaded.')
parser.add_argument('-m', '--mean', action='store_true',help='Whether expected free energy should be calculated using the mean instead of sampling..')
parser.add_argument('-d', '--duration', type=int, default=50001, help='Duration of experiment.')
parser.add_argument('-method', '--method', type=str, default='mcts', help='Pre-select method used by the agent for action selection. Available: t1, t12, ai, mcts or habit!')
parser.add_argument('-steps', '--steps', type=int, default=7, help='How many steps ahead the agent can imagine!')
parser.add_argument('-temp', '--temperature', type=float, default=1, help='Initialize testing routine!')
parser.add_argument('-jumps', '--jumps', type=int, default=5, help='Mental jumps: How many steps ahead the agent has learnt to predict in a singe step!')

# MCTS arguments
parser.add_argument("-C", "--C", type=float, help="MCTS parameter: C: Balance between exploration and exploitation..", default=1.0)
parser.add_argument("-repeats", "--repeats", type=int, help="MCTS parameter: Simulation repeats", default=300)
parser.add_argument("-threshold", "--threshold", type=float, help="MCTS parameter: Threshold to make decision prematurely", default=0.5)
parser.add_argument("-depth", "--depth", type=int, help="MCTS parameter: Simulation depth", default=3)
parser.add_argument("-no_habit", "--no_habit", action='store_true', help="MCTS parameter: Disable habitual control as a first choice of the MCTS algorithm.")

# Parse the command line arguments
args = parser.parse_args()

# Check validity of parsed arguments
if args.network[-1] in ['/', '\\']:
    args.network = args.network[:-1]
if args.method not in ['t1', 't12', 'ai', 'mcts', 'habit']:
    raise Exception("Invalid method type.")

# Start tensorflow session
config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

# Hyper-parameters
s_dim = 10
pi_dim = 4
batch_size = 1
debug = False
last_pi = None
samples = 1
if args.steps == -1:
    args.steps = 1

# Create the environment
game = Game(1)
game.randomize_environment(0)
game.current_s[0, -1] = 0.0

# Create the agent
params = mcts.MCTS_Params(args)
model = ActiveInferenceModel(s_dim=s_dim, pi_dim=pi_dim, gamma=1.0, beta_s=1.0, beta_o=1.0, colour_channels=1, resolution=64)
model.load_all(args.network)

# Create the graphical interface
if debug is True:
    cv2.namedWindow('demo', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('demo', 500, 500)

pi0 = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
o0 = game.current_frame(0).reshape(1, 64, 64, 1)
qs0_mean, qs0_logvar = model.model_down.encoder(o0)
s0 = model.model_down.reparameterize(qs0_mean, qs0_logvar)

start_time = time.time()

nb_simulations = 100
nb_action_perception_cycles = 30 * 8
exec_times = []
tot_reward = 0

for i in range(0, nb_simulations):

    game.randomize_environment(0)
    game.current_s[0, 6] = 0.0
    actions = []
    start_time = time.time()

    for t in range(0, nb_action_perception_cycles):

        if len(actions) <= 0:
            # Get observation from the environment
            o_single = game.current_frame(0)
            mcts_path, _, _, _, _ = mcts.active_inference_mcts(model=model, frame=o_single, params=params, o_shape=(64, 64, 1))

            # Push the sequence of actions returned by MCTS into the list of action to execute
            for action in mcts_path:
                for _ in range(args.jumps):
                    actions.append(action)
        else:
            # Execute an action in the environment
            if game.execute_action(actions[0], 0):
                break
            else:
                actions = actions[1:]

        # Display the current frame using Open CV.
        if debug is True and u.display_GUI(game):
            break

        # If the agent have not solved
        if t + 1 >= nb_action_perception_cycles:
            game.current_s[0, 6] = -1

    tot_reward += game.get_reward(0)
    exec_times.append(time.time() - start_time)

avg = sum(exec_times) / len(exec_times)
var = sum((x - avg)**2 for x in exec_times) / (len(exec_times) - 1)
print(exec_times)
print("Time: " + str(avg) + " +/- " + str(math.sqrt(var)))
print("P(solved): " + str((tot_reward + nb_simulations) / (2 * nb_simulations)))

if debug is True:
    cv2.destroyAllWindows()

exit('Exiting ok...!')
