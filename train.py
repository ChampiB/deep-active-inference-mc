import os, argparse
from sys import argv
from distutils.dir_util import copy_tree
from src.game_environment import Game
import src.util as u
import src.tfloss as loss
from src.tfmodel import ActiveInferenceModel
from src.tfutils import *

parser = argparse.ArgumentParser(description='Training script.')
parser.add_argument('-r', '--resume', action='store_true', help='If this is used, the script tries to load existing weights and resume training.')
parser.add_argument('-b', '--batch', type=int, default=50, help='Select batch size.')
args = parser.parse_args()

var_a = 1.0
var_b = 25.0
var_c = 5.0
var_d = 1.5
s_dim = 10
pi_dim = 4
beta_s = 1.0
beta_o = 1.0
gamma = 0.0
gamma_rate = 1e-05
gamma_max = 0.8
gamma_delay = 30000
deepness = 1
samples = 1
repeats = 5
l_rate_top = 1e-04
l_rate_mid = 1e-04
l_rate_down = 0.001
epochs = 1000

signature = 'final_model_'
signature += str(gamma_rate)+'_'+str(gamma_delay)+'_'+str(var_a)+'_'+str(args.batch)+'_'+str(s_dim)+'_'+str(repeats)
folder = 'figs_'+signature
folder_chp = folder + '/checkpoints'

try:
    os.mkdir(folder)
except:
    print('Folder already exists!!')
try:
    os.mkdir(folder_chp)
except:
    print('Folder chp creation error')

# Create environment.
games = Game(args.batch)

# Create agent.
model = ActiveInferenceModel(s_dim=s_dim, pi_dim=pi_dim, gamma=gamma, beta_s=beta_s, beta_o=beta_o, colour_channels=1, resolution=64)

# Load model if a save exists.
if args.resume:
    stats, optimizers = model.load_all(folder_chp)
    start_epoch = len(stats['F']) + 1
else:
    start_epoch = 1
    optimizers = {}

# Create optimizers.
if optimizers == {}:
    optimizers['top'] = tf.keras.optimizers.Adam(learning_rate=l_rate_top)
    optimizers['mid'] = tf.keras.optimizers.Adam(learning_rate=l_rate_mid)
    optimizers['down'] = tf.keras.optimizers.Adam(learning_rate=l_rate_down)

for epoch in range(start_epoch, (epochs + 1) * 1000):

    # Increase gamma by gamma_rate if at least gamma_delay epoches have passed
    # and gamma is inferior to gamma_max.
    if epoch > gamma_delay and model.model_down.gamma < gamma_max:
        model.model_down.gamma.assign(model.model_down.gamma+gamma_rate)

    # -- MAKE TRAINING DATA FOR THIS BATCH ---------------------------------
    games.randomize_environment_all()
    o0, o1, pi0, log_Ppi = u.make_batch_dsprites_active_inference(
        games=games, model=model, deepness=deepness, samples=samples, calc_mean=True, repeats=repeats
    )

    # -- TRAIN TOP LAYER ---------------------------------------------------
    qs0, _, _ = model.model_down.encoder_with_sample(o0)
    D_KL_pi = loss.train_model_top(
        model_top=model.model_top, s=qs0, log_Ppi=log_Ppi, optimizer=optimizers['top']
    )
    D_KL_pi = D_KL_pi.numpy()

    current_omega = loss.compute_omega(D_KL_pi, a=var_a, b=var_b, c=var_c, d=var_d).reshape(-1, 1)

    # -- TRAIN MIDDLE LAYER ------------------------------------------------
    qs1_mean, qs1_logvar = model.model_down.encoder(o1)
    ps1_mean, ps1_logvar = loss.train_model_mid(
        model_mid=model.model_mid, s0=qs0, qs1_mean=qs1_mean,
        qs1_logvar=qs1_logvar, Ppi_sampled=pi0, omega=current_omega,
        optimizer=optimizers['mid']
    )

    # -- TRAIN DOWN LAYER --------------------------------------------------
    loss.train_model_down(
        model_down=model.model_down, o1=o1, ps1_mean=ps1_mean,
        ps1_logvar=ps1_logvar, omega=current_omega, optimizer=optimizers['down']
    )

    if epoch % 2 == 0:
        model.save_all(folder_chp, os.path.basename(argv[0]))
    if epoch % 2 == 25:
        # keep the checkpoints every 25 steps
        copy_tree(folder_chp, folder_chp+'_epoch_'+str(epoch))
        os.remove(folder_chp+'_epoch_'+str(epoch)+'/optimizers.pkl')
