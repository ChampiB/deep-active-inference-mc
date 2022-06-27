import tensorflow as tf
import numpy as np

log_2_pi = np.log(2.0 * np.pi)
log_2_pi_e = np.log(2.0 * np.pi * np.e)


def kl_div_from_logvar_and_precision(mean_hat, log_var_hat, mean, log_var, omega):
    return 0.5 * (log_var - tf.math.log(omega) - log_var_hat) \
           + tf.exp(log_var_hat) / (2.0 * tf.exp(log_var) / omega) \
           + tf.math.square(mean_hat - mean) / (2.0 * tf.exp(log_var) / omega) - 0.5


def kl_div_from_logvar(mu1, logvar1, mu2, logvar2):
    return 0.5*(logvar2 - logvar1) + (tf.exp(logvar1) + tf.math.square(mu1 - mu2)) / (2.0 * tf.exp(logvar2)) - 0.5


@tf.function
def entropy_normal_from_logvar(logvar):
    return 0.5*(log_2_pi_e + logvar)


def entropy_bernoulli(p, displacement=0.00001):
    return - (1 - p) * tf.math.log(displacement + 1 - p) \
           - p * tf.math.log(displacement + p)


def log_bernoulli(x, p, displacement=0.00001):
    return x*tf.math.log(displacement + p) + (1-x)*tf.math.log(displacement + 1 - p)


def calc_reward(o, resolution=64):
    perfect_reward = np.zeros((3, resolution, 1), dtype=np.float32)
    perfect_reward[:, :int(resolution/2)] = 1.0
    return log_bernoulli(o[:, 0:3, 0:resolution, :], perfect_reward)


def total_correlation(data):
    Cov = np.cov(data.T)
    return 0.5*(np.log(np.diag(Cov)).sum() - np.linalg.slogdet(Cov)[1])
