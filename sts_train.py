import matplotlib as mpl
from matplotlib import pylab as plt
import matplotlib.dates as mdates

import collections

import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

from tensorflow_probability import distributions as tfd
from tensorflow_probability import sts

tf.enable_v2_behavior()

if tf.test.gpu_device_name() != '/device:GPU:0':
  print('WARNING: GPU device not found.')
else:
  print('SUCCESS: Found GPU: {}'.format(tf.test.gpu_device_name()))

num_forecast_steps = 100


def build_model(observed_time_series, aux_data):
  hour_of_day_effect = sts.Seasonal(
      num_seasons=24,
      observed_time_series=observed_time_series,
      name='hour_of_day_effect')
  day_of_week_effect = sts.Seasonal(
      num_seasons=7, num_steps_per_season=24,
      observed_time_series=observed_time_series,
      name='day_of_week_effect')
  temperature_effect = sts.LinearRegression(
      design_matrix=tf.reshape(aux_data - np.mean(aux_data),
                               (-1, 1)), name='temperature_effect')
  autoregressive = sts.Autoregressive(
      order=1,
      observed_time_series=observed_time_series,
      name='autoregressive')
  model = sts.Sum([hour_of_day_effect,
                   day_of_week_effect,
                   temperature_effect,
                   autoregressive],
                   observed_time_series=observed_time_series)
  return model


if __name__ == '__main__':

    dates = range(0, 1000)
    train_data = np.random.rand(100, 1000)
    aux_data = np.random.rand(100, 1000)

    c1, c2 = 'blue', 'red'

    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(2, 1, 1)
    ax.plot(dates[:-num_forecast_steps],
            train_data[0][:-num_forecast_steps], lw=2, label="training data")
    ax.set_ylabel("Hourly demand (GW)")

    ax = fig.add_subplot(2, 1, 2)

    ax.plot(dates[:-num_forecast_steps],
            aux_data[0][:-num_forecast_steps], lw=2, label="training data", c=c2)
    ax.set_ylabel("Temperature (deg C)")
    ax.set_title("Temperature")
    fig.suptitle("Electricity Demand in Victoria, Australia (2014)",
                fontsize=15)
    fig.autofmt_xdate()

    plt.show()

    demand_model = build_model(train_data[:-num_forecast_steps], aux_data[:-num_forecast_steps])

    # Build the variational surrogate posteriors `qs`.
    variational_posteriors = tfp.sts.build_factored_surrogate_posterior(
        model=demand_model)

    #@title Minimize the variational loss.

    # Allow external control of optimization to reduce test runtimes.
    num_variational_steps = 200 # @param { isTemplate: true}
    num_variational_steps = int(num_variational_steps)

    optimizer = tf.optimizers.Adam(learning_rate=.1)
    # Using fit_surrogate_posterior to build and optimize the variational loss function.
    @tf.function(experimental_compile=True)
    def train():
        elbo_loss_curve = tfp.vi.fit_surrogate_posterior(
            target_log_prob_fn=demand_model.joint_log_prob(
                observed_time_series=train_data[:-num_forecast_steps]),
            surrogate_posterior=variational_posteriors,
            optimizer=optimizer,
            num_steps=num_variational_steps)
        return elbo_loss_curve

    elbo_loss_curve = train()

    plt.plot(elbo_loss_curve)
    plt.show()

    # Draw samples from the variational posterior.
    q_samples_demand_ = variational_posteriors.sample(50)