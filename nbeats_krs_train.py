import os
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
import random

import data_preprocess
from nbeats_krs_model import NBeatsNet


def get_script_arguments():
    parser = ArgumentParser()
    parser.add_argument('--task', choices=['m5', 'dummy'], required=True)
    parser.add_argument('--test', action='store_true')
    return parser.parse_args()


def get_metrics(y_true, y_hat):
    error = np.mean(np.square(y_true - y_hat))
    smape = np.mean(2 * np.abs(y_true - y_hat) / (np.abs(y_true) + np.abs(y_hat)))
    return smape, error


def ensure_results_dir():
    if not os.path.exists('results/test'):
        os.makedirs('results/test')


def reshape_array(x):
    assert len(x.shape) == 2, 'input np.array should be in the format: samples, timesteps'
    if len(x.shape) == 2:
        nb_samples, nb_timestamps = x.shape
        return x.reshape((nb_samples, nb_timestamps, 1))


def generate_data(backcast_length, forecast_length, input_dim):
    def gen(num_samples):
        x = np.random.uniform(size=(num_samples, backcast_length, input_dim))
        y = np.random.uniform(size=(num_samples, forecast_length, input_dim))
        # return next(dummy_data_generator_multivariate(backcast_length, forecast_length,
        #                                               signal_type='seasonality', random=True, batch_size=num_samples))
        return x, y

    x_train, y_train = gen(6_000)
    x_test, y_test = gen(1_000)

    # x_train, y_train, x_test, y_test = reshape_array(x_train), reshape_array(y_train), reshape_array(
    #     x_test), reshape_array(y_test)
    return x_train, None, y_train, x_test, None, y_test


# simple batcher.
def data_generator(x_full, y_full, bs):
    def split(arr, size):
        arrays = []
        while len(arr) > size:
            slice_ = arr[:size]
            arrays.append(slice_)
            arr = arr[size:]
        arrays.append(arr)
        return arrays

    while True:
        for rr in split((x_full, y_full), bs):
            yield rr


def get_m5_data_multivariate(backcast_length, forecast_length, batch_size, is_training):
    
    sell_prices_df, calendar_df, sales_train_validation_df, submission_df = data_preprocess.read_data()
    dates_list = data_preprocess.get_dates_list(calendar_df)
    data_df = data_preprocess.get_data_for_store_dept(sales_train_validation_df, dates_list, 'CA_1', 'HOBBIES_1')
    print(data_df.head())
    # data_preprocess.plot_item(data_df, dates_list, 1)

    data = data_df.values.astype(float)
    norm_constant = np.max(data[:-batch_size*forecast_length], axis=0)
    print('Norm constant: ', norm_constant)
    if is_training:
        data = data[:-batch_size*forecast_length]
    else:
        data = data[-backcast_length-batch_size*forecast_length:]
    data = data / norm_constant
    print('Data shape: ', data.shape)

    # data = data[:, :5]

    sample_indexes = list(range(len(data) - forecast_length - backcast_length + 1))
    random.shuffle(sample_indexes)
    print('sample indexes: ', sample_indexes)

    strt = 0

    def m5_datagen():
        nonlocal strt
        should_run = True
        while should_run:
            x_batch, y_batch = [], []
            for _ in range(batch_size):
                start_index = sample_indexes[strt]
                x_batch.append(data[start_index:start_index+backcast_length])
                y_batch.append(data[start_index+backcast_length:start_index+backcast_length+forecast_length])
                strt = strt + 1
                if strt >= len(sample_indexes):
                    if is_training:
                        strt = 0
                    else:
                        should_run = False
                        break
            x = np.array(x_batch)
            y = np.array(y_batch)
            # print('x, y batch shapes: ', x.shape, y.shape)
            yield x, y

    return m5_datagen

def train_model(model: NBeatsNet, task: str, best_perf=np.inf, max_steps=10001, plot_results=500, is_test=False):
    ensure_results_dir()
    # if is_test then override max_steps argument
    if is_test:
        max_steps = 5

    if task == 'dummy':
        x_train, e_train, y_train, x_test, e_test, y_test = generate_data(
            model.backcast_length, model.forecast_length, model.input_dim)
    # TODO return test data
    elif task == 'm5':
        # x_test, e_test, y_test = get_m4_data_multivariate(model.backcast_length, model.forecast_length,
        #                                                   is_training=False)

        m5_data_test = get_m5_data_multivariate(
            model.backcast_length, model.forecast_length, batch_size=8, is_training=False)
        x_test, y_test = next(m5_data_test())
        # x_test = np.zeros(10) # TODO remove
        # pass
    else:
        raise ValueError('Invalid task.')

    print('x_test.shape=', x_test.shape)

    x_train, y_train, e_train = None, None, None
    m5_data_train = get_m5_data_multivariate(
        model.backcast_length, model.forecast_length, batch_size=8, is_training=True)
    # print(m5_data)
    # azz = m5_data()
    # print(len(azz), azz[0].shape, azz[1].shape)

    for step in range(max_steps):
        if task == 'dummy':
            x_train, e_train, y_train, x_test, e_test, y_test = generate_data(model.backcast_length,
                                                                              model.forecast_length,
                                                                              model.input_dim)
        elif task == 'm5':
            x_train, y_train = next(m5_data_train())
            # print('x_train.shape=', x_train.shape)
        else:
            raise ValueError('Invalid task.')

        if model.has_exog():
            model.train_on_batch([x_train, e_train], y_train)
        else:
            model.train_on_batch(x_train, y_train)

        if step % plot_results == 0:
            print('step=', step)
            model.save('results/n_beats_model_' + str(step) + '.h5')
            if model.has_exog():
                predictions = model.predict([x_train, e_train])
                validation_predictions = model.predict([x_test, e_test])
            else:
                predictions = model.predict(x_train)
                validation_predictions = model.predict(x_test)
                clip_validation_predictions = np.maximum(1e-8, validation_predictions)
                # print(clip_validation_predictions.shape)
            smape = get_metrics(y_test, clip_validation_predictions)[0]
            print('test smape=', smape)
            if smape < best_perf:
                best_perf = smape
                model.save('results/n_beats_model_ongoing.h5')
            # for k in range(model.input_dim):
            for k in range(10):
                plot_keras_model_predictions(model, False, step, x_train[0, :, k], y_train[0, :, k],
                                             predictions[0, :, k], axis=k)
                plot_keras_model_predictions(model, True, step, x_test[0, :, k], y_test[0, :, k],
                                             clip_validation_predictions[0, :, k], axis=k)

    model.save('results/n_beats_model.h5')

    if model.has_exog():
        predictions = model.predict([x_train, e_train])
        validation_predictions = model.predict([x_test, e_test])
    else:
        predictions = model.predict(x_train)
        validation_predictions = model.predict(x_test)

    # for k in range(model.input_dim):
    for k in range(5):
        plot_keras_model_predictions(model, False, max_steps, x_train[0, :, k], y_train[0, :, k],
                                     predictions[0, :, k], axis=k)
        plot_keras_model_predictions(model, True, max_steps, x_test[0, :, k], y_test[0, :, k],
                                     validation_predictions[0, :, k], axis=k)
    print('smape=', get_metrics(y_test, validation_predictions)[0])
    print('error=', get_metrics(y_test, validation_predictions)[1])


def plot_keras_model_predictions(model, is_test, step, backcast, forecast, prediction, axis):
    legend = ['backcast', 'forecast', 'predictions of forecast']
    if is_test:
        title = 'results/test/' + 'step_' + str(step) + '_axis_' + str(axis) + '.png'
    else:
        title = 'results/' + 'step_' + str(step) + '_axis_' + str(axis) + '.png'
    plt.figure()
    plt.grid(True)
    x_y = np.concatenate([backcast, forecast], axis=-1).flatten()
    plt.plot(list(range(model.backcast_length)), backcast.flatten(), color='b')
    plt.plot(list(range(len(x_y) - model.forecast_length, len(x_y))), forecast.flatten(), color='g')
    plt.plot(list(range(len(x_y) - model.forecast_length, len(x_y))), prediction.flatten(), color='r')
    plt.scatter(range(len(x_y)), x_y.flatten(), color=['b'] * model.backcast_length + ['g'] * model.forecast_length)
    plt.scatter(list(range(len(x_y) - model.forecast_length, len(x_y))), prediction.flatten(),
                color=['r'] * model.forecast_length)
    plt.legend(legend)
    plt.savefig(title)
    plt.close()


def main():
    args = get_script_arguments()

    if args.task in ['m5', 'dummy']:
        model = NBeatsNet(backcast_length=365, forecast_length=7, input_dim=416,
                          stack_types=(NBeatsNet.GENERIC_BLOCK, NBeatsNet.GENERIC_BLOCK), nb_blocks_per_stack=2,
                          thetas_dim=(4, 4), share_weights_in_stack=True, hidden_layer_units=128)
        # model = NBeatsNet(input_dim=5, backcast_length=365, forecast_length=7,
        #                   stack_types=(NBeatsNet.TREND_BLOCK, NBeatsNet.SEASONALITY_BLOCK), nb_blocks_per_stack=3,
        #                   thetas_dim=(4, 4), share_weights_in_stack=True,
        #                   hidden_layer_units=256)
    # elif args.task == 'kcg':
    #     model = NBeatsNet(input_dim=2, backcast_length=360, forecast_length=10,
    #                       stack_types=(NBeatsNet.TREND_BLOCK, NBeatsNet.SEASONALITY_BLOCK), nb_blocks_per_stack=3,
    #                       thetas_dim=(4, 8), share_weights_in_stack=False,
    #                       hidden_layer_units=256)
    # elif args.task == 'nrj':
    #     model = NBeatsNet(input_dim=1, exo_dim=2, backcast_length=10, forecast_length=1,
    #                       stack_types=(NBeatsNet.TREND_BLOCK, NBeatsNet.SEASONALITY_BLOCK), nb_blocks_per_stack=2,
    #                       thetas_dim=(4, 8), share_weights_in_stack=False, hidden_layer_units=128,
    #                       nb_harmonics=10)
    else:
        raise ValueError('Unknown task.')

    # model.compile_model(loss='mae', learning_rate=1e-5)
    model.compile_model(loss='mse', learning_rate=1e-4)
    train_model(model, args.task, is_test=args.test)


if __name__ == '__main__':
    main()