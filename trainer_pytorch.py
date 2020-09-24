import os
from src.nbeats_theirs.model import NBeatsNet
import matplotlib.pyplot as plt
import torch
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch import tensor

CHECKPOINT_NAME = 'nbeats-training-checkpoint.th'


def split(arr, size):
    arrays = []
    while len(arr) > size:
        slice_ = arr[:size]
        arrays.append(slice_)
        arr = arr[size:]
    arrays.append(arr)
    return arrays


def batcher(dataset, batch_size, infinite=False):
    while True:
        x, y = dataset
        for x_, y_ in zip(split(x, batch_size), split(y, batch_size)):
            yield x_, y_
        if not infinite:
            break


def main():

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # TODO These go in the training setting
    forecast_length = 2
    backcast_length = 2 * forecast_length
    batch_size = 16  # greater than 4 for viz

    from src.data_management import DataHandler, Settings
    data_settings = {'dataset': 'm4',
                     'horizon': forecast_length,
                     'm4_time_series_idx': 0,
                     'n_horizons_lookback': 3}
    settings = Settings(data_settings, 'data')
    data = DataHandler(settings)

    from src.utilities import lag_features, prune_data
    lags = backcast_length
    features_tr = lag_features(data.features_tr.multioutput, lags)
    features_tr, labels_tr = prune_data(features_tr, data.labels_tr.multioutput)

    dataset = TensorDataset(tensor(features_tr.values), tensor(labels_tr.values))
    trainloader = DataLoader(dataset, shuffle=True, batch_size=batch_size)

    print('--- Model ---')
    net = NBeatsNet(device=device,
                    stack_types=[NBeatsNet.TREND_BLOCK, NBeatsNet.SEASONALITY_BLOCK, NBeatsNet.GENERIC_BLOCK],
                    forecast_length=forecast_length,
                    thetas_dims=[2, 8, 3],
                    nb_blocks_per_stack=3,
                    backcast_length=backcast_length,
                    hidden_layer_units=64,  # 1024
                    share_weights_in_stack=False,
                    nb_harmonics=None)

    optimiser = optim.Adam(net.parameters())

    def plot_model(x, target, grad_step):
        print('plot()')
        plot(net, x, target, backcast_length, forecast_length, grad_step)

    max_grad_steps = 10000

    net = simple_fit(net, optimiser, trainloader, plot_model, device, max_grad_steps)

    # prediction:
    features_ts = lag_features(data.features_ts.multioutput, lags)
    features_ts, labels_ts = prune_data(features_ts, data.labels_ts.multioutput)

    torch.no_grad()
    _, forecast = net(torch.tensor(features_ts.values, dtype=torch.float).to(device))
    import pandas as pd
    import numpy as np
    forecast = pd.DataFrame([np.array(forecast[i].data[0]) for i in range(len(forecast))], index=features_ts.index)

    labels_ts = data.labels_ts.single_output
    import matplotlib.pyplot as plt
    plt.plot(labels_ts, 'k')
    plt.plot(forecast)
    k = 1



    # TODO Complete the training pipeline
    # TODO Complete the forecasting pipeline
    # TODO Create a training/forecasting class for NBeats
    # TODO Redo the air-polution dataset (along with training splits etc
    # TODO Benchmark all methods


def simple_fit(net, optimiser, data_generator, on_save_callback, device, max_grad_steps=10000):
    print('--- Training ---')
    # initial_grad_step = load(net, optimiser)
    initial_grad_step = 0
    max_epochs = 1
    losses = []
    for epoch in range(max_epochs):
        for grad_step, (x, target) in enumerate(data_generator):
            grad_step += initial_grad_step
            optimiser.zero_grad()
            net.train()
            backcast, forecast = net(torch.tensor(x, dtype=torch.float).to(device))
            loss = F.mse_loss(forecast, torch.tensor(target, dtype=torch.float).to(device))
            loss.backward()
            optimiser.step()
            print(f'grad_step = {str(grad_step).zfill(6)}, loss = {loss.item():.6f}')
            losses.append(loss)
            # if grad_step % 1000 == 0 or (grad_step < 1000 and grad_step % 100 == 0):
            #     with torch.no_grad():
            #         save(net, optimiser, grad_step)
            #         if on_save_callback is not None:
            #             on_save_callback(x, target, grad_step)
            if grad_step > max_grad_steps:
                print('Finished.')
                break
        print(epoch, 'done')
    import matplotlib.pyplot as plt
    plt.semilogy(losses[5:])
    plt.pause(0.1)
    return net
    k = 1





# def save(model, optimiser, grad_step):
#     torch.save({
#         'grad_step': grad_step,
#         'model_state_dict': model.state_dict(),
#         'optimizer_state_dict': optimiser.state_dict(),
#     }, CHECKPOINT_NAME)
#
#
# def load(model, optimiser):
#     if os.path.exists(CHECKPOINT_NAME):
#         checkpoint = torch.load(CHECKPOINT_NAME)
#         model.load_state_dict(checkpoint['model_state_dict'])
#         optimiser.load_state_dict(checkpoint['optimizer_state_dict'])
#         grad_step = checkpoint['grad_step']
#         print(f'Restored checkpoint from {CHECKPOINT_NAME}.')
#         return grad_step
#     return 0


def plot(net, x, target, backcast_length, forecast_length, grad_step):
    net.eval()
    _, f = net(torch.tensor(x, dtype=torch.float))
    subplots = [221, 222, 223, 224]

    plt.figure(1)
    plt.subplots_adjust(top=0.88)
    for i in range(4):
        ff, xx, yy = f.cpu().numpy()[i], x[i], target[i]
        plt.subplot(subplots[i])
        plt.plot(range(0, backcast_length), xx, color='b')
        plt.plot(range(backcast_length, backcast_length + forecast_length), yy, color='g')
        plt.plot(range(backcast_length, backcast_length + forecast_length), ff, color='r')
        # plt.title(f'step #{grad_step} ({i})')

    output = 'n_beats_{}.png'.format(grad_step)
    plt.savefig(output)
    plt.clf()
    print('Saved image to {}.'.format(output))


if __name__ == '__main__':
    main()
