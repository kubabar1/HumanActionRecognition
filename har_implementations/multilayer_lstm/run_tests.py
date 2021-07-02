import torch
import os

from impl.LSTM import LSTM
from impl.test import test
from impl.utils import classes, get_batch


def main():
    dataset_path = '../../datasets/berkeley_mhad/3d'
    batch_cache_path = 'batch_cache'
    checkpoints_dir = 'checkpoints'
    n_layer = 3
    time_stamp = 20
    hidden_dim = 128
    n_categories = len(classes)

    train_rp, train_jjd, train_jld, _ = get_batch(batch_size=1, batch_cache_path=batch_cache_path)
    input_size_rp = train_rp.shape[2]
    input_size_jjd = train_jjd.shape[2]
    input_size_jld = train_jld.shape[2]

    lstm_rp = LSTM(input_size_rp, hidden_dim, n_categories, n_layer, time_stamp)
    lstm_jjd = LSTM(input_size_jjd, hidden_dim, n_categories, n_layer, time_stamp)
    lstm_jld = LSTM(input_size_jld, hidden_dim, n_categories, n_layer, time_stamp)

    lstm_rp.load_state_dict(torch.load(os.path.join(checkpoints_dir, 'lstm_rp_model.pth')))
    lstm_jjd.load_state_dict(torch.load(os.path.join(checkpoints_dir, 'lstm_jjd_model.pth')))
    lstm_jld.load_state_dict(torch.load(os.path.join(checkpoints_dir, 'lstm_jld_model.pth')))

    lstm_rp.eval()
    lstm_jjd.eval()
    lstm_jld.eval()

    test(lstm_rp, lstm_jjd, lstm_jld, batch_cache_path=batch_cache_path)


if __name__ == '__main__':
    main()
