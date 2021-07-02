import torch
import numpy as np

from .utils import get_test_data, classes


def test(lstm_rp, lstm_jjd, lstm_jld, batch_cache_path='batch_cache'):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    correct_rp = 0
    correct_jjd = 0
    correct_jld = 0

    with torch.no_grad():
        test_rp_x, test_jjd_x, test_jld_x, labels = get_test_data(batch_cache_path=batch_cache_path)
        all_count = len(labels)
        for i in range(len(labels)):
            category = classes[labels[i]]

            tensor_test_rp_x = torch.from_numpy(np.array([test_rp_x[i]])).float()
            tensor_test_jjd_x = torch.from_numpy(np.array([test_jjd_x[i]])).float()
            tensor_test_jld_x = torch.from_numpy(np.array([test_jld_x[i]])).float()

            tensor_test_rp_x.to(device)
            tensor_test_jjd_x.to(device)
            tensor_test_jld_x.to(device)

            output_rp = lstm_rp(tensor_test_rp_x)
            output_jjd = lstm_jjd(tensor_test_jjd_x)
            output_jld = lstm_jld(tensor_test_jld_x)

            guess_rp = classes[int(torch.argmax(torch.exp(output_rp)[0]).item())]
            guess_jjd = classes[int(torch.argmax(torch.exp(output_jjd)[0]).item())]
            guess_jld = classes[int(torch.argmax(torch.exp(output_jld)[0]).item())]

            if guess_rp == category:
                correct_rp += 1
            if guess_jjd == category:
                correct_jjd += 1
            if guess_jld == category:
                correct_jld += 1

    print('Mean accuracy RP: {}%'.format(correct_rp / all_count * 100))
    print('Mean accuracy JJD: {}%'.format(correct_jjd / all_count * 100))
    print('Mean accuracy JLD: {}%'.format(correct_jld / all_count * 100))

