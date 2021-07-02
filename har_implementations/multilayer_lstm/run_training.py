from impl.train import train
import torch
import os


def main():
    dataset_path = '../../datasets/berkeley_mhad/3d'
    batch_cache_path = 'batch_cache'
    checkpoints_dir = 'checkpoints'
    n_iters = 5000

    lstm_rp, lstm_jjd, lstm_jld = train(dataset_path, batch_cache_path=batch_cache_path, n_iters=n_iters)

    if not os.path.exists(checkpoints_dir):
        os.mkdir(checkpoints_dir)

    torch.save(lstm_rp.state_dict(), os.path.join(checkpoints_dir, 'lstm_rp_model_{}.pth'.format(n_iters)))
    torch.save(lstm_jjd.state_dict(), os.path.join(checkpoints_dir, 'lstm_jjd_model_{}.pth'.format(n_iters)))
    torch.save(lstm_jld.state_dict(), os.path.join(checkpoints_dir, 'lstm_jld_model_{}.pth'.format(n_iters)))




if __name__ == '__main__':
    main()
