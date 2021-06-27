from impl.evaluate import evaluate
from impl.test import test


def main():
    dataset_path = '../../datasets/berkeley_mhad/3d'
    model_alexnet_front, model_alexnet_top, model_alexnet_side = evaluate(dataset_path, loss_print_step=10, samples_count=20, cycles=5)
    test(model_alexnet_front, model_alexnet_top, model_alexnet_side, dataset_path)


if __name__ == '__main__':
    main()
