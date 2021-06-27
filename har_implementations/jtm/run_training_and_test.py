from impl.evaluate import evaluate
from impl.test import test


def main():
    dataset_path = '../../datasets/berkeley_mhad/3d'
    model_alexnet_front, model_alexnet_top, model_alexnet_side = evaluate(dataset_path)
    test(model_alexnet_front, model_alexnet_top, model_alexnet_side, dataset_path)


if __name__ == '__main__':
    main()
