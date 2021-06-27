from impl.evaluate import evaluate


def main():
    dataset_path = '../../datasets/berkeley_mhad/3d'
    model_alexnet_xy, model_alexnet_xz, model_alexnet_yz, model_alexnet_xyz = evaluate(dataset_path)


if __name__ == '__main__':
    main()
