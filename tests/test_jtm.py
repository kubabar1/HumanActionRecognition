import os
import unittest
from random import randrange

from har.impl.jtm.evaluate import evaluate_tests, fit, load_model, ModelType
from har.impl.jtm.train import train
from har.utils.dataset_util import berkeley_mhad_classes, SetType, video_pose_3d_kpts, get_berkeley_dataset, berkeley_frame_width, \
    berkeley_frame_height
from tests.utils import setup_test_resources

unittest.TestLoader.sortTestMethodsUsing = None


class TestLSTMSimple(unittest.TestCase):
    test_results_path = 'tests/results'
    test_dataset_resource_path = 'tests/test_resources/berkeley-3D.zip'
    test_dataset_path = 'tests/test_dataset'

    @classmethod
    def setup_class(cls):
        setup_test_resources(cls.test_results_path, cls.test_dataset_path, cls.test_dataset_resource_path)

    def test_run_training_evaluate_fit(self):
        'model_jtm_en_1_bs_64_lr_0.0001_op_SGD_momentum_0.9_wd_0_ar_5_stlr_30_gammastlr_0.1_network_ALEXNET_front.pth'

        generated_model_name_front = 'model_jtm_en_1_bs_32_lr_0.0001_op_SGD_momentum_0.9_wd_0_ar_5_stlr_30_gammastlr_0.1_network_ALEXNET_front.pth'
        generated_acc_diagram_front = 'model_jtm_en_1_bs_32_lr_0.0001_op_SGD_momentum_0.9_wd_0_ar_5_stlr_30_gammastlr_0.1_network_ALEXNET_front_acc.png'
        generated_loss_diagram_front = 'model_jtm_en_1_bs_32_lr_0.0001_op_SGD_momentum_0.9_wd_0_ar_5_stlr_30_gammastlr_0.1_network_ALEXNET_front_loss.png'
        generated_train_acc_front = 'model_jtm_en_1_bs_32_lr_0.0001_op_SGD_momentum_0.9_wd_0_ar_5_stlr_30_gammastlr_0.1_network_ALEXNET_front_train_acc.npy'
        generated_train_loss_front = 'model_jtm_en_1_bs_32_lr_0.0001_op_SGD_momentum_0.9_wd_0_ar_5_stlr_30_gammastlr_0.1_network_ALEXNET_front_train_loss.npy'
        generated_val_acc_front = 'model_jtm_en_1_bs_32_lr_0.0001_op_SGD_momentum_0.9_wd_0_ar_5_stlr_30_gammastlr_0.1_network_ALEXNET_front_val_acc.npy'
        generated_val_loss_front = 'model_jtm_en_1_bs_32_lr_0.0001_op_SGD_momentum_0.9_wd_0_ar_5_stlr_30_gammastlr_0.1_network_ALEXNET_front_val_loss.npy'

        generated_model_name_top = 'model_jtm_en_1_bs_32_lr_0.0001_op_SGD_momentum_0.9_wd_0_ar_5_stlr_30_gammastlr_0.1_network_ALEXNET_top.pth'
        generated_acc_diagram_top = 'model_jtm_en_1_bs_32_lr_0.0001_op_SGD_momentum_0.9_wd_0_ar_5_stlr_30_gammastlr_0.1_network_ALEXNET_top_acc.png'
        generated_loss_diagram_top = 'model_jtm_en_1_bs_32_lr_0.0001_op_SGD_momentum_0.9_wd_0_ar_5_stlr_30_gammastlr_0.1_network_ALEXNET_top_loss.png'
        generated_train_acc_top = 'model_jtm_en_1_bs_32_lr_0.0001_op_SGD_momentum_0.9_wd_0_ar_5_stlr_30_gammastlr_0.1_network_ALEXNET_top_train_acc.npy'
        generated_train_loss_top = 'model_jtm_en_1_bs_32_lr_0.0001_op_SGD_momentum_0.9_wd_0_ar_5_stlr_30_gammastlr_0.1_network_ALEXNET_top_train_loss.npy'
        generated_val_acc_top = 'model_jtm_en_1_bs_32_lr_0.0001_op_SGD_momentum_0.9_wd_0_ar_5_stlr_30_gammastlr_0.1_network_ALEXNET_top_val_acc.npy'
        generated_val_loss_top = 'model_jtm_en_1_bs_32_lr_0.0001_op_SGD_momentum_0.9_wd_0_ar_5_stlr_30_gammastlr_0.1_network_ALEXNET_top_val_loss.npy'

        generated_model_name_side = 'model_jtm_en_1_bs_32_lr_0.0001_op_SGD_momentum_0.9_wd_0_ar_5_stlr_30_gammastlr_0.1_network_ALEXNET_side.pth'
        generated_acc_diagram_side = 'model_jtm_en_1_bs_32_lr_0.0001_op_SGD_momentum_0.9_wd_0_ar_5_stlr_30_gammastlr_0.1_network_ALEXNET_side_acc.png'
        generated_loss_diagram_side = 'model_jtm_en_1_bs_32_lr_0.0001_op_SGD_momentum_0.9_wd_0_ar_5_stlr_30_gammastlr_0.1_network_ALEXNET_side_loss.png'
        generated_train_acc_side = 'model_jtm_en_1_bs_32_lr_0.0001_op_SGD_momentum_0.9_wd_0_ar_5_stlr_30_gammastlr_0.1_network_ALEXNET_side_train_acc.npy'
        generated_train_loss_side = 'model_jtm_en_1_bs_32_lr_0.0001_op_SGD_momentum_0.9_wd_0_ar_5_stlr_30_gammastlr_0.1_network_ALEXNET_side_train_loss.npy'
        generated_val_acc_side = 'model_jtm_en_1_bs_32_lr_0.0001_op_SGD_momentum_0.9_wd_0_ar_5_stlr_30_gammastlr_0.1_network_ALEXNET_side_val_acc.npy'
        generated_val_loss_side = 'model_jtm_en_1_bs_32_lr_0.0001_op_SGD_momentum_0.9_wd_0_ar_5_stlr_30_gammastlr_0.1_network_ALEXNET_side_val_loss.npy'
        generated_evaluate_path = os.path.join(self.test_results_path, 'evaluate.png')

        training_data, training_labels = get_berkeley_dataset(os.path.join(self.test_dataset_path, 'berkeley-3D'),
                                                              set_type=SetType.TRAINING)
        validation_data, validation_labels = get_berkeley_dataset(os.path.join(self.test_dataset_path, 'berkeley-3D'),
                                                                  set_type=SetType.VALIDATION)
        test_data, test_labels = get_berkeley_dataset(os.path.join(self.test_dataset_path, 'berkeley-3D'), set_type=SetType.TEST)

        run_train_test(training_data, training_labels, validation_data, validation_labels, self.test_results_path)

        jtm_model_front = run_load_model_test(os.path.join(self.test_results_path, generated_model_name_front))
        jtm_model_top = run_load_model_test(os.path.join(self.test_results_path, generated_model_name_top))
        jtm_model_side = run_load_model_test(os.path.join(self.test_results_path, generated_model_name_side))

        run_evaluation_test(jtm_model_front, test_data, test_labels, self.test_results_path, print_results=True)

        assert os.path.exists(generated_evaluate_path)

        os.remove(generated_evaluate_path)

        run_evaluation_test(jtm_model_top, test_data, test_labels, self.test_results_path, print_results=True)

        assert os.path.exists(generated_evaluate_path)

        os.remove(generated_evaluate_path)

        run_evaluation_test(jtm_model_side, test_data, test_labels, self.test_results_path, print_results=True)

        assert os.path.exists(generated_evaluate_path)

        run_fit_test(jtm_model_front, test_data, test_labels, ModelType.FRONT, print_results=True)
        run_fit_test(jtm_model_top, test_data, test_labels, ModelType.TOP, print_results=True)
        run_fit_test(jtm_model_side, test_data, test_labels, ModelType.SIDE, print_results=True)

        assert os.path.exists(os.path.join(self.test_results_path, generated_model_name_front))
        assert os.path.exists(os.path.join(self.test_results_path, generated_acc_diagram_front))
        assert os.path.exists(os.path.join(self.test_results_path, generated_loss_diagram_front))
        assert os.path.exists(os.path.join(self.test_results_path, generated_train_acc_front))
        assert os.path.exists(os.path.join(self.test_results_path, generated_train_loss_front))
        assert os.path.exists(os.path.join(self.test_results_path, generated_val_acc_front))
        assert os.path.exists(os.path.join(self.test_results_path, generated_val_loss_front))

        assert os.path.exists(os.path.join(self.test_results_path, generated_model_name_top))
        assert os.path.exists(os.path.join(self.test_results_path, generated_acc_diagram_top))
        assert os.path.exists(os.path.join(self.test_results_path, generated_loss_diagram_top))
        assert os.path.exists(os.path.join(self.test_results_path, generated_train_acc_top))
        assert os.path.exists(os.path.join(self.test_results_path, generated_train_loss_top))
        assert os.path.exists(os.path.join(self.test_results_path, generated_val_acc_top))
        assert os.path.exists(os.path.join(self.test_results_path, generated_val_loss_top))

        assert os.path.exists(os.path.join(self.test_results_path, generated_model_name_side))
        assert os.path.exists(os.path.join(self.test_results_path, generated_acc_diagram_side))
        assert os.path.exists(os.path.join(self.test_results_path, generated_loss_diagram_side))
        assert os.path.exists(os.path.join(self.test_results_path, generated_train_acc_side))
        assert os.path.exists(os.path.join(self.test_results_path, generated_train_loss_side))
        assert os.path.exists(os.path.join(self.test_results_path, generated_val_acc_side))
        assert os.path.exists(os.path.join(self.test_results_path, generated_val_loss_side))


def run_train_test(training_data, training_labels, validation_data, validation_labels, test_results_path):
    train(berkeley_mhad_classes, training_data, training_labels, validation_data, validation_labels, video_pose_3d_kpts,
          berkeley_frame_width, berkeley_frame_height, epoch_nb=1, results_path=test_results_path, val_every=1, print_every=1,
          show_diagram=False, print_results=False, action_repetitions=5, add_timestamp=False, batch_size=32)


def run_load_model_test(generated_model_path):
    return load_model(generated_model_path, len(berkeley_mhad_classes))


def run_evaluation_test(jtm_model, test_data, test_labels, results_path, print_results=False, show_diagram=False):
    accuracy = evaluate_tests(berkeley_mhad_classes, test_data, test_labels, jtm_model, video_pose_3d_kpts, berkeley_frame_width,
                              berkeley_frame_height, ModelType.FRONT, result_path=results_path, show_diagram=show_diagram)
    if print_results:
        print('Test accuracy: {}'.format(accuracy))


def run_fit_test(jtm_model, test_data, test_labels, model_type, print_results=False):
    random_id = randrange(len(test_labels))
    test_sequence, test_label = test_data[random_id], test_labels[random_id]
    predicted = fit(berkeley_mhad_classes, test_sequence, jtm_model, video_pose_3d_kpts, berkeley_frame_width, berkeley_frame_height,
                    model_type)
    if print_results:
        print('CORRECT: {}'.format(berkeley_mhad_classes[test_label]))
        print('PREDICTED: {}'.format(predicted))
