import os
import shutil
import unittest
import zipfile
from random import randrange

from har.impl.lstm_simple.evaluate import evaluate_tests, load_model, fit
from har.impl.lstm_simple.train import train
from har.utils.dataset_util import berkeley_mhad_classes, SetType, video_pose_3d_kpts, get_berkeley_dataset

unittest.TestLoader.sortTestMethodsUsing = None


class TestLSTMSimple(unittest.TestCase):
    test_results_path = 'tests/results'
    test_dataset_resource_path = 'tests/test_resources/berkeley-3D.zip'
    test_dataset_path = 'tests/test_dataset'

    def setUp(self):
        if os.path.exists(self.test_results_path):
            shutil.rmtree(self.test_results_path)
        if os.path.exists(self.test_dataset_path):
            shutil.rmtree(self.test_dataset_path)
        with zipfile.ZipFile(self.test_dataset_resource_path, 'r') as zip_ref:
            zip_ref.extractall(self.test_dataset_path)

    def test_run_training_evaluate_fit(self):
        generated_model_name = 'model_lstm_simple_en_5_bs_128_lr_0.0001_op_RMSPROP_geo_JOINT_COORDINATE_hs_128_hl_3_it_SPLIT_dropout_0.5_momentum_0.9_wd_0_split_20_steps_32_3D.pth'
        generated_acc_diagram = 'model_lstm_simple_en_5_bs_128_lr_0.0001_op_RMSPROP_geo_JOINT_COORDINATE_hs_128_hl_3_it_SPLIT_dropout_0.5_momentum_0.9_wd_0_split_20_steps_32_3D_acc.png'
        generated_loss_diagram = 'model_lstm_simple_en_5_bs_128_lr_0.0001_op_RMSPROP_geo_JOINT_COORDINATE_hs_128_hl_3_it_SPLIT_dropout_0.5_momentum_0.9_wd_0_split_20_steps_32_3D_loss.png'
        generated_train_acc = 'model_lstm_simple_en_5_bs_128_lr_0.0001_op_RMSPROP_geo_JOINT_COORDINATE_hs_128_hl_3_it_SPLIT_dropout_0.5_momentum_0.9_wd_0_split_20_steps_32_3D_train_acc.npy'
        generated_train_loss = 'model_lstm_simple_en_5_bs_128_lr_0.0001_op_RMSPROP_geo_JOINT_COORDINATE_hs_128_hl_3_it_SPLIT_dropout_0.5_momentum_0.9_wd_0_split_20_steps_32_3D_train_loss.npy'
        generated_val_acc = 'model_lstm_simple_en_5_bs_128_lr_0.0001_op_RMSPROP_geo_JOINT_COORDINATE_hs_128_hl_3_it_SPLIT_dropout_0.5_momentum_0.9_wd_0_split_20_steps_32_3D_val_acc.npy'
        generated_val_loss = 'model_lstm_simple_en_5_bs_128_lr_0.0001_op_RMSPROP_geo_JOINT_COORDINATE_hs_128_hl_3_it_SPLIT_dropout_0.5_momentum_0.9_wd_0_split_20_steps_32_3D_val_loss.npy'


        training_data, training_labels = get_berkeley_dataset(os.path.join(self.test_dataset_path, 'berkeley-3D'),
                                                              set_type=SetType.TRAINING)
        validation_data, validation_labels = get_berkeley_dataset(os.path.join(self.test_dataset_path, 'berkeley-3D'),
                                                                  set_type=SetType.VALIDATION)
        test_data, test_labels = get_berkeley_dataset(os.path.join(self.test_dataset_path, 'berkeley-3D'), set_type=SetType.TEST)

        run_train_test(training_data, training_labels, validation_data, validation_labels, self.test_results_path)

        lstm_simple_model = run_load_model_test(os.path.join(self.test_results_path, generated_model_name))

        run_evaluation_test(lstm_simple_model, test_data, test_labels, self.test_results_path)

        run_fit_test(lstm_simple_model, test_data, test_labels)

        assert os.path.exists(os.path.join(self.test_results_path, generated_model_name))
        assert os.path.exists(os.path.join(self.test_results_path, generated_acc_diagram))
        assert os.path.exists(os.path.join(self.test_results_path, generated_loss_diagram))
        assert os.path.exists(os.path.join(self.test_results_path, generated_train_acc))
        assert os.path.exists(os.path.join(self.test_results_path, generated_train_loss))
        assert os.path.exists(os.path.join(self.test_results_path, generated_val_acc))
        assert os.path.exists(os.path.join(self.test_results_path, generated_val_loss))

        assert os.path.exists(os.path.join(self.test_results_path, 'evaluate.png'))


def run_train_test(training_data, training_labels, validation_data, validation_labels, test_results_path):
    train(berkeley_mhad_classes, training_data, training_labels, validation_data, validation_labels, video_pose_3d_kpts, epoch_nb=5,
          show_diagram=False, results_path=test_results_path, val_every=2, print_every=2, print_results=False)


def run_load_model_test(generated_model_path):
    return load_model(generated_model_path, berkeley_mhad_classes, video_pose_3d_kpts)


def run_evaluation_test(lstm_simple_model, test_data, test_labels, results_path, print_results=False):
    accuracy = evaluate_tests(berkeley_mhad_classes, test_data, test_labels, lstm_simple_model, video_pose_3d_kpts,
                              results_path=results_path, show_diagram=False)
    if print_results:
        print('Test accuracy: {}'.format(accuracy))


def run_fit_test(lstm_simple_model, test_data, test_labels, print_results=False):
    random_id = randrange(len(test_labels))
    test_sequence, test_label = test_data[random_id], test_labels[random_id]
    predicted = fit(berkeley_mhad_classes, test_sequence, lstm_simple_model, video_pose_3d_kpts)
    if print_results:
        print('CORRECT: {}'.format(berkeley_mhad_classes[test_label]))
        print('PREDICTED: {}'.format(predicted))

