import time

import torch
import torch.nn as nn
import torch.optim as optim

from .model.LSTMSimpleModel import LSTMSimpleModel
from .utils.LSTMSimpleDataset import LSTMSimpleDataset, get_input_size
from ...utils.dataset_util import DatasetInputType, GeometricFeature, SetType, normalise_skeleton_3d_batch
from ...utils.model_name_generator import ModelNameGenerator
from ...utils.training_utils import save_model_common, save_diagram_common, print_train_results, \
    Optimizer, save_loss_common, validate_model, get_training_batch_accuracy


def train(classes, training_data, training_labels, validation_data, validation_labels, analysed_kpts_description, hidden_layers=3,
          dropout=0.5, epoch_nb=10000, batch_size=128, hidden_size=128, learning_rate=0.0001, print_every=50, weight_decay=0, momentum=0.9,
          val_every=5, steps=32, split=20, input_type=DatasetInputType.SPLIT, optimizer_type=Optimizer.RMSPROP,
          geometric_feature=GeometricFeature.JOINT_COORDINATE, results_path='results', model_name_suffix='', save_loss=True,
          save_diagram=True, save_model=True, save_model_for_inference=False, add_random_rotation_y=False, use_cache=False, is_3d=True,
          show_diagram=True, print_results=True, remove_cache=False, use_normalization=True, add_timestamp=True):
    method_name = 'lstm_simple'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    input_size = get_input_size(geometric_feature, len(analysed_kpts_description.values()), is_3d)

    lstm_model = LSTMSimpleModel(input_size, hidden_size, hidden_layers, len(classes), dropout).to(device)

    criterion = nn.NLLLoss()

    if optimizer_type == Optimizer.RMSPROP:
        optimizer = optim.RMSprop(lstm_model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    elif optimizer_type == Optimizer.SGD:
        optimizer = optim.SGD(lstm_model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    elif optimizer_type == Optimizer.ADAM:
        optimizer = optim.Adam(lstm_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:
        raise Exception('Unknown optimizer')

    all_train_losses = []
    all_val_losses = []

    all_batch_training_accuracies = []
    all_batch_val_accuracies = []

    start_time = time.time()
    epoch = 0

    if use_normalization:
        training_data = normalise_skeleton_3d_batch(training_data, analysed_kpts_description['left_hip'],
                                                    analysed_kpts_description['right_hip'])
        validation_data = normalise_skeleton_3d_batch(validation_data, analysed_kpts_description['left_hip'],
                                                      analysed_kpts_description['right_hip'])

    train_data_loader = LSTMSimpleDataset(training_data, training_labels, batch_size, analysed_kpts_description, SetType.TRAINING,
                                          input_type, steps=steps, split=split, geometric_feature=geometric_feature,
                                          add_random_rotation_y=add_random_rotation_y, use_cache=use_cache, remove_cache=remove_cache)
    validation_data_loader = LSTMSimpleDataset(validation_data, validation_labels, batch_size, analysed_kpts_description,
                                               SetType.VALIDATION, input_type, steps=steps, split=split,
                                               geometric_feature=geometric_feature, use_cache=use_cache, remove_cache=remove_cache)

    for epoch in range(epoch_nb):
        data, train_y = next(iter(train_data_loader))
        tensor_train_y = torch.from_numpy(train_y).to(device)

        optimizer.zero_grad()

        tensor_train_x = torch.tensor(data.reshape((data.shape[0], data.shape[1], -1)), dtype=torch.float, device=device)

        output = lstm_model(tensor_train_x)

        loss = criterion(output, tensor_train_y)

        loss.backward()

        optimizer.step()

        all_train_losses.append(loss.item())

        if epoch % print_every == 0 and epoch > 0 and print_results:
            print_train_results(classes, output, tensor_train_y, epoch, epoch_nb, start_time, loss, batch_size, print_every)

        if epoch % val_every == 0 and epoch > 0:
            all_batch_training_accuracies.append(get_training_batch_accuracy(classes, output, tensor_train_y, batch_size)[1])
            with torch.no_grad():
                data_val, val_y = next(iter(validation_data_loader))
                tensor_val_y = torch.from_numpy(val_y).to(device)
                tensor_val_x = torch.tensor(data_val.reshape((data_val.shape[0], data_val.shape[1], -1)), dtype=torch.float, device=device)
                output_val = lstm_model(tensor_val_x)
                loss_val = criterion(output_val, tensor_val_y)
                val_loss, batch_acc = validate_model(tensor_val_y, output_val, classes, epoch, epoch_nb, print_every, start_time,
                                                     batch_size, loss_val, print_results=print_results)
                all_val_losses.append(val_loss)
                all_batch_val_accuracies.append(batch_acc)

    model_name = ModelNameGenerator(method_name, model_name_suffix, add_timestamp) \
        .add_epoch_number(epoch_nb) \
        .add_batch_size(batch_size) \
        .add_learning_rate(learning_rate) \
        .add_optimizer_name(optimizer_type.name) \
        .add_geometric_feature(geometric_feature.name) \
        .add_hidden_size(hidden_size) \
        .add_hidden_layers(hidden_layers) \
        .add_input_type(input_type.name) \
        .add_dropout(dropout) \
        .add_momentum(momentum) \
        .add_weight_decay(weight_decay) \
        .add_split(split) \
        .add_steps(steps) \
        .add_random_rotation_y(add_random_rotation_y) \
        .add_is_3d(is_3d) \
        .add_is_normalization_used(use_normalization) \
        .generate()

    if save_model:
        save_model_common(lstm_model, optimizer, epoch, val_every, all_train_losses, all_val_losses,
                          save_model_for_inference, results_path, model_name)

    if save_loss:
        save_loss_common(all_train_losses, all_val_losses, model_name, results_path, all_batch_training_accuracies,
                         all_batch_val_accuracies)

    if save_diagram:
        save_diagram_common(all_train_losses, all_val_losses, model_name, val_every, epoch_nb, results_path,
                            all_batch_training_accuracies, all_batch_val_accuracies, show_diagram=show_diagram)

    return lstm_model
