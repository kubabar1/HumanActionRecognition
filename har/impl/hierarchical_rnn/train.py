import time

import torch
import torch.nn as nn
import torch.optim as optim

from .model.HierarchicalRNNModel import HierarchicalRNNModel
from .utils.HierarchicalRNNDataset import HierarchicalRNNDataset
from ...utils.dataset_util import DatasetInputType, normalise_skeleton_3d_batch
from ...utils.model_name_generator import ModelNameGenerator
from ...utils.training_utils import save_model_common, save_diagram_common, print_train_results, Optimizer, save_loss_common, \
    validate_model, get_training_batch_accuracy


def train(classes, training_data, training_labels, validation_data, validation_labels, analysed_kpts_description, epoch_nb=10000,
          batch_size=128, hidden_size=128, learning_rate=0.0001, print_every=50, weight_decay=0, momentum=0.9, val_every=5, steps=32,
          split=20, input_type=DatasetInputType.SPLIT, optimizer_type=Optimizer.RMSPROP, results_path='results', model_name_suffix='',
          save_loss=True, save_diagram=True, save_model=True, save_model_for_inference=False, add_random_rotation_y=False, is_3d=True,
          show_diagram=True, print_results=True, use_normalisation=True, add_timestamp=True):
    method_name = 'hierarchical_rnn'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    input_size = 9 if is_3d else 6

    hrnn_model = HierarchicalRNNModel(input_size, hidden_size, len(classes)).to(device)

    criterion = nn.NLLLoss()

    if optimizer_type == Optimizer.RMSPROP:
        optimizer = optim.RMSprop(hrnn_model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    elif optimizer_type == Optimizer.SGD:
        optimizer = optim.SGD(hrnn_model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    elif optimizer_type == Optimizer.ADAM:
        optimizer = optim.Adam(hrnn_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:
        raise Exception('Unknown optimizer')

    all_train_losses = []
    all_val_losses = []

    all_batch_training_accuracies = []
    all_batch_val_accuracies = []

    start_time = time.time()
    epoch = 0

    if use_normalisation:
        training_data = normalise_skeleton_3d_batch(training_data, analysed_kpts_description['left_hip'],
                                                    analysed_kpts_description['right_hip'])
        validation_data = normalise_skeleton_3d_batch(validation_data, analysed_kpts_description['left_hip'],
                                                      analysed_kpts_description['right_hip'])

    train_data_loader = HierarchicalRNNDataset(training_data, training_labels, batch_size, analysed_kpts_description,
                                               add_random_rotation_y=add_random_rotation_y, steps=steps, split=split)
    validation_data_loader = HierarchicalRNNDataset(validation_data, validation_labels, batch_size, analysed_kpts_description,
                                                    steps=steps, split=split)

    for epoch in range(epoch_nb):
        data, train_y = next(iter(train_data_loader))
        tensor_train_y = torch.from_numpy(train_y).to(device)

        optimizer.zero_grad()

        left_arms = data[0]
        right_arms = data[1]
        left_legs = data[2]
        right_legs = data[3]

        tensor_left_arms = torch.tensor(left_arms, dtype=torch.float, device=device)
        tensor_right_arms = torch.tensor(right_arms, dtype=torch.float, device=device)
        tensor_left_legs = torch.tensor(left_legs, dtype=torch.float, device=device)
        tensor_right_legs = torch.tensor(right_legs, dtype=torch.float, device=device)

        tensor_train_x = [tensor_left_arms, tensor_right_arms, tensor_left_legs, tensor_right_legs]

        output = hrnn_model(tensor_train_x)

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

                left_arms_val = data_val[0]
                right_arms_val = data_val[1]
                left_legs_val = data_val[2]
                right_legs_val = data_val[3]

                tensor_left_arms_val = torch.tensor(left_arms_val, dtype=torch.float, device=device)
                tensor_right_arms_val = torch.tensor(right_arms_val, dtype=torch.float, device=device)
                tensor_left_legs_val = torch.tensor(left_legs_val, dtype=torch.float, device=device)
                tensor_right_legs_val = torch.tensor(right_legs_val, dtype=torch.float, device=device)

                tensor_val_x = [tensor_left_arms_val, tensor_right_arms_val, tensor_left_legs_val, tensor_right_legs_val]

                output_val = hrnn_model(tensor_val_x)
                loss_val = criterion(output_val, tensor_val_y)
                val_loss, batch_acc = validate_model(tensor_val_y, output_val, classes, epoch, epoch_nb, print_every,
                                                     start_time, batch_size, loss_val, print_results=print_results)
                all_val_losses.append(val_loss)
                all_batch_val_accuracies.append(batch_acc)

    model_name = ModelNameGenerator(method_name, model_name_suffix, add_timestamp) \
        .add_epoch_number(epoch_nb) \
        .add_batch_size(batch_size) \
        .add_learning_rate(learning_rate) \
        .add_optimizer_name(optimizer_type.name) \
        .add_hidden_size(hidden_size) \
        .add_input_type(input_type.name) \
        .add_momentum(momentum) \
        .add_weight_decay(weight_decay) \
        .add_split(split) \
        .add_steps(steps) \
        .add_random_rotation_y(add_random_rotation_y) \
        .add_is_3d(is_3d) \
        .add_is_normalization_used(use_normalisation) \
        .generate()

    if save_model:
        save_model_common(hrnn_model, optimizer, epoch, val_every, all_train_losses, all_val_losses,
                          save_model_for_inference, results_path, model_name)

    if save_loss:
        save_loss_common(all_train_losses, all_val_losses, model_name, results_path, all_batch_training_accuracies,
                         all_batch_val_accuracies)

    if save_diagram:
        save_diagram_common(all_train_losses, all_val_losses, model_name, val_every, epoch_nb, results_path,
                            all_batch_training_accuracies, all_batch_val_accuracies, show_diagram)

    return hrnn_model
