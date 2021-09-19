import time

import torch
import torch.nn as nn
import torch.optim as optim

from .model.GeometricLSTM import GeometricLSTM
from .utils.GeometricLSTMDataset import GeometricLSTMDataset
from ...utils.dataset_util import SetType, DatasetInputType, GeometricFeature
from ...utils.training_utils import save_model_common, save_diagram_common, generate_model_name, print_train_results, \
    Optimizer, save_loss_common, validate_model, get_training_batch_accuracy


def train(classes, training_data, training_labels, validation_data, validation_labels,
          analysed_kpts_description, input_size=36, hidden_layers=3, dropout=0.5,
          epoch_nb=10000, batch_size=128, hidden_size=128, learning_rate=0.0001,
          geometric_feature=GeometricFeature.JOINT_COORDINATE,
          print_every=50, weight_decay=0, momentum=0.9, val_every=5, input_type=DatasetInputType.SPLIT, save_loss=True,
          save_diagram=True, results_path='results', optimizer_type=Optimizer.RMSPROP, save_model=True, use_cache=True,
          save_model_for_inference=False, add_random_rotation_y=False, steps=32, split=20):
    method_name = 'geometric_multilayer_lstm'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    lstm_model = GeometricLSTM(input_size, hidden_size, len(classes), hidden_layers, dropout).to(device)

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

    train_data_loader = GeometricLSTMDataset(training_data, training_labels, batch_size, SetType.TRAINING,
                                             analysed_kpts_description, geometric_feature, input_type,
                                             add_random_rotation_y=add_random_rotation_y,
                                             steps=steps, split=split, use_cache=use_cache)
    validation_data_loader = GeometricLSTMDataset(validation_data, validation_labels, batch_size, SetType.TRAINING,
                                                  analysed_kpts_description, geometric_feature, input_type,
                                                  steps=steps, split=split, use_cache=use_cache)

    for epoch in range(epoch_nb):
        data, train_y = next(iter(train_data_loader))
        tensor_train_y = torch.from_numpy(train_y).to(device)

        data = data.reshape((data.shape[0], data.shape[1], -1))

        optimizer.zero_grad()

        tensor_train_x = torch.tensor(data, dtype=torch.float, device=device)

        output = lstm_model(tensor_train_x)

        loss = criterion(output, tensor_train_y)

        loss.backward()

        optimizer.step()

        all_train_losses.append(loss.item())

        if epoch % print_every == 0 and epoch > 0:
            print_train_results(classes, output, tensor_train_y, epoch, epoch_nb, start_time, loss, batch_size, print_every,
                                geometric_feature.value)

        if epoch % val_every == 0 and epoch > 0:
            all_batch_training_accuracies.append(get_training_batch_accuracy(classes, output, tensor_train_y, batch_size)[1])
            with torch.no_grad():
                data_val, val_y = next(iter(validation_data_loader))
                tensor_val_y = torch.from_numpy(val_y).to(device)
                data_val = data.reshape((data_val.shape[0], data_val.shape[1], -1))
                tensor_val_x = torch.tensor(data_val, dtype=torch.float, device=device)

                output_val = lstm_model(tensor_val_x)
                loss_val_rp = criterion(output_val, tensor_val_y)
                val_loss, batch_acc = validate_model(tensor_val_y, output_val, classes, epoch, epoch_nb, print_every, start_time,
                                                     batch_size, loss_val_rp, geometric_feature.value)

                all_val_losses.append(val_loss)
                all_batch_val_accuracies.append(batch_acc)

    model_name = generate_model_name(method_name, epoch_nb, batch_size, learning_rate, optimizer_type.name, hidden_size)

    if save_diagram:
        save_diagram_common(all_train_losses, all_val_losses, model_name, val_every, epoch_nb, results_path,
                            all_batch_training_accuracies, all_batch_val_accuracies)

    if save_model:
        save_model_common(lstm_model, optimizer, epoch, val_every, all_train_losses, all_val_losses,
                          save_model_for_inference, results_path, model_name)

    if save_loss:
        save_loss_common(all_train_losses, all_val_losses, model_name, results_path, all_batch_training_accuracies,
                         all_batch_val_accuracies)

    return lstm_model
