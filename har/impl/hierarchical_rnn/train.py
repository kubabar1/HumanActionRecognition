import time

import torch
import torch.nn as nn
import torch.optim as optim

from .model.HierarchicalRNNModel import HierarchicalRNNModel
from ...utils.training_utils import save_model_common, save_diagram_common, generate_model_name, print_train_results, \
    Optimizer, save_loss_common, test_model


def train(classes, get_batch, dataset_path, epoch_nb=2000, batch_size=128, hidden_size=256, learning_rate=0.00001,
          print_every=50, weight_decay=0, momentum=0.9, train_every=10, test_every=5, save_loss=True,
          save_diagram=True, results_path='results', optimizer_type=Optimizer.RMSPROP, save_model=True,
          save_model_for_inference=False):
    method_name = 'hierarchical_rnn'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    hierarchical_rnn_model = HierarchicalRNNModel(hidden_size, len(classes)).to(device)

    criterion = nn.NLLLoss()

    if optimizer_type == Optimizer.RMSPROP:
        optimizer = optim.RMSprop(hierarchical_rnn_model.parameters(), lr=learning_rate, momentum=momentum,
                                  weight_decay=weight_decay)
    elif optimizer_type == Optimizer.SGD:
        optimizer = optim.SGD(hierarchical_rnn_model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    elif optimizer_type == Optimizer.ADAM:
        optimizer = optim.Adam(hierarchical_rnn_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:
        raise Exception('Unknown optimizer')

    all_train_losses = []
    all_test_losses = []

    all_batch_training_accuracies = []
    all_batch_test_accuracies = []

    start_time = time.time()
    epoch = 0

    for epoch in range(epoch_nb):
        data, train_y = get_batch(dataset_path, batch_size)
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

        output = hierarchical_rnn_model(tensor_train_x)

        loss = criterion(output, tensor_train_y)

        loss.backward()

        optimizer.step()

        all_train_losses.append(loss.item())

        if epoch % test_every == 0 and epoch > 0:
            train_accuracy = print_train_results(classes, output, tensor_train_y, epoch, epoch_nb, start_time, loss, batch_size,
                                                 print_every)
            all_batch_training_accuracies.append(train_accuracy)
            with torch.no_grad():
                data_test, test_y = get_batch(dataset_path, batch_size, is_training=False)
                tensor_test_y = torch.from_numpy(test_y).to(device)

                left_arms_test = data_test[0]
                right_arms_test = data_test[1]
                left_legs_test = data_test[2]
                right_legs_test = data_test[3]

                tensor_left_arms_test = torch.tensor(left_arms_test, dtype=torch.float, device=device)
                tensor_right_arms_test = torch.tensor(right_arms_test, dtype=torch.float, device=device)
                tensor_left_legs_test = torch.tensor(left_legs_test, dtype=torch.float, device=device)
                tensor_right_legs_test = torch.tensor(right_legs_test, dtype=torch.float, device=device)

                tensor_test_x = [tensor_left_arms_test, tensor_right_arms_test, tensor_left_legs_test, tensor_right_legs_test]

                output_test = hierarchical_rnn_model(tensor_test_x)
                loss_test = criterion(output_test, tensor_test_y)
                test_loss, batch_acc = test_model(tensor_test_y, output_test, classes, epoch, epoch_nb, print_every,
                                                  start_time, batch_size, loss_test)
                all_test_losses.append(test_loss)
                all_batch_test_accuracies.append(batch_acc)

    model_name = generate_model_name(method_name, epoch_nb, batch_size, hidden_size, learning_rate, optimizer_type.name)

    if save_diagram:
        save_diagram_common(all_train_losses, all_test_losses, model_name, test_every, epoch_nb, results_path,
                            all_batch_training_accuracies, all_batch_test_accuracies)

    if save_model:
        save_model_common(hierarchical_rnn_model, optimizer, epoch, train_every, test_every, all_train_losses, all_test_losses,
                          save_model_for_inference, results_path, model_name)

    if save_loss:
        save_loss_common(all_train_losses, all_test_losses, model_name, results_path)

    return hierarchical_rnn_model
