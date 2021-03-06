import time
from enum import Enum, auto

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms

from .utils.JTMDataset import JTMDataset
from ...utils.dataset_util import SetType, normalise_skeleton_3d_batch
from ...utils.model_name_generator import ModelNameGenerator
from ...utils.training_utils import save_model_common, save_diagram_common, Optimizer, save_loss_common, time_since


class NeuralNetworkModel(Enum):
    ALEXNET = auto()
    RESNET152 = auto()
    MOBILENETV2 = auto()
    SHUFFLENETV2 = auto()


def train(classes, training_data, training_labels, validation_data, validation_labels,
          analysed_kpts_description, image_width, image_height, epoch_nb=100, batch_size=64, learning_rate=0.0001, gamma_step_lr=0.1,
          step_size_lr=30, print_every=5, val_every=5, weight_decay=0, momentum=0.9, action_repetitions=100, results_path='results',
          model_name_suffix='', optimizer_type=Optimizer.SGD, neural_network_model=NeuralNetworkModel.ALEXNET,
          save_loss=True, save_diagram=True, save_model=True, save_model_for_inference=False, use_cache=False, show_diagram=True,
          print_results=True, remove_cache=False, add_timestamp=True, use_normalization=True):
    method_name = 'jtm'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])

    if use_normalization:
        training_data = normalise_skeleton_3d_batch(training_data, analysed_kpts_description['left_hip'],
                                                    analysed_kpts_description['right_hip'])
        validation_data = normalise_skeleton_3d_batch(validation_data, analysed_kpts_description['left_hip'],
                                                      analysed_kpts_description['right_hip'])

    model_front, model_top, model_side = get_neural_network_models(neural_network_model, classes, device)

    if optimizer_type == Optimizer.RMSPROP:
        optimizer_front = optim.RMSprop(model_front.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
        optimizer_top = optim.RMSprop(model_top.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
        optimizer_side = optim.RMSprop(model_side.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    elif optimizer_type == Optimizer.SGD:
        optimizer_front = optim.SGD(model_front.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
        optimizer_top = optim.SGD(model_top.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
        optimizer_side = optim.SGD(model_side.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    elif optimizer_type == Optimizer.ADAM:
        optimizer_front = optim.Adam(model_front.parameters(), lr=learning_rate, weight_decay=weight_decay)
        optimizer_top = optim.Adam(model_top.parameters(), lr=learning_rate, weight_decay=weight_decay)
        optimizer_side = optim.Adam(model_side.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:
        raise Exception('Unknown optimizer')

    criterion_front = nn.CrossEntropyLoss()
    criterion_top = nn.CrossEntropyLoss()
    criterion_side = nn.CrossEntropyLoss()

    scheduler_front = optim.lr_scheduler.StepLR(optimizer_front, step_size=step_size_lr, gamma=gamma_step_lr)
    scheduler_top = optim.lr_scheduler.StepLR(optimizer_top, step_size=step_size_lr, gamma=gamma_step_lr)
    scheduler_side = optim.lr_scheduler.StepLR(optimizer_side, step_size=step_size_lr, gamma=gamma_step_lr)

    train_data_loader = JTMDataset(training_data, training_labels, image_width, image_height, batch_size, SetType.TRAINING,
                                   analysed_kpts_description, action_repetitions, use_cache, remove_cache)
    val_data_loader = JTMDataset(validation_data, validation_labels, image_width, image_height, batch_size, SetType.VALIDATION,
                                 analysed_kpts_description, action_repetitions, use_cache, remove_cache)

    all_train_losses_front = []
    all_train_losses_top = []
    all_train_losses_side = []

    all_val_losses_front = []
    all_val_losses_top = []
    all_val_losses_side = []

    all_batch_training_accuracies_front = []
    all_batch_training_accuracies_top = []
    all_batch_training_accuracies_side = []

    all_batch_val_accuracies_front = []
    all_batch_val_accuracies_top = []
    all_batch_val_accuracies_side = []

    start_time = time.time()

    epoch = 0
    for epoch in range(epoch_nb):
        data, labels = next(iter(train_data_loader))

        all_train_losses_front_batch = []
        all_train_losses_top_batch = []
        all_train_losses_side_batch = []

        train_acc_front = 0.0
        train_acc_top = 0.0
        train_acc_side = 0.0

        for i, (img, lbl) in enumerate(zip(data, labels)):
            img_tensor_front = torch.unsqueeze(transform(img[0]), 0).to(device)
            img_tensor_top = torch.unsqueeze(transform(img[1]), 0).to(device)
            img_tensor_side = torch.unsqueeze(transform(img[2]), 0).to(device)
            lbl_tensor = torch.tensor([lbl]).to(device)

            optimizer_front.zero_grad()
            optimizer_top.zero_grad()
            optimizer_side.zero_grad()

            output_front = model_front(img_tensor_front)
            output_top = model_top(img_tensor_top)
            output_side = model_side(img_tensor_side)

            if torch.argmax(output_front).item() == lbl:
                train_acc_front += 1
            if torch.argmax(output_top).item() == lbl:
                train_acc_top += 1
            if torch.argmax(output_side).item() == lbl:
                train_acc_side += 1

            loss_front = criterion_front(output_front, lbl_tensor)
            loss_top = criterion_top(output_top, lbl_tensor)
            loss_side = criterion_side(output_side, lbl_tensor)

            loss_front.backward()
            loss_top.backward()
            loss_side.backward()

            optimizer_front.step()
            optimizer_top.step()
            optimizer_side.step()

            all_train_losses_front_batch.append(loss_front.item())
            all_train_losses_top_batch.append(loss_top.item())
            all_train_losses_side_batch.append(loss_side.item())

        all_train_losses_front.append(np.mean(all_train_losses_front_batch))
        all_train_losses_top.append(np.mean(all_train_losses_top_batch))
        all_train_losses_side.append(np.mean(all_train_losses_side_batch))

        if epoch % print_every == 0 and epoch > 0 and print_results:
            print('TRAIN_FRONT: %d %d%% (%s) %.4f [%d/%d -> %.2f%%]' % (
                epoch, epoch / epoch_nb * 100, time_since(start_time), loss_front, train_acc_front, batch_size,
                train_acc_front / batch_size * 100))
            print('TRAIN_TOP: %d %d%% (%s) %.4f [%d/%d -> %.2f%%]' % (
                epoch, epoch / epoch_nb * 100, time_since(start_time), loss_top, train_acc_top, batch_size,
                train_acc_top / batch_size * 100))
            print('TRAIN_SIDE: %d %d%% (%s) %.4f [%d/%d -> %.2f%%]' % (
                epoch, epoch / epoch_nb * 100, time_since(start_time), loss_side, train_acc_side, batch_size,
                train_acc_side / batch_size * 100))

        if epoch % val_every == 0 and epoch > 0:
            all_batch_training_accuracies_front.append(train_acc_front / batch_size)
            all_batch_training_accuracies_top.append(train_acc_top / batch_size)
            all_batch_training_accuracies_side.append(train_acc_side / batch_size)
            with torch.no_grad():
                data_valid, labels_valid = next(iter(val_data_loader))

                all_val_losses_front_tmp = []
                all_val_losses_top_tmp = []
                all_val_losses_side_tmp = []

                val_acc_front = 0.0
                val_acc_top = 0.0
                val_acc_side = 0.0

                for i, (val_img, val_lbl) in enumerate(zip(data_valid, labels_valid)):
                    val_img_tensor_front = torch.unsqueeze(transform(val_img[0]), 0).to(device)
                    val_img_tensor_top = torch.unsqueeze(transform(val_img[1]), 0).to(device)
                    val_img_tensor_side = torch.unsqueeze(transform(val_img[2]), 0).to(device)
                    val_lbl_tensor = torch.tensor([val_lbl]).to(device)

                    val_output_front = model_front(val_img_tensor_front)
                    val_output_top = model_top(val_img_tensor_top)
                    val_output_side = model_side(val_img_tensor_side)

                    if torch.argmax(val_output_front).item() == val_lbl:
                        val_acc_front += 1
                    if torch.argmax(val_output_top).item() == val_lbl:
                        val_acc_top += 1
                    if torch.argmax(val_output_side).item() == val_lbl:
                        val_acc_side += 1

                    val_loss_front = criterion_front(val_output_front, val_lbl_tensor)
                    val_loss_top = criterion_top(val_output_top, val_lbl_tensor)
                    val_loss_side = criterion_side(val_output_side, val_lbl_tensor)

                    all_val_losses_front_tmp.append(val_loss_front.item())
                    all_val_losses_top_tmp.append(val_loss_top.item())
                    all_val_losses_side_tmp.append(val_loss_side.item())

                if print_results:
                    print('VALIDATION_FRONT: %d %d%% (%s) %.4f [%d/%d -> %.2f%%]' % (
                        epoch, epoch / epoch_nb * 100, time_since(start_time), val_loss_front, val_acc_front, batch_size,
                        val_acc_front / batch_size * 100))
                    print('VALIDATION_TOP: %d %d%% (%s) %.4f [%d/%d -> %.2f%%]' % (
                        epoch, epoch / epoch_nb * 100, time_since(start_time), val_loss_top, val_acc_top, batch_size,
                        val_acc_top / batch_size * 100))
                    print('VALIDATION_SIDE: %d %d%% (%s) %.4f [%d/%d -> %.2f%%]\n' % (
                        epoch, epoch / epoch_nb * 100, time_since(start_time), val_loss_side, val_acc_side, batch_size,
                        val_acc_side / batch_size * 100))
                all_batch_val_accuracies_front.append(val_acc_front / batch_size)
                all_batch_val_accuracies_top.append(val_acc_top / batch_size)
                all_batch_val_accuracies_side.append(val_acc_side / batch_size)

                all_val_losses_front.append(np.mean(all_val_losses_front_tmp))
                all_val_losses_top.append(np.mean(all_val_losses_top_tmp))
                all_val_losses_side.append(np.mean(all_val_losses_side_tmp))

        scheduler_front.step()
        scheduler_top.step()
        scheduler_side.step()

    model_name = ModelNameGenerator(method_name, model_name_suffix, add_timestamp) \
        .add_epoch_number(epoch_nb) \
        .add_batch_size(batch_size) \
        .add_learning_rate(learning_rate) \
        .add_optimizer_name(optimizer_type.name) \
        .add_momentum(momentum) \
        .add_weight_decay(weight_decay) \
        .add_action_repetitions(action_repetitions) \
        .add_step_size_lr(step_size_lr) \
        .add_gamma_step_lr(gamma_step_lr) \
        .add_neural_network_model(neural_network_model.name) \
        .add_is_normalization_used(use_normalization) \
        .generate()

    if save_model:
        save_model_common(model_front, optimizer_front, epoch, val_every, all_train_losses_front,
                          all_val_losses_front, save_model_for_inference, results_path, model_name + '_front')
        save_model_common(model_top, optimizer_top, epoch, val_every, all_train_losses_top,
                          all_val_losses_top, save_model_for_inference, results_path, model_name + '_top')
        save_model_common(model_side, optimizer_side, epoch, val_every, all_train_losses_side,
                          all_val_losses_side, save_model_for_inference, results_path, model_name + '_side')

    if save_loss:
        save_loss_common(all_train_losses_front, all_val_losses_front, model_name + '_front', results_path,
                         all_batch_training_accuracies_front, all_batch_val_accuracies_front)
        save_loss_common(all_train_losses_top, all_val_losses_top, model_name + '_top', results_path,
                         all_batch_training_accuracies_top, all_batch_val_accuracies_top)
        save_loss_common(all_train_losses_side, all_val_losses_side, model_name + '_side', results_path,
                         all_batch_training_accuracies_side, all_batch_val_accuracies_side)

    if save_diagram:
        save_diagram_common(all_train_losses_front, all_val_losses_front, model_name + '_front', val_every, epoch_nb,
                            results_path, all_batch_training_accuracies_front, all_batch_val_accuracies_front, show_diagram)
        save_diagram_common(all_train_losses_top, all_val_losses_top, model_name + '_top', val_every, epoch_nb, results_path,
                            all_batch_training_accuracies_top, all_batch_val_accuracies_top, show_diagram)
        save_diagram_common(all_train_losses_side, all_val_losses_side, model_name + '_side', val_every, epoch_nb,
                            results_path, all_batch_training_accuracies_side, all_batch_val_accuracies_side, show_diagram)

    return model_front, model_top, model_side


def get_neural_network_models(neural_network_model_type, classes, device):
    if neural_network_model_type == NeuralNetworkModel.ALEXNET:
        # Load AlexNet torch.hub.load('pytorch/vision:v0.6.0', 'alexnet', pretrained=True)
        model_front = models.alexnet(pretrained=True)
        model_top = models.alexnet(pretrained=True)
        model_side = models.alexnet(pretrained=True)

        # Updating the third and the last classifier that is the output layer of the network.
        model_front.classifier[6] = nn.Linear(4096, len(classes))
        model_top.classifier[6] = nn.Linear(4096, len(classes))
        model_side.classifier[6] = nn.Linear(4096, len(classes))
    elif neural_network_model_type == NeuralNetworkModel.RESNET152:
        model_front = models.resnet152(pretrained=True)
        model_top = models.resnet152(pretrained=True)
        model_side = models.resnet152(pretrained=True)

        model_front.fc = nn.Linear(2048, len(classes))
        model_top.fc = nn.Linear(2048, len(classes))
        model_side.fc = nn.Linear(2048, len(classes))
    elif neural_network_model_type == NeuralNetworkModel.MOBILENETV2:
        model_front = models.mobilenet_v2(pretrained=True)
        model_top = models.mobilenet_v2(pretrained=True)
        model_side = models.mobilenet_v2(pretrained=True)

        model_front.classifier[1] = nn.Linear(1280, len(classes))
        model_top.classifier[1] = nn.Linear(1280, len(classes))
        model_side.classifier[1] = nn.Linear(1280, len(classes))
    elif neural_network_model_type == NeuralNetworkModel.SHUFFLENETV2:
        model_front = models.shufflenet_v2_x1_0(pretrained=True)
        model_top = models.shufflenet_v2_x1_0(pretrained=True)
        model_side = models.shufflenet_v2_x1_0(pretrained=True)

        model_front.fc = nn.Linear(1024, len(classes))
        model_top.fc = nn.Linear(1024, len(classes))
        model_side.fc = nn.Linear(1024, len(classes))
    else:
        raise ValueError('Unknown NN model')

    model_front.to(device)
    model_top.to(device)
    model_side.to(device)
    return model_front, model_top, model_side
