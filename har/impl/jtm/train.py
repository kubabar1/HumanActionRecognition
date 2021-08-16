import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms

from .utils.JTMDataset import JTMDataset
from ...utils.dataset_util import get_analysed_keypoints, SetType
from ...utils.training_utils import save_model_common, save_diagram_common, generate_model_name, Optimizer, save_loss_common, \
    time_since


def train(classes, training_data, training_labels, validation_data, validation_labels, image_width, image_height,
          epoch_nb=200, batch_size=64, action_repetitions=100,
          learning_rate=0.00001, print_every=2, weight_decay=0.0005, momentum=0.9, step_size=30, gamma=0.1, validate_every=2,
          save_loss=True, save_diagram=True, results_path='results', optimizer_type=Optimizer.RMSPROP,
          save_model=True, save_model_for_inference=False, use_cache=True):
    method_name = 'jtm'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    analysed_kpts_left, analysed_kpts_right = get_analysed_keypoints()

    # Load AlexNet torch.hub.load('pytorch/vision:v0.6.0', 'alexnet', pretrained=True)
    model_alexnet_front = models.alexnet(pretrained=True)
    model_alexnet_top = models.alexnet(pretrained=True)
    model_alexnet_side = models.alexnet(pretrained=True)

    # Define transform
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])

    # Model description
    # print(model_alexnet_front.eval())

    # Updating the third and the last classifier that is the output layer of the network.
    model_alexnet_front.classifier[6] = nn.Linear(4096, len(classes))
    model_alexnet_top.classifier[6] = nn.Linear(4096, len(classes))
    model_alexnet_side.classifier[6] = nn.Linear(4096, len(classes))

    # Move the input and AlexNet_model to GPU
    model_alexnet_front.to(device)
    model_alexnet_top.to(device)
    model_alexnet_side.to(device)

    # Loss
    criterion_front = nn.CrossEntropyLoss()
    criterion_top = nn.CrossEntropyLoss()
    criterion_side = nn.CrossEntropyLoss()

    # Optimizer(SGD)
    optimizer_front = optim.SGD(model_alexnet_front.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    optimizer_top = optim.SGD(model_alexnet_top.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    optimizer_side = optim.SGD(model_alexnet_side.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)

    # Scheduler to change learning rate depends on epoch number
    scheduler_front = optim.lr_scheduler.StepLR(optimizer_front, step_size=step_size, gamma=gamma)
    scheduler_top = optim.lr_scheduler.StepLR(optimizer_top, step_size=step_size, gamma=gamma)
    scheduler_side = optim.lr_scheduler.StepLR(optimizer_side, step_size=step_size, gamma=gamma)

    train_data_loader = JTMDataset(training_data, training_labels, image_width, image_height, action_repetitions, batch_size,
                                   SetType.TRAINING, use_cache)
    val_data_loader = JTMDataset(validation_data, validation_labels, image_width, image_height, action_repetitions, batch_size,
                                 SetType.VALIDATION, use_cache)

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

    for epoch in range(epoch_nb):
        data, labels = next(iter(train_data_loader))

        all_train_losses_front_tmp = []
        all_train_losses_top_tmp = []
        all_train_losses_side_tmp = []

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

            output_front = model_alexnet_front(img_tensor_front)
            output_top = model_alexnet_top(img_tensor_top)
            output_side = model_alexnet_side(img_tensor_side)

            _, pred_front = torch.max(output_front, 1)
            _, pred_top = torch.max(output_top, 1)
            _, pred_side = torch.max(output_side, 1)

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

            all_train_losses_front_tmp.append(loss_front.item())
            all_train_losses_top_tmp.append(loss_top.item())
            all_train_losses_side_tmp.append(loss_side.item())

        all_train_losses_front.append(np.mean(all_train_losses_front_tmp))
        all_train_losses_top.append(np.mean(all_train_losses_top_tmp))
        all_train_losses_side.append(np.mean(all_train_losses_side_tmp))

        if epoch % validate_every == 0 and epoch > 0:
            all_batch_training_accuracies_front.append(train_acc_front / batch_size)
            all_batch_training_accuracies_top.append(train_acc_top / batch_size)
            all_batch_training_accuracies_side.append(train_acc_side / batch_size)

            if epoch % print_every == 0:
                print('TRAIN_FRONT: %d %d%% (%s) %.4f [%d/%d -> %.2f%%]' % (
                    epoch, epoch / epoch_nb * 100, time_since(start_time), loss_front, train_acc_front, batch_size,
                    train_acc_front / batch_size * 100))
                print('TRAIN_TOP: %d %d%% (%s) %.4f [%d/%d -> %.2f%%]' % (
                    epoch, epoch / epoch_nb * 100, time_since(start_time), loss_top, train_acc_top, batch_size,
                    train_acc_top / batch_size * 100))
                print('TRAIN_SIDE: %d %d%% (%s) %.4f [%d/%d -> %.2f%%]' % (
                    epoch, epoch / epoch_nb * 100, time_since(start_time), loss_side, train_acc_side, batch_size,
                    train_acc_side / batch_size * 100))
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

                    val_output_front = model_alexnet_front(val_img_tensor_front)
                    val_output_top = model_alexnet_top(val_img_tensor_top)
                    val_output_side = model_alexnet_side(val_img_tensor_side)

                    _, val_pred_front = torch.max(val_output_front, 1)
                    _, val_pred_top = torch.max(val_output_top, 1)
                    _, val_pred_side = torch.max(val_output_side, 1)

                    if torch.argmax(val_output_front).item() == val_lbl:
                        val_acc_front += 1
                    if torch.argmax(val_output_top).item() == val_lbl:
                        val_acc_top += 1
                    if torch.argmax(val_pred_side).item() == val_lbl:
                        val_acc_side += 1

                    val_loss_front = criterion_front(val_output_front, val_lbl_tensor)
                    val_loss_top = criterion_top(val_output_top, val_lbl_tensor)
                    val_loss_side = criterion_side(val_output_side, val_lbl_tensor)

                    all_val_losses_front_tmp.append(val_loss_front.item())
                    all_val_losses_top_tmp.append(val_loss_top.item())
                    all_val_losses_side_tmp.append(val_loss_side.item())

                if epoch % print_every == 0:
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

    model_name = generate_model_name(method_name, epoch_nb, batch_size, learning_rate, optimizer_type.name)

    if save_diagram:
        save_diagram_common(all_train_losses_front, all_val_losses_front, model_name + '_front', validate_every, epoch_nb,
                            results_path, all_batch_training_accuracies_front, all_batch_val_accuracies_front)
        save_diagram_common(all_train_losses_top, all_val_losses_top, model_name + '_top', validate_every, epoch_nb, results_path,
                            all_batch_training_accuracies_top, all_batch_val_accuracies_top)
        save_diagram_common(all_train_losses_side, all_val_losses_side, model_name + '_side', validate_every, epoch_nb,
                            results_path, all_batch_training_accuracies_side, all_batch_val_accuracies_side)

    if save_model:
        save_model_common(model_alexnet_front, optimizer_front, epoch, validate_every, all_train_losses_front,
                          all_val_losses_front, save_model_for_inference, results_path, model_name + '_front')
        save_model_common(model_alexnet_top, optimizer_top, epoch, validate_every, all_train_losses_top,
                          all_val_losses_top, save_model_for_inference, results_path, model_name + '_top')
        save_model_common(model_alexnet_side, optimizer_side, epoch, validate_every, all_train_losses_side,
                          all_val_losses_side, save_model_for_inference, results_path, model_name + '_side')

    if save_loss:
        save_loss_common(all_train_losses_front, all_val_losses_front, model_name + '_front', results_path,
                         all_batch_training_accuracies_front, all_batch_val_accuracies_front)
        save_loss_common(all_train_losses_top, all_val_losses_top, model_name + '_top', results_path,
                         all_batch_training_accuracies_top, all_batch_val_accuracies_top)
        save_loss_common(all_train_losses_side, all_val_losses_side, model_name + '_side', results_path,
                         all_batch_training_accuracies_side, all_batch_val_accuracies_side)

    return model_alexnet_front, model_alexnet_top, model_alexnet_side
