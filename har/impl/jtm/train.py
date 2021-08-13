import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms

from .utils.JTMDataset import JTMDataset
from ...utils.training_utils import save_model_common, save_diagram_common, generate_model_name, print_train_results, \
    Optimizer, save_loss_common, test_model, time_since


def train(classes, training_data, training_labels, validation_data, validation_labels, image_width, image_height,
          analysed_kpts_left, analysed_kpts_right, epoch_nb=100, batch_size=64, action_repetitions=100, hidden_size=256,
          learning_rate=0.00001, print_every=2, weight_decay=0.0005, momentum=0.9, step_size=30, gamma=0.1, train_every=10,
          validate_every=2, save_loss=True, save_diagram=True, results_path='results', optimizer_type=Optimizer.RMSPROP,
          save_model=True, save_model_for_inference=False):
    method_name = 'jtm'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

    train_data_loader = JTMDataset(training_data, training_labels, image_width, image_height, analysed_kpts_left,
                                   analysed_kpts_right, action_repetitions, batch_size)

    validation_data_loader = JTMDataset(validation_data, validation_labels, image_width, image_height, analysed_kpts_left,
                                        analysed_kpts_right, action_repetitions, batch_size)

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

            all_train_losses_front.append(loss_front.item())
            all_train_losses_top.append(loss_top.item())
            all_train_losses_side.append(loss_side.item())

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
                data_valid, labels_valid = next(iter(validation_data_loader))

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

                    all_val_losses_front.append(val_loss_front.item())
                    all_val_losses_top.append(val_loss_top.item())
                    all_val_losses_side.append(val_loss_side.item())

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

        scheduler_front.step()
        scheduler_top.step()
        scheduler_side.step()

    #     all_train_losses.append(loss.item())
    #
    #     running_loss_front = 0.0
    #     running_loss_top = 0.0
    #     running_loss_side = 0.0
    #
    #     for i, (img, lbl) in enumerate(zip(shuffled_images, shuffled_labels)):
    #         img_tensor_front = torch.unsqueeze(transform(img[0]), 0).to(device)
    #         img_tensor_top = torch.unsqueeze(transform(img[1]), 0).to(device)
    #         img_tensor_side = torch.unsqueeze(transform(img[2]), 0).to(device)
    #         lbl_tensor = torch.tensor([lbl]).to(device)
    #
    #         # zero the parameter gradients
    #         optimizer_front.zero_grad()
    #         optimizer_top.zero_grad()
    #         optimizer_side.zero_grad()
    #
    #         # forward + backward + optimize
    #         output_front = model_alexnet_front(img_tensor_front)
    #         output_top = model_alexnet_top(img_tensor_top)
    #         output_side = model_alexnet_side(img_tensor_side)
    #
    #         loss_front = criterion_front(output_front, lbl_tensor)
    #         loss_top = criterion_top(output_top, lbl_tensor)
    #         loss_side = criterion_side(output_side, lbl_tensor)
    #
    #         loss_front.backward()
    #         loss_top.backward()
    #         loss_side.backward()
    #
    #         optimizer_front.step()
    #         optimizer_top.step()
    #         optimizer_side.step()
    #
    #         # print statistics
    #         running_loss_front += loss_front.item()
    #         running_loss_top += loss_top.item()
    #         running_loss_side += loss_side.item()
    #
    #         if epoch % test_every == 0 and epoch > 0:
    #             print('####################################################')
    #             print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss_front / loss_print_step))
    #             print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss_top / loss_print_step))
    #             print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss_side / loss_print_step))
    #             print('####################################################')
    #             # plt.imshow(img[0])
    #             # plt.show()
    #             # plt.imshow(img[1])
    #             # plt.show()
    #             # plt.imshow(img[2])
    #             # plt.show()
    #             # print(classes[lbl])
    #             running_loss_front = 0.0
    #             running_loss_top = 0.0
    #             running_loss_side = 0.0
    #
    #     scheduler_front.step()
    #     scheduler_top.step()
    #     scheduler_side.step()
    #
    # model_name = generate_model_name(method_name, epoch_nb, batch_size, hidden_size, learning_rate, optimizer_type.name)
    #
    # if save_diagram:
    #     save_diagram_common(all_train_losses, all_test_losses, model_name, test_every, epoch_nb, results_path,
    #                         all_batch_training_accuracies, all_batch_test_accuracies)
    #
    # if save_model:
    #     save_model_common(lstm_model, optimizer, epoch, train_every, test_every, all_train_losses, all_test_losses,
    #                       save_model_for_inference, results_path, model_name)
    #
    # if save_loss:
    #     save_loss_common(all_train_losses, all_test_losses, model_name, results_path, all_batch_training_accuracies,
    #                      all_batch_test_accuracies)
    #
    # return model_alexnet_front, model_alexnet_top, model_alexnet_side
    #
    # if epoch % test_every == 0 and epoch > 0:
    #     train_accuracy = print_train_results(classes, output, tensor_train_y, epoch, epoch_nb, start_time, loss, batch_size,
    #                                          print_every)
    #     all_batch_training_accuracies.append(train_accuracy)
    #     with torch.no_grad():
    #         data_test, test_y = get_batch(dataset_path, batch_size, is_training=False)
    #         tensor_test_y = torch.from_numpy(test_y).to(device)
    #         tensor_test_x = torch.tensor(data_test.reshape((data_test.shape[0], data_test.shape[1], -1)), dtype=torch.float,
    #                                      device=device)
    #         output_test = lstm_model(tensor_test_x)
    #         loss_test = criterion(output_test, tensor_test_y)
    #         test_loss, batch_acc = test_model(tensor_test_y, output_test, classes, epoch, epoch_nb, print_every,
    #                                           start_time, batch_size, loss_test)
    #         all_test_losses.append(test_loss)
    #         all_batch_test_accuracies.append(batch_acc)
