import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from .jtm import get_mini_batch
from .utils import image_height, image_width, classes


def evaluate(dataset_path, loss_print_step=128, samples_count=256, cycles=100):
    # Load AlexNet
    model_alexnet_front = models.alexnet(pretrained=True)  # torch.hub.load('pytorch/vision:v0.6.0', 'alexnet', pretrained=True)
    model_alexnet_top = models.alexnet(pretrained=True)
    model_alexnet_side = models.alexnet(pretrained=True)

    # Define transform
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Model description
    print(model_alexnet_front.eval())

    # Updating the third and the last classifier that is the output layer of the network.
    model_alexnet_front.classifier[6] = nn.Linear(4096, len(classes))
    model_alexnet_top.classifier[6] = nn.Linear(4096, len(classes))
    model_alexnet_side.classifier[6] = nn.Linear(4096, len(classes))

    # Change dropout to 0.9
    # dropout_p=0.9
    # model_alexnet_front.classifier[0] = nn.Dropout(p=dropout_p)
    # model_alexnet_front.classifier[3] = nn.Dropout(p=dropout_p)

    # model_alexnet_top.classifier[0] = nn.Dropout(p=dropout_p)
    # model_alexnet_top.classifier[3] = nn.Dropout(p=dropout_p)

    # model_alexnet_side.classifier[0] = nn.Dropout(p=dropout_p)
    # model_alexnet_side.classifier[3] = nn.Dropout(p=dropout_p)

    print(model_alexnet_front.eval())

    # Instantiating CUDA device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Move the input and AlexNet_model to GPU
    model_alexnet_front.to(device)
    model_alexnet_top.to(device)
    model_alexnet_side.to(device)

    # Loss
    criterion_front = nn.CrossEntropyLoss()
    criterion_top = nn.CrossEntropyLoss()
    criterion_side = nn.CrossEntropyLoss()

    # Optimizer(SGD)
    lr = 0.001
    momentum = 0.9
    weight_decay = 0.0005
    optimizer_front = optim.SGD(model_alexnet_front.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    optimizer_top = optim.SGD(model_alexnet_top.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    optimizer_side = optim.SGD(model_alexnet_side.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

    # Scheduler to change learning rate depends on epoch number
    step_size = 30
    gamma = 0.1
    scheduler_front = optim.lr_scheduler.StepLR(optimizer_front, step_size=step_size, gamma=gamma)
    scheduler_top = optim.lr_scheduler.StepLR(optimizer_top, step_size=step_size, gamma=gamma)
    scheduler_side = optim.lr_scheduler.StepLR(optimizer_side, step_size=step_size, gamma=gamma)

    images, labels = get_mini_batch(dataset_path, classes, image_width, image_height, samples_count=samples_count, training=True)
    shuffled_array = np.array(range(len(labels)))

    for epoch in range(cycles):
        np.random.shuffle(shuffled_array)

        running_loss_front = 0.0
        running_loss_top = 0.0
        running_loss_side = 0.0

        shuffled_images = [images[idx] for idx in shuffled_array]
        shuffled_labels = [labels[idx] for idx in shuffled_array]

        for i, (img, lbl) in enumerate(zip(shuffled_images, shuffled_labels)):
            img_tensor_front = torch.unsqueeze(transform(img[0]), 0).to(device)
            img_tensor_top = torch.unsqueeze(transform(img[1]), 0).to(device)
            img_tensor_side = torch.unsqueeze(transform(img[2]), 0).to(device)
            lbl_tensor = torch.tensor([lbl]).to(device)

            # zero the parameter gradients
            optimizer_front.zero_grad()
            optimizer_top.zero_grad()
            optimizer_side.zero_grad()

            # forward + backward + optimize
            output_front = model_alexnet_front(img_tensor_front)
            output_top = model_alexnet_top(img_tensor_top)
            output_side = model_alexnet_side(img_tensor_side)

            loss_front = criterion_front(output_front, lbl_tensor)
            loss_top = criterion_top(output_top, lbl_tensor)
            loss_side = criterion_side(output_side, lbl_tensor)

            loss_front.backward()
            loss_top.backward()
            loss_side.backward()

            optimizer_front.step()
            optimizer_top.step()
            optimizer_side.step()

            # print statistics
            running_loss_front += loss_front.item()
            running_loss_top += loss_top.item()
            running_loss_side += loss_side.item()

            if i % loss_print_step == (loss_print_step - 1):  # print every 'loss_print_step' mini-batches
                print('####################################################')
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss_front / loss_print_step))
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss_top / loss_print_step))
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss_side / loss_print_step))
                print('####################################################')
                # plt.imshow(img[0])
                # plt.show()
                # plt.imshow(img[1])
                # plt.show()
                # plt.imshow(img[2])
                # plt.show()
                # print(classes[lbl])
                running_loss_front = 0.0
                running_loss_top = 0.0
                running_loss_side = 0.0

        scheduler_front.step()
        scheduler_top.step()
        scheduler_side.step()

    return model_alexnet_front, model_alexnet_top, model_alexnet_side
