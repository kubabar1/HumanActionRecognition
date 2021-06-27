import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from impl.jdm import get_mini_batch
from impl.utils import classes


def evaluate(dataset_path, loss_print_step=128, samples_count=256, cycles=100):
    # Load AlexNet
    model_alexnet_xy = models.alexnet(pretrained=True)  # torch.hub.load('pytorch/vision:v0.6.0', 'alexnet', pretrained=True)
    model_alexnet_xz = models.alexnet(pretrained=True)
    model_alexnet_yz = models.alexnet(pretrained=True)
    model_alexnet_xyz = models.alexnet(pretrained=True)

    # Define transform
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Model description
    print(model_alexnet_xy.eval())

    # Updating the third and the last classifier that is the output layer of the network.
    model_alexnet_xy.classifier[6] = nn.Linear(4096, len(classes))
    model_alexnet_xz.classifier[6] = nn.Linear(4096, len(classes))
    model_alexnet_yz.classifier[6] = nn.Linear(4096, len(classes))
    model_alexnet_xyz.classifier[6] = nn.Linear(4096, len(classes))

    # Change dropout to 0.9
    # dropout_p=0.9
    # model_alexnet_front.classifier[0] = nn.Dropout(p=dropout_p)
    # model_alexnet_front.classifier[3] = nn.Dropout(p=dropout_p)

    # model_alexnet_top.classifier[0] = nn.Dropout(p=dropout_p)
    # model_alexnet_top.classifier[3] = nn.Dropout(p=dropout_p)

    # model_alexnet_side.classifier[0] = nn.Dropout(p=dropout_p)
    # model_alexnet_side.classifier[3] = nn.Dropout(p=dropout_p)

    print(model_alexnet_xy.eval())

    # Instantiating CUDA device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Verifying CUDA
    print(device)

    # Move the input and AlexNet_model to GPU
    model_alexnet_xy.to(device)
    model_alexnet_xz.to(device)
    model_alexnet_yz.to(device)
    model_alexnet_xyz.to(device)

    # Loss
    criterion_xy = nn.CrossEntropyLoss()
    criterion_xz = nn.CrossEntropyLoss()
    criterion_yz = nn.CrossEntropyLoss()
    criterion_xyz = nn.CrossEntropyLoss()

    # Optimizer(SGD)
    lr = 0.001
    momentum = 0.9
    weight_decay = 0.0005
    optimizer_xy = optim.SGD(model_alexnet_xy.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    optimizer_xz = optim.SGD(model_alexnet_xz.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    optimizer_yz = optim.SGD(model_alexnet_yz.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    optimizer_xyz = optim.SGD(model_alexnet_xyz.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

    # Scheduler to change learning rate depends on epoch number
    step_size = 400
    gamma = 0.1
    scheduler_xy = optim.lr_scheduler.StepLR(optimizer_xy, step_size=step_size, gamma=gamma)
    scheduler_xz = optim.lr_scheduler.StepLR(optimizer_xz, step_size=step_size, gamma=gamma)
    scheduler_yz = optim.lr_scheduler.StepLR(optimizer_yz, step_size=step_size, gamma=gamma)
    scheduler_xyz = optim.lr_scheduler.StepLR(optimizer_xyz, step_size=step_size, gamma=gamma)

    print(transform)

    images, labels = get_mini_batch(dataset_path, classes, samples_count=samples_count, training=True)
    shuffled_array = np.array(range(len(labels)))

    for epoch in range(cycles):
        np.random.shuffle(shuffled_array)

        running_loss_xy = 0.0
        running_loss_xz = 0.0
        running_loss_yz = 0.0
        running_loss_xyz = 0.0

        shuffled_images = [images[idx] for idx in shuffled_array]
        shuffled_labels = [labels[idx] for idx in shuffled_array]

        for i, (img, lbl) in enumerate(zip(shuffled_images, shuffled_labels)):
            img_tensor_xy = torch.unsqueeze(transform(img[0]), 0).to(device)
            img_tensor_xz = torch.unsqueeze(transform(img[1]), 0).to(device)
            img_tensor_yz = torch.unsqueeze(transform(img[2]), 0).to(device)
            img_tensor_xyz = torch.unsqueeze(transform(img[3]), 0).to(device)
            lbl_tensor = torch.tensor([lbl]).to(device)

            # zero the parameter gradients
            optimizer_xy.zero_grad()
            optimizer_xz.zero_grad()
            optimizer_yz.zero_grad()
            optimizer_xyz.zero_grad()

            # forward + backward + optimize
            output_xy = model_alexnet_xy(img_tensor_xy)
            output_xz = model_alexnet_xz(img_tensor_xz)
            output_yz = model_alexnet_yz(img_tensor_yz)
            output_xyz = model_alexnet_xyz(img_tensor_xyz)

            loss_xy = criterion_xy(output_xy, lbl_tensor)
            loss_xz = criterion_xz(output_xz, lbl_tensor)
            loss_yz = criterion_yz(output_yz, lbl_tensor)
            loss_xyz = criterion_xyz(output_xyz, lbl_tensor)

            loss_xy.backward()
            loss_xz.backward()
            loss_yz.backward()
            loss_xyz.backward()

            optimizer_xy.step()
            optimizer_xz.step()
            optimizer_yz.step()
            optimizer_xyz.step()

            # print statistics
            running_loss_xy += loss_xy.item()
            running_loss_xz += loss_xz.item()
            running_loss_yz += loss_yz.item()
            running_loss_xyz += loss_xyz.item()

            if i % loss_print_step == (loss_print_step - 1):  # print every 'loss_print_step' mini-batches
                print('####################################################')
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss_xy / loss_print_step))
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss_xz / loss_print_step))
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss_yz / loss_print_step))
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss_xyz / loss_print_step))
                print('####################################################')

                # tmp = transform(img[0]).detach().numpy()
                # tmp = np.moveaxis(tmp * 255, 0, -1).astype(np.uint8)
                # plt.imshow(transforms.functional.to_pil_image(tmp, mode="RGB"))
                # plt.show()
                # plt.imshow(transforms.functional.to_pil_image(np.array(img[0]), "RGB"))
                # plt.show()

                running_loss_xy = 0.0
                running_loss_xz = 0.0
                running_loss_yz = 0.0
                running_loss_xyz = 0.0

        scheduler_xy.step()
        scheduler_xz.step()
        scheduler_yz.step()
        scheduler_xyz.step()

    return model_alexnet_xy, model_alexnet_xz, model_alexnet_yz, model_alexnet_xyz
