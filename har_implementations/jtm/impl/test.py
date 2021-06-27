import numpy as np
import torch
import torchvision.transforms as transforms

from .jtm import get_mini_batch
from .utils import image_height, image_width, classes


def test(model_alexnet_front, model_alexnet_top, model_alexnet_side, dataset_path, samples_count=256):
    correct = 0
    correct_front = 0
    correct_top = 0
    correct_side = 0
    total = 0

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        images, labels = get_mini_batch(dataset_path, classes, image_width, image_height, samples_count=samples_count,
                                        training=False)
        for i, (img, lbl) in enumerate(zip(images, labels)):
            img_tensor_front = torch.unsqueeze(transform(img[0]), 0).to(device)
            img_tensor_top = torch.unsqueeze(transform(img[1]), 0).to(device)
            img_tensor_side = torch.unsqueeze(transform(img[2]), 0).to(device)
            lbl_tensor = torch.tensor([lbl]).to(device)

            outputs1 = model_alexnet_front(img_tensor_front).cpu().detach().numpy()[0]
            outputs2 = model_alexnet_top(img_tensor_top).cpu().detach().numpy()[0]
            outputs3 = model_alexnet_side(img_tensor_side).cpu().detach().numpy()[0]

            outputs = np.multiply(np.multiply(outputs1, outputs2), outputs3)

            predicted = outputs.argmax()

            # print('##################################################')
            # print(np.argmax(outputs1))
            # print(np.argmax(outputs2))
            # print(np.argmax(outputs3))
            # print(lbl_tensor.item())
            # print('##################################################')

            # print(predicted)
            # print(np.argmax(outputs3))
            # print(lbl_tensor.item())
            # print(outputs[predicted])

            # _, predicted1 = torch.max(outputs1.data, 1)
            # _, predicted2 = torch.max(outputs2.data, 1)
            # _, predicted3 = torch.max(outputs3.data, 1)

            total += lbl_tensor.size(0)
            correct += 1 if predicted == lbl_tensor.item() else 0
            correct_front += 1 if np.argmax(outputs1) == lbl_tensor.item() else 0
            correct_top += 1 if np.argmax(outputs2) == lbl_tensor.item() else 0
            correct_side += 1 if np.argmax(outputs3) == lbl_tensor.item() else 0

    print('Accuracy of the network on the {} images front: {} %'.format(samples_count, (100 * correct_front / total)))
    print('Accuracy of the network on the {} images top: {} %'.format(samples_count, (100 * correct_top / total)))
    print('Accuracy of the network on the {} images side: {} %'.format(samples_count, (100 * correct_side / total)))
    print('Summary accuracy of the network on the {} images: {} %'.format(samples_count, (100 * correct / total)))

    # Testing classification accuracy for individual classes.
    class_correct = list(0. for i in range(len(classes)))
    class_total = list(0. for i in range(len(classes)))
    samples_count = 256
    with torch.no_grad():
        images, labels = get_mini_batch(dataset_path, classes, image_width, image_height, samples_count=samples_count,
                                        training=False)
        for i, (img, lbl) in enumerate(zip(images, labels)):
            img_tensor_front = torch.unsqueeze(transform(img[0]), 0).to(device)
            img_tensor_top = torch.unsqueeze(transform(img[1]), 0).to(device)
            img_tensor_side = torch.unsqueeze(transform(img[2]), 0).to(device)
            lbl_tensor = torch.tensor([lbl]).to(device)

            outputs1 = model_alexnet_front(img_tensor_front).cpu().detach().numpy()[0]
            outputs2 = model_alexnet_top(img_tensor_top).cpu().detach().numpy()[0]
            outputs3 = model_alexnet_side(img_tensor_side).cpu().detach().numpy()[0]

            outputs = np.multiply(np.multiply(outputs1, outputs2), outputs3)

            predicted = outputs.argmax()
            c = 1 if predicted == lbl_tensor.item() else 0
            label = lbl_tensor[0]
            class_correct[label] += c
            class_total[label] += 1

    for i in range(len(classes)):
        if class_total[i]:
            print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))

    avg = 0
    for i in range(10):
        temp = (100 * class_correct[i] / class_total[i])
        avg = avg + temp
    avg = avg / 10
    print('Average accuracy = ', avg)
