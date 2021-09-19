from enum import Enum
from random import randrange

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

from .utils.JTMDataset import JTMDataset, generate_sample_images
from ...utils.dataset_util import DatasetInputType, SetType
from ...utils.evaluation_utils import draw_confusion_matrix


class ModelType(Enum):
    FRONT = 0
    TOP = 1
    SIDE = 2


def evaluate_tests(classes, test_data, test_labels, model_path, analysed_kpts_description, image_width, image_height,
                   model_type: ModelType):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])

    jtm_model = models.AlexNet()
    jtm_model.classifier[6] = nn.Linear(4096, len(classes))
    jtm_model.load_state_dict(torch.load(model_path))
    jtm_model.eval()
    jtm_model.to(device)

    test_data_loader = JTMDataset(test_data, test_labels, image_width, image_height, len(test_data),
                                  SetType.TEST, analysed_kpts_description, is_test=True)

    test_x, test_y = next(iter(test_data_loader))

    test_acc = 0
    pred = []

    for i, (test_img, test_lbl) in enumerate(zip(test_x, test_y)):
        test_img_tensor = torch.unsqueeze(transform(test_img[0]), model_type.value).to(device)

        test_output = jtm_model(test_img_tensor)
        pred.append(torch.argmax(test_output).item())

        if torch.argmax(test_output).item() == test_lbl:
            test_acc += 1

    correct_arr = [classes[int(i)] for i in test_y]
    predicted_arr = [classes[int(i)] for i in pred]

    draw_confusion_matrix(correct_arr, predicted_arr, classes)

    return test_acc / len(test_x)


def fit(classes, data, model_path, analysed_kpts_description, image_width, image_height, model_type: ModelType):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])

    jtm_model = models.AlexNet()
    jtm_model.classifier[6] = nn.Linear(4096, len(classes))
    jtm_model.load_state_dict(torch.load(model_path))
    jtm_model.eval()
    jtm_model.to(device)

    smpl_img = generate_sample_images(data, analysed_kpts_description, image_width, image_height)[model_type.value]

    test_img_tensor = torch.unsqueeze(transform(smpl_img), model_type.value).to(device)
    test_output = jtm_model(test_img_tensor)

    return classes[torch.argmax(test_output).item()]
