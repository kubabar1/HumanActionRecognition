import torch
import torchvision.transforms as transforms

from ..jtm.evaluate import ModelType
from ..jtm.utils.JTMDataset import JTMDataset, generate_sample_images
from ..lstm_simple.utils.LSTMSimpleDataset import LSTMSimpleDataset
from ...utils.dataset_util import SetType, DatasetInputType, GeometricFeature, normalise_skeleton_3d_batch, normalise_skeleton_3d
from ...utils.evaluation_utils import draw_confusion_matrix

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor()
])


def evaluate_tests(classes, test_data, test_labels, jtm_model_front, jtm_model_top, jtm_model_side, lstm_simple_model_rp,
                   lstm_simple_model_jjd, lstm_simple_model_jld, analysed_kpts_description, image_width, image_height,
                   result_path='results', show_diagram=True, input_type=DatasetInputType.SPLIT, split=20, steps=32, use_normalization=True):
    if use_normalization:
        test_data = normalise_skeleton_3d_batch(test_data, analysed_kpts_description['left_hip'], analysed_kpts_description['right_hip'])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    test_data_loader = JTMDataset(test_data, test_labels, image_width, image_height, len(test_data),
                                  SetType.TEST, analysed_kpts_description, is_test=True)

    test_data_loader_rp = LSTMSimpleDataset(test_data, test_labels, len(test_data), analysed_kpts_description, steps=steps,
                                            split=split, input_type=input_type, is_test=True, set_type=SetType.TEST,
                                            geometric_feature=GeometricFeature.RELATIVE_POSITION)
    test_data_loader_jjd = LSTMSimpleDataset(test_data, test_labels, len(test_data), analysed_kpts_description, steps=steps,
                                             split=split, input_type=input_type, is_test=True, set_type=SetType.TEST,
                                             geometric_feature=GeometricFeature.JOINT_JOINT_DISTANCE)
    test_data_loader_jld = LSTMSimpleDataset(test_data, test_labels, len(test_data), analysed_kpts_description, steps=steps,
                                             split=split, input_type=input_type, is_test=True, set_type=SetType.TEST,
                                             geometric_feature=GeometricFeature.JOINT_LINE_DISTANCE)

    test_x_jtm, test_y_jtm = next(iter(test_data_loader))
    test_x_lstm_rp, _ = next(iter(test_data_loader_rp))
    test_x_lstm_jjd, _ = next(iter(test_data_loader_jjd))
    test_x_lstm_jld, _ = next(iter(test_data_loader_jld))

    test_acc = 0
    pred = []

    for i, (test_img_jtm, test_img_lstm_rp, test_img_lstm_jjd, test_img_lstm_jld, test_lbl_jtm) in \
            enumerate(zip(test_x_jtm, test_x_lstm_rp, test_x_lstm_jjd, test_x_lstm_jld, test_y_jtm)):
        test_img_tensor_front = torch.unsqueeze(transform(test_img_jtm[ModelType.FRONT.value]), 0).to(device)
        test_img_tensor_top = torch.unsqueeze(transform(test_img_jtm[ModelType.TOP.value]), 0).to(device)
        test_img_tensor_side = torch.unsqueeze(transform(test_img_jtm[ModelType.SIDE.value]), 0).to(device)

        test_output_front = torch.softmax(jtm_model_front(test_img_tensor_front), 1)
        test_output_top = torch.softmax(jtm_model_top(test_img_tensor_top), 1)
        test_output_side = torch.softmax(jtm_model_side(test_img_tensor_side), 1)

        tensor_val_x_rp = torch.tensor(torch.unsqueeze(torch.tensor(test_img_lstm_rp), 0), dtype=torch.float, device=device)
        tensor_val_x_jjd = torch.tensor(torch.unsqueeze(torch.tensor(test_img_lstm_jjd), 0), dtype=torch.float, device=device)
        tensor_val_x_jld = torch.tensor(torch.unsqueeze(torch.tensor(test_img_lstm_jld), 0), dtype=torch.float, device=device)

        output_val_rp = torch.exp(lstm_simple_model_rp(tensor_val_x_rp)[0])
        output_val_jjd = torch.exp(lstm_simple_model_jjd(tensor_val_x_jjd)[0])
        output_val_jld = torch.exp(lstm_simple_model_jld(tensor_val_x_jld)[0])

        test_output = test_output_front * test_output_top * test_output_side * output_val_rp * output_val_jjd * output_val_jld

        pred.append(torch.argmax(test_output).item())

        if torch.argmax(test_output).item() == test_lbl_jtm:
            test_acc += 1

    correct_arr = [classes[int(i)] for i in test_y_jtm]
    predicted_arr = [classes[int(i)] for i in pred]

    draw_confusion_matrix(correct_arr, predicted_arr, classes, result_path=result_path, show_diagram=show_diagram)

    return test_acc / len(test_x_jtm)
