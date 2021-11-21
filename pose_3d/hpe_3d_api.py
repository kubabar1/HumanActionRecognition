import os
import sys


def load_model(video_pose_path, num_joints_in, num_joints_out=None, in_features=2):
    """
    Load VideoPose3D model pretrained on h36m dataset

    :param video_pose_path: path to VideoPose3D
    :param num_joints_in: number of input joints (e.g. 17 for Human3.6M)
    :param num_joints_out: number of output joints (can be different than input) - if not defined 'num_joints_out == num_joints_in'
    :param in_features: number of input features for each joint (typically 2 for 2D input)
    :return: VideoPose3D model pretrained on h36m dataset
    """
    sys.path.append(video_pose_path)
    from common.loss import torch
    from common.model import TemporalModel
    if num_joints_out is None:
        num_joints_out = num_joints_in
    architecture = '3,3,3,3,3'
    dropout = 0.25
    channels = 1024
    filter_widths = [int(x) for x in architecture.split(',')]
    model_pos = TemporalModel(num_joints_in, in_features, num_joints_out, filter_widths=filter_widths, dropout=dropout, channels=channels)
    chk_filename = os.path.join(video_pose_path, 'checkpoint', 'pretrained_h36m_detectron_coco.bin')
    checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
    model_pos.load_state_dict(checkpoint['model_pos'])
    return model_pos


def process_2d_to_3d(video_pose_path, keypoints, model_pos, joints_left, joints_right, frame_width, frame_height):
    """
    Process 2D keypoints to 3D coordinates

    :param video_pose_path: path to VideoPose3D
    :param keypoints: input keypoints used to generate 3D coordinates - shape (frames_count, joints_count, 2)
    :param model_pos: VideoPose3D model pretrained loaded using load_model function
    :param joints_left: array of left joints indexes (corresponding to input key points)
    :param joints_right: array of right joints indexes (corresponding to input key points)
    :param frame_width: width of input frame
    :param frame_height: height of input frame
    :return: generated predictions - shape (frames_count, joints_count, 3)
    """
    sys.path.append(video_pose_path)
    from common.camera import normalize_screen_coordinates
    from common.generators import UnchunkedGenerator
    from common.loss import torch
    keypoints = normalize_screen_coordinates(keypoints[..., :2], w=frame_width, h=frame_height)
    receptive_field = model_pos.receptive_field()
    pad = (receptive_field - 1) // 2

    if torch.cuda.is_available():
        model_pos = model_pos.cuda()

    gen = UnchunkedGenerator(None, None, [keypoints.copy()], pad=pad, kps_left=joints_left, kps_right=joints_right)
    prediction = evaluate(video_pose_path, gen, model_pos, None, joints_left, joints_right, return_predictions=True)

    return prediction


def evaluate(video_pose_path, test_generator, model_pos, model_traj, joints_left, joints_right, action=None, return_predictions=False,
             use_trajectory_model=False):
    sys.path.append(video_pose_path)
    from common.loss import torch, mpjpe, p_mpjpe, n_mpjpe, mean_velocity_error
    epoch_loss_3d_pos = 0
    epoch_loss_3d_pos_procrustes = 0
    epoch_loss_3d_pos_scale = 0
    epoch_loss_3d_vel = 0
    with torch.no_grad():
        if not use_trajectory_model:
            model_pos.eval()
        else:
            model_traj.eval()
        N = 0
        for _, batch, batch_2d in test_generator.next_epoch():
            inputs_2d = torch.from_numpy(batch_2d.astype('float32'))
            if torch.cuda.is_available():
                inputs_2d = inputs_2d.cuda()

            # Positional model
            if not use_trajectory_model:
                predicted_3d_pos = model_pos(inputs_2d)
            else:
                predicted_3d_pos = model_traj(inputs_2d)

            # Test-time augmentation (if enabled)
            if test_generator.augment_enabled():
                # Undo flipping and take average with non-flipped version
                predicted_3d_pos[1, :, :, 0] *= -1
                if not use_trajectory_model:
                    predicted_3d_pos[1, :, joints_left + joints_right] = predicted_3d_pos[1, :, joints_right + joints_left]
                predicted_3d_pos = torch.mean(predicted_3d_pos, dim=0, keepdim=True)

            if return_predictions:
                return predicted_3d_pos.squeeze(0).cpu().numpy()

            inputs_3d = torch.from_numpy(batch.astype('float32'))
            if torch.cuda.is_available():
                inputs_3d = inputs_3d.cuda()
            inputs_3d[:, :, 0] = 0
            if test_generator.augment_enabled():
                inputs_3d = inputs_3d[:1]

            error = mpjpe(predicted_3d_pos, inputs_3d)
            epoch_loss_3d_pos_scale += inputs_3d.shape[0] * inputs_3d.shape[1] * n_mpjpe(predicted_3d_pos, inputs_3d).item()

            epoch_loss_3d_pos += inputs_3d.shape[0] * inputs_3d.shape[1] * error.item()
            N += inputs_3d.shape[0] * inputs_3d.shape[1]

            inputs = inputs_3d.cpu().numpy().reshape(-1, inputs_3d.shape[-2], inputs_3d.shape[-1])
            predicted_3d_pos = predicted_3d_pos.cpu().numpy().reshape(-1, inputs_3d.shape[-2], inputs_3d.shape[-1])

            epoch_loss_3d_pos_procrustes += inputs_3d.shape[0] * inputs_3d.shape[1] * p_mpjpe(predicted_3d_pos, inputs)

            # Compute velocity error
            epoch_loss_3d_vel += inputs_3d.shape[0] * inputs_3d.shape[1] * mean_velocity_error(predicted_3d_pos, inputs)

    if action is None:
        print('----------')
    else:
        print('----' + action + '----')
    e1 = (epoch_loss_3d_pos / N) * 1000
    e2 = (epoch_loss_3d_pos_procrustes / N) * 1000
    e3 = (epoch_loss_3d_pos_scale / N) * 1000
    ev = (epoch_loss_3d_vel / N) * 1000
    print('Test time augmentation:', test_generator.augment_enabled())
    print('Protocol #1 Error (MPJPE):', e1, 'mm')
    print('Protocol #2 Error (P-MPJPE):', e2, 'mm')
    print('Protocol #3 Error (N-MPJPE):', e3, 'mm')
    print('Velocity Error (MPJVE):', ev, 'mm')
    print('----------')

    return e1, e2, e3, ev
