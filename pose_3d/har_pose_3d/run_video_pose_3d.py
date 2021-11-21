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


def process_2d_to_3d(video_pose_path, keypoints, model_pos, joints_left, joints_right, frame_width, frame_height, kps_left=None,
                     kps_right=None):
    """
    Process 2D keypoints to 3D coordinates

    :param video_pose_path: path to VideoPose3D
    :param keypoints: input keypoints used to generate 3D coordinates - shape (frames_count, joints_count, 2)
    :param model_pos: VideoPose3D model pretrained loaded using load_model function
    :param joints_left: array of left joints indexes (corresponding to input key points)
    :param joints_right: array of right joints indexes (corresponding to input key points)
    :param frame_width: width of input frame
    :param frame_height: height of input frame
    :param kps_left: array of left joints indexes (corresponding to input key points) - if less points should be taken into account
    :param kps_right: array of right joints indexes (corresponding to input key points) - if less points should be taken into account
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

    kps_left = kps_left if kps_left is not None else joints_left
    kps_right = kps_right if kps_right is not None else joints_right

    gen = UnchunkedGenerator(None, None, [keypoints.copy()], pad=pad, kps_left=kps_left, kps_right=kps_right)
    prediction = evaluate(video_pose_path, gen, model_pos, None, joints_left, joints_right, return_predictions=True)

    return prediction


def process_custom_2d_to_3d_npz(keypoints_npz_path, video_pose_path, viz_subject, viz_action, viz_video=None,
                                viz_output='output.mp4', viz_export='output'):
    sys.path.append(video_pose_path)
    from common.camera import normalize_screen_coordinates, world_to_camera
    from common.custom_dataset import CustomDataset
    from common.generators import UnchunkedGenerator
    from common.loss import np, torch
    from common.model import TemporalModel, TemporalModelOptimized1f
    dataset = CustomDataset(keypoints_npz_path)
    subjects_test = [viz_subject]
    architecture = '3,3,3,3,3'
    dropout = 0.25
    channels = 1024
    viz_camera = 0
    dense = None
    causal = None
    filter_widths = [int(x) for x in architecture.split(',')]
    actions = '*'
    action_filter = None if actions == '*' else actions.split(',')
    model_params = 0

    for subject in dataset.subjects():
        for action in dataset[subject].keys():
            anim = dataset[subject][action]

            if 'positions' in anim:
                positions_3d = []
                for cam in anim['cameras']:
                    pos_3d = world_to_camera(anim['positions'], R=cam['orientation'], t=cam['translation'])
                    pos_3d[:, 1:] -= pos_3d[:, :1]  # Remove global offset, but keep trajectory in first position
                    positions_3d.append(pos_3d)
                anim['positions_3d'] = positions_3d

    keypoints = np.load(keypoints_npz_path, allow_pickle=True)
    keypoints_metadata = keypoints['metadata'].item()
    keypoints_symmetry = keypoints_metadata['keypoints_symmetry']
    kps_left, kps_right = list(keypoints_symmetry[0]), list(keypoints_symmetry[1])
    joints_left, joints_right = list(dataset.skeleton().joints_left()), list(dataset.skeleton().joints_right())
    keypoints = keypoints['positions_2d'].item()

    for subject in dataset.subjects():
        assert subject in keypoints, 'Subject {} is missing from the 2D detections dataset'.format(subject)
        for action in dataset[subject].keys():
            assert action in keypoints[subject], 'Action {} of subject {} is missing from the 2D detections dataset'.format(
                action, subject)
            if 'positions_3d' not in dataset[subject][action]:
                continue

            for cam_idx in range(len(keypoints[subject][action])):
                # We check for >= instead of == because some videos in H3.6M contain extra frames
                mocap_length = dataset[subject][action]['positions_3d'][cam_idx].shape[0]
                assert keypoints[subject][action][cam_idx].shape[0] >= mocap_length

                if keypoints[subject][action][cam_idx].shape[0] > mocap_length:
                    # Shorten sequence
                    keypoints[subject][action][cam_idx] = keypoints[subject][action][cam_idx][:mocap_length]

            assert len(keypoints[subject][action]) == len(dataset[subject][action]['positions_3d'])

    for subject in keypoints.keys():
        for action in keypoints[subject]:
            for cam_idx, kps in enumerate(keypoints[subject][action]):
                # Normalize camera frame
                cam = dataset.cameras()[subject][cam_idx]
                kps[..., :2] = normalize_screen_coordinates(kps[..., :2], w=cam['res_w'], h=cam['res_h'])
                keypoints[subject][action][cam_idx] = kps

    _, _, poses_valid_2d = fetch(video_pose_path, subjects_test, keypoints, dataset, action_filter)

    # Use optimized model for single-frame predictions
    model_pos_train = TemporalModelOptimized1f(poses_valid_2d[0].shape[-2], poses_valid_2d[0].shape[-1],
                                               dataset.skeleton().num_joints(), filter_widths=filter_widths, causal=causal,
                                               dropout=dropout, channels=channels)

    model_pos = TemporalModel(poses_valid_2d[0].shape[-2], poses_valid_2d[0].shape[-1], dataset.skeleton().num_joints(),
                              filter_widths=filter_widths, causal=causal, dropout=dropout, channels=channels, dense=dense)

    receptive_field = model_pos.receptive_field()
    pad = (receptive_field - 1) // 2

    if causal:
        causal_shift = pad
    else:
        causal_shift = 0

    for parameter in model_pos.parameters():
        model_params += parameter.numel()

    if torch.cuda.is_available():
        model_pos = model_pos.cuda()
        model_pos_train = model_pos_train.cuda()

    chk_filename = os.path.join(video_pose_path, 'checkpoint', 'pretrained_h36m_detectron_coco.bin')
    checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
    model_pos_train.load_state_dict(checkpoint['model_pos'])
    model_pos.load_state_dict(checkpoint['model_pos'])

    if 'model_traj' in checkpoint:
        model_traj = TemporalModel(poses_valid_2d[0].shape[-2], poses_valid_2d[0].shape[-1], 1,
                                   filter_widths=filter_widths, causal=causal, dropout=dropout, channels=channels, dense=dense)
        if torch.cuda.is_available():
            model_traj = model_traj.cuda()
        model_traj.load_state_dict(checkpoint['model_traj'])
    else:
        model_traj = None

    input_keypoints = keypoints[viz_subject][viz_action][viz_camera].copy()
    ground_truth = None
    if viz_subject in dataset.subjects() and viz_action in dataset[viz_subject]:
        if 'positions_3d' in dataset[viz_subject][viz_action]:
            ground_truth = dataset[viz_subject][viz_action]['positions_3d'][viz_camera].copy()

    test_time_augmentation = None
    gen = UnchunkedGenerator(None, None, [input_keypoints],
                             pad=pad, causal_shift=causal_shift, augment=test_time_augmentation,
                             kps_left=kps_left, kps_right=kps_right, joints_left=joints_left, joints_right=joints_right)
    prediction = evaluate(video_pose_path, gen, model_pos, model_traj, joints_left, joints_right, return_predictions=True)
    if model_traj is not None and ground_truth is None:
        prediction_traj = evaluate(video_pose_path, gen, model_pos, model_traj, joints_left, joints_right, return_predictions=True,
                                   use_trajectory_model=True)
        prediction += prediction_traj

    if viz_export is not None:
        np.save(viz_export, prediction)

    if viz_output is not None and viz_video is not None:
        render_video(video_pose_path, dataset, prediction, viz_subject, viz_camera, viz_video, viz_output, input_keypoints,
                     keypoints_metadata, ground_truth)

    dataset._skeleton._parents = [-1, 0, 1, 2, 3, 4, 0, 6, 7, 8, 9, 0, 11, 12, 13, 14, 12,
                                  16, 17, 18, 19, 20, 19, 22, 12, 24, 25, 26, 27, 28, 27, 30]


def render_video(video_pose_path, dataset, prediction, viz_subject, viz_camera, viz_video, viz_output, input_keypoints, keypoints_metadata,
                 ground_truth):
    sys.path.append(video_pose_path)
    from common.camera import camera_to_world, image_coordinates
    from common.loss import np
    viz_limit = -1
    viz_bitrate = 3000
    viz_downsample = 1
    viz_skip = 0
    viz_size = 5

    if ground_truth is not None:
        # Reapply trajectory
        trajectory = ground_truth[:, :1]
        ground_truth[:, 1:] += trajectory
        prediction += trajectory

    # Invert camera transformation
    cam = dataset.cameras()[viz_subject][viz_camera]
    if ground_truth is not None:
        prediction = camera_to_world(prediction, R=cam['orientation'], t=cam['translation'])
    else:
        # If the ground truth is not available, take the camera extrinsic params from a random subject.
        # They are almost the same, and anyway, we only need this for visualization purposes.
        for subject in dataset.cameras():
            if 'orientation' in dataset.cameras()[subject][viz_camera]:
                rot = dataset.cameras()[subject][viz_camera]['orientation']
                break
        prediction = camera_to_world(prediction, R=rot, t=0)
        # We don't have the trajectory, but at least we can rebase the height
        prediction[:, :, 2] -= np.min(prediction[:, :, 2])

    anim_output = {'Reconstruction': prediction}

    input_keypoints = image_coordinates(input_keypoints[..., :2], w=cam['res_w'], h=cam['res_h'])

    from common.visualization import render_animation

    render_animation(input_keypoints, keypoints_metadata, anim_output,
                     dataset.skeleton(), dataset.fps(), viz_bitrate, cam['azimuth'], viz_output,
                     limit=viz_limit, downsample=viz_downsample, size=viz_size,
                     input_video_path=viz_video, viewport=(cam['res_w'], cam['res_h']),
                     input_video_skip=viz_skip)


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


def fetch(video_pose_path, subjects, keypoints, dataset, action_filter=None, subset=1, parse_3d_poses=True, downsample=1):
    sys.path.append(video_pose_path)
    from common.utils import deterministic_random
    out_poses_3d = []
    out_poses_2d = []
    out_camera_params = []
    for subject in subjects:
        for action in keypoints[subject].keys():
            if action_filter is not None:
                found = False
                for a in action_filter:
                    if action.startswith(a):
                        found = True
                        break
                if not found:
                    continue

            poses_2d = keypoints[subject][action]
            for i in range(len(poses_2d)):  # Iterate across cameras
                out_poses_2d.append(poses_2d[i])

            if subject in dataset.cameras():
                cams = dataset.cameras()[subject]
                assert len(cams) == len(poses_2d), 'Camera count mismatch'
                for cam in cams:
                    if 'intrinsic' in cam:
                        out_camera_params.append(cam['intrinsic'])

            if parse_3d_poses and 'positions_3d' in dataset[subject][action]:
                poses_3d = dataset[subject][action]['positions_3d']
                assert len(poses_3d) == len(poses_2d), 'Camera count mismatch'
                for i in range(len(poses_3d)):  # Iterate across cameras
                    out_poses_3d.append(poses_3d[i])

    if len(out_camera_params) == 0:
        out_camera_params = None
    if len(out_poses_3d) == 0:
        out_poses_3d = None

    stride = downsample
    if subset < 1:
        for i in range(len(out_poses_2d)):
            n_frames = int(round(len(out_poses_2d[i]) // stride * subset) * stride)
            start = deterministic_random(0, len(out_poses_2d[i]) - n_frames + 1, str(len(out_poses_2d[i])))
            out_poses_2d[i] = out_poses_2d[i][start:start + n_frames:stride]
            if out_poses_3d is not None:
                out_poses_3d[i] = out_poses_3d[i][start:start + n_frames:stride]
    elif stride > 1:
        # Downsample as requested
        for i in range(len(out_poses_2d)):
            out_poses_2d[i] = out_poses_2d[i][::stride]
            if out_poses_3d is not None:
                out_poses_3d[i] = out_poses_3d[i][::stride]

    return out_camera_params, out_poses_3d, out_poses_2d
