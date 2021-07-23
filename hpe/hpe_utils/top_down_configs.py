import os
from configparser import ConfigParser

import httplib2

top_down_configs = lambda mmpose_path: {
    'alexnet_coco_256x192': [
        os.path.join(mmpose_path, 'configs/top_down/alexnet/coco/alexnet_coco_256x192.py'),
        'https://download.openmmlab.com/mmpose/top_down/alexnet/alexnet_coco_256x192-a7b1fd15_20200727.pth'
    ],
    'cpm_coco_384x288': [
        os.path.join(mmpose_path, 'configs/top_down/cpm/coco/cpm_coco_384x288.py'),
        'https://download.openmmlab.com/mmpose/top_down/cpm/cpm_coco_384x288-80feb4bc_20200821.pth'
    ],
    'res152_coco_384x288_dark': [
        os.path.join(mmpose_path, 'configs/top_down/darkpose/coco/res152_coco_384x288_dark.py'),
        'https://download.openmmlab.com/mmpose/top_down/resnet/res152_coco_384x288_dark-d3b8ebd7_20210203.pth'
    ],
    'hrnet_w48_coco_384x288_dark': [
        os.path.join(mmpose_path, 'configs/top_down/darkpose/coco/hrnet_w48_coco_384x288_dark.py'),
        'https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_384x288_dark-e881a4b6_20210203.pth'
    ],
    'deeppose_res152_coco_256x192': [
        os.path.join(mmpose_path, 'configs/top_down/deeppose/coco/deeppose_res152_coco_256x192.py'),
        'https://download.openmmlab.com/mmpose/top_down/deeppose/deeppose_res152_coco_256x192-7df89a88_20210205.pth'
    ],
    'hourglass52_coco_384x384': [
        os.path.join(mmpose_path, 'configs/top_down/hourglass/coco/hourglass52_coco_384x384.py'),
        'https://download.openmmlab.com/mmpose/top_down/hourglass/hourglass52_coco_384x384-be91ba2b_20200812.pth'
    ],
    'hrnet_w48_coco_384x288': [
        os.path.join(mmpose_path, 'configs/top_down/hrnet/coco/hrnet_w48_coco_384x288.py'),
        'https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_384x288-314c8528_20200708.pth'
    ],
    'mobilenetv2_coco_384x288': [
        os.path.join(mmpose_path, 'configs/top_down/mobilenet_v2/coco/mobilenetv2_coco_384x288.py'),
        'https://download.openmmlab.com/mmpose/top_down/mobilenetv2/mobilenetv2_coco_384x288-26be4816_20200727.pth'
    ],
    '4xmspn50_coco_256x192': [
        os.path.join(mmpose_path, 'configs/top_down/mspn/coco/4xmspn50_coco_256x192.py'),
        'https://download.openmmlab.com/mmpose/top_down/mspn/4xmspn50_coco_256x192-7b837afb_20201123.pth'
    ],
    'res152_coco_384x288': [
        os.path.join(mmpose_path, 'configs/top_down/resnet/coco/res152_coco_384x288.py'),
        'https://download.openmmlab.com/mmpose/top_down/resnet/res152_coco_384x288-3860d4c9_20200709.pth'
    ],
    'resnetv1d152_coco_384x288': [
        os.path.join(mmpose_path, 'configs/top_down/resnetv1d/coco/resnetv1d152_coco_384x288.py'),
        'https://download.openmmlab.com/mmpose/top_down/resnetv1d/resnetv1d152_coco_384x288-626c622d_20200730.pth'
    ],
    'resnext152_coco_384x288': [
        os.path.join(mmpose_path, 'configs/top_down/resnext/coco/resnext152_coco_384x288.py'),
        'https://download.openmmlab.com/mmpose/top_down/resnext/resnext152_coco_384x288-806176df_20200727.pth'
    ],
    '3xrsn50_coco_256x192': [
        os.path.join(mmpose_path, 'configs/top_down/rsn/coco/3xrsn50_coco_256x192.py'),
        'https://download.openmmlab.com/mmpose/top_down/rsn/3xrsn50_coco_256x192-58f57a68_20201127.pth'
    ],
    'scnet101_coco_384x288': [
        os.path.join(mmpose_path, 'configs/top_down/scnet/coco/scnet101_coco_384x288.py'),
        'https://download.openmmlab.com/mmpose/top_down/scnet/scnet101_coco_384x288-0b6e631b_20200709.pth'
    ],
    'seresnet152_coco_384x288': [
        os.path.join(mmpose_path, 'configs/top_down/seresnet/coco/seresnet152_coco_384x288.py'),
        'https://download.openmmlab.com/mmpose/top_down/seresnet/seresnet152_coco_384x288-58b23ee8_20200727.pth'
    ],
    'shufflenetv1_coco_384x288': [
        os.path.join(mmpose_path, 'configs/top_down/shufflenet_v1/coco/shufflenetv1_coco_384x288.py'),
        'https://download.openmmlab.com/mmpose/top_down/shufflenetv1/shufflenetv1_coco_384x288-b2930b24_20200804.pth'
    ],
    'shufflenetv2_coco_384x288': [
        os.path.join(mmpose_path, 'configs/top_down/shufflenet_v2/coco/shufflenetv2_coco_384x288.py'),
        'https://download.openmmlab.com/mmpose/top_down/shufflenetv2/shufflenetv2_coco_384x288-fb38ac3a_20200921.pth'
    ]
}


def validate_data():
    config = ConfigParser()
    config.read('../../config.ini')
    mmpose_path = config.get('main', 'MMPOSE_PATH')
    error_count = 0
    configs = top_down_configs(mmpose_path)
    for model_name in configs:
        config_path = configs[model_name][0]
        checkpoint_url = configs[model_name][1]
        resp = httplib2.Http().request(checkpoint_url, 'HEAD')
        if not os.path.isfile(config_path):
            print('CONFIG ERROR -> ' + config_path)
            error_count += 1
        if int(resp[0]['status']) != 200:
            print('WEBSITE DOES NOT EXISTS -> ' + checkpoint_url)
            error_count += 1
    if not error_count:
        print('DATA VALIDATION FINISHED WITH SUCCESS')


if __name__ == '__main__':
    validate_data()
