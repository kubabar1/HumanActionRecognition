import os
from configparser import ConfigParser

import httplib2

bottom_up_configs = lambda mmpose_path: {
    'higher_hrnet32_coco_512x512': [
        os.path.join(mmpose_path, 'configs/bottom_up/higherhrnet/coco/higher_hrnet32_coco_512x512.py'),
        'https://download.openmmlab.com/mmpose/bottom_up/higher_hrnet32_coco_512x512-8ae85183_20200713.pth'
    ],
    'higher_hrnet32_coco_640x640': [
        os.path.join(mmpose_path, 'configs/bottom_up/higherhrnet/coco/higher_hrnet32_coco_640x640.py'),
        'https://download.openmmlab.com/mmpose/bottom_up/higher_hrnet32_coco_640x640-a22fe938_20200712.pth'
    ],
    'higher_hrnet48_coco_512x512': [
        os.path.join(mmpose_path, 'configs/bottom_up/higherhrnet/coco/higher_hrnet48_coco_512x512.py'),
        'https://download.openmmlab.com/mmpose/bottom_up/higher_hrnet48_coco_512x512-60fedcbc_20200712.pth'
    ],
    'hrnet_w32_coco_512x512': [
        os.path.join(mmpose_path, 'configs/bottom_up/hrnet/coco/hrnet_w32_coco_512x512.py'),
        'https://download.openmmlab.com/mmpose/bottom_up/hrnet_w32_coco_512x512-bcb8c247_20200816.pth'
    ],
    'hrnet_w48_coco_512x512': [
        os.path.join(mmpose_path, 'configs/bottom_up/hrnet/coco/hrnet_w48_coco_512x512.py'),
        'https://download.openmmlab.com/mmpose/bottom_up/hrnet_w48_coco_512x512-cf72fcdf_20200816.pth'
    ],
    'mobilenetv2_coco_512x512': [
        os.path.join(mmpose_path, 'configs/bottom_up/mobilenet/coco/mobilenetv2_coco_512x512.py'),
        'https://download.openmmlab.com/mmpose/bottom_up/mobilenetv2_coco_512x512-4d96e309_20200816.pth'
    ],
    'res50_coco_512x512': [
        os.path.join(mmpose_path, 'configs/bottom_up/resnet/coco/res50_coco_512x512.py'),
        'https://download.openmmlab.com/mmpose/bottom_up/res50_coco_512x512-5521bead_20200816.pth'
    ],
    'res50_coco_640x640': [
        os.path.join(mmpose_path, 'configs/bottom_up/resnet/coco/res50_coco_640x640.py'),
        'https://download.openmmlab.com/mmpose/bottom_up/res50_coco_640x640-2046f9cb_20200822.pth'
    ],
    'res101_coco_512x512': [
        os.path.join(mmpose_path, 'configs/bottom_up/resnet/coco/res101_coco_512x512.py'),
        'https://download.openmmlab.com/mmpose/bottom_up/res101_coco_512x512-e0c95157_20200816.pth'
    ],
    'res152_coco_512x512': [
        os.path.join(mmpose_path, 'configs/bottom_up/resnet/coco/res152_coco_512x512.py'),
        'https://download.openmmlab.com/mmpose/bottom_up/res152_coco_512x512-364eb38d_20200822.pth'
    ],
    'higher_hrnet32_coco_512x512_udp': [
        os.path.join(mmpose_path, 'configs/bottom_up/udp/coco/higher_hrnet32_coco_512x512_udp.py'),
        'https://download.openmmlab.com/mmpose/bottom_up/higher_hrnet32_coco_512x512_udp-8cc64794_20210222.pth'
    ],
    'hrnet_w32_coco_512x512_udp': [
        os.path.join(mmpose_path, 'configs/bottom_up/udp/coco/hrnet_w32_coco_512x512_udp.py'),
        'https://download.openmmlab.com/mmpose/bottom_up/hrnet_w32_coco_512x512_udp-91663bf9_20210220.pth'
    ]
}


def validate_data():
    config = ConfigParser()
    config.read('../../config.ini')
    mmpose_path = config.get('main', 'MMPOSE_PATH')
    error_count = 0
    configs = bottom_up_configs(mmpose_path)
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
