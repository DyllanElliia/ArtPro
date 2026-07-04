import os
import json
import math

def paris2nerf(paris_cameras_path: str, nerf_transforms_path: str, resolution_x: int, set_type='train'):
    with open(paris_cameras_path, 'r') as json_file:
        cameras = json.load(json_file)

    f_x = cameras['K'][0][0]
    horizontal_fov = 2 * math.atan(resolution_x / (2 * f_x))
    transforms = {
        'camera_angle_x': horizontal_fov,
        'frames': [],
    }

    for img_name_no_ext, transform_mat in cameras.items():
        if img_name_no_ext == 'K':
            continue
        transforms['frames'].append(
            {
                'file_path': './' + set_type + '/' + img_name_no_ext,
                'rotation': 0.0,
                'transform_matrix': transform_mat
            }
        )

    with open(nerf_transforms_path, 'w') as outfile:
        json.dump(transforms, outfile, indent=4)

if __name__ == '__main__':
    res = 800
    input_path = 'data/storage45135/end/camera_train.json'
    output_path = 'data/storage45135/end/transforms_train.json'
    paris2nerf(input_path, output_path, res, set_type='train')

    input_path = 'data/storage45135/end/camera_test.json'
    output_path = 'data/storage45135/end/transforms_test.json'
    paris2nerf(input_path, output_path, res, set_type='test')
