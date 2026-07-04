import os
import json
import shutil

def save_dict_to_json(data: dict, file_path: str) -> None:
    with open(file_path, 'w') as outfile:
        json.dump(data, outfile, indent=4)

def load_json_to_dict(file_path: str) -> dict:
    with open(file_path, 'r') as json_file:
        info = json.load(json_file)
    return info

def gen_empty_dict(horizontal_fov: float) -> dict:
    return {'camera_angle_x': horizontal_fov, 'frames': []}

def safe_move(source: str, target: str):
    if os.path.exists(target) or not os.path.exists(source):
        return
    shutil.move(source, target)

def move_dir(source_dir: str, target_dir: str, rm=False) -> None:
    if os.path.exists(target_dir) or not os.path.exists(source_dir):
        return
    os.makedirs(target_dir, exist_ok=True)
    for item in os.listdir(source_dir):
        source_path = os.path.join(source_dir, item)
        destination_path = os.path.join(target_dir, item)
        safe_move(source_path, destination_path)
    if rm:
        os.rmdir(source_dir)

def convert_to_nerf(path: str) -> None:
    for state in ['start', 'end']:
        trans_source = os.path.join(path, f'transforms_train_{state}.json')
        trans_target = os.path.join(path, f'{state}/transforms_train.json')
        trans_target_test = os.path.join(path, f'{state}/transforms_test.json')

        trans = load_json_to_dict(trans_source)
        horizontal_fov = trans['camera_angle_x']
        for frame in trans['frames']:
            frame['file_path'] = './train/' + os.path.basename(frame['file_path'])

        save_dict_to_json(trans, trans_target)
        save_dict_to_json(gen_empty_dict(horizontal_fov), trans_target_test)
        os.makedirs(os.path.join(path, f'{state}/train'), exist_ok=True)
        os.makedirs(os.path.join(path, f'{state}/train_d'), exist_ok=True)
        os.makedirs(os.path.join(path, f'{state}/test'), exist_ok=True)
        os.makedirs(os.path.join(path, f'{state}/test_d'), exist_ok=True)
        move_dir(os.path.join(path, f'{state}/train/depth'), os.path.join(path, f'{state}/train_d'), rm=True)
        move_dir(os.path.join(path, f'{state}/train/rgba'), os.path.join(path, f'{state}/train'), rm=True)
    
    

if __name__ == '__main__':
    # data_path = 'data/artgs/oven_101908'  # False
    # data_path = 'data/artgs/storage_45503'  # True
    # data_path = 'data/artgs/storage_47648'  # False
    # data_path = 'data/artgs/table_25493'  # False
    # data_path = 'data/artgs/table_31249'  # False
    # convert_to_nerf(data_path)

    pass
