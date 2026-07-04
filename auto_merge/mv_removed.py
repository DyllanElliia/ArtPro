import os
from os.path import join as pjoin
import shutil
import sys
sys.path.append('.')
import numpy as np

from utils.general_utils import find_files_with_suffix, estimate_normals_o3d
from scene.dataset_readers import fetchPly, storePly

def run_dataset(dataset, models):
    for _, out_name, data_name, _ in models:
        src_dir = f'auto_merge/debug_3_{dataset}/{data_name}'
        tgt_dir = f'output/{dataset}/{out_name}/clustering/clusters'
        os.makedirs(tgt_dir, exist_ok=True)

        for idx, filename in enumerate(sorted(find_files_with_suffix(src_dir, '.ply'))):
            pcd = fetchPly(pjoin(src_dir, filename))
            normals = estimate_normals_o3d(pcd.points)
            storePly(pjoin(tgt_dir, f'points3d_{idx}.ply'),
                     pcd.points, np.zeros_like(pcd.points), normals)
    print('Done with all models.')

if __name__ == '__main__':
    models_artgs = [
        # (3, 'oven', 'oven_101908', False),
        # (3, 'sto3', 'storage_45503', True),

        # (6, 'sto6', 'storage_47648', True),
        # (3, 'tbl3', 'table_25493', True),
        # (4, 'tbl4', 'table_31249', False),
    ]

    models_paris = [
        # (1, 'blade_103706', 'blade_103706', False),
        # (1, 'foldchair_102255', 'foldchair_102255', False),
        # (1, 'fridge_10905', 'fridge_10905', False),
        # (1, 'laptop_10211', 'laptop_10211', True),
        # (1, 'oven_101917', 'oven_101917', False),
        # (1, 'stapler_103111', 'stapler_103111', True),
        # (1, 'storage_45135', 'storage_45135', True),
        # (1, 'USB_100109', 'USB_100109', False),
        # (1, 'washer_103776', 'washer_103776', True),

        # (1, 'scissor_11100','scissor_11100', True),
    ]

    models_ours = [
        # (2, 'mado', 'uindou103238', False),
        # (3, 'te3', 'teeburu33116', False),
        # (4, 'tbr4', 'teeburu34178', False),
        # (4, 'sto4', 'sutoreeji45759', False),
        # (4, 'tee', 'teeburu23372', False),
        # (6, 'sut', 'sutoreeji40417', False),
        # (5, 'tbr5', 'teeburu34610', False),

        (10, 'str', 'sutoreeji47585', False),
    ]

    run_dataset('artgs', models_artgs)
    run_dataset('paris', models_paris)
    run_dataset('ours', models_ours)

    pass
