
gpu_id=7

# ===== artpro =====
CUDA_VISIBLE_DEVICES=$gpu_id python train.py --source ./datasets/artpro/window103238 --output outputs/artpro/window103238 --gt_movable_part 2
CUDA_VISIBLE_DEVICES=$gpu_id python train.py --source ./datasets/artpro/table33116 --output outputs/artpro/table33116 --gt_movable_part 3 --segment_level fine
CUDA_VISIBLE_DEVICES=$gpu_id python train.py --source ./datasets/artpro/table34178 --output outputs/artpro/table34178 --gt_movable_part 4 --segment_level fine
CUDA_VISIBLE_DEVICES=$gpu_id python train.py --source ./datasets/artpro/storage45759 --output outputs/artpro/storage45759 --gt_movable_part 4
CUDA_VISIBLE_DEVICES=$gpu_id python train.py --source ./datasets/artpro/table23372 --output outputs/artpro/table23372 --gt_movable_part 4
CUDA_VISIBLE_DEVICES=$gpu_id python train.py --source ./datasets/artpro/storage40417 --output outputs/artpro/storage40417 --gt_movable_part 6 --segment_level fine
CUDA_VISIBLE_DEVICES=$gpu_id python train.py --source ./datasets/artpro/table34610 --output outputs/artpro/table34610 --gt_movable_part 5 --segment_level fine --rel_tol 0.04 --tau_merge 1e-3
CUDA_VISIBLE_DEVICES=$gpu_id python train.py --source ./datasets/artpro/storage47585 --output outputs/artpro/storage47585 --gt_movable_part 10 --segment_level medium --rel_tol 0.01 --tau_merge 1e-4

# ===== artgs =====
CUDA_VISIBLE_DEVICES=$gpu_id python train.py --source ./datasets/artgs/oven_101908 --output outputs/artgs/oven_101908 --gt_movable_part 3
CUDA_VISIBLE_DEVICES=$gpu_id python train.py --source ./datasets/artgs/storage_45503 --output outputs/artgs/storage_45503 --gt_movable_part 3 --reverse
CUDA_VISIBLE_DEVICES=$gpu_id python train.py --source ./datasets/artgs/storage_47648 --output outputs/artgs/storage_47648 --gt_movable_part 6 --reverse
CUDA_VISIBLE_DEVICES=$gpu_id python train.py --source ./datasets/artgs/table_25493 --output outputs/artgs/table_25493 --gt_movable_part 3 --reverse
CUDA_VISIBLE_DEVICES=$gpu_id python train.py --source ./datasets/artgs/table_31249 --output outputs/artgs/table_31249 --gt_movable_part 4

# ===== real (note: out and single both live under the paris dir) =====
CUDA_VISIBLE_DEVICES=$gpu_id python train.py --source ./datasets/paris/real_fridge --output outputs/paris/real_fridge --gt_movable_part 1
CUDA_VISIBLE_DEVICES=$gpu_id python train.py --source ./datasets/paris/real_storage --output outputs/paris/real_storage --gt_movable_part 1

# ===== paris (gt_movable_part=1) =====
CUDA_VISIBLE_DEVICES=$gpu_id python train.py --source ./datasets/paris/blade_103706 --output outputs/paris/blade_103706 --gt_movable_part 1
CUDA_VISIBLE_DEVICES=$gpu_id python train.py --source ./datasets/paris/foldchair_102255 --output outputs/paris/foldchair_102255 --gt_movable_part 1
CUDA_VISIBLE_DEVICES=$gpu_id python train.py --source ./datasets/paris/fridge_10905 --output outputs/paris/fridge_10905 --gt_movable_part 1 --segment_level coarse
CUDA_VISIBLE_DEVICES=$gpu_id python train.py --source ./datasets/paris/laptop_10211 --output outputs/paris/laptop_10211 --gt_movable_part 1 --reverse
CUDA_VISIBLE_DEVICES=$gpu_id python train.py --source ./datasets/paris/oven_101917 --output outputs/paris/oven_101917 --gt_movable_part 1
CUDA_VISIBLE_DEVICES=$gpu_id python train.py --source ./datasets/paris/stapler_103111 --output outputs/paris/stapler_103111 --gt_movable_part 1 --reverse
CUDA_VISIBLE_DEVICES=$gpu_id python train.py --source ./datasets/paris/storage_45135 --output outputs/paris/storage_45135 --gt_movable_part 1 --reverse --segment_level coarse
CUDA_VISIBLE_DEVICES=$gpu_id python train.py --source ./datasets/paris/USB_100109 --output outputs/paris/USB_100109 --gt_movable_part 1
CUDA_VISIBLE_DEVICES=$gpu_id python train.py --source ./datasets/paris/washer_103776 --output outputs/paris/washer_103776 --gt_movable_part 1 --reverse
CUDA_VISIBLE_DEVICES=$gpu_id python train.py --source ./datasets/paris/scissor_11100 --output outputs/paris/scissor_11100 --gt_movable_part 1 --reverse --segment_level coarse
