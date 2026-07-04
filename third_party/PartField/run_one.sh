source_path=$1
gpu_id=$2
echo "Input path: ${source_path}"
echo "GPU ID: ${gpu_id}"

# mkdir -p ${source_path}/partfield_featuresfee1

CUDA_VISIBLE_DEVICES=${gpu_id} python partfield_inference.py \
  -c configs/final/demo.yaml \
  --opts \
    continue_ckpt model/model_objaverse.ckpt \
    result_name ../${source_path}/partfield_features \
    dataset.data_path ${source_path} \
    is_pc True

# sh run_one.sh ../datasets/paris/real_fridge/start 1
# sh run_one.sh ../datasets/paris/real_storage/start 1
# sh run_one.sh ../datasets/real/real_fridge/end 1
# sh run_one.sh ../datasets/real/real_storage/end 2

# sh run_one.sh ../datasets/artgs/oven_101908/start 1
# sh run_one.sh ../datasets/artgs/storage_45503/start 1
# sh run_one.sh ../datasets/artgs/storage_47648/start 1
# sh run_one.sh ../datasets/artgs/table_25493/start 1
# sh run_one.sh ../datasets/artgs/table_31249/start 1
# sh run_one.sh ../datasets/artgs/table_31249/end 1

# sh run_one.sh ../datasets/ours/sutoreeji40417/start 1
# sh run_one.sh ../datasets/ours/sutoreeji45759/start 1
# sh run_one.sh ../datasets/ours/sutoreeji47585/start 1
# sh run_one.sh ../datasets/ours/teeburu23372/start 1
# sh run_one.sh ../datasets/ours/teeburu33116/start 1
# sh run_one.sh ../datasets/ours/teeburu34178/start 1
# sh run_one.sh ../datasets/ours/teeburu34610/start 1
# sh run_one.sh ../datasets/ours/uindou103238/start 1

# sh run_one.sh ../datasets/ours/teeburu34610_v2/start 1
# sh run_one.sh ../datasets/ours/sutoreeji47585_v2/start 1

# sh run_one.sh ../datasets/real_irc/Storage/start 0

# sh run_one.sh ../datasets/paris/blade_103706/start 2
# sh run_one.sh ../datasets/paris/foldchair_102255/start 2
# sh run_one.sh ../datasets/paris/fridge_10905/start 2
# sh run_one.sh ../datasets/paris/laptop_10211/start 2
# sh run_one.sh ../datasets/paris/oven_101917/start 2
# sh run_one.sh ../datasets/paris/scissor_11100/start 2
# sh run_one.sh ../datasets/paris/stapler_103111/start 2
# sh run_one.sh ../datasets/paris/storage_45135/start 2
# sh run_one.sh ../datasets/paris/USB_100109/start 2
# sh run_one.sh ../datasets/paris/washer_103776/start 2
