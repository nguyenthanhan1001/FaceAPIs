DATASET_DIR=/home/nptai/aflw/data/flickr/
OUTPUT_DIR=./aflw
python tf_convert_data.py \
    --dataset_name=aflw \
    --dataset_dir=${DATASET_DIR} \
    --output_name=aflw_train \
    --output_dir=${OUTPUT_DIR}