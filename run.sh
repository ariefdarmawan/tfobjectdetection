#! /bin/bash

export model_name='faster_rcnn_1'

# convert xml to csv
python3 py/xml2csv.py --image_dir=images --output=data

# generate tfrecord
python3 py/generate_tfrecord.py --csv_input=data/train_labels.csv \
    --image_dir=images/train \
    --output_path=data/train.record \
    --label=data/label_map.csv

python3 py/generate_tfrecord.py --csv_input=data/test_labels.csv \
    --image_dir=images/test \
    --output_path=data/test.record \
    --label=data/label_map.csv

# generate config
python3 py/generate_config.py --template data/model/$model_name.config.template \
    --output data/model/$model_name.config \
    --label=data/label_map.csv \
    --map=data/label.pbtxt \
    --ckpt=data/$model_name/model.ckpt \
    --train=data/train.record \
    --test=data/test.record \
    --test_images_dir=images/test 

# delete folder
rm -fR data/training 
rm -fR data/graph

# train the model
#python3 py/model_main.py --logtostderr \
#    --model_dir=data/training/ \
#    --pipeline_config_path=data/model/$model_name.config
python3 py/train.py --logtostderr \
   --train_dir=data/training/ \
   --pipeline_config_path=data/model/$model_name.config

# generate interference
export model_name='faster_rcnn_1' 
python py/export_inference_graph.py \
    --input_type image_tensor \
    --pipeline_config_path data/model/$model_name.config \
    --trained_checkpoint_prefix data/training/model.ckpt-6926 \
    --output_directory data/graph
    