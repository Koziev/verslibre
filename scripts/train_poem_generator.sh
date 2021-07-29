#! /bin/bash


CUDA_VISIBLE_DEVICES=0 \
PYTHON_PATH=/home/inkoziev/polygon/ru-gpts \
python /home/inkoziev/polygon/ru-gpts/pretrain_transformers.py \
    --output_dir=/home/inkoziev/polygon/text_generator/models/rugpt_poem_generator \
    --overwrite_output_dir \
    --model_type=gpt2 \
    --model_name_or_path='sberbank-ai/rugpt3small_based_on_gpt2' \
    --do_train \
    --line_by_line \
    --train_data_file='/home/inkoziev/polygon/text_generator/tmp/poem_generator_dataset.dat' \
    --block_size 150 \
    --per_gpu_train_batch_size 48 \
    --save_steps 1000000 \
    --num_train_epochs 2 \
