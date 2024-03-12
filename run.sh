# conda activate med_sam
export CUDA_VISIBLE_DEVICES="1,2"

python train_3dsam.py --data lung58 \
--snapshot_path logs --rand_crop_size 512 \
--max_epoch 100 --split_model 1