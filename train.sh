python -m torch.distributed.launch --nproc_per_node=4 train_seq.py --config ./config/nuscenes_seq_final.yaml --ddp --gpu '0,1,2,3'