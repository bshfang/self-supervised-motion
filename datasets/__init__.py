from datasets.nuscenes_flow_sequence import NuScenes_sequence
# from datasets.waymo_flow_sequsence import Waymo_sequence

def build_dataset(dataset_cfg, train=True, merge=False, split='train'):

    dataset_name = dataset_cfg['train_dataset']
    if dataset_name == 'nuscenes':
        static_mask_path = None if 'static_mask_path' not in dataset_cfg else dataset_cfg['static_mask_path']
        rigid_piece_path = None if 'rigid_piece_path' not in dataset_cfg else dataset_cfg['rigid_piece_path']
        dataset = NuScenes_sequence(dataset_cfg=dataset_cfg,
                                    split=split,
                                    data_root=dataset_cfg['train_data_root'],
                                    static_mask_root=static_mask_path,
                                    rigid_piece_root=rigid_piece_path)

    return dataset
