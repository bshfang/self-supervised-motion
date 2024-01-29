import os
import os.path as osp
from tqdm import tqdm
import yaml
import time
import socket
import argparse
import numpy as np
import random

import torch
from torch import optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, DistributedSampler
from tensorboardX import SummaryWriter

from datasets import build_dataset
from losses import build_criterion
from models.flowpred import FlowPred

from utils.train_util import to_device
from utils.eval_util import AverageMeter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./config/nusc_self.yaml', type=str, help='Config files')
    parser.add_argument('--gpu', type=str, default='0', help='gpu')
    parser.add_argument('--ddp', action='store_true', default=False)
    parser.add_argument("--local_rank", type=int, default=3)
    args = parser.parse_args()


    # === device === 
    if args.ddp:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        device = torch.device('cuda', args.local_rank)
        world_size = torch.distributed.get_world_size()
        args.world_size = world_size
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # === config ===
    with open(args.config) as f:
        configs = yaml.load(f, Loader=yaml.FullLoader)

    # === Fix the random seed ===
    if configs['random_seed']:
        print('random seed: ', configs['random_seed'])
        seed = configs['random_seed']
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        print('No random seed')

    # === log ===
    if args.local_rank == 3:
        log_path = os.path.join('logs', configs['tag'])
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        save_dir = os.path.join(
            log_path,
            time.strftime('%d%B%Yat%H:%M:%S%Z') + '_' + socket.gethostname())
        writer = SummaryWriter(save_dir)
        with open(osp.join(save_dir, 'config.yaml'), mode='w') as f:
            yaml.dump(configs, f)
    else:
        save_dir, writer = None, None

    # === Dataset Setup ===
    train_dataset = build_dataset(configs['data'], split='train')
    val_dataset = build_dataset(configs['data'], split='val')
    if args.ddp:
        sampler_train = DistributedSampler(train_dataset)
        sampler_val = DistributedSampler(val_dataset, shuffle=False)

        train_loader = DataLoader(train_dataset,
                                batch_size=configs['batch_size'],
                                num_workers=configs['num_workers'],
                                collate_fn=train_dataset.collate_batch_train,
                                pin_memory=True,
                                sampler=sampler_train,
                                drop_last=True)
        val_loader = DataLoader(val_dataset,
                                batch_size=configs['batch_size'],
                                num_workers=configs['num_workers'],
                                collate_fn=val_dataset.collate_batch_train,
                                pin_memory=True,
                                sampler=sampler_val)
    else:
        train_loader = DataLoader(train_dataset,
                                batch_size=configs['batch_size'],
                                num_workers=configs['num_workers'],
                                collate_fn=train_dataset.collate_batch_train,
                                pin_memory=True,
                                shuffle=True)
        val_loader = DataLoader(val_dataset,
                                # batch_size=1,
                                batch_size=configs['batch_size'],
                                num_workers=configs['num_workers'],
                                collate_fn=val_dataset.collate_batch_train,
                                pin_memory=True,
                                shuffle=False)
    
    print('train loader:', len(train_loader), ', val loader:', len(val_loader))

    # === model setup ===
    model_config = configs['model']
    input_frame = configs['data']['past_frame']
    output_frame = configs['data']['future_frame']
    if configs['use_backward_loss']:
        model = FlowPred(model_config, input_frame, output_frame+1).cuda()
    else:
        model = FlowPred(model_config, input_frame, output_frame).cuda()
    if args.ddp:
        model = torch.nn.parallel.DistributedDataParallel(model, 
                                                          device_ids=[args.local_rank], 
                                                          output_device=args.local_rank, 
                                                          find_unused_parameters=True)

    # === loss setup ===
    loss_config = configs['loss']
    criterion = build_criterion(loss_config)

    # === optimizer & scheduler ===
    optimizer = optim.AdamW(model.parameters(), lr=configs['optimizer']['lr'],
                           weight_decay=configs['optimizer']['weight_decay'])
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, 
                                               milestones=configs['scheduler']['milestones'],
                                               gamma=configs['scheduler']['gamma'])

    # === load from checkpoint ===
    if configs['load_model'] != "":
        pass
        # load model

    # === Training ===
    epochs = configs['epochs']
    itrs = 0
    best_eval = 1e3
    for epoch in range(epochs):
        if args.local_rank == 3:
            print('epoch:', epoch)
        if args.ddp:
            train_loader.sampler.set_epoch(epoch)

        for i, batch_data in tqdm(enumerate(train_loader)):
            itrs += 1
            model.train()
            model.zero_grad()
            optimizer.zero_grad()

            batch_data = to_device(batch_data, device)
            output = model(batch_data)

            loss = criterion(batch_data, output)
            loss_all = sum(loss.values())

            if args.local_rank == 3:
                for key, value in loss.items():
                    writer.add_scalar('train_loss_'+key, value, global_step=itrs)
                writer.add_scalar('train_loss_all', loss_all, global_step=itrs)

            loss_all.backward()
            optimizer.step()

        current_lr = optimizer.param_groups[0]['lr']
        if args.local_rank == 3:
            writer.add_scalar('lr', current_lr, global_step=epoch)

        scheduler.step()
        
        # val
        if epoch % configs['eval_freq'] == 0:
            if args.local_rank == 3:
                print("Epoch {} validation...".format(epoch))
            eval_result = eval_flow_motionnet(args, model, val_loader, device, future_frame_num=output_frame)
            for s, s_value in eval_result.items():
                for key, value in s_value.items():
                    for v_i, v in enumerate(value):
                        if args.local_rank == 3:
                            writer.add_scalar('eval_rank_frame_{}_{}_{}'.format(v_i, s, key), v, global_step=epoch)
            eval_metrics = eval_result['fast']['mean'][-1]
            if args.local_rank == 3:
                if eval_metrics < best_eval:
                    best_eval = eval_metrics
                    if args.ddp:
                        torch.save(model.module.state_dict(),
                                os.path.join(save_dir, 'model_best.pth'))
                    else:
                        torch.save(model.state_dict(),
                                os.path.join(save_dir, 'model_best.pth'))

        if args.local_rank == 3 and epoch % configs['save_freq'] == 0 and epoch > 0:
            if args.ddp:
                torch.save(model.module.state_dict(),
                        os.path.join(save_dir,
                                        'model_epoch%d.pth' % (epoch)))
            else:
                torch.save(model.state_dict(),
                        os.path.join(save_dir,
                                        'model_epoch%d.pth' % (epoch)))


def eval_flow_motionnet(args, model, data_loader, device, future_frame_num=2):
    """
    this function is only used for predict 2 frames
    """
    # The speed intervals for grouping the cellss
    # speed_intervals = np.array([[0.0, 0.0], [0, 5.0], [5.0, 20.0]])  # unit: m/s
    # We do not consider > 20m/s, since objects in nuScenes appear inside city and rarely exhibit very high speed
    speed_intervals = np.array([[0.0, 0.0], [0, 5.0], [5.0, 20.0]])
    intervals_name = ['static', 'slow', 'fast']
    distance_intervals = speed_intervals

    cell_groups = list()  # grouping the cells with different speeds
    for i in range(distance_intervals.shape[0]):
        cell_statistics = list()

        for j in range(future_frame_num+1):
            # corresponds to each row, which records the MSE, median etc.
            cell_statistics.append([])
        cell_groups.append(cell_statistics)

    with torch.no_grad():
        for i, batch_data in tqdm(enumerate(data_loader)):
            model.eval()

            batch_data = to_device(batch_data, device)
            gt_flows = batch_data['gt_flows']
            gt_valid_maps = batch_data['gt_valid_maps']

            output = model(batch_data)
            pred_flows = output['pillar_flow_output']
            b, h, w, _ = pred_flows.shape
            pred_flows = pred_flows.view(b, h, w, -1, 2)
            pred_flows = pred_flows.permute(0, 3, 4, 1, 2).contiguous()  # b,5,2,h,w
            pred_flows = pred_flows.cpu().numpy()

            # === static mask (align with motionnet) ===
            last_pred = pred_flows[:, -1, :, :, :]
            last_pred_norm = np.linalg.norm(last_pred, ord=2, axis=1)  # b,h,w

            thd_mask = last_pred_norm <= 0.2  # b,h,w
            weight_map = np.ones_like(thd_mask, dtype=np.float32)
            weight_map[thd_mask] = 0.0
            weight_map = weight_map[:, np.newaxis, np.newaxis, ...]  # b,1,1,h,w
            pred_flows = pred_flows * weight_map
            # =======================

            # pre-processing
            all_disp_field_gt = gt_flows.cpu().numpy()
            all_disp_field_gt = np.transpose(all_disp_field_gt, (0, 1, 4, 2, 3))
            # print(all_disp_field_gt.shape)  # b,5,2,h,w
            all_disp_field_gt_norm = np.linalg.norm(all_disp_field_gt, ord=2, axis=2)  # b,5,h,w

            # compute flow prediction error
            upper_thresh = 0.2  # 0.2m/s
            upper_bound = 0.5 * upper_thresh
            static_cell_mask = all_disp_field_gt_norm <= upper_bound
            static_cell_mask = np.all(static_cell_mask, axis=1)  # along the temporal axis
            moving_cell_mask = np.logical_not(static_cell_mask)

            for j, d in enumerate(distance_intervals):
                for s in range(future_frame_num+1):
                    if s == 2:
                        curr_valid_pixel_map = gt_valid_maps[:, 1].cpu().numpy()
                    else:
                        curr_valid_pixel_map = gt_valid_maps[:, s].cpu().numpy()

                    if j == 0:  # corresponds to static cells
                        curr_mask = np.logical_and(curr_valid_pixel_map, static_cell_mask)
                    else:
                        # We use the displacement between keyframe and the last sample frame as metrics
                        last_gt_norm = all_disp_field_gt_norm[:, -1]
                        mask = np.logical_and(d[0] <= last_gt_norm, last_gt_norm < d[1])

                        curr_mask = np.logical_and(curr_valid_pixel_map, mask)
                        curr_mask = np.logical_and(curr_mask, moving_cell_mask)
                    border = 8
                    roi_mask = np.zeros_like(curr_mask, dtype=np.bool)
                    roi_mask[:, border:-border, border:-border] = True
                    curr_mask = np.logical_and(curr_mask, roi_mask)
                    cell_idx = np.where(curr_mask == True)

                    if s == 2:
                        pred = pred_flows[:, 0] * 2
                        gt = all_disp_field_gt[:, 1]
                    else:
                        pred = pred_flows[:, s]
                        gt = all_disp_field_gt[:, s]

                    norm_error = np.linalg.norm(gt - pred, ord=2, axis=1)
                    # static error
                    cell_groups[j][s].append(norm_error[cell_idx])

    dump_res = {}
    for i, d in enumerate(speed_intervals):
        group = cell_groups[i]
        if args.local_rank == 3:
            print("--------------------------------------------------------------")
            print("Rank {}; For cells within speed range [{}, {}]:".format(args.local_rank, d[0], d[1]))

        dump_error = []
        dump_error_quantile_50 = []

        for s in range(future_frame_num+1):
            row = group[s]
            errors = np.concatenate(row) if len(row) != 0 else row
            mean_error = np.average(errors)
            error_quantile_50 = np.quantile(errors, 0.5) if len(errors) != 0 else 0

            dump_error.append(mean_error)
            dump_error_quantile_50.append(error_quantile_50)
            if args.local_rank == 3:
                print("Rank {}; Frame {}:\nThe mean error is {}\nThe median is {}".format(args.local_rank, s, mean_error, error_quantile_50))
        dump_res[intervals_name[i]] = {'mean': dump_error, 'median': dump_error_quantile_50}

    return dump_res


if __name__=='__main__':
    main()