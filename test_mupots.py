from opts import Opts
from models.s2mpmr import S2MPMR
from progress.bar import Bar
from datasets.dataloader_mupots import MuPoTS
import torch.nn as nn
import torch
import ref
import numpy as np
import os.path as osp
import scipy.io as sio

inds = [13,-1,8,7,6,9,10,11,2,1,0,3,4,5,-2,-3,12]

def Test(coord):
    # Options
    opt = Opts().parse()
    opt.dataset = 'fusion2'
    if coord == 'cam':
        opt.coord = 'cam'
    elif coord == 'relative':
        opt.coord = 'relative'

    # Set gpu number
    gpus = [0]

    # Build models
    generator = S2MPMR(opt).cuda()
    generator = nn.DataParallel(generator, device_ids=gpus)

    file_name = 'final_model.pth'
    state = torch.load(file_name)
    generator.load_state_dict(state['generator'])

    loader_test = torch.utils.data.DataLoader(
        dataset=MuPoTS('test'),
        batch_size=1,
        shuffle=False,
        num_workers=ref.num_threads
    )

    generator.eval()

    num_iters = len(loader_test)
    bar = Bar('==>', max=num_iters)

    pred_2d_save = {}
    pred_3d_save = {}

    with torch.no_grad():
        for i, data_test in enumerate(loader_test):
            inp = data_test['img'].cuda()
            gt_root = data_test['root_img'].cuda()
            k_value = data_test['k_value'].cuda()

            cam_f = data_test['cam_f'][0].cuda()
            cam_c = data_test['cam_c'][0].cuda()
            gt_index = data_test['gt_index'][0].cuda()
            ori_shape = data_test['ori_shape'][0].cuda()

            img_name = data_test['img_name'][0]
            spine3d = data_test['spine3d'][0].cuda()
            spine2d = data_test['spine2d'][0].cuda()


            res_out = ref.res_out
            res_in = ref.res_in
            res_ratio = res_in / res_out

            # Forward propagation
            generator_output = generator(inp, k_value)
            (pred_detect, pred_thetas, pred_camera, pred_verts, pred_j3d, pred_j2d, pred_root,
             pred_bbox) = generator_output

            grid_x = int(gt_index[1].item() * res_ratio + res_ratio / 2)
            grid_y = int(gt_index[0].item() * res_ratio + res_ratio / 2)

            if gt_index[0].item() > 13 or gt_index[1].item() > 13 or gt_index[0].item() < 0 or gt_index[1].item() < 0:
                continue

            # camera coordinate of predicted root joint
            pred_root = pred_root[0, int(gt_index[0].item()), int(gt_index[1].item())].detach().view(3)
            pred_root[0] += grid_x
            pred_root[1] += grid_y
            pred_root[0] = pred_root[0] * ori_shape[0] / res_in
            if ori_shape[0] == ori_shape[1]:
                pred_root[1] = pred_root[1] * ori_shape[1] / res_in
            else:
                pred_root[1] = (pred_root[1] - 98.) * ori_shape[0] / res_in
            pred_root[0] = (pred_root[0] - cam_c[0]) / cam_f[0] * pred_root[2]
            pred_root[1] = (pred_root[1] - cam_c[1]) / cam_f[1] * pred_root[2]

            # camera coordinate of gt root joint
            gt_root = gt_root[0, int(gt_index[0].item()), int(gt_index[1].item())].detach().view(3)
            gt_root[0] += grid_x
            gt_root[1] += grid_y
            gt_root[0] = gt_root[0] * ori_shape[0] / res_in
            if ori_shape[0] == ori_shape[1]:
                gt_root[1] = gt_root[1] * ori_shape[1] / res_in
            else:
                gt_root[1] = (gt_root[1] - 98.) * ori_shape[0] / res_in
            gt_root[0] = (gt_root[0] - cam_c[0]) / cam_f[0] * gt_root[2]
            gt_root[1] = (gt_root[1] - cam_c[1]) / cam_f[1] * gt_root[2]

            pred3d = pred_j3d[0, int(gt_index[0].item()), int(gt_index[1].item())].detach().view(19, 3)

            if opt.coord == 'relative':
                pred3d_abs = pred3d + gt_root
            elif opt.coord == 'cam':
                pred3d_abs = pred3d + pred_root

            pred3d_cam = pred3d_abs.cpu().numpy().copy()
            pred2d = np.zeros((19, 3))

            pred2d[:, 0] = pred3d_cam[:, 0] * cam_f[0].item() / pred3d_cam[:, 2] + cam_c[0].item()
            pred2d[:, 1] = pred3d_cam[:, 1] * cam_f[1].item() / pred3d_cam[:, 2] + cam_c[1].item()
            pred2d[:, 2] = pred3d_cam[:, 2]

            pred_2d_kpt = np.zeros((17, 3))
            pred_3d_kpt = np.zeros((17, 3))

            for j in range(17):
                if inds[j] == -1:
                    pred_2d_kpt[j] = (pred2d[8] + pred2d[9]) * .5
                elif inds[j] == -2:
                    pred_2d_kpt[j] = (pred2d[2] + pred2d[3]) * .5
                elif inds[j] == -3:
                    pred_2d_kpt[j][:2] = spine2d.cpu().numpy().copy()
                    pred_2d_kpt[j][2] = spine3d[2].cpu().numpy().copy()
                else:
                    pred_2d_kpt[j] = pred2d[inds[j]]
            if img_name in pred_2d_save:
                pred_2d_save[img_name].append(pred_2d_kpt[:, :2])
            else:
                pred_2d_save[img_name] = [pred_2d_kpt[:, :2]]

            for j in range(17):
                if inds[j] == -1:
                    pred_3d_kpt[j] = (pred3d_cam[8] + pred3d_cam[9]) * .5
                elif inds[j] == -2:
                    pred_3d_kpt[j] = (pred3d_cam[2] + pred3d_cam[3]) * .5
                elif inds[j] == -3:
                    pred_3d_kpt[j] = spine3d.cpu().numpy().copy()
                else:
                    pred_3d_kpt[j] = pred3d_cam[inds[j]]
            if img_name in pred_3d_save:
                pred_3d_save[img_name].append(pred_3d_kpt)
            else:
                pred_3d_save[img_name] = [pred_3d_kpt]

            Bar.suffix = '[{0}/{1}]| Tot: {total:} | ETA: {eta:} '.format(
                i, num_iters, total=bar.elapsed_td, eta=bar.eta_td)

            bar.next()

        bar.finish()

        output_path = osp.join('./matlab', 'preds_2d_kpt_mupots.mat')
        sio.savemat(output_path, pred_2d_save)
        print("Testing result is saved at " + output_path)
        output_path = osp.join('./matlab', 'preds_3d_kpt_mupots.mat')
        sio.savemat(output_path, pred_3d_save)
        print("Testing result is saved at " + output_path)

if __name__ == "__main__":
    coord = 'cam' # cam or relative
    Test(coord)
