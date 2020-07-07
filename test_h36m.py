from opts import Opts
from models.s2mpmr import S2MPMR
from datasets.dataloader_h36m import H36M14
from utils.eval import AverageMeter, compute_error3d, compute_error2d, reconstruction_error
from progress.bar import Bar
import torch.nn as nn
import torch
import ref
import os


def Test(protocol):
    # Options
    opt = Opts().parse()
    opt.dataset = 'fusion'
    if protocol == 'p1':
        opt.protocol = 'p1'
    elif protocol == 'p2':
        opt.protocol = 'p2'

    # Set gpu number
    gpus = [0]

    # Build models
    generator = S2MPMR(opt).cuda()
    generator = nn.DataParallel(generator, device_ids=gpus)


    file_name = 'final_model.pth'
    state = torch.load(file_name)
    generator.load_state_dict(state['generator'])

    loader_test = torch.utils.data.DataLoader(
        dataset=H36M14('val',opt),
        batch_size=1,
        shuffle=False,
        num_workers=ref.num_threads
    )

    generator.eval()

    # Initialize evaluations
    error2d, error3d, mrpe_x, mrpe_y, mrpe_z, mrpe, recon_err = \
        AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()

    num_iters = len(loader_test)
    bar = Bar('==>', max=num_iters)


    with torch.no_grad():
        for i, data_test in enumerate(loader_test):
            inp = data_test['img'].cuda()
            gt_j3d = data_test['pose3d'].cuda()
            gt_j2d = data_test['pose2d'].cuda()
            gt_root = data_test['root_img'].cuda()
            valid3d = data_test['valid3d'].cuda()
            valid2d = data_test['valid2d'].cuda()
            k_value = data_test['k_value'].cuda()

            cam_f = data_test['cam_f'][0].cuda()
            cam_c = data_test['cam_c'][0].cuda()
            gt_index = data_test['gt_index'][0].cuda()
            ori_shape = data_test['ori_shape'][0].cuda()

            nb = inp.shape[0]

            res_out = ref.res_out
            res_in = ref.res_in
            res_ratio = res_in / res_out

            # Forward propagation
            generator_output = generator(inp, k_value)
            (pred_detect, pred_thetas, pred_camera, pred_verts, pred_j3d, pred_j2d, pred_root,
             pred_bbox) = generator_output

            grid_x = int(gt_index[1].item() * res_ratio + res_ratio / 2)
            grid_y = int(gt_index[0].item() * res_ratio + res_ratio / 2)

            pred_root = pred_root[0, int(gt_index[0].item()), int(gt_index[1].item())].reshape(3)
            gt_root = gt_root[0, int(gt_index[0].item()), int(gt_index[1].item())].reshape(3)

            # convert to network input coordinate
            pred_root[0] += grid_x
            pred_root[1] += grid_y
            gt_root[0] += grid_x
            gt_root[1] += grid_y

            # convert to original image coordinate
            pred_root[0] = pred_root[0] * ori_shape[0] / res_in
            pred_root[1] = pred_root[1] * ori_shape[1] / res_in
            gt_root[0] = gt_root[0] * ori_shape[0] / res_in
            gt_root[1] = gt_root[1] * ori_shape[1] / res_in

            # convert to camera coordinate
            pred_root[0] = (pred_root[0] - cam_c[0]) / cam_f[0] * pred_root[2]
            pred_root[1] = (pred_root[1] - cam_c[1]) / cam_f[1] * pred_root[2]
            gt_root[0] = (gt_root[0] - cam_c[0]) / cam_f[0] * gt_root[2]
            gt_root[1] = (gt_root[1] - cam_c[1]) / cam_f[1] * gt_root[2]

            recon_err.update(
                reconstruction_error(pred_j3d.view(nb * ref.res_out * ref.res_out, ref.num_joints, 3).cpu().numpy(),
                                     gt_j3d.view(nb * ref.res_out * ref.res_out, ref.num_joints, 3).cpu().numpy(),
                                     reduction='sum'))
            mrpe_x.update(
                compute_error3d(pred_root[0].view(1, 1, 1), gt_root[0].view(1, 1, 1), torch.ones((1, 1)).cuda()))
            mrpe_y.update(
                compute_error3d(pred_root[1].view(1, 1, 1), gt_root[1].view(1, 1, 1), torch.ones((1, 1)).cuda()))
            mrpe_z.update(
                compute_error3d(pred_root[2].view(1, 1, 1), gt_root[2].view(1, 1, 1), torch.ones((1, 1)).cuda()))
            mrpe.update(compute_error3d(pred_root.view(1, 1, 3), gt_root.view(1, 1, 3), torch.ones((1, 1)).cuda()))

            error3d.update(compute_error3d(pred_j3d.detach().view(nb * ref.res_out * ref.res_out, ref.num_joints, 3),
                                           gt_j3d.view(nb * ref.res_out * ref.res_out, ref.num_joints, 3),
                                           valid3d.view(nb * ref.res_out * ref.res_out, ref.num_joints)))
            error2d.update(compute_error2d(pred_j2d.detach().view(nb * ref.res_out * ref.res_out, ref.num_joints, 2),
                                           gt_j2d.view(nb * ref.res_out * ref.res_out, ref.num_joints, 2),
                                           valid2d.view(nb * ref.res_out * ref.res_out, ref.num_joints)))

            Bar.suffix = '[{0}/{1}]| Tot: {total:} | ETA: {eta:} | MRPE_X {mrpe_x.avg:.2f} | MRPE_Y {mrpe_y.avg:.2f} | MRPE_Z {mrpe_z.avg:.2f} | MRPE {mrpe.avg:.2f} | Recon_error {recon_err.avg:.2f} | MPJPE {error3d.avg:.2f} | E2D {error2d.avg:.2f} '.format(
                i, num_iters, total=bar.elapsed_td, eta=bar.eta_td, mrpe_x=mrpe_x, mrpe_y=mrpe_y,
                mrpe_z=mrpe_z, mrpe=mrpe, recon_err=recon_err, error3d=error3d, error2d=error2d)

            bar.next()
        bar.finish()


if __name__ == "__main__":
    protocol = 'p1'  # p1 or p2
    Test(protocol)

