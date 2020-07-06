import os
import torch
from utils.eval import AverageMeter, compute_error3d, compute_error2d, reconstruction_error
from progress.bar import Bar
import ref
import numpy as np
import os.path as osp
import scipy.io as sio


inds = [13,-1,8,7,6,9,10,11,2,1,0,3,4,5,-2,-3,12]


def _weighted_l1_loss(prediction, target, weight=1.0):
    return torch.sum(weight*torch.abs(prediction-target))/prediction.shape[0]

def _weighted_l2_loss(prediction, target, weight=1.0):
    return torch.sum(weight*((prediction-target)**2.0))/prediction.shape[0]

def _adv_l2_loss(prediction, weight=1.0):
    return torch.sum(weight*((prediction-1.0)**2.0))/weight.sum()

def _disc_l2_loss(real, fake, weight_real=1.0, weight_fake=1.0):
    return torch.sum(weight_real*((real-1.0)**2.0))/real.shape[0] + torch.sum(weight_fake*(fake**2.0))/weight_fake.sum()

def step(split, epoch, opt, loader_joint, generator, g_optimizer=None, loader_mosh=None, discriminator=None, d_optimizer=None):
    # Training mode
    if split == 'train':
        generator.train()
        loader_mosh = iter(loader_mosh)
    else:
        generator.eval()

    # Initialize evaluations
    cost_detect, cost_3d, cost_2d, cost_adv, cost_disc , cost_root, cost_bbox = \
        AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    error2d, error3d,  mrpe_x, mrpe_y, mrpe_z, mrpe, recon_err = \
        AverageMeter(),AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()


    num_iters = len(loader_joint)
    bar = Bar('==>', max=num_iters)

    pred_2d_save = {}
    pred_3d_save = {}

    # For each mini-batch,
    for i, data_train in enumerate(loader_joint):
        inp = data_train['img'].cuda()
        gt_detect = data_train['detect'].cuda()
        gt_j3d = data_train['pose3d'].cuda()
        gt_j2d = data_train['pose2d'].cuda()
        gt_root = data_train['root_img'].cuda()
        gt_bbox = data_train['xywh'].cuda()
        valid3d = data_train['valid3d'].cuda()
        valid2d = data_train['valid2d'].cuda()
        validroot = data_train['validroot'].cuda()
        valid_bbox = data_train['valid_xywh'].cuda()
        k_value = data_train['k_value'].cuda()



        if split== 'test':
            cam_f = data_train['cam_f'][0].cuda()
            cam_c = data_train['cam_c'][0].cuda()
            gt_index = data_train['gt_index'][0].cuda()
            ori_shape = data_train['ori_shape'][0].cuda()
            if opt.dataset == 'fusion2':
                img_name = data_train['img_name'][0]
                spine3d = data_train['spine3d'][0].cuda()
                spine2d = data_train['spine2d'][0].cuda()


        nb = inp.shape[0]

        res_out = ref.res_out
        res_in = ref.res_in
        res_ratio = res_in / res_out


        # Forward propagation
        generator_output = generator(inp, k_value)
        (pred_detect, pred_thetas, pred_camera, pred_verts, pred_j3d, pred_j2d, pred_root, pred_bbox) = generator_output


        # Compute generator loss
        loss_root = _weighted_l1_loss(pred_root.view(nb,ref.res_out,ref.res_out,1,3),gt_root.view(nb,ref.res_out,ref.res_out,1,3),validroot.view(nb,ref.res_out,ref.res_out,1,3))
        loss_bbox = _weighted_l2_loss(pred_bbox.view(nb, ref.res_out, ref.res_out, 1, 4), gt_bbox.view(nb, ref.res_out, ref.res_out, 1, 4), valid_bbox.view(nb, ref.res_out, ref.res_out, 1, 4))
        loss_detect = _weighted_l2_loss(pred_detect, gt_detect)
        loss_joint_3d = _weighted_l1_loss(pred_j3d.view(nb, ref.res_out, ref.res_out, ref.num_joints, 3), gt_j3d.view(nb, ref.res_out, ref.res_out, ref.num_joints, 3), valid3d.view(nb, ref.res_out, ref.res_out, ref.num_joints, 1))
        loss_joint_2d = _weighted_l1_loss(pred_j2d.view(nb, ref.res_out, ref.res_out, ref.num_joints, 2), gt_j2d.view(nb, ref.res_out, ref.res_out, ref.num_joints, 2), valid2d.view(nb, ref.res_out, ref.res_out, ref.num_joints, 1))
        loss_adv = _adv_l2_loss(discriminator(pred_thetas.view(nb * ref.res_out * ref.res_out , -1)),gt_detect.view(nb * ref.res_out * ref.res_out, -1))
        g_loss = loss_bbox * opt.weight_bbox + loss_root * opt.weight_root + loss_detect * opt.weight_score + loss_joint_3d + loss_joint_2d*opt.weight_2d + loss_adv*opt.weight_adv

        # Compute discriminator loss
        if split == 'train':
            data_mosh = next(loader_mosh)
            real_thetas = data_mosh['theta'].cuda()
            fake_thetas = pred_thetas.view(nb * ref.res_out * ref.res_out , -1).detach()
            real_disc_val, fake_disc_val = discriminator(real_thetas), discriminator(fake_thetas)
            loss_disc = _disc_l2_loss(real_disc_val, fake_disc_val, 1.0, gt_detect.view(nb*ref.res_out*ref.res_out,-1))
            d_loss = loss_disc*opt.weight_disc

        # Update model parameters with backpropagation
        if split == 'train':
            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()
            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()

        # Update evaluations
        cost_bbox.update(loss_bbox.detach().item()* opt.weight_bbox, nb)
        cost_root.update(loss_root.detach().item()* opt.weight_root,nb)
        cost_detect.update(loss_detect.detach().item()* opt.weight_score, nb)
        cost_3d.update(loss_joint_3d.detach().item(), nb)
        cost_2d.update(loss_joint_2d.detach().item()*opt.weight_2d, nb)
        cost_adv.update(loss_adv.detach().item()*opt.weight_adv, nb)
        if split == 'train':
            cost_disc.update(loss_disc.detach().item() * opt.weight_disc, nb)




        if split == 'test' and opt.dataset == 'fusion':
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


            recon_err.update(reconstruction_error(pred_j3d.view(nb*ref.res_out*ref.res_out,ref.num_joints,3).cpu().numpy(),gt_j3d.view(nb*ref.res_out*ref.res_out, ref.num_joints, 3).cpu().numpy(),reduction='sum'))
            mrpe_x.update(compute_error3d(pred_root[0].view(1,1,1),gt_root[0].view(1,1,1),torch.ones((1,1)).cuda()))
            mrpe_y.update(compute_error3d(pred_root[1].view(1,1,1),gt_root[1].view(1,1,1),torch.ones((1,1)).cuda()))
            mrpe_z.update(compute_error3d(pred_root[2].view(1,1,1),gt_root[2].view(1,1,1),torch.ones((1,1)).cuda()))
            mrpe.update(compute_error3d(pred_root.view(1,1,3),gt_root.view(1,1,3),torch.ones((1,1)).cuda()))

        if split == 'test' and opt.dataset == 'fusion2':
            grid_x = int(gt_index[1].item() * res_ratio + res_ratio / 2)
            grid_y = int(gt_index[0].item() * res_ratio + res_ratio / 2)

            if gt_index[0].item() >13 or gt_index[1].item() >13 or gt_index[0].item() <0 or gt_index[1].item() <0:
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
                pred_2d_save[img_name].append(pred_2d_kpt[:,:2])
            else:
                pred_2d_save[img_name] = [pred_2d_kpt[:,:2]]


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





        error3d.update(compute_error3d(pred_j3d.detach().view(nb*ref.res_out*ref.res_out, ref.num_joints, 3), gt_j3d.view(nb*ref.res_out*ref.res_out, ref.num_joints, 3), valid3d.view(nb*ref.res_out*ref.res_out, ref.num_joints)))
        error2d.update(compute_error2d(pred_j2d.detach().view(nb*ref.res_out*ref.res_out, ref.num_joints, 2), gt_j2d.view(nb*ref.res_out*ref.res_out, ref.num_joints, 2), valid2d.view(nb*ref.res_out*ref.res_out, ref.num_joints)))

        if split == 'test':
            if opt.dataset == 'fusion':
                Bar.suffix = '{split} Epoch: [{0}][{1}/{2}]| Tot: {total:} | ETA: {eta:} | MRPE_X {mrpe_x.avg:.2f} | MRPE_Y {mrpe_y.avg:.2f} | MRPE_Z {mrpe_z.avg:.2f} | MRPE {mrpe.avg:.2f} | Recon_error {recon_err.avg:.2f} | MPJPE {error3d.avg:.2f} | E2D {error2d.avg:.2f} '.format(
                    epoch, i, num_iters, split=split, total=bar.elapsed_td, eta=bar.eta_td,mrpe_x = mrpe_x, mrpe_y = mrpe_y, mrpe_z = mrpe_z, mrpe=mrpe, recon_err=recon_err, error3d=error3d, error2d = error2d)
            elif opt.dataset == 'fusion2':
                Bar.suffix = '{split} Epoch: [{0}][{1}/{2}]| Tot: {total:} | ETA: {eta:} '.format(
                    epoch, i, num_iters, split=split, total=bar.elapsed_td, eta=bar.eta_td)
        else:
            Bar.suffix = '{split} Epoch: [{0}][{1}/{2}]| Tot: {total:} | ETA: {eta:} | CS {cost_detect.avg:.2f} | CB {cost_bbox.avg:.2f} | CR {cost_root.avg:.2f} C3D {cost_3d.avg:.2f} | C2D {cost_2d.avg:.2f} | C_adv {cost_adv.avg:.2f} | C_disc {cost_disc.avg:.2f} | E3D {error3d.avg:.2f} | E2D {error2d.avg:.2f}'.\
                format(epoch, i, num_iters, split=split, total=bar.elapsed_td, eta=bar.eta_td, cost_bbox=cost_bbox, cost_root=cost_root, cost_detect=cost_detect, cost_3d=cost_3d, cost_2d=cost_2d, cost_adv=cost_adv, cost_disc=cost_disc, error3d=error3d, error2d=error2d)
        bar.next()

    bar.finish()


    if split=='test' and opt.dataset == 'fusion2':
        output_path = osp.join('./matlab', 'preds_2d_kpt_mupots.mat')
        sio.savemat(output_path, pred_2d_save)
        print("Testing result is saved at " + output_path)
        output_path = osp.join('./matlab', 'preds_3d_kpt_mupots.mat')
        sio.savemat(output_path, pred_3d_save)
        print("Testing result is saved at " + output_path)

    # Save final test results
    if split == 'test':
        file = open(os.path.join(opt.save_dir, 'result_%s.txt'.format(opt.protocol)), 'w')
        file.write('MRPE_X for test set = %.6f\n' % (mrpe_x.avg))
        file.write('MRPE_Y for test set = %.6f\n' % (mrpe_y.avg))
        file.write('MRPE_Z for test set = %.6f\n' % (mrpe_z.avg))
        file.write('MRPE for test set = %.6f\n' % (mrpe.avg))
        file.write('Reconstruction error for test set = %.6f\n' % (recon_err.avg))
        file.write('MPJPE for test set = %.6f\n' % (error3d.avg))
        file.write('2d error for test set = %.6f\n' % (error2d.avg))
        file.write('---------------------------------------------------\n')
        file.close()


    return cost_root.avg, cost_detect.avg, cost_3d.avg, cost_2d.avg, cost_adv.avg, cost_disc.avg, error3d.avg, error2d.avg

def train(epoch, opt, loader_joint, generator, g_optimizer, loader_mosh, discriminator, d_optimizer):
    return step('train', epoch, opt, loader_joint, generator, g_optimizer, loader_mosh, discriminator, d_optimizer)

def val(epoch, opt, loader_joint, generator, discriminator):
    return step('val', epoch, opt, loader_joint, generator, discriminator=discriminator)

def test(epoch, opt, loader_joint, generator, discriminator):
    return step('test', epoch, opt, loader_joint, generator, discriminator=discriminator)

