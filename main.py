import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import ref
from models.s2mpmr import S2MPMR
from models.discriminator import Discriminator
from datasets.dataloader_h36m import H36M14
from datasets.dataloader_fusion import Fusion
from datasets.dataloader_fusion2 import Fusion2
from datasets.dataloader_mupots import MuPoTS
from datasets.dataloader_mosh import Mosh
from train import train, val, test
from opts import Opts
import matplotlib.pyplot as plt


def main():
    # For repeatable experiments
    cudnn.benchmark = False
    cudnn.deterministic = True
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    # Options
    opt = Opts().parse()


    # Set gpu number
    gpus = [0]

    # Build models
    generator = S2MPMR(opt).cuda()
    generator = nn.DataParallel(generator, device_ids=gpus)
    discriminator = Discriminator().cuda()
    discriminator = nn.DataParallel(discriminator, device_ids=gpus)

    # Optimizer
    g_optimizer = torch.optim.Adam(
        generator.parameters(),
        lr=opt.lr_g
    )
    d_optimizer = torch.optim.Adam(
        discriminator.parameters(),
        lr=opt.lr_d
    )


    if opt.dataset == 'fusion':
        # Scheduler
        g_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            g_optimizer,
            milestones=[50, 70],
            gamma=0.1
        )
        d_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            d_optimizer,
            milestones=[50, 70],
            gamma=0.1
        )
        # Create data loaders
        loader_train = torch.utils.data.DataLoader(
            dataset=Fusion('train'),
            batch_size=opt.batch_size * len(gpus),
            shuffle=True,
            num_workers=ref.num_threads
        )
        loader_val = torch.utils.data.DataLoader(
            dataset=H36M14('val'),
            batch_size=opt.batch_size * len(gpus),
            shuffle=False,
            num_workers=ref.num_threads
        )

    elif opt.dataset == 'fusion2':
        g_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            g_optimizer,
            milestones=[17, 25],
            gamma=0.1
        )
        d_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            d_optimizer,
            milestones=[17, 25],
            gamma=0.1
        )
        # Create data loaders
        loader_train = torch.utils.data.DataLoader(
            dataset=Fusion2('train'),
            batch_size=opt.batch_size * len(gpus),
            shuffle=True,
            num_workers=ref.num_threads
        )
        loader_val = torch.utils.data.DataLoader(
            dataset=MuPoTS('test'),
            batch_size=opt.batch_size * len(gpus),
            shuffle=False,
            num_workers=ref.num_threads
        )


    loader_mosh = torch.utils.data.DataLoader(
        dataset=Mosh(),
        batch_size=opt.batch_size*len(gpus)  ,
        shuffle=True,
        num_workers=ref.num_threads
    )



    # History
    history = []
    history.append([]) # epoch
    history.append([]) # cost_detect (train)
    history.append([]) # cost_3d (train)
    history.append([]) # cost_2d (train)
    history.append([]) # cost_adv (train)
    history.append([]) # cost_disc (train)
    history.append([]) # error3d (train)
    history.append([]) # error2d (train)
    history.append([]) # cost_detect (val)
    history.append([]) # cost_3d (val)
    history.append([]) # cost_2d (val)
    history.append([]) # cost_adv (val)
    history.append([]) # cost_disc (val)
    history.append([]) # error3d (val)
    history.append([]) # error2d (val)



    # Load model
    idx_start = opt.num_epochs
    while idx_start > 0:
        file_name = os.path.join(opt.save_dir, 'model_{}.pth'.format(idx_start))
        if os.path.exists(file_name):
            state = torch.load(file_name)
            generator.load_state_dict(state['generator'])
            discriminator.load_state_dict(state['discriminator'])
            g_optimizer.load_state_dict(state['g_optimizer'])
            d_optimizer.load_state_dict(state['d_optimizer'])
            g_scheduler.load_state_dict(state['g_scheduler'])
            d_scheduler.load_state_dict(state['d_scheduler'])

            history_name = os.path.join(opt.save_dir, 'history_{}.pkl'.format(idx_start))
            if os.path.exists(history_name):
                with open(history_name, 'rb') as fin:
                    history = pickle.load(fin)
            break
        idx_start -= 1



    # Train
    for epoch in range(idx_start+1, opt.num_epochs+1):
        # For repeatable experiments
        np.random.seed(epoch)
        torch.manual_seed(epoch)
        torch.cuda.manual_seed(epoch)



        # Perform one epoch of training
        cost_root_train, cost_detect_train, cost_3d_train, cost_2d_train, cost_adv_train, cost_disc_train, error3d_train, error2d_train = train(
            epoch, opt, loader_train, generator, g_optimizer, loader_mosh, discriminator, d_optimizer)

        # Do scheduler
        g_scheduler.step()
        d_scheduler.step()

        # Perform one epoch of validation
        with torch.no_grad():
            cost_root_val, cost_detect_val, cost_3d_val, cost_2d_val, cost_adv_val, cost_disc_val, error3d_val, error2d_val = val(
                epoch, opt, loader_val, generator, discriminator)




        # Store history
        history[0].append(epoch)
        history[1].append(cost_detect_train)
        history[2].append(cost_3d_train)
        history[3].append(cost_2d_train)
        history[4].append(cost_adv_train)
        history[5].append(cost_disc_train)
        history[6].append(error3d_train)
        history[7].append(error2d_train)
        history[8].append(cost_detect_val)
        history[9].append(cost_3d_val)
        history[10].append(cost_2d_val)
        history[11].append(cost_adv_val)
        history[12].append(cost_disc_val)
        history[13].append(error3d_val)
        history[14].append(error2d_val)


        # Save model
        if epoch % opt.save_intervals == 0:
            state = {
                'epoch': epoch,
                'generator': generator.state_dict(),
                'discriminator': discriminator.state_dict(),
                'g_optimizer': g_optimizer.state_dict(),
                'd_optimizer': d_optimizer.state_dict(),
                'g_scheduler': g_scheduler.state_dict(),
                'd_scheduler': d_scheduler.state_dict()
            }
            torch.save(state, os.path.join(opt.save_dir, 'model_{}.pth'.format(epoch)))
            history_name = os.path.join(opt.save_dir, 'history_{}.pkl'.format(epoch))
            with open(history_name, 'wb') as fout:
                pickle.dump(history, fout)


    # Name of final model
    file_name = os.path.join(opt.save_dir, 'final_model.pth')

    # Save final model
    torch.save(state, file_name)

    # Name of final history
    history_name = os.path.join(opt.save_dir, 'final_history.pkl')

    # Save final history
    with open(history_name, 'wb') as fout:
        pickle.dump(history, fout)



    # Save histotry in matplot graph format
    x = range(1, opt.num_epochs+1)
    cost_detect_train = np.array(history[1])
    cost_3d_train = np.array(history[2])
    cost_2d_train = np.array(history[3])
    cost_adv_train = np.array(history[4])
    cost_disc_train = np.array(history[5])
    error3d_train = np.array(history[6])
    error2d_train = np.array(history[7])
    cost_detect_val = np.array(history[8])
    cost_3d_val = np.array(history[9])
    cost_2d_val = np.array(history[10])
    cost_adv_val = np.array(history[11])
    cost_disc_val = np.array(history[12])
    error3d_val = np.array(history[13])
    error2d_val = np.array(history[14])



    # Save train loss graph
    fig, ax = plt.subplots()
    ax.plot(x, cost_detect_train, 'k')
    ax.plot(x, cost_3d_train, 'r')
    ax.plot(x, cost_2d_train, 'b')
    ax.plot(x, cost_adv_train, 'm')
    ax.plot(x, cost_disc_train, 'c')
    ax.set(xlabel='epoch', ylabel='cost', title='training')
    plt.legend(('cost_detect', 'cost_3d', 'cost_2d', 'cost_adv', 'cost_disc'))
    ax.grid()
    fig.savefig(os.path.join(opt.save_dir, 'cost_train.png'))

    # Save validation loss graph
    fig, ax = plt.subplots()
    ax.plot(x, cost_detect_val, 'k')
    ax.plot(x, cost_3d_val, 'r')
    ax.plot(x, cost_2d_val, 'b')
    ax.plot(x, cost_adv_val, 'm')
    ax.plot(x, cost_disc_val, 'c')
    ax.set(xlabel='epoch', ylabel='cost', title='valid')
    plt.legend(('cost_detect', 'cost_3d', 'cost_2d', 'cost_adv', 'cost_disc'))
    ax.grid()
    fig.savefig(os.path.join(opt.save_dir, 'cost_val.png'))

    # Save 3D error graph
    fig, ax = plt.subplots()
    ax.plot(x, error3d_train, 'r')
    ax.plot(x, error3d_val, 'b')
    ax.set(xlabel='epoch', ylabel='error3d', title='MPJPE (mm)')
    plt.legend(('error3d_train', 'error3d_val'))
    ax.grid()
    fig.savefig(os.path.join(opt.save_dir, 'error3d.png'))

    # Save 2D error graph
    fig, ax = plt.subplots()
    ax.plot(x, error2d_train, 'r')
    ax.plot(x, error2d_val, 'b')
    ax.set(xlabel='epoch', ylabel='error2d', title='2D Error (pixels)')
    plt.legend(('error2d_train', 'error2d_val'))
    ax.grid()
    fig.savefig(os.path.join(opt.save_dir, 'error2d.png'))



    #--------------------------------------------------------------------
    # Test loader for final prediction
    if opt.dataset == 'fusion':
        loader_test = torch.utils.data.DataLoader(
            dataset=H36M14('val'),
            batch_size=1,
            shuffle=False,
            num_workers=ref.num_threads
        )

    elif opt.dataset == 'fusion2':
        loader_test = torch.utils.data.DataLoader(
            dataset=MuPoTS('test'),
            batch_size=1,
            shuffle=False,
            num_workers=ref.num_threads
        )

    # Generate final prediction
    with torch.no_grad():
        test(opt.num_epochs, opt, loader_test, generator, discriminator)


if __name__ == '__main__':
    main()


