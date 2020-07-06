import argparse
import os
import ref

class Opts():
    def __init__(self):
        self.parser = argparse.ArgumentParser()

    def init(self):
        # Miscellaneous
        self.parser.add_argument('-dataset', default='h36m', help='Dataset: h36m | fusion')
        self.parser.add_argument('-protocol', default='p1', help='Dataset: p1 | p2')

        # Network structure
        self.parser.add_argument('-network', default='s2mpmr',help='Network to use: s2mpmr')

        # Loss
        self.parser.add_argument('-weight_score', type=float, default=1e3, help='Weight for score loss of generator')
        self.parser.add_argument('-weight_bbox', type=float, default=1e3, help='Weight for bounding box loss of generator')
        self.parser.add_argument('-weight_root', type=float, default=1, help='Weight for root joint loss of generator')
        self.parser.add_argument('-weight_2d', type=float, default=1e1, help='Weight for 2D joint loss of generator')
        self.parser.add_argument('-weight_adv', type=float, default=1e3, help='Weight for discriminator loss of generator')
        self.parser.add_argument('-weight_disc', type=float, default=1e3, help='Weight for discriminator loss or discriminator')

        # Optimization
        self.parser.add_argument('-opt_method', default='adam', help='Optimization method')
        self.parser.add_argument('-lr_g', type=float, default=1e-4, help='Learning rate for generator')
        self.parser.add_argument('-lr_d', type=float, default=1e-4, help='Learning rate for discriminator')

        # Training options
        self.parser.add_argument('-num_epochs', type=int, default=100, help='Number of training epochs')
        self.parser.add_argument('-batch_size', type=int, default=10, help='Mini-batch size')
        self.parser.add_argument('-save_intervals', type=int, default=10, help='Number of iterations for saving model')

        # Test option
        self.parser.add_argument('-coord', default='relative', help='relative | cam')


    def parse(self):
        self.init()
        self.opt = self.parser.parse_args()

        # Set directories for experiments
        self.opt.save_dir = '%s' % (ref.exp_dir)
        if not os.path.exists(self.opt.save_dir):
            os.makedirs(self.opt.save_dir)
        self.opt.save_dir = os.path.join(self.opt.save_dir, '%s' % self.opt.network)
        if not os.path.exists(self.opt.save_dir):
            os.makedirs(self.opt.save_dir)
        self.opt.save_dir = os.path.join(self.opt.save_dir, '%s' % (self.opt.dataset))
        if not os.path.exists(self.opt.save_dir):
            os.makedirs(self.opt.save_dir)
        self.opt.save_dir = '%s/2d%1.1e_adv%1.1e_disc%1.1e_%s_lrg%1.1e_lrd%1.1e_batch%d' % (
        self.opt.save_dir, self.opt.weight_2d, self.opt.weight_adv, self.opt.weight_disc, self.opt.opt_method,
        self.opt.lr_g, self.opt.lr_d, self.opt.batch_size)
        if not os.path.exists(self.opt.save_dir):
            os.makedirs(self.opt.save_dir)

        # Save options
        args = dict((name, getattr(self.opt, name)) for name in dir(self.opt)
                    if not name.startswith('_'))
        refs = dict((name, getattr(ref, name)) for name in dir(ref)
                    if not name.startswith('_'))
        file_name = os.path.join(self.opt.save_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('==> Args:\n')
            for k, v in sorted(args.items()):
                opt_file.write('  %s: %s\n' % (str(k), str(v)))
            opt_file.write('==> Args:\n')
            for k, v in sorted(refs.items()):
                opt_file.write('  %s: %s\n' % (str(k), str(v)))

        return self.opt

