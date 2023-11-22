"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

from models.networks.sync_batchnorm import DataParallelWithCallback
from models.pix2pix_model import Pix2PixModel

import math
import torch 

class Pix2PixTrainer():
    """
    Trainer creates the model and optimizers, and uses them to
    updates the weights of the network while reporting losses
    and the latest visuals to visualize the progress in training.
    """

    def __init__(self, opt):
        self.opt = opt
        self.pix2pix_model = Pix2PixModel(opt)
        if len(opt.gpu_ids) > 0:
            self.pix2pix_model = DataParallelWithCallback(self.pix2pix_model,
                                                          device_ids=opt.gpu_ids)
            self.pix2pix_model_on_one_gpu = self.pix2pix_model.module
        else:
            self.pix2pix_model_on_one_gpu = self.pix2pix_model

        self.generated = torch.empty(0).cuda()
        self.g_losses = {}
        self.d_losses = {}
        if opt.isTrain:
            self.optimizer_G, self.optimizer_D = \
                self.pix2pix_model_on_one_gpu.create_optimizers(opt)
            self.old_lr = opt.lr

        self.mini_batch_size = 8

    def run_generator_one_step(self, data):
        self.optimizer_G.zero_grad()
        self.g_losses = {}
        batch_size = data['label'].size()[0]
        num_mini_batch = math.ceil(batch_size / self.mini_batch_size) 
        self.generated = torch.empty(0).cuda()
        for mini_batch_idx in range(num_mini_batch):
            # split minibatch
            start_idx = mini_batch_idx * self.mini_batch_size
            end_idx = start_idx + self.mini_batch_size
            mini_data = {'label': data['label'][start_idx: end_idx],
                        'instance': data['instance'][start_idx: end_idx],
                        'image': data['image'][start_idx: end_idx],
                        'path': data['path'][start_idx: end_idx]
                        }
            mini_g_losses, mini_generated = self.pix2pix_model(mini_data, mode='generator')
            # accumulate gradient
            mini_g_loss = sum(mini_g_losses.values()).mean() * mini_data['label'].size()[0] / batch_size
            mini_g_loss.backward()
            # record losses: KL, GAN, GAN_Feat, VGG
            for key, l in mini_g_losses.items():
                if key in self.g_losses:
                    self.g_losses[key] = self.g_losses[key] + mini_g_losses[key] * mini_data['label'].size()[0] / batch_size
                else:
                    self.g_losses[key] = mini_g_losses[key] * mini_data['label'].size()[0] / batch_size
            # collect generated images 
            self.generated = torch.cat((self.generated, mini_generated), 0)
                
        # update with accumulated gradient
        self.optimizer_G.step()

    def run_discriminator_one_step(self, data):
        self.optimizer_D.zero_grad()
        self.d_losses = {}
        batch_size = data['label'].size()[0]
        num_mini_batch = math.ceil(batch_size / self.mini_batch_size) 
        for mini_batch_idx in range(num_mini_batch):
            # split minibatch
            start_idx = mini_batch_idx * self.mini_batch_size
            end_idx = start_idx + self.mini_batch_size
            mini_data = {'label': data['label'][start_idx: end_idx],
                        'instance': data['instance'][start_idx: end_idx],
                        'image': data['image'][start_idx: end_idx],
                        'path': data['path'][start_idx: end_idx]
                        }
            mini_d_losses = self.pix2pix_model(mini_data, mode='discriminator')
            # accumulate gradient
            mini_d_loss = sum(mini_d_losses.values()).mean() * mini_data['label'].size()[0] / batch_size
            mini_d_loss.backward()
            # record losses: D_Fake, D_real
            for key, l in mini_d_losses.items():
                if key in self.d_losses:
                    self.d_losses[key] = self.d_losses[key] + mini_d_losses[key] * mini_data['label'].size()[0] / batch_size
                else:
                    self.d_losses[key] = mini_d_losses[key] * mini_data['label'].size()[0] / batch_size
        # update with accumulated gradient
        self.optimizer_D.step()

    def get_latest_losses(self):
        return {**self.g_losses, **self.d_losses}

    def get_latest_generated(self):
        return self.generated

    def update_learning_rate(self, epoch):
        self.update_learning_rate(epoch)

    def save(self, epoch):
        self.pix2pix_model_on_one_gpu.save(epoch)

    ##################################################################
    # Helper functions
    ##################################################################

    def update_learning_rate(self, epoch):
        if epoch > self.opt.niter:
            lrd = self.opt.lr / self.opt.niter_decay
            new_lr = self.old_lr - lrd
        else:
            new_lr = self.old_lr

        if new_lr != self.old_lr:
            if self.opt.no_TTUR:
                new_lr_G = new_lr
                new_lr_D = new_lr
            else:
                new_lr_G = new_lr / 2
                new_lr_D = new_lr * 2

            for param_group in self.optimizer_D.param_groups:
                param_group['lr'] = new_lr_D
            for param_group in self.optimizer_G.param_groups:
                param_group['lr'] = new_lr_G
            print('update learning rate: %f -> %f' % (self.old_lr, new_lr))
            self.old_lr = new_lr