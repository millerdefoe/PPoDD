""" This file contains the implementation of the PoDD model """

import torch
import higher
import numpy as np
import torch.nn as nn
import torch.optim as optim

from src.data_utils import get_arch


class PoDD(nn.Module):
    def __init__(self, distilled_data, y_init, cropping_function, arch, window, lr, num_train_eval, total_patch_num,
                 distill_batch_size, num_classes=2, train_y=False, train_lr=False, channel=3, im_size=(32, 32),
                 inner_optim='SGD', cctype=0, syn_intervention=None, real_intervention=None, decay=False,
                 label_cropping_function=None, num_stages=5):
        super(PoDD, self).__init__()

        self.data = distilled_data
        self.samples_num = total_patch_num
        self.get_crops = cropping_function
        self.get_labels = label_cropping_function
        self.distill_batch_size = distill_batch_size

        self.label = y_init
        self.train_y = train_y
        if train_y:
            self.label = self.label.float().cuda().requires_grad_(True)

        self.arch = arch
        self.decay = decay
        self.cctype = cctype
        self.window = window
        self.im_size = im_size
        self.channel = channel
        self.curriculum = window
        self.num_classes = num_classes
        self.inner_optim = inner_optim
        self.num_train_eval = num_train_eval
        self.syn_intervention = syn_intervention
        self.real_intervention = real_intervention

        self.criterion = nn.CrossEntropyLoss()
        self.lr = lr if not train_lr else torch.nn.Parameter(lr)
        self.net = get_arch(arch, self.num_classes, self.channel, self.im_size)
        self.stage_proportions = np.array([1, 0.0, 0.0, 0.0, 0.0])  # Default for 5 stages
        self.num_stages = num_stages
        self.stored_posters = []  
        self.stored_labels = []  
        self.current_stage = 0


    def get_overlapping_patches_and_labels(self):
        """
        Fetch patches proportionally from completed stages only.
        Ensures only the existing stored posters (up to self.current_stage) are sampled.
        Uses dynamic probabilities for stage selection.
        """

        # ✅ Use only available stages up to `self.current_stage`
        available_stages = min(self.current_stage + 1, self.num_stages)  # e.g., if stage=0, only 0 is available, if stage=1, use 0 & 1
        
        # ✅ Normalize selection probabilities from stored proportions
        stage_probs = self.stage_proportions[:available_stages].copy()
        stage_probs = stage_probs / stage_probs.sum()  # Ensure sum is 1

        # ✅ Sample which stage each patch should be drawn from
        stage_indices = np.random.choice(available_stages, size=self.distill_batch_size, p=stage_probs)

        imgs_list, labels_list = [], []
        device = self.data.device

        # ✅ Select from both the current poster and stored ones
        all_posters = [self.data] + self.stored_posters[:self.current_stage]  
        all_labels = [self.label] + self.stored_labels[:self.current_stage]

        for stage_idx in range(available_stages):
            num_samples_from_stage = np.sum(stage_indices == stage_idx)

            if num_samples_from_stage > 0:
                perm = torch.randperm(self.samples_num, device=device)
                selected_indices = perm[:num_samples_from_stage].sort()[0]

                # ✅ Fetch patches and labels from the corresponding stored poster
                imgs_stage = self.get_crops(all_posters[stage_idx], selected_indices).to(device)
                labels_stage = all_labels[stage_idx].to(device)[selected_indices].to(device) if not self.train_y else self.get_labels(all_labels[stage_idx], selected_indices).to(device)

                imgs_list.append(imgs_stage)
                labels_list.append(labels_stage)

        # ✅ Combine all selected patches from different stages
        imgs = torch.cat(imgs_list, dim=0)
        labels = torch.cat(labels_list, dim=0)

        return imgs, labels


    def forward(self, x):
        self.net = get_arch(self.arch, self.num_classes, self.channel, self.im_size).cuda()
        self.net.train()

        if self.inner_optim == 'SGD':
            self.optimizer = optim.SGD(self.net.parameters(), lr=self.lr, momentum=0.9, weight_decay=5e-4)
            self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[200],
                                                            gamma=0.2) if self.decay else None
        elif self.inner_optim == 'Adam':
            self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)

        else:
            raise ValueError(f'inner_optim={self.inner_optim} is not supported')

        if self.dd_type not in ['curriculum', 'standard']:
            print('The dataset distillation method is not implemented!')
            raise NotImplementedError()

        if self.dd_type == 'curriculum':
            for i in range(self.curriculum):
                self.optimizer.zero_grad()
                imgs, label = self.get_overlapping_patches_and_labels()
                imgs = self.syn_intervention(imgs, dtype='syn')
                out, pres = self.net(imgs)
                loss = self.criterion(out, label)
                
                loss.backward()
                self.optimizer.step()
                if self.inner_optim == 'SGD' and self.scheduler is not None:
                    self.scheduler.step()

        loss_coef = 1
        with higher.innerloop_ctx(self.net, self.optimizer, copy_initial_weights=True) as (fnet, diffopt):
            for i in range(self.window):
                imgs, label = self.get_overlapping_patches_and_labels()
                imgs = self.syn_intervention(imgs, dtype='syn')

                if i + self.curriculum == 150 or i + self.curriculum == 240:
                    if self.inner_optim == 'SGD':
                        loss_coef = loss_coef * 0.2

                out, pres = fnet(imgs)
                loss = self.criterion(out, label)

                diffopt.step(loss)
                if self.inner_optim == 'SGD' and self.scheduler is not None:
                    self.scheduler.step()

            x = self.real_intervention(x, dtype='real')
            return fnet(x)

    def init_train(self, epoch, init=False):
        if init:
            self.net = get_arch(self.arch, self.num_classes, self.channel, self.im_size).cuda()
            if self.inner_optim == 'SGD':
                self.optimizer = optim.SGD(self.net.parameters(), lr=self.lr, momentum=0.9, weight_decay=5e-4)
                self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[600],
                                                                gamma=0.2) if self.decay else None
            elif self.inner_optim == 'Adam':
                self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)

        for i in range(epoch):
            self.optimizer.zero_grad()
            imgs, label = self.get_overlapping_patches_and_labels()
            imgs = self.syn_intervention(imgs, dtype='syn')
            out, pres = self.net(imgs)
            loss = self.criterion(out, label)
            loss.backward()
            self.optimizer.step()
            if self.inner_optim == 'SGD' and self.scheduler is not None:
                self.scheduler.step()

    # initialize the EMA with the currect data value
    def ema_init(self, ema_coef):
        self.shadow = -1e5
        self.ema_coef = ema_coef

    # update the EMA value
    def ema_update(self, grad_norm):
        if self.shadow == -1e5:
            self.shadow = grad_norm
        else:
            self.shadow -= (1 - self.ema_coef) * (self.shadow - grad_norm)
        return self.shadow

    def test(self, x):
        with torch.no_grad():
            out = self.net(x)
        return out

    def update_stage_proportions(self, stage_accuracies, min_weight=0.1, new_stage_boost=1.5):
        """
        Dynamically updates stage proportions based on accuracy across all completed stages.
        Ensures the newest poster has the highest probability initially.
        
        Args:
            stage_accuracies (list): Accuracy values for each completed stage.
            min_weight (float): Minimum probability for any stage.
            new_stage_boost (float): Factor to boost the newest stage's probability.
        """
        completed_stages = self.current_stage + 1  # Include only completed stages

        if completed_stages == 1:
            print(f"No update needed if only 1 stage exists")
            return  # No update needed if only 1 stage exists

        # 🛑 Assign initial accuracy to new stage
        if len(stage_accuracies) < completed_stages:
            prev_stage_acc = stage_accuracies[-1]  # Take last known accuracy
            stage_accuracies.append(prev_stage_acc * 0.90)  # Assign 90% of previous stage accuracy

        # 🔥 1. Compute accuracy-based weights
        acc_weights = np.array(stage_accuracies[:completed_stages]) + 1e-5  # Avoid zero division
        acc_weights /= acc_weights.sum()  # Normalize to sum to 1

        # 🔥 2. Apply a decay factor (older stages get lower weight)
        decay_factor = np.exp(-0.5 * np.arange(completed_stages))  # Exponential decay
        decay_factor = decay_factor[:completed_stages]  # Ensure correct shape

        # 🔥 3. Apply the new stage boost
        dynamic_probs = acc_weights * decay_factor
        dynamic_probs[-1] *= new_stage_boost  # Boost the newest stage
        dynamic_probs /= dynamic_probs.sum()  # Re-normalize to sum to 1

        # 🔥 4. Enforce a minimum probability for all stages
        min_probs = np.full(completed_stages, min_weight / completed_stages)  # Minimum weight per stage
        dynamic_probs = np.maximum(dynamic_probs, min_probs)  # Ensure each stage gets at least min_weight
        dynamic_probs /= dynamic_probs.sum()  # Re-normalize to sum to 1

        # 🔥 5. Update stage proportions
        self.stage_proportions[:completed_stages] = dynamic_probs  

        print(f"Updated Stage Proportions: {self.stage_proportions}")


    
    def save_current_poster(self):
        """ Store a snapshot of the current poster and label at this stage. """
        self.stored_posters.append(self.data.clone().detach())
        self.stored_labels.append(self.label.clone().detach())




def random_indices(y, nclass=10, intraclass=False, device='ca'):
    n = len(y)
    if intraclass:
        index = torch.arange(n).to(device)
        for c in range(nclass):
            index_c = index[y == c]
            if len(index_c) > 0:
                randidx = torch.randperm(len(index_c))
                index[y == c] = index_c[randidx]
    else:
        index = torch.randperm(n)
    return index


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

