# date: 2022-01-04 19:25
# author: liucc
import os
import logging
import cv2
import torch
import random
import numpy as np

from dataset.dataset import get_training_data_loader, get_and_align_fp_landmark_tuples_from_fp_list, \
    get_fp_landmark_tuples_from_json_list, prepare_img_stack, SamplesCache, get_testing_data_loader
from net import FasNet

from tqdm import tqdm
from torchvision.utils import save_image
import torch.distributed as torch_ddp

from utils_common import *


def kl_distance(p: torch.Tensor, q: torch.Tensor):
    assert p.shape == q.shape, (p.shape, q.shape)
    assert len(p.shape) == 2, p.shape  # shape: [B, feat_dim]
    d = (p * torch.log(1e-5 + p / q)).sum(dim=1, keepdim=True)
    d = (d + (q * torch.log(1e-5 + q / p)).sum(dim=1, keepdim=True)) / 2.0
    return d  # shape: [B, 1]


def my_activation(x: torch.Tensor, min_val=0.5):
    # x -> min_val, then y -> +inf 
    # x -> +inf, then y -> 1.0
    y = 1.0 / (1.0 - torch.exp(min_val - x - 1e-6))
    return y - 1.0  # range: (0, +inf)

acc_test_fake_best = 0.0
manual_r_best = 0.0
weight_best = ''

fw_save_path_all_result = None
correct_cnt_all_live = 0
total_cnt_all_live = 0
correct_cnt_all_2d = 0
total_cnt_all_2d = 0
correct_cnt_all_3d = 0
total_cnt_all_3d = 0

class Solver:
    def __init__(self, live_list_fp, fake_2d_list_fp, fake_3d_list_fp, ratio_3d, img_dim, batch_size, base_lr,
                 weights_fp=None, phase='train', need_augment=True, device='cuda', using_ddp=False,
                 work_root='./tmp/base'):
        self.device, self.work_root, self.img_dim = device, work_root, img_dim
        self.live_list_fp, self.fake_2d_list_fp = live_list_fp, fake_2d_list_fp
        self.fake_3d_list_fp, self.ratio_3d, self.batch_size = fake_3d_list_fp, ratio_3d, batch_size
        self.weights_fp = weights_fp
        self.batch_size = batch_size
        self.ds, self.dl = None, None
        self.hard_samples_cache = None
        self.start_epoch = 0
        self.using_ddp = using_ddp

        local_rank = 0
        if self.using_ddp:
            torch_ddp.init_process_group(backend='nccl')
            local_rank = torch_ddp.get_rank()
            self.device = torch.device(local_rank)
            torch.cuda.set_device(self.device)
            print('[rank %d] Using DDP: ref cmd: CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch '
                  '--nproc_per_node=4 xx.py' % local_rank)
        print('preparing net and optimizer...')
        self.fas_net = FasNet(input_ch=1, img_dim=self.img_dim, hidden_dim=128).to(self.device)
        self.optimizer = torch.optim.SGD(self.fas_net.parameters(), lr=base_lr, momentum=0.9, weight_decay=5e-4)
        # self.optimizer = torch.optim.RMSprop(self.fas_net.parameters(), lr=base_lr, alpha=0.9, weight_decay=5e-4)
        if weights_fp is not None:
            self.try_to_load_weights(weights_fp=weights_fp)
        if self.using_ddp:
            self.fas_net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.fas_net)
            self.fas_net = torch.nn.parallel.DistributedDataParallel(
                self.fas_net, broadcast_buffers=False, device_ids=[local_rank]).to(self.device)

        if 'train' in phase:
            print('preparing training data set...')
            self.ds, self.dl = get_training_data_loader(
                self.live_list_fp, self.fake_2d_list_fp, self.fake_3d_list_fp, self.ratio_3d,
                self.img_dim, self.batch_size, num_workers=12, max_samples_each_id=168,
                max_samples_each_list=4000, ddp=using_ddp, need_augment=need_augment
            )
            self.hard_samples_cache = SamplesCache(
                max_num_caches=1, num_sample_types=4, sample_shape=[1, self.img_dim, self.img_dim],
                device=self.device)

        self.weights_dir = os.path.join(self.work_root, 'weights')
        self.images_dir = os.path.join(self.work_root, 'images')
        if not os.path.exists(self.weights_dir):
            os.makedirs(self.weights_dir)
        if not os.path.exists(self.images_dir):
            os.makedirs(self.images_dir)

        self.test_dl = None
        self.max_acc = 0.0
        self.tensor_0 = torch.tensor(0.0).to(self.device)

    def train_one_epoch(self, epoch):
        assert self.dl is not None and self.ds is not None
        self.ds.shuffle(reload=True)

        self.fas_net.train()
        print('\n%5s %5s %10s %10s %10s %10s %10s %10s %10s' % (
            'epoch', 'acc', 'lr', 'fake', 'random', 'twin', 'radius', 'live-m', 'fake-m'))
        bar = tqdm(enumerate(self.dl), total=len(self.dl))
        # bar = enumerate(self.dl)
        center = torch.tensor(0.0)
        for b_idx, (live_stack, fake_stack, random_fake_stack, fake_twin_stack) in bar:
            live_stack, fake_stack = live_stack.to(self.device), fake_stack.to(self.device)
            random_fake_stack = random_fake_stack.to(self.device)
            fake_twin_stack = fake_twin_stack.to(self.device)

            # loading hard samples from cache and using these samples for training
            if self.hard_samples_cache is not None:
                hard_samples = self.hard_samples_cache.get_caches()
                if hard_samples is not None:
                    hard_lv_stk, hard_fk_stk, hard_rd_fk_stk, hard_fk_tw_stk = hard_samples
                    for hd_i in range(len(hard_lv_stk)):
                        live_stack = torch.cat([live_stack, hard_lv_stk[hd_i]], dim=0)
                        fake_stack = torch.cat([fake_stack, hard_fk_stk[hd_i]], dim=0)
                        random_fake_stack = torch.cat([random_fake_stack, hard_rd_fk_stk[hd_i]], dim=0)
                        fake_twin_stack = torch.cat([fake_twin_stack, hard_fk_tw_stk[hd_i]], dim=0)
            bs = live_stack.size(0)

            # p shape: [N*bs, hidden_dim], center shape: [1, hidden_dim], radius/margin: [1, 1]
            p, center, radius, margin = self.fas_net(
                torch.cat([live_stack, fake_stack, random_fake_stack, fake_twin_stack], dim=0))
            # dist
            live_dist = kl_distance(p[:bs], center.repeat((bs, 1)))  # shape: [bs, 1]
            fake_dist = kl_distance(p[bs:2 * bs], center.repeat((bs, 1)))  # shape: [bs, 1]
            random_fake_dist = kl_distance(p[2 * bs:3 * bs], center.repeat((bs, 1)))  # shape: [bs, 1]
            # fake and its twin sample should be close
            fake_twin_dist = kl_distance(p[bs:2 * bs], p[3 * bs:])

            # loss
            live_fake_vl = torch.sigmoid(fake_dist - live_dist)  # should be close to 1.0
            live_random_vl = torch.sigmoid(random_fake_dist - live_dist)  # same as above

            live_fake_loss = -torch.log(live_fake_vl + 1e-6).mean()
            # NOTE: min(xx_dist) = 0, min(sigmoid(xx_dist)) = 0.5
            # live_fake_loss = live_fake_loss - torch.log(torch.sigmoid(fake_dist) - 0.5 + 1e-6).mean()  
            live_fake_loss = live_fake_loss + my_activation(fake_dist, 0.0).mean()
            live_random_loss = -torch.log(live_random_vl + 1e-6).mean()
            # live_random_loss = live_random_loss - torch.log(torch.sigmoid(random_fake_dist) - 0.5 + 1e-6).mean()
            live_random_loss = live_random_loss + my_activation(random_fake_dist, 0.0).mean()
            live_random_loss *= 0.5
            # radius loss
            twin_loss = fake_twin_dist.pow(2).mean()

            loss = live_fake_loss + live_random_loss + live_dist.abs().mean()  + twin_loss
            if torch.isnan(loss):
                print('loss is nan')
                exit(1)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            with torch.no_grad():
                sv_lv_idx, sv_fk_idx, sv_rd_idx = 0, 0, 0
                # cache hard samples
                if self.hard_samples_cache is not None:
                    sv_lv_idx = torch.argmax(live_dist.squeeze()[:self.batch_size]).item()
                    sv_fk_idx = torch.argmin(fake_dist.squeeze()[:self.batch_size]).item()
                    sv_rd_idx = torch.argmin(random_fake_dist.squeeze()[:self.batch_size]).item()
                    self.hard_samples_cache.set_caches(
                        samples=[
                            live_stack[sv_lv_idx][None, ...],
                            fake_stack[sv_fk_idx][None, ...],
                            random_fake_stack[sv_rd_idx][None, ...],
                            fake_twin_stack[sv_fk_idx][None, ...]
                        ])
                # debug info
                auto_radius = (live_dist.max() + fake_dist.min()) / 2.0
                # auto_radius = radius.abs().mean().detach().item()
                num_live_correct = (live_dist <= auto_radius).float().sum()
                num_fake_correct = (fake_dist >= auto_radius).float().sum()
                num_random_correct = (random_fake_dist >= auto_radius).float().sum()
                acc = (num_live_correct + num_fake_correct + num_random_correct) / (3 * bs)
                if np.random.random() < 0.08:  # random visible debug batch-0
                    save_image((live_stack[sv_lv_idx:sv_lv_idx + 1].detach().cpu().transpose(1, 0) + 1.0) / 2.0,
                               os.path.join(self.images_dir, 'live_stack.jpg'))
                    save_image((fake_stack[sv_fk_idx:sv_fk_idx + 1].detach().cpu().transpose(1, 0) + 1.0) / 2.0,
                               os.path.join(self.images_dir, 'fake_stack.jpg'))
                    save_image((random_fake_stack[sv_rd_idx:sv_rd_idx + 1].detach().cpu().transpose(1, 0) + 1.0) / 2.0,
                               os.path.join(self.images_dir, 'random_fake_stack.jpg'))
                    with open(os.path.join(self.images_dir, 'dist.txt'), 'w+') as f:
                        f.write('dist: live(%.5f), fake(%.5f), random(%.5f), twin: %.5f\n' % (
                            live_dist[sv_lv_idx].squeeze().item(), fake_dist[sv_fk_idx].squeeze().item(),
                            random_fake_dist[sv_rd_idx].squeeze().item(), twin_loss.item()
                        ))
                        f.write('acc: live(%.5f), fake(%.5f), random(%.5f)\n' % (
                            float(num_live_correct) / bs, float(num_fake_correct) / bs,
                            float(num_random_correct) / bs))
                        if self.hard_samples_cache is not None:
                            f.write('hard samples cache len(%d), idx(%d %d %d)\n' % (
                                len(self.hard_samples_cache.caches[0]), self.hard_samples_cache.cache_nums[0],
                                self.hard_samples_cache.cache_nums[1], self.hard_samples_cache.cache_nums[2]))
                            l_stk, f_stk, rf_stk, ft_stk = self.hard_samples_cache.get_caches()
                            save_image((l_stk[0].detach().cpu().transpose(1, 0) + 1.0) / 2.0,
                                       os.path.join(self.images_dir, 'hard_live_stack.jpg'))
                            save_image((f_stk[0].detach().cpu().transpose(1, 0) + 1.0) / 2.0,
                                       os.path.join(self.images_dir, 'hard_fake_stack.jpg'))
                            save_image((rf_stk[0].detach().cpu().transpose(1, 0) + 1.0) / 2.0,
                                       os.path.join(self.images_dir, 'hard_random_fake_stack.jpg'))
                            save_image((ft_stk[0].detach().cpu().transpose(1, 0) + 1.0) / 2.0,
                                       os.path.join(self.images_dir, 'hard_fake_twin_stack.jpg'))
                    # print('fake dist:', fake_dist.detach().squeeze()[:8])
                    # print('live dist:', live_dist.detach().squeeze()[:8])
            c_lr = self.optimizer.state_dict()['param_groups'][0]['lr']
            desc = '%5s %5s %10s %10s %10s %10s %10s %10s %10s' % (
                '%d' % epoch, '%.3f' % acc.item(), '%.6f' % c_lr,
                '%.5f' % live_fake_loss.item(), '%.5f' % live_random_loss.item(),
                '%.5f' % twin_loss.item(), '%.5f' % radius.abs().mean().item(),
                '%.5f' % live_dist.mean().item(), '%.5f' % fake_dist.mean().item()
            )
            bar.set_description(desc=desc)
            if (b_idx + 1) % 500:
                continue
            self.save_weights(epoch, 'epoch-%d.pth' % epoch)
        print('center:', center.detach().squeeze().cpu().numpy().tolist()[:20])
        return

    def set_optimizer_lr(self, lr):
        for p in self.optimizer.param_groups:
            p['lr'] = lr
        print('optimizer[0] lr is been set to %.5f.' % self.optimizer.state_dict()['param_groups'][0]['lr'])

    def save_weights(self, epoch, save_fn):
        if self.using_ddp:
            rk = torch_ddp.get_rank()
            if rk != 0:
                print('rank[%d] skips saving weights.' % rk)
                return  # only one process has change to save, avoiding file broken
        if hasattr(self.fas_net, 'module'):
            eqv_net_wt = self.fas_net.module.state_dict()
        else:
            eqv_net_wt = self.fas_net.state_dict()
        s_dict = {
            'epoch': epoch,
            'max_acc': self.max_acc,
            'fas_net': eqv_net_wt,
            'optimizer': self.optimizer.state_dict(),
        }
        torch.save(s_dict, os.path.join(self.weights_dir, save_fn))

    def get_running_radius_margin(self):
        if hasattr(self.fas_net, 'module'):
            return self.fas_net.module.running_radius, self.fas_net.module.running_margin
        return self.fas_net.running_radius, self.fas_net.running_margin

    def set_running_radius_margin(self, radius, margin):
        if hasattr(self.fas_net, 'module'):
            self.fas_net.module.running_radius = radius
            self.fas_net.module.running_margin = margin
        self.fas_net.running_radius = radius
        self.fas_net.running_margin = margin

    def try_to_load_weights(self, weights_fp):
        def _load_weights(_net, s_dict):
            from collections import OrderedDict
            weights_dict = OrderedDict()
            for rk, v in s_dict.items():
                k = rk[7:] if rk[:7] == 'module.' else rk
                if k not in _net.state_dict().keys():
                    # pass
                    print(k, ' key mismatch')
                elif _net.state_dict()[k].numel() != v.numel():
                    # pass
                    print('numel mismatch', _net.state_dict()[k].numel(), v.numel())
                else:
                    weights_dict[k] = v
            _net.load_state_dict(weights_dict, strict=False)
            return _net

        whole_dict = torch.load(weights_fp, map_location='cpu')
        # load net
        self.fas_net = _load_weights(self.fas_net, whole_dict['fas_net'])    # base_model   fas_net
        self.start_epoch = whole_dict['epoch'] + 1
        self.max_acc = whole_dict['max_acc']
        print('load net from {} done.'.format(weights_fp))
        try:
            # load optimizer
            self.optimizer.load_state_dict(whole_dict['optimizer'])
            print('load optimizers from {} done.'.format(weights_fp))
        except Exception as e:
            print('load from dict failed,', e)
        return

    def train(self, max_epochs=99999):
        for epoch in range(self.start_epoch, max_epochs):
            self.train_one_epoch(epoch)
            if epoch % 5 == 0 or True:
                self.save_weights(epoch, 'epoch-%d.pth' % epoch)
                # acc = self.test_one_epoch(verbose=True)
                # self.save_weights(epoch, 'last.pth')
                # if acc < self.max_acc:
                #     continue
                # self.max_acc = acc
                # self.save_weights(epoch, 'epoch_%d-acc_%.4f.pth' % (epoch, acc))
            # if epoch not in [15, 30, 90]:
            if epoch not in [1000, 5000]:
                continue
            try:
                c_lr = self.optimizer.state_dict()['param_groups'][0]['lr']
                self.set_optimizer_lr(c_lr * 0.1)
            except Exception as e:
                print('modify lr:', e)
        return

    def fast_test_old(self, fp_list_fp, batch_size, src_json=False, manual_r=0.0, expected_type='fake', desc=None):
        self.fas_net.eval()
        test_dl = get_testing_data_loader(
            fp_list_fp, batch_size, 8, self.img_dim, src_json, ddp=False)

        dst_dir = os.path.join(self.work_root, 'testing')
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)

        # if desc is None:
        #     fn = 'testing1_%s_dist.txt' % expected_type
        # else:
        #     fn = '%s_testing_%s_dist.txt' % (desc, expected_type)
        # dst_fp = os.path.join(dst_dir, fn)
        # fw = open(dst_fp, 'w+')
        # assert fw is not None, dst_fp
        save_path = self.weights_fp.replace('.pth', '_' + os.path.basename(fp_list_fp))
        print('Test result will saved in {}'.format(save_path))
        fw = open(save_path, 'w')
        assert fw is not None, save_path
        total_cnt, correct_cnt = 1e-6, 0
        data_bar = tqdm(enumerate(test_dl), total=len(test_dl), desc='testing')
        for idx, (img_stack, img_fps) in data_bar:
            img_t = img_stack.to(self.device)
            with torch.no_grad():
                bs = img_t.size(0)
                # p shape: [1, hidden_dim], center shape: [1, hidden_dim], radius/margin: [1, 1]
                p, center, radius, margin = self.fas_net(img_t)
                # cal dist
                dist = kl_distance(p, center.repeat((p.size(0), 1))).squeeze()  # shape: [bs]
                running_radius, running_margin = self.get_running_radius_margin()
                # r_s = running_radius.squeeze().abs().item()
                m_s = running_margin.squeeze().abs().item()
                r_s = radius.abs().mean().item()
                if manual_r > 0.0:
                    r_s = manual_r
                total_cnt += bs
                judge = torch.ge if 'fake' == expected_type else torch.lt
                is_correct = judge(dist, r_s).long()
                correct_cnt += float(is_correct.sum())
                for k in range(bs):
                    try:
                        # 0/1(error/correct), img_fp, dist, radius, margin
                        fw.write('%d,%s,%f,%f,%f\n' % (
                            is_correct[k].item(), img_fps[k], dist[k].item(), r_s, m_s))
                    except Exception as e:
                        print('fast test saving:', e)
                data_bar.set_description(desc='correct: %d / %d, acc: %.4f' % (
                    correct_cnt, total_cnt, float(correct_cnt) / total_cnt))
        fw.close()
        print('test acc: %f.' % (float(correct_cnt / total_cnt)))
        # print('testing dist is saved in %s.' % dst_fp)
        print('testing dist is saved in %s.' % save_path)


    def fast_test(self, fp_list_fp, batch_size, src_json=False, manual_r=0.0, expected_type='fake', desc=None, datalist_normal=True):
        self.fas_net.eval()
        fake_flag=True if expected_type == 'fake' else False
        test_dl = get_testing_data_loader(fp_list_fp, batch_size, 8, self.img_dim, src_json, ddp=False, fake_flag=fake_flag, datalist_normal=datalist_normal)

        dst_dir = os.path.join(self.work_root, 'testing')
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)

        save_path = self.weights_fp.replace('.pth', '_' + os.path.basename(fp_list_fp))
        print('Test result will saved in {}'.format(save_path))
        fw = open(save_path, 'w')
        assert fw is not None, save_path
        total_cnt, correct_cnt = 1e-6, 0
        data_bar = tqdm(enumerate(test_dl), total=len(test_dl), desc='testing')

        for idx, (img_stack, img_fps) in data_bar:
            img_t = img_stack.to(self.device)
            with torch.no_grad():
                bs = img_t.size(0)
                # p shape: [1, hidden_dim], center shape: [1, hidden_dim], radius/margin: [1, 1]
                p, center, radius, margin = self.fas_net(img_t)
                # cal dist
                dist = kl_distance(p, center.repeat((p.size(0), 1))).squeeze()  # shape: [bs]
                running_radius, running_margin = self.get_running_radius_margin()
                # r_s = running_radius.squeeze().abs().item()
                m_s = running_margin.squeeze().abs().item()
                r_s = radius.abs().mean().item()
                if manual_r > 0.0:
                    r_s = manual_r
                total_cnt += bs
                judge = torch.ge if 'fake' == expected_type else torch.lt
                is_correct = judge(dist, r_s).long()
                correct_cnt += float(is_correct.sum())
                for k in range(bs):
                    try:
                        # 0/1(error/correct), img_fp, dist, radius, margin
                        fw.write('%d,%s,%f,%f,%f\n' % (is_correct[k].item(), img_fps[k], dist[k].item(), r_s, m_s))
                    except Exception as e:
                        print('fast test saving:', e)
                data_bar.set_description(desc='correct: %d / %d, acc: %.4f' % (correct_cnt, total_cnt, float(correct_cnt) / total_cnt))

        fw.close()
        print('test acc: %f.' % (float(correct_cnt / total_cnt)))
        # print('testing dist is saved in %s.' % dst_fp)
        print('testing dist is saved in %s.' % save_path)

        # 保存各文件的准确率及结果
        global fw_save_path_all_result
        global correct_cnt_all_live
        global total_cnt_all_live
        global correct_cnt_all_2d
        global total_cnt_all_2d
        global correct_cnt_all_3d
        global total_cnt_all_3d

        fw_save_path_all_result.write("%s, correct:%d/%d, acc:%.4f\n" % (fp_list_fp, correct_cnt, total_cnt, float(correct_cnt)/total_cnt))
        if '_G_' in fp_list_fp:
            correct_cnt_all_live += correct_cnt
            total_cnt_all_live += total_cnt
        elif '_P_' in fp_list_fp:
            correct_cnt_all_2d += correct_cnt
            total_cnt_all_2d += total_cnt
        elif '_M_' in fp_list_fp:
            correct_cnt_all_3d += correct_cnt
            total_cnt_all_3d += total_cnt
        else:
            print("=====> %s name is not right" % (fp_list_fp))

    def fast_test_evaluate(self, fp_list_fp, batch_size, src_json=False, manual_r=0.0, expected_type='fake', desc=None, datalist_normal=True):
        self.fas_net.eval()
        # test_dl = get_testing_data_loader(fp_list_fp, batch_size, 12, self.img_dim, src_json, ddp=False)
        fake_flag=False if expected_type is 'live' else True
        test_dl = get_testing_data_loader(fp_list_fp, batch_size, 8, self.img_dim, src_json, ddp=False, fake_flag=fake_flag, datalist_normal=datalist_normal)

        dst_dir = os.path.join(self.work_root, 'testing')
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)

        save_path = self.weights_fp.replace('.pth', '_' + os.path.basename(fp_list_fp))
        total_cnt, correct_cnt = 1e-10, 0
        data_bar = tqdm(enumerate(test_dl), total=len(test_dl), desc='testing')

        # 将活体的准确率卡在99.322%左右，计算出对应的manual_r(r_s)
        if fake_flag == False:
            dist_all = []

        for idx, (img_stack, img_fps) in data_bar:
            img_t = img_stack.to(self.device)
            with torch.no_grad():
                bs = img_t.size(0)

                p, center, radius, margin = self.fas_net(img_t)
                dist = kl_distance(p, center.repeat((p.size(0), 1))).squeeze()  # shape: [bs]
                running_radius, running_margin = self.get_running_radius_margin()
                m_s = running_margin.squeeze().abs().item()
                r_s = radius.abs().mean().item()
                if manual_r > 0.0:
                    r_s = manual_r
                total_cnt += bs
                judge = torch.ge if 'fake' == expected_type else torch.lt
                is_correct = judge(dist, r_s).long()
                correct_cnt += float(is_correct.sum())
                data_bar.set_description(desc='correct: %d / %d, acc: %.4f' % (
                    correct_cnt, total_cnt, float(correct_cnt) / total_cnt))

            if fake_flag == False:
                for k in range(bs):
                    dist_all.append(dist[k].item())
        if fake_flag == False:
            dist_all.sort(reverse=True)
            total_cnt_99_322 = int(total_cnt*(1-0.9962))     # 0.99322   0.995  0.996   0.99616
            print("===>total_cnt_99_322: ", total_cnt_99_322)
            print("===>Get set live_99.322 manual_r: %5f" % (dist_all[total_cnt_99_322]))
            manual_r = dist_all[total_cnt_99_322]

        correct = (float(correct_cnt / total_cnt))
        print('===>%s test acc: %f.\n' % (expected_type, correct))
        if fake_flag == True:
            global acc_test_fake_best
            global manual_r_best
            global weight_best
            if correct > acc_test_fake_best:
                acc_test_fake_best = correct
                manual_r_best = manual_r
                weight_best = self.weights_fp
            print("=====================================")    
            print("acc_test_fake_best: ", acc_test_fake_best)
            print("self.manual_r_best: ", manual_r_best)
            print("self.weight_best: ", weight_best)
            print("=====================================\n") 

        return manual_r

    def fast_detect(self, list_fp, list_json=False, shuffle=True, name_with_dist=False, manual_r=0.0, max_cnt=0):
        self.fas_net.eval()
        if list_json:
            fp_landmark_tuples = get_fp_landmark_tuples_from_json_list(list_fp, max_samples_each_id=30)
        else:
            fp_landmark_tuples = get_and_align_fp_landmark_tuples_from_fp_list(
                list_fp, max_samples_each_list=5000, need_align=False)
        dst_dir = os.path.join(self.images_dir, 'fast_detect')
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)
        if shuffle:
            np.random.shuffle(fp_landmark_tuples)
        for idx, (fp, landmark) in tqdm(
                enumerate(fp_landmark_tuples), total=len(fp_landmark_tuples), desc='detecting'):
            img = cv2.imread(fp)
            assert img is not None, fp
            img_stack = prepare_img_stack(img, landmark, self.img_dim, need_augment=False)
            img_t = torch.from_numpy(img_stack[None, ...]).to(self.device)

            with torch.no_grad():
                # p shape: [1, hidden_dim], center shape: [1, hidden_dim], radius/margin: [1, 1]
                p, center, radius, margin = self.fas_net(img_t)
                # cal dist
                dist = kl_distance(p, center.repeat((p.size(0), 1))).squeeze()  # NOTE squeeze() works due to bs=1
                dist_s = dist.item()
                running_radius, running_margin = self.get_running_radius_margin()
                # r_s = running_radius.squeeze().abs().item()
                m_s = running_margin.squeeze().abs().item()
                r_s = radius.abs().mean().item()
                if manual_r > 0.0:
                    r_s = manual_r

                landmark = np.array(landmark).astype(np.int32)
                for li in range(5):
                    img = cv2.circle(img, (landmark[2 * li], landmark[2 * li + 1]), 2, (0, 255, 0), 2)
                img = cv2.resize(img, (224, 224))  # for convenient view
                img = cv2.putText(img, 'r:%.4f m:%.4f' % (r_s, m_s)
                                  , (5, 15), cv2.FONT_HERSHEY_DUPLEX, 0.4, (0, 255, 0))
                img = cv2.putText(img, '%.5f %s' % (dist_s, "L" if dist_s < r_s else "F"), (5, 40),
                                  cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 0), 2)
                img = cv2.putText(img, '%.5f %s' % (dist_s, "L" if dist_s < r_s else "F"), (5, 40),
                                  cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 0) if dist_s < r_s else (0, 0, 255), 1)
            if name_with_dist:
                dst_fp = os.path.join(dst_dir, '%.5f-%d.jpg' % (dist_s, idx))
            else:
                dst_fp = os.path.join(dst_dir, '%d.jpg' % idx)
            cv2.imwrite(dst_fp, img)
            if 0 < max_cnt < idx:
                break
        return

    def scan_abnormal_data_and_save(self, list_fp, list_json=False, manual_r=0.0, expected_type='live'):
        self.fas_net.eval()
        if list_json:
            fp_landmark_tuples = get_fp_landmark_tuples_from_json_list(list_fp, max_samples_each_id=300000)
        else:
            fp_landmark_tuples = get_and_align_fp_landmark_tuples_from_fp_list(
                list_fp, max_samples_each_list=50000000, need_align=False)
        dst_dir = os.path.join(self.images_dir, 'scanned_abnormal_data')
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)
        abn_fn_fp_w = open(os.path.join(dst_dir, 'fn_src-fp.txt'), 'w+')
        assert abn_fn_fp_w is not None, dst_dir
        for idx, (fp, landmark) in tqdm(
                enumerate(fp_landmark_tuples), total=len(fp_landmark_tuples), desc='scanning'):
            img = cv2.imread(fp)
            assert img is not None, fp
            img_stack = prepare_img_stack(img, landmark, self.img_dim, need_augment=False)
            img_t = torch.from_numpy(img_stack[None, ...]).to(self.device)

            with torch.no_grad():
                # p shape: [1, hidden_dim], center shape: [1, hidden_dim], radius/margin: [1, 1]
                p, center, radius, margin = self.fas_net(img_t)
                # cal dist
                dist = kl_distance(p, center.repeat((p.size(0), 1))).squeeze()  # NOTE squeeze() works due to bs=1
                dist_s = dist.item()
                running_radius, running_margin = self.get_running_radius_margin()
                # r_s = running_radius.squeeze().abs().item()
                r_s = radius.abs().mean().item()
                if manual_r > 0.0:
                    r_s = manual_r
                if expected_type == 'live':
                    if dist_s < r_s:
                        continue
                else:
                    if dist_s >= r_s:
                        continue
                landmark = np.array(landmark).astype(np.int32)
                for li in range(5):
                    img = cv2.circle(img, (landmark[2 * li], landmark[2 * li + 1]), 2, (0, 255, 0), 2)
                img = cv2.resize(img, (224, 224))  # for convenient view
            dst_fn = '%.5f-%d.jpg' % (dist_s, idx)
            dst_fp = os.path.join(dst_dir, dst_fn)
            cv2.imwrite(dst_fp, img)
            abn_fn_fp_w.write('%s,%s\n' % (dst_fn, fp))
        abn_fn_fp_w.close()
        return

    def fast_test_for_transplantation(self, fp_list_fp, batch_size, src_json=False, manual_r=0.0, expected_type='fake', desc=None, datalist_normal=True):
        # Save key information of preprocess, inference and postprocess for 
        # verifying the correctness of each part of code on board.
        import utils_np
        self.fas_net.eval()

        fake_flag=True if expected_type == 'fake' else False
        test_dl = get_testing_data_loader(fp_list_fp, batch_size, 1, self.img_dim, src_json, ddp=False, fake_flag=fake_flag, datalist_normal=datalist_normal)

        center = 0
        total_cnt, correct_cnt = 1e-6, 0
        data_bar = tqdm(enumerate(test_dl), total=len(test_dl), desc='testing')
        for idx, (img_stack, img_fps) in data_bar:
            utils_np.save_batch_as_bin(img_stack, img_fps)
            img_t = img_stack.to(self.device)
            with torch.no_grad():
                bs = img_t.size(0)
                # p shape: [1, hidden_dim], center shape: [1, hidden_dim], radius/margin: [1, 1]
                p, center, radius, margin = self.fas_net(img_t)
                if idx == 0:
                    log_info('center', center.cpu().detach().numpy().ravel(), False)
                # cal dist
                dist = kl_distance(p, center.repeat((p.size(0), 1))).squeeze()  # shape: [bs]
                running_radius, running_margin = self.get_running_radius_margin()
                # r_s = running_radius.squeeze().abs().item()
                m_s = running_margin.squeeze().abs().item()
                r_s = radius.abs().mean().item()
                if manual_r > 0.0:
                    r_s = manual_r
                total_cnt += bs
                judge = torch.ge if 'fake' == expected_type else torch.lt
                is_correct = judge(dist, r_s).long()
                correct_cnt += float(is_correct.sum())
                for k in range(bs):
                    try:
                        t = p[k].cpu().detach().numpy().ravel()
                        t = [str(tt) for tt in t]
                        logging.info('[{}] {}'.format(idx, img_fps[k]))
                        logging.info('probs: {}'.format(','.join(t)))
                        logging.info('dist: {:.4f}'.format(dist.item()))
                    except Exception as e:
                        print('fast test saving:', e)
                data_bar.set_description(desc='correct: %d / %d, acc: %.4f' % (
                    correct_cnt, total_cnt, float(correct_cnt) / total_cnt))
        print('test acc: %f.' % (float(correct_cnt / total_cnt)))
        return



def to_onnx2(model, ckpt_path):
    loc = 'cuda:{}'.format(0)
    checkpoint = torch.load(ckpt_path, map_location=loc)
    model.load_state_dict(checkpoint['fas_net'])
    model.eval()
    x = torch.randn([1, 1, 256, 256], requires_grad=True)
    # x = torch.randn([1, 1, 112, 112], requires_grad=True)
    dst_path = ckpt_path.replace('.pth', '.onnx')
    torch.onnx.export(model, x, dst_path, verbose=True)

def fast_run():
    # config
    phase = 'test'     # test, detect, save, yizhi, pytorch_onnx
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    gpu_id = 0

    # jiadu model
    weights_fp = '/ssd1/wangwj/work_dilusense/RGBD/fas/tmp/lvmi_mobileNet_autoRadius_k0_test1/weights/epoch-43.pth'
    manual_r_test = 1.5209492444992065
    weights_fp = 'pre_checkpoint/epoch-32.pth'
    manual_r_test = 0.487713
    weights_fp = 'pre_checkpoint/epoch-45.pth'
    manual_r_test = 0.8927957
    weights_fp = 'pre_checkpoint/epoch-52.pth'
    manual_r_test = 0.983765
    weights_fp = '/ssd1/wangwj/work_dilusense/RGBD/fas/tmp/lvmi_mobileNet_autoRadius_k0_test4_solver_tain/weights/epoch-68.pth'
    manual_r_test = 0.987585
    weights_fp = 'pre_checkpoint/epoch-78.pth'
    manual_r_test = 1.018727
    # weights_fp = 'pre_checkpoint/epoch-223.pth'
    # manual_r_test = 0.670
    # weights_fp = '/ssd1/wangwj/work_dilusense/RGBD/fas/tmp/lvmi_mobileNet_autoRadius_k0_test4_solver_tain_epoch236/weights/epoch-251.pth'
    # manual_r_test = 0.555
    weights_fp = '/ssd1/wangwj/work_dilusense/RGBD/fas/tmp/lvmi_mobileNet_autoRadius_k0_test4_solver_tain_epoch236/weights/epoch-251.pth'
    manual_r_test = 0.83886

    # lvmi DM8
    # weights_fp = '/ssd1/wangwj/work_dilusense/DM8/fas_use_server5_data/tmp/lvmi_mobileNet_autoRadius_k0_test_7_big_error/weights/epoch-324.pth'
    # manual_r_test = 0.2582

    k_num = 0
    idx = 1
    param = {
        'work_root': './tmp_test/lvmi_mobileNet_autoRadius_k{}/{}'.format(k_num, idx),
        'live_list_fp': '/home/liucc/work/workspace/dataset/face_antiSpoofing/lvmi/live/training_set_dict_link/json_list.txt',
        'fake_2d_list_fp': '/work/liucc/workspace/dataset/face_antiSpoofing/lvmi/fake_images_file_path_list/2d_fake_list_training.txt',
        'fake_3d_list_fp': '/work/liucc/workspace/dataset/face_antiSpoofing/lvmi/fake_images_file_path_list/3d_fake_list_training_k%d.txt' % k_num
    }

    using_ddp = True if torch.cuda.device_count() > 1 else False
    using_ddp = False
    solver = Solver(
        live_list_fp=param['live_list_fp'],
        fake_2d_list_fp=param['fake_2d_list_fp'],
        fake_3d_list_fp=param['fake_3d_list_fp'],
        weights_fp=weights_fp,
        ratio_3d=0.8,
        img_dim=256,
        batch_size=64,
        base_lr=1e-2,
        phase=phase,
        need_augment=True,  # set false when fine mining(seems got worse res, better keeps True)
        work_root=param['work_root'],
        using_ddp=using_ddp,
        device='cuda:{}'.format(gpu_id)
    )

    if 'test' in phase:
        # '''
        # DM8项目
        test_files = [
            # # 2d fake
            # '/ssd1/wangwj/dataset/lvmi_dm8/ir/fake/DM8_lvmi046_20220220_ir_fake_faceinfo_epoch-117_352_320_0.88_P_test.txt',
            # '/ssd1/wangwj/dataset/lvmi_dm8/ir/fake/DM8_lvmi089_20220220_ir_fake_faceinfo_epoch-117_352_320_0.88_P_test.txt',

            # # 3d fake
            # '/ssd1/wangwj/dataset/crop_save_dir_wjh/test_k0/IFAS_lvmi_FAS2022_M_test.txt',
            # '/ssd1/wangwj/dataset/lvmi_dm8/ir/fake/DM8_lvmi046_20220220_ir_fake_faceinfo_epoch-117_352_320_0.88_M_test.txt',
            # '/ssd1/wangwj/dataset/lvmi_dm8/ir/fake/DM8_lvmi089_20220220_ir_fake_faceinfo_epoch-117_352_320_0.88_M_test.txt',
            # '/ssd1/wangwj/dataset/lvmi_dm8/ir/fake/DM8_lvmi102_20220128_ir_fake_faceinfo_epoch-117_352_320_0.88_M_test.txt',
            # '/ssd1/wangwj/dataset/lvmi_dm8/ir/fake/DM8_lvmi125_20220127_ir_fake_faceinfo_epoch-117_352_320_0.88_M_test.txt',

            # # live
            # '/ssd1/wangwj/dataset/lvmi_dm8/ir/live/lv_20220225_ir_live_faceinfo_epoch-117_352_320_0.88_test_G_all_noocc.txt',
            # '/ssd1/wangwj/dataset/lvmi_dm8/ir/live/lv_20220311_ir_live_faceinfo_epoch-117_352_320_0.88_test_G_all_noocc.txt',
            # '/ssd1/wangwj/dataset/lvmi_dm8/ir/live/lv_20220315_ir_live_faceinfo_epoch-117_352_320_0.88_test_G_all_noocc.txt',
            # '/ssd1/wangwj/dataset/lvmi_dm8/ir/live/lv_20220312_ir_live_faceinfo_epoch-117_352_320_0.88_test_G_kids_all_noocc.txt',
            # '/ssd1/wangwj/dataset/lvmi_dm8/ir/live/lv_20220312_ir_live_faceinfo_epoch-117_352_320_0.88_test_G_old_man_all_noocc.txt'

            # 统一测试数据
            # live
            # '/storage-server5/temp_data/hewu/lvmi_dm8_bctc/flist/live/DM8203_BCTC_20220527_G_all_faceinfo_epoch-117_352_320_0.88_test.json',    # 使用下面的txt替换该json
            '/ssd1/wangwj/dataset/lvmi_dm8/ir/BCTC/DM8203_BCTC_20220527_G_all_faceinfo_epoch-117_352_320_0.88_test.txt',
            #'/storage-server5/temp_data/hewu/lvmi_dm8_bctc/flist/live/dm8_lv_20220527_ir_live_faceinfo_hefei_office_test_G_all.txt',
            '/storage-server5/temp_data/hewu/lvmi_dm8_bctc/flist/live/dm8_lv_20220315_ir_live_faceinfo_epoch-117_352_320_0.88_test_G_all_noocc.txt',
            '/storage-server5/temp_data/hewu/lvmi_dm8_bctc/flist/live/dm8_lv_20220312_ir_live_faceinfo_epoch-117_352_320_0.88_test_G_old_man_all_noocc.txt',
            '/storage-server5/temp_data/hewu/lvmi_dm8_bctc/flist/live/dm8_lv_20220312_ir_live_faceinfo_epoch-117_352_320_0.88_test_G_kids_all_noocc.txt',
            '/storage-server5/temp_data/hewu/lvmi_dm8_bctc/flist/live/dm8_lv_20220311_ir_live_faceinfo_epoch-117_352_320_0.88_test_G_all_noocc.txt',
            '/storage-server5/temp_data/hewu/lvmi_dm8_bctc/flist/live/dm8_lv_20220225_ir_live_faceinfo_epoch-117_352_320_0.88_test_G_all_noocc.txt',

            # 2d fake
            '/storage-server5/temp_data/hewu/lvmi_dm8_bctc/flist/2d_fake/DM8_lvmi089_20220220_ir_fake_faceinfo_epoch-117_352_320_0.88_P_test.txt',
            '/storage-server5/temp_data/hewu/lvmi_dm8_bctc/flist/2d_fake/DM8_lvmi046_20220220_ir_fake_faceinfo_epoch-117_352_320_0.88_P_test.txt',

            # 3d fake
            '/storage-server5/temp_data/hewu/lvmi_dm8_bctc/flist/live/DM8_lvmi_20220504_ir_fake_faceinfo_M_LSZ_test.txt',
            '/storage-server5/temp_data/hewu/lvmi_dm8_bctc/flist/3d_fake/IFAS_lvmi_FAS2022_M_test.txt',
            '/storage-server5/temp_data/hewu/lvmi_dm8_bctc/flist/3d_fake/DM8_lvmi125_20220127_ir_fake_faceinfo_epoch-117_352_320_0.88_M_test.txt',
            '/storage-server5/temp_data/hewu/lvmi_dm8_bctc/flist/3d_fake/DM8_lvmi102_20220128_ir_fake_faceinfo_epoch-117_352_320_0.88_M_test.txt',
            '/storage-server5/temp_data/hewu/lvmi_dm8_bctc/flist/3d_fake/DM8_lvmi089_20220220_ir_fake_faceinfo_epoch-117_352_320_0.88_M_test.txt',
            '/storage-server5/temp_data/hewu/lvmi_dm8_bctc/flist/3d_fake/DM8_lvmi046_20220220_ir_fake_faceinfo_epoch-117_352_320_0.88_M_test.txt'   
        ]

        # test_files = [
        #     '/ssd1/wangwj/work_dilusense/DM8/fas_test_1/test_data/live_error/IFAS_BCTC_test_G_faceinfo.txt',
        #     '/ssd1/wangwj/work_dilusense/DM8/fas_test_1/test_data/live_error/IFAS_BCTC_test_M_faceinfo.txt',
        #     '/ssd1/wangwj/dataset/lvmi_dm8/DM8203_BCTC_20220527/DM8203_BCTC_20220527_G_all_faceinfo_epoch-117_352_320_0.88.txt'
        # ]
        # '''

        # RGBD项目
        # test_files = [
        #     # '/ssd1/wangwj/dataset/dataset_new/Fake_anti_dataset/FAS_Jiadu/shifang/IFAS_112x112/jiadu_ir_fake_test_real_v1.txt',
        #     '/ssd1/wangwj/dataset/dataset_new/Fake_anti_dataset/FAS_Jiadu/shifang/IFAS_112x112/jiadu_ir_fake_test_real_v1_G_.txt',
        #     '/ssd1/wangwj/dataset/dataset_new/Fake_anti_dataset/FAS_Jiadu/shifang/IFAS_112x112/jiadu_ir_fake_test_real_v1_M_.txt',
        #     '/ssd1/wangwj/dataset/dataset_new/Fake_anti_dataset/FAS_Jiadu/rgbd/rgbd_20211010/ir/IFAS_rbgd_20211010_G_test.txt',
        #     '/ssd1/wangwj/dataset/dataset_new/Fake_anti_dataset/FAS_Jiadu/ir_fake/rgbd/rgbd_20211111/ir/IFAS_rgbd_20211111_G_test.txt',
        #     '/ssd1/wangwj/dataset/dataset_new/Fake_anti_dataset/FAS_Jiadu/ir_fake/rgbd/rgbd_20211014/IFAS_rbgd_20211012_M_test.txt',
        #     '/ssd1/wangwj/dataset/dataset_new/Fake_anti_dataset/FAS_Jiadu/ir_fake/rgbd/rgbd_20211014/IFAS_rbgd_20211014_M_test.txt',
        #     '/ssd1/wangwj/dataset/dataset_new/Fake_anti_dataset/FAS_Jiadu/ir_fake/rgbd/rgbd_20211111/ir/IFAS_rgbd_20211111_M_test.txt',
        #     '/ssd1/wangwj/dataset/dataset_new/Fake_anti_dataset/FAS_Jiadu/app/rgbd640/20210929/ir/IFAS_rgbd640_20210929_M_test.txt',
        #     # '/ssd1/wangwj/dataset/dataset_new/Fake_anti_dataset/FAS_Jiadu/ir_fake/rgbd/rgbd_20211109/ir/IFAS_rgbd_20211109_GM_test.txt',
        #     '/ssd1/wangwj/dataset/dataset_new/Fake_anti_dataset/FAS_Jiadu/ir_fake/rgbd/rgbd_20211109/ir/IFAS_rgbd_20211109_GM_test_G_.txt',
        #     '/ssd1/wangwj/dataset/dataset_new/Fake_anti_dataset/FAS_Jiadu/ir_fake/rgbd/rgbd_20211109/ir/IFAS_rgbd_20211109_GM_test_M_.txt',
        #     '/ssd1/wangwj/dataset/dataset_new/Fake_anti_dataset/FAS_Jiadu/ir_fake/rgbd/rgbd_20211015/IFAS_rgbd_20211015_P_test.txt',
        #     '/ssd1/wangwj/dataset/dataset_new/Fake_anti_dataset/FAS_Jiadu/ir_fake/rgbd/rgbd_20211016/IFAS_rgbd_20211016_P_test.txt',
        #     '/ssd1/wangwj/dataset/dataset_new/Fake_anti_dataset/FAS_Jiadu/rgbd/rgbd_20211227/ir/IFAS_rgbd_20211227_G_test.txt',    
        #     '/ssd1/wangwj/dataset/dataset_new/Fake_anti_dataset/FAS_Jiadu/ir_fake/rgbd/rgbd_20211223/ir/IFAS_rgbd_20211223_M_test.txt',
        #     '/ssd1/wangwj/dataset/dataset_new/Fake_anti_dataset/FAS_Jiadu/ir_fake/rgbd/rgbd_20211224/ir/IFAS_rgbd_20211224_M_test.txt'
        # ]

        # test_files = [
        #     # '/ssd1/wangwj/work_dilusense/RGBD/fas/dataset/ir_RGBD/list/ir_RGBD_faceinfo_epoch-117_352_320_0.88_G_RGBD_tongyikeypoint.txt'
        #     # '/ssd1/wangwj/work_dilusense/RGBD/fas/dataset/ir_RGBD/list/ir_RGBD_faceinfo_epoch-117_352_320_0.88_G_RGBD.txt'
        #     '/ssd1/wangwj/work_dilusense/RGBD/fas/dataset/ir_RGBD/list/ir_RGBD_all_faceinfo_epoch-117_352_320_0.88_G_RGBD.txt'
        # ]

        global fw_save_path_all_result
        global correct_cnt_all_live
        global total_cnt_all_live
        global correct_cnt_all_2d
        global total_cnt_all_2d
        global correct_cnt_all_3d
        global total_cnt_all_3d
        save_path_all_result = weights_fp.replace('.pth', '_result_acc.txt')
        fw_save_path_all_result = open(save_path_all_result, 'w')

        for f in test_files:
            is_json = True if f.endswith('.json') else False
            is_fake = 'live' if '_G_' in os.path.basename(f) else 'fake'

            solver.fast_test(
                fp_list_fp=f,
                src_json=is_json,
                expected_type=is_fake,
                batch_size=64, 
                manual_r = manual_r_test,
                datalist_normal = True    # mormal为True表示list是类似于DM8的那种标准的数据集，其他均为非标准的数据集为False，如RGBD数据
            )

        # 统计总的结果
        if total_cnt_all_live != 0:
            fw_save_path_all_result.write("live correct:%d/%d, acc:%.4f\n" % (correct_cnt_all_live, total_cnt_all_live, float(correct_cnt_all_live)/total_cnt_all_live))
        if total_cnt_all_2d != 0:
            fw_save_path_all_result.write("2d correct:%d/%d, acc:%.4f\n" % (correct_cnt_all_2d, total_cnt_all_2d, float(correct_cnt_all_2d)/total_cnt_all_2d))
        if total_cnt_all_3d != 0:
            fw_save_path_all_result.write("3d correct:%d/%d, acc:%.4f\n" % (correct_cnt_all_3d, total_cnt_all_3d, float(correct_cnt_all_3d)/total_cnt_all_3d)) 

        # 关闭文件句柄
        fw_save_path_all_result.close()
        print("Test over")
    
    if 'yizhi' in phase:
        level = logging.INFO
        # file_list_path = '/ssd1/wangjh/data/lvmi_dm8/lvmi_dm8_verify_SDK/ir/IFAS_lvmi_dm8_verify.txt'
        # file_list_path = './yizhi/IFAS_lvmi_dm8_verify.txt'
        # file_list_path = '/ssd1/wangwj/data/lvmi_dm8/ir/images_verify/IFAS_lvmi_dm8_verify.txt'
        # file_list_path = '/ssd1/wangwj/dataset/lvmi_dm8/ir/images_verify/IFAS_lvmi_dm8_M_verify.txt'

        # RGBD
        file_list_path = './dataset/datalist/images_verify/RGBD_G_verify.txt'
        is_json = True if file_list_path.endswith('.json') else False
        is_fake = 'live' if '_G_' in os.path.basename(file_list_path) else 'fake'
        set_logger(file_list_path.replace('.txt', '_output.txt'), level=level)
        logging.basicConfig(level=level, format='%(message)s')
        solver.fast_test_for_transplantation(
            fp_list_fp=file_list_path, desc='3d',
            # fp_list_fp='/ssd1/wangjh/data/ir_fake/lvmi_DM8/lvmi_20220301/test.txt', desc='3d',
            src_json=is_json,
            expected_type=is_fake,
            batch_size=1,
            manual_r=manual_r_test,     # 1.595906
            datalist_normal = False    # mormal为True表示list是类似于DM8的那种标准的数据集，其他均为非标准的数据集为False，如RGBD数据
        )
    
    if 'detect' in phase:
        solver.fast_detect(
            # list_fp='/home/liucc/work/workspace/dataset/face_antiSpoofing/lvmi/live/testing_set_dict_link/json_list.txt',
            # list_json=True,
            list_fp='./tmp/lvmi_1/fast_test/fast_test_list.txt',
            # list_fp='./tmp/lvmi_mobileNet_autoRadius_4_k0/hard-samples/hard_samples_SLTM.txt',
            # list_fp='/home/liucc/work/workspace/dataset/face_antiSpoofing/lvmi/fake_images_file_path_list/fake_list_testing_k0_only_3d.txt',
            # list_fp='/home/liucc/work/workspace/dataset/face_antiSpoofing/lvmi/fake_images_file_path_list/fake_list_testing_k2_only_3d.txt',
            list_json=False,
            shuffle=True,
            name_with_dist=True,
            manual_r=1.0,
            max_cnt=2000
        )
        
    if 'pytorch_onnx' in phase:

        from net_for_onnx import FasNet

        model = FasNet(input_ch=1, img_dim=256, hidden_dim=128)
        # model = FasNet(input_ch=1, img_dim=112, hidden_dim=128)
        to_onnx2(model, weights_fp)


def fast_run_new():
    # config
    os.environ['CUDA_VISIBLE_DEVICES'] = '6'
    gpu_id = 0

    # model dir
    # RGBD
    # weights_fp = '/ssd1/wangwj/work_dilusense/RGBD/fas/tmp/lvmi_mobileNet_autoRadius_k0_test4_solver_tain_epoch236/weights/'
    # manual_r_test = 0.5 # 初始值

    # DM8
    weights_fp = '/ssd1/wangwj/work_dilusense/DM8/fas_use_server5_data/tmp/lvmi_mobileNet_autoRadius_k0_test_8_big_error_add_live_twin/weights/'
    weights_fp = '/ssd1/wangwj/work_dilusense/DM8/fas_use_server5_data/tmp/lvmi_mobileNet_autoRadius_k0_test_9_big_error_add_live_twin_epoch-316/weights/'
    weights_fp = '/ssd1/wangwj/work_dilusense/DM8/fas_use_server5_data/tmp/lvmi_mobileNet_autoRadius_k0_test_9_big_error_add_live_twin_epoch-320/weights/'
    manual_r_test = 0.5 # 初始值

    # liucc
    # weights_fp = '/ssd1/wangwj/work_dilusense/DM8/fas_test_1/checkpoints/dm8/cdl/'
    # manual_r_test = 0.5 # 初始值

    k_num = 0
    idx = 1
    param = {
        'work_root': './tmp_test/lvmi_mobileNet_autoRadius_k{}/{}'.format(k_num, idx),
        'live_list_fp': '/home/liucc/work/workspace/dataset/face_antiSpoofing/lvmi/live/training_set_dict_link/json_list.txt',
        'fake_2d_list_fp': '/work/liucc/workspace/dataset/face_antiSpoofing/lvmi/fake_images_file_path_list/2d_fake_list_training.txt',
        'fake_3d_list_fp': '/work/liucc/workspace/dataset/face_antiSpoofing/lvmi/fake_images_file_path_list/3d_fake_list_training_k%d.txt' % k_num
    }

    using_ddp = True if torch.cuda.device_count() > 1 else False
    using_ddp = False

    '''
    # RGBD
    test_files = [           
        # test thd
        './dataset/datalist/rgbd_20220507_all_G_test.txt',
        './dataset/datalist/rgbd_20220507_all_M_test.txt'
    ]
    '''

    #'''
    # DM8
    test_files = [           
        # test thd
        '/ssd1/wangwj/dataset/lvmi_dm8/ir/live/lv_ir_live_faceinfo_test_epoch-117_G_all_noocc.txt',
        '/ssd1/wangwj/dataset/lvmi_dm8/ir/fake/lv_ir_live_faceinfo_test_epoch-117_M_all_noocc.txt'
    ]
    #'''

    if '.pth' not in weights_fp:
        pth_count = 0
        # manual_r_best = 0.0
        # acc_test_fake_best = 0.0
        # weight_best = ""
        weights_dict = {}
        for rootDir, subrootDirlist, filelist in os.walk(weights_fp): 
            for weight_fp in filelist:
                if '.pth' not in weight_fp:
                    continue

                weight_fp = rootDir + weight_fp
                print("weight_fp: %s(%d/%d)" % (weight_fp, pth_count, len(filelist)))
                pth_count = pth_count + 1

                solver = Solver(
                            live_list_fp=param['live_list_fp'],
                            fake_2d_list_fp=param['fake_2d_list_fp'],
                            fake_3d_list_fp=param['fake_3d_list_fp'],
                            weights_fp=weight_fp,
                            ratio_3d=0.8,
                            img_dim=256,
                            batch_size=512,
                            base_lr=1e-2,
                            phase='test',
                            need_augment=True,  # set false when fine mining(seems got worse res, better keeps True)
                            work_root=param['work_root'],
                            using_ddp=using_ddp,
                            device='cuda:{}'.format(gpu_id))
                
                for f in test_files:
                    is_json = True if f.endswith('.json') else False
                    is_fake = 'live' if '_G_' in os.path.basename(f) else 'fake'
                    print("use manual_r_test is : ", manual_r_test)
                    manual_r_test = solver.fast_test_evaluate(
                        fp_list_fp=f,
                        src_json=is_json,
                        expected_type=is_fake,
                        batch_size=512,   # 
                        manual_r = manual_r_test,
                        datalist_normal = True    # mormal表示list是类似于DM8的那种标准的数据集，其他均为非标准的数据集，如RGBD数据
                    )
                print("manual_r_test：", manual_r_test)
                weights_dict[weight_fp] = manual_r_test
                print(weight_fp+' : ' , weights_dict[weight_fp])
                
        # 保存结果
        weights_dict_sort = sorted(weights_dict.items(), key = lambda kv:(kv[1], kv[0]))
        print(weights_dict_sort)
        result_evaluate = weights_fp + '/' + 'evaluate_result.json'
        with open(result_evaluate, 'w') as evaluate_fp:
            json.dump(weights_dict_sort, evaluate_fp, indent=4)
                
    else:
        print("Error weights_fp..., The path should be dic not file\n")

if __name__ == '__main__':
    # fast_run()           # 正常测试与发布
    fast_run_new()         # 阈值测试方法
