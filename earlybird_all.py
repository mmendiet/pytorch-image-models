import timm.models.layers.slimmable_ops_v2 as sov2
import torch
import torch.nn as nn
import numpy as np


class EarlyBird():
    def __init__(self, percent_list, epoch_keep=5, threshold=0.1):
        self.percent_list = percent_list
        self.epoch_keep = epoch_keep
        self.threshold = threshold
        self.masks = {percent:[] for percent in percent_list}
        self.dists = {percent:[1 for i in range(1, self.epoch_keep)] for percent in percent_list}
        self.cfgs = {}
        self.complete = False

    def pruning(self, model, percent):
        total = 0
        for n, m in model.named_modules():
            if isinstance(m, nn.BatchNorm2d) and 'resobn.0' in n:
                total += m.weight.data.shape[0]

        bn = torch.zeros(total)
        index = 0
        for n, m in model.named_modules():
            if isinstance(m, nn.BatchNorm2d) and 'resobn.0' in n:
                size = m.weight.data.shape[0]
                bn[index:(index+size)] = m.weight.data.abs().clone()
                index += size

        y, i = torch.sort(bn)
        thre_index = int(total * percent)
        thre = y[thre_index]
        # print('Pruning threshold: {}'.format(thre))

        mask = torch.zeros(total)
        index = 0
        for k, nm in enumerate(model.named_modules()):
            n = nm[0]
            m = nm[1]
            if isinstance(m, nn.BatchNorm2d) and 'resobn.0' in n:
                size = m.weight.data.numel()
                weight_copy = m.weight.data.abs().clone()
                _mask = weight_copy.gt(thre.cuda()).float().cuda()
                mask[index:(index+size)] = _mask.view(-1)
                # print('layer index: {:d} \t total channel: {:d} \t remaining channel: {:d}'.format(k, _mask.shape[0], int(torch.sum(_mask))))
                index += size

        # print('Pre-processing Successful!')
        return mask

    def put(self, mask, p):
        if len(self.masks[p]) < self.epoch_keep:
            self.masks[p].append(mask)
        else:
            self.masks[p].pop(0)
            self.masks[p].append(mask)

    def cal_dist(self, p):
        if len(self.masks[p]) == self.epoch_keep:
            for i in range(len(self.masks[p])-1):
                mask_i = self.masks[p][-1]
                mask_j = self.masks[p][i]
                self.dists[p][i] = 1 - float(torch.sum(mask_i==mask_j)) / mask_j.size(0)
            return True
        else:
            return False

    def early_bird_emerge(self, model, logger, mode=3):
        found = {}
        for percent in sorted(self.percent_list, reverse=True):
            mask = self.pruning(model, percent)
            self.put(mask, percent)
            flag = self.cal_dist(percent)
            if flag == True:
                logger.info(str(percent) + ': ' + str(self.dists[percent]))
                for i in range(len(self.dists[percent])):
                    if self.dists[percent][i] > self.threshold:
                        # return False
                        found[percent] = False
                if percent not in found:
                    found[percent] = True
            else:
                found[percent] = False

        if mode==0:
            #Grab once
            if found[max(self.percent_list)]:
                for percent, value in found.items():
                    self.cfgs[percent] = self.prune(model, percent)
        elif mode==1:
            # Grab as emerge
            for percent, value in found.items():
                if value and percent not in self.cfgs:
                    self.cfgs[percent] = self.prune(model, percent)
        elif mode==2:
            #grab as emerge but replace later
            for percent, value in found.items():
                if value and percent not in self.cfgs:
                    #grab everything from there up
                    for p, v in found.items():
                        if p <= percent:
                            self.cfgs[p] = self.prune(model, p)
                    break
        elif mode==3:
            if found[max(self.percent_list)]:
                for percent, value in found.items():
                    self.cfgs[percent] = self.prune(model, percent)
            else:
                # Grab as emerge
                for percent, value in found.items():
                    if value and percent not in self.cfgs:
                        self.cfgs[percent] = self.prune(model, percent)
        else:
            print('Invalid mode: 0, 1, or 2')
        #If all are found, mark as complete
        if set(self.cfgs.keys()) == set(self.percent_list):
            self.complete = True
        # Get final cfg list
        return [cfg for percent, cfg in sorted(self.cfgs.items())]

    def prune(self, model, pr):
        total = 0

        for n, m in model.named_modules():
            if isinstance(m, nn.BatchNorm2d) and 'resobn.0' in n:
                total += m.weight.data.shape[0]

        bn = torch.zeros(total)
        index = 0
        for n, m in model.named_modules():
            if isinstance(m, nn.BatchNorm2d) and 'resobn.0' in n:
                size = m.weight.data.shape[0]
                bn[index:(index+size)] = m.weight.data.abs().clone()
                index += size

        p_flops = 0
        y, i = torch.sort(bn)
        p_flops += total * np.log2(total) * 3
        thre_index = int(total * pr)
        thre = y[thre_index]


        if 'mobilenet' in str(type(model)):
            p_list = sorted(self.percent_list)
            dec = (0.2-0.1)/len(p_list)
            prune_min = 0.2-(p_list.index(pr)*dec)
            pruned = 0
            cfg = []
            keyword = 'body.7'
            layers = [x for x in model.named_modules()]
            for k, nm in enumerate(model.named_modules()):
                if isinstance(nm[1], nn.BatchNorm2d) and 'resobn.0' in nm[0]:
                    m = nm[1]
                    n = nm[0]
                    weight_copy = m.weight.data.abs().clone()
                    mask = weight_copy.gt(thre.cuda()).float().cuda()
                    pruned = pruned + mask.shape[0] - torch.sum(mask)
                    num = int(torch.sum(mask))
                    if keyword in n:# and layers[k-offset][1].residual_connection:
                        #grouped layer
                        cfg.append(mask.shape[0])
                    else:
                        if num < prune_min*mask.shape[0]:
                            num = int(prune_min*mask.shape[0])
                        if layers[k-3][1].groups > 1:
                            cfg.append(max(num,cfg[-1]))
                            cfg[-2] = cfg[-1]
                        else:
                            cfg.append(max(num,1))
        else:
            pruned = 0
            cfg = []
            # if FLAGS.model== 'models.resnet' and FLAGS.depth == 50:
            #     keyword = 'bn3'
            #     last = 0
            #     offset = 10 + 2*len(FLAGS.test_subnet_idx)
            # else:
            keyword = 'bn2'
            last = -1
            offset = 8 + len(sov2.test_subnet_idx)
            layers = [x for x in model.named_modules()]
            for k, nm in enumerate(model.named_modules()):
                if isinstance(nm[1], nn.BatchNorm2d) and 'resobn.0' in nm[0]:
                    m = nm[1]
                    n = nm[0]
                    weight_copy = m.weight.data.abs().clone()
                    mask = weight_copy.gt(thre.cuda()).float().cuda()
                    pruned = pruned + mask.shape[0] - torch.sum(mask)
                    num = int(torch.sum(mask))
                    if keyword in n and layers[k-offset][1].downsample is None:
                        #grouped layer
                        cfg.append(mask.shape[0])
                    elif keyword in n and layers[k-offset][1].downsample is not None:
                        # final of group, reset
                        last = -1
                    elif last == -1:
                        # First of group
                        last = max(num, 1)
                        cfg.append(mask.shape[0])
                    # elif num != 0:
                    else:
                        cfg.append(max(num,1))
        return cfg