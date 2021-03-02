import torch.nn as nn
import os

resolutions = [224, 192, 160, 128]
subnet_chnls = [[64, 64, 64, 64, 64, 128, 128, 128, 128, 256, 256, 256, 256, 512, 512, 512, 512]]
test_subnet_idx = [0,1,2,3,4,5,6,7,8,9,10]

def gen_subnet_cfgs(model='wideresnet', pruning='gate'):
    cfgs = []
    prune_chnls = subnet_chnls
    for subnetidx in range(len(prune_chnls)):
        sc = prune_chnls[subnetidx]
        subcfg = [[3, sc[0]]]
        for i in range(len(sc) - 1):
            io_channles = [sc[i], sc[i + 1]]
            subcfg.append(io_channles)
        subcfg.append([sc[-1], sc[-1]])
        cfgs.append(subcfg)

    return cfgs

cfgs = gen_subnet_cfgs()

def make_divisible(v, divisor=8, min_value=1):
    """
    forked from slim:
    https://github.com/tensorflow/models/blob/\
    0344c5503ee55e24f0de7f37336a6e08f10976fd/\
    research/slim/nets/mobilenet/mobilenet.py#L62-L69
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class USConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, depthwise=False, bias=True, layer_idx=0, shortcut=False):
        super(USConv2d, self).__init__(
            in_channels, out_channels,
            kernel_size, stride=stride, padding=padding, dilation=dilation,
            groups=groups, bias=bias)
        self.depthwise = depthwise
        self.layer_idx = layer_idx
        self.subnet_idx = 0
        self.reso_idx = 0
        self.shortcut = shortcut

    def forward(self, input):
        # cfg = cfgs[self.reso_idx][self.subnet_idx]
        cfg = cfgs[self.subnet_idx]
        if self.shortcut:
            # if FLAGS.model == 'models.wideresnet' or FLAGS.model == 'models.resnet_cifar':  # basic block
            #     shift = 1
            # elif FLAGS.model == 'models.resnet':  # bottleneck block
            #     if FLAGS.depth==18:
            #         shift = 1
            #     else:
            #         shift = 2
            # else:
                # raise ValueError('Model type error!')
            in_channels = cfg[self.layer_idx-1][0]
            out_channels = cfg[self.layer_idx][1]
        else:
            in_channels = cfg[self.layer_idx][0]
            out_channels = cfg[self.layer_idx][1]
        self.groups = in_channels if self.depthwise else 1
        weight = self.weight[:out_channels, :in_channels, :, :]
        # if not self.shortcut:
        #     print('layer_index:{}, weight_shape:{}'.format(self.layer_idx, weight.shape))
        if self.bias is not None:
            bias = self.bias[:out_channels]
        else:
            bias = self.bias
        y = nn.functional.conv2d(
            input[:, :in_channels, :, :], weight, bias, self.stride, self.padding,
            self.dilation, self.groups)
        return y


class USBatchNorm2d(nn.BatchNorm2d):
    def __init__(self, num_features, layer_idx=0):
        super(USBatchNorm2d, self).__init__(
            num_features, affine=True, track_running_stats=True)
        self.num_features_basic = num_features
        # if FLAGS.model == 'models.wideresnet':  # preact network
        #     self.shift = 0
        # elif FLAGS.model == 'models.resnet' or FLAGS.model == 'models.resnet_cifar' \
            #  or FLAGS.model == 'models.mobilenet_v2':  # postact network
        self.shift = 1
        # else:
            # raise ValueError('Model type error!')
        self.resobn = nn.ModuleList(
            [nn.BatchNorm2d(num_features, affine=True, track_running_stats=True)
            #  for i in range(len(FLAGS.resolution_list))]
            for i in range(len(test_subnet_idx))]
        )
        self.ignore_model_profiling = True
        self.layer_idx = layer_idx
        self.subnet_idx = 0
        self.reso_idx = 0
        self.postbn = False


    def forward(self, input):
        # cfg = cfgs[self.reso_idx][self.subnet_idx]
        cfg = cfgs[self.subnet_idx]
        c = cfg[self.layer_idx][self.shift]
        # weight = self.weight[:c]
        # bias = self.bias[:c]
        weight = self.resobn[self.reso_idx].weight[:c]
        bias = self.resobn[self.reso_idx].bias[:c]
        # if self.postbn or self.training is False:
        y = nn.functional.batch_norm(
            input[:, :c, :, :],
            self.resobn[self.reso_idx].running_mean[:c],
            self.resobn[self.reso_idx].running_var[:c],
            weight[:c],
            bias[:c],
            self.training,
            self.momentum,
            self.eps)
        # else:
        #     y = nn.functional.batch_norm(
        #         input[:, :c, :, :],
        #         None,
        #         None,
        #         weight[:c],
        #         bias[:c],
        #         self.training,
        #         self.momentum,
        #         self.eps)
        return y


class USLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(USLinear, self).__init__(
            in_features, out_features, bias=bias)
        self.subnet_idx = 0
        self.reso_idx = 0

    def forward(self, input):
        # cfg = cfgs[self.reso_idx][self.subnet_idx]
        cfg = cfgs[self.subnet_idx]
        in_features = cfg[-1][1]
        out_features = self.out_features
        weight = self.weight[:out_features, :in_features]
        if self.bias is not None:
            bias = self.bias[:out_features]
        else:
            bias = self.bias
        return nn.functional.linear(input[:, :in_features], weight, bias)