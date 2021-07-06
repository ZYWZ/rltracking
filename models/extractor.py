import torch
from torch import nn

from torchvision.models import resnet50
from roi_pooling.functions.roi_pooling import roi_pooling_2d
from collections import OrderedDict
import torch.nn.functional as F
import scipy.io
import os
import numpy as np


torch.set_grad_enabled(False)

class Extractor(nn.Module):
    def __init__(self):
        self.backbone = resnet50(pretrained=True)
        self.backbone.layer4[0].conv2 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),
                                                  bias=False)
        self.backbone.layer4[0].downsample[0] = nn.Conv2d(1024, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.conv1 = nn.Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        # self.extract_feat = self.mlp([25088, 8192, 2048, 512])
        self.extract_feat = nn.Linear(25088, 512)
        del self.backbone.fc

    def mlp(self, sizes, activation=nn.Tanh, output_activation=nn.Identity):
        # Build a feedforward neural network.
        layers = []
        for j in range(len(sizes) - 1):
            act = activation if j < len(sizes) - 2 else output_activation
            layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
        return nn.Sequential(*layers)

    def forward(self, det, img):
        x = self.backbone.conv1(img)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        x = self.conv1(x)

        output_size = (7, 7)
        spatial_scale = 0.0625
        # roi_pooling_2d() function only works for gpu
        y = roi_pooling_2d(x, det, output_size,
                           spatial_scale=spatial_scale).unsqueeze(0)

        y = torch.flatten(y, start_dim=2)
        feat = self.extract_feat(y)

        return feat


def append_params(params, module, prefix):
    for child in module.children():
        for k,p in child._parameters.items():
            if p is None: continue

            if isinstance(child, nn.BatchNorm2d):
                name = prefix + '_bn_' + k
            else:
                name = prefix + '_' + k

            if name not in params:
                params[name] = p
            else:
                raise RuntimeError('Duplicated param name: {:s}'.format(name))


class MDNet(nn.Module):
    def __init__(self, model_path=None, K=1):
        super(MDNet, self).__init__()
        self.K = K
        self.layers = nn.Sequential(OrderedDict([
            ('conv1', nn.Sequential(nn.Conv2d(3, 96, kernel_size=7, stride=2),
                                    nn.ReLU(inplace=True),
                                    nn.LocalResponseNorm(2),
                                    nn.MaxPool2d(kernel_size=3, stride=2))),
            ('conv2', nn.Sequential(nn.Conv2d(96, 256, kernel_size=5, stride=2),
                                    nn.ReLU(inplace=True),
                                    nn.LocalResponseNorm(2),
                                    nn.MaxPool2d(kernel_size=3, stride=2))),
            ('conv3', nn.Sequential(nn.Conv2d(256, 512, kernel_size=3, stride=1),
                                    nn.ReLU(inplace=True))),
            ('fc4', nn.Sequential(nn.Linear(512 * 3 * 3, 512),
                                  nn.ReLU(inplace=True))),
            ('fc5', nn.Sequential(nn.Dropout(0.5),
                                  nn.Linear(512, 512),
                                  nn.ReLU(inplace=True)))]))

        self.branches = nn.ModuleList([nn.Sequential(nn.Dropout(0.5),
                                                     nn.Linear(512, 2)) for _ in range(K)])

        for m in self.layers.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0.1)
        for m in self.branches.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

        if model_path is not None:
            if os.path.splitext(model_path)[1] == '.pth':
                self.load_model(model_path)
            elif os.path.splitext(model_path)[1] == '.mat':
                self.load_mat_model(model_path)
            else:
                raise RuntimeError('Unkown model format: {:s}'.format(model_path))
        self.build_param_dict()

    def build_param_dict(self):
        self.params = OrderedDict()
        for name, module in self.layers.named_children():
            append_params(self.params, module, name)
        for k, module in enumerate(self.branches):
            append_params(self.params, module, 'fc6_{:d}'.format(k))

    def set_learnable_params(self, layers):
        for k, p in self.params.items():
            if any([k.startswith(l) for l in layers]):
                p.requires_grad = True
            else:
                p.requires_grad = False

    def get_learnable_params(self):
        params = OrderedDict()
        for k, p in self.params.items():
            if p.requires_grad:
                params[k] = p
        return params

    def get_all_params(self):
        params = OrderedDict()
        for k, p in self.params.items():
            params[k] = p
        return params

    def forward(self, x, k=0, in_layer='conv1', out_layer='fc5'):
        # forward model from in_layer to out_layer
        run = False
        for name, module in self.layers.named_children():
            if name == in_layer:
                run = True
            if run:
                x = module(x)
                if name == 'conv3':
                    x = x.view(x.size(0), -1)
                if name == out_layer:
                    return x

        x = self.branches[k](x)
        if out_layer == 'fc6':
            return x
        elif out_layer == 'fc6_softmax':
            return F.softmax(x, dim=1)

    def load_model(self, model_path):
        states = torch.load(model_path)
        shared_layers = states['shared_layers']
        self.layers.load_state_dict(shared_layers)

    def load_mat_model(self, matfile):
        mat = scipy.io.loadmat(matfile)
        mat_layers = list(mat['layers'])[0]

        # copy conv weights
        for i in range(3):
            weight, bias = mat_layers[i * 4]['weights'].item()[0]
            self.layers[i][0].weight.data = torch.from_numpy(np.transpose(weight, (3, 2, 0, 1)))
            self.layers[i][0].bias.data = torch.from_numpy(bias[:, 0])


"""
    Extractor of the given detections. Use MDNet as extractor.
"""
class ExtractorV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = MDNet()
        self.backbone.load_model("model_state_dict/mdnet_imagenet_vid.pth")

    def forward(self, det, img):
        output_size = (107, 107)
        spatial_scale = 1

        y = roi_pooling_2d(img, det, output_size,
                           spatial_scale=spatial_scale)
        output = self.backbone(y)
        return output


def build_extractor():
    return ExtractorV2()

def build_MDNet():
    return MDNet()