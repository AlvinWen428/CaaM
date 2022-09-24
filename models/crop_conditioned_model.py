import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

from .resnet224 import ResNet, BasicBlock, Bottleneck, conv3x3


class FusionModule(nn.Module):
    def __init__(self, net, if_fusion=False):
        super(FusionModule, self).__init__()
        self.net = net
        self._if_fusion = if_fusion

    def set_if_fusion(self, fusion):
        self._if_fusion = fusion

    def forward(self, x, condition):
        if self._if_fusion:
            condition_map = condition.view((*condition.shape, 1, 1))
            condition_map = condition_map.expand((*condition.shape, *x.shape[-2:]))
            input_tensor = torch.cat((x, condition_map), dim=1)
            return self.net(input_tensor)
        else:
            return self.net(x)


class FusionFCModule(FusionModule):
    def __init__(self, net, if_fusion=False):
        super(FusionFCModule, self).__init__(net, if_fusion)

    def forward(self, x, condition):
        if self._if_fusion:
            input_tensor = torch.cat((x, condition), dim=1)
            return self.net(input_tensor)
        else:
            return self.net(x)


class ResNetFusionModel(ResNet):
    def __init__(self, block, layers, fusion_layer, additional_channel, num_classes=1000):
        super(ResNetFusionModel, self).__init__(block, layers, num_classes)
        self.fusion_layer = fusion_layer
        assert self.fusion_layer in [1, 2, 3, 4, 5]
        self.additional_channel = additional_channel

        self.layer1 = FusionModule(self.layer1)
        self.layer2 = FusionModule(self.layer2)
        self.layer3 = FusionModule(self.layer3)
        self.layer4 = FusionModule(self.layer4)
        if self.fusion_layer <= 4:
            self._modify_in_channel(getattr(self, 'layer{}'.format(self.fusion_layer)), block)
        else:
            self.fc = FusionFCModule(self.fc)
            self._modify_fc_layer(self.fc)

    def _modify_in_channel(self, layer: FusionModule, block):
        # conv1
        in_channels, out_channels, stride = layer.net[0].conv1.in_channels, layer.net[0].conv1.out_channels, \
                                            layer.net[0].conv1.stride
        del layer.net[0].conv1
        if block == BasicBlock:
            layer.net[0].conv1 = conv3x3(in_channels + self.additional_channel, out_channels, stride)
        else:
            layer.net[0].conv1 = nn.Conv2d(in_channels + self.additional_channel, out_channels, kernel_size=1, bias=False)

        # downsample
        if layer.net[0].downsample is not None:
            in_channels = layer.net[0].downsample[0].in_channels
            out_channels = layer.net[0].downsample[0].out_channels
            stride = layer.net[0].downsample[0].stride
            del layer.net[0].downsample
            layer.net[0].downsample = nn.Sequential(
                nn.Conv2d(in_channels + self.additional_channel, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        # set this layer as a fusion layer
        layer.set_if_fusion(fusion=True)

    def _modify_fc_layer(self, layer: FusionFCModule):
        in_features, out_features = layer.net.in_features, layer.net.out_features
        del layer.net
        layer.net = nn.Linear(in_features + self.additional_channel, out_features)
        layer.set_if_fusion(fusion=True)

    def forward(self, x, condition):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x, condition)
        x = self.layer2(x, condition)
        x = self.layer3(x, condition)
        x = self.layer4(x, condition)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        if self.fusion_layer == 5:
            x = self.fc(x, condition)
        else:
            x = self.fc(x)

        return x


class TwoBranchResNet(nn.Module):
    def __init__(self, block, layers, fusion_layer, crop_size, condition_activation=None,
                 stop_gradient=True, num_classes=1000):
        super(TwoBranchResNet, self).__init__()
        self.coarse_model = ResNet(block, layers, num_classes)
        self.refine_model = ResNetFusionModel(block, layers, fusion_layer, num_classes, num_classes)
        self.condition_activation = condition_activation
        self.stop_gradient = stop_gradient
        self.crop_size = crop_size
        assert self.crop_size % 2 == 0

        self.transform = nn.Sequential(
            T.CenterCrop(crop_size),
            T.Pad((224-crop_size)//2)
        )

    def activation_on_condition(self, condition):
        if self.condition_activation == 'relu':
            return F.relu(condition)
        elif self.condition_activation == 'softmax':
            return F.softmax(condition, dim=1)
        elif self.condition_activation is None:
            return condition
        else:
            raise ValueError

    def forward(self, x, train_mode=False):
        cropped_x = self.transform(x.clone())
        coarse_output = self.coarse_model(cropped_x)
        if self.stop_gradient:
            input_condition = coarse_output.detach()
        else:
            input_condition = coarse_output
        input_condition = self.activation_on_condition(input_condition)

        output = self.refine_model(x, input_condition)
        if train_mode:
            return output, coarse_output
        else:
            return output


class UncropConditionedAblation(TwoBranchResNet):
    def __init__(self, block, layers, fusion_layer, crop_size, condition_activation=None,
                 stop_gradient=True, num_classes=1000):
        super(UncropConditionedAblation, self).__init__(block, layers, fusion_layer, crop_size, condition_activation,
                                                        stop_gradient, num_classes)

    def forward(self, x, train_mode=False):
        coarse_output = self.coarse_model(x)
        if self.stop_gradient:
            input_condition = coarse_output.detach()
        else:
            input_condition = coarse_output
        input_condition = self.activation_on_condition(input_condition)

        output = self.refine_model(x, input_condition)
        if train_mode:
            return output, coarse_output
        else:
            return output


def crop_conditioned_resnet18(pretrained=False, fusion_layer=4, **kwargs):
    model = TwoBranchResNet(BasicBlock, [2, 2, 2, 2], fusion_layer=fusion_layer, **kwargs)
    if pretrained:
        raise ValueError
    return model


def uncrop_conditioned_resnet18(pretrained=False, fusion_layer=4, **kwargs):
    model = UncropConditionedAblation(BasicBlock, [2, 2, 2, 2], fusion_layer=fusion_layer, **kwargs)
    if pretrained:
        raise ValueError
    return model
