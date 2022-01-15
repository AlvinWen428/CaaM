import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .resnet224 import ResNet, BasicBlock, ResNet_Feature
from .crop_conditioned_model import ResNetFusionModel


class TwoInputResNet(nn.Module):
    def __init__(self, block, layers, fusion_layer, condition_activation=None,
                 stop_gradient=True, num_classes=1000):
        super(TwoInputResNet, self).__init__()
        self.coarse_model = ResNet(block, layers, num_classes)
        self.refine_model = ResNetFusionModel(block, layers, fusion_layer, num_classes, num_classes)
        self.condition_activation = condition_activation
        self.stop_gradient = stop_gradient

    def activation_on_condition(self, condition):
        if self.condition_activation == 'relu':
            return F.relu(condition)
        elif self.condition_activation == 'softmax':
            return F.softmax(condition, dim=1)
        elif self.condition_activation is None:
            return condition
        else:
            raise ValueError

    def forward(self, x, processed_x, train_mode=False):
        coarse_output = self.coarse_model(processed_x)
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


class TwoInputSharedResNet(nn.Module):
    def __init__(self, block, layers, condition_activation=None, stop_gradient=True, num_classes=1000):
        super(TwoInputSharedResNet, self).__init__()
        self.feature_extractor = ResNet_Feature(block, layers, num_classes)
        self.coarse_classifier = nn.Linear(512 * block.expansion, num_classes)
        self.refine_classifier = nn.Linear(512 * block.expansion + num_classes, num_classes)
        self.condition_activation = condition_activation
        self.stop_gradient = stop_gradient

    def activation_on_condition(self, condition):
        if self.condition_activation == 'relu':
            return F.relu(condition)
        elif self.condition_activation == 'softmax':
            return F.softmax(condition, dim=1)
        elif self.condition_activation is None:
            return condition
        else:
            raise ValueError

    def forward(self, x, processed_x, train_mode=False):
        coarse_feature = self.feature_extractor(processed_x)
        coarse_output = self.coarse_classifier(coarse_feature)
        if self.stop_gradient:
            input_condition = coarse_output.detach()
        else:
            input_condition = coarse_output
        input_condition = self.activation_on_condition(input_condition)

        refine_feature = self.feature_extractor(x)
        refine_output = self.refine_classifier(torch.cat([refine_feature, input_condition], dim=1))
        if train_mode:
            return refine_output, coarse_output
        else:
            return refine_output


class NoSaliencyConditionedAblation(TwoInputResNet):
    def __init__(self, block, layers, fusion_layer, condition_activation=None,
                 stop_gradient=True, num_classes=1000):
        super(NoSaliencyConditionedAblation, self).__init__(block, layers, fusion_layer, condition_activation,
                                                            stop_gradient, num_classes)

    def forward(self, x, processed_x, train_mode=False):
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


class NoSaliencyConditionedSharedAblation(TwoInputSharedResNet):
    def __init__(self, block, layers, condition_activation=None, stop_gradient=True, num_classes=1000):
        super(NoSaliencyConditionedSharedAblation, self).__init__(block, layers, condition_activation,
                                                                  stop_gradient, num_classes)

    def forward(self, x, processed_x, train_mode=False):
        coarse_feature = self.feature_extractor(x)
        coarse_output = self.coarse_classifier(coarse_feature)
        if self.stop_gradient:
            input_condition = coarse_output.detach()
        else:
            input_condition = coarse_output
        input_condition = self.activation_on_condition(input_condition)

        refine_feature = self.feature_extractor(x)
        refine_output = self.refine_classifier(torch.cat([refine_feature, input_condition], dim=1))
        if train_mode:
            return refine_output, coarse_output
        else:
            return refine_output


class OnlySaliencySharedAblation(TwoInputSharedResNet):
    def __init__(self, block, layers, condition_activation=None, stop_gradient=True, num_classes=1000):
        super(OnlySaliencySharedAblation, self).__init__(block, layers, condition_activation,
                                                         stop_gradient, num_classes)

    def forward(self, x, processed_x, train_mode=False):
        coarse_feature = self.feature_extractor(processed_x)
        coarse_output = self.coarse_classifier(coarse_feature)
        if self.stop_gradient:
            input_condition = coarse_output.detach()
        else:
            input_condition = coarse_output
        input_condition = self.activation_on_condition(input_condition)

        refine_feature = self.feature_extractor(processed_x)
        refine_output = self.refine_classifier(torch.cat([refine_feature, input_condition], dim=1))
        if train_mode:
            return refine_output, coarse_output
        else:
            return refine_output


def saliency_conditioned_resnet18(pretrained=False, fusion_layer=4, **kwargs):
    model = TwoInputResNet(BasicBlock, [2, 2, 2, 2], fusion_layer=fusion_layer, **kwargs)
    if pretrained:
        raise ValueError
    return model


def saliency_conditioned_shared_resnet18(pretrained=False, **kwargs):
    model = TwoInputSharedResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        raise ValueError
    return model


def nosaliency_conditioned_resnet18(pretrained=False, fusion_layer=4, **kwargs):
    model = NoSaliencyConditionedAblation(BasicBlock, [2, 2, 2, 2], fusion_layer=fusion_layer, **kwargs)
    if pretrained:
        raise ValueError
    return model


def nosaliency_conditioned_shared_resnet18(pretrained=False, **kwargs):
    model = NoSaliencyConditionedSharedAblation(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        raise ValueError
    return model


def only_saliency_conditioned_shared_resnet18(pretrained=False, **kwargs):
    model = OnlySaliencySharedAblation(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        raise ValueError
    return model

