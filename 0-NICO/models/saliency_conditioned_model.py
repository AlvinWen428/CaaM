import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .resnet224 import ResNet, BasicBlock, ResNet_Feature, ResNetWithInterFeature
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


class SaliencyConditionedDifferentZetaAblation(TwoInputResNet):
    def __init__(self, block, layers, condition_activation=None, stop_gradient=True, num_classes=1000,
                 select_feature='avgpool'):
        super(SaliencyConditionedDifferentZetaAblation, self).__init__(block, layers, 5,
                                                                       condition_activation, stop_gradient, num_classes)
        self.select_feature = select_feature
        del self.coarse_model
        del self.refine_model

        if self.select_feature == 'avgpool':
            additional_channel = 512 * block.expansion
            self.feature_idx = 4
            self.extractor = self.extract_avgpool
        elif self.select_feature == 'layer2':
            additional_channel = 128
            self.feature_idx = 1
            self.extractor = self.extract_conv_layer
        elif self.select_feature == 'layer3':
            additional_channel = 256
            self.feature_idx = 2
            self.extractor = self.extract_conv_layer
        elif self.select_feature == 'layer4':
            additional_channel = 512
            self.feature_idx = 3
            self.extractor = self.extract_conv_layer
        else:
            raise ValueError

        self.coarse_model = ResNetWithInterFeature(block, layers, num_classes)
        self.refine_model = ResNetFusionModel(block, layers, 5, additional_channel, num_classes)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

    def extract_avgpool(self, feature_tuple):
        return feature_tuple[4]

    def extract_conv_layer(self, feature_tuple):
        layer_output = feature_tuple[self.feature_idx]
        extract_feature = self.global_avg_pool(layer_output).view(layer_output.shape[0], layer_output.shape[1])
        return extract_feature

    def forward(self, x, processed_x, train_mode=False):
        coarse_output, inter = self.coarse_model(processed_x)
        extract_feature = self.extractor(inter)

        if self.stop_gradient:
            input_condition = extract_feature.detach()
        else:
            input_condition = extract_feature

        # use no activation on the intermediate features
        # input_condition = self.activation_on_condition(input_condition)

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


class InverseConditionedSharedAblation(TwoInputSharedResNet):
    def __init__(self, block, layers, condition_activation=None, stop_gradient=True, num_classes=1000):
        super(InverseConditionedSharedAblation, self).__init__(block, layers, condition_activation,
                                                                  stop_gradient, num_classes)

    def forward(self, x, processed_x, train_mode=False):
        coarse_feature = self.feature_extractor(x)
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


class NoPrimeLossSharedAblation(TwoInputSharedResNet):
    def __init__(self, block, layers, condition_activation=None, stop_gradient=True, num_classes=1000):
        super(NoPrimeLossSharedAblation, self).__init__(block, layers, condition_activation,
                                                         stop_gradient, num_classes)

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

        # output a detached coarse output to avoid the coarse loss back-prop to the model
        coarse_output_no_loss = coarse_output.detach()
        if train_mode:
            return refine_output, coarse_output_no_loss
        else:
            return refine_output


class NoConcatSharedAblation(TwoInputSharedResNet):
    def __init__(self, block, layers, condition_activation=None, stop_gradient=True, num_classes=1000):
        super(NoConcatSharedAblation, self).__init__(block, layers, condition_activation,
                                                     stop_gradient, num_classes)
        del self.refine_classifier
        self.refine_classifier = nn.Linear(512 * block.expansion, num_classes)

    def forward(self, x, processed_x, train_mode=False):
        coarse_feature = self.feature_extractor(processed_x)
        coarse_output = self.coarse_classifier(coarse_feature)

        refine_feature = self.feature_extractor(x)
        refine_output = self.refine_classifier(refine_feature)

        if train_mode:
            return refine_output, coarse_output
        else:
            return refine_output


class MixTwoInputSharedEnsemble(ResNet):
    def __init__(self, block, layers, num_classes=1000):
        super(MixTwoInputSharedEnsemble, self).__init__(block, layers, num_classes)

    def original_forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

    def forward(self, x, processed_x=None, train_mode=False):
        if train_mode and processed_x is None:
            output = self.original_forward(x)
        elif not train_mode and processed_x is not None:
            output_raw = self.original_forward(x)
            output_processed = self.original_forward(processed_x)
            output = (output_raw + output_processed) / 2
        else:
            raise ValueError
        return output


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


def saliency_conditioned_inverse_shared_resnet18(pretrained=False, **kwargs):
    model = InverseConditionedSharedAblation(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        raise ValueError
    return model


def saliency_conditioned_different_zeta_resnet18(pretrained=False, **kwargs):
    model = SaliencyConditionedDifferentZetaAblation(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        raise ValueError
    return model


def saliency_conditioned_mix_inputs_ensemble_resnet18(pretrained=False, **kwargs):
    model = MixTwoInputSharedEnsemble(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        raise ValueError
    return model


def saliency_conditioned_no_prime_loss_shared_resnet18(pretrained=False, **kwargs):
    model = NoPrimeLossSharedAblation(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        raise ValueError
    return model


def saliency_conditioned_no_concat_shared_resnet18(pretrained=False, **kwargs):
    model = NoConcatSharedAblation(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        raise ValueError
    return model
