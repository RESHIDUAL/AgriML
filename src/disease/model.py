"""
Model architecture for the leaf disease classification pipeline.
Supports MobileNetV3-Large, EfficientNet-B0, and ResNet-34 backbones
with utilities for head swapping, checkpoint loading, and parameter freezing.
"""

import torch
import torch.nn as nn
import torchvision.models as models


def build_model(
    num_classes: int,
    backbone: str = "mobilenet_v3_large",
    pretrained_imagenet: bool = True,
) -> nn.Module:
    if backbone == "mobilenet_v3_large":
        weights = (
            models.MobileNet_V3_Large_Weights.IMAGENET1K_V2
            if pretrained_imagenet
            else None
        )
        model = models.mobilenet_v3_large(weights=weights)
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, num_classes)

    elif backbone == "efficientnet_b0":
        weights = (
            models.EfficientNet_B0_Weights.IMAGENET1K_V1
            if pretrained_imagenet
            else None
        )
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)

    elif backbone == "resnet34":
        weights = (
            models.ResNet34_Weights.IMAGENET1K_V1
            if pretrained_imagenet
            else None
        )
        model = models.resnet34(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features, num_classes),
        )

    else:
        raise ValueError(
            f"Unsupported backbone '{backbone}'. "
            "Choose from: 'mobilenet_v3_large', 'efficientnet_b0', 'resnet34'."
        )

    return model


def load_stage1_and_swap_head(
    checkpoint_path: str,
    new_num_classes: int,
    backbone: str = "mobilenet_v3_large",
) -> nn.Module:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    old_num_classes = checkpoint["num_classes"]

    model = build_model(old_num_classes, backbone, pretrained_imagenet=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    swap_head(model, new_num_classes, backbone)
    return model


def swap_head(
    model: nn.Module,
    new_num_classes: int,
    backbone: str = "mobilenet_v3_large",
) -> nn.Module:
    if backbone == "mobilenet_v3_large":
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, new_num_classes)

    elif backbone == "efficientnet_b0":
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, new_num_classes)

    elif backbone == "resnet34":
        in_features = (
            model.fc[-1].in_features
            if isinstance(model.fc, nn.Sequential)
            else model.fc.in_features
        )
        model.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features, new_num_classes),
        )
    else:
        raise ValueError(
            f"Unsupported backbone '{backbone}'. "
            "Choose from: 'mobilenet_v3_large', 'efficientnet_b0', 'resnet34'."
        )

    return model


def get_num_params(model: nn.Module) -> tuple[int, int]:
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    return total_params, trainable_params


def freeze_backbone(
    model: nn.Module,
    backbone: str = "mobilenet_v3_large",
) -> None:
    for param in model.parameters():
        param.requires_grad = False

    if backbone == "mobilenet_v3_large":
        for param in model.classifier[3].parameters():
            param.requires_grad = True
    elif backbone == "efficientnet_b0":
        for param in model.classifier[1].parameters():
            param.requires_grad = True
    elif backbone == "resnet34":
        for param in model.fc.parameters():
            param.requires_grad = True


def unfreeze_all(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = True


if __name__ == "__main__":
    net = build_model(num_classes=38, backbone="mobilenet_v3_large", pretrained_imagenet=False)
    x = torch.randn(2, 3, 224, 224)
    out = net(x)
    print("Forward pass successful! Output shape:", out.shape)
