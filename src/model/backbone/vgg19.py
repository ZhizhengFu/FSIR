import torch
import torchvision
import torch.nn as nn


# modify from https://github.com/eezkni/FDL/blob/main/FDL_pytorch/models.py
class VGG(nn.Module):
    def __init__(self, requires_grad=False, pretrained=True):
        super().__init__()

        vgg_pretrained_features = torchvision.models.vgg19(
            pretrained=pretrained
        ).features

        self.stage1 = torch.nn.Sequential()
        self.stage2 = torch.nn.Sequential()
        self.stage3 = torch.nn.Sequential()
        self.stage4 = torch.nn.Sequential()
        self.stage5 = torch.nn.Sequential()

        for x in range(0, 4):
            self.stage1.add_module(str(x), vgg_pretrained_features[x])  # type: ignore
        for x in range(4, 9):
            self.stage2.add_module(str(x), vgg_pretrained_features[x])  # type: ignore
        for x in range(9, 18):
            self.stage3.add_module(str(x), vgg_pretrained_features[x])  # type: ignore
        for x in range(18, 27):
            self.stage4.add_module(str(x), vgg_pretrained_features[x])  # type: ignore
        for x in range(27, 36):
            self.stage5.add_module(str(x), vgg_pretrained_features[x])  # type: ignore
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1)
        )

        self.chns = [64, 128, 256, 512, 512]

    def get_features(self, x):
        # normalize the data
        h = (x - self.mean) / self.std

        h = self.stage1(h)
        h_relu1_2 = h

        h = self.stage2(h)
        h_relu2_2 = h

        h = self.stage3(h)
        h_relu3_3 = h

        h = self.stage4(h)
        h_relu4_3 = h

        h = self.stage5(h)
        h_relu5_3 = h

        # get the features of each layer
        outs = [h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3]

        return outs

    def forward(self, x):
        feats_x = self.get_features(x)
        return feats_x
