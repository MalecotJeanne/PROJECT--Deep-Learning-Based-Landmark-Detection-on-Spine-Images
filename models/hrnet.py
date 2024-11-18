"""
    define the HRNet model
    based on the implementation of Team iFLYTEK for the AASCE2019 
    (https://github.com/YJY-CV/Spine/blob/master/lib/models/pose_hrnet.py)
"""

import torch
import torch.nn as nn

from monai.networks.blocks import SEResNetBottleneck
from monai.networks.nets import HighResBlock


class HRNet(nn.Module):
    """
    My implementation of the HRNet model, similar to the one used by Team iFLYTEK for the AASCE2019
    but using some blocks from MONAI instead of creating them from scratch.
    """

    def __init__(self):
        super(HRNet, self).__init__()

        # TODO: remove hard-coded values and make them configurable

        # -------------------------------------------------------
        # stage 1
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=0)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64, momentum=0)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

        # using bottleneck implementation from MONAI -> the expension  is always 4 in this class.
        self.bottleneck1 = SEResNetBottleneck(
            spatial_dims=2,
            inplanes=64,
            planes=32 // 4,  # TODO: define in a config file
            groups=1,
            reduction=2,  # 2 is default value in SEBlock
            downsample=None,
        )

        # -------------------------------------------------------
        # stage 2
        num_channels = [32, 64]  # eventually add expansion, if BottleNeck?
        self.num_branches_2 = len(num_channels)

        # no expansion with a Basic Block
        self.transition1 = self._make_transition_layer([32], num_channels)
        self.stage2, pre_stage_channels = self._make_stage(
            1, self.num_branches_2, num_channels, num_channels
        )  # two times num_channels because no expansion

        # -------------------------------------------------------
        # stage 3
        num_channels = [32, 64, 128]
        self.num_branches_3 = len(num_channels)

        self.transition2 = self._make_transition_layer(pre_stage_channels, num_channels)
        self.stage3, pre_stage_channels = self._make_stage(
            1, self.num_branches_3, num_channels, num_channels
        )

        # -------------------------------------------------------
        # stage 4
        num_channels = [32, 64, 128, 256]
        self.num_branches_4 = len(num_channels)

        self.transition3 = self._make_transition_layer(pre_stage_channels, num_channels)
        self.stage4, pre_stage_channels = self._make_stage(
            1,
            self.num_branches_4,
            num_channels,
            num_channels,
            multi_scale_output=False,
        )

        # -------------------------------------------------------
        # final layer
        self.final_layer = nn.Conv2d(
            in_channels=pre_stage_channels[0],
            out_channels=68,  # nb of landmarks
            kernel_size=1,
            stride=1,
            padding="same",
        )

    def _make_transition_layer(self, num_channels_pre_layer, num_channels_cur_layer):
        """
        Create the transition layer between two stages of the HRNet model, to ensure e that the output channels of the previous stage match the input channels required by the next stage.
        """
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if (
                i < num_branches_pre
            ):  # the current branch is existing in the previous stage
                if (
                    num_channels_cur_layer[i] != num_channels_pre_layer[i]
                ):  # we apply a convolution if the number of channels is different, to make them match.
                    transition_layers.append(
                        nn.Sequential(
                            nn.Conv2d(
                                num_channels_pre_layer[i],
                                num_channels_cur_layer[i],
                                3,
                                1,
                                1,
                                bias=False,
                            ),
                            nn.BatchNorm2d(num_channels_cur_layer[i]),
                            nn.ReLU(inplace=True),
                        )
                    )
                else:  # no transition needed (existing branch with the same number of channels)
                    transition_layers.append(None)
            else:  # the current branch needs to be created
                conv3x3s = []
                inchannels = num_channels_pre_layer[
                    -1
                ]  # number of channel in the last branch of the previous stage
                for j in range(i + 1 - num_branches_pre):
                    outchannels = (
                        num_channels_cur_layer[i]
                        if j == i - num_branches_pre
                        else inchannels
                    )
                    conv3x3s.append(
                        nn.Sequential(
                            nn.Conv2d(inchannels, outchannels, 3, 2, 1, bias=False),
                            nn.BatchNorm2d(outchannels),
                            nn.ReLU(inplace=True),
                        )
                    )
                transition_layers.append(nn.Sequential(*conv3x3s))

        return nn.ModuleList(transition_layers)

    def _make_stage(
        self,
        num_modules,
        num_branches,
        num_channels,
        num_inchannels,
        multi_scale_output=True,
    ):
        """
        Create a stage of the HRNet model.
        """
        modules = []
        for i in range(num_modules):
            # multi_scale_output is only used last module
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True

            modules.append(
                HighResModule(
                    num_branches=num_branches,
                    in_channels=num_inchannels,
                    out_channels=num_channels,
                    spatial_dims=2,
                    multi_scale_output=reset_multi_scale_output,
                )
            )
            num_inchannels = modules[-1].get_num_inchannels()

        return nn.Sequential(*modules), num_inchannels

    def forward(self, x):
        """
        Forward pass of the HRNet model.
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.bottleneck1(x)

        x_list = []
        for i in range(self.num_branches_2):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)

        y_list = self.stage2(x_list)

        x_list = []
        for i in range(self.num_branches_3):
            if self.transition2[i] is not None:
                x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])

        y_list = self.stage3(x_list)

        x_list = []
        for i in range(self.num_branches_4):
            if self.transition3[i] is not None:
                x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage4(x_list)

        x = self.final_layer(y_list[0])
        
        #convert into a probability map, flatten (last two dims) it and apply softmax
        H,W = x.size()[-2:]
        x = x.view(x.size(0), x.size(1), -1)
        x = self.softmax(x)
        x = x.view(x.size(0), x.size(1), H, W)

        return x

        


class HighResModule(nn.Module):
    """
    HighResModule is a module that contains multiple branches, each with its own HighResBlock.
    """

    def __init__(
        self,
        num_branches,
        in_channels,
        out_channels,
        spatial_dims=2,
        multi_scale_output=True,
    ):

        super(HighResModule, self).__init__()
        self.num_branches = num_branches
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.spatial_dims = spatial_dims
        self.multi_scale_output = multi_scale_output
        self.relu = nn.ReLU()

        self.branches = nn.ModuleList()
        self.fuse_layers = nn.ModuleList()

        self._check_branches()
        self.branches = self._make_branches()
        self.fuse_layers = self._make_fuse_layers()

    def _check_branches(self):

        if self.num_branches != len(self.in_channels):
            raise ValueError(
                "num_branches({}) is not equal to the length of num_channels({})".format(
                    self.num_branches, len(self.in_channels)
                )
            )  # TODO : add a logger

        if self.num_branches != len(self.out_channels):
            raise ValueError(
                "num_branches({}) is not equal to the length of num_inchannels({})".format(
                    self.num_branches, len(self.out_channels)
                )
            )  # TODO : add a logger

    def _make_one_branch(self, branch_index):

        return HighResBlock(
            spatial_dims=self.spatial_dims,
            in_channels=self.in_channels[branch_index],
            out_channels=self.out_channels[branch_index],
            dilation=1,
            acti_type="relu",
            norm_type="batch",
            channel_matching="project",  # in the case of an expansion, the channel_matching should be "project" (convolution, with kernel size 1)
        )

    def _make_branches(self):

        branches = []
        for i in range(self.num_branches):
            branches.append(self._make_one_branch(i))

        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None
        else:
            fuse_layers = []
            for i in range(self.num_branches if self.multi_scale_output else 1):
                layer = []
                for j in range(self.num_branches):
                    if j > i:  # Upsample to match the current branch's resolution
                        layer.append(
                            nn.Sequential(
                                nn.Conv2d(
                                    in_channels=self.in_channels[j],
                                    out_channels=self.out_channels[i],
                                    kernel_size=1,
                                    stride=1,
                                    bias=False,
                                ),
                                nn.BatchNorm2d(self.out_channels[i]),
                                nn.Upsample(scale_factor=2 ** (j - i), mode="nearest"),
                            )
                        )
                    elif j < i:  # Downsample to match the current branch's resolution
                        conv3x3s = []
                        in_channels = self.in_channels[j]
                        for k in range(i - j):  # Add more downsampling layers
                            out_channels = (
                                self.out_channels[i] if k == i - j - 1 else in_channels
                            )
                            conv3x3s.append(
                                nn.Sequential(
                                    nn.Conv2d(
                                        in_channels=in_channels,
                                        out_channels=out_channels,
                                        kernel_size=3,
                                        stride=2,  # downsample by stride of 2
                                        padding=1,
                                        bias=False,
                                    ),
                                    nn.BatchNorm2d(out_channels),
                                )
                            )
                            in_channels = out_channels
                        layer.append(nn.Sequential(*conv3x3s))
                    else:
                        layer.append(None)
                fuse_layers.append(nn.ModuleList(layer))
            return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        """
        Get the number of input channels.
        """
        return self.in_channels

    def forward(self, x):
        """
        Forward pass of the HighResModule.
        """
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x_fuse = []

        for i, fuse_layer in enumerate(self.fuse_layers):
            y = x[0] if i == 0 else fuse_layer[0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                else:
                    y = y + fuse_layer[j](x[j])

            x_fuse.append(self.relu(y))

        return x_fuse
