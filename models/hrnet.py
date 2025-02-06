import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from monai.networks.blocks import SEResNetBottleneck
from monai.networks.nets import HighResBlock



class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = self._conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = self._conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=0.1)
        self.downsample = downsample
        self.stride = stride
    
    def _conv3x3(self, in_planes, out_planes, stride=1):
        return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu(out)

        return out

class HighResModule(nn.Module):
    """
    HighResModule contains multiple branches with HighResBlock and fusion layers.
    """

    def __init__(
        self, num_branches, in_channels, out_channels, n_blocks, spatial_dims=2, multi_scale_output=True
    ):
        super(HighResModule, self).__init__()
        self.num_branches = num_branches
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_blocks = n_blocks
        self.spatial_dims = spatial_dims
        self.multi_scale_output = multi_scale_output
        self.relu = nn.ReLU()
        self._check_branches()
        self.branches = self._make_branches()
        self.fuse_layers = self._make_fuse_layers()

    def _check_branches(self):
        if self.num_branches != len(self.in_channels):
            raise ValueError(
                f"num_branches({self.num_branches}) does not match length of in_channels({len(self.in_channels)})"
            )
        if self.num_branches != len(self.out_channels):
            raise ValueError(
                f"num_branches({self.num_branches}) does not match length of out_channels({len(self.out_channels)})"
            )

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
    
        # downsample = None
        # stride = 1 

        # num_channels = self.out_channels

        # if stride != 1 or \
        #    self.in_channels[branch_index] != num_channels[branch_index]:
        #     downsample = nn.Sequential(
        #         nn.Conv2d(
        #             self.in_channels[branch_index],
        #             num_channels[branch_index] ,
        #             kernel_size=1, stride=stride, bias=False
        #         ),
        #         nn.BatchNorm2d(
        #             num_channels[branch_index],
        #             momentum=0.1
        #         ),
        #     )

        # layers = []
        # layers.append(
        #     BasicBlock(
        #         self.in_channels[branch_index],
        #         num_channels[branch_index],
        #         stride,
        #         downsample
        #     )
        # )
        # self.in_channels[branch_index] = \
        #     num_channels[branch_index] 
        # for i in range(1, self.n_blocks):
        #     layers.append(
        #         BasicBlock(
        #             self.in_channels[branch_index],
        #             num_channels[branch_index]
        #         )
        #     )

        # return nn.Sequential(*layers)

    def _make_branches(self):
        return nn.ModuleList([self._make_one_branch(i) for i in range(self.num_branches)])

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None

        fuse_layers = []
        for i in range(self.num_branches if self.multi_scale_output else 1):
            layer = []
            for j in range(self.num_branches):
                if j > i:
                    layer.append(
                        nn.Sequential(
                            nn.Conv2d(
                                in_channels=self.in_channels[j],
                                out_channels=self.out_channels[i],
                                kernel_size=1,
                                stride=1,
                                bias=False,
                            ),
                            nn.BatchNorm2d(self.out_channels[i], momentum=0.1),
                            nn.Upsample(scale_factor=2 ** (j - i), mode="nearest"),
                        )
                    )
                elif j < i:
                    conv3x3s = []
                    in_channels = self.in_channels[j]
                    for k in range(i - j):
                        out_channels = self.out_channels[i] if k == i - j - 1 else in_channels
                        conv3x3s.append(
                            nn.Sequential(
                                nn.Conv2d(
                                    in_channels=in_channels,
                                    out_channels=out_channels,
                                    kernel_size=3,
                                    stride=2,
                                    padding=1,
                                    bias=False
                                ),
                                nn.BatchNorm2d(out_channels, momentum=0.1),
                                nn.ReLU(inplace=True),
                            )
                        )
                        in_channels = out_channels
                    layer.append(nn.Sequential(*conv3x3s))
                else:
                    layer.append(None)
            fuse_layers.append(nn.ModuleList(layer))

        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        return self.out_channels

    def forward(self, x):

        x = [self.branches[i](x[i]) for i in range(self.num_branches)]

        if self.num_branches == 1:
            return [x[0]]

        x_fuse = []
        for i, fuse_layer in enumerate(self.fuse_layers):
            y = x[0] if i == 0 else fuse_layer[0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                elif fuse_layer[j] is not None:
                    y = y + fuse_layer[j](x[j])
            x_fuse.append(self.relu(y))

        return x_fuse

class HRNet(nn.Module):
    """
    HRNet implementation for spine X-ray landmark detection.
    """

    def __init__(self, config):
        super(HRNet, self).__init__()

        self.config = config
        self.spatial_dims = config["spatial_dims"]
        self.in_channels = config["in_channels"]
        self.num_classes = config["out_channels"]
        self.channels = config["channels"]
        self.n_blocks = config["n_blocks"]
        self.kernel_size = config.get("kernel_size", 3)
        self.dropout = config.get("dropout", 0.0)
        self.bn_momentum = config.get("bn_momentum", 0.1)
        
        self.expansion = config.get("expansion", 4)  # SE block reduction ratio

        if isinstance(self.n_blocks, int): 
            self.n_blocks = [self.n_blocks] * 4

        # Stage 1
        channels_stage1 = 64  # self.channels[0] * self.expansion
        self.conv1 = nn.Conv2d(self.in_channels, channels_stage1, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels_stage1, momentum=self.bn_momentum)
        self.conv2 = nn.Conv2d(channels_stage1, channels_stage1, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels_stage1, momentum=self.bn_momentum)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(p=self.dropout)

        self.layer1 = self._make_layer(channels_stage1, channels_stage1, self.n_blocks[0])

        # Stage 2
        num_channels = self.channels[:2]
        #num_channels = [c * self.expansion for c in num_channels]
        self.num_branches_2 = len(num_channels)
        self.transition1 = self._make_transition_layer([channels_stage1 * self.expansion], num_channels)
        self.stage2, pre_stage_channels = self._make_stage(1, self.n_blocks[1], self.num_branches_2, num_channels, num_channels)

        # Stage 3
        num_channels = self.channels[:3]
        #num_channels = [c * self.expansion for c in num_channels]
        self.num_branches_3 = len(num_channels)
        self.transition2 = self._make_transition_layer(pre_stage_channels, num_channels)
        self.stage3, pre_stage_channels = self._make_stage(1, self.n_blocks[2], self.num_branches_3, num_channels, num_channels)

        # Stage 4
        num_channels = self.channels
        #num_channels = [c * self.expansion for c in num_channels]
        self.num_branches_4 = len(num_channels)
        self.transition3 = self._make_transition_layer(pre_stage_channels, num_channels)
        self.stage4, pre_stage_channels = self._make_stage(
            1, self.n_blocks[3], self.num_branches_4, num_channels, num_channels, multi_scale_output=False
        )

        # Final layer
        self.final_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=pre_stage_channels[0],
                out_channels=self.num_classes,
                kernel_size=1,
                stride=1,
                padding="same",
            ),
            nn.Dropout(p=self.dropout),
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_uniform_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                init.ones_(m.weight)
                init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                init.zeros_(m.bias)

    def _make_layer(self, inplanes, planes, blocks, stride=1):
        downsample = None
        reduction = 16

        if stride != 1 or inplanes != planes * self.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * self.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * self.expansion),
            )

        layers = []
        layers.append(SEResNetBottleneck(
            spatial_dims=self.spatial_dims,
            inplanes=inplanes,
            planes=planes,  
            groups=1,
            reduction=reduction,
            downsample=downsample,
        ))
        
        inplanes = planes * self.expansion
        for i in range(1, blocks):
            layers.append(SEResNetBottleneck(
                spatial_dims=self.spatial_dims,
                inplanes=inplanes,
                planes=planes,  
                groups=1,
                reduction=reduction, 
                downsample=None,
            ))

        return nn.Sequential(*layers)
        

    def _make_transition_layer(self, num_channels_pre_layer, num_channels_cur_layer):
        transition_layers = []
        num_branches_pre = len(num_channels_pre_layer)
        num_branches_cur = len(num_channels_cur_layer)

        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(
                        nn.Sequential(
                            nn.Conv2d(
                                num_channels_pre_layer[i],
                                num_channels_cur_layer[i],
                                kernel_size=3,
                                stride=1,
                                padding=1,
                                bias=False,
                            ),
                            nn.BatchNorm2d(num_channels_cur_layer[i], momentum=self.bn_momentum),
                            nn.ReLU(inplace=True),
                            nn.Dropout(p=self.dropout),
                        )
                    )
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = nn.ModuleList()
                inchannels = num_channels_pre_layer[-1]
                for j in range(i + 1 - num_branches_pre):
                    outchannels = num_channels_cur_layer[i] if j == i - num_branches_pre else inchannels
                    conv3x3s = nn.Sequential(
                        *list(conv3x3s.children()),
                        nn.Conv2d(
                            inchannels, outchannels, kernel_size=3, stride=2, padding=1, bias=False
                        ),
                        nn.BatchNorm2d(outchannels, momentum=self.bn_momentum),
                        nn.ReLU(inplace=True),
                        nn.Dropout(p=self.dropout),
                        )
                
                    inchannels = outchannels
                transition_layers.append(nn.Sequential(*conv3x3s))

        return nn.ModuleList(transition_layers)
    

    def _make_stage(self, num_modules, num_blocks, num_branches, num_channels, num_inchannels, multi_scale_output=True):
        modules = []
        for i in range(num_modules):
            reset_multi_scale_output = multi_scale_output if i < num_modules - 1 else False
            modules.append(
                HighResModule(
                    num_branches=num_branches,
                    in_channels=num_inchannels,
                    out_channels=num_channels,
                    n_blocks=num_blocks,
                    spatial_dims=2,
                    multi_scale_output=True,
                )
            )
            num_inchannels = modules[-1].get_num_inchannels()

        return nn.Sequential(*modules), num_inchannels

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        # x = self.bottleneck1(x)
        x = self.layer1(x)

        x = F.interpolate(x, scale_factor=4, mode="bilinear", align_corners=True)   

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
        return x