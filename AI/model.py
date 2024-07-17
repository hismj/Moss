import torch
import torch.nn as nn


#ShuffleUnit
def channel_shuffle(x, groups):
    # 获得特征图的所以维度的数据
    batch_size, num_channels, height, width = x.shape
    # 对特征通道进行分组
    channels_per_group = num_channels // groups
    # reshape新增特征图的维度
    x = x.view(batch_size, groups, channels_per_group, height, width)
    # 通道混洗(将输入张量的指定维度进行交换)
    x = torch.transpose(x, 1, 2).contiguous()
    # reshape降低特征图的维度
    x = x.view(batch_size, -1, height, width)
    return x


class ShuffleUnit(nn.Module):
    def __init__(self, input_c: int, output_c: int, stride: int):
        super(ShuffleUnit, self).__init__()
        # 步长必须在1和2之间
        if stride not in [1, 2]:
            raise ValueError("illegal stride value.")
        self.stride = stride

        # 输出通道必须能二被等分
        assert output_c % 2 == 0
        branch_features = output_c // 2

        # 当stride为1时，input_channel是branch_features的两倍
        # '<<' 是位运算，可理解为计算×2的快速方法
        assert (self.stride != 1) or (input_c == branch_features << 1)

        # 捷径分支
        if self.stride == 2:
            # 进行下采样:3×3深度卷积+1×1卷积
            self.branch1 = nn.Sequential(
                self.depthwise_conv(input_c, input_c, kernel_s=3, stride=self.stride, padding=1),
                nn.BatchNorm2d(input_c),
                nn.Conv2d(input_c, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(branch_features),
                nn.ReLU(inplace=True)
            )
        else:
            # 不进行下采样:保持原状
            self.branch1 = nn.Sequential()

        # 主干分支
        self.branch2 = nn.Sequential(
            # 1×1卷积+3×3深度卷积+1×1卷积
            nn.Conv2d(input_c if self.stride > 1 else branch_features, branch_features, kernel_size=1,
                      stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
            self.depthwise_conv(branch_features, branch_features, kernel_s=3, stride=self.stride, padding=1),
            nn.BatchNorm2d(branch_features),
            nn.Conv2d(branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True)
        )

    # 深度卷积
    @staticmethod
    def depthwise_conv(input_c, output_c, kernel_s, stride, padding, bias=False):
        return nn.Conv2d(in_channels=input_c, out_channels=output_c, kernel_size=kernel_s,
                         stride=stride, padding=padding, bias=bias, groups=input_c)

    def forward(self, x):
        if self.stride == 1:
            # 通道切分
            x1, x2 = x.chunk(2, dim=1)
            # 主干分支和捷径分支拼接
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            # 通道切分被移除
            # 主干分支和捷径分支拼接
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)
        # 通道混洗
        out = channel_shuffle(out, 2)
        return out

# ShuffleNetV2定义
class ShuffleNetV2(nn.Module):
    def __init__(self, stages_repeats, stages_out_channels, num_classes=2, ShuffleUnit=ShuffleUnit):
        super(ShuffleNetV2, self).__init__()

        if len(stages_repeats) != 3:
            raise ValueError("expected stages_repeats as list of 3 positive ints")
        if len(stages_out_channels) != 5:
            raise ValueError("expected stages_out_channels as list of 5 positive ints")
        self._stage_out_channels = stages_out_channels

        input_channels = 3
        output_channels = self._stage_out_channels[0]

        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)
        )
        input_channels = output_channels
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.stage2: nn.Sequential
        self.stage3: nn.Sequential
        self.stage4: nn.Sequential

        stage_names = ["stage{}".format(i) for i in [2, 3, 4]]
        for name, repeats, output_channels in zip(stage_names, stages_repeats,
                                                  self._stage_out_channels[1:]):
            seq = [ShuffleUnit(input_channels, output_channels, 2)]
            for i in range(repeats - 1):
                seq.append(ShuffleUnit(output_channels, output_channels, 1))
            setattr(self, name, nn.Sequential(*seq))
            input_channels = output_channels
        output_channels = self._stage_out_channels[-1]
        self.conv5 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)
        )
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(output_channels, num_classes)
        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.conv5(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# ShuffleUnit定义

# 创建ShuffleNetV2模型函数
def shufflenet_v2_x2_0(num_classes=2):
    model = ShuffleNetV2(stages_repeats=[4, 8, 4],
                         stages_out_channels=[24, 244, 488, 976, 2048],
                         num_classes=num_classes)
    return model
