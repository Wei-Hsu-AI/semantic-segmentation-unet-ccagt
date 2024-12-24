import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        """
        基本的殘差模組
        :param in_channels: 輸入通道數
        :param out_channels: 輸出通道數
        :param stride: 卷積步幅
        :param downsample: 下採樣模組，用於匹配跳接通道數
        """
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x  # 保留輸入，用於殘差跳接
        if self.downsample:  # 如果需要下採樣，調整尺寸和通道數
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity  # 殘差相加
        out = self.relu(out)

        return out


class ResNetBackbone(nn.Module):
    def __init__(self):
        super(ResNetBackbone, self).__init__()
        # 初始卷積層
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # 殘差層
        self.layer1 = self._make_layer(64, 64, 2, stride=1)  # 輸入: bs, 64, 256, 256 -> 輸出: bs, 64, 256, 256
        self.layer2 = self._make_layer(64, 128, 2, stride=2)  # 輸入: bs, 64, 256, 256 -> 輸出: bs, 128, 128, 128
        self.layer3 = self._make_layer(128, 256, 2, stride=2)  # 輸入: bs, 128, 128, 128 -> 輸出: bs, 256, 64, 64
        self.layer4 = self._make_layer(256, 512, 2, stride=2)  # 輸入: bs, 256, 64, 64 -> 輸出: bs, 512, 32, 32

    def _make_layer(self, in_channels, out_channels, blocks, stride):
        """
        構建殘差層
        :param in_channels: 輸入通道數
        :param out_channels: 輸出通道數
        :param blocks: 殘差塊的數量
        :param stride: 卷積步幅
        """
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

        layers = [ResidualBlock(in_channels, out_channels, stride, downsample)]  # 第一個殘差塊
        for _ in range(1, blocks):  # 其餘殘差塊
            layers.append(ResidualBlock(out_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x1 = self.layer1(x)  # bs, 64, 256, 256
        x2 = self.layer2(x1)  # bs, 128, 128, 128
        x3 = self.layer3(x2)  # bs, 256, 64, 64
        x4 = self.layer4(x3)  # bs, 512, 32, 32

        return x1, x2, x3, x4


class UNetDecoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetDecoder, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)  # 上採樣
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)  # 上採樣: 尺寸加倍
        x = torch.cat([x2, x1], dim=1)  # 通道拼接
        x = self.conv(x)  # 卷積處理
        return x


class UNet(nn.Module):
    def __init__(self, num_classes):
        """
        基於 ResNet 的 U-Net 模型
        :param num_classes: 分割類別數
        """
        super(UNet, self).__init__()
        self.encoder = ResNetBackbone()

        self.decoder3 = UNetDecoder(512, 256)  # bs, 256, 64, 64
        self.decoder2 = UNetDecoder(256, 128)  # bs, 128, 128, 128
        self.decoder1 = UNetDecoder(128, 64)  # bs, 64, 256, 256
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)  # 輸出分割結果，bs, num_classes, 256, 256

        self.apply(initialize_weights)

    def forward(self, x):
        """
        前向傳播
        :param x: 輸入圖片 [bs, 3, 256, 256]
        :return: 分割結果 [bs, num_classes, 256, 256]
        """
        enc1, enc2, enc3, enc4 = self.encoder(x)
        dec3 = self.decoder3(enc4, enc3)  # bs, 256, 64, 64
        dec2 = self.decoder2(dec3, enc2)  # bs, 128, 128, 128
        dec1 = self.decoder1(dec2, enc1)  # bs, 64, 256, 256
        out = self.final_conv(dec1)  # bs, num_classes, 256, 256
        return out
    
def initialize_weights(module):
    """
    初始化模型權重
    :param module: nn.Module 中的子模塊
    """
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
        # Kaiming Initialization for Conv layers
        torch.nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        if module.bias is not None:
            torch.nn.init.constant_(module.bias, 0)

    elif isinstance(module, nn.BatchNorm2d):
        # BatchNorm Initialization
        torch.nn.init.constant_(module.weight, 1)
        torch.nn.init.constant_(module.bias, 0)

    elif isinstance(module, nn.Linear):
        # Xavier Initialization for Linear layers
        torch.nn.init.xavier_normal_(module.weight)
        if module.bias is not None:
            torch.nn.init.constant_(module.bias, 0)
