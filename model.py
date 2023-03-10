import torch.nn as nn
import torch.nn.functional as F


class Block_test(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.norm = nn.InstanceNorm2d(dim, affine=True)
        self.sigmoid = nn.Sigmoid()
        dw_channel = 3 * dim
        self.conv1 = nn.Conv2d(in_channels=dim, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        inter_channels = int(dim // 4)
        self.att1 = nn.Sequential(
            nn.Conv2d(dim, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.GELU(),
            nn.Conv2d(inter_channels, dim, kernel_size=1, stride=1, padding=0),
        )

        self.att2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.GELU(),
            nn.Conv2d(inter_channels, dim, kernel_size=1, stride=1, padding=0),
        )
        self.conv3 = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)

    def forward(self, x):
        x1 = self.norm(x)
        x1 = self.conv1(x1)
        x1 = self.conv2(x1)
        e, f, g = x1.chunk(3, dim=1)
        # dim = (x1.shape[1]) // 3
        # e = x1[:, 0:dim, :, :]
        # f = x1[:, dim: dim * 2, :, :]
        # g = x1[:, dim * 2: dim * 3, :, :]
        e = self.att1(e)
        f = self.att2(f)
        ef = e + f
        ef = self.sigmoid(ef)
        efg = ef * g
        efg = self.conv3(efg)
        out = efg + x
        return out


class FNAFNet(nn.Module):

    def __init__(self, img_channel=3, width=32, middle_blk_num=1, enc_blk_nums=None, dec_blk_nums=None):
        super().__init__()

        if dec_blk_nums is None:
            dec_blk_nums = [1, 1, 1, 1]
        if enc_blk_nums is None:
            enc_blk_nums = [1, 1, 1, 1]

        NAFBlock = Block_test

        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1,
                               groups=1,
                               bias=True)
        self.ending = nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1,
                                groups=1,
                                bias=True)
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        chan = width

        for i in range(len(enc_blk_nums)):
            self.encoders.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in
                      range(enc_blk_nums[i])]
                )
            )
            self.downs.append(
                nn.Conv2d(chan, 2 * chan, 2, 2)
            )
            chan = chan * 2

        self.middle_blks = \
            nn.Sequential(
                *[NAFBlock(chan) for _ in
                  range(middle_blk_num)]
            )

        for j in range(len(dec_blk_nums)):
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in
                      range(dec_blk_nums[j])]
                )
            )

        self.padder_size = 2 ** len(self.encoders)

    def forward(self, inp):
        B, C, H, W = inp.shape

        inp = self.check_image_size(inp)

        x = self.intro(inp)

        encs = []

        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)

        x = self.ending(x)
        output = x + inp
        return output[:, :, :H, :W]

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x