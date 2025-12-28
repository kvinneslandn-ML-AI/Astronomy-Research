# ---------- Small helper block ----------
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


# ---------- Encoder: 4-level downsampling ----------
class Encoder4L(nn.Module):
    """
    Input:  (B, 1, 125, 125)
    Output: bottleneck z: (B, 256, 7, 7)
    Also returns skip features for U-Net-style decoding.
    """
    def __init__(self, in_channels=1, base_channels=32):
        super().__init__()
        c1 = base_channels          # 32
        c2 = base_channels * 2      # 64
        c3 = base_channels * 4      # 128
        c4 = base_channels * 8      # 256

        # Downsample convs: stride=2, kernel=4, pad=1  -> 125→62→31→15→7
        self.down1 = nn.Conv2d(in_channels, c1, kernel_size=4, stride=2, padding=1)  # 125→62
        self.down2 = nn.Conv2d(c1, c2, kernel_size=4, stride=2, padding=1)           # 62→31
        self.down3 = nn.Conv2d(c2, c3, kernel_size=4, stride=2, padding=1)           # 31→15
        self.down4 = nn.Conv2d(c3, c4, kernel_size=4, stride=2, padding=1)           # 15→7

        self.conv1 = ConvBlock(c1, c1)
        self.conv2 = ConvBlock(c2, c2)
        self.conv3 = ConvBlock(c3, c3)
        self.conv4 = ConvBlock(c4, c4)

    def forward(self, x):
        # x: (B,1,125,125)
        x1 = self.down1(x)   # (B,32,62,62)
        x1 = self.conv1(x1)

        x2 = self.down2(x1)  # (B,64,31,31)
        x2 = self.conv2(x2)

        x3 = self.down3(x2)  # (B,128,15,15)
        x3 = self.conv3(x3)

        x4 = self.down4(x3)  # (B,256,7,7)
        x4 = self.conv4(x4)  # bottleneck

        # return bottleneck + skips
        return x4, (x3, x2, x1)


# ---------- Decoder: upsample with skips ----------
class Decoder4L(nn.Module):
    """
    Input:  bottleneck z: (B, 256, 7, 7)
    Output: logits: (B, 1, 125, 125)
    Uses skip connections from encoder.
    """
    def __init__(self, out_channels=1, base_channels=32):
        super().__init__()
        c1 = base_channels          # 32
        c2 = base_channels * 2      # 64
        c3 = base_channels * 4      # 128
        c4 = base_channels * 8      # 256

        # Transposed convs chosen so:
        # 7→15 (op=1), 15→31 (op=1), 31→62 (op=0), 62→125 (op=1)
        self.up1 = nn.ConvTranspose2d(c4, c3, kernel_size=4, stride=2, padding=1, output_padding=1)  # 7→15
        self.up2 = nn.ConvTranspose2d(c3, c2, kernel_size=4, stride=2, padding=1, output_padding=1)  # 15→31
        self.up3 = nn.ConvTranspose2d(c2, c1, kernel_size=4, stride=2, padding=1, output_padding=0)  # 31→62
        self.up4 = nn.ConvTranspose2d(c1, c1, kernel_size=4, stride=2, padding=1, output_padding=1)  # 62→125

        # after concat with skip, double the channels
        self.dec3 = ConvBlock(c3 + c3, c3)
        self.dec2 = ConvBlock(c2 + c2, c2)
        self.dec1 = ConvBlock(c1 + c1, c1)

        # final 1×1 conv to produce logits
        self.out_conv = nn.Conv2d(c1, out_channels, kernel_size=1)

    def forward(self, z, skips):
        x3, x2, x1 = skips  # shapes: (B,128,15,15), (B,64,31,31), (B,32,62,62)

        d3 = self.up1(z)                 # (B,128,15,15)
        d3 = torch.cat([d3, x3], dim=1)  # (B,256,15,15)
        d3 = self.dec3(d3)               # (B,128,15,15)

        d2 = self.up2(d3)                # (B,64,31,31)
        d2 = torch.cat([d2, x2], dim=1)  # (B,128,31,31)
        d2 = self.dec2(d2)               # (B,64,31,31)

        d1 = self.up3(d2)                # (B,32,62,62)
        d1 = torch.cat([d1, x1], dim=1)  # (B,64,62,62)
        d1 = self.dec1(d1)               # (B,32,62,62)

        out = self.up4(d1)               # (B,32,125,125)
        logits = self.out_conv(out)      # (B,1,125,125)
        return logits

class AE(nn.Module):
    """
    4-level downsampling autoencoder for arc segmentation.
    Forward:
        logits, loss (if target provided)
    """
    def __init__(self, base_channels=32, pos_weight=4.0):
        super().__init__()
        self.encoder = Encoder4L(in_channels=1, base_channels=base_channels)
        self.decoder = Decoder4L(out_channels=1, base_channels=base_channels)
        self.register_buffer("pos_weight", torch.tensor([pos_weight], dtype=torch.float32))

    def forward(self, noisy_input, arc_target=None):
        # noisy_input: (B,1,125,125)
        z, skips = self.encoder(noisy_input)
        logits = self.decoder(z, skips)

        if arc_target is not None:
            loss = F.binary_cross_entropy_with_logits(
                logits,
                arc_target,
                pos_weight=self.pos_weight
            )
            return logits, loss

        return logits


class LensRadiusNet(nn.Module):
    def __init__(self, encoder, hidden=256):
        super().__init__()
        self.encoder = encoder  # Encoder4L

        z_dim = 256 * 7 * 7
        self.shared = nn.Sequential(
            nn.Linear(z_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, hidden),
            nn.ReLU(inplace=True),
        )
        self.cls_head = nn.Linear(hidden, 1)  # lens logit
        self.rad_head = nn.Linear(hidden, 1)  # radius (scalar)

    def forward(self, x):
        z, _ = self.encoder(x)          # (B,256,7,7)
        z = z.flatten(1)                # (B, 256*7*7 = 12544)
        h = self.shared(z)
        lens_logit = self.cls_head(h).squeeze(1)
        radius     = F.softplus(self.rad_head(h)).squeeze(1)
        return lens_logit, radius
