# ============================================================
# EfficientNetV2 backbone (supports v2_s / v2_m / v2_l) + 1-channel input
# ============================================================
class EfficientNetV2Backbone(nn.Module):
    """
    Returns a pooled feature vector (B, feat_dim) from EfficientNetV2.
    Supports:
      - "efficientnet_v2_s"
      - "efficientnet_v2_m"
      - "efficientnet_v2_l"
    """
    def __init__(self, name="efficientnet_v2_s", in_channels=1, pretrained=False):
        super().__init__()
        name = name.lower().strip()
        assert name in {"efficientnet_v2_s", "efficientnet_v2_m", "efficientnet_v2_l"}, \
            f"Unknown EfficientNetV2 variant: {name}"

        # ---- build model (optionally pretrained) ----
        # NOTE: torchvision weights may need downloading; if unavailable, we fall back to random init.
        net = None
        if pretrained:
            try:
                if name == "efficientnet_v2_s":
                    net = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.DEFAULT)
                elif name == "efficientnet_v2_m":
                    net = models.efficientnet_v2_m(weights=models.EfficientNet_V2_M_Weights.DEFAULT)
                else:
                    net = models.efficientnet_v2_l(weights=models.EfficientNet_V2_L_Weights.DEFAULT)
            except Exception as e:
                print(f"[warn] pretrained weights unavailable ({e}); using random init instead.")
                pretrained = False

        if net is None:
            if name == "efficientnet_v2_s":
                net = models.efficientnet_v2_s(weights=None)
            elif name == "efficientnet_v2_m":
                net = models.efficientnet_v2_m(weights=None)
            else:
                net = models.efficientnet_v2_l(weights=None)

        # ---- replace first conv to accept 1-channel input ----
        # EfficientNetV2 stem conv lives at net.features[0][0]
        old_conv = net.features[0][0]
        new_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            dilation=old_conv.dilation,
            groups=old_conv.groups,
            bias=(old_conv.bias is not None),
            padding_mode=old_conv.padding_mode,
        )

        # If pretrained and in_channels==1, initialize by averaging RGB weights.
        # Otherwise, Kaiming init.
        if pretrained and in_channels == 1 and old_conv.weight.shape[1] == 3:
            with torch.no_grad():
                new_conv.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))
                if old_conv.bias is not None and new_conv.bias is not None:
                    new_conv.bias.copy_(old_conv.bias)
        else:
            nn.init.kaiming_normal_(new_conv.weight, mode="fan_out", nonlinearity="relu")
            if new_conv.bias is not None:
                nn.init.zeros_(new_conv.bias)

        net.features[0][0] = new_conv

        self.features = net.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # Figure out feat_dim from classifier input
        # net.classifier is (Dropout, Linear(in_features, 1000))
        self.feat_dim = net.classifier[1].in_features

    def forward(self, x):
        x = self.features(x)       # (B,C,h,w)
        x = self.pool(x)           # (B,C,1,1)
        return x.flatten(1)        # (B,C)

class LensRadiusEffNetV2(nn.Module):
    """
    Multitask heads with separate trunks:
      - lens_logit: (B,) raw logit for BCEWithLogitsLoss
      - r_pred:     (B,) radius prediction
      - ser_pred:   (B,)
      - lensr_pred: (B,)
    """
    def __init__(
        self,
        backbone="efficientnet_v2_s",
        in_channels=1,
        hidden=256,
        pretrained=False,
        add_noise=False,
        gauss_std_range=(0.0, 0.1),
        exp_log10_range=(3.0, 6.0),
        trunk_mid=512,
        trunk_dropout=0.2,
    ):
        super().__init__()
        self.backbone = EfficientNetV2Backbone(backbone, in_channels, pretrained)

        def make_trunk():
            return nn.Sequential(
                nn.Linear(self.backbone.feat_dim, trunk_mid),
                nn.ReLU(inplace=True),
                nn.Dropout(trunk_dropout),
                nn.Linear(trunk_mid, hidden),
                nn.ReLU(inplace=True),
            )

        # One trunk per head
        self.trunks = nn.ModuleDict({
            "cls":   make_trunk(),
            "rad":   make_trunk(),
            "ser":   make_trunk(),
            "lensr": make_trunk(),
        })

        # Heads
        self.cls_head   = nn.Linear(hidden, 1)
        self.rad_head   = nn.Linear(hidden, 1)
        self.ser_head   = nn.Linear(hidden, 1)
        self.lensr_head = nn.Linear(hidden, 1)

        # Noise config
        self.add_noise = add_noise
        self.gauss_std_range = gauss_std_range
        self.exp_log10_range = exp_log10_range

    def forward(self, x):
        # --- Noise + Poisson-like noise (TRAIN ONLY) ---
        if self.training and self.add_noise:
            a, b = self.gauss_std_range
            std = torch.empty(1, device=x.device).uniform_(a, b).item()
            x = x + torch.randn_like(x) * std

            lo, hi = self.exp_log10_range
            exp_log10 = torch.empty(1, device=x.device).uniform_(lo, hi).item()
            exp_time = 10.0 ** exp_log10

            sigma = torch.sqrt(torch.abs(x) / exp_time + 1e-8)
            x = x + torch.randn_like(x) * sigma

        # --- Backbone ---
        f = self.backbone(x)

        # --- Separate trunks per task ---
        h_cls   = self.trunks["cls"](f)
        h_rad   = self.trunks["rad"](f)
        h_ser   = self.trunks["ser"](f)
        h_lensr = self.trunks["lensr"](f)

        out = {}
        out["lens_logit"] = self.cls_head(h_cls).squeeze(1)

        # keep your softplus choice (only if targets are >= 0)
        out["r_pred"]     = F.softplus(self.rad_head(h_rad)).squeeze(1)
        out["ser_pred"]   = F.softplus(self.ser_head(h_ser)).squeeze(1)
        out["lensr_pred"] = F.softplus(self.lensr_head(h_lensr)).squeeze(1)

        return out
