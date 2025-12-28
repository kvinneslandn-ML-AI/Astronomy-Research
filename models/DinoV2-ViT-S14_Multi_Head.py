import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file
from timm.layers.pos_embed import resample_abs_pos_embed

class DinoV2LensRadius(nn.Module):
    def __init__(self, weights_path, in_chans=1, hidden=256, img_size=224):
        super().__init__()

        # Build ViT with fixed img_size=224 (so PatchEmbed assert passes)
        backbone = timm.create_model(
            "vit_small_patch14_dinov2",
            pretrained=False,
            num_classes=0,
            in_chans=3,          # match checkpoint first
            img_size=img_size,
            global_pool="avg",
        )

        # Load local weights
        state = load_file(weights_path)

        # --- Resample absolute pos_embed if needed ---
        if "pos_embed" in state and state["pos_embed"].shape != backbone.pos_embed.shape:
            # grid sizes (exclude class token)
            old_tokens = state["pos_embed"].shape[1] - 1
            new_tokens = backbone.pos_embed.shape[1] - 1
        
            old_g = int(old_tokens ** 0.5)
            new_g = int(new_tokens ** 0.5)
        
            assert old_g * old_g == old_tokens, f"Checkpoint pos tokens not square: {old_tokens}"
            assert new_g * new_g == new_tokens, f"Model pos tokens not square: {new_tokens}"
        
            state["pos_embed"] = resample_abs_pos_embed(
                state["pos_embed"],
                new_size=(new_g, new_g),
                old_size=(old_g, old_g),
                num_prefix_tokens=1,
            )

        missing, unexpected = backbone.load_state_dict(state, strict=False)
        print(f"Loaded weights. missing={len(missing)} unexpected={len(unexpected)}")

        # Convert patch embedding to 1-channel by averaging RGB weights
        old = backbone.patch_embed.proj
        new = nn.Conv2d(
            1, old.out_channels,
            kernel_size=old.kernel_size,
            stride=old.stride,
            padding=old.padding,
            bias=(old.bias is not None),
        )
        with torch.no_grad():
            new.weight.copy_(old.weight.mean(dim=1, keepdim=True))
            if old.bias is not None:
                new.bias.copy_(old.bias)
        backbone.patch_embed.proj = new

        self.backbone = backbone
        feat_dim = backbone.num_features

        self.shared = nn.Sequential(
            nn.Linear(feat_dim, 512), nn.ReLU(True), nn.Dropout(0.2),
            nn.Linear(512, hidden), nn.ReLU(True),
        )
        self.cls_head = nn.Linear(hidden, 1)
        self.rad_head = nn.Linear(hidden, 1)

    def forward(self, x):
        f = self.backbone(x)
        h = self.shared(f)
        lens_logit = self.cls_head(h).squeeze(1)
        theta_pred = F.softplus(self.rad_head(h)).squeeze(1)
        return lens_logit, theta_pred
