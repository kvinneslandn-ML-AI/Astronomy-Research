class Encoder(nn.Module):
    def __init__(self, in_channels=1, hidden_channels=64, z_channels=128):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=4, stride=2, padding=1)  # 125 → 63
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, stride=1, padding=1)  # stays 63
        self.conv3 = nn.Conv2d(hidden_channels, z_channels, kernel_size=3, stride=1, padding=1)  # stays 63

        self.relu = nn.ReLU()

    def forward(self, x):
        out1 = self.relu(self.conv1(x))
        out2 = self.relu(self.conv2(out1))
        z = self.conv3(out2)

        return z, out1 

class Decoder(nn.Module):
    def __init__(self, z_channels=128, hidden_channels=64, out_channels=1):
        super().__init__()

        self.decode_main = nn.Sequential(
            nn.ConvTranspose2d(z_channels, hidden_channels, kernel_size=4, stride=2, padding=1),  # 63 → 126
            nn.ReLU(),
        )

        # After upsampling, combine with skip connection from encoder
        self.final_layers = nn.Sequential(
            nn.Conv2d(hidden_channels + hidden_channels, hidden_channels, kernel_size=3, stride=1, padding=1),  # Skip concat
            nn.ReLU(),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=3, stride=1, padding=1)  # Output: (1, 125, 125)
        )

    def forward(self, z, skip_connection):
        x = self.decode_main(z)  # Upsample

        # Crop skip connection to match (in case of size mismatch)
        skip_upsampled = F.interpolate(skip_connection, size=x.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip_upsampled], dim=1)
        x = self.final_layers(x)

        return x[:, :, :125, :125] 

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings=64, embedding_dim=32, commitment_cost=0.50):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1/self.num_embeddings, 1/self.num_embeddings)
        self.commitment_cost = commitment_cost

    def forward(self, z):
        # z: [B, C, H, W] → [B, H, W, C]
        z_perm = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z_perm.view(-1, self.embedding_dim)
    
        # Compute distances
        distances = (
            torch.sum(z_flattened**2, dim=1, keepdim=True)
            - 2 * torch.matmul(z_flattened, self.embedding.weight.t())
            + torch.sum(self.embedding.weight**2, dim=1)
        )
    
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.size(0), self.num_embeddings, device=z.device)
        encodings.scatter_(1, encoding_indices, 1)
    
        quantized = torch.matmul(encodings, self.embedding.weight)
        quantized = quantized.view(z_perm.shape)
    
        # Undo permute → back to [B, C, H, W]
        quantized = quantized.permute(0, 3, 1, 2).contiguous()
    
        # Losses (quantized and z now match: [B, C, H, W])
        commitment_loss = self.commitment_cost * F.mse_loss(quantized.detach(), z)
        codebook_loss = F.mse_loss(quantized, z.detach())
    
        # Straight-through estimator
        quantized = z + (quantized - z).detach()
    
        return quantized, commitment_loss + codebook_loss

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 64, 4, stride=2, padding=1),  # (1, 125, 125) → (64, ~63, ~63)
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 1, 4, stride=1, padding=1),  # → (1, ~61, ~61)
        )

    def forward(self, x):
        return self.model(x)

class VQVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.quantizer = VectorQuantizer()

    def forward(self, noisy_input, clean_target):
        z, skip = self.encoder(noisy_input)
        quantized, vq_loss = self.quantizer(z)

        residual = self.decoder(quantized, skip)
        # Ensure residual matches input size exactly
        residual_upsampled = F.interpolate(residual, size=noisy_input.shape[2:], mode='bilinear', align_corners=False)
        
        x_recon = noisy_input + residual_upsampled
        recon_loss = F.l1_loss(x_recon, clean_target)

        return x_recon, recon_loss + vq_loss
