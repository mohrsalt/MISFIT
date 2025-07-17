import os
import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from monai.losses import SSIMLoss
from tqdm import tqdm

# === Refinement Network ===
class RefinementNet(nn.Module):
    def __init__(self, in_ch=1, base=16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(in_ch, base, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(base, base, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.Conv3d(base, base, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(base, in_ch, 3, padding=1)
        )

    def forward(self, x):
        return self.decoder(self.encoder(x)) + x

# === Dataset for pseudo train set ===
class GeneratedToTargetDataset(Dataset):
    def __init__(self, pseudo_train_dir, patch_size=(64, 64, 64)):
        self.samples = []
        self.patch_size = patch_size

        for subject in os.listdir(pseudo_train_dir):
            subject_path = os.path.join(pseudo_train_dir, subject)
            files = os.listdir(subject_path)

            # Get paths to generated and missing target
            gen_path = None
            tgt_path = None

            for f in files:
                if f.endswith('-generated.nii.gz'):
                    gen_path = os.path.join(subject_path, f)
                elif f.startswith('Missing_Target_') and f.endswith('.nii.gz'):
                    tgt_path = os.path.join(subject_path, f)

            if gen_path and tgt_path:
                self.samples.append((gen_path, tgt_path))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        gen_path, tgt_path = self.samples[idx]
        gen_vol = nib.load(gen_path).get_fdata().astype(np.float32)
        tgt_vol = nib.load(tgt_path).get_fdata().astype(np.float32)

        # Normalize
        gen_vol = (gen_vol - np.mean(gen_vol)) / (np.std(gen_vol) + 1e-5)
        tgt_vol = (tgt_vol - np.mean(tgt_vol)) / (np.std(tgt_vol) + 1e-5)

        # Center crop
        cz, cy, cx = [d // 2 for d in gen_vol.shape]
        pz, py, px = [p // 2 for p in self.patch_size]
        gen_crop = gen_vol[cz - pz:cz + pz, cy - py:cy + py, cx - px:cx + px]
        tgt_crop = tgt_vol[cz - pz:cz + pz, cy - py:cy + py, cx - px:cx + px]

        return (
            torch.from_numpy(gen_crop[None, ...]),
            torch.from_numpy(tgt_crop[None, ...])
        )


# === Main Training ===
def train():
    pseudo_train_dir = '/home/users/ntu/mohor001/scratch/Task8DataBrats/pseudo_train_set'
    dataset = GeneratedToTargetDataset(pseudo_train_dir)
    loader = DataLoader(dataset, batch_size=1, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RefinementNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    mse_loss = nn.MSELoss()
    ssim_loss = SSIMLoss(spatial_dims=3)

    for epoch in range(5):
        epoch_loss = 0.0
        for x, y in tqdm(loader, desc=f"Epoch {epoch+1}"):
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = 0.7 * mse_loss(pred, y) + 0.3 * ssim_loss(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1}: Avg Loss = {epoch_loss / len(loader):.4f}")

    torch.save(model.state_dict(), "refinement_net.pt")

if __name__ == "__main__":
    train()
