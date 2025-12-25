import os
import torch
from torch.utils.data import Dataset
import imageio.v3 as iio
import matplotlib.pyplot as plt
from torchvision.utils import make_grid, save_image
from PIL import Image


class CustomDataset(Dataset):
    def __init__(self, image_path, transform=None, extensions=(".png", ".jpg", ".jpeg")):
        self.image_path = image_path
        self.transform = transform

        self.images = sorted([
            f for f in os.listdir(image_path)
            if f.lower().endswith(extensions)
        ])

        self.images = self._filter_valid_images()

    def _filter_valid_images(self):
        valid = []
        for img_name in self.images:
            img_path = os.path.join(self.image_path, img_name)
            try:
                img = iio.imread(img_path)
                if img.ndim == 2:
                    continue  # grayscale
                if img.shape[-1] != 3:
                    continue  # skip RGBA or weird formats
                valid.append(img_name)
            except Exception:
                pass
        return valid

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_path, self.images[idx])
        img = iio.imread(img_path)

        if self.transform is not None:
            img = self.transform(img)

        return img
    

def visualize_dataset(loader, num_images=16, nrow=4):
    images = []

    for batch in loader:
        if isinstance(batch, (list, tuple)):
            batch = batch[0]
        images.append(batch)
        if sum(b.size(0) for b in images) >= num_images:
            break

    images = torch.cat(images, dim=0)[:num_images]

    grid = make_grid(
        images,
        nrow=nrow,
        padding=0,
        normalize=True,
        value_range=(-1, 1) if images.min() < 0 else None,
    )

    np_img = grid.permute(1, 2, 0).cpu().numpy()

    plt.figure(figsize=(nrow * 2, (num_images // nrow) * 2))
    plt.imshow(np_img)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.show()


def cherry_pick_samples(
    generator,
    nz,
    device,
    total=25,
    save_path="cherry_picked.png",
):
    generator.eval()

    kept = []

    print(f"Cherry-picking {total} images.")
    print("Type 'y' + Enter to keep an image, anything else to skip.\n")

    with torch.no_grad():
        while len(kept) < total:
            noise = torch.randn(1, nz, 1, 1, device=device)
            img = generator(noise).cpu()

            # visualize
            vis = (img + 1) / 2  # [-1,1] -> [0,1]
            plt.imshow(vis[0].permute(1, 2, 0))
            plt.axis("off")
            plt.show()

            choice = input(f"Keep image {len(kept)+1}/{total}? [y/N]: ")

            if choice.lower() == "y":
                kept.append(img)
                print("✓ kept\n")
            else:
                print("✗ skipped\n")

    # stack and save grid
    kept = torch.cat(kept, dim=0)
    grid = make_grid(kept, nrow=5, normalize=True, value_range=(-1, 1))
    save_image(grid, save_path)

    print(f"Saved cherry-picked grid to: {save_path}")

    return kept


def samples_to_gif(
    samples_dir,
    out_path="samples.gif",
    duration=200,
    loop=0,
):
    # collect image files
    image_files = sorted(
        [
            os.path.join(samples_dir, f)
            for f in os.listdir(samples_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
    )

    if len(image_files) == 0:
        raise RuntimeError("No images found in samples directory")

    frames = [Image.open(f).convert("RGB") for f in image_files]

    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=loop,
    )

    print(f"GIF saved to: {out_path}")