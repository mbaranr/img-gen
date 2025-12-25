import os
import torch.optim as optim
import torch
from tqdm import tqdm
from torchvision.utils import save_image


def train_gan(
    g,
    d,
    loss_fn,
    dataloader,
    epochs,
    device,
    nz=100,
    lr=2e-4,
    beta1=0.5,
    out_dir="outputs",
):
    os.makedirs(out_dir, exist_ok=True)
    sample_dir = os.path.join(out_dir, "samples")
    ckpt_dir = os.path.join(out_dir, "checkpoints")
    os.makedirs(sample_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    g = g.to(device)
    d = d.to(device)

    opt_g = optim.Adam(g.parameters(), lr=lr, betas=(beta1, 0.999))
    opt_d = optim.Adam(d.parameters(), lr=lr, betas=(beta1, 0.999))

    fixed_noise = torch.randn(25, nz, 1, 1, device=device)

    # per-batch loss
    g_losses = []
    d_losses = []

    iters = 0
    epoch_bar = tqdm(range(epochs), desc="Epochs")

    for epoch in epoch_bar:
        batch_bar = tqdm(dataloader, desc="Batches", leave=False)

        for real, _ in batch_bar:
            real = real.to(device)
            bsz = real.size(0)

            # train discriminator
            d.zero_grad()

            real_labels = torch.ones(bsz, device=device)
            fake_labels = torch.zeros(bsz, device=device)

            out_real = d(real)
            loss_real = loss_fn(out_real, real_labels)

            noise = torch.randn(bsz, nz, 1, 1, device=device)
            fake = g(noise)

            out_fake = d(fake.detach())
            loss_fake = loss_fn(out_fake, fake_labels)

            loss_d = loss_real + loss_fake
            loss_d.backward()
            opt_d.step()

            # train generator
            g.zero_grad()

            out_fake = d(fake)
            loss_g = loss_fn(out_fake, real_labels)
            loss_g.backward()
            opt_g.step()

            # logging losses
            g_losses.append(loss_g.item())
            d_losses.append(loss_d.item())

            batch_bar.set_postfix(
                loss_d=f"{loss_d.item():.4f}",
                loss_g=f"{loss_g.item():.4f}",
            )

            # save samples every 150 batches
            if iters % 150 == 0:
                g.eval()
                with torch.no_grad():
                    samples = g(fixed_noise).cpu()
                g.train()

                save_image(
                    samples,
                    f"{sample_dir}/iter_{iters:06d}.png",
                    nrow=5,
                    normalize=True,
                    value_range=(-1, 1),
                )

            iters += 1

        # save checkpoint every 2 epochs
        if (epoch + 1) % 2 == 0:
            torch.save(
                {
                    "epoch": epoch + 1,
                    "generator": g.state_dict(),
                    "discriminator": d.state_dict(),
                },
                f"{ckpt_dir}/dcgan_epoch_{epoch+1}.pt",
            )

        epoch_bar.set_postfix(
            loss_d=f"{loss_d.item():.4f}",
            loss_g=f"{loss_g.item():.4f}",
        )

    return g, d, g_losses, d_losses