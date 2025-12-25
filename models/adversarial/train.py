import os
import torch.optim as optim
import torch
from tqdm import tqdm
from torchvision.utils import save_image


def prepare_output_dirs(out_dir):
    sample_dir = os.path.join(out_dir, "samples")
    ckpt_dir = os.path.join(out_dir, "checkpoints")
    os.makedirs(sample_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    return sample_dir, ckpt_dir


def make_optimizers(g, d, lr, beta1):
    opt_g = optim.Adam(g.parameters(), lr=lr, betas=(beta1, 0.999))
    opt_d = optim.Adam(d.parameters(), lr=lr, betas=(beta1, 0.999))
    return opt_g, opt_d


def sample_and_save(
    generator,
    fixed_noise,
    sample_dir,
    step,
    nrow=5,
):
    generator.eval()
    with torch.no_grad():
        samples = generator(fixed_noise).cpu()
    generator.train()

    save_image(
        samples,
        f"{sample_dir}/iter_{step:06d}.png",
        nrow=nrow,
        normalize=True,
        value_range=(-1, 1),
    )


def save_checkpoint(
    g,
    d,
    ckpt_dir,
    epoch,
    prefix,
):
    torch.save(
        {
            "epoch": epoch,
            "generator": g.state_dict(),
            "discriminator": d.state_dict(),
        },
        f"{ckpt_dir}/{prefix}_epoch_{epoch}.pt",
    )


def train_dcgan(
    g,
    d,
    dataloader,
    epochs,
    device,
    loss_fn,
    nz=100,
    lr=2e-4,
    beta1=0.5,
    out_dir="outputs",
    sample_every=150,
):
    sample_dir, ckpt_dir = prepare_output_dirs(out_dir)

    g, d = g.to(device), d.to(device)
    opt_g, opt_d = make_optimizers(g, d, lr, beta1)

    fixed_noise = torch.randn(25, nz, 1, 1, device=device)

    g_losses, d_losses = [], []
    iters = 0

    epoch_bar = tqdm(range(epochs), desc="DCGAN Epochs")

    for epoch in epoch_bar:
        batch_bar = tqdm(dataloader, desc="Batches", leave=False)

        for real, _ in batch_bar:
            real = real.to(device)
            bsz = real.size(0)

            # discriminator
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

            # generator
            g.zero_grad()
            out_fake = d(fake)
            loss_g = loss_fn(out_fake, real_labels)
            loss_g.backward()
            opt_g.step()

            d_losses.append(loss_d.item())
            g_losses.append(loss_g.item())

            batch_bar.set_postfix(
                loss_d=f"{loss_d.item():.4f}",
                loss_g=f"{loss_g.item():.4f}",
            )

            if iters % sample_every == 0:
                sample_and_save(g, fixed_noise, sample_dir, iters)

            iters += 1

        save_checkpoint(g, d, ckpt_dir, epoch + 1, "dcgan")

        epoch_bar.set_postfix(
            loss_d=f"{loss_d.item():.4f}",
            loss_g=f"{loss_g.item():.4f}",
        )

    return g, d, g_losses, d_losses


def gradient_penalty(critic, real, fake, device):
    batch_size = real.size(0)
    eps = torch.rand(batch_size, 1, 1, 1, device=device)
    interpolated = fake + eps * (real - fake)
    interpolated.requires_grad_(True)

    scores = critic(interpolated)

    grad = torch.autograd.grad(
        outputs=scores,
        inputs=interpolated,
        grad_outputs=torch.ones_like(scores),
        create_graph=True,
    )[0]

    grad = grad.view(batch_size, -1)
    gp = ((grad.norm(2, dim=1) - 1) ** 2).mean()
    return gp


def train_wgan_gp(
    g,
    c,
    dataloader,
    epochs,
    device,
    nz=100,
    lr=2e-4,
    beta1=0.0,
    lambda_gp=10.0,
    n_critic=5,
    out_dir="outputs",
    sample_every=150,
):
    sample_dir, ckpt_dir = prepare_output_dirs(out_dir)

    g, c = g.to(device), c.to(device)
    opt_g, opt_c = make_optimizers(g, c, lr, beta1)

    fixed_noise = torch.randn(25, nz, 1, 1, device=device)

    g_losses, c_losses = [], []
    iters = 0

    epoch_bar = tqdm(range(epochs), desc="WGAN-GP Epochs")

    for epoch in epoch_bar:
        batch_bar = tqdm(dataloader, desc="Batches", leave=False)

        for real, _ in batch_bar:
            real = real.to(device)
            bsz = real.size(0)

            # critic updates
            for _ in range(n_critic):
                c.zero_grad()

                noise = torch.randn(bsz, nz, 1, 1, device=device)
                fake = g(noise)

                loss_real = c(real).mean()
                loss_fake = c(fake.detach()).mean()
                gp = gradient_penalty(c, real, fake.detach(), device)

                loss_c = loss_fake - loss_real + lambda_gp * gp
                loss_c.backward()
                opt_c.step()

            # generator update
            g.zero_grad()
            noise = torch.randn(bsz, nz, 1, 1, device=device)
            fake = g(noise)

            loss_g = -c(fake).mean()
            loss_g.backward()
            opt_g.step()

            c_losses.append(loss_c.item())
            g_losses.append(loss_g.item())

            batch_bar.set_postfix(
                loss_c=f"{loss_c.item():.4f}",
                loss_g=f"{loss_g.item():.4f}",
            )

            if iters % sample_every == 0:
                sample_and_save(g, fixed_noise, sample_dir, iters)

            iters += 1

        save_checkpoint(g, c, ckpt_dir, epoch + 1, "wgan_gp")

        epoch_bar.set_postfix(
            loss_c=f"{loss_c.item():.4f}",
            loss_g=f"{loss_g.item():.4f}",
        )

    return g, c, g_losses, c_losses