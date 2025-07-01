import torch
import torch.nn as nn
import torch.nn.functional as F
from taming.modules.discriminator.model import NLayerDiscriminator, weights_init


class DummyLoss(nn.Module):
    def __init__(self):
        super().__init__()


def adopt_weight(weight, global_step, threshold=0, value=0.):
    if global_step < threshold:
        weight = value
    return weight


def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1. - logits_real))
    loss_fake = torch.mean(F.relu(1. + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def vanilla_d_loss(logits_real, logits_fake):
    d_loss = 0.5 * (
        torch.mean(torch.nn.functional.softplus(-logits_real)) +
        torch.mean(torch.nn.functional.softplus(logits_fake)))
    return d_loss


class VQLPIPSWithDiscriminator(nn.Module):
    def __init__(self, disc_start, codebook_weight=1.0, pixelloss_weight=1.0,
                 disc_num_layers=3, disc_in_channels=3, disc_factor=1.0, disc_weight=1.0,
                 use_actnorm=False, disc_conditional=False,
                 disc_ndf=32, disc_loss="hinge", num_classes=1):
        super().__init__()
        assert disc_loss in ["hinge", "vanilla"]
        
        self.pixel_weight = pixelloss_weight
        self.num_classes = num_classes
        self.discriminator = NLayerDiscriminator(input_nc=disc_in_channels,
                                                 n_layers=disc_num_layers,
                                                 use_actnorm=use_actnorm,
                                                 ndf=disc_ndf,
                                                 out_ch=num_classes
                                                 ).apply(weights_init)
        self.discriminator_iter_start = disc_start
        if disc_loss == "hinge":
            self.disc_loss = hinge_d_loss
        elif disc_loss == "vanilla":
            self.disc_loss = vanilla_d_loss
        else:
            raise ValueError(f"Unknown GAN loss '{disc_loss}'.")
        print(f"VQLPIPSWithDiscriminator running with {disc_loss} loss.")
        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight
     

    def mean_flat(tensor):
        """
        Take the mean over all non-batch dimensions.
        """
        return tensor.mean(dim=list(range(2, len(tensor.shape))))
    def forward(self,inputs, reconstructions, optimizer_idx,
                global_step, last_layer=None, label=None, split="train"):
        rec_loss = torch.mean(self.mean_flat((inputs - reconstructions) ** 2), dim=0)
        weights = torch.ones(len(rec_loss)).cuda()
        nll_loss = rec_loss
        nll_loss = (rec_loss * weights).mean()

        batch_size = inputs.shape[0]
        # now the GAN part
        if optimizer_idx == 0:
            # generator update

            # multi-class classification
            logits_fake = self.discriminator(reconstructions.contiguous())
            targets = label.clone()
            logits_reshaped = logits_fake.view(batch_size, self.num_classes, -1).mean(dim=2)
            g_loss = F.cross_entropy(logits_reshaped, targets)

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            loss = nll_loss + disc_factor * g_loss 

            log = {"{}/total_loss".format(split): loss.clone().detach().mean(),
                   "{}/nll_loss".format(split): nll_loss.detach().mean(),
                   "{}/rec_loss".format(split): rec_loss.detach().mean(),
                   "{}/disc_factor".format(split): torch.tensor(disc_factor),
                   "{}/g_loss".format(split): g_loss.detach().mean(),
                   }
            return loss, log

        if optimizer_idx == 1:
            # second pass for discriminator update

         
            logits_real = self.discriminator(inputs.contiguous().detach())
            logits_real_reshaped = logits_real.view(batch_size, self.num_classes, -1).mean(dim=2)
            real_targets = label.clone()
            real_loss = F.cross_entropy(logits_real_reshaped, real_targets)
            
            logits_fake = self.discriminator(reconstructions.contiguous().detach())
            logits_fake_reshaped = logits_fake.view(batch_size, self.num_classes, -1).mean(dim=2)
            fake_targets = torch.zeros(batch_size, dtype=torch.long, device=inputs.device)  # fake class (0)
            fake_loss = F.cross_entropy(logits_fake_reshaped, fake_targets)
            
            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            d_loss = disc_factor * 0.5 * (real_loss + fake_loss)


            log = {"{}/disc_loss".format(split): d_loss.clone().detach().mean(),
                   "{}/logits_real".format(split): logits_real.detach().mean(),
                   "{}/logits_fake".format(split): logits_fake.detach().mean()
                   }
            return d_loss, log