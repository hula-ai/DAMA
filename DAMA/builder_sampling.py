# DAMA adaptive sampling
# References: MAE https://github.com/facebookresearch/mae

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

class DAMA(nn.Module):
    """
    Build a Masked Autoencoder ViTs backbone with mutual Information approach
    """

    def __init__(self, base_encoder, loss_beta=2, last_k_blocks=6, loss_alpha=1, norm_pix_loss=False, in_chans=7):
        super(DAMA, self).__init__()

        self.loss_beta = loss_beta
        self.last_k_blocks = last_k_blocks
        self.loss_alpha = loss_alpha
        self.norm_pix_loss = norm_pix_loss
        self.in_chans = in_chans

        # build encoders
        self.base_encoder = base_encoder()
        self.momentum_encoder = base_encoder()
        embed_dim = self.base_encoder.embed_dim

        projs = []
        projs.append(nn.Linear(embed_dim, embed_dim*2))
        projs.append(nn.LayerNorm(embed_dim*2))
        projs.append(nn.ReLU(inplace=True))
        projs.append(nn.Linear(embed_dim*2, embed_dim))
        projs.append(nn.LayerNorm(embed_dim))
        projs.append(nn.ReLU(inplace=True))
        projs.append(nn.Softmax(dim=-1))
        self.projs_head = nn.Sequential(*projs)

    # uncomment for model1 -> ema & model1 -> model1 (shared weights)
    #     for param_b, param_m in zip(self.base_encoder.parameters(), self.momentum_encoder.parameters()):
    #         param_m.data.copy_(param_b.data)  # initialize
    #         param_m.requires_grad = False  # not update by gradient
    #
    # @torch.no_grad()
    # def _update_momentum_encoder(self, m):
    #     """Momentum update of the momentum encoder"""
    #     for param_b, param_m in zip(self.base_encoder.parameters(), self.momentum_encoder.parameters()):
    #         param_m.data = param_m.data * m + param_b.data * (1. - m)


    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.base_encoder.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], self.in_chans, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * self.in_chans))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.base_encoder.patch_embed.patch_size[0]
        h = w = int(x.shape[1] ** .5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, self.in_chans))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], self.in_chans, h * p, h * p))
        return imgs

    def forward_recon_loss(self, imgs, pred, mask):
        """
        MAE https://github.com/facebookresearch/mae/blob/main/models_mae.py#L198
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss_per_patch = loss.mean(dim=-1)  # [N, L], mean loss per patch
        loss = (loss_per_patch * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss, loss_per_patch

    def forward_regression_loss(self, x, y):
        """
        regression loss between student's and teacher's feature

        """
        x = self.projs_head(x)

        y = y[-self.last_k_blocks:]
        y = [F.layer_norm(feat.float(), feat.shape[-1:]) for feat in y]
        y = sum(y) / len(y)
        y = F.softmax(y, dim=-1)

        if self.loss_beta == 0:
            loss = F.mse_loss(x.float(), y.float(), reduction="none").sum(dim=-1)
        else:
            loss = F.smooth_l1_loss(
                x.float(), y.float(), reduction="none", beta=self.loss_beta
            ).sum(dim=-1)

        if self.loss_alpha is None or self.loss_alpha <= 0:
            loss = loss.sum() / math.sqrt(x.size(-1))
        else:
            loss = loss.sum() * self.loss_alpha
        return loss

    # model1 -> ema
    # def forward(self, imgs, m):
    #     '''
    #     :param imgs: same images for student and teacher
    #     :param m:    momentum
    #     :return:     loss
    #     named 2 branches as img and txt for clarification
    #     '''
    #
    #     # random sampling
    #     pred_img, x_img, mask_img, _, ids_shuffle_img, ids_restore_img = self.base_encoder(imgs, None, None, False) # compute features
    #     recons_img, loss_per_patch_img = self.forward_recon_loss(imgs, pred_img, mask_img) # reconstruction loss for student
    #
    #     # adaptive sampling
    #     with torch.no_grad():  # no gradient
    #         self._update_momentum_encoder(m)  # update the momentum encoder
    #         # compute momentum features as targets
    #         pred_txt, _, mask_txt, featout_txt, _, _ = self.momentum_encoder(imgs, mask_img, loss_per_patch_img, True)
    #
    #     # reconstruction loss for teacher
    #     recons_txt, _ = self.forward_recon_loss(imgs, pred_txt, mask_txt)
    #
    #     # regression loss (mutual info)
    #     info_loss = self.forward_regression_loss(x_img, featout_txt)
    #
    #     return info_loss, recons_img, recons_txt, pred_img, mask_img, pred_txt, mask_txt

    # model1 -> model1
    # def forward(self, imgs):
    #     '''
    #     named 2 branches as img and txt for clarification
    #     '''
    #
    #     # random sampling
    #     pred_img, x_img, mask_img, _, ids_shuffle_img, ids_restore_img = self.base_encoder(imgs, None, None, False) # compute features
    #     recons_img, loss_per_patch_img = self.forward_recon_loss(imgs, pred_img, mask_img) # reconstruction loss for student
    #
    #     # adaptive sampling
    #     pred_txt, _, mask_txt, featout_txt, _, _ = self.base_encoder(imgs, mask_img, loss_per_patch_img, True)
    #
    #     # reconstruction loss for teacher
    #     recons_txt, _ = self.forward_recon_loss(imgs, pred_txt, mask_txt)
    #
    #     # regression loss (mutual info)
    #     info_loss = self.forward_regression_loss(x_img, featout_txt)
    #
    #     return info_loss, recons_img, recons_txt, pred_img, mask_img, pred_txt, mask_txt

    # model1 -> model2
    def forward(self, imgs, m):
        '''
        named 2 branches as img and txt for clarification
        '''
        # random sampling
        pred_img, x_img, mask_img, _, ids_shuffle_img, ids_restore_img = self.base_encoder(imgs, None, None, False) # compute features
        recons_img, loss_per_patch_img = self.forward_recon_loss(imgs, pred_img, mask_img) # reconstruction loss for student

        # adaptive sampling
        pred_txt, _, mask_txt, featout_txt, _, _ = self.momentum_encoder(imgs, mask_img, loss_per_patch_img, True)

        # reconstruction loss for teacher
        recons_txt, _ = self.forward_recon_loss(imgs, pred_txt, mask_txt)

        # regression loss (mutual info)
        info_loss = self.forward_regression_loss(x_img, featout_txt)

        return info_loss, recons_img, recons_txt, pred_img, mask_img, pred_txt, mask_txt
