"""
Lightning Module for LadderVAE.
"""

from typing import Any, Dict

import ml_collections
import numpy as np
import pytorch_lightning as L
import torch
import torchvision.transforms.functional as F

from careamics.models.lvae.likelihoods import LikelihoodModule
from careamics.models.lvae.lvae import LadderVAE
from careamics.models.lvae.utils import (
    LossType,
    compute_batch_mean,
    free_bits_kl,
    torch_nanmean,
)

from .metrics import RangeInvariantPsnr, RunningPSNR
from .train_utils import MetricMonitor


class LadderVAELight(L.LightningModule):

    def __init__(
        self,
        config: ml_collections.ConfigDict,
        data_mean: Dict[str, torch.Tensor],
        data_std: Dict[str, torch.Tensor],
        target_ch: int,
    ):
        """
        Here we will do the following:
            - initialize the model (from LadderVAE class)
            - initialize the parameters related to the training and loss.

        NOTE:
        Some of the model attributes are defined in the model object itself, while some others will be defined here.
        Note that all the attributes related to the training and loss that were already defined in the model object
        are redefined here as Lightning module attributes (e.g., self.some_attr = model.some_attr).
        The attributes related to the model itself are treated as model attributes (e.g., self.model.some_attr).

        NOTE: HC stands for Hard Coded attribute.
        """
        super().__init__()

        self.data_mean = data_mean
        self.data_std = data_std
        self.target_ch = target_ch

        # Initialize LVAE model
        self.model = LadderVAE(
            data_mean=data_mean, data_std=data_std, config=config, target_ch=target_ch
        )

        ##### Define attributes from config #####
        self.workdir = config.workdir
        self._input_is_sum = False
        self.kl_loss_formulation = config.loss.kl_loss_formulation
        assert self.kl_loss_formulation in [
            None,
            "",
            "usplit",
            "denoisplit",
            "denoisplit_usplit",
        ], f"""
            Invalid kl_loss_formulation. {self.kl_loss_formulation}"""

        ##### Define loss attributes #####
        # Parameters already defined in the model object
        self.loss_type = self.model.loss_type
        self._denoisplit_w = self._usplit_w = None
        if self.loss_type == LossType.DenoiSplitMuSplit:
            self._usplit_w = 0
            self._denoisplit_w = 1 - self._usplit_w
            assert self._denoisplit_w + self._usplit_w == 1
        self._restricted_kl = self.model._restricted_kl

        # General loss parameters
        self.channel_1_w = 1
        self.channel_2_w = 1

        # About Reconsruction Loss
        self.reconstruction_mode = False
        self.skip_nboundary_pixels_from_loss = None
        self.reconstruction_weight = 1.0
        self._exclusion_loss_weight = 0
        self.ch1_recons_w = 1
        self.ch2_recons_w = 1
        self.enable_mixed_rec = False
        self.mixed_rec_w_step = 0

        # About KL Loss
        self.kl_weight = 1.0  # HC
        self.usplit_kl_weight = None  # HC
        self.free_bits = 1.0  # HC
        self.kl_annealing = False  # HC
        self.kl_annealtime = self.kl_start = None
        if self.kl_annealing:
            self.kl_annealtime = 10  # HC
            self.kl_start = -1  # HC

        ##### Define training attributes #####
        self.lr = config.training.lr
        self.lr_scheduler_patience = config.training.lr_scheduler_patience
        self.lr_scheduler_monitor = config.model.get("monitor", "val_loss")
        self.lr_scheduler_mode = MetricMonitor(self.lr_scheduler_monitor).mode()

        # Initialize object for keeping track of PSNR for each output channel
        self.channels_psnr = [RunningPSNR() for _ in range(self.model.target_ch)]

    def forward(self, x: Any) -> Any:
        return self.model(x)

    def training_step(
        self, batch: torch.Tensor, batch_idx: int, enable_logging: bool = True
    ) -> Dict[str, torch.Tensor]:

        if self.current_epoch == 0 and batch_idx == 0:
            self.log("val_psnr", 1.0, on_epoch=True)

        # Pre-processing of inputs
        x, target = batch[:2]
        self.set_params_to_same_device_as(x)
        x_normalized = self.normalize_input(x)
        if self.reconstruction_mode:  # just for experimental purpose
            target_normalized = x_normalized[:, :1].repeat(1, 2, 1, 1)
            target = None
            mask = None
        else:
            target_normalized = self.normalize_target(target)
            mask = ~((target == 0).reshape(len(target), -1).all(dim=1))

        # Forward pass
        out, td_data = self.forward(x_normalized)

        if (
            self.model.encoder_no_padding_mode
            and out.shape[-2:] != target_normalized.shape[-2:]
        ):
            target_normalized = F.center_crop(target_normalized, out.shape[-2:])

        # Loss Computations
        # mask = torch.isnan(target.reshape(len(x), -1)).all(dim=1)
        recons_loss_dict, imgs = self.get_reconstruction_loss(
            reconstruction=out,
            target=target_normalized,
            input=x_normalized,
            splitting_mask=mask,
            return_predicted_img=True,
        )

        # This `if` is not used by default config
        if self.skip_nboundary_pixels_from_loss:
            pad = self.skip_nboundary_pixels_from_loss
            target_normalized = target_normalized[:, :, pad:-pad, pad:-pad]

        recons_loss = recons_loss_dict["loss"] * self.reconstruction_weight

        if torch.isnan(recons_loss).any():
            recons_loss = 0.0

        if self.model.non_stochastic_version:
            kl_loss = torch.Tensor([0.0]).cuda()
            net_loss = recons_loss
        else:
            if self.loss_type == LossType.DenoiSplitMuSplit:
                msg = f"For the loss type {LossType.name(self.loss_type)}, kl_loss_formulation must be denoisplit_usplit"
                assert self.kl_loss_formulation == "denoisplit_usplit", msg
                assert self._denoisplit_w is not None and self._usplit_w is not None

                kl_key_denoisplit = "kl_restricted" if self._restricted_kl else "kl"
                # NOTE: 'kl' key stands for the 'kl_samplewise' key in the TopDownLayer class.
                # The different naming comes from `top_down_pass()` method in the LadderVAE class.
                denoisplit_kl = self.get_kl_divergence_loss(
                    topdown_layer_data_dict=td_data, kl_key=kl_key_denoisplit
                )
                usplit_kl = self.get_kl_divergence_loss_usplit(
                    topdown_layer_data_dict=td_data
                )
                kl_loss = (
                    self._denoisplit_w * denoisplit_kl + self._usplit_w * usplit_kl
                )
                kl_loss = self.kl_weight * kl_loss

                recons_loss = self.reconstruction_loss_musplit_denoisplit(
                    out, target_normalized
                )
                # recons_loss = self._denoisplit_w * recons_loss_nm + self._usplit_w * recons_loss_gm

            elif self.kl_loss_formulation == "usplit":
                kl_loss = self.get_kl_weight() * self.get_kl_divergence_loss_usplit(
                    td_data
                )
            elif self.kl_loss_formulation in ["", "denoisplit"]:
                kl_loss = self.get_kl_weight() * self.get_kl_divergence_loss(td_data)
            net_loss = recons_loss + kl_loss

        # Logging
        if enable_logging:
            for i, x in enumerate(td_data["debug_qvar_max"]):
                self.log(f"qvar_max:{i}", x.item(), on_epoch=True)

            self.log("reconstruction_loss", recons_loss_dict["loss"], on_epoch=True)
            self.log("kl_loss", kl_loss, on_epoch=True)
            self.log("training_loss", net_loss, on_epoch=True)
            self.log("lr", self.lr, on_epoch=True)
            if self.model._tethered_ch2_scalar is not None:
                self.log(
                    "tethered_ch2_scalar",
                    self.model._tethered_ch2_scalar,
                    on_epoch=True,
                )
                self.log(
                    "tethered_ch1_scalar",
                    self.model._tethered_ch1_scalar,
                    on_epoch=True,
                )

            # self.log('grad_norm_bottom_up', self.grad_norm_bottom_up, on_epoch=True)
            # self.log('grad_norm_top_down', self.grad_norm_top_down, on_epoch=True)

        output = {
            "loss": net_loss,
            "reconstruction_loss": (
                recons_loss.detach()
                if isinstance(recons_loss, torch.Tensor)
                else recons_loss
            ),
            "kl_loss": kl_loss.detach(),
        }
        # https://github.com/openai/vdvae/blob/main/train.py#L26
        if torch.isnan(net_loss).any():
            return None

        return output

    def validation_step(self, batch: torch.Tensor, batch_idx: int):
        # Pre-processing of inputs
        x, target = batch[:2]
        self.set_params_to_same_device_as(x)
        x_normalized = self.normalize_input(x)
        if self.reconstruction_mode:  # only for experimental purpose
            target_normalized = x_normalized[:, :1].repeat(1, 2, 1, 1)
            target = None
            mask = None
        else:
            target_normalized = self.normalize_target(target)
            mask = ~((target == 0).reshape(len(target), -1).all(dim=1))

        # Forward pass
        out, _ = self.forward(x_normalized)

        if self.model.predict_logvar is not None:
            out_mean, _ = out.chunk(2, dim=1)
        else:
            out_mean = out

        if (
            self.model.encoder_no_padding_mode
            and out.shape[-2:] != target_normalized.shape[-2:]
        ):
            target_normalized = F.center_crop(target_normalized, out.shape[-2:])

        if self.loss_type == LossType.DenoiSplitMuSplit:
            recons_loss = self.reconstruction_loss_musplit_denoisplit(
                out, target_normalized
            )
            recons_loss_dict = {"loss": recons_loss}
            recons_img = out_mean
        else:
            # Metrics computation
            recons_loss_dict, recons_img = self.get_reconstruction_loss(
                reconstruction=out_mean,
                target=target_normalized,
                input=x_normalized,
                splitting_mask=mask,
                return_predicted_img=True,
            )

        # This `if` is not used by default config
        if self.skip_nboundary_pixels_from_loss:
            pad = self.skip_nboundary_pixels_from_loss
            target_normalized = target_normalized[:, :, pad:-pad, pad:-pad]

        channels_rinvpsnr = []
        for i in range(target_normalized.shape[1]):
            self.channels_psnr[i].update(recons_img[:, i], target_normalized[:, i])
            psnr = RangeInvariantPsnr(
                target_normalized[:, i].clone(), recons_img[:, i].clone()
            )
            channels_rinvpsnr.append(psnr)
            psnr = torch_nanmean(psnr).item()
            self.log(f"val_psnr_l{i+1}", psnr, on_epoch=True)

        recons_loss = recons_loss_dict["loss"]
        if torch.isnan(recons_loss).any():
            return

        self.log("val_loss", recons_loss, on_epoch=True)
        # self.log('val_psnr', (val_psnr_l1 + val_psnr_l2) / 2, on_epoch=True)

        # if batch_idx == 0 and self.power_of_2(self.current_epoch):
        #     all_samples = []
        #     for i in range(20):
        #         sample, _ = self(x_normalized[0:1, ...])
        #         sample = self.likelihood.get_mean_lv(sample)[0]
        #         all_samples.append(sample[None])

        #     all_samples = torch.cat(all_samples, dim=0)
        #     all_samples = all_samples * self.data_std + self.data_mean
        #     all_samples = all_samples.cpu()
        #     img_mmse = torch.mean(all_samples, dim=0)[0]
        #     self.log_images_for_tensorboard(all_samples[:, 0, 0, ...], target[0, 0, ...], img_mmse[0], 'label1')
        #     self.log_images_for_tensorboard(all_samples[:, 0, 1, ...], target[0, 1, ...], img_mmse[1], 'label2')

        # return net_loss

    def on_validation_epoch_end(self):
        psnr_arr = []
        for i in range(len(self.channels_psnr)):
            psnr = self.channels_psnr[i].get()
            if psnr is None:
                psnr_arr = None
                break
            psnr_arr.append(psnr.cpu().numpy())
            self.channels_psnr[i].reset()

        if psnr_arr is not None:
            psnr = np.mean(psnr_arr)
            self.log("val_psnr", psnr, on_epoch=True)
        else:
            self.log("val_psnr", 0.0, on_epoch=True)

        if self.mixed_rec_w_step:
            self.mixed_rec_w = max(self.mixed_rec_w - self.mixed_rec_w_step, 0.0)
            self.log("mixed_rec_w", self.mixed_rec_w, on_epoch=True)

    def predict_step(self, batch: torch.Tensor, batch_idx: Any) -> Any:
        raise NotImplementedError("predict_step is not implemented")

    def configure_optimizers(self):
        optimizer = torch.optim.Adamax(self.parameters(), lr=self.lr, weight_decay=0)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            self.lr_scheduler_mode,
            patience=self.lr_scheduler_patience,
            factor=0.5,
            min_lr=1e-12,
            verbose=True,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": self.lr_scheduler_monitor,
        }

    ##### REQUIRED Methods for Loss Computation #####
    def get_reconstruction_loss(
        self,
        reconstruction: torch.Tensor,
        target: torch.Tensor,
        input: torch.Tensor,
        splitting_mask: torch.Tensor = None,
        return_predicted_img: bool = False,
        likelihood_obj: LikelihoodModule = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        reconstruction: torch.Tensor,
        target: torch.Tensor
        input: torch.Tensor
        splitting_mask: torch.Tensor = None
            A boolean tensor that indicates which items to keep for reconstruction loss computation.
            If `None`, all the elements of the items are considered (i.e., the mask is all `True`).
        return_predicted_img: bool = False
        likelihood_obj: LikelihoodModule = None
        """
        output = self._get_reconstruction_loss_vector(
            reconstruction=reconstruction,
            target=target,
            input=input,
            return_predicted_img=return_predicted_img,
            likelihood_obj=likelihood_obj,
        )
        loss_dict = output[0] if return_predicted_img else output

        if splitting_mask is None:
            splitting_mask = torch.ones_like(loss_dict["loss"]).bool()

        # print(len(target) - (torch.isnan(loss_dict['loss'])).sum())

        loss_dict["loss"] = loss_dict["loss"][splitting_mask].sum() / len(
            reconstruction
        )
        for i in range(1, 1 + target.shape[1]):
            key = f"ch{i}_loss"
            loss_dict[key] = loss_dict[key][splitting_mask].sum() / len(reconstruction)

        if "mixed_loss" in loss_dict:
            loss_dict["mixed_loss"] = torch.mean(loss_dict["mixed_loss"])
        if return_predicted_img:
            assert len(output) == 2
            return loss_dict, output[1]
        else:
            return loss_dict

    def _get_reconstruction_loss_vector(
        self,
        reconstruction: torch.Tensor,
        target: torch.Tensor,
        input: torch.Tensor,
        return_predicted_img: bool = False,
        likelihood_obj: LikelihoodModule = None,
    ):
        """
        Parameters
        ----------
        return_predicted_img: bool
            If set to `True`, the besides the loss, the reconstructed image is also returned.
            Default is `False`.
        """
        output = {
            "loss": None,
            "mixed_loss": None,
        }

        for i in range(1, 1 + target.shape[1]):
            output[f"ch{i}_loss"] = None

        if likelihood_obj is None:
            likelihood_obj = self.model.likelihood

        # Log likelihood
        ll, like_dict = likelihood_obj(reconstruction, target)
        ll = self._get_weighted_likelihood(ll)
        if (
            self.skip_nboundary_pixels_from_loss is not None
            and self.skip_nboundary_pixels_from_loss > 0
        ):
            pad = self.skip_nboundary_pixels_from_loss
            ll = ll[:, :, pad:-pad, pad:-pad]
            like_dict["params"]["mean"] = like_dict["params"]["mean"][
                :, :, pad:-pad, pad:-pad
            ]

        # assert ll.shape[1] == 2, f"Change the code below to handle >2 channels first. ll.shape {ll.shape}"
        output = {"loss": compute_batch_mean(-1 * ll)}
        if ll.shape[1] > 1:
            for i in range(1, 1 + target.shape[1]):
                output[f"ch{i}_loss"] = compute_batch_mean(-ll[:, i - 1])
        else:
            assert ll.shape[1] == 1
            output["ch1_loss"] = output["loss"]
            output["ch2_loss"] = output["loss"]

        if (
            self.channel_1_w is not None
            and self.channel_2_w is not None
            and (self.channel_1_w != 1 or self.channel_2_w != 1)
        ):
            assert ll.shape[1] == 2, "Only 2 channels are supported for now."
            output["loss"] = (
                self.channel_1_w * output["ch1_loss"]
                + self.channel_2_w * output["ch2_loss"]
            ) / (self.channel_1_w + self.channel_2_w)

        # This `if` is not used by default config
        if self.enable_mixed_rec:
            mixed_pred, mixed_logvar = self.get_mixed_prediction(
                like_dict["params"]["mean"],
                like_dict["params"]["logvar"],
                self.data_mean,
                self.data_std,
            )
            if (
                self.model._multiscale_count is not None
                and self.model._multiscale_count > 1
            ):
                assert input.shape[1] == self.model._multiscale_count
                input = input[:, :1]

            assert (
                input.shape == mixed_pred.shape
            ), "No fucking room for vectorization induced bugs."
            mixed_recons_ll = self.model.likelihood.log_likelihood(
                input, {"mean": mixed_pred, "logvar": mixed_logvar}
            )
            output["mixed_loss"] = compute_batch_mean(-1 * mixed_recons_ll)

        # This `if` is not used by default config
        if self._exclusion_loss_weight:
            raise NotImplementedError(
                "Exclusion loss is not well defined here, so it should not be used."
            )
            imgs = like_dict["params"]["mean"]
            exclusion_loss = compute_exclusion_loss(imgs[:, :1], imgs[:, 1:])
            output["exclusion_loss"] = exclusion_loss

        if return_predicted_img:
            return output, like_dict["params"]["mean"]

        return output

    def reconstruction_loss_musplit_denoisplit(self, out, target_normalized):
        if self.model.predict_logvar is not None:
            out_mean, _ = out.chunk(2, dim=1)
        else:
            out_mean = out

        recons_loss_nm = (
            -1 * self.model.likelihood_NM(out_mean, target_normalized)[0].mean()
        )
        recons_loss_gm = -1 * self.model.likelihood_gm(out, target_normalized)[0].mean()
        recons_loss = (
            self._denoisplit_w * recons_loss_nm + self._usplit_w * recons_loss_gm
        )
        return recons_loss

    def _get_weighted_likelihood(self, ll):
        """
        Each of the channels gets multiplied with a different weight.
        """
        if self.ch1_recons_w == 1 and self.ch2_recons_w == 1:
            return ll

        assert ll.shape[1] == 2, "This function is only for 2 channel images"

        mask1 = torch.zeros((len(ll), ll.shape[1], 1, 1), device=ll.device)
        mask1[:, 0] = 1
        mask2 = torch.zeros((len(ll), ll.shape[1], 1, 1), device=ll.device)
        mask2[:, 1] = 1

        return ll * mask1 * self.ch1_recons_w + ll * mask2 * self.ch2_recons_w

    def get_kl_weight(self):
        """
        KL loss can be weighted depending whether any annealing procedure is used.
        This function computes the weight of the KL loss in case of annealing.
        """
        if self.kl_annealing == True:
            # calculate relative weight
            kl_weight = (self.current_epoch - self.kl_start) * (
                1.0 / self.kl_annealtime
            )
            # clamp to [0,1]
            kl_weight = min(max(0.0, kl_weight), 1.0)

            # if the final weight is given, then apply that weight on top of it
            if self.kl_weight is not None:
                kl_weight = kl_weight * self.kl_weight
        elif self.kl_weight is not None:
            return self.kl_weight
        else:
            kl_weight = 1.0
        return kl_weight

    def get_kl_divergence_loss_usplit(
        self, topdown_layer_data_dict: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """ """
        kl = torch.cat(
            [kl_layer.unsqueeze(1) for kl_layer in topdown_layer_data_dict["kl"]], dim=1
        )
        # NOTE: kl.shape = (16,4) 16 is batch size. 4 is number of layers.
        # Values are sum() and so are of the order 30000
        # Example values: 30626.6758, 31028.8145, 29509.8809, 29945.4922, 28919.1875, 29075.2988

        nlayers = kl.shape[1]
        for i in range(nlayers):
            # topdown_layer_data_dict['z'][2].shape[-3:] = 128 * 32 * 32
            norm_factor = np.prod(topdown_layer_data_dict["z"][i].shape[-3:])
            # if self._restricted_kl:
            #     pow = np.power(2,min(i + 1, self._multiscale_count-1))
            #     norm_factor /= pow * pow

            kl[:, i] = kl[:, i] / norm_factor

        kl_loss = free_bits_kl(kl, 0.0).mean()
        return kl_loss

    def get_kl_divergence_loss(self, topdown_layer_data_dict, kl_key="kl"):
        """
        kl[i] for each i has length batch_size
        resulting kl shape: (batch_size, layers)
        """
        kl = torch.cat(
            [kl_layer.unsqueeze(1) for kl_layer in topdown_layer_data_dict[kl_key]],
            dim=1,
        )

        # As compared to uSplit kl divergence,
        # more by a factor of 4 just because we do sum and not mean.
        kl_loss = free_bits_kl(kl, self.free_bits).sum()
        # NOTE: at each hierarchy, it is more by a factor of 128/i**2).
        # 128/(2*2) = 32 (bottommost layer)
        # 128/(4*4) = 8
        # 128/(8*8) = 2
        # 128/(16*16) = 0.5 (topmost layer)

        # Normalize the KL-loss w.r.t. the  latent space
        kl_loss = kl_loss / np.prod(self.model.img_shape)
        return kl_loss

    ##### UTILS Methods #####
    def normalize_input(self, x):
        if self.model.normalized_input:
            return x
        return (x - self.data_mean["input"].mean()) / self.data_std["input"].mean()

    def normalize_target(self, target, batch=None):
        return (target - self.data_mean["target"]) / self.data_std["target"]

    def unnormalize_target(self, target_normalized):
        return target_normalized * self.data_std["target"] + self.data_mean["target"]

    ##### ADDITIONAL Methods #####
    # def log_images_for_tensorboard(self, pred, target, img_mmse, label):
    #     clamped_pred = torch.clamp((pred - pred.min()) / (pred.max() - pred.min()), 0, 1)
    #     clamped_mmse = torch.clamp((img_mmse - img_mmse.min()) / (img_mmse.max() - img_mmse.min()), 0, 1)
    #     if target is not None:
    #         clamped_input = torch.clamp((target - target.min()) / (target.max() - target.min()), 0, 1)
    #         img = wandb.Image(clamped_input[None].cpu().numpy())
    #         self.logger.experiment.log({f'target_for{label}': img})
    #         # self.trainer.logger.experiment.add_image(f'target_for{label}', clamped_input[None], self.current_epoch)
    #     for i in range(3):
    #         # self.trainer.logger.experiment.add_image(f'{label}/sample_{i}', clamped_pred[i:i + 1], self.current_epoch)
    #         img = wandb.Image(clamped_pred[i:i + 1].cpu().numpy())
    #         self.logger.experiment.log({f'{label}/sample_{i}': img})

    #     img = wandb.Image(clamped_mmse[None].cpu().numpy())
    #     self.trainer.logger.experiment.log({f'{label}/mmse (100 samples)': img})

    @property
    def global_step(self) -> int:
        """Global step."""
        return self._global_step

    def increment_global_step(self):
        """Increments global step by 1."""
        self._global_step += 1

    def set_params_to_same_device_as(self, correct_device_tensor: torch.Tensor):

        self.model.likelihood.set_params_to_same_device_as(correct_device_tensor)
        if isinstance(self.data_mean, torch.Tensor):
            if self.data_mean.device != correct_device_tensor.device:
                self.data_mean = self.data_mean.to(correct_device_tensor.device)
                self.data_std = self.data_std.to(correct_device_tensor.device)
        elif isinstance(self.data_mean, dict):
            for k, v in self.data_mean.items():
                if v.device != correct_device_tensor.device:
                    self.data_mean[k] = v.to(correct_device_tensor.device)
                    self.data_std[k] = self.data_std[k].to(correct_device_tensor.device)

    def get_mixed_prediction(
        self, prediction, prediction_logvar, data_mean, data_std, channel_weights=None
    ):
        pred_unorm = prediction * data_std["target"] + data_mean["target"]
        if channel_weights is None:
            channel_weights = 1

        if self._input_is_sum:
            mixed_prediction = torch.sum(
                pred_unorm * channel_weights, dim=1, keepdim=True
            )
        else:
            mixed_prediction = torch.mean(
                pred_unorm * channel_weights, dim=1, keepdim=True
            )

        mixed_prediction = (mixed_prediction - data_mean["input"].mean()) / data_std[
            "input"
        ].mean()

        if prediction_logvar is not None:
            if data_std["target"].shape == data_std["input"].shape and torch.all(
                data_std["target"] == data_std["input"]
            ):
                assert channel_weights == 1
                logvar = prediction_logvar
            else:
                var = torch.exp(prediction_logvar)
                var = var * (data_std["target"] / data_std["input"]) ** 2
                if channel_weights != 1:
                    var = var * torch.square(channel_weights)

                # sum of variance.
                mixed_var = 0
                for i in range(var.shape[1]):
                    mixed_var += var[:, i : i + 1]

                logvar = torch.log(mixed_var)
        else:
            logvar = None
        return mixed_prediction, logvar
