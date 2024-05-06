##### TO MOVE IN PL.LIGHTNINGMODULE (training_step, validation_step, losses, ...)
    def log_images_for_tensorboard(self, pred, target, img_mmse, label):
        clamped_pred = torch.clamp((pred - pred.min()) / (pred.max() - pred.min()), 0, 1)
        clamped_mmse = torch.clamp((img_mmse - img_mmse.min()) / (img_mmse.max() - img_mmse.min()), 0, 1)
        if target is not None:
            clamped_input = torch.clamp((target - target.min()) / (target.max() - target.min()), 0, 1)
            img = wandb.Image(clamped_input[None].cpu().numpy())
            self.logger.experiment.log({f'target_for{label}': img})
            # self.trainer.logger.experiment.add_image(f'target_for{label}', clamped_input[None], self.current_epoch)
        for i in range(3):
            # self.trainer.logger.experiment.add_image(f'{label}/sample_{i}', clamped_pred[i:i + 1], self.current_epoch)
            img = wandb.Image(clamped_pred[i:i + 1].cpu().numpy())
            self.logger.experiment.log({f'{label}/sample_{i}': img})

        img = wandb.Image(clamped_mmse[None].cpu().numpy())
        self.trainer.logger.experiment.log({f'{label}/mmse (100 samples)': img})

    @property
    def global_step(self) -> int:
        """Global step."""
        return self._global_step

    def increment_global_step(self):
        """Increments global step by 1."""
        self._global_step += 1

    def _init_lr_scheduler_params(self, config):
        self.lr_scheduler_monitor = config.model.get('monitor', 'val_loss')
        self.lr_scheduler_mode = MetricMonitor(self.lr_scheduler_monitor).mode()

    def configure_optimizers(self):
        optimizer = optim.Adamax(self.parameters(), lr=self.lr, weight_decay=0)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                         self.lr_scheduler_mode,
                                                         patience=self.lr_scheduler_patience,
                                                         factor=0.5,
                                                         min_lr=1e-12,
                                                         verbose=True)

        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': self.lr_scheduler_monitor}

    def get_kl_weight(self):
        if (self.kl_annealing == True):
            # calculate relative weight
            kl_weight = (self.current_epoch - self.kl_start) * (1.0 / self.kl_annealtime)
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

    def get_reconstruction_loss(self,
                                reconstruction,
                                target,
                                input,
                                splitting_mask=None,
                                return_predicted_img=False,
                                likelihood_obj=None):
        output = self._get_reconstruction_loss_vector(reconstruction,
                                                      target,
                                                      input,
                                                      return_predicted_img=return_predicted_img,
                                                      likelihood_obj=likelihood_obj)
        loss_dict = output[0] if return_predicted_img else output
        if splitting_mask is None:
            splitting_mask = torch.ones_like(loss_dict['loss']).bool()

        # print(len(target) - (torch.isnan(loss_dict['loss'])).sum())

        loss_dict['loss'] = loss_dict['loss'][splitting_mask].sum() / len(reconstruction)
        for i in range(1, 1 + target.shape[1]):
            key = 'ch{}_loss'.format(i)
            loss_dict[key] = loss_dict[key][splitting_mask].sum() / len(reconstruction)

        if 'mixed_loss' in loss_dict:
            loss_dict['mixed_loss'] = torch.mean(loss_dict['mixed_loss'])
        if return_predicted_img:
            assert len(output) == 2
            return loss_dict, output[1]
        else:
            return loss_dict

    def _get_reconstruction_loss_vector(self,
                                        reconstruction,
                                        target,
                                        input,
                                        return_predicted_img=False,
                                        likelihood_obj=None):
        """
        Args:
            return_predicted_img: If set to True, the besides the loss, the reconstructed image is also returned.
        """

        output = {
            'loss': None,
            'mixed_loss': None,
        }
        for i in range(1, 1 + target.shape[1]):
            output['ch{}_loss'.format(i)] = None

        if likelihood_obj is None:
            likelihood_obj = self.likelihood

        # Log likelihood
        ll, like_dict = likelihood_obj(reconstruction, target)
        ll = self._get_weighted_likelihood(ll)
        if self.skip_nboundary_pixels_from_loss is not None and self.skip_nboundary_pixels_from_loss > 0:
            pad = self.skip_nboundary_pixels_from_loss
            ll = ll[:, :, pad:-pad, pad:-pad]
            like_dict['params']['mean'] = like_dict['params']['mean'][:, :, pad:-pad, pad:-pad]

        # assert ll.shape[1] == 2, f"Change the code below to handle >2 channels first. ll.shape {ll.shape}"
        output = {
            'loss': compute_batch_mean(-1 * ll),
        }
        if ll.shape[1] > 1:
            for i in range(1, 1 + target.shape[1]):
                output['ch{}_loss'.format(i)] = compute_batch_mean(-ll[:, i - 1])
        else:
            assert ll.shape[1] == 1
            output['ch1_loss'] = output['loss']
            output['ch2_loss'] = output['loss']

        if self.channel_1_w is not None and self.channel_2_w is not None and (self.channel_1_w != 1
                                                                              or self.channel_2_w != 1):
            assert ll.shape[1] == 2, "Only 2 channels are supported for now."
            output['loss'] = (self.channel_1_w * output['ch1_loss'] +
                              self.channel_2_w * output['ch2_loss']) / (self.channel_1_w + self.channel_2_w)

        if self.enable_mixed_rec:
            mixed_pred, mixed_logvar = self.get_mixed_prediction(like_dict['params']['mean'],
                                                                 like_dict['params']['logvar'], self.data_mean,
                                                                 self.data_std)
            if self._multiscale_count is not None and self._multiscale_count > 1:
                assert input.shape[1] == self._multiscale_count
                input = input[:, :1]

            assert input.shape == mixed_pred.shape, "No fucking room for vectorization induced bugs."
            mixed_recons_ll = self.likelihood.log_likelihood(input, {'mean': mixed_pred, 'logvar': mixed_logvar})
            output['mixed_loss'] = compute_batch_mean(-1 * mixed_recons_ll)

        if self._exclusion_loss_weight:
            imgs = like_dict['params']['mean']
            exclusion_loss = compute_exclusion_loss(imgs[:, :1], imgs[:, 1:])
            output['exclusion_loss'] = exclusion_loss

        if return_predicted_img:
            return output, like_dict['params']['mean']

        return output

    def get_kl_divergence_loss_usplit(self, topdown_layer_data_dict):
        kl = torch.cat([kl_layer.unsqueeze(1) for kl_layer in topdown_layer_data_dict['kl']], dim=1)
        # kl.shape = (16,4) 16 is batch size. 4 is number of layers. Values are sum() and so are of the order 30000
        # Example values: 30626.6758, 31028.8145, 29509.8809, 29945.4922, 28919.1875, 29075.2988
        nlayers = kl.shape[1]
        for i in range(nlayers):
            # topdown_layer_data_dict['z'][2].shape[-3:] = 128 * 32 * 32
            norm_factor = np.prod(topdown_layer_data_dict['z'][i].shape[-3:])
            # if self._restricted_kl:
            #     pow = np.power(2,min(i + 1, self._multiscale_count-1))
            #     norm_factor /= pow * pow
            
            kl[:, i] = kl[:, i] / norm_factor

        kl_loss = free_bits_kl(kl, 0.0).mean()
        return kl_loss

    def get_kl_divergence_loss(self, topdown_layer_data_dict, kl_key='kl'):
        # kl[i] for each i has length batch_size
        # resulting kl shape: (batch_size, layers)
        kl = torch.cat([kl_layer.unsqueeze(1) for kl_layer in topdown_layer_data_dict[kl_key]], dim=1)
        # As compared to uSplit kl divergence,
        # more by a factor of 4 just because we do sum and not mean.
        kl_loss = free_bits_kl(kl, self.free_bits).sum()
        # at each hierarchy, it is more by a factor of 128/i**2).
        # 128/(2*2) = 32 (bottommost layer)
        # 128/(4*4) = 8
        # 128/(8*8) = 2
        # 128/(16*16) = 0.5 (topmost layer)
        kl_loss = kl_loss / np.prod(self.img_shape)
        return kl_loss

    def training_step(self, batch, batch_idx, enable_logging=True):
        if self.current_epoch == 0 and batch_idx == 0:
            self.log('val_psnr', 1.0, on_epoch=True)

        x, target = batch[:2]
        x_normalized = self.normalize_input(x)
        if self.reconstruction_mode:
            target_normalized = x_normalized[:, :1].repeat(1, 2, 1, 1)
            target = None
            mask = None
        else:
            target_normalized = self.normalize_target(target)
            mask = ~((target == 0).reshape(len(target), -1).all(dim=1))

        out, td_data = self.forward(x_normalized)
        if self.encoder_no_padding_mode and out.shape[-2:] != target_normalized.shape[-2:]:
            target_normalized = F.center_crop(target_normalized, out.shape[-2:])

        # mask = torch.isnan(target.reshape(len(x), -1)).all(dim=1)
        recons_loss_dict, imgs = self.get_reconstruction_loss(out,
                                                              target_normalized,
                                                              x_normalized,
                                                              mask,
                                                              return_predicted_img=True)
        if self.skip_nboundary_pixels_from_loss:
            pad = self.skip_nboundary_pixels_from_loss
            target_normalized = target_normalized[:, :, pad:-pad, pad:-pad]

        recons_loss = recons_loss_dict['loss'] * self.reconstruction_weight
        if torch.isnan(recons_loss).any():
            recons_loss = 0.0

        if self.loss_type == LossType.ElboMixedReconstruction:
            recons_loss += self.mixed_rec_w * recons_loss_dict['mixed_loss']

            if enable_logging:
                self.log('mixed_reconstruction_loss', recons_loss_dict['mixed_loss'], on_epoch=True)

        if self._exclusion_loss_weight:
            exclusion_loss = recons_loss_dict['exclusion_loss']
            recons_loss += self._exclusion_loss_weight * exclusion_loss
            if enable_logging:
                self.log('exclusion_loss', exclusion_loss, on_epoch=True)

        assert self.loss_type != LossType.ElboWithNbrConsistency

        if self.non_stochastic_version:
            kl_loss = torch.Tensor([0.0]).cuda()
            net_loss = recons_loss
        else:
            if self.loss_type == LossType.DenoiSplitMuSplit:
                msg = f"For the loss type {LossType.name(self.loss_type)}, kl_loss_formulation must be denoisplit_usplit"
                assert self.kl_loss_formulation == 'denoisplit_usplit', msg
                assert self._denoisplit_w is not None and self._usplit_w is not None

                if self.predict_logvar is not None:
                    out_mean, _ = out.chunk(2, dim=1)
                else:
                    out_mean  = out
                
                kl_key_denoisplit = 'kl_restricted' if self._restricted_kl else 'kl'
                denoisplit_kl = self.get_kl_divergence_loss(td_data, kl_key=kl_key_denoisplit)
                usplit_kl = self.get_kl_divergence_loss_usplit(td_data)
                kl_loss = self._denoisplit_w * denoisplit_kl + self._usplit_w * usplit_kl
                kl_loss = self.kl_weight * kl_loss

                recons_loss_nm = -1*self.likelihood_NM(out_mean, target_normalized)[0].mean()
                recons_loss_gm = -1*self.likelihood_gm(out, target_normalized)[0].mean()
                recons_loss = self._denoisplit_w * recons_loss_nm + self._usplit_w * recons_loss_gm
                
            elif self.kl_loss_formulation == 'usplit':
                kl_loss = self.get_kl_weight() * self.get_kl_divergence_loss_usplit(td_data)
            elif self.kl_loss_formulation in ['', 'denoisplit']:
                kl_loss = self.get_kl_weight() * self.get_kl_divergence_loss(td_data)
            net_loss = recons_loss + kl_loss

        if enable_logging:
            for i, x in enumerate(td_data['debug_qvar_max']):
                self.log(f'qvar_max:{i}', x.item(), on_epoch=True)

            self.log('reconstruction_loss', recons_loss_dict['loss'], on_epoch=True)
            self.log('kl_loss', kl_loss, on_epoch=True)
            self.log('training_loss', net_loss, on_epoch=True)
            self.log('lr', self.lr, on_epoch=True)
            if self._tethered_ch2_scalar is not None:
                self.log('tethered_ch2_scalar', self._tethered_ch2_scalar, on_epoch=True)
                self.log('tethered_ch1_scalar', self._tethered_ch1_scalar, on_epoch=True)

            # self.log('grad_norm_bottom_up', self.grad_norm_bottom_up, on_epoch=True)
            # self.log('grad_norm_top_down', self.grad_norm_top_down, on_epoch=True)

        output = {
            'loss': net_loss,
            'reconstruction_loss': recons_loss.detach() if isinstance(recons_loss, torch.Tensor) else recons_loss,
            'kl_loss': kl_loss.detach(),
        }
        # https://github.com/openai/vdvae/blob/main/train.py#L26
        if torch.isnan(net_loss).any():
            return None

        return output

    def set_params_to_same_device_as(self, correct_device_tensor):
        self.likelihood.set_params_to_same_device_as(correct_device_tensor)
        if isinstance(self.data_mean, torch.Tensor):
            if self.data_mean.device != correct_device_tensor.device:
                self.data_mean = self.data_mean.to(correct_device_tensor.device)
                self.data_std = self.data_std.to(correct_device_tensor.device)
        elif isinstance(self.data_mean, dict):
            for k, v in self.data_mean.items():
                if v.device != correct_device_tensor.device:
                    self.data_mean[k] = v.to(correct_device_tensor.device)
                    self.data_std[k] = self.data_std[k].to(correct_device_tensor.device)

    def validation_step(self, batch, batch_idx):
        x, target = batch[:2]
        self.set_params_to_same_device_as(x)
        x_normalized = self.normalize_input(x)
        if self.reconstruction_mode:
            target_normalized = x_normalized[:, :1].repeat(1, 2, 1, 1)
            target = None
            mask = None
        else:
            target_normalized = self.normalize_target(target)
            mask = ~((target == 0).reshape(len(target), -1).all(dim=1))

        out, td_data = self.forward(x_normalized)
        if self.encoder_no_padding_mode and out.shape[-2:] != target_normalized.shape[-2:]:
            target_normalized = F.center_crop(target_normalized, out.shape[-2:])

        recons_loss_dict, recons_img = self.get_reconstruction_loss(out,
                                                                    target_normalized,
                                                                    x_normalized,
                                                                    mask,
                                                                    return_predicted_img=True)
        if self._dump_kth_frame_prediction is not None:
            if self.current_epoch == 0:
                self._val_frame_creator.update_target(target.cpu().numpy().astype(np.int32),
                                                      batch[-1].cpu().numpy().astype(np.int32))
            if self.current_epoch == 0 or self.current_epoch % self._dump_epoch_interval == 0:
                imgs = self.unnormalize_target(recons_img).cpu().numpy().astype(np.int32)
                self._val_frame_creator.update(imgs, batch[-1].cpu().numpy().astype(np.int32))

        if self.skip_nboundary_pixels_from_loss:
            pad = self.skip_nboundary_pixels_from_loss
            target_normalized = target_normalized[:, :, pad:-pad, pad:-pad]

        channels_rinvpsnr = []
        for i in range(recons_img.shape[1]):
            self.channels_psnr[i].update(recons_img[:, i], target_normalized[:, i])
            psnr = RangeInvariantPsnr(target_normalized[:, i].clone(), recons_img[:, i].clone())
            channels_rinvpsnr.append(psnr)
            psnr = torch_nanmean(psnr).item()
            self.log(f'val_psnr_l{i+1}', psnr, on_epoch=True)

        recons_loss = recons_loss_dict['loss']
        if torch.isnan(recons_loss).any():
            return

        self.log('val_loss', recons_loss, on_epoch=True)
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
            self.log('val_psnr', psnr, on_epoch=True)
        else:
            self.log('val_psnr', 0.0, on_epoch=True)

        if self._dump_kth_frame_prediction is not None:
            if self.current_epoch == 1:
                self._val_frame_creator.dump_target()
            if self.current_epoch == 0 or self.current_epoch % self._dump_epoch_interval == 0:
                self._val_frame_creator.dump(self.current_epoch)
                self._val_frame_creator.reset()

        if self.mixed_rec_w_step:
            self.mixed_rec_w = max(self.mixed_rec_w - self.mixed_rec_w_step, 0.0)
            self.log('mixed_rec_w', self.mixed_rec_w, on_epoch=True)
