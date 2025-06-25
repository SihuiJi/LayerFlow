from typing import Dict, Union

import torch
import torch.nn as nn

from ...util import append_dims, instantiate_from_config


class Denoiser(nn.Module):
    def __init__(self, weighting_config, scaling_config):
        super().__init__()

        self.weighting = instantiate_from_config(weighting_config)
        self.scaling = instantiate_from_config(scaling_config)

    def possibly_quantize_sigma(self, sigma):
        return sigma

    def possibly_quantize_c_noise(self, c_noise):
        return c_noise

    def w(self, sigma):
        return self.weighting(sigma)

    def forward(
        self,
        network: nn.Module,
        input: torch.Tensor,
        sigma: torch.Tensor,
        cond: Dict,
        **additional_model_inputs,
    ) -> torch.Tensor:
        sigma = self.possibly_quantize_sigma(sigma)
        sigma_shape = sigma.shape
        sigma = append_dims(sigma, input.ndim)
        c_skip, c_out, c_in, c_noise = self.scaling(sigma, **additional_model_inputs)
        c_noise = self.possibly_quantize_c_noise(c_noise.reshape(sigma_shape))
        # print('train cond')
        # for key in cond:
        #     print(key, cond[key].shape)
        #     crossattn 6 226 4096
        # print('input',input.shape)
        # input torch.Size([3, 4, 16, 60, 90])
        # input torch.Size([2, 4, 16, 60, 90])
        
        # input torch.Size([2, 12, 16, 60, 90])
        # print('cond_input',additional_model_inputs["cond_input"].shape)
        # torch.Size([3, 2, 16, 60, 90])
        # torch.Size([2, 2, 16, 60, 90])
        # print('c_in',c_in.shape)
        # print('c_noise',c_noise.shape)
        # print('cond',cond)
        # print('c_out',c_out.shape)
        # print('c_skip',c_skip.shape)
        # from IPython import embed
        # embed()
        # print(network(input * c_in, c_noise, cond, **additional_model_inputs).shape)
        # torch.Size([8, 14, 16, 60, 90])
        # print(c_out.shape)
        # print(input.shape)
        # torch.Size([8, 12, 16, 60, 90])
        # print(c_skip.shape)

        # input torch.Size([1, 8, 16, 60, 90])
        # cond_input torch.Size([1, 9, 16, 60, 90])
        # torch.Size([1, 12, 16, 60, 90]) torch.Size([1, 8, 16, 60, 90])
        return network(input * c_in, c_noise, cond, **additional_model_inputs) * c_out + input * c_skip


class DiscreteDenoiser(Denoiser):
    def __init__(
        self,
        weighting_config,
        scaling_config,
        num_idx,
        discretization_config,
        do_append_zero=False,
        quantize_c_noise=True,
        flip=True,
        is_stage1=False
    ):
        super().__init__(weighting_config, scaling_config)
        sigmas = instantiate_from_config(discretization_config)(num_idx, do_append_zero=do_append_zero, flip=flip)
        self.sigmas = sigmas
        # self.register_buffer("sigmas", sigmas)
        self.quantize_c_noise = quantize_c_noise
        self.is_stage1 = is_stage1

    def sigma_to_idx(self, sigma):
        dists = sigma - self.sigmas.to(sigma.device)[:, None]
        return dists.abs().argmin(dim=0).view(sigma.shape)

    def idx_to_sigma(self, idx):
        return self.sigmas.to(idx.device)[idx]

    def possibly_quantize_sigma(self, sigma):
        return self.idx_to_sigma(self.sigma_to_idx(sigma))

    def possibly_quantize_c_noise(self, c_noise):
        if self.quantize_c_noise:
            return self.sigma_to_idx(c_noise)
        else:
            return c_noise
