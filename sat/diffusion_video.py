import random

import math
from typing import Any, Dict, List, Tuple, Union
from omegaconf import ListConfig
import torch.nn.functional as F

from sat.helpers import print_rank0
import torch
from torch import nn

from sgm.modules import UNCONDITIONAL_CONFIG
from sgm.modules.autoencoding.temporal_ae import VideoDecoder
from sgm.modules.diffusionmodules.wrappers import OPENAIUNETWRAPPER
from sgm.util import (
    default,
    disabled_train,
    get_obj_from_str,
    instantiate_from_config,
    log_txt_as_img,
)
import gc
from sat import mpu
from einops import rearrange
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution

class SATVideoDiffusionEngine(nn.Module):
    def __init__(self, args, **kwargs):
        super().__init__()

        model_config = args.model_config
        # model args preprocess
        log_keys = model_config.get("log_keys", None)
        input_key = model_config.get("input_key", "mp4")
        network_config = model_config.get("network_config", None)
        network_wrapper = model_config.get("network_wrapper", None)
        denoiser_config = model_config.get("denoiser_config", None)
        sampler_config = model_config.get("sampler_config", None)
        conditioner_config = model_config.get("conditioner_config", None)
        first_stage_config = model_config.get("first_stage_config", None)
        loss_fn_config = model_config.get("loss_fn_config", None)
        scale_factor = model_config.get("scale_factor", 1.0)
        latent_input = model_config.get("latent_input", False)
        disable_first_stage_autocast = model_config.get("disable_first_stage_autocast", False)
        no_cond_log = model_config.get("disable_first_stage_autocast", False)
        not_trainable_prefixes = model_config.get("not_trainable_prefixes", ["first_stage_model", "conditioner"])
        compile_model = model_config.get("compile_model", False)
        en_and_decode_n_samples_a_time = model_config.get("en_and_decode_n_samples_a_time", None)
        lr_scale = model_config.get("lr_scale", None)
        lora_train = model_config.get("lora_train", False)
        self.use_pd = model_config.get("use_pd", False)  # progressive distillation

        self.log_keys = log_keys
        self.input_key = input_key
        self.not_trainable_prefixes = not_trainable_prefixes
        self.en_and_decode_n_samples_a_time = en_and_decode_n_samples_a_time
        self.lr_scale = lr_scale
        self.lora_train = lora_train
        self.noised_image_input = model_config.get("noised_image_input", False)
        self.ref_image_input = model_config.get("ref_image_input", False)
        self.train_all_tasks = model_config.get("train_all_tasks", False)
        self.train_2_tasks = model_config.get("train_2_tasks", False)
        self.bg2fg = model_config.get("bg2fg", False)
        self.fg2bg = model_config.get("fg2bg", False)
        self.noised_image_all_concat = model_config.get("noised_image_all_concat", False) # False
        self.cat_mode = model_config.get("cat_mode", False)
        self.noised_image_dropout = model_config.get("noised_image_dropout", 0.0)
        self.image_cond = model_config.get("image_cond", False)
        self.video_cond = model_config.get("video_cond", False)
        self.seg_only = model_config.get("seg_only", False)
        if args.fp16:
            dtype = torch.float16
            dtype_str = "fp16"
        elif args.bf16:
            dtype = torch.bfloat16
            dtype_str = "bf16"
        else:
            dtype = torch.float32
            dtype_str = "fp32"
        self.dtype = dtype
        self.dtype_str = dtype_str

        network_config["params"]["dtype"] = dtype_str
        model = instantiate_from_config(network_config)
        self.model = get_obj_from_str(default(network_wrapper, OPENAIUNETWRAPPER))(
            model, compile_model=compile_model, dtype=dtype
        )

        self.denoiser = instantiate_from_config(denoiser_config)
        self.sampler = instantiate_from_config(sampler_config) if sampler_config is not None else None
        self.conditioner = instantiate_from_config(default(conditioner_config, UNCONDITIONAL_CONFIG))

        self._init_first_stage(first_stage_config)

        self.loss_fn = instantiate_from_config(loss_fn_config) if loss_fn_config is not None else None

        self.latent_input = latent_input
        self.scale_factor = scale_factor
        self.disable_first_stage_autocast = disable_first_stage_autocast
        self.no_cond_log = no_cond_log
        self.device = args.device

    def disable_untrainable_params(self):
        
        total_trainable = 0
        for n, p in self.named_parameters():
            if p.requires_grad == False:
                continue
            flag = False
            for prefix in self.not_trainable_prefixes:
                if n.startswith(prefix) or prefix == "all" or prefix=='stage1' or prefix=='stage2' or prefix=='stage1_v3' or prefix=='stage2_v3' or prefix=='part_6' or prefix=='stage1_v3_part_6' or prefix=='stage2_v3_part_6' or prefix=='part_6_proj' or prefix=='part_6_txt' or prefix=='stage1_v3_part_6_txt' or prefix=='stage2_v3_part_6_txt' or prefix=='stage2_v3_part_6_new':
                    flag = True
                    break

            lora_prefix_1 = ["matrix_A", "matrix_B"]
            lora_prefix_2 = ["matrix_C", "matrix_D"]
            if self.not_trainable_prefixes==['all_no_lora']:
                for prefix in lora_prefix_1:
                    if prefix in n:
                        flag = True
                        break
            elif self.not_trainable_prefixes==['part_6']:
                if 'model.diffusion_model.transformer.layers.' in n and int(n.split('.')[4])%6==5:
                    flag = False
                elif 'model.diffusion_model.mixins.adaln_layer.key_layernorm_list.' in n and int(n.split('.')[5])%6==5:
                    flag = False
                elif 'model.diffusion_model.mixins.adaln_layer.query_layernorm_list.' in n and int(n.split('.')[5])%6==5:
                    flag = False
                elif 'model.diffusion_model.mixins.adaln_layer.adaLN_modulations.' in n and int(n.split('.')[5])%6==5:
                    flag = False
            elif self.not_trainable_prefixes==['part_6_txt']:
                if 'model.diffusion_model.transformer.layers.' in n and int(n.split('.')[4])%6==5:
                    flag = False
                elif 'model.diffusion_model.mixins.adaln_layer.key_layernorm_list.' in n and int(n.split('.')[5])%6==5:
                    flag = False
                elif 'model.diffusion_model.mixins.adaln_layer.query_layernorm_list.' in n and int(n.split('.')[5])%6==5:
                    flag = False
                elif 'model.diffusion_model.mixins.adaln_layer.adaLN_modulations.' in n and int(n.split('.')[5])%6==5:
                    flag = False
                elif 'txt_embedder' in n:
                    flag = False
            elif self.not_trainable_prefixes==['part_6_proj']:
                if 'model.diffusion_model.transformer.layers.' in n and int(n.split('.')[4])%6==5:
                    flag = False
                elif 'model.diffusion_model.mixins.adaln_layer.key_layernorm_list.' in n and int(n.split('.')[5])%6==5:
                    flag = False
                elif 'model.diffusion_model.mixins.adaln_layer.query_layernorm_list.' in n and int(n.split('.')[5])%6==5:
                    flag = False
                elif 'model.diffusion_model.mixins.adaln_layer.adaLN_modulations.' in n and int(n.split('.')[5])%6==5:
                    flag = False
                elif 'model.diffusion_model.mixins.patch_embed.new_proj' in n:
                    flag = False
                elif 'model.diffusion_model.mixins.final_layer.new_linear' in n:
                    flag = False
            elif self.not_trainable_prefixes==['stage1']:
                for prefix in lora_prefix_1:
                    if prefix in n and int(n.split('.')[4])%2==1:
                        flag = False
                        break
            elif self.not_trainable_prefixes==['stage2']:
                for prefix in lora_prefix_1:
                    if prefix in n and int(n.split('.')[4])%2==0:
                        flag = False
                        break
            elif self.not_trainable_prefixes==['stage1_v3']:
                for prefix in lora_prefix_1:
                    if prefix in n:
                        flag = False
                        break
            elif self.not_trainable_prefixes==['stage1_v3_part_6']:
                for prefix in lora_prefix_1:
                    if prefix in n:
                        if 'model.diffusion_model.transformer.layers.' in n and int(n.split('.')[4])%6==5:
                            flag = False
                            break
                        elif 'model.diffusion_model.mixins.adaln_layer.key_layernorm_list.' in n and int(n.split('.')[5])%6==5:
                            flag = False
                            break
                        elif 'model.diffusion_model.mixins.adaln_layer.query_layernorm_list.' in n and int(n.split('.')[5])%6==5:
                            flag = False
                            break
                        elif 'model.diffusion_model.mixins.adaln_layer.adaLN_modulations.' in n and int(n.split('.')[5])%6==5:
                            flag = False
                            break
            elif self.not_trainable_prefixes==['stage1_v3_part_6_txt']:
                for prefix in lora_prefix_1:
                    if prefix in n:
                        if 'model.diffusion_model.transformer.layers.' in n and int(n.split('.')[4])%6==5:
                            flag = False
                            break
                        elif 'model.diffusion_model.mixins.adaln_layer.key_layernorm_list.' in n and int(n.split('.')[5])%6==5:
                            flag = False
                            break
                        elif 'model.diffusion_model.mixins.adaln_layer.query_layernorm_list.' in n and int(n.split('.')[5])%6==5:
                            flag = False
                            break
                        elif 'model.diffusion_model.mixins.adaln_layer.adaLN_modulations.' in n and int(n.split('.')[5])%6==5:
                            flag = False
                            break
                if 'txt_embedder' in n:
                    flag = False
            elif self.not_trainable_prefixes==['stage2_v3_part_6']:
                for prefix in lora_prefix_2:
                    if prefix in n:
                        if 'model.diffusion_model.transformer.layers.' in n and int(n.split('.')[4])%6==5:
                            flag = False
                            break
                        elif 'model.diffusion_model.mixins.adaln_layer.key_layernorm_list.' in n and int(n.split('.')[5])%6==5:
                            flag = False
                            break
                        elif 'model.diffusion_model.mixins.adaln_layer.query_layernorm_list.' in n and int(n.split('.')[5])%6==5:
                            flag = False
                            break
                        elif 'model.diffusion_model.mixins.adaln_layer.adaLN_modulations.' in n and int(n.split('.')[5])%6==5:
                            flag = False
                            break
            elif self.not_trainable_prefixes==['stage2_v3_part_6_new']:
                if 'model.diffusion_model.transformer.layers.' in n and int(n.split('.')[4])%6==5 and 'matrix_' not in n:
                    flag = False
                elif 'model.diffusion_model.mixins.adaln_layer.key_layernorm_list.' in n and int(n.split('.')[5])%6==5 and 'matrix_' not in n:
                    flag = False
                elif 'model.diffusion_model.mixins.adaln_layer.query_layernorm_list.' in n and int(n.split('.')[5])%6==5 and 'matrix_' not in n:
                    flag = False
                elif 'model.diffusion_model.mixins.adaln_layer.adaLN_modulations.' in n and int(n.split('.')[5])%6==5 and 'matrix_' not in n:
                    flag = False
            elif self.not_trainable_prefixes==['stage2_v3_part_6_txt']:
                for prefix in lora_prefix_2:
                    if prefix in n:
                        if 'model.diffusion_model.transformer.layers.' in n and int(n.split('.')[4])%6==5:
                            flag = False
                            break
                        elif 'model.diffusion_model.mixins.adaln_layer.key_layernorm_list.' in n and int(n.split('.')[5])%6==5:
                            flag = False
                            break
                        elif 'model.diffusion_model.mixins.adaln_layer.query_layernorm_list.' in n and int(n.split('.')[5])%6==5:
                            flag = False
                            break
                        elif 'model.diffusion_model.mixins.adaln_layer.adaLN_modulations.' in n and int(n.split('.')[5])%6==5:
                            flag = False
                            break
                        elif 'txt_embedder' in n:
                            flag = False
                            break
                if 'txt_embedder' in n:
                    flag = False
            elif self.not_trainable_prefixes==['stage2_v3']:
                for prefix in lora_prefix_2:
                    if prefix in n:
                        flag = False
                        break
            else:
                for prefix in lora_prefix_1:
                    if prefix in n:
                        flag = False
                        break
                
            if flag:
                p.requires_grad_(False)
            else:
                total_trainable += p.numel()
        
        trainable_dict = {}
        for name, para in self.named_parameters():
            if para.requires_grad:
                trainable_dict[name] = para
        with open(f'trainable_params_{self.not_trainable_prefixes[0]}.txt', 'w') as f:
            for k,v in trainable_dict.items():
                f.write(k+'\t'+str(v.shape)+'\n')
        print_rank0("***** Total trainable parameters: " + str(total_trainable) + " *****")
        
    def reinit(self, mixin_names=[], parent_model=None):
        pass

    def _init_first_stage(self, config):
        model = instantiate_from_config(config).eval()
        model.train = disabled_train
        for param in model.parameters():
            param.requires_grad = False
        self.first_stage_model = model

    def forward(self, x, batch):
        loss = self.loss_fn(self.model, self.denoiser, self.conditioner, x, batch)
        loss_mean = loss.mean()
        loss_dict = {"loss": loss_mean}
        return loss_mean, loss_dict

    def add_noise_to_first_frame(self, image):
        sigma = torch.normal(mean=-3.0, std=0.5, size=(image.shape[0],)).to(self.device)
        sigma = torch.exp(sigma).to(image.dtype)
        image_noise = torch.randn_like(image) * sigma[:, None, None, None, None]
        image = image + image_noise
        return image

    def shared_step(self, batch: Dict) -> Any:
        x = self.get_input(batch)

        if self.lr_scale is not None:
            lr_x = F.interpolate(x, scale_factor=1 / self.lr_scale, mode="bilinear", align_corners=False)
            lr_x = F.interpolate(lr_x, scale_factor=self.lr_scale, mode="bilinear", align_corners=False)
            lr_z = self.encode_first_stage(lr_x, batch)
            batch["lr_input"] = lr_z

        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = self.encode_first_stage(x, batch)
        x = x.permute(0, 2, 1, 3, 4).contiguous()

        if self.noised_image_input:
            if self.train_all_tasks:
                task_idx = random.choice([0, 1, 2, 3])
                batch["task_idx"] = task_idx
                if task_idx!=0:
                    if self.video_cond:
                        video = x[:, (task_idx-1)*4:task_idx*4]
                        batch["cond_input"] = video
                        if task_idx ==1:
                            x = x[:, 4:] 
                        elif task_idx ==2:
                            x = torch.cat([x[:, :4],x[:, -4:]],dim=1)
                        elif task_idx ==3:
                            x = x[:, :-4] 
            elif self.train_2_tasks:
                task_idx = random.choice([1, 2])
                batch["task_idx"] = task_idx
                if self.video_cond:
                    video = x[:, (task_idx-1)*4:task_idx*4]
                    batch["cond_input"] = video
                    if task_idx ==1:
                        x = x[:, 4:] 
                    elif task_idx ==2:
                        x = torch.cat([x[:, :4],x[:, -4:]],dim=1)
            elif self.bg2fg:
                batch["task_idx"] = 2
                if self.video_cond:
                    video = x[:, 9:13]
                    batch["cond_input"] = video
                    x = torch.cat([x[:, :9],x[:, -4:]],dim=1)
            elif self.fg2bg:
                batch["task_idx"] = 1
                if self.video_cond:
                    video = x[:, :9]
                    batch["cond_input"] = video
                    x = x[:, 9:]
            elif self.seg_only:
                if self.video_cond:
                    video = x[:, -4:]
                    batch["cond_input"] = video
                    x = x[:, :-4]
            else:
                is_seg = random.choice([True, False])
                if is_seg:
                    if self.video_cond:
                        video = x[:, -4:]
                        batch["cond_input"] = video
                        x = x[:, :-4]
        if self.ref_image_input:
            if self.video_cond:
                video = x[:, -2:]
                if random.random() < self.noised_image_dropout:
                    video = torch.zeros_like(video)
                batch["concat_images"] = video
                x = x[:, :-2]
        if self.cat_mode:
            batch["cat_mode"]=True
        gc.collect()
        torch.cuda.empty_cache()
        loss, loss_dict = self(x, batch)
        return loss, loss_dict

    def get_input(self, batch):
        return batch[self.input_key].to(self.dtype)

    @torch.no_grad()
    def decode_first_stage(self, z):
        z = 1.0 / self.scale_factor * z
        n_samples = default(self.en_and_decode_n_samples_a_time, z.shape[0])
        n_rounds = math.ceil(z.shape[0] / n_samples)
        all_out = []
        with torch.autocast("cuda", enabled=not self.disable_first_stage_autocast):
            for n in range(n_rounds):
                if isinstance(self.first_stage_model.decoder, VideoDecoder):
                    kwargs = {"timesteps": len(z[n * n_samples : (n + 1) * n_samples])}
                else:
                    kwargs = {}
                use_cp = False
                out = self.first_stage_model.decode(z[n * n_samples : (n + 1) * n_samples], **kwargs)
                all_out.append(out)
                gc.collect()
                torch.cuda.empty_cache()
        out = torch.cat(all_out, dim=0)
        return out

    @torch.no_grad()
    def encode_first_stage(self, x, batch):
        if not batch["load_tensors"][0]:
            if self.seg_only or (not self.noised_image_input):
                pass
            else:
                x = torch.cat([x[:,:,0:1,:,:], x], dim=2)
            frame = x.shape[2]
        else:
            frame = x.shape[1]
        if frame > 1 and batch["load_tensors"][0]:
            x = x.permute(0, 2, 1, 3, 4).contiguous()
            x = DiagonalGaussianDistribution(x).sample()
            return x * self.scale_factor  # already encoded
        use_cp = False
        n_samples = default(self.en_and_decode_n_samples_a_time, x.shape[0])
        n_rounds = math.ceil(x.shape[0] / n_samples)
        all_out = []
        with torch.autocast("cuda", enabled=not self.disable_first_stage_autocast):
            for n in range(n_rounds):
                out = self.first_stage_model.encode(x[n * n_samples : (n + 1) * n_samples])
                all_out.append(out)
                gc.collect()
                torch.cuda.empty_cache()
        z = torch.cat(all_out, dim=0)
        z = self.scale_factor * z

        return z
    
    @torch.no_grad()
    def sample(
        self,
        cond: Dict,
        uc: Union[Dict, None] = None,
        batch_size: int = 16,
        shape: Union[None, Tuple, List] = None,
        prefix=None,
        cond_image=None,
        concat_images=None,
        task_idx=0,
        cat_mode=False,
        **kwargs,
    ):

        randn = torch.randn(batch_size, *shape).to(torch.float32).to(self.device)
        if hasattr(self, "seeded_noise"):
            randn = self.seeded_noise(randn)
        
        if prefix is not None:
            randn = torch.cat([randn[:, :(randn.shape[1] - prefix.shape[1])], prefix], dim=1)
        # broadcast noise
        mp_size = mpu.get_model_parallel_world_size()
        if mp_size > 1:
            global_rank = torch.distributed.get_rank() // mp_size
            src = global_rank * mp_size
            torch.distributed.broadcast(randn, src=src, group=mpu.get_model_parallel_group())

        scale = None
        scale_emb = None

        denoiser = lambda input, sigma, c, **addtional_model_inputs: self.denoiser(
            self.model, input, sigma, c, concat_images=concat_images, **addtional_model_inputs
        )

        samples = self.sampler(denoiser, randn, cond, uc=uc, scale=scale, scale_emb=scale_emb, is_stage1 = self.denoiser.is_stage1, cond_image=cond_image, task_idx=task_idx, cat_mode=cat_mode)
        samples = samples.to(self.dtype)
        return samples

    @torch.no_grad()
    def log_conditionings(self, batch: Dict, n: int) -> Dict:
        """
        Defines heuristics to log different conditionings.
        These can be lists of strings (text-to-image), tensors, ints, ...
        """
        image_h, image_w = batch[self.input_key].shape[3:]
        log = dict()

        for embedder in self.conditioner.embedders:
            if ((self.log_keys is None) or (embedder.input_key in self.log_keys)) and not self.no_cond_log:
                x = batch[embedder.input_key][:n]
                if isinstance(x, torch.Tensor):
                    if x.dim() == 1:
                        # class-conditional, convert integer to string
                        x = [str(x[i].item()) for i in range(x.shape[0])]
                        xc = log_txt_as_img((image_h, image_w), x, size=image_h // 4)
                    elif x.dim() == 2:
                        # size and crop cond and the like
                        x = ["x".join([str(xx) for xx in x[i].tolist()]) for i in range(x.shape[0])]
                        xc = log_txt_as_img((image_h, image_w), x, size=image_h // 20)
                    else:
                        raise NotImplementedError()
                elif isinstance(x, (List, ListConfig)):
                    if isinstance(x[0], str):
                        xc = log_txt_as_img((image_h, image_w), x, size=image_h // 20)
                    else:
                        raise NotImplementedError()
                else:
                    raise NotImplementedError()
                log[embedder.input_key] = xc
        return log

    @torch.no_grad()
    def log_video(
        self,
        batch: Dict,
        N: int = 8,
        ucg_keys: List[str] = None,
        only_log_video_latents=False,
        **kwargs,
    ) -> Dict:
        conditioner_input_keys = [e.input_key for e in self.conditioner.embedders]
        if ucg_keys:
            assert all(map(lambda x: x in conditioner_input_keys, ucg_keys)), (
                "Each defined ucg key for sampling must be in the provided conditioner input keys,"
                f"but we have {ucg_keys} vs. {conditioner_input_keys}"
            )
        else:
            ucg_keys = conditioner_input_keys
        log = dict()
        x = self.get_input(batch)
        
        c, uc = self.conditioner.get_unconditional_conditioning(
            batch,
            force_uc_zero_embeddings=ucg_keys if len(self.conditioner.embedders) > 0 else [],
        )

        sampling_kwargs = {}

        N = min(x.shape[0], N)
        for k in c:
            frames = c[k].shape[0]//N
        
        x = x.to(self.device)[:N]

        if not self.latent_input:
            log["inputs"] = x.to(torch.float32)
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        z = self.encode_first_stage(x, batch)
        if not only_log_video_latents:
            log["reconstructions"] = self.decode_first_stage(z).to(torch.float32)
            log["reconstructions"] = log["reconstructions"].permute(0, 2, 1, 3, 4).contiguous()
        z = z.permute(0, 2, 1, 3, 4).contiguous()
        log.update(self.log_conditionings(batch, N))
        for k in c:
            if isinstance(c[k], torch.Tensor):
                c[k], uc[k] = map(lambda y: y[k][:N*frames].to(self.device), (c, uc))
        if self.cat_mode:
            sampling_kwargs["cat_mode"] = self.cat_mode
        if 'is_seg' in batch:
            if batch['is_seg'][0]=="True":
                if self.noised_image_input:
                    if self.image_cond:
                        image = x[:, :, 0:1]
                        image = self.add_noise_to_first_frame(image)
                        image = self.encode_first_stage(image, batch)
                        image = image.permute(0, 2, 1, 3, 4).contiguous()
                        image = torch.concat([image, torch.zeros_like(z[:, 1:])], dim=1)
                        samples = self.sample(c, shape=z.shape[1:], uc=uc, batch_size=N, cond_image=image, **sampling_kwargs)  # b t c h w  
                    elif self.video_cond:
                        video = z[:,-4:]
                        z = z[:,:-4]
                        samples = self.sample(c, shape=z.shape[1:], uc=uc, batch_size=N, cond_image=video, **sampling_kwargs)  # b t c h w
                elif self.ref_image_input:
                    if self.video_cond:
                        video = z[:,-2:]
                        c["concat"] = video
                        uc["concat"] = video
                        z = z[:,:-2]
                        samples = self.sample(c, shape=z.shape[1:], uc=uc, batch_size=N, **sampling_kwargs)  # b t c h w  
                else:
                    samples = self.sample(c, shape=z.shape[1:], uc=uc, batch_size=N, **sampling_kwargs)  # b t c h w
            elif batch['is_seg'][0]=="False":
                samples = self.sample(c, shape=z.shape[1:], uc=uc, batch_size=N, **sampling_kwargs)  # b t c h w
            elif 0<=batch['is_seg'][0]<=3:
                if self.train_all_tasks or self.train_2_tasks or self.bg2fg or self.fg2bg:
                    task_idx = batch['is_seg'][0]
                    if task_idx ==1:
                        video = z[:, :9]
                        z = z[:, 9:] 
                    elif task_idx ==2:
                        video = z[:, 9:13]
                        z = torch.cat([z[:, :9],z[:, -4:]],dim=1)
                    sampling_kwargs["task_idx"] = task_idx
                    samples = self.sample(c, shape=z.shape[1:], uc=uc, batch_size=N, cond_image=video, **sampling_kwargs)  # b t c h w
            else:
                samples = self.sample(c, shape=z.shape[1:], uc=uc, batch_size=N, **sampling_kwargs)  # b t c h w
        else:
            samples = self.sample(c, shape=z.shape[1:], uc=uc, batch_size=N, **sampling_kwargs)  # b t c h w
        if 'is_seg' in batch:
            if batch['is_seg'][0]=="True":
                if self.ref_image_input or self.noised_image_input:
                    samples = torch.cat([samples,video],dim=1)
            elif batch['is_seg'][0]=="False":
                pass
            elif 0<=batch['is_seg'][0]<=3:
                if self.ref_image_input or self.noised_image_input:
                    if self.train_all_tasks or self.train_2_tasks or self.bg2fg or self.fg2bg:
                        task_idx = batch['is_seg'][0]
                        if task_idx ==1:
                            samples = torch.cat([video, samples],dim=1)
                        elif task_idx ==2:
                            samples = torch.cat([samples[:, :9],video,samples[:, -4:]],dim=1)

        samples = samples.permute(0, 2, 1, 3, 4).contiguous()
        if only_log_video_latents:
            latents = 1.0 / self.scale_factor * samples
            log["latents"] = latents
        else:
            samples = self.decode_first_stage(samples).to(torch.float32)
            samples = samples.permute(0, 2, 1, 3, 4).contiguous()
            log["samples"] = samples
        return log
