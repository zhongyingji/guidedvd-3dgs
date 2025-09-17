import numpy as np
from tqdm import tqdm
import torch
from lvdm.models.utils_diffusion import make_ddim_sampling_parameters, make_ddim_timesteps, rescale_noise_cfg
from lvdm.common import noise_like
from lvdm.common import extract_into_tensor
import copy

class DDIMSamplerGuidance(object):
    def __init__(self, model, schedule="linear", **kwargs):
        super().__init__()
        self.model = model
        self.ddpm_num_timesteps = model.num_timesteps
        self.schedule = schedule
        self.counter = 0

    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != torch.device("cuda"):
                attr = attr.to(torch.device("cuda"))
        setattr(self, name, attr)

    def make_schedule(self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0., verbose=True):
        self.ddim_timesteps = make_ddim_timesteps(ddim_discr_method=ddim_discretize, num_ddim_timesteps=ddim_num_steps,
                                                  num_ddpm_timesteps=self.ddpm_num_timesteps,verbose=verbose)
        alphas_cumprod = self.model.alphas_cumprod
        assert alphas_cumprod.shape[0] == self.ddpm_num_timesteps, 'alphas have to be defined for each timestep'
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(self.model.device)

        if self.model.use_dynamic_rescale:
            self.ddim_scale_arr = self.model.scale_arr[self.ddim_timesteps]
            # self.ddim_scale_arr_prev = torch.cat([self.ddim_scale_arr[0:1], self.ddim_scale_arr[:-1]])
            # fix a bug
            self.ddim_scale_arr_prev = torch.cat([self.model.scale_arr[0:1], self.ddim_scale_arr[:-1]])

        self.register_buffer('betas', to_torch(self.model.betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(self.model.alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod.cpu())))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod.cpu())))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu() - 1)))

        # ddim sampling parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(alphacums=alphas_cumprod.cpu(),
                                                                                   ddim_timesteps=self.ddim_timesteps,
                                                                                   eta=ddim_eta,verbose=verbose)
        self.register_buffer('ddim_sigmas', ddim_sigmas)
        self.register_buffer('ddim_alphas', ddim_alphas)
        self.register_buffer('ddim_alphas_prev', ddim_alphas_prev)
        self.register_buffer('ddim_sqrt_one_minus_alphas', np.sqrt(1. - ddim_alphas))
        sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
            (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod) * (
                        1 - self.alphas_cumprod / self.alphas_cumprod_prev))
        self.register_buffer('ddim_sigmas_for_original_num_steps', sigmas_for_original_sampling_steps)

    # @torch.no_grad()
    def sample(self,
               S,
               batch_size,
               shape,
               conditioning=None,
               callback=None,
               normals_sequence=None,
               img_callback=None,
               quantize_x0=False,
               eta=0.,
               mask=None,
               x0=None,
               temperature=1.,
               noise_dropout=0.,
               score_corrector=None,
               corrector_kwargs=None,
               verbose=True,
               schedule_verbose=False,
               x_T=None,
               log_every_t=100,
               unconditional_guidance_scale=1.,
               unconditional_conditioning=None,
               precision=None,
               fs=None,
               timestep_spacing='uniform', #uniform_trailing for starting from last timestep
               guidance_rescale=0.0,
               **kwargs
               ):
        
        # check condition bs
        if conditioning is not None:
            if isinstance(conditioning, dict):
                try:
                    cbs = conditioning[list(conditioning.keys())[0]].shape[0]
                except:
                    cbs = conditioning[list(conditioning.keys())[0]][0].shape[0]

                if cbs != batch_size:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
            else:
                if conditioning.shape[0] != batch_size:
                    print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")

        self.make_schedule(ddim_num_steps=S, ddim_discretize=timestep_spacing, ddim_eta=eta, verbose=schedule_verbose)
        
        # make shape
        if len(shape) == 3:
            C, H, W = shape
            size = (batch_size, C, H, W)
        elif len(shape) == 4:
            C, T, H, W = shape
            size = (batch_size, C, T, H, W)

        samples, intermediates = self.ddim_sampling(conditioning, size,
                                                    callback=callback,
                                                    img_callback=img_callback,
                                                    quantize_denoised=quantize_x0,
                                                    mask=mask, x0=x0,
                                                    ddim_use_original_steps=False,
                                                    noise_dropout=noise_dropout,
                                                    temperature=temperature,
                                                    score_corrector=score_corrector,
                                                    corrector_kwargs=corrector_kwargs,
                                                    x_T=x_T,
                                                    log_every_t=log_every_t,
                                                    unconditional_guidance_scale=unconditional_guidance_scale,
                                                    unconditional_conditioning=unconditional_conditioning,
                                                    verbose=verbose,
                                                    precision=precision,
                                                    fs=fs,
                                                    guidance_rescale=guidance_rescale,
                                                    **kwargs)
        return samples, intermediates

    # @torch.no_grad()
    def ddim_sampling(self, cond, shape,
                      x_T=None, ddim_use_original_steps=False,
                      callback=None, timesteps=None, quantize_denoised=False,
                      mask=None, x0=None, img_callback=None, log_every_t=100,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None, verbose=True,precision=None,fs=None,guidance_rescale=0.0,
                      **kwargs):
        device = self.model.betas.device        
        b = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T
        if precision is not None:
            if precision == 16:
                img = img.to(dtype=torch.float16)

        if timesteps is None:
            timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
        elif timesteps is not None and not ddim_use_original_steps:
            subset_end = int(min(timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) - 1
            timesteps = self.ddim_timesteps[:subset_end]
            
        intermediates = {'x_inter': [img], 'pred_x0': [img]}
        time_range = reversed(range(0,timesteps)) if ddim_use_original_steps else np.flip(timesteps)
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
        if verbose:
            iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps)
        else:
            iterator = time_range

        clean_cond = kwargs.pop("clean_cond", False)

        # cond_copy, unconditional_conditioning_copy = copy.deepcopy(cond), copy.deepcopy(unconditional_conditioning)
        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((b,), step, device=device, dtype=torch.long)

            ## use mask to blend noised original latent (img_orig) & new sampled latent (img)
            if mask is not None:
                assert x0 is not None
                if clean_cond:
                    img_orig = x0
                else:
                    img_orig = self.model.q_sample(x0, ts)  # TODO: deterministic forward pass? <ddim inversion>
                img = img_orig * mask + (1. - mask) * img # keep original & modify use img


            outs = self.p_sample_ddim(img, cond, ts, index=index, use_original_steps=ddim_use_original_steps,
                                quantize_denoised=quantize_denoised, temperature=temperature,
                                noise_dropout=noise_dropout, score_corrector=score_corrector,
                                corrector_kwargs=corrector_kwargs,
                                unconditional_guidance_scale=unconditional_guidance_scale,
                                unconditional_conditioning=unconditional_conditioning,
                                mask=mask,x0=x0,fs=fs,guidance_rescale=guidance_rescale,
                                **kwargs)

            img, pred_x0 = outs
            if callback: callback(i)
            if img_callback: img_callback(pred_x0, i)

            if index % log_every_t == 0 or index == total_steps - 1:
                intermediates['x_inter'].append(img)
                intermediates['pred_x0'].append(pred_x0)
        

        return img, intermediates

    # @torch.no_grad()
    def p_sample_ddim(self, x, c, t, index, repeat_noise=False, use_original_steps=False, quantize_denoised=False,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None,
                      uc_type=None, conditional_guidance_scale_temporal=None,mask=None,x0=None,guidance_rescale=0.0,**kwargs):
        b, *_, device = *x.shape, x.device
    
        if x.dim() == 5:
            is_video = True
        else:
            is_video = False

        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
        # sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas
        sigmas = self.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas
        # select parameters corresponding to the currently considered timestep
    
        if is_video:
            size = (b, 1, 1, 1, 1)
        else:
            size = (b, 1, 1, 1)
        a_t = torch.full(size, alphas[index], device=device)
        a_prev = torch.full(size, alphas_prev[index], device=device)
        sigma_t = torch.full(size, sigmas[index], device=device)
        sqrt_one_minus_at = torch.full(size, sqrt_one_minus_alphas[index],device=device)

        beta_t = a_t / a_prev

        start = 101
        end = -1

        def print_grad(grad):
            print("***> Gradient at this point:", grad)

        # loss guidances
        if "loss_guidance_fn" in kwargs: 
            loss_guidance_fn = kwargs["loss_guidance_fn"]
            verbose = loss_guidance_fn.verbose
            loss_guidance_batch = 1
            assert loss_guidance_batch==1, "tv loss only supports batchsize = 1"
            repeat = loss_guidance_fn.recur_steps

            if repeat > 1: 
                assert repeat == 2, "only support 2 recur steps"
            
            scale_guidance_weight = 1.0
            if loss_guidance_fn.scale_guidance_weight: 
                scale_guidance_weight = loss_guidance_fn.guidance_weight_fn(
                    loss_guidance_fn.current_train_iter
                )
                if index == 1: 
                    print("=> Scale guidance weight: ", scale_guidance_weight)
        
            self.model.model.requires_grad_(True)
            self.model.first_stage_model.requires_grad_(True)
            
            for j in range(repeat): 
                
                x = x.detach().requires_grad_(True) # [1, 4, n_frames, H//8, W//8]

                e_t_cond = self.model.apply_model(x, t, c, **kwargs)
                e_t_uncond = self.model.apply_model(x, t, unconditional_conditioning, **kwargs)

                model_output = e_t_uncond + unconditional_guidance_scale * (e_t_cond - e_t_uncond)
                correction = e_t_cond - e_t_uncond

                model_output = rescale_noise_cfg(model_output, e_t_cond, guidance_rescale=guidance_rescale)

                e_t = self.model.predict_eps_from_z_and_v(x, t, model_output)

                pred_x0 = self.model.predict_start_from_z_and_v(x, t, model_output)
                if self.model.use_dynamic_rescale:
                    scale_t = torch.full(size, self.ddim_scale_arr[index], device=device)
                    prev_scale_t = torch.full(size, self.ddim_scale_arr_prev[index], device=device)
                    rescale = (prev_scale_t / scale_t)
                    pred_x0 *= rescale
                
                # direction pointing to x_t
                with torch.no_grad(): 
                    dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t

                    noise = sigma_t * noise_like(x.shape, device, repeat_noise) * temperature
                    if noise_dropout > 0.:
                        noise = torch.nn.functional.dropout(noise, p=noise_dropout)
                
                    x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise

                batch, dim, n_frames, H, W = pred_x0.shape

                accum_guided_grad = [] # [1, 4, 25, h, w]
                accum_D_x0_t = [] # [1, 4, 25, H, W]
                accum_grad_loss2x0 = []
                uncertaint_map_grad = 0.
                for batch_idx in range(n_frames)[::loss_guidance_batch]: 

                    batch_pred_x0 = pred_x0[:, :, batch_idx:(batch_idx+loss_guidance_batch)]# [1, 4, n_frames, h, w]
                    batch_pred_x0_clone = batch_pred_x0.clone().detach().requires_grad_(True) # do not bp to unet and free it

                    if start > index >= end: 
                        
                        D_x0_t = self.model.differentiable_decode_first_stage(batch_pred_x0_clone) # [1, 3, n_frames, H, W]
                        # L11 in Algorithm 1 of the paper. 
                        accum_D_x0_t.append(D_x0_t)
                        loss_dict, numel = loss_guidance_fn(D_x0_t[0], index, batch_idx, batch_idx+loss_guidance_batch) # [1, 3, H, W]
                        guided_loss_l2 = loss_dict["recon"]

                        # NOTE: very strange bug. if .mean() in loss_guidance_fn, then the grad will all be zeros
                        grad_loss2x0_l2 = torch.autograd.grad(outputs=guided_loss_l2, inputs=batch_pred_x0_clone)[0] 
                        # [1, 4, 1, h, w]
                        
                        if not loss_guidance_fn.mean_loss: 
                            grad_loss2x0_l2 = grad_loss2x0_l2 / numel
                            
                        grad_loss2x0 = grad_loss2x0_l2.clone()
                        accum_grad_loss2x0.append(grad_loss2x0)
                        
                        if verbose: 
                            if batch_idx == 0: 
                                print("=> gradient sum: ", index, grad_loss2x0.shape, grad_loss2x0_l2[0, :, 0].abs().sum())
                                print(grad_loss2x0[:, 0])
                        
                        del batch_pred_x0, D_x0_t, batch_pred_x0_clone, grad_loss2x0, grad_loss2x0_l2
                   
                if start > index >= end: 
                    accum_D_x0_t = torch.cat(accum_D_x0_t, dim=2) # [1, 3, n_frames, H, W]
                    loss_guidance_fn.save_pred_x0(accum_D_x0_t, index)

                    accum_grad_loss2x0 = torch.cat(accum_grad_loss2x0, dim=2) # [1, 4, n_frames, h, w]
                    if verbose: 
                        print("=> accum_grad: ", accum_grad_loss2x0.shape, accum_grad_loss2x0.abs().sum())
                     
                    pred_x0.backward(gradient=accum_grad_loss2x0, inputs=x)
                    
                    accum_guided_grad = x.grad.clone().detach() # L12 in Algorithm 1 of the paper. 
                    x.grad.zero_()
                    
                    if verbose: 
                        print("=> guided shape: ", accum_guided_grad.shape) # [1, 4, 25, h, w]
                        print("=> check: ", (accum_guided_grad * accum_guided_grad).mean().sqrt().item())
                    
                    tmp_s = (accum_guided_grad * accum_guided_grad).mean().sqrt().item()
                    if tmp_s == 0:
                        rho = 0
                    else: 
                        rho_scale = 0.2 * scale_guidance_weight
                        rho = (correction.detach() ** 2).mean().sqrt().item() * unconditional_guidance_scale / tmp_s * rho_scale

                    x_prev = x_prev - rho * accum_guided_grad
                    # L13 in Algorithm 1 of the paper. 

                    del accum_D_x0_t, accum_grad_loss2x0, accum_guided_grad
                
                torch.cuda.empty_cache()

                x = beta_t.sqrt() * x_prev + (1 - beta_t).sqrt() * noise_like(x.shape, device, False)
        
        return x_prev.detach(), pred_x0.detach()
        # TODO: check why removeing the detach here will incur OOM error
    
    @torch.no_grad()
    def decode(self, x_latent, cond, t_start, unconditional_guidance_scale=1.0, unconditional_conditioning=None,
               use_original_steps=False, callback=None):

        timesteps = np.arange(self.ddpm_num_timesteps) if use_original_steps else self.ddim_timesteps
        timesteps = timesteps[:t_start]

        time_range = np.flip(timesteps)
        total_steps = timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='Decoding image', total=total_steps)
        x_dec = x_latent
        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((x_latent.shape[0],), step, device=x_latent.device, dtype=torch.long)
            x_dec, _ = self.p_sample_ddim(x_dec, cond, ts, index=index, use_original_steps=use_original_steps,
                                          unconditional_guidance_scale=unconditional_guidance_scale,
                                          unconditional_conditioning=unconditional_conditioning)
            if callback: callback(i)
        return x_dec

    @torch.no_grad()
    def stochastic_encode(self, x0, t, use_original_steps=False, noise=None):
        # fast, but does not allow for exact reconstruction
        # t serves as an index to gather the correct alphas
        if use_original_steps:
            sqrt_alphas_cumprod = self.sqrt_alphas_cumprod
            sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod
        else:
            sqrt_alphas_cumprod = torch.sqrt(self.ddim_alphas)
            sqrt_one_minus_alphas_cumprod = self.ddim_sqrt_one_minus_alphas

        if noise is None:
            noise = torch.randn_like(x0)
        return (extract_into_tensor(sqrt_alphas_cumprod, t, x0.shape) * x0 +
                extract_into_tensor(sqrt_one_minus_alphas_cumprod, t, x0.shape) * noise)
