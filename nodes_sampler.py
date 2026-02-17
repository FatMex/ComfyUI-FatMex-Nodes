"""
Fat Mex Nodes - Samplers
Content, Reference, Image Edit, and Inpaint samplers.
Each consolidates multiple raw ComfyUI nodes into a single node.
"""

import math
import torch
import logging

import comfy.sample
import comfy.samplers
import comfy.utils
import comfy.model_management
import latent_preview
import node_helpers
import folder_paths

from .presets import MODEL_PRESETS, PRESET_NAMES, RESOLUTION_PRESETS, RESOLUTION_NAMES

logger = logging.getLogger("FatMex")

def _resolve_preset(preset_hint):
    """Resolve a preset config from hint string."""
    if preset_hint != "auto" and preset_hint in MODEL_PRESETS:
        return MODEL_PRESETS[preset_hint]
    return MODEL_PRESETS.get("Klein 9B", {
        "sampler": "euler", "scheduler": "beta", "steps": 6, "cfg": 1.0,
    })


def _resolve_sampling_params(preset_config, steps=0, cfg=0.0, sampler_name="auto", scheduler="auto"):
    """Resolve actual sampling parameters from preset + overrides."""
    return (
        steps if steps > 0 else preset_config.get("steps", 6),
        cfg if cfg > 0 else preset_config.get("cfg", 1.0),
        sampler_name if sampler_name != "auto" else preset_config.get("sampler", "euler"),
        scheduler if scheduler != "auto" else preset_config.get("scheduler", "beta"),
    )


def _vae_decode(vae, sampled):
    """Decode latent samples to images."""
    decoded = sampled["samples"]
    if decoded.is_nested:
        decoded = decoded.unbind()[0]
    images = vae.decode(decoded)
    if len(images.shape) == 5:
        images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
    return images


def _scale_image_to_total_pixels(samples_bchw, total_pixels, multiple_of=16):
    """
    Scale image tensor [B,C,H,W] to target total pixel count with dimensions
    divisible by `multiple_of`. Uses area interpolation (best for downscaling).
    Matches ComfyUi-Scale-Image-to-Total-Pixels-Advanced behaviour.
    """
    h, w = samples_bchw.shape[2], samples_bchw.shape[3]
    current_pixels = h * w
    if current_pixels == 0:
        return samples_bchw
    scale = math.sqrt(total_pixels / current_pixels)
    new_w = max(multiple_of, round(w * scale / multiple_of) * multiple_of)
    new_h = max(multiple_of, round(h * scale / multiple_of) * multiple_of)
    if new_w == w and new_h == h:
        return samples_bchw
    return comfy.utils.common_upscale(samples_bchw, new_w, new_h, "area", "disabled")


def _qwen_edit_encode(clip, vae, prompt, negative_prompt, images,
                       ref_latent_mask=None, use_zero_out_negative=False):
    """
    Encode prompt with Qwen Image Edit Plus style conditioning.
    Replicates TextEncodeQwenImageEditPlus logic internally.

    Args:
        ref_latent_mask: optional list of bools, same length as images.
            True = include this image as a reference latent (identity conditioning).
            False = VL-only (visual context, no latent conditioning).
            None = all images get reference latents (original behaviour).
        use_zero_out_negative: if True, use ConditioningZeroOut of the positive
            as negative conditioning instead of encoding negative_prompt text.
            This dramatically improves Qwen 2509 quality per community research.
    """
    ref_latents = []
    images_vl = []
    llama_template = (
        "<|im_start|>system\nDescribe the key features of the input image "
        "(color, shape, size, texture, objects, background), then explain how "
        "the user's text instruction should alter or modify the image. Generate "
        "a new image that meets the user's requirements while maintaining "
        "consistency with the original input where appropriate."
        "<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
    )
    image_prompt = ""

    for i, image in enumerate(images):
        if image is not None:
            samples = image.movedim(-1, 1)

            # VL resize (384x384 target) — always included so the LLM sees the image
            total_vl = int(384 * 384)
            scale_vl = math.sqrt(total_vl / (samples.shape[3] * samples.shape[2]))
            w_vl = round(samples.shape[3] * scale_vl)
            h_vl = round(samples.shape[2] * scale_vl)
            s_vl = comfy.utils.common_upscale(samples, w_vl, h_vl, "area", "disabled")
            images_vl.append(s_vl.movedim(1, -1))

            # Ref latent: scale to ~1mpx with dims divisible by 16
            include_latent = ref_latent_mask[i] if ref_latent_mask is not None else True
            if vae is not None and include_latent:
                s_ref = _scale_image_to_total_pixels(samples, total_pixels=1024 * 1024, multiple_of=16)
                ref_latents.append(vae.encode(s_ref.movedim(1, -1)[:, :, :, :3]))

            image_prompt += "Picture {}: <|vision_start|><|image_pad|><|vision_end|>".format(i + 1)

    tokens = clip.tokenize(image_prompt + prompt, images=images_vl, llama_template=llama_template)
    positive = clip.encode_from_tokens_scheduled(tokens)
    if len(ref_latents) > 0:
        positive = node_helpers.conditioning_set_values(
            positive, {"reference_latents": ref_latents}, append=True
        )

    if use_zero_out_negative:
        # ConditioningZeroOut: zero the positive conditioning tensors.
        # Forces the model to use the high-quality reference latent instead
        # of low-quality internal image copies. Critical for Qwen 2509 quality.
        negative = []
        for t in positive:
            d = t[1].copy()
            pooled_output = d.get("pooled_output", None)
            if pooled_output is not None:
                d["pooled_output"] = torch.zeros_like(pooled_output)
            conditioning_lyrics = d.get("conditioning_lyrics", None)
            if conditioning_lyrics is not None:
                d["conditioning_lyrics"] = torch.zeros_like(conditioning_lyrics)
            negative.append([torch.zeros_like(t[0]), d])
    else:
        neg_tokens = clip.tokenize(negative_prompt)
        negative = clip.encode_from_tokens_scheduled(neg_tokens)

    return positive, negative


def _common_sample(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent, denoise=1.0):
    """Shared sampling logic matching ComfyUI's common_ksampler."""
    latent_image = latent["samples"]
    latent_image = comfy.sample.fix_empty_latent_channels(model, latent_image, latent.get("downscale_ratio_spacial", None))

    batch_inds = latent.get("batch_index", None)
    noise = comfy.sample.prepare_noise(latent_image, seed, batch_inds)

    noise_mask = latent.get("noise_mask", None)

    callback = latent_preview.prepare_callback(model, steps)
    disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
    samples = comfy.sample.sample(
        model, noise, steps, cfg, sampler_name, scheduler,
        positive, negative, latent_image,
        denoise=denoise, noise_mask=noise_mask,
        callback=callback, disable_pbar=disable_pbar, seed=seed
    )
    out = latent.copy()
    out.pop("downscale_ratio_spacial", None)
    out["samples"] = samples
    return out


class FatMexContentSampler:
    """
    All-in-one text-to-image sampler.
    Replaces: EmptyLatentImage + CLIPTextEncode (pos) + CLIPTextEncode (neg) + KSampler + VAEDecode
    """

    def __init__(self):
        self.device = comfy.model_management.intermediate_device()

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The model from Fat Mex Model Loader."}),
                "clip": ("CLIP", {"tooltip": "The CLIP model from Fat Mex Model Loader."}),
                "vae": ("VAE", {"tooltip": "The VAE from Fat Mex Model Loader."}),
                "positive_prompt": ("STRING", {"multiline": True, "dynamicPrompts": True, "tooltip": "Describe what you want in the image.", "default": ""}),
                "negative_prompt": ("STRING", {"multiline": True, "tooltip": "Describe what you don't want.", "default": ""}),
                "resolution": (RESOLUTION_NAMES, {"default": "1024x1024 (Square)", "tooltip": "Image resolution preset."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True, "tooltip": "Random seed for reproducibility."}),
            },
            "optional": {
                "preset_hint": (["auto"] + PRESET_NAMES, {"default": "auto", "tooltip": "Sampling preset hint. 'auto' uses reasonable defaults."}),
                "steps": ("INT", {"default": 0, "min": 0, "max": 100, "tooltip": "Override steps (0 = use preset default)."}),
                "cfg": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100.0, "step": 0.1, "tooltip": "Override CFG (0 = use preset default)."}),
                "sampler_name": (["auto"] + comfy.samplers.KSampler.SAMPLERS, {"default": "auto", "tooltip": "Override sampler (auto = use preset)."}),
                "scheduler": (["auto"] + comfy.samplers.KSampler.SCHEDULERS, {"default": "auto", "tooltip": "Override scheduler (auto = use preset)."}),
                "custom_width": ("INT", {"default": 0, "min": 0, "max": 16384, "step": 8, "tooltip": "Custom width (only when resolution is 'Custom')."}),
                "custom_height": ("INT", {"default": 0, "min": 0, "max": 16384, "step": 8, "tooltip": "Custom height (only when resolution is 'Custom')."}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 64, "tooltip": "Number of images to generate."}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Denoise strength. 1.0 for full generation."}),
            }
        }

    RETURN_TYPES = ("IMAGE", "LATENT")
    RETURN_NAMES = ("images", "latent")
    OUTPUT_TOOLTIPS = ("The generated images.", "The raw latent (for chaining).")
    FUNCTION = "generate"
    CATEGORY = "Fat Mex"
    DESCRIPTION = "All-in-one text-to-image generator. Enter your prompt, pick a resolution, and get images. Automatically handles prompt encoding, latent creation, sampling, and VAE decoding."

    def generate(self, model, clip, vae, positive_prompt, negative_prompt, resolution, seed,
                 preset_hint="auto", steps=0, cfg=0.0, sampler_name="auto", scheduler="auto",
                 custom_width=0, custom_height=0, batch_size=1, denoise=1.0):

        # Resolve preset for sampling defaults
        preset_config = None
        if preset_hint != "auto" and preset_hint in MODEL_PRESETS:
            preset_config = MODEL_PRESETS[preset_hint]
        else:
            # Default to Klein-like settings
            preset_config = MODEL_PRESETS.get("Klein 9B", {
                "sampler": "euler", "scheduler": "beta", "steps": 6, "cfg": 1.0,
            })

        # Resolve sampling parameters
        actual_steps = steps if steps > 0 else preset_config.get("steps", 6)
        actual_cfg = cfg if cfg > 0 else preset_config.get("cfg", 1.0)
        actual_sampler = sampler_name if sampler_name != "auto" else preset_config.get("sampler", "euler")
        actual_scheduler = scheduler if scheduler != "auto" else preset_config.get("scheduler", "beta")

        # Resolve resolution
        if resolution == "Custom":
            width = custom_width if custom_width > 0 else 1024
            height = custom_height if custom_height > 0 else 1024
        else:
            width, height = RESOLUTION_PRESETS.get(resolution, (1024, 1024))

        # Encode prompts
        pos_tokens = clip.tokenize(positive_prompt)
        positive = clip.encode_from_tokens_scheduled(pos_tokens)

        neg_tokens = clip.tokenize(negative_prompt)
        negative = clip.encode_from_tokens_scheduled(neg_tokens)

        # Create empty latent (standard 4-channel, //8 spatial approach)
        # fix_empty_latent_channels() will auto-adjust channels and spatial dims
        # based on the loaded model (e.g. Flux2: 4->128ch, spatial 8->16)
        latent = torch.zeros(
            [batch_size, 4, height // 8, width // 8],
            device=self.device
        )
        latent_dict = {"samples": latent, "downscale_ratio_spacial": 8}

        # Sample
        sampled = _common_sample(
            model, seed, actual_steps, actual_cfg,
            actual_sampler, actual_scheduler,
            positive, negative, latent_dict,
            denoise=denoise
        )

        # Decode
        decoded = sampled["samples"]
        if decoded.is_nested:
            decoded = decoded.unbind()[0]
        images = vae.decode(decoded)
        if len(images.shape) == 5:
            images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])

        return (images, sampled)


class FatMexReferenceSampler:
    """
    Reference-guided image generation.
    Replaces: VAEEncode + ReferenceLatent + CLIPTextEncode + KSampler + VAEDecode

    Two modes based on denoise:
    - denoise < 1.0: img2img mode (starts from encoded reference, refines it)
    - denoise = 1.0: full generation mode (starts from empty latent, guided by reference conditioning)
    """

    def __init__(self):
        self.device = comfy.model_management.intermediate_device()

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The model from Fat Mex Model Loader."}),
                "clip": ("CLIP", {"tooltip": "The CLIP model."}),
                "vae": ("VAE", {"tooltip": "The VAE model."}),
                "reference_image": ("IMAGE", {"tooltip": "The reference image to guide generation."}),
                "positive_prompt": ("STRING", {"multiline": True, "dynamicPrompts": True, "tooltip": "Describe what you want.", "default": ""}),
                "negative_prompt": ("STRING", {"multiline": True, "tooltip": "Describe what you don't want.", "default": ""}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True}),
                "denoise": ("FLOAT", {"default": 0.75, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Below 1.0 = img2img refinement from reference. 1.0 = full generation with reference conditioning."}),
            },
            "optional": {
                "preset_hint": (["auto"] + PRESET_NAMES, {"default": "auto"}),
                "steps": ("INT", {"default": 0, "min": 0, "max": 100, "tooltip": "Override steps (0 = preset default)."}),
                "cfg": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "sampler_name": (["auto"] + comfy.samplers.KSampler.SAMPLERS, {"default": "auto"}),
                "scheduler": (["auto"] + comfy.samplers.KSampler.SCHEDULERS, {"default": "auto"}),
                "width": ("INT", {"default": 0, "min": 0, "max": 16384, "step": 8, "tooltip": "Output width (0 = same as reference)."}),
                "height": ("INT", {"default": 0, "min": 0, "max": 16384, "step": 8, "tooltip": "Output height (0 = same as reference)."}),
            }
        }

    RETURN_TYPES = ("IMAGE", "LATENT")
    RETURN_NAMES = ("images", "latent")
    FUNCTION = "generate"
    CATEGORY = "Fat Mex"
    DESCRIPTION = "Generate images guided by a reference. Low denoise = img2img refinement (adds detail/texture), 1.0 = new image with reference conditioning."

    def generate(self, model, clip, vae, reference_image, positive_prompt, negative_prompt, seed, denoise=0.75,
                 preset_hint="auto", steps=0, cfg=0.0, sampler_name="auto", scheduler="auto",
                 width=0, height=0):

        preset_config = MODEL_PRESETS.get(preset_hint, MODEL_PRESETS.get("Klein 9B", {})) if preset_hint != "auto" else MODEL_PRESETS.get("Klein 9B", {})

        actual_steps = steps if steps > 0 else preset_config.get("steps", 6)
        actual_cfg = cfg if cfg > 0 else preset_config.get("cfg", 1.0)
        actual_sampler = sampler_name if sampler_name != "auto" else preset_config.get("sampler", "euler")
        actual_scheduler = scheduler if scheduler != "auto" else preset_config.get("scheduler", "beta")

        # Encode reference image to latent
        ref_latent = vae.encode(reference_image[:, :, :, :3])

        # Determine output size
        if width <= 0 or height <= 0:
            # Use reference image dimensions
            _, h, w, _ = reference_image.shape
            width = w
            height = h

        # Encode prompts
        pos_tokens = clip.tokenize(positive_prompt)
        positive = clip.encode_from_tokens_scheduled(pos_tokens)

        neg_tokens = clip.tokenize(negative_prompt)
        negative = clip.encode_from_tokens_scheduled(neg_tokens)

        # Apply ReferenceLatent conditioning (for character consistency)
        positive = node_helpers.conditioning_set_values(
            positive,
            {"reference_latents": [ref_latent]},
            append=True
        )

        if denoise < 1.0:
            # img2img mode: start from the VAE-encoded reference image
            # KSampler adds noise proportional to denoise, then refines
            latent_dict = {"samples": ref_latent}
            logger.info("[ReferenceSampler] img2img mode: denoise=%.2f, starting from encoded ref", denoise)
        else:
            # Full generation mode: start from empty latent, guided by ref conditioning
            latent = torch.zeros(
                [1, 4, height // 8, width // 8],
                device=self.device
            )
            latent_dict = {"samples": latent, "downscale_ratio_spacial": 8}
            logger.info("[ReferenceSampler] generation mode: starting from empty latent")

        # Sample
        sampled = _common_sample(
            model, seed, actual_steps, actual_cfg,
            actual_sampler, actual_scheduler,
            positive, negative, latent_dict,
            denoise=denoise
        )

        images = _vae_decode(vae, sampled)
        return (images, sampled)


class FatMexImageEditSampler:
    """
    All-in-one Qwen Image Edit sampler.
    Replaces: TextEncodeQwenImageEditPlus + CLIPTextEncode + GetImageSize +
              EmptyQwenImageLayeredLatentImage + KSampler + VAEDecode

    Feed it reference images and a prompt describing the edit.
    Uses Qwen's vision-language conditioning internally.
    """

    def __init__(self):
        self.device = comfy.model_management.intermediate_device()

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "Model from Fat Mex Model Loader (Qwen Edit preset)."}),
                "clip": ("CLIP", {"tooltip": "CLIP from Fat Mex Model Loader."}),
                "vae": ("VAE", {"tooltip": "VAE from Fat Mex Model Loader."}),
                "prompt": ("STRING", {"multiline": True, "dynamicPrompts": True,
                                      "tooltip": "Describe the edit. Reference images as 'image 1', 'image 2', etc.",
                                      "default": ""}),
                "negative_prompt": ("STRING", {"multiline": True,
                                               "tooltip": "What to avoid.",
                                               "default": "ugly, blurry, deformed, bad anatomy, low quality"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True}),
            },
            "optional": {
                "image1": ("IMAGE", {"tooltip": "Reference image 1 (e.g. the face to use)."}),
                "image2": ("IMAGE", {"tooltip": "Reference image 2 (optional)."}),
                "image3": ("IMAGE", {"tooltip": "Reference image 3 (optional)."}),
                "preset_hint": (["auto"] + PRESET_NAMES, {"default": "auto"}),
                "steps": ("INT", {"default": 0, "min": 0, "max": 100, "tooltip": "Override steps (0 = preset)."}),
                "cfg": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "sampler_name": (["auto"] + comfy.samplers.KSampler.SAMPLERS, {"default": "auto"}),
                "scheduler": (["auto"] + comfy.samplers.KSampler.SCHEDULERS, {"default": "auto"}),
                "width": ("INT", {"default": 1024, "min": 64, "max": 16384, "step": 8,
                                  "tooltip": "Output width."}),
                "height": ("INT", {"default": 1024, "min": 64, "max": 16384, "step": 8,
                                   "tooltip": "Output height."}),
                "layers": ("INT", {"default": 3, "min": 0, "max": 16,
                                   "tooltip": "Qwen layered latent layers (usually = number of images)."}),
            }
        }

    RETURN_TYPES = ("IMAGE", "LATENT")
    RETURN_NAMES = ("images", "latent")
    FUNCTION = "generate"
    CATEGORY = "Fat Mex"
    DESCRIPTION = (
        "All-in-one Qwen Image Edit sampler. Feed reference images + a prompt "
        "describing the edit. Handles vision-language encoding, layered latent "
        "creation, sampling, and decoding in one node."
    )

    def generate(self, model, clip, vae, prompt, negative_prompt, seed,
                 image1=None, image2=None, image3=None,
                 preset_hint="auto", steps=0, cfg=0.0,
                 sampler_name="auto", scheduler="auto",
                 width=1024, height=1024, layers=3):

        logger.info(f"[ImageEditSampler] prompt={prompt[:60]}... images={[x is not None for x in [image1,image2,image3]]}")

        preset_config = _resolve_preset(preset_hint)
        actual_steps, actual_cfg, actual_sampler, actual_scheduler = _resolve_sampling_params(
            preset_config, steps, cfg, sampler_name, scheduler
        )
        logger.info(f"[ImageEditSampler] steps={actual_steps} sampler={actual_sampler} scheduler={actual_scheduler} size={width}x{height} layers={layers}")

        # Qwen Edit encoding with image references
        positive, negative = _qwen_edit_encode(
            clip, vae, prompt, negative_prompt, [image1, image2, image3]
        )
        logger.info("[ImageEditSampler] Encoding complete")

        # Create Qwen layered latent: [batch, 16, layers+1, H//8, W//8]
        latent = torch.zeros(
            [1, 16, layers + 1, height // 8, width // 8],
            device=self.device
        )
        latent_dict = {"samples": latent}

        # Sample
        sampled = _common_sample(
            model, seed, actual_steps, actual_cfg,
            actual_sampler, actual_scheduler,
            positive, negative, latent_dict,
            denoise=1.0
        )
        logger.info("[ImageEditSampler] Sampling complete")

        images = _vae_decode(vae, sampled)
        # Only keep the last image — earlier ones are intermediate layer reconstructions
        images = images[-1:]
        return (images, sampled)


class FatMexInpaintSampler:
    """
    All-in-one inpaint sampler with Qwen Edit conditioning.
    Replaces: VAEEncode + SetLatentNoiseMask + TextEncodeQwenImageEditPlus +
              CLIPTextEncode + KSampler + VAEDecode

    Feed it a target image, a mask, reference images, and an inpaint prompt.
    Used for head swaps, face replacement, object editing, etc.
    """

    def __init__(self):
        self.device = comfy.model_management.intermediate_device()

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "Model from Fat Mex Model Loader (Qwen Edit preset)."}),
                "clip": ("CLIP", {"tooltip": "CLIP from Fat Mex Model Loader."}),
                "vae": ("VAE", {"tooltip": "VAE from Fat Mex Model Loader."}),
                "target_image": ("IMAGE", {"tooltip": "The image to inpaint on (e.g. the body image)."}),
                "mask": ("MASK", {"tooltip": "Mask defining the area to regenerate (e.g. from FatMexHeadMask)."}),
                "prompt": ("STRING", {"multiline": True, "dynamicPrompts": True,
                                      "tooltip": "Describe the inpaint. Use 'image 1' to reference the face.",
                                      "default": "Replace the masked head area with the person of image 1. Match the head shape, rotation and lighting of the source image."}),
                "negative_prompt": ("STRING", {"multiline": True,
                                               "default": "ugly, blurry, deformed, bad anatomy, plastic skin"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True}),
            },
            "optional": {
                "image1": ("IMAGE", {"tooltip": "Reference image 1 (e.g. the face to swap in)."}),
                "image2": ("IMAGE", {"tooltip": "Reference image 2 (optional \u2014 e.g. the target body for context)."}),
                "grow_mask_by": ("INT", {"default": 6, "min": 0, "max": 64, "step": 1,
                                          "tooltip": "Expand mask edges for seamless inpainting (matches VAEEncodeForInpaint)."}),
                "preset_hint": (["auto"] + PRESET_NAMES, {"default": "auto"}),
                "steps": ("INT", {"default": 8, "min": 1, "max": 100, "tooltip": "Inpaint steps."}),
                "cfg": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "sampler_name": (["auto"] + comfy.samplers.KSampler.SAMPLERS, {"default": "auto"}),
                "scheduler": (["auto"] + comfy.samplers.KSampler.SCHEDULERS, {"default": "auto"}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                                      "tooltip": "Inpaint denoise strength. 1.0 fully regenerates masked area."}),
            }
        }

    RETURN_TYPES = ("IMAGE", "LATENT")
    RETURN_NAMES = ("images", "latent")
    FUNCTION = "generate"
    CATEGORY = "Fat Mex"
    DESCRIPTION = (
        "All-in-one inpaint sampler with Qwen Edit conditioning. "
        "Feed a target image + mask + reference images. "
        "Perfect for head swaps, face edits, and object replacement."
    )

    def generate(self, model, clip, vae, target_image, mask, prompt, negative_prompt, seed,
                 image1=None, image2=None, grow_mask_by=6,
                 preset_hint="auto", steps=8, cfg=0.0,
                 sampler_name="auto", scheduler="auto", denoise=1.0):

        preset_config = _resolve_preset(preset_hint)
        actual_steps, actual_cfg, actual_sampler, actual_scheduler = _resolve_sampling_params(
            preset_config, steps, cfg, sampler_name, scheduler
        )
        logger.info(f"[InpaintSampler] steps={actual_steps} cfg={actual_cfg} "
                     f"sampler={actual_sampler} scheduler={actual_scheduler}")

        # === Matches the video workflow exactly ===
        # TextEncodeQwenImageEditPlus receives:
        #   image1 = body (clean, no mask overlay)
        #   image2 = face reference
        # KSampler receives:
        #   Empty Latent at body's exact dimensions
        #   No noise_mask — pure Qwen Edit generation
        # SAM3 mask is used ONLY for post-compositing.

        body = target_image[:, :, :, :3]
        _, h_orig, w_orig, _ = target_image.shape

        # Qwen Edit encoding: body as Picture 1, face as Picture 2.
        # Body = VL-only (pose/scene context, no ref latent competing with face).
        # Face = VL + reference latent (STRONG identity: hair color, skin, features).
        # This prevents the body's dark hair / skin tone from diluting the
        # face reference's blonde hair / lighter skin in the latent space.
        positive, negative = _qwen_edit_encode(
            clip, vae, prompt, negative_prompt,
            [body, image1],
            ref_latent_mask=[False, True],
            use_zero_out_negative=True
        )

        # Standard 4D empty latent — EXACTLY like the video's EmptyLatentImage.
        # fix_empty_latent_channels() in _common_sample auto-converts this to
        # the right format for Qwen Edit (16ch, 5D, correct spatial ratio).
        # We must NOT manually create a 5D latent because that bypasses the
        # critical spatial downscale ratio adjustment.
        latent = torch.zeros(
            [1, 4, h_orig // 8, w_orig // 8],
            device=self.device
        )
        latent_dict = {"samples": latent, "downscale_ratio_spacial": 8}
        logger.info(f"[InpaintSampler] Empty latent 4ch {w_orig // 8}x{h_orig // 8} "
                     f"(body is {w_orig}x{h_orig}, will auto-convert for Qwen)")

        # Generate from scratch — no noise_mask, denoise 1.0.
        sampled = _common_sample(
            model, seed, actual_steps, actual_cfg,
            actual_sampler, actual_scheduler,
            positive, negative, latent_dict,
            denoise=1.0
        )

        edited = _vae_decode(vae, sampled)
        logger.info(f"[InpaintSampler] Generated {edited.shape}")

        # Return raw Qwen Edit output — no compositing.
        # Qwen Edit preserves the body/background through its own conditioning.
        # The mask is not needed here; the next pipeline stage (detail pass)
        # handles any remaining cleanup.
        return (edited, sampled)


SAMPLER_CLASS_MAPPINGS = {
    "FatMexContentSampler": FatMexContentSampler,
    "FatMexReferenceSampler": FatMexReferenceSampler,
    "FatMexImageEditSampler": FatMexImageEditSampler,
    "FatMexInpaintSampler": FatMexInpaintSampler,
}

SAMPLER_NAME_MAPPINGS = {
    "FatMexContentSampler": "Fat Mex Content Sampler",
    "FatMexReferenceSampler": "Fat Mex Reference Sampler",
    "FatMexImageEditSampler": "Fat Mex Image Edit Sampler",
    "FatMexInpaintSampler": "Fat Mex Inpaint Sampler",
}
