"""
Fat Mex Nodes - Model Loader
All-in-one model loader that loads model/clip/vae by preset,
supports LoRA stacking, sage attention, and model sampling shift.
"""

import logging

import comfy.sd
import comfy.utils
import comfy.samplers
import comfy.model_sampling
import folder_paths

from .presets import MODEL_PRESETS, PRESET_NAMES

logger = logging.getLogger("FatMex")

# Weight dtype options matching ComfyUI's UNETLoader
WEIGHT_DTYPES = ["default", "fp8_e4m3fn", "fp8_e5m2", "fp8_e4m3fn_fast", "gguf"]


def _apply_sage_attention(model):
    """Check and log sage attention status. Sage attention is applied globally via --use-sage-attention."""
    from comfy.ldm.modules.attention import SAGE_ATTENTION_IS_AVAILABLE
    if SAGE_ATTENTION_IS_AVAILABLE:
        logger.info("FatMex: Sage attention is available and active")
    else:
        logger.warning("FatMex: Sage attention requested but not available. Install sageattention or use --use-sage-attention")
    return model


def _apply_model_sampling_shift(model, shift):
    """Apply AuraFlow-style model sampling shift (used by Qwen Image 2512).
    Equivalent to ComfyUI's ModelSamplingAuraFlow node with multiplier=1.0."""
    m = model.clone()

    sampling_base = comfy.model_sampling.ModelSamplingDiscreteFlow
    sampling_type = comfy.model_sampling.CONST

    class ModelSamplingAdvanced(sampling_base, sampling_type):
        pass

    model_sampling = ModelSamplingAdvanced(model.model.model_config)
    model_sampling.set_parameters(shift=shift, multiplier=1.0)
    m.add_object_patch("model_sampling", model_sampling)
    return m


class FatMexModelLoader:
    """
    All-in-one model loader. Pick a preset and get MODEL + CLIP + VAE.
    Supports LoRA stacking, sage attention, and model sampling shift.
    Replaces: UNETLoader + CLIPLoader + VAELoader + LoraLoaderModelOnly (x2)
    """

    @classmethod
    def INPUT_TYPES(s):
        lora_list = ["none"] + folder_paths.get_filename_list("loras")

        return {
            "required": {
                "preset": (PRESET_NAMES, {
                    "default": PRESET_NAMES[0] if PRESET_NAMES else "Klein 9B",
                    "tooltip": "Model preset to load. Each preset defines the UNET, CLIP, VAE, and default sampling settings."
                }),
            },
            "optional": {
                "sage_attention": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable sage attention for faster inference (requires sageattention package)."
                }),
                "lora_1": (lora_list, {
                    "default": "none",
                    "tooltip": "First LoRA to apply to the model."
                }),
                "lora_1_strength": ("FLOAT", {
                    "default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01,
                    "tooltip": "Strength of the first LoRA."
                }),
                "lora_2": (lora_list, {
                    "default": "none",
                    "tooltip": "Second LoRA to apply to the model."
                }),
                "lora_2_strength": ("FLOAT", {
                    "default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01,
                    "tooltip": "Strength of the second LoRA."
                }),
                "weight_dtype_override": (WEIGHT_DTYPES, {
                    "default": "default",
                    "tooltip": "Override weight data type. 'default' uses the preset's setting."
                }),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    RETURN_NAMES = ("model", "clip", "vae")
    OUTPUT_TOOLTIPS = (
        "The loaded diffusion model.",
        "The CLIP text encoder.",
        "The VAE for encoding/decoding images.",
    )
    FUNCTION = "load_model"
    CATEGORY = "Fat Mex"
    DESCRIPTION = (
        "All-in-one model loader. Pick a preset to load the correct UNET, CLIP, "
        "and VAE. Optionally stack up to 2 LoRAs. Supports all Fat Mex model "
        "families: Klein 9B, Qwen Image Edit, Z-Image Turbo, Chroma HD, etc."
    )

    def load_model(self, preset, sage_attention=True,
                   lora_1="none", lora_1_strength=1.0,
                   lora_2="none", lora_2_strength=1.0,
                   weight_dtype_override="default"):

        config = MODEL_PRESETS.get(preset)
        if config is None:
            raise ValueError(f"FatMex: Unknown preset '{preset}'. Available: {PRESET_NAMES}")

        logger.info(f"FatMex: Loading preset '{preset}'")

        # --- Determine weight dtype ---
        weight_dtype = weight_dtype_override if weight_dtype_override != "default" else config.get("weight_dtype", "default")
        model_options = {}
        if weight_dtype == "fp8_e4m3fn":
            model_options["dtype"] = {"torch.float8_e4m3fn": True}
        elif weight_dtype == "fp8_e5m2":
            model_options["dtype"] = {"torch.float8_e5m2": True}
        elif weight_dtype == "fp8_e4m3fn_fast":
            model_options["dtype"] = {"torch.float8_e4m3fn": True}
            model_options["fp8_optimizations"] = True

        # --- Load Model ---
        unet_source = config.get("unet_source", "diffusion_models")

        if unet_source == "checkpoints" and config.get("checkpoint"):
            # Load from checkpoint (includes model + clip + vae)
            ckpt_name = config["checkpoint"]
            ckpt_path = folder_paths.get_full_path_or_raise("checkpoints", ckpt_name)
            out = comfy.sd.load_checkpoint_guess_config(
                ckpt_path,
                output_vae=True,
                output_clip=True,
                embedding_directory=folder_paths.get_folder_paths("embeddings"),
            )
            model, clip, vae = out[:3]
            logger.info(f"FatMex: Loaded checkpoint '{ckpt_name}'")
        else:
            # Load UNET separately
            unet_name = config.get("unet")
            if unet_name is None:
                raise ValueError(f"FatMex: Preset '{preset}' has no UNET or checkpoint configured")

            unet_path = folder_paths.get_full_path_or_raise(unet_source, unet_name)
            model = comfy.sd.load_diffusion_model(unet_path, model_options=model_options)
            logger.info(f"FatMex: Loaded UNET '{unet_name}' from {unet_source}")

            # --- Load CLIP ---
            clip_name = config.get("clip")
            if clip_name:
                clip_type_str = config.get("clip_type", "").upper()
                clip_type = getattr(comfy.sd.CLIPType, clip_type_str, comfy.sd.CLIPType.STABLE_DIFFUSION)

                clip_path = folder_paths.get_full_path_or_raise("text_encoders", clip_name)
                clip = comfy.sd.load_clip(
                    ckpt_paths=[clip_path],
                    embedding_directory=folder_paths.get_folder_paths("embeddings"),
                    clip_type=clip_type,
                )
                logger.info(f"FatMex: Loaded CLIP '{clip_name}' (type: {clip_type_str})")
            else:
                clip = None

            # --- Load VAE ---
            vae_name = config.get("vae")
            if vae_name:
                vae_path = folder_paths.get_full_path_or_raise("vae", vae_name)
                sd = comfy.utils.load_torch_file(vae_path)
                vae = comfy.sd.VAE(sd=sd)
                logger.info(f"FatMex: Loaded VAE '{vae_name}'")
            else:
                vae = None

        # --- Apply Sage Attention ---
        if sage_attention:
            model = _apply_sage_attention(model)

        # --- Apply Model Sampling Shift (e.g. AuraFlow for Qwen 2512) ---
        sampling_shift = config.get("model_sampling_shift")
        if sampling_shift is not None:
            model = _apply_model_sampling_shift(model, sampling_shift)
            logger.info(f"FatMex: Applied model sampling shift={sampling_shift} for preset '{preset}'")

        # --- Apply LoRAs ---
        for lora_name, lora_strength, label in [
            (lora_1, lora_1_strength, "LoRA 1"),
            (lora_2, lora_2_strength, "LoRA 2"),
        ]:
            if lora_name and lora_name != "none":
                lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
                lora_sd = comfy.utils.load_torch_file(lora_path, safe_load=True)
                model, clip = comfy.sd.load_lora_for_models(
                    model, clip, lora_sd,
                    strength_model=lora_strength,
                    strength_clip=lora_strength,
                )
                logger.info(f"FatMex: Applied {label} '{lora_name}' (strength={lora_strength})")

        logger.info(f"FatMex: Preset '{preset}' fully loaded")
        return (model, clip, vae)


LOADER_CLASS_MAPPINGS = {
    "FatMexModelLoader": FatMexModelLoader,
}

LOADER_NAME_MAPPINGS = {
    "FatMexModelLoader": "Fat Mex Model Loader",
}
