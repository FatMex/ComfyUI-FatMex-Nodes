"""
Fat Mex Nodes - Image Upscaler
Wraps UpscaleModelLoader + ImageUpscaleWithModel + ImageScale into a single node.
"""

import torch
import logging

import comfy.utils
import comfy.model_management
import folder_paths

logger = logging.getLogger("FatMex")

_upscale_model_cache = {}

RESIZE_METHODS = ["lanczos", "bicubic", "bilinear", "nearest-exact"]


def _load_upscale_model(model_name):
    """Load and cache an upscale model by name."""
    if model_name in _upscale_model_cache:
        return _upscale_model_cache[model_name]

    from spandrel import ModelLoader, ImageModelDescriptor
    try:
        from spandrel_extra_arches import EXTRA_REGISTRY
        from spandrel import MAIN_REGISTRY
        MAIN_REGISTRY.add(*EXTRA_REGISTRY)
    except Exception:
        pass

    model_path = folder_paths.get_full_path_or_raise("upscale_models", model_name)
    sd = comfy.utils.load_torch_file(model_path, safe_load=True)
    if "module.layers.0.residual_group.blocks.0.norm1.weight" in sd:
        sd = comfy.utils.state_dict_prefix_replace(sd, {"module.": ""})
    out = ModelLoader().load_from_state_dict(sd).eval()

    if not isinstance(out, ImageModelDescriptor):
        raise Exception("Upscale model must be a single-image model.")

    _upscale_model_cache[model_name] = out
    logger.info(f"Loaded upscale model: {model_name} (scale: {out.scale}x)")
    return out


def _upscale_image(upscale_model, image):
    """Upscale image using tiled processing with OOM recovery."""
    device = comfy.model_management.get_torch_device()

    memory_required = comfy.model_management.module_size(upscale_model.model)
    memory_required += (512 * 512 * 3) * image.element_size() * max(upscale_model.scale, 1.0) * 384.0
    memory_required += image.nelement() * image.element_size()
    comfy.model_management.free_memory(memory_required, device)

    upscale_model.to(device)
    in_img = image.movedim(-1, -3).to(device)

    tile = 512
    overlap = 32

    oom = True
    try:
        while oom:
            try:
                steps = in_img.shape[0] * comfy.utils.get_tiled_scale_steps(
                    in_img.shape[3], in_img.shape[2],
                    tile_x=tile, tile_y=tile, overlap=overlap
                )
                pbar = comfy.utils.ProgressBar(steps)
                s = comfy.utils.tiled_scale(
                    in_img,
                    lambda a: upscale_model(a),
                    tile_x=tile, tile_y=tile,
                    overlap=overlap,
                    upscale_amount=upscale_model.scale,
                    pbar=pbar,
                )
                oom = False
            except comfy.model_management.OOM_EXCEPTION as e:
                tile //= 2
                if tile < 128:
                    raise e
    finally:
        upscale_model.to("cpu")

    return torch.clamp(s.movedim(-3, -1), min=0, max=1.0)


def _resize_image(image, width, height, method="lanczos"):
    """Resize image to target dimensions."""
    samples = image.movedim(-1, 1)
    s = comfy.utils.common_upscale(samples, width, height, method, "disabled")
    return s.movedim(1, -1)


class FatMexUpscaler:
    """
    All-in-one image upscaler.
    Replaces: UpscaleModelLoader + ImageUpscaleWithModel + ImageScale

    Load an upscale model, upscale the image, then optionally resize
    to a final target size.
    """

    @classmethod
    def INPUT_TYPES(s):
        model_list = folder_paths.get_filename_list("upscale_models")
        if not model_list:
            model_list = ["(no models found)"]

        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Image to upscale."}),
            },
            "optional": {
                "upscale_model": (model_list, {
                    "default": model_list[0],
                    "tooltip": "Upscale model to use. Place .pth files in ComfyUI/models/upscale_models/."
                }),
                "rescale_factor": ("FLOAT", {
                    "default": 1.0, "min": 0.1, "max": 1.0, "step": 0.05,
                    "tooltip": "Scale down after upscaling. 1.0 = keep full size, 0.5 = half the upscaled size. E.g. 4x model + 0.5 rescale = effective 2x."
                }),
                "resize_method": (RESIZE_METHODS, {
                    "default": "lanczos",
                    "tooltip": "Interpolation method for final resize."
                }),
                "max_dimension": ("INT", {
                    "default": 0, "min": 0, "max": 16384, "step": 8,
                    "tooltip": "Cap the longest side to this value (0 = no cap). Preserves aspect ratio."
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "upscale"
    CATEGORY = "Fat Mex"
    DESCRIPTION = (
        "All-in-one image upscaler. Loads a model, upscales with tiled "
        "processing (handles large images), then optionally resizes to a "
        "final target size. Model is cached for speed."
    )

    def upscale(self, image, upscale_model=None, rescale_factor=1.0,
                resize_method="lanczos", max_dimension=0):

        if upscale_model is None or upscale_model == "(no models found)":
            logger.warning("[Upscaler] No upscale model selected, returning original image")
            return (image,)

        model = _load_upscale_model(upscale_model)
        logger.info(f"[Upscaler] Input: {image.shape[2]}x{image.shape[1]}, model: {upscale_model} ({model.scale}x)")

        upscaled = _upscale_image(model, image)

        h, w = upscaled.shape[1], upscaled.shape[2]
        logger.info(f"[Upscaler] After upscale: {w}x{h}")

        target_w, target_h = w, h

        if rescale_factor < 1.0:
            target_w = int(w * rescale_factor)
            target_h = int(h * rescale_factor)

        if max_dimension > 0 and max(target_w, target_h) > max_dimension:
            scale = max_dimension / max(target_w, target_h)
            target_w = int(target_w * scale)
            target_h = int(target_h * scale)

        target_w = max(8, (target_w // 8) * 8)
        target_h = max(8, (target_h // 8) * 8)

        if target_w != w or target_h != h:
            upscaled = _resize_image(upscaled, target_w, target_h, resize_method)
            logger.info(f"[Upscaler] After resize: {target_w}x{target_h}")

        logger.info(f"[Upscaler] Done")
        return (upscaled,)


UPSCALE_CLASS_MAPPINGS = {
    "FatMexUpscaler": FatMexUpscaler,
}

UPSCALE_NAME_MAPPINGS = {
    "FatMexUpscaler": "Fat Mex Upscaler",
}
