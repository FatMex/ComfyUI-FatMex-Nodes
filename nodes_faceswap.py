"""
Fat Mex Nodes - Face Swap
Two modes:
  Quick  – InsightFace inswapper paste (no MODEL/CLIP/VAE needed)
  Quality – YOLOv8 face detect → inpaint with image-editing model + optional 4x face upscale
            Matches icekub's FaceDetailer approach.
"""

import torch
import numpy as np
import logging
import os
from PIL import Image

logger = logging.getLogger("FatMex")

# Lazy-loaded singletons
_insightface = None
_face_analyser = None
_yolo_model = None
_upscale_model = None


# ── helpers ──────────────────────────────────────────────────────────────

def _get_insightface():
    global _insightface
    if _insightface is None:
        try:
            import insightface
            _insightface = insightface
        except ImportError:
            raise RuntimeError("FatMex Face Swap requires insightface. pip install insightface onnxruntime")
    return _insightface


def _get_onnx_providers():
    providers = []
    try:
        import onnxruntime
        available = onnxruntime.get_available_providers()
        if "CUDAExecutionProvider" in available:
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")
    except Exception:
        providers = ["CPUExecutionProvider"]
    return providers


def _get_face_analyser(model_path=None):
    global _face_analyser
    if _face_analyser is None:
        import folder_paths
        insightface = _get_insightface()
        _face_analyser = insightface.app.FaceAnalysis(
            name="buffalo_l",
            root=model_path or os.path.join(folder_paths.models_dir, "insightface"),
            providers=_get_onnx_providers()
        )
        _face_analyser.prepare(ctx_id=0, det_size=(640, 640))
    return _face_analyser


def _get_yolo_detector():
    """Lazy-load YOLOv8 face detector."""
    global _yolo_model
    if _yolo_model is None:
        import folder_paths
        model_path = os.path.join(folder_paths.models_dir, "ultralytics", "bbox", "face_yolov8m.pt")
        if not os.path.exists(model_path):
            raise RuntimeError(
                f"YOLOv8 face model not found at {model_path}. "
                "Download face_yolov8m.pt to ComfyUI/models/ultralytics/bbox/"
            )
        from ultralytics import YOLO
        _yolo_model = YOLO(model_path)
    return _yolo_model


def _get_upscale_model():
    """Lazy-load 4xFaceUpDAT upscale model."""
    global _upscale_model
    if _upscale_model is None:
        import folder_paths
        import comfy.utils
        from spandrel import ModelLoader as SpandrelLoader

        model_path = os.path.join(folder_paths.models_dir, "upscale_models", "4xFaceUpDAT.pth")
        if not os.path.exists(model_path):
            logger.warning("4xFaceUpDAT.pth not found — skipping face upscale.")
            return None
        _upscale_model = SpandrelLoader().load_from_file(model_path)
        if hasattr(_upscale_model, 'eval'):
            _upscale_model.eval()
    return _upscale_model


def _tensor_to_pil(tensor):
    if len(tensor.shape) == 4:
        tensor = tensor[0]
    img_np = (tensor.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(img_np)


def _pil_to_tensor(pil_image):
    img_np = np.array(pil_image).astype(np.float32) / 255.0
    return torch.from_numpy(img_np).unsqueeze(0)


# ── Quick mode: InsightFace inswapper ────────────────────────────────────

def _quick_swap(target_image, face_reference, face_index=0, blend_strength=1.0):
    """Original inswapper-based face swap."""
    insightface = _get_insightface()

    target_pil = _tensor_to_pil(target_image)
    ref_pil = _tensor_to_pil(face_reference)

    target_np = np.array(target_pil)[:, :, ::-1]
    ref_np = np.array(ref_pil)[:, :, ::-1]

    analyser = _get_face_analyser()
    target_faces = analyser.get(target_np)
    ref_faces = analyser.get(ref_np)

    if len(ref_faces) == 0:
        logger.warning("FatMex FaceSwap (quick): No face in reference. Returning target.")
        return target_image
    if len(target_faces) == 0:
        logger.warning("FatMex FaceSwap (quick): No face in target. Returning target.")
        return target_image

    target_faces = sorted(target_faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]), reverse=True)
    ref_face = sorted(ref_faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]), reverse=True)[0]

    face_idx = min(face_index, len(target_faces) - 1)
    target_face = target_faces[face_idx]

    import folder_paths
    swapper_path = None
    for search_dir in ["insightface", "reactor"]:
        model_dir = os.path.join(folder_paths.models_dir, search_dir)
        for fname in ["inswapper_128.onnx", "inswapper_128_fp16.onnx"]:
            candidate = os.path.join(model_dir, fname)
            if os.path.exists(candidate):
                swapper_path = candidate
                break
        if swapper_path is None:
            models_subdir = os.path.join(model_dir, "models")
            if os.path.exists(models_subdir):
                for fname in ["inswapper_128.onnx", "inswapper_128_fp16.onnx"]:
                    candidate = os.path.join(models_subdir, fname)
                    if os.path.exists(candidate):
                        swapper_path = candidate
                        break
        if swapper_path:
            break

    if swapper_path is None:
        logger.warning("FatMex FaceSwap: inswapper_128.onnx not found.")
        return target_image

    swapper = insightface.model_zoo.get_model(swapper_path, providers=_get_onnx_providers())
    result = swapper.get(target_np, target_face, ref_face, paste_back=True)

    if blend_strength < 1.0:
        result = (result * blend_strength + target_np * (1.0 - blend_strength)).astype(np.uint8)

    result_rgb = result[:, :, ::-1]
    result_pil = Image.fromarray(result_rgb)
    return _pil_to_tensor(result_pil)


# ── Quality mode: YOLOv8 detect → inpaint face region ───────────────────

def _quality_swap(target_image, face_reference, model, clip, vae,
                  denoise=0.6, feather=13, steps=4, cfg=1.0,
                  sampler_name="euler", scheduler="beta",
                  upscale_face=True):
    """
    FaceDetailer-style swap: detect face bbox with YOLOv8, then inpaint
    the face region using the provided image-editing model with a
    face-matching prompt.
    """
    import comfy.samplers
    import comfy.sample
    import comfy.model_management
    import nodes
    import torch

    # 1. Detect face in target with YOLOv8
    target_pil = _tensor_to_pil(target_image)
    target_np = np.array(target_pil)

    yolo = _get_yolo_detector()
    results = yolo(target_np, verbose=False)

    if len(results) == 0 or len(results[0].boxes) == 0:
        logger.warning("FatMex FaceSwap (quality): No face detected in target. Returning unchanged.")
        return target_image

    # Get the largest face bbox
    boxes = results[0].boxes
    areas = (boxes.xyxy[:, 2] - boxes.xyxy[:, 0]) * (boxes.xyxy[:, 3] - boxes.xyxy[:, 1])
    best_idx = areas.argmax().item()
    x1, y1, x2, y2 = boxes.xyxy[best_idx].cpu().numpy().astype(int)

    h, w = target_np.shape[:2]

    # 2. Expand bbox (dilation + crop factor like icekub)
    bbox_w = x2 - x1
    bbox_h = y2 - y1
    dilation = 22
    crop_factor = 2.5

    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    crop_w = bbox_w * crop_factor + dilation * 2
    crop_h = bbox_h * crop_factor + dilation * 2

    # Expanded crop region
    cx1 = max(0, int(cx - crop_w / 2))
    cy1 = max(0, int(cy - crop_h / 2))
    cx2 = min(w, int(cx + crop_w / 2))
    cy2 = min(h, int(cy + crop_h / 2))

    # 3. Create feathered mask for the face region
    mask = np.zeros((h, w), dtype=np.float32)
    # Inner face region (fully masked)
    ix1 = max(0, x1 - dilation)
    iy1 = max(0, y1 - dilation)
    ix2 = min(w, x2 + dilation)
    iy2 = min(h, y2 + dilation)
    mask[iy1:iy2, ix1:ix2] = 1.0

    # Apply gaussian feathering
    if feather > 0:
        from scipy.ndimage import gaussian_filter
        mask = gaussian_filter(mask, sigma=feather)
        mask = np.clip(mask, 0.0, 1.0)

    mask_tensor = torch.from_numpy(mask).unsqueeze(0)  # (1, H, W)

    # 4. Encode the face-matching prompt
    prompt_text = (
        "Adjust the source image lighting and skin tone to ensure the new face "
        "fits in the new body seamlessly. Match the face shape of the source image. "
        "Keep everything else the same."
    )
    negative_text = ""

    tokens_pos = clip.tokenize(prompt_text)
    cond_pos, pooled_pos = clip.encode_from_tokens(tokens_pos, return_pooled=True, return_dict=False)
    positive = [[cond_pos, {"pooled_output": pooled_pos}]]

    tokens_neg = clip.tokenize(negative_text)
    cond_neg, pooled_neg = clip.encode_from_tokens(tokens_neg, return_pooled=True, return_dict=False)
    negative = [[cond_neg, {"pooled_output": pooled_neg}]]

    # 5. Encode the target image to latent
    target_for_vae = target_image  # (B, H, W, C)
    encoded = vae.encode(target_for_vae[:, :, :, :3])

    # 6. Create noise mask from our feathered mask
    latent_h = encoded.shape[2]
    latent_w = encoded.shape[3]
    mask_resized = torch.nn.functional.interpolate(
        mask_tensor.unsqueeze(0), size=(latent_h, latent_w), mode='bilinear'
    ).squeeze(0)

    # 7. Sample (inpaint the face region)
    noise = comfy.sample.prepare_noise(encoded, seed=torch.randint(0, 2**32, (1,)).item())

    sampler_obj = comfy.samplers.KSampler(
        model, steps=steps, device=comfy.model_management.get_torch_device(),
        sampler=sampler_name, scheduler=scheduler, denoise=denoise
    )

    samples = sampler_obj.sample(
        noise, positive, negative, cfg=cfg,
        latent_image=encoded, noise_mask=mask_resized
    )

    # 8. Decode back to image
    result = vae.decode(samples)

    # 9. Blend: use our feathered mask to composite result onto original
    mask_blend = mask_tensor.unsqueeze(-1)  # (1, H, W, 1)
    if result.shape[1] != h or result.shape[2] != w:
        result = torch.nn.functional.interpolate(
            result.permute(0, 3, 1, 2), size=(h, w), mode='bilinear'
        ).permute(0, 2, 3, 1)

    final = result * mask_blend + target_image * (1.0 - mask_blend)

    # 10. Optional face upscale
    if upscale_face:
        try:
            up_model = _get_upscale_model()
            if up_model is not None:
                face_crop = final[:, iy1:iy2, ix1:ix2, :]
                if face_crop.shape[1] > 0 and face_crop.shape[2] > 0:
                    face_for_upscale = face_crop.permute(0, 3, 1, 2).to(comfy.model_management.get_torch_device())
                    with torch.no_grad():
                        upscaled = up_model(face_for_upscale)
                    upscaled = torch.nn.functional.interpolate(
                        upscaled, size=(iy2-iy1, ix2-ix1), mode='lanczos', antialias=True
                    )
                    upscaled = upscaled.permute(0, 2, 3, 1).cpu()
                    face_mask = np.zeros((h, w), dtype=np.float32)
                    face_mask[iy1:iy2, ix1:ix2] = 1.0
                    if feather > 0:
                        face_mask = gaussian_filter(face_mask, sigma=feather//2)
                    face_mask_t = torch.from_numpy(face_mask[iy1:iy2, ix1:ix2]).unsqueeze(0).unsqueeze(-1)
                    final[:, iy1:iy2, ix1:ix2, :] = (
                        upscaled * face_mask_t + final[:, iy1:iy2, ix1:ix2, :] * (1.0 - face_mask_t)
                    )
        except Exception as e:
            logger.warning(f"FatMex FaceSwap: Face upscale failed ({e}), skipping.")

    return final


# ── Main Node ────────────────────────────────────────────────────────────

class FatMexFaceSwap:
    """
    One-node face swap with two modes:
      Quick  – InsightFace inswapper (fast, no model needed)
      Quality – YOLOv8 detect + inpaint with editing model (matches icekub quality)
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "target_image": ("IMAGE", {"tooltip": "The generated image to swap a face onto."}),
                "face_reference": ("IMAGE", {"tooltip": "Clear photo of the face you want to use."}),
            },
            "optional": {
                "model": ("MODEL", {"tooltip": "For quality mode: connect MODEL from a Qwen Image Edit loader."}),
                "clip": ("CLIP", {"tooltip": "For quality mode: connect CLIP from the same loader."}),
                "vae": ("VAE", {"tooltip": "For quality mode: connect VAE from the same loader."}),
                "face_index": ("INT", {"default": 0, "min": 0, "max": 10,
                                       "tooltip": "Which face to swap (0 = largest)."}),
                "blend_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                                             "tooltip": "Quick mode: blend strength. Quality mode: ignored."}),
                "method": (["auto", "quick", "quality"], {
                    "default": "auto",
                    "tooltip": "auto = quality if MODEL connected, else quick."
                }),
                "denoise": ("FLOAT", {"default": 0.6, "min": 0.1, "max": 1.0, "step": 0.05,
                                      "tooltip": "Quality mode: how much to regenerate the face (0.6 = natural)."}),
                "feather": ("INT", {"default": 13, "min": 0, "max": 50, "step": 1,
                                    "tooltip": "Quality mode: mask edge feathering in pixels."}),
                "steps": ("INT", {"default": 4, "min": 1, "max": 30,
                                  "tooltip": "Quality mode: sampling steps for face inpaint."}),
                "upscale_face": ("BOOLEAN", {"default": True,
                                             "tooltip": "Quality mode: upscale face region with 4xFaceUpDAT."}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("swapped_image",)
    FUNCTION = "swap_face"
    CATEGORY = "Fat Mex"
    DESCRIPTION = (
        "Face swap with two modes:\n"
        "• Quick – InsightFace inswapper paste (fast, no extra model needed)\n"
        "• Quality – YOLOv8 detect → inpaint face with editing model + 4x upscale\n"
        "Connect MODEL/CLIP/VAE from a Qwen Image Edit loader for quality mode."
    )

    def swap_face(self, target_image, face_reference,
                  model=None, clip=None, vae=None,
                  face_index=0, blend_strength=1.0,
                  method="auto", denoise=0.6, feather=13, steps=4,
                  upscale_face=True):

        # Determine mode
        use_quality = False
        if method == "quality":
            if model is None or clip is None or vae is None:
                logger.warning("FatMex FaceSwap: Quality mode requires MODEL/CLIP/VAE. Falling back to quick.")
            else:
                use_quality = True
        elif method == "auto":
            use_quality = (model is not None and clip is not None and vae is not None)

        # Process batch
        batch_size = target_image.shape[0]
        results = []

        for i in range(batch_size):
            single_target = target_image[i:i+1]

            if use_quality:
                result = _quality_swap(
                    single_target, face_reference, model, clip, vae,
                    denoise=denoise, feather=feather, steps=steps,
                    cfg=1.0, sampler_name="euler", scheduler="beta",
                    upscale_face=upscale_face
                )
            else:
                result = _quick_swap(
                    single_target, face_reference,
                    face_index=face_index, blend_strength=blend_strength
                )

            results.append(result)

        return (torch.cat(results, dim=0),)


FACESWAP_CLASS_MAPPINGS = {
    "FatMexFaceSwap": FatMexFaceSwap,
}

FACESWAP_NAME_MAPPINGS = {
    "FatMexFaceSwap": "Fat Mex Face Swap",
}
