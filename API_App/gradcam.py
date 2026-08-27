"""Minimal Grad-CAM implementation based on forward/backward hooks.

Written by hand instead of pulling in the `grad-cam` package: it is ~80 lines, removes a
dependency from the API image, and makes the target-layer choice explicit.
"""

from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


class GradCAM:
    """
    Grad-CAM for any CNN whose target layer emits an NCHW feature map.

    For torchvision's ConvNeXt the natural target is `model.features[-1]`, which outputs
    [B, 768, 7, 7] at 224x224 input. No reshape_transform is needed (unlike ViT-style
    backbones), because ConvNeXt keeps the spatial layout throughout.
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self._handles = [
            target_layer.register_forward_hook(self._save_activation),
            target_layer.register_full_backward_hook(self._save_gradient),
        ]

    def _save_activation(self, module, inputs, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.remove()

    def __call__(self, input_tensor: torch.Tensor, class_idx: Optional[int] = None):
        """
        Returns
        -------
        cam : np.ndarray, shape (H, W), values in [0, 1]
        class_idx : int
            The class the explanation was computed for.
        """
        self.model.zero_grad(set_to_none=True)

        # Gradients are required even though the model is in eval mode
        with torch.enable_grad():
            logits = self.model(input_tensor)
            if class_idx is None:
                class_idx = int(logits.argmax(dim=1).item())
            logits[:, class_idx].sum().backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks captured nothing: wrong target layer?")

        # Channel importance = spatially averaged gradient
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))

        cam = F.interpolate(cam, size=input_tensor.shape[-2:],
                            mode='bilinear', align_corners=False)
        cam = cam[0, 0].cpu().numpy()

        cam_min, cam_max = float(cam.min()), float(cam.max())
        if cam_max - cam_min < 1e-8:
            # Flat map: nothing to show, return zeros rather than amplifying noise
            return np.zeros_like(cam), class_idx
        cam = (cam - cam_min) / (cam_max - cam_min)
        return cam, class_idx


def _jet_colormap(values: np.ndarray) -> np.ndarray:
    """Piecewise-linear JET approximation, so matplotlib is not needed at serving time."""
    v = np.clip(values, 0.0, 1.0)
    four = 4.0 * v
    r = np.clip(np.minimum(four - 1.5, -four + 4.5), 0, 1)
    g = np.clip(np.minimum(four - 0.5, -four + 3.5), 0, 1)
    b = np.clip(np.minimum(four + 0.5, -four + 2.5), 0, 1)
    return (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)


def overlay_cam_on_image(pil_image: Image.Image, cam: np.ndarray, alpha: float = 0.5,
                         size: int = 224) -> Image.Image:
    """Blend the heatmap over the resized original slice."""
    base = pil_image.convert('RGB').resize((size, size))
    heat = Image.fromarray(_jet_colormap(cam)).resize((size, size))
    return Image.blend(base, heat, alpha=float(np.clip(alpha, 0.0, 1.0)))
