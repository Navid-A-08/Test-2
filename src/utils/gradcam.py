"""
Grad-CAM (Gradient-weighted Class Activation Mapping) for the ResNet18
classifier.

What it does:
    Produces a heatmap over the input image showing which regions most
    influenced the model's prediction for a given class. This works by
    hooking into the last convolutional layer (layer4 for ResNet18),
    capturing its output activations and the gradients that flow back into
    it during backprop, then combining them into a spatial importance map.

Why this matters here:
    A "malignant" prediction with no indication of *what tissue pattern*
    drove it is hard to trust or debug. Grad-CAM lets you sanity-check
    whether the model is actually looking at tumor-relevant tissue
    structure, or picking up on something spurious (background artifacts,
    staining variation, slide edges, etc.) — a real risk with a training
    set this size.

Reference: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep
Networks via Gradient-based Localization" (2017).
"""

import numpy as np
import torch
import torch.nn.functional as F


class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        self._forward_handle = target_layer.register_forward_hook(self._save_activations)
        self._backward_handle = target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, input, output):
        self.activations = output.detach()

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor: torch.Tensor, target_class: int = None):
        """
        Args:
            input_tensor: a single preprocessed image, shape (1, C, H, W).
            target_class: class index to explain. If None, uses the model's
                predicted class.

        Returns:
            cam: numpy array, shape (H, W), values in [0, 1] — the heatmap,
                 resized to match the input image's spatial size.
            predicted_class: the class index used.
        """
        self.model.eval()
        input_tensor = input_tensor.requires_grad_(True)

        output = self.model(input_tensor)  # shape (1, num_classes)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        self.model.zero_grad()
        score = output[0, target_class]
        score.backward()

        # Global-average-pool the gradients over spatial dims to get one
        # importance weight per channel, then weight the activations by it.
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (1, 1, h, w)
        cam = F.relu(cam)  # only care about features that positively support the class

        # Resize to input resolution.
        cam = F.interpolate(cam, size=input_tensor.shape[2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        # Normalize to [0, 1] for visualization.
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam, target_class

    def remove_hooks(self):
        self._forward_handle.remove()
        self._backward_handle.remove()
