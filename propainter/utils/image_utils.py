from dataclasses import dataclass, field

import numpy as np
import scipy
import torch
from numpy.typing import NDArray
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image


@dataclass
class ImageConfig:
    width: int
    height: int
    mask_dilates: int
    flow_mask_dilates: int
    input_size: tuple[int, int]
    video_length: int
    process_size: tuple[int, int] = field(init=False)

    def __post_init__(self) -> None:
        """Initialize process size."""
        self.process_size = (
            self.width - self.width % 8,
            self.height - self.height % 8,
        )


@dataclass
class ImageOutpaintConfig(ImageConfig):
    width_scale: float
    height_scale: float
    process_size: tuple[int, int] = field(init=False)
    outpaint_size: tuple[int, int] = field(init=False)

    # TODO: Refactor
    def __post_init__(self) -> None:
        """Initialize output size for outpainting."""
        self.process_size = (
            self.width - self.width % 8,
            self.height - self.height % 8,
        )
        pad_image_width = int(self.width_scale * self.width)
        pad_image_height = int(self.height_scale * self.height)
        self.outpaint_size = (
            pad_image_width - pad_image_width % 8,
            pad_image_height - pad_image_height % 8,
        )


class Stack:
    """Stack images based on number of channels."""

    def __init__(self, roll=False):
        self.roll = roll

    def __call__(self, img_group) -> NDArray:
        mode = img_group[0].mode
        if mode == "1":
            img_group = [img.convert("L") for img in img_group]
            mode = "L"
        if mode == "L":
            return np.stack([np.expand_dims(x, 2) for x in img_group], axis=2)
        if mode == "RGB":
            if self.roll:
                return np.stack([np.array(x)[:, :, ::-1] for x in img_group], axis=2)
            return np.stack(img_group, axis=2)
        raise NotImplementedError(f"Image mode {mode}")


class ToTorchFormatTensor:
    """Converts a PIL.Image (RGB) or numpy.ndarray (H x W x C) in the range [0, 255] to a torch FloatTensor of shape (C x H x W) in the range [0.0, 1.0]."""

    # TODO: Check if this function is necessary with comfyUI workflow.
    def __init__(self, div=True):
        self.div = div

    def __call__(self, pic) -> torch.Tensor:
        if isinstance(pic, np.ndarray):
            # numpy img: [L, C, H, W]
            img = torch.from_numpy(pic).permute(2, 3, 0, 1).contiguous()
        else:
            # handle PIL Image
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
            img = img.view(pic.size[1], pic.size[0], len(pic.mode))
            # put it from HWC to CHW format
            # yikes, this transpose takes 80% of the loading time/CPU
            img = img.transpose(0, 1).transpose(0, 2).contiguous()
        img = img.float().div(255) if self.div else img.float()
        return img


def to_tensors():
    return transforms.Compose([Stack(), ToTorchFormatTensor()])


def resize_images(images: list[NDArray], config: ImageConfig, is_mask: bool = False) -> list[NDArray]:
    """Resizes images using OpenCV for better performance.
    
    Works directly with numpy arrays instead of PIL Images for better performance.
    
    Args:
        images: List of numpy arrays to resize
        config: ImageConfig with process_size and input_size
        is_mask: If True, use INTER_NEAREST for masks (preserves sharp edges)
                 If False, use INTER_LINEAR for regular images (smoother interpolation)
    """
    if config.process_size != config.input_size:
        import cv2
        # Use INTER_NEAREST for masks to preserve hard edges, INTER_LINEAR for images
        interpolation = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
        resized = []
        for img in images:
            resized_img = cv2.resize(img, config.process_size, interpolation=interpolation)
            resized.append(resized_img)
        return resized
    return images


def convert_image_to_frames(images: torch.Tensor) -> list[NDArray]:
    """Convert a batch of PyTorch tensors directly to numpy arrays (no PIL conversion).
    
    This is more efficient than converting to PIL Images as an intermediate step.
    
    Args:
        images: PyTorch tensor of shape [B, H, W, C] with values in [0, 1] or [0, 255]
        
    Returns:
        List of numpy arrays in uint8 format [0, 255] range
        
    Note:
        Automatically detects input range (0-1 or 0-255) and converts accordingly
    """
    frames = []
    for image in images:
        # Move to CPU and convert to numpy
        np_frame = image.detach().cpu().numpy()
        # Convert from [0, 1] to [0, 255] if needed
        if np_frame.max() <= 1.0:
            np_frame = (np_frame * 255).clip(0, 255).astype(np.uint8)
        else:
            np_frame = np_frame.clip(0, 255).astype(np.uint8)
        frames.append(np_frame)

    return frames


def binary_mask(mask: np.ndarray, th: float = 0.1) -> np.ndarray:
    return (mask > th).astype(mask.dtype)


def convert_mask_to_frames(images: torch.Tensor) -> list[NDArray]:
    """Convert mask tensors to numpy arrays (no PIL conversion).
    
    More efficient than PIL Image conversion.
    Handles both float (0-1 range) and uint8 masks across all float types.
    """
    frames = []
    for image in images:
        image = image.detach().cpu()

        # Adjust scaling based on dtype - handle all float types (float16, float32, float64, bfloat16)
        if image.dtype in (torch.float16, torch.float32, torch.float64, torch.bfloat16):
            # Assume float masks are in [0, 1] range, scale to [0, 255]
            image = (image * 255).clamp(0, 255).byte()
        elif image.dtype == torch.uint8:
            # Already in correct range
            pass
        else:
            # For other dtypes, convert to float32 first
            image = (image.float() * 255).clamp(0, 255).byte()
        
        np_frame = image.numpy()
        # Ensure proper shape (H, W) for grayscale
        if np_frame.ndim == 3:
            np_frame = np_frame.squeeze()
        
        frames.append(np_frame)

    return frames


def read_masks(
    masks: torch.Tensor, config: ImageConfig
) -> tuple[list[NDArray], list[NDArray]]:
    """Process masks with dilation using numpy arrays instead of PIL."""
    mask_images = convert_mask_to_frames(masks)
    mask_images = resize_images(mask_images, config, is_mask=True)
    masks_dilated: list[NDArray] = []
    flow_masks: list[NDArray] = []

    for mask_array in mask_images:
        # Ensure 2D array
        if mask_array.ndim > 2:
            mask_array = mask_array.squeeze()

        # Dilate for flow mask
        if config.flow_mask_dilates > 0:
            flow_mask_img = scipy.ndimage.binary_dilation(
                mask_array, iterations=config.flow_mask_dilates
            ).astype(np.uint8) * 255
        else:
            flow_mask_img = binary_mask(mask_array.astype(np.float32)).astype(np.uint8) * 255
        flow_masks.append(flow_mask_img)

        # Dilate for inpainting mask
        if config.mask_dilates > 0:
            mask_array_dilated = scipy.ndimage.binary_dilation(
                mask_array, iterations=config.mask_dilates
            ).astype(np.uint8) * 255
        else:
            mask_array_dilated = binary_mask(mask_array.astype(np.float32)).astype(np.uint8) * 255
        masks_dilated.append(mask_array_dilated)

    if len(mask_images) == 1:
        flow_masks = flow_masks * config.video_length
        masks_dilated = masks_dilated * config.video_length

    return flow_masks, masks_dilated


def prepare_frames_and_masks(
    frames: list[NDArray],
    mask: torch.Tensor,
    config: ImageConfig,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[NDArray]]:
    frames = resize_images(frames, config)

    flow_masks, masks_dilated = read_masks(mask, config)

    original_frames = frames  # Already numpy arrays
    
    # Stack frames: list[(H,W,C)] -> (H, W, T, C) then permute to (T, C, H, W)
    frames_stacked = np.stack(frames, axis=2)  # (H, W, T, C)
    frames_tensor = torch.from_numpy(frames_stacked).permute(2, 3, 0, 1).contiguous().float() / 255.0
    frames_tensor = frames_tensor.unsqueeze(0) * 2 - 1  # (1, T, C, H, W)
    
    # Masks are 2D (H, W), add channel dimension: (H, W, 1) before stacking
    flow_masks_with_c = [m if m.ndim == 3 else np.expand_dims(m, axis=2) for m in flow_masks]
    flow_masks_stacked = np.stack(flow_masks_with_c, axis=2)  # (H, W, T, C)
    flow_masks_tensor = torch.from_numpy(flow_masks_stacked).permute(2, 3, 0, 1).contiguous().float() / 255.0
    flow_masks_tensor = flow_masks_tensor.unsqueeze(0)  # (1, T, C, H, W)
    
    masks_dilated_with_c = [m if m.ndim == 3 else np.expand_dims(m, axis=2) for m in masks_dilated]
    masks_dilated_stacked = np.stack(masks_dilated_with_c, axis=2)  # (H, W, T, C)
    masks_dilated_tensor = torch.from_numpy(masks_dilated_stacked).permute(2, 3, 0, 1).contiguous().float() / 255.0
    masks_dilated_tensor = masks_dilated_tensor.unsqueeze(0)  # (1, T, C, H, W)
    
    frames_tensor, flow_masks_tensor, masks_dilated_tensor = (
        frames_tensor.to(device),
        flow_masks_tensor.to(device),
        masks_dilated_tensor.to(device),
    )
    return frames_tensor, flow_masks_tensor, masks_dilated_tensor, original_frames


def extrapolation(
    resized_frames: list[NDArray], image_config: ImageOutpaintConfig
) -> tuple[list[NDArray], list[NDArray], list[NDArray]]:
    """Prepares the data for video outpainting using numpy arrays."""
    resized_frames = resize_images(resized_frames, image_config)

    resized_height, resized_width = resized_frames[0].shape[:2]
    pad_image_width, pad_image_height = image_config.outpaint_size

    # Defines new FOV.
    width_start = int((pad_image_width - resized_width) / 2)
    height_start = int((pad_image_height - resized_height) / 2)

    # Extrapolates the FOV for video.
    extrapolated_frames = []
    for v in resized_frames:
        frame = np.zeros(((pad_image_height, pad_image_width, 3)), dtype=np.uint8)
        frame[
            height_start : height_start + resized_height,
            width_start : width_start + resized_width,
            :,
        ] = v
        extrapolated_frames.append(frame)

    # Generates the mask for missing region.
    masks_dilated = []
    flow_masks = []

    dilate_h = 4 if height_start > 10 else 0
    dilate_w = 4 if width_start > 10 else 0
    mask = np.ones(((pad_image_height, pad_image_width)), dtype=np.uint8)

    mask[
        height_start + dilate_h : height_start + resized_height - dilate_h,
        width_start + dilate_w : width_start + resized_width - dilate_w,
    ] = 0
    flow_masks.append(mask * 255)

    mask[
        height_start : height_start + resized_height,
        width_start : width_start + resized_width,
    ] = 0
    masks_dilated.append(mask * 255)

    flow_masks = flow_masks * image_config.video_length
    masks_dilated = masks_dilated * image_config.video_length

    return (
        extrapolated_frames,
        flow_masks,
        masks_dilated,
    )


def prepare_frames_and_masks_for_outpaint(
    frames: list[NDArray],
    flow_masks: list[NDArray],
    masks_dilated: list[NDArray],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[NDArray]]:
    # frames and masks are already numpy arrays
    original_frames = frames
    
    # Stack frames: list[(H,W,C)] -> (H, W, T, C) then permute to (T, C, H, W)
    frames_stacked = np.stack(frames, axis=2)  # (H, W, T, C)
    frames_tensor = torch.from_numpy(frames_stacked).permute(2, 3, 0, 1).contiguous().float() / 255.0
    frames_tensor = frames_tensor.unsqueeze(0) * 2 - 1  # (1, T, C, H, W)
    
    # Masks are 2D (H, W), add channel dimension: (H, W, 1) before stacking
    flow_masks_with_c = [m if m.ndim == 3 else np.expand_dims(m, axis=2) for m in flow_masks]
    flow_masks_stacked = np.stack(flow_masks_with_c, axis=2)  # (H, W, T, C)
    flow_masks_tensor = torch.from_numpy(flow_masks_stacked).permute(2, 3, 0, 1).contiguous().float() / 255.0
    flow_masks_tensor = flow_masks_tensor.unsqueeze(0)  # (1, T, C, H, W)
    
    masks_dilated_with_c = [m if m.ndim == 3 else np.expand_dims(m, axis=2) for m in masks_dilated]
    masks_dilated_stacked = np.stack(masks_dilated_with_c, axis=2)  # (H, W, T, C)
    masks_dilated_tensor = torch.from_numpy(masks_dilated_stacked).permute(2, 3, 0, 1).contiguous().float() / 255.0
    masks_dilated_tensor = masks_dilated_tensor.unsqueeze(0)  # (1, T, C, H, W)
    
    frames_tensor, flow_masks_tensor, masks_dilated_tensor = (
        frames_tensor.to(device),
        flow_masks_tensor.to(device),
        masks_dilated_tensor.to(device),
    )
    return frames_tensor, flow_masks_tensor, masks_dilated_tensor, original_frames


def handle_output(
    composed_frames: list[NDArray],
    flow_masks: torch.Tensor,
    masks_dilated: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    output_array = np.stack(composed_frames, axis=0).astype(np.float32) / 255.0
    output_images = torch.from_numpy(output_array)

    output_flow_masks = flow_masks.squeeze()
    output_masks_dilated = masks_dilated.squeeze()

    return output_images, output_flow_masks, output_masks_dilated
