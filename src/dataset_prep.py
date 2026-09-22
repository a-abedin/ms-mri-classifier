
import os
from typing import Dict, Generator, List, Optional, Tuple
import nibabel as nib
import numpy as np
from PIL import Image


class NiftiSliceExtractor:
    """Extracts, normalizes, and labels 2D slices from 3D NIfTI MRI volumes."""

    PLANES: Dict[str, int] = {
        "sagittal": 0,
        "coronal": 1,
        "axial": 2,
    }

    def __init__(
        self,
        target_size: Tuple[int, int] = (224, 224),
        crop_boundary_ratio: float = 0.15,
    ) -> None:
        self.target_size = target_size
        self.crop_boundary_ratio = crop_boundary_ratio

    @staticmethod
    def load_nifti_data(file_path: str) -> np.ndarray:
        """Loads NIfTI volume data safely as a float array."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        nimg = nib.load(file_path)
        return np.asanyarray(nimg.dataobj, dtype=np.float32)

    @staticmethod
    def safe_normalize_to_uint8(slice_img: np.ndarray) -> np.ndarray:
        """Normalizes slice intensity to [0, 255] uint8 without division by zero."""
        max_val = np.max(slice_img)
        min_val = np.min(slice_img)

        if max_val - min_val > 1e-6:
            norm = (slice_img - min_val) / (max_val - min_val) * 255.0
            return norm.astype(np.uint8)
        return np.zeros_like(slice_img, dtype=np.uint8)

    def extract_slices_from_volume(
        self,
        image_vol: np.ndarray,
        mask_vol: np.ndarray,
        plane: str = "sagittal",
    ) -> Generator[Tuple[int, np.ndarray, int], None, None]:
        """Extracts filtered 2D slices and binary labels across the specified anatomical plane."""
        axis = self.PLANES.get(plane.lower())
        if axis is None:
            raise ValueError(f"Invalid plane '{plane}'. Supported: {list(self.PLANES.keys())}")

        total_slices = image_vol.shape[axis]
        lower_bound = int(total_slices * self.crop_boundary_ratio)
        upper_bound = int(total_slices * (1.0 - self.crop_boundary_ratio))

        for idx in range(lower_bound, upper_bound):
            if axis == 0:
                im_slice = image_vol[idx, :, :]
                lb_slice = mask_vol[idx, :, :]
            elif axis == 1:
                im_slice = image_vol[:, idx, :]
                lb_slice = mask_vol[:, idx, :]
            else:
                im_slice = image_vol[:, :, idx]
                lb_slice = mask_vol[:, :, idx]

            norm_img = self.safe_normalize_to_uint8(im_slice)
            binary_label = 1 if np.max(lb_slice) > 0 else 0

            yield idx, norm_img, binary_label

    def save_slice_as_png(self, slice_array: np.ndarray, output_path: str) -> None:
        """Saves a 2D uint8 numpy array as a PNG image."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        img = Image.fromarray(slice_array)
        img.save(output_path)
