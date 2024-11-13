import torch
from monai.transforms import Resize


class ResizeWithLandmarksd(Resize):
    """
    TODO: Docstring
    """

    def __init__(
        self,
        spatial_size,
        mode="bilinear",
        align_corners=False,
        keys=["image", "landmarks"],
    ):
        super().__init__(
            spatial_size=spatial_size, mode=mode, align_corners=align_corners
        )
        self.keys = keys

    def __call__(self, data, **kwargs):
        image, landmarks = data[self.keys[0]], data[self.keys[1]]
        original_height, original_width = image.shape[-2], image.shape[-1]

        image = super().__call__(image, **kwargs) 
        data[self.keys[0]] = image

        resized_height, resized_width = image.shape[-2], image.shape[-1]

        scaling_factors = torch.tensor(
            [resized_height / original_height, resized_width / original_width],
            dtype=landmarks.dtype,
        )
        data[self.keys[1]] = landmarks * scaling_factors

        return data
