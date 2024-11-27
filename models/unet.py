"""
    define an adapted UNet model
    based on the implementation of MONAI
    (https://docs.monai.io/en/stable/networks.html)
"""

from monai.networks.nets import UNet

class UNet(UNet):
    """
    Unet using the MONAI implementation, working with specific config file.
    """

    def __init__(self, config):
        """
        Initialize the UNet model with the provided configuration.
        """
        super().__init__(
            spatial_dims=config["spatial_dims"],
            in_channels=config["in_channels"],
            out_channels=config["out_channels"],
            channels=config["channels"],
            strides=config["strides"],
            kernel_size=config["kernel_size"],
        )

    def forward(self, x):
        """
        Forward pass of the adapted UNet model.
        """
        return super().forward(x)
