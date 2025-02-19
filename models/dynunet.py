from monai.networks.nets import DynUNet

class DynUNet(DynUNet):
    """
    DynUnet using the MONAI implementation, working with specific config file.
    """
    def __init__(self, config):
        """
        Initialize the DynUNet model with the provided configuration.
        """
        super().__init__(
            spatial_dims=config["spatial_dims"],
            in_channels=config["in_channels"],
            out_channels=config["out_channels"],
            kernel_size=config["kernel_sizes"],  
            strides=config["strides"],
            upsample_kernel_size=config["upsample_kernels"],
            filters=config["filters"], 
            deep_supervision=config["deep_supervision"],  
            norm_name=config["norm"],  
            res_block=config["res_block"],  
        )

    def forward(self, x):
        return super().forward(x)
