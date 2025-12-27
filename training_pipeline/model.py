"""
Model wrapper for VRT - Video Super-Resolution
Version 1: Direct import from existing network_vrt.py
"""

from models.network_vrt import VRT

def build_vrt_sr_model(config):
    """Build VRT model for video super-resolution"""
    model = VRT(
        upscale=config.scale,  # 4x super-resolution
        img_size=[config.num_frames, config.patch_size, config.patch_size],
        window_size=config.window_size,
        depths=config.depths,
        embed_dims=config.embed_dims,
        num_heads=config.num_heads,
        pa_frames=config.pa_frames,  # Parallel warping frames
        deformable_groups=config.deformable_groups,
        nonblind_denoising=False,  # False for SR
        use_checkpoint_attn=config.use_checkpoint_attn,
        use_checkpoint_ffn=config.use_checkpoint_ffn
    )
    return model