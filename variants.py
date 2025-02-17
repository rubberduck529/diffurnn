# variants.py

from model import DEN

def imagenet_base(num_classes=1000, **kwargs):
    """
    Constructs the ImageNet Base variant of the DEN model.
    
    Hyperparameters:
      - img_size: 224
      - patch_size: 16 (224/16 = 14x14 patches)
      - embed_dim: 768
      - depth: 12 layers
      - num_classes: 1000 (for ImageNet)
    """
    return DEN(
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=num_classes,
        embed_dim=768,
        depth=12,
        **kwargs
    )

def imagenet_large(num_classes=1000, **kwargs):
    """
    Constructs the ImageNet Large variant of the DEN model.
    
    Hyperparameters:
      - img_size: 224
      - patch_size: 16 (224/16 = 14x14 patches)
      - embed_dim: 1024
      - depth: 24 layers
      - num_classes: 1000 (for ImageNet)
    """
    return DEN(
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=num_classes,
        embed_dim=1024,
        depth=24,
        **kwargs
    )

def imagenet_huge(num_classes=1000, **kwargs):
    """
    Constructs the ImageNet Huge variant of the DEN model.
    
    Hyperparameters:
      - img_size: 224
      - patch_size: 16 (224/16 = 14x14 patches)
      - embed_dim: 1280
      - depth: 32 layers
      - num_classes: 1000 (for ImageNet)
    """
    return DEN(
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=num_classes,
        embed_dim=1280,
        depth=32,
        **kwargs
    )

# Dictionary mapping variant names to builder functions.
VARIANTS = {
    "imagenet_base": imagenet_base,
    "imagenet_large": imagenet_large,
    "imagenet_huge": imagenet_huge,
}

if __name__ == "__main__":
    try:
        from ptflops import get_model_complexity_info
    except ImportError:
        raise ImportError("Please install ptflops to compute FLOPs (pip install ptflops)")

    for variant_name, build_fn in VARIANTS.items():
        model = build_fn()
        # Use the image size defined in the patch embedding (default is 224 for ImageNet variants).
        img_size = model.patch_embed.img_size if hasattr(model.patch_embed, "img_size") else 224
        input_res = (3, img_size, img_size)
        flops, params = get_model_complexity_info(model, input_res, as_strings=True, print_per_layer_stat=False)
        print(f"{variant_name}: {params} parameters, {flops} FLOPs")
