def get_config(ds='unet'):
    if ds == 'unet':
        return dict(
            image_size=224
        )