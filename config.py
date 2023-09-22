
unet_config=dict(
    image_size=224,
    batch_size=32,
    num_epochs=15
)

vae_config=dict(
    image_size=32,
    batch_size=32,
    num_epochs=20,
    enc_out_dim=512,
    latent_dim=256,
    num_classes=10
)