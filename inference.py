from munch import Munch
from core.data_loader import get_train_loader, get_test_loader
from core.data_loader import InputFetcher
import copy, os

# setup the test_loader 
# src_dir = 'assets/representative/celeba_hq/src'
# ref_dir = 'assets/representative/celeba_hq/ref'
src_dir = 'tests1/src'
ref_dir = 'tests1/ref'

img_size = 256
val_batch_size = 16
num_workers = 4
latent_dim = 16 # this is the dimension of the latent code (z)
style_dim = 64 # this is the dimension of the style code (s)
num_domains = 2
w_hpf = 1

loaders = Munch(src=get_test_loader(root=src_dir,
                                    img_size=img_size,
                                    batch_size=val_batch_size,
                                    shuffle=False,
                                    num_workers=num_workers),
               ref=get_test_loader(root=ref_dir,
                                  img_size=img_size,
                                  batch_size=val_batch_size,
                                  shuffle=False,
                                  num_workers=num_workers))

src = next(InputFetcher(loaders.src, None, latent_dim, 'test'))
ref = next(InputFetcher(loaders.ref, None, latent_dim, 'test'))

print(f'Able to load the dataloaders successfully')

# what we are essentially doing is generating the style code from the source image 
# and using that to apply the style code on the src image 

# The style code can be generated in two ways 
# Use the encoder to generate the style code from the image 
# Use a mapping network to sample random z from the latent space and condition the generation on the code

# load the generator 
import torch.nn as nn 
from core.model import Generator, MappingNetwork, StyleEncoder, Discriminator

generator = nn.DataParallel(Generator(img_size=img_size, style_dim=style_dim, w_hpf=w_hpf))
 
# load the mapping network 
mapping_network = nn.DataParallel(MappingNetwork(latent_dim, style_dim, num_domains))

# load the style encoder network 
style_encoder = nn.DataParallel(StyleEncoder(img_size, style_dim, num_domains))

# load the discriminator 
discriminator = nn.DataParallel(Discriminator(img_size, num_domains))

generator_ema = copy.deepcopy(generator)
mapping_network_ema = copy.deepcopy(mapping_network)
style_encoder_ema = copy.deepcopy(style_encoder)

nets = Munch(generator=generator, mapping_network=mapping_network, style_encoder=style_encoder, discriminator=discriminator)
nets_ema = Munch(generator=generator_ema, mapping_network=mapping_network_ema, style_encoder=style_encoder_ema)

from core.wing import FAN
wing_path = 'expr/checkpoints/wing.ckpt'

fan = nn.DataParallel(FAN(fname_pretrained=wing_path).eval())
fan.get_heatmap = fan.module.get_heatmap
nets.fan = fan
nets_ema.fan = fan

from core.checkpoint import CheckpointIO
checkpoint_dir = 'expr/checkpoints/celeba_hq'

ckptios = [
    CheckpointIO(os.path.join(checkpoint_dir, '{:06d}_nets_ema.ckpt'), data_parallel=True, **nets_ema)
    # CheckpointIO(os.path.join(checkpoint_dir, '{:06d}_nets_ema.ckpt'), data_parallel=True, **nets_ema),
    # CheckpointIO(os.path.join(checkpoint_dir, '{:06d}_optims.ckpt'), )
]

# load the checkpoints
resume_iter = 100000
for ckptio in ckptios:
    ckptio.load(resume_iter)

# translate a source image using the style generated from a reference image 
# 0 represents the female domain
# src_x, src_y = src.x[0].unsqueeze(0), src.y[0].unsqueeze(0) # 0 represents female
# ref_x, ref_y = ref.x[0].unsqueeze(0), ref.y[0].unsqueeze(0)

# 1 represents the male domain, sample images from the male domain
src_x, src_y = src.x[1].unsqueeze(0), src.y[1].unsqueeze(0) # 0 represents female
ref_x, ref_y = ref.x[1].unsqueeze(0), ref.y[1].unsqueeze(0)

save_dir = '/home2/aditya1/cvit/content_sync/stargan-v2/custom2'
os.makedirs(save_dir, exist_ok=True)

original_src = os.path.join(save_dir, 'source.jpg')
original_ref = os.path.join(save_dir, 'reference.jpg')

import cv2 
# save the files using cv2
# cv2.imwrite(original_src, src_x.squeeze(0).permute(1, 2, 0).cpu().detach().numpy())
# cv2.imwrite(original_ref, ref_x.squeeze(0).permute(1, 2, 0).cpu().detach().numpy())

# using the torchvision utility for saving the image to the disk 
import torchvision.utils as vision_utils

# normalize the source and reference images before saving them
from utils import denormalize

vision_utils.save_image(denormalize(src.x[1].cpu()), original_src) # save the original src image 
vision_utils.save_image(denormalize(ref.x[1].cpu()), original_ref) # save the reference image

# generate the style code using the reference image 
style_code = nets_ema.style_encoder(ref_x, ref_y)
print(f'Style code dimension : {style_code.shape}')

masks = nets_ema.fan.get_heatmap(src_x) if w_hpf > 0 else None

# generate the fake image using the generator (translate source image using reference style code)
fake_x = nets_ema.generator(src_x, style_code, masks=masks)


denormalized = denormalize(fake_x)
# fake_generated = denormalized.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()

# write the generated fake image to the disk 
import cv2

filename = os.path.join(save_dir, 'generated.jpg')

# cv2.imwrite(filename, fake_generated)
vision_utils.save_image(denormalized, filename) # save the generated image using the reference style
# vision_utils.save_image(fake_x, filename) # save the image without normalization
print(f'Generated fake image written successfully')

# reconstruct the image back again using the style vector of the source image
style_src = nets_ema.style_encoder(src_x, src_y)
masks = nets_ema.fan.get_heatmap(fake_x) if w_hpf > 0 else None
x_rec = nets_ema.generator(fake_x, style_src, masks=masks)

# denormalize the reconstructed image 
denorm = denormalize(x_rec)
# fake_reconstructed = denorm.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()

# save the image 
filename_rec = os.path.join(save_dir, 'reconstructed.jpg')

# cv2.imwrite(filename_rec, fake_reconstructed)
vision_utils.save_image(denorm, filename_rec) # save the reconstructed image
# vision_utils.save_image(x_rec, filename_rec) # save the image without denormalization
print(f'Generated fake reconstructed successfully')