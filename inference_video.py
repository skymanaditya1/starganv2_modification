# A sequence of source frames and a reference frame is given as input
# The style vector to condition on is generated from the reference frame 
from munch import Munch
from core.data_loader import get_train_loader, get_test_loader
from core.data_loader import InputFetcher
import copy, os
import torch
from utils import denormalize

img_size = 256
val_batch_size = 16
num_workers = 4
latent_dim = 16 # this is the dimension of the latent code (z)
style_dim = 64 # this is the dimension of the style code (s)
num_domains = 2
w_hpf = 1

image_path = 'assets/representative/celeba_hq/src/male/005735.jpg'
# ref_image_path = 'assets/representative/celeba_hq/src/male/006930.jpg'
ref_image_path = 'assets/representative/celeba_hq/src/male/005735.jpg'
from PIL import Image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

img = Image.open(image_path).convert('RGB')
ref_img = Image.open(ref_image_path).convert('RGB')

import torchvision.transforms as transforms
import torchvision.utils as vision_utils

transform = transforms.Compose([transforms.Resize([256, 256]), 
            transforms.ToTensor(), 
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

img = transform(img).unsqueeze(0).to(device) # Adding the dummy batch dimension
ref_img = transform(ref_img).unsqueeze(0).to(device) # Adding the dummy batch dimension
print(img.shape)

import torch.nn as nn 
from core.model import Generator, MappingNetwork, StyleEncoder, Discriminator

generator = nn.DataParallel(Generator(img_size=img_size, style_dim=style_dim, w_hpf=w_hpf))
mapping_network = nn.DataParallel(MappingNetwork(latent_dim, style_dim, num_domains))
style_encoder = nn.DataParallel(StyleEncoder(img_size, style_dim, num_domains))
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
    ]

resume_iter = 100000
for ckptio in ckptios:
    ckptio.load(resume_iter)

ref_y = torch.tensor([1]).to(device)
style_code = nets_ema.style_encoder(ref_img, ref_y)
print(f'Style code dim : {style_code.shape}')

masks = nets_ema.fan.get_heatmap(img) if w_hpf > 0 else None

# Generate the fake image using the content from the source image and the style from the ref image
fake_x = nets_ema.generator(img, style_code, masks=masks)
fake_denormalized = denormalize(fake_x)

import cv2

save_dir = '/home2/aditya1/cvit/content_sync/stargan-v2/custom4'
os.makedirs(save_dir, exist_ok=True)

filename = os.path.join(save_dir, 'generated.jpg')
vision_utils.save_image(fake_denormalized, filename)
print(f'Generated fake image written successfully')

src_filename = os.path.join(save_dir, 'source.jpg')
ref_filename = os.path.join(save_dir, 'reference.jpg')

vision_utils.save_image(denormalize(img).cpu(), src_filename)
vision_utils.save_image(denormalize(ref_img).cpu(), ref_filename)

# reconstruct the image using fake_image and style from source 
src_y = torch.tensor([1]).to(device)
style_source = nets_ema.style_encoder(img, src_y)
masks = nets_ema.fan.get_heatmap(fake_x) if w_hpf > 0 else None
x_rec = nets_ema.generator(fake_x, style_source, masks=masks)

rec_filename = os.path.join(save_dir, 'reconstructed.jpg')
vision_utils.save_image(denormalize(x_rec).cpu(), rec_filename)