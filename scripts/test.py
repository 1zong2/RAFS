import sys
sys.path.append("./")
sys.path.append("../")
sys.path.append("./submodel")
sys.path.append("./submodel/stylegan2_pytorch")
import torch
import torch.nn as nn
from submodel.stylegan2_pytorch.model import Generator, Discriminator
from submodel.psp import GradualStyleEncoder
from torchvision import transforms


transforms = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

def get_keys(d, name):
	if 'state_dict' in d:
		d = d['state_dict']
	d_filt = {k[len(name) + 1:]: v for k, v in d.items() if k[:len(name)] == name}
	return d_filt


def weight_init(m):
    if isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 0.001)
        m.bias.data.zero_()
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight.data)

    if isinstance(m, nn.ConvTranspose2d):
        nn.init.xavier_normal_(m.weight.data)

# psp
pSpEncoder = GradualStyleEncoder().cuda()
ckpts = torch.load('ptnn/psp_ffhq_encode.pt', map_location="cuda")
pSpEncoder.load_state_dict(get_keys(ckpts, 'encoder'), strict=True)
pSpEncoder.eval()
for param in pSpEncoder.parameters():
    param.requires_grad = False
del ckpts

# styleGAN
size = 1024
generator = Generator(size, 512, 8, channel_multiplier=2).cuda()
discriminator = Discriminator(size)
ckpts = torch.load("./ptnn/stylegan2-ffhq-config-f.pt", map_location="cuda")
avg = ckpts["latent_avg"].cuda()
print(avg.size())
generator.load_state_dict(ckpts["g_ema"], strict=False)
for param in generator.parameters():
    param.requires_grad = False
generator.eval()

discriminator.load_state_dict(ckpts["d"], strict=False)
for param in discriminator.parameters():
    param.requires_grad = False
discriminator.eval()

del ckpts

from PIL import Image
import cv2
img = Image.open("/home/compu/RAFS/assets/k-celeb/00000.png").convert("RGB")
img = transforms(img).unsqueeze(0).cuda()

code = pSpEncoder(img)
recon, fmaps = generator(code+avg, input_is_latent=True)
recon = recon.squeeze().detach().cpu().numpy().transpose([1,2,0])/2+0.5
cv2.imwrite("test.jpg", recon[:, :, ::-1]*255)