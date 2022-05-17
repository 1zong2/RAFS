import torch
import torch.nn as nn
import torch.nn.functional as F
from submodel import arcface
from submodel.faceparser import BiSeNet
from submodel.stylegan2_pytorch.model import Generator, Discriminator
from submodel.psp import GradualStyleEncoder
from submodel.transformer_t2t_vit import Token_transformer, Attention_3
from torchvision import transforms

def weight_init(m):
    if isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 0.001)
        m.bias.data.zero_()
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight.data)

    if isinstance(m, nn.ConvTranspose2d):
        nn.init.xavier_normal_(m.weight.data)


def get_keys(d, name):
	if 'state_dict' in d:
		d = d['state_dict']
	d_filt = {k[len(name) + 1:]: v for k, v in d.items() if k[:len(name)] == name}
	return d_filt


class Interpolate(nn.Module):
    def __init__(self, scale_factor, mode="bilinear"):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode
        
    def forward(self, x):
        x = self.interp(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=False)
        return x

class RAFSGenerator(nn.Module):
    def __init__(self):
        super(RAFSGenerator, self).__init__()
        
        self.GAP = GAP()
        self.RAT = RAT()
        self.RAP = RAP()
        self.FMP = FMP()

        weight_init(self.GAP)
        weight_init(self.RAT)
        weight_init(self.RAP)
        weight_init(self.FMP)

        # self.transformer = Transformer(dim=12, depth=2, heads=8, dim_head=64, mlp_dim=512, dropout = 0.)
        self.transformer = Token_transformer(dim=512, in_dim=512, num_heads=8, mlp_ratio=1.)
        self.face_pool = torch.nn.AdaptiveAvgPool2d((256, 256)).eval()
        self.load_pretrained_models()

        self.blur = transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 5))
        self.kernel = torch.ones((1,1,5,5), device="cuda")

    def get_token(self, img):
        M = self.get_mask(img) # [B, 4, 64, 64]
        F = self.pSpEncoder.get_fmap(img)
        T = self.RAT(F, M) # [B, 512, 12]
        return T

    def forward(self, I_s, I_t):
        M_s = self.get_mask(I_s) # [B, 4, 64, 64]
        M_t = self.get_mask(I_t) # [B, 4, 64, 64]
        face_mask = self.get_face_mask(I_t)
        F_s = self.pSpEncoder.get_fmap(I_s) # [[B, 512, 16, 16] [B, 512, 32, 32] [B, 512, 64, 64]]
        F_t = self.pSpEncoder.get_fmap(I_t) # [[B, 512, 16, 16] [B, 512, 32, 32] [B, 512, 64, 64]]

        T_s = self.RAT(F_s, M_s) # [B, 512, 12]

        T_s_hat = T_s + self.transformer(T_s) # [B, 12, 512]
        F_l = self.RAP(F_t, M_t, T_s_hat)
        F_g = self.GAP(F_s[0]) # [[B, 512, 16, 16] [B, 512, 32, 32] [B, 512, 64, 64]]
        F_t_hat = [f_g + f_l for (f_g, f_l) in zip(F_l, F_g)] # [[B, 512, 16, 16] [B, 512, 32, 32] [B, 512, 64, 64]]
        
        latent = self.pSpEncoder.get_latent(*F_t_hat)
        I_st, fmaps = self.generator(latent+self.avg_code, input_is_latent=True)
        I_st = self.face_pool(I_st) # [B, 3, 1024, 1024] --> [B, 3, 256, 256]

        # fmaps: [B, 512, 8, 8] [B, 512, 16, 16] [B, 512, 32, 32] [B, 512, 64, 64] 
        # [B, 256, 128, 128] [B, 128, 256, 256] [B, 64, 512, 512] [B, 32, 1024, 1024]

        # M = self.FMP(fmaps[2:6]) # [B, 3, 32, 32] ~ [B, 3, 256, 256] takes only 4 fmaps

        # I_out = M * I_st + (1-M) * I_t
        I_out = face_mask * I_st + (1-face_mask) * I_t 

        return I_out, I_st, M_s, M_t
        # return I_out, I_st, M_s, M_t, M, fmaps

    def get_id(self, I):
        return self.arcface(F.interpolate(I[:, :, 32:224, 32:224], [112, 112], mode='bilinear', align_corners=True))

    def get_face_mask(self, I):
        with torch.no_grad():
            parsing = self.segmentation_net(F.interpolate(I, size=(512,512), mode='bilinear', align_corners=True)).max(1)[1]
            mask = torch.where(parsing>0, 1, 0)
            mask-= torch.where(parsing>13, 1, 0)
            mask = F.interpolate(mask.unsqueeze(1).float(), size=(256,256), mode='nearest')
            mask = torch.clamp(F.conv2d(mask, self.kernel, padding=(2, 2)), 0, 1)
            mask = self.blur(mask)
        return mask.repeat(1,3,1,1)

    def get_mask(self, I):
        parsing = self.segmentation_net(F.interpolate(I, size=(512,512), mode='bilinear', align_corners=True)).max(1)[1]
        parsing = parsing.unsqueeze(1)
        brows = torch.where(parsing==2, 1, 0) + torch.where(parsing==3, 1, 0)
        eyes = torch.where(parsing==4, 1, 0) + torch.where(parsing==5, 1, 0)
        nose = torch.where(parsing==10, 1, 0)
        lips = torch.where(parsing==12, 1, 0) + torch.where(parsing==13, 1, 0)
        mask = torch.cat([brows, eyes, nose, lips], dim=1).float()
        mask = F.interpolate(mask, size=(64, 64), mode='nearest')
        return mask
    
    def load_pretrained_models(self):

        # psp
        self.pSpEncoder = GradualStyleEncoder()
        ckpts = torch.load('ptnn/psp_ffhq_encode.pt', map_location="cuda")
        self.avg_code = ckpts["latent_avg"]
        self.pSpEncoder.load_state_dict(get_keys(ckpts, 'encoder'), strict=True)
        self.pSpEncoder.eval()
        for param in self.pSpEncoder.parameters():
            param.requires_grad = False
        del ckpts

        # styleGAN
        self.size = 1024
        self.generator = Generator(self.size, 512, 8, channel_multiplier=2)
        self.discriminator = Discriminator(self.size)
        ckpts = torch.load("./ptnn/stylegan2-ffhq-config-f.pt", map_location="cuda")

        self.generator.load_state_dict(ckpts["g_ema"], strict=False)
        for param in self.generator.parameters():
            param.requires_grad = False
        self.generator.eval()

        # self.discriminator.load_state_dict(ckpts["d"], strict=False)
        # for param in self.discriminator.parameters():
        #     param.requires_grad = False
        # self.discriminator.eval()

        del ckpts

        # face parser
        self.segmentation_net = BiSeNet(n_classes=19)
        ckpts = torch.load('ptnn/faceparser.pth', map_location="cuda")
        self.segmentation_net.load_state_dict(ckpts)
        self.segmentation_net.eval()
        for param in self.segmentation_net.parameters():
            param.requires_grad = False
        del ckpts
        
        # face recognition model: arcface
        self.arcface = arcface.Backbone(50, 0.6, 'ir_se')
        ckpts = torch.load('ptnn/arcface.pth', map_location="cuda")
        self.arcface.load_state_dict(ckpts, strict=False)
        self.arcface.eval()
        for param in self.arcface.parameters():
            param.requires_grad = False
        del ckpts


class GAP(nn.Module):
    def __init__(self):
        super(GAP, self).__init__()
        
        MLP = []
        MLP.append(nn.Linear(512, 1024))
        MLP.append(nn.LeakyReLU(0.2))
        MLP.append(nn.Linear(1024, 1024))
        MLP.append(nn.LeakyReLU(0.2))
        MLP.append(nn.Linear(1024, 1024))
        MLP.append(nn.LeakyReLU(0.2))

        self.MLP = nn.Sequential(*MLP)
    
    def forward(self, F_s0):
        code = F.adaptive_avg_pool2d(F_s0, (1, 1)).squeeze(2).squeeze(2) # [B, 512]
        F_l1 = self.MLP(code).reshape(-1, 1, 32, 32).repeat(1, 512, 1, 1) # [B, 512, 16, 16]
        F_l0 = F.interpolate(F_l1, scale_factor=0.5) # [B, 512, 32, 32]
        F_l2 = F.interpolate(F_l1, scale_factor=2) # [B, 512, 64, 64]
        
        return F_l0, F_l1, F_l2

class RAT(nn.Module):
    def __init__(self):
        super(RAT, self).__init__()

        MLPs = []
        for _ in range(12):
            MLPs.append(nn.Sequential(
                nn.Linear(512, 512),
                nn.LeakyReLU(0.2),
                nn.Linear(512, 512),
                nn.LeakyReLU(0.2),
                nn.Linear(512, 512),
                nn.LeakyReLU(0.2),
            ))
        self.MLPs = nn.ModuleList(MLPs)

    def forward(self, F_s, M_s):
        T_s_locals = []
        mlp_idx = 0
        for fmap in F_s:
            for local_index in range(4):
                masked_local = F.interpolate(fmap, size=(64, 64)) * M_s[:, local_index, :, :].unsqueeze(1) # [B, 512, 64, 64]
                code = F.adaptive_avg_pool2d(masked_local, (1, 1)) # [B, 512, 1, 1]
                code = self.MLPs[mlp_idx](code.squeeze(2).squeeze(2)) # [B, 512]
                T_s_locals.append(code.unsqueeze(1)) # [B, 1, 512]
                # code = self.MLPs[mlp_idx](code.squeeze(2).squeeze(2)) # [B, 512]
                # T_s_locals.append(code.unsqueeze(2)) # [B, 512, 1]
                mlp_idx += 1
        T_s = torch.cat(T_s_locals, dim=1) # [B, 12, 512]
        return T_s

class RAP(nn.Module):
    def __init__(self):
        super(RAP, self).__init__()
        
        self.attentions = nn.ModuleList([Attention_3(q_dim=512, k_dim=512, v_dim=512, in_dim=512) for _ in range(3)])
        self.norm_q = nn.LayerNorm(512)
        self.norm_k = nn.LayerNorm(512)
        self.norm_v = nn.LayerNorm(512)

    def forward(self, F_t, M_t, T_s_hat):
        F_l = []
        size = [16, 32, 64]
        M_t = M_t.sum(dim=1, keepdim=True) # [B, 1, 64, 64]
        for i in range(3):
            fmap = F.interpolate(F_t[i], size=(64,64))
            masked_fmap = fmap * M_t  # [B, 512, 64, 64]
            masked_fmap_flatten = torch.flatten(masked_fmap, start_dim=2).permute(0, 2, 1) # [B, 512, 4096] --> [B, 4096, 512]
            # attn = self.attentions[i](masked_fmap_flatten, T_s_hat, T_s_hat)  # [B, 512, 4096], [B, 12, 512],  [B, 12, 512]
            updated_fmap = self.attentions[i](self.norm_q(masked_fmap_flatten), self.norm_k(T_s_hat), self.norm_v(T_s_hat))  # [B, 4096, 512], [B, 12, 512], [B, 12, 512]
            # updated_fmap size: [B, 512, 4096]
            updated_fmap = updated_fmap.reshape(-1, 512, 64, 64)
            # final_fmap = F.interpolate((M_t*updated_fmap + (1-M_t)*fmap), (size[i], size[i]))
            final_fmap = F.interpolate((updated_fmap + fmap), (size[i], size[i]))
            F_l.append(final_fmap)

        return F_l # [[B, 512, 16, 16], [B, 512, 32, 32], [B, 512, 64, 64]]

class FMP(nn.Module):
    def __init__(self):
        super(FMP, self).__init__()

        fmap_dims = [512, 512, 256, 128]
        self.conv1x1 = nn.Conv2d(512, 1, kernel_size=1, stride=1, padding=0)
        bottlenecks = []
        for i in range(4):
            bottlenecks.append(nn.Sequential(
                nn.Conv2d(fmap_dims[i], 128, kernel_size=1, stride=1, padding=0),
                nn.LeakyReLU(0.2)
            ))
        self.bottlenecks = nn.ModuleList(bottlenecks)

    def forward(self, fmaps):
        inters = []
        for i in range(4):
            inters.append(F.interpolate(self.bottlenecks[i](fmaps[i]), (256,256)))
        M = torch.sigmoid(self.conv1x1(torch.cat(inters, dim=1)))
        return M

