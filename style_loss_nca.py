import copy
import random
from NCA import *
import style_loss
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import cv2
import numpy as np
import utils
import torch
from torchvision.models import vgg19, VGG19_Weights
from AdEMAMix import AdEMAMix
import matplotlib
matplotlib.use("TkAgg")
DEVICE = "cuda:0"
HEIGHT = 300
WIDTH = 300
CHANNELS = 16

"""Model and base style initialisation"""


in_features,  hidden_dim, out_features = 4*CHANNELS, 128,CHANNELS
img_base = style_loss.image_loader("knd2.jpg", HEIGHT)

img_base = style_loss.to_nchw(img_base)
style_loss.show(img_base.clone().detach().cpu().permute(0,2,3,1).numpy()[0,:,:,:3], "base")

nca = CA(CHANNELS, hidden_n=96)
nca = nca.to(DEVICE)
params = sum([np.prod(p.size()) for p in nca.parameters()])
print(f"Num params: {params}")
#optim = AdEMAMix(nca.parameters(), lr = 1e-3, weight_decay=0.00)
optim = torch.optim.AdamW(filter(lambda p: p.requires_grad, nca.parameters()), 1e-3, capturable=False)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=[4000,8000], gamma=0.3)

num_points = 60
batch_size = 5
#x_prime = torch.rand((1,CHANNELS, HEIGHT, WIDTH), dtype=torch.float32).cuda()



pool = torch.rand((64,CHANNELS, HEIGHT, WIDTH), dtype=torch.float32).cuda()



#x_prime[:,:3,random_locs[0], random_locs[1]] = 1

save_memory = True
loss_log = []
with torch.no_grad():
    vgg_loss = style_loss.create_vgg_loss(img_base)

for i in range(8000 + 1):
    #optim.zero_grad()poop
    with torch.no_grad():

        idx = np.random.choice(len(pool),batch_size,replace=False)
        x = pool[idx]
        if i %8 == 0:
            x[:1] = torch.rand((1,CHANNELS,HEIGHT,WIDTH))


    #with torch.autocast(device_type="cuda", dtype=torch.float16):

    rand_n = random.randint(32,64)
    if not save_memory:
        for j in range(rand_n):
            x = nca.visualize(x)
    else:
        x.requires_grad = True  # https://github.com/pytorch/pytorch/issues/42812
        x = torch.utils.checkpoint.checkpoint_sequential([nca] * rand_n, 16, x, use_reentrant = False)
    s_loss = vgg_loss(style_loss.to_rgb(x))
    total_loss = s_loss #+ (x - x.clip(-2.0,2.0)).abs().sum()



    with torch.no_grad():
        total_loss.backward()
        for p in nca.parameters():
            p.grad /= (p.grad.norm() + 1e-8)  # normalize gradients
        #torch.nn.utils.clip_grad_norm_(nca.parameters(), 0.1)
        optim.step()

        #scheduler.step()
        pool[idx] = x.clone().detach()
        loss_log.append(total_loss.log().item())
        optim.zero_grad()

    if i % 10 == 0:
        print(f"Loss for step {i}: {total_loss.item()}")
        style_loss.show(style_loss.to_rgb(x).clone().detach().cpu().permute(0,2,3,1).numpy()[0,:,:,:3], "optim")
        plt.plot(loss_log, '.', alpha=0.5, color="b")

        plt.show(block=False)
        plt.pause(0.01)
    if i % 100 == 0:
        torch.save(nca.state_dict(), "style_nca106.pth")

