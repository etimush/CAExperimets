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
import matplotlib
from AdEMAMix import AdEMAMix
matplotlib.use("TkAgg")
DEVICE = "cuda:0"
HEIGHT = 140
WIDTH = 140
CHANNELS = 16
PADDING = 0
INFER_DIM = True
"""Model and base style initialisation"""


if INFER_DIM:
    img_base = style_loss.image_loader("knd7.jpg", [113])
    print(img_base.shape)
    HEIGHT,WIDTH = img_base.shape[-3], img_base.shape[-2]
    print(HEIGHT, WIDTH)
else:
    img_base = style_loss.image_loader("knd7.jpg", [HEIGHT, WIDTH])

style_loss.show(img_base,"any", waitkey=1,resize=[WIDTH*4,HEIGHT*4])

img_base = style_loss.to_nchw(img_base)
#8.65
#style_loss.show(img_base.clone().detach().cpu().permute(0,2,3,1).numpy()[0,:,:,:3], "base")


nca = IsoCA(CHANNELS, hidden_n=158)
#nca.load_state_dict(torch.load("./style_nca36.pth"))
nca = nca.to(DEVICE)
params = sum([np.prod(p.size()) for p in nca.parameters()])
print(f"Num params: {params}")

#optim = torch.optim.AdamW(nca.parameters(), 2e-4, capturable=False)
optim = AdEMAMix(nca.parameters(), lr = 1e-4, weight_decay=0.00)
    #scheduler = torch.optim.lr_scheduler.CyclicLR(optim, 1e-5, 1e-3, step_size_up=125, mode='triangular2')
scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=[20000,40000,100000], gamma=0.3)
batch_size = 4
num_points = 81
rand_p = style_loss.generate_constellation([HEIGHT,WIDTH], num_points)
#rand_p = np.random.randint(0,WIDTH, [2,num_points])
#rand_p = np.load("./constellation15.npy")
img_base = img_base.tile((batch_size,1,1,1))
with torch.no_grad():
    #patches_base = style_loss.get_patches(img_base.to("cpu"),100, step=50)

    x_prime = torch.zeros((batch_size,CHANNELS, HEIGHT, WIDTH), dtype=torch.float32).cuda()
    x_prime[:, :, rand_p[0], rand_p[1]] = 1

    #patches_x_prime = style_loss.get_patches(x_prime.to("cpu"),100, step=50)
    #print(patches_x_prime.shape)

np.save("constellation19", rand_p)
#with torch.no_grad():
    #s_loss = style_loss.create_vgg_loss(img_base)




loss_log = []
def L2(T1, T2): return (T1 - T2).pow(2).sum()
#L2 = torch.nn.MSELoss()
LAP = torch.tensor([[1., 2., 1.], [2., -12., 2.], [1., 2., 1.]], device=DEVICE)
SOBEL = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], device=DEVICE)
def perchannel_conv(x, filters):
  b, ch, h, w = x.shape
  y = x.reshape(b * ch, 1, h, w)
  y = torch.nn.functional.pad(y, (1, 1, 1, 1), 'circular')
  y = torch.nn.functional.conv2d(y, filters[:, None])
  return y.reshape(b, -1, h, w)
def laplacian(x):
  state_lap = perchannel_conv(x, LAP[None, :])
  return  state_lap

def gradnorm(x):
  grad = perchannel_conv(x, torch.stack([SOBEL, SOBEL.T]))
  gx, gy = grad[:, ::2], grad[:, 1::2]

  return (gx*gx+gy*gy+1e-8).sqrt()


for i in range(2000000 + 1):

    with torch.no_grad():
        rand_shift = [random.randint(0,dim) for dim in [WIDTH,HEIGHT]]
        x = torch.nn.functional.pad(x_prime, [PADDING, PADDING, PADDING, PADDING])
        #x = x.roll(rand_shift, (-1,-2))
        #rand_patch = np.random.choice(patches_base.shape[1], size=batch_size, replace=False)
        #rand_patch -= 1
        #patch_base = patches_base[0,rand_patch,...].cuda()
        #x = patches_x_prime[0, rand_patch,...].cuda()



    for j in range(513):
        x = nca.visualize(x, update_rate=0.5)

        if (j % 64 == 0) and (j != 0) :
            #x_up = x.roll((-rand_shift[0], -rand_shift[1]), (-1,-2))
            x_up = x[:,:,PADDING:x.shape[-2]-PADDING,PADDING:x.shape[-1]-PADDING]
            total_loss = L2(img_base, style_loss.to_rgb(x_up)) + 0.1*(L2(laplacian(img_base),laplacian(style_loss.to_rgb(x_up))) + L2(gradnorm(img_base), gradnorm(style_loss.to_rgb(x_up)))) #+ 0.01* s_loss(style_loss.to_rgb(x_up))


            with torch.no_grad():
                total_loss.backward()
                for p in nca.parameters():
                    p.grad /= (p.grad.norm() + 1e-8)
                optim.step()
                if total_loss.log() < 12:
                    loss_log.append(total_loss.log().item())
                x = x.detach()
                optim.zero_grad()
    scheduler.step()


    if i % 100 == 0:
        print(f"Loss for step {i}: {loss_log[-1]}")
        #style_loss.show(patch_base.clone().detach().cpu().permute(0, 2, 3, 1).numpy()[-1, :, :, :3], "patch",
                        #resize=[WIDTH * 2, HEIGHT * 2])
        style_loss.show(style_loss.to_rgb(x).clone().detach().cpu().permute(0,2,3,1).numpy()[-1,:,:,:3], "optim", resize=[WIDTH*4, HEIGHT*4])
        plt.clf()
        plt.plot(loss_log, '.', alpha=0.5, color="b")

        plt.show(block=False)
        plt.pause(0.01)

        torch.save(nca.state_dict(), "style_nca20.pth")
