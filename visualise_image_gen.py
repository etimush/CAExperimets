import cv2
import torch

import style_loss
from NCA import *
import utils
import numpy as np

DEVICE = "cuda:0"
HEIGHT =  500
WIDTH = 500
CHANNELS = 16
FULL_SCREEN = True
PADDING = 0
if FULL_SCREEN:
    cv2.namedWindow("optim", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("optim", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
in_features, hidden_dim, out_features = 4*CHANNELS, 128,CHANNELS

x_prime = torch.zeros((1,CHANNELS, HEIGHT, WIDTH), dtype=torch.float32).cuda()
random_locs =  np.load("constellation17.npy")
#random_locs = style_loss.generate_constellation([HEIGHT,WIDTH], 2025)
random_locs = np.random.randint(0,HEIGHT, [2,2025])
print(random_locs)
constellation = torch.rand((1,3,random_locs.shape[-1]), device=DEVICE)

x_prime[:,:,random_locs[0], random_locs[1]] = 1


#img_base = style_loss.to_nchw(img_base)
#x_prime[:,:3,:,:] = img_base


nca = IsoCA(CHANNELS, hidden_n=158)
nca.load_state_dict(torch.load("style_nca37.pth"))
#nca2 = IsoCA(CHANNELS, hidden_n=96)
#nca2.load_state_dict(torch.load("./style_nca24.pth"))
nca.to(DEVICE).eval()
#nca2.to(DEVICE).eval()
#x = x_prime

x = torch.nn.functional.pad(x_prime, [PADDING, PADDING, PADDING, PADDING],"circular")
while True:

    x = nca.visualize(x, update_rate=0.5)
    #x2 = nca2.visualize(x)
    x = x.detach()#*0.9 + x2.detach()* 0.1
    image = style_loss.to_rgb( x).clone().detach().cpu().permute(0,2,3,1).numpy()[0,:,:,:]
    style_loss.show(image,"optim", waitkey=1, resize=[1000,1000])