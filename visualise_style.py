import cv2
import torch

import style_loss
from NCA import *
import utils
import numpy as np

DEVICE = "cuda:0"
HEIGHT = 1080//2
WIDTH = 1920//2
CHANNELS = 16
FULL_SCREEN = True
if FULL_SCREEN:
    cv2.namedWindow("optim", cv2.WINDOW_NORMAL)
    #cv2.setWindowProperty("optim", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
in_features, hidden_dim, out_features = 4*CHANNELS, 128,CHANNELS

x_prime = torch.rand((1,CHANNELS, HEIGHT, WIDTH), dtype=torch.float32).cuda()
img_base = style_loss.image_loader("knd2.jpg", HEIGHT)

#img_base = style_loss.to_nchw(img_base)
#x_prime[:,:3,:,:] = img_base

#nca = uSNCA(in_features, hidden_dim, out_features)
nca = CA(CHANNELS, hidden_n=96)
#nca2 = CA(CHANNELS, hidden_n=96)
nca.load_state_dict(torch.load("./style_nca106.pth"))
#nca2.load_state_dict(torch.load("./style_nca7.pth"))
nca.to(DEVICE).eval()
#nca2.to(DEVICE).eval()
x = x_prime

for i in range(9000 + 1):

    x = nca(x)


    x = x.detach()
    #x = x[:,:,2:-2,2:-2]
    #x = style_loss.resize(x, [HEIGHT,WIDTH])
    image = style_loss.to_rgb( x).clone().detach().cpu().permute(0,2,3,1).numpy()[0,:,:,:3]
    style_loss.show(image,"optim", waitkey=1, resize=[1920,1080])