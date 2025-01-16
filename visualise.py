import cv2
import torch

import style_loss
from NCA import *
import utils
import numpy as np

DEVICE = "cuda:0"
HEIGHT = 50
WIDTH = 50
CHANNELS = 16
LEARNABLE_FILTERS = 2
in_features, hidden_dim, out_features = 4*CHANNELS, 128,CHANNELS
mouseX,mouseY, is_clicked = 0,0,False
def draw_circle(event,x,y,flags,param):
    global mouseX,mouseY, is_clicked
    if event == cv2.EVENT_LBUTTONDOWN:
        mouseX,mouseY = x,y
        is_clicked = True

x_prime = torch.zeros((1,CHANNELS, HEIGHT, WIDTH), dtype=torch.float32).cuda()
# x_prime[3:,WIDTH-6, int(HEIGHT / 2)] = 1
x_prime[:,3:, int(WIDTH / 2), int(HEIGHT / 2)] = 1.0
nca = CA(CHANNELS,hidden_n=96,mask_n=12)
nca.load_state_dict(torch.load("./lizard_sobel_batch.pth"))
nca.to(DEVICE).eval()
x = x_prime

while True:

    x = nca.visualize(x)
    x = x.detach()
    image = x.clone().detach().cpu().permute(0,3,2,1).numpy()[0,:,:,:3]
    image = cv2.resize(image, [500,500], cv2.INTER_CUBIC)
    if is_clicked:
        print(mouseX, mouseY)
        is_clicked = False
        mask = style_loss.create_circular_mask([HEIGHT,WIDTH], mouseY//10, mouseX//10, 10)

        x = x * ~mask[None,None,...]
    cv2.imshow("image", image)
    cv2.setMouseCallback('image', draw_circle)
    cv2.waitKey(10)
