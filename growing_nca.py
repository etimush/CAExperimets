import random
from AdEMAMix import AdEMAMix
import cv2
import torch
import matplotlib.pyplot as plt
import numpy as np
from NCA import *
import utils
import matplotlib
matplotlib.use("TkAgg")
DEVICE = "cuda:0"
HEIGHT = 50
WIDTH = 50
CHANNELS = 16
LEARNABLE_FILTERS = 2
in_features,  hidden_dim, out_features = 4*CHANNELS, 128,CHANNELS

base = cv2.imread(f"./lizard.png", cv2.IMREAD_UNCHANGED)
base = cv2.resize(base, (int(HEIGHT), int(WIDTH)), interpolation=cv2.INTER_AREA)
base_2 = base / 255
base_2[..., :3] *= base_2[..., 3:]
base_torch = torch.tensor(base_2, dtype=torch.float32, requires_grad=True).permute((2, 0, 1)).cuda()
base_tt = base_torch.cpu().permute((1, 2, 0)).clone().detach().numpy()

batch_size = 8
x_prime = torch.zeros((CHANNELS, HEIGHT, WIDTH), dtype=torch.float32).cuda()
# x_prime[3:,WIDTH-6, int(HEIGHT / 2)] = 1
x_prime[3:, int(WIDTH / 2), int(HEIGHT / 2)] = 1.0
plt.figure(3)
plt.imshow(base_tt)

nca = CA(CHANNELS,96, 12)

nca = nca.to(DEVICE)
#optim = torch.optim.Adam(nca.parameters(), lr=1e-3)
optim = AdEMAMix(nca.parameters(), lr = 1e-3, weight_decay=0.00)
scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=2500, gamma=0.3)


pool = torch.tile(x_prime, (4000, 1, 1, 1))
loss = 0
loss_log = []
for i in range(8000 + 1):
    loss = 0
    x, idxs = utils.get_batch(pool,x_prime,batch_size)
    optim.zero_grad()
    for _ in range( random.randrange(64,92)):
        x = nca.visualize(x)
    loss = (base_torch - x[:, :4, :, :]).pow(2).sum()
    loss_log.append(loss.log().item())
    pool = utils.update_pool(pool, x.clone().detach(), idxs)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(nca.parameters(), 0.1)
    optim.step()
    scheduler.step()
    x = x.detach()
    loss = loss.detach()
    if i % 10 == 0:
        print(f"Training itter {i}, loss = {loss.item()}")

        plt.figure(1,figsize=(10, 4))

        plt.title('Loss history)')
        plt.plot(loss_log, '.', alpha=0.5, color = "b")
        utils.show_batch(x)
        plt.show(block=False)
        plt.pause(0.01)
    if i % 100 == 0:
        torch.save(nca.state_dict(), "lizard_sobel_batch.pth")
