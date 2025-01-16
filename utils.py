import matplotlib.pyplot as plt
import numpy as np
def show_batch(results,channels = 4):
  x = results.cpu().clone().permute((0,2,3,1)).detach().numpy()
  plt.figure(2)
  plt.clf()
  for i in range(results.shape[0]):
    img = x[i,:, :, 0:channels]
    plt.figure(2)
    plt.subplot(2, 4, i+1)
    plt.imshow(img)

def get_batch(pool,x_prime,batch_size):
    idxs = np.random.randint(0,pool.shape[0],batch_size)
    batch = pool[idxs,:,:,:]
    batch[0:1,:,:,:] = x_prime
    return batch, idxs

def update_pool(pool,results, idxs):
    pool[idxs] = results.clone().detach()
    return pool