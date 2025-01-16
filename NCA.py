import torch
import torchvision
class NCA(torch.nn.Module):

    def __init__(self,in_features, hidden_dim, out_features):
        super(NCA, self).__init__()
        self.conv_sense = torch.nn.Conv2d(out_features, out_features*3, kernel_size=3, padding="same")
        self.conv1 = torch.nn.Conv2d(in_features,hidden_dim,1,padding="same")
        self.relu = torch.nn.ReLU()
        self.conv3 = torch.nn.Conv2d(hidden_dim, hidden_dim, 1, padding="same")
        self.relu1 = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(hidden_dim,out_features,1,padding="same")
        self.grid = None
        self.out_features = out_features


    def forward(self, x):

        x = self.conv1(x)
        x = self.relu(x)
        x =self.conv3(x)
        x = self.relu1(x)
        x = self.conv2(x)
        return x

    def __call__(self,x, num):

        for _ in range(num):
            x = self.sense(x)
            x = self.forward(x)
            x = self.update(x)
        return x
    def visualize(self,x):
        x = self.sense(x)
        x = self.forward(x)
        x = self.update(x)
        return x

    def sense(self,x, rotation = 0):

        self.grid = x
        """k1 = torchvision.transforms.functional.rotate(torch.tensor([[1, 2, 1],
                                                                    [2, -12, 2],
                                                                    [+1, 2, +1]], dtype=torch.float32).tile(
            (self.out_features, 1, 1, 1)) / 8, rotation).cuda()
        k2 = torchvision.transforms.functional.rotate(torch.tensor([[-1, -1, -1],
                            [0, 0, 0],
                            [+1, 1, +1]],dtype=torch.float32).tile((self.out_features,1,1,1))/8, rotation).cuda()
        k3 = torchvision.transforms.functional.rotate(torch.tensor([[-1, 0, +1],
                            [-1, 0, +1],
                            [-1, 0, +1]],dtype=torch.float32).tile((self.out_features,1,1,1))/8, rotation).cuda()
        sense1 = torch.nn.functional.conv2d(x, k1, groups=self.out_features, stride=1, padding="same")
        sense2 = torch.nn.functional.conv2d(x, k2, groups=self.out_features, stride=1, padding="same")
        sense3 = torch.nn.functional.conv2d(x, k3, groups=self.out_features, stride=1, padding="same")
        sense_x = torch.cat((x,sense1,sense2,sense3), 1)"""
        sense = self.conv_sense(x)
        sense_x = torch.cat((x,sense), dim=1)

        return sense_x

    def update(self,update):
        pre_life_mask = torch.nn.functional.max_pool2d(self.grid[:,None,3,:,:],3,1,1).cuda() >0.1
        async_update = torch.rand((self.out_features,self.grid.shape[2], self.grid.shape[3])).cuda() > 0.5
        grid_update = self.grid + (update*async_update)
        post_life_mask = torch.nn.functional.max_pool2d(self.grid[:,None,3,:,:],3,1,1).cuda() >0.1
        self.grid = grid_update * post_life_mask * pre_life_mask
        return self.grid


class uNCA(torch.nn.Module):

    def __init__(self,in_features, hidden_dim, out_features):
        super(uNCA, self).__init__()
        """self.k1 = torchvision.transforms.functional.rotate(torch.tensor([[1, 2, 1],
                                                                    [2, -12, 2],
                                                                    [+1, 2, +1]], dtype=torch.float32).tile(
            (out_features, 1, 1, 1)) / 8,0).cuda()
        self.k2 = torchvision.transforms.functional.rotate(torch.tensor([[-1, -1, -1],
                                                                    [0, 0, 0],
                                                                    [+1, 1, +1]], dtype=torch.float32).tile(
            (out_features, 1, 1, 1)) / 8, 0).cuda()
        self.k3 = torchvision.transforms.functional.rotate(torch.tensor([[-1, 0, +1],
                                                                    [-1, 0, +1],
                                                                    [-1, 0, +1]], dtype=torch.float32).tile(
            (out_features, 1, 1, 1)) / 8,0).cuda()"""

        self.k1 = torch.nn.Conv2d(out_features, out_features, kernel_size=3, stride=1, groups=out_features, padding_mode="circular", padding="same")
        self.k2 = torch.nn.Conv2d(out_features, out_features, kernel_size=3, stride=1, groups=out_features,
                                  padding_mode="circular", padding="same")
        self.k3 = torch.nn.Conv2d(out_features, out_features, kernel_size=3, stride=1, groups=out_features,
                                  padding_mode="circular", padding="same")
        self.k1.weight.data = make_kernel([[1, 2, 1], [2, -12, 2],[+1, 2, +1]], out_features)
        self.k1.requires_grad_(False)
        self.k2.weight.data = make_kernel([[-1, 0, 1], [-2, 0, 2], [-1, 0, +1]], out_features)
        self.k2.requires_grad_(False)

        self.out_features = out_features
        self.k3.weight.data = make_kernel([[-1, -2, -1], [0, 0, 0], [+1, 2, +1]], out_features)
        self.k3.requires_grad_(False)
        self.conv2 = torch.nn.Conv2d(8*out_features, out_features, kernel_size=1, padding="same", padding_mode="circular")
        self.grid = None

    def forward(self, x):
        sense1 = self.k1(x)
        sense2 = self.k2(x)
        sense3 = self.k3(x)
        p = torch.cat((x,sense1,sense2,sense3), dim =1)
        y = torch.cat((p,p.abs()), dim=1)
        y =  self.conv2(y)
        x = self.update(y)
        return x

    def __call__(self,x, num):

        for _ in range(num):
            self.grid = x
            x = self.forward(x)
            torch.nn.utils.clip_grad_norm_(x, 0.1)


        return x


    def visualize(self,x):
        self.grid = x
        x = self.forward(x)

        return x

    def update(self,update):
        pre_life_mask = torch.nn.functional.max_pool2d(self.grid[:,None,3,:,:],3,1,1).cuda() >0.1
        async_update = torch.rand((self.out_features,self.grid.shape[2], self.grid.shape[3])).cuda() > 0.5
        grid_update = self.grid + (update*async_update)
        post_life_mask = torch.nn.functional.max_pool2d(self.grid[:,None,3,:,:],3,1,1).cuda() >0.1
        self.grid = grid_update * post_life_mask * pre_life_mask
        return self.grid

class uSNCA(torch.nn.Module):

    def __init__(self,in_features, hidden_dim, out_features ):
        super(uSNCA, self).__init__()


        self.k1 = torch.nn.Conv2d(out_features, out_features, kernel_size=3, stride=1, groups=out_features, padding="same", padding_mode="circular", bias=False)
        print(f"k1 shape {self.k1.weight.shape}")
        self.k2 = torch.nn.Conv2d(out_features, out_features, kernel_size=3, stride=1, groups=out_features,
                                   padding="same", padding_mode="circular", bias=False)
        self.k3 = torch.nn.Conv2d(out_features, out_features, kernel_size=3, stride=1, groups=out_features, padding="same", padding_mode="circular",bias=False)
        self.k1.weight.data[:,:] = torch.tensor([[1, 2, 1], [2, -12, 2],[+1, 2, +1]])
        #self.k1.requires_grad_(False)
        self.k2.weight.data[:,:] = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, +1]])
        #self.k2.requires_grad_(False)

        self.out_features = out_features
        self.k3.weight.data[:,:] = torch.tensor([[-1, -2, -1], [0, 0, 0], [+1, 2, +1]])
        #self.k3.requires_grad_(False)
        self.conv2 = torch.nn.Conv2d((4*out_features)*2, out_features, kernel_size=1, padding="same", padding_mode="circular", groups=1)
        self.grid = None


    def forward(self, x):

        sense1 = self.k1(x)
        sense2 = self.k2(x)
        sense3 = self.k3(x)
        p = torch.cat((x,sense1,sense2,sense3), dim =1)
        y = torch.cat((p,p.abs()), dim=1)
        x = x + self.conv2(y)

        return x

    def __call__(self,x, num):

        for _ in range(num):
            self.grid = x
            x = self.forward(x)
            torch.nn.utils.clip_grad_norm_(x, 0.1)


        return x


    def visualize(self,x):
        self.grid = x
        x = self.forward(x)

        return x



def make_kernel(kernel, channels):
   k = torch.tensor(kernel, dtype=torch.float32).tile((channels, 1, 1, 1)).cuda() / 8
   return k


def perchannel_conv(x, filters):
  '''filters: [filter_n, h, w]'''
  b, ch, h, w = x.shape
  y = x.reshape(b*ch, 1, h, w)
  y = torch.nn.functional.pad(y, [1, 1, 1, 1], 'circular')
  y = torch.nn.functional.conv2d(y, filters[:,None])
  return y.reshape(b, -1, h, w)


ident = torch.tensor([[0.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,0.0]], dtype=torch.float32,device="cuda:0")
sobel_x = torch.tensor([[-1.0,0.0,1.0],[-2.0,0.0,2.0],[-1.0,0.0,1.0]], dtype=torch.float32,device="cuda:0")
lap = torch.tensor([[1.0,2.0,1.0],[2.0,-12,2.0],[1.0,2.0,1.0]], dtype=torch.float32,device="cuda:0")
gaus =torch.tensor([[1.0,2.0,1.0],[2.0,4.0,2.0],[1.0,2.0,1.0]], dtype=torch.float32,device="cuda:0")

def perception(x, mask_n = 0):
  filters = torch.stack([sobel_x, sobel_x.T, lap])
  n = x.shape[1]
  padd = torch.zeros((x.shape[0], 3*mask_n, x.shape[2], x.shape[3]),device="cuda:0")

  obs =perchannel_conv(x[:,0:n-mask_n], filters)

  return torch.cat((x,obs, padd), dim=1)

class CA(torch.nn.Module):
  def __init__(self, chn=12, hidden_n=96, mask_n = 0):
    super().__init__()
    self.chn = chn
    self.w1 = torch.nn.Conv2d(4*chn, hidden_n, 1)
    self.w2 = torch.nn.Conv2d(hidden_n, chn, 1, bias=False)
    self.w2.weight.data.zero_()
    self.mask_n = mask_n

  def forward(self, x, update_rate=0.5):
      y = perception(x, self.mask_n)
      y = self.w2(torch.relu(self.w1(y)))
      b, c, h, w = y.shape
      update_mask = (torch.rand(b, 1, h, w, device="cuda:0") + update_rate).floor()
      pre_life_mask = torch.nn.functional.max_pool2d(x[:, None, 3, ...], 3, 1, 1).cuda() > 0.1

      # Perform update
      x = x + y * update_mask * pre_life_mask
      return x

  def visualize(self,x, update_rate=0.5):
      y = perception(x, self.mask_n)
      y = self.w2(torch.relu(self.w1(y)))
      b, c, h, w = y.shape
      update_mask = (torch.rand(b, 1, h, w, device="cuda:0") + update_rate).floor()
      pre_life_mask = torch.nn.functional.max_pool2d(x[:, None, 3, ...], 3, 1, 1).cuda() > 0.1

      # Perform update
      x = x + y * update_mask * pre_life_mask
      return x






def gradnorm_perception(x):
  grad = perchannel_conv(x, torch.stack([sobel_x, sobel_x.T]))
  gx, gy = grad[:, ::2], grad[:, 1::2]
  state_lap = perchannel_conv(x, torch.stack([ident, lap]))
  return torch.cat([ state_lap, (gx*gx+gy*gy+1e-8).sqrt()], 1)






class IsoCA(torch.nn.Module):
  def __init__(self, chn=12, hidden_n=128):
    super().__init__()
    self.chn = chn

    # Determine the number of perceived channels
    perc_n = gradnorm_perception(torch.zeros([1, chn, 8, 8], device="cuda:0")).shape[1]

    # Approximately equalize the parameter count between model variants


    # Model layers
    self.w1 = torch.nn.Conv2d(perc_n, hidden_n, 1)
    self.w2 = torch.nn.Conv2d(hidden_n, chn, 1, bias=False)
    self.w2.weight.data.zero_()



  def visualize(self, x, update_rate=0.5):
    y = gradnorm_perception(x)
    y = self.w1(y)
    #y = torch.cat([y,-y], dim)
    y = self.w2(torch.nn.functional.leaky_relu(y))
    b, c, h, w = y.shape
    update_mask = (torch.rand(b, 1, h, w, device="cuda:0") + update_rate).floor()
    #pre_life_mask = torch.nn.functional.max_pool2d(x[:,None,3,...], 3, 1, 1).cuda() > 0.1


      # Perform update
    x = x + y * update_mask #* pre_life_mask

    return  x



class IsoGrowCA(torch.nn.Module):
  def __init__(self, chn=12, hidden_n=128):
    super().__init__()
    self.chn = chn

    # Determine the number of perceived channels
    perc_n = gradnorm_perception(torch.zeros([1, chn, 8, 8], device="cuda:0")).shape[1]

    # Approximately equalize the parameter count between model variants


    # Model layers
    self.w1 = torch.nn.Conv2d(perc_n, hidden_n, 1)
    self.w2 = torch.nn.Conv2d(hidden_n, chn, 1, bias=False)
    self.w2.weight.data.zero_()



  def visualize(self, x, update_rate=0.5):
    y = gradnorm_perception(x)
    y = self.w1(y)
    #y = torch.cat([y,-y], dim)
    y = self.w2(torch.nn.functional.leaky_relu(y))
    b, c, h, w = y.shape
    update_mask = (torch.rand(b, 1, h, w, device="cuda:0") + update_rate).floor()
    pre_life_mask = torch.nn.functional.max_pool2d(x[:,None,3,...], 3, 1, 1).cuda() > 0.1


      # Perform update
    x = x + y * update_mask * pre_life_mask

    return  x