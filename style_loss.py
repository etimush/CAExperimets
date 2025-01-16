import cv2
import PIL
import PIL.Image
import torchvision.transforms
from torchvision import transforms
import numpy as np
import torch
def image_loader(path, size):
    img = PIL.Image.open(path)

    transform = transforms.Compose([transforms.PILToTensor()])
    img = transform(img)

    res = torchvision.transforms.Resize(size, interpolation=torchvision.transforms.InterpolationMode.BICUBIC)
    img = res(img)
    img = torchvision.transforms.functional.adjust_sharpness(img, 1)
    img = img.permute(1,2,0)
    img = np.float32(img)/255
    return img

def to_nchw(img):
  img = torch.as_tensor(img,device="cuda:0")
  if len(img.shape) == 3:
    img = img[None,...]
  return img.permute(0, 3, 1, 2)
# show image
def show(img, title, resize = [500, 500], waitkey = 1):
    # Convert from BGR to RGB

    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # imshow() only accepts float [0,1] or int [0,255]
    img = np.array(img).clip(0, 1)


    open_cv_image = img[:, :, ::-1].copy()
    open_cv_image = cv2.resize(open_cv_image, resize, cv2.INTER_CUBIC)
    cv2.imshow(title, open_cv_image)
    cv2.waitKey(waitkey)

mse_loss = torch.nn.MSELoss()
def to_rgb(x):
  return x[:,:3,:,:]

def color_loss(g,h):
    loss = 0

    for i in range(g.shape[1]):
        gstd = torch.std(g[0,i,...])
        hstd = torch.std(h[0,i,...])
        loss += mse_loss(hstd, gstd)
        gmean = torch.mean(g[0,i,...])
        hmean = torch.mean(h[0,i,...])
        loss += mse_loss(hmean, gmean)
    loss += mse_loss(torch.std(h.sum(dim=1)), torch.std(g.sum(dim=1)))
    return loss

def extra_features(I:torch.Tensor, n_levels, k):
    F = []
    Ii = I
    GB = torchvision.transforms.GaussianBlur(5)
    for _ in range(n_levels):
        Isharp = Ii + 2*(Ii-GB.forward(Ii))
        Fl = Isharp.unfold(1,3,1).unfold(2,k,1).unfold(3,k,1)

        F.append(Fl)
        transform = transforms.Resize(size=Ii.shape[-1]//2)
        Ii = transform(Ii)
    return F


vgg16 = torchvision.models.vgg16(weights='IMAGENET1K_V1').features
vgg16.to("cuda:0")

def calc_styles_vgg(imgs):
  style_layers = [1, 6, 11, 18, 25]
  mean = torch.tensor([0.485, 0.456, 0.406], device="cuda:0")[:,None,None]
  std = torch.tensor([0.229, 0.224, 0.225], device="cuda:0")[:,None,None]
  x = (imgs-mean) / std
  b, c, h, w = x.shape
  features = [x.reshape(b, c, h*w)]
  for i, layer in enumerate(vgg16[:max(style_layers)+1]):
    x = layer(x)
    if i in style_layers:
      b, c, h, w = x.shape
      features.append(x.reshape(b, c, h*w))
  return features
def project_sort(x, proj):
  return torch.einsum('bcn,cp->bpn', x, proj).sort()[0]
def ot_loss(source, target, proj_n=32):
  ch, n = source.shape[-2:]
  projs = torch.nn.functional.normalize(torch.randn(ch, proj_n, device="cuda:0"), dim=0)
  source_proj = project_sort(source, projs)
  target_proj = project_sort(target, projs)
  target_interp = torch.nn.functional.interpolate(target_proj, n, mode='nearest')
  return (source_proj-target_interp).square().sum()

def create_vgg_loss(target_img):
  yy = calc_styles_vgg(target_img)
  def loss_f(imgs):
    xx = calc_styles_vgg(imgs)
    return sum(ot_loss(x, y) for x, y in zip(xx, yy))
  return loss_f

def resize(T:torch.Tensor, size: []) -> torch.Tensor:
    res = torchvision.transforms.Resize(size, interpolation=torchvision.transforms.InterpolationMode.BICUBIC, antialias=True)
    return res.forward(T)




def generate_constellation(size : list, num_points : int) -> np.array:
    x_interval = size[0]//int(np.sqrt(num_points))
    y_interval = size[1]//int(np.sqrt(num_points))
    x = np.linspace(0,size[0]-x_interval,int(np.sqrt(num_points)), dtype=int)
    y = np.linspace(0, size[1]-y_interval, int(np.sqrt(num_points)), dtype=int)
    xx,yy = np.meshgrid(x,y)
    xx+= x_interval//2
    yy += y_interval//2

    x_jigle  = np.random.randint(-x_interval//5, x_interval//2, size=xx.shape)
    y_jigle = np.random.randint(-y_interval // 2, y_interval // 2, size=yy.shape)
    xx += x_jigle
    yy += y_jigle

    return np.stack([xx.flat,yy.flat])


def get_patches(img: torch.Tensor, k, step = 20) -> torch.Tensor:
    patches = img.unfold(1, img.shape[1], 1).unfold(2, k, step).unfold(3, k, step)
    patches = patches.squeeze()
    return patches.reshape((-1, patches.shape[1]* patches.shape[2],patches.shape[3],patches.shape[4],patches.shape[5] ))


def create_circular_mask(image_size, center_x, center_y, radius):
    height, width = image_size
    y_grid, x_grid = torch.meshgrid(torch.arange(height, device="cuda:0"), torch.arange(width, device="cuda:0"), indexing='ij')
    distance_squared = (x_grid - center_x) ** 2 + (y_grid - center_y) ** 2
    mask = distance_squared <= radius ** 2
    return mask