import torch
from torch import nn
import torch.nn.functional as F


class ZApproximator:
    """Approximate (z, classes) of an image."""
    def __init__(self, biggan, vgg=None):
        super().__init__()
        self.biggan = biggan
        self.vgg = vgg
        self.vgg_mean = torch.Tensor([0.485, 0.456, 0.406])[..., None, None][None,...].to(self.device)
        self.vgg_std = torch.Tensor([0.229, 0.224, 0.225])[..., None, None][None,...].to(self.device)

    def __call__(self, imgs, z=None, classes=None, iterations=3, lr=0.1,
                truncation=0.01, learn_classes=False):
        batch_size = imgs.shape[0]
        if z is None:
            z = self.initial_z(batch_size)
        if classes is None:
            if self.vgg is not None:
                classes = self.vgg(self.preprocess_vgg(imgs)).detach()
                classes.requires_grad_(True)
            else:
                classes = self.initial_classes(batch_size)

        params = [z]
        if learn_classes:
            params.append(classes)
        optimizer = torch.optim.Adam(params, lr=lr)
        mse_loss = nn.MSELoss()
        z0 = z.clone()
        c0 = classes.clone()
        for i in range(iterations):
            optimizer.zero_grad()
            p_imgs = self.biggan(z, F.softmax(classes, dim=-1), truncation)
            p_imgs = (p_imgs + 1) / 2
            img_err = mse_loss(imgs, p_imgs)
            #img_err = (imgs - p_imgs).pow(2).sum((1,2,3)).mean()
            img_err.backward()
            optimizer.step()
        zd = (z0 - z.to(z0.device))
        print('z', *map(float, [zd.min(), zd.mean(), zd.max()]))
        cd = (c0 - classes.to(c0.device))
        print('c', *map(float, [cd.min(), cd.mean(), cd.max()]))
        return z, classes, p_imgs, img_err

    def initial_z(self, n):
        return torch.randn(n, self.z_dim, requires_grad=True, device=self.device)

    def initial_classes(self, n, requires_grad=True):
        return torch.rand(n, self.num_classes, requires_grad=requires_grad, device=self.device)

    def preprocess_vgg(self, x):
        return (x - self.vgg_mean) / self.vgg_std

    @property
    def device(self):
        return next(self.biggan.parameters()).device

    @property
    def z_dim(self):
        return self.biggan.config.z_dim

    @property
    def num_classes(self):
        return self.biggan.config.num_classes
