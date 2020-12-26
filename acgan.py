from util import show_grid
import torch
import torch.nn as nn

def weights_init(m):
  classname = m.__class__.__name__
  if classname.find('Conv') != -1 or classname.find('Linear') != -1:
    m.weight.data.normal_(0.0, 0.02)

def to_oh(labels, n_classes):
  oh = torch.zeros(labels.size(0), n_classes)
  for (o, l) in zip(oh, labels):
    o[l] = 1.0
  return oh
    
class Generator(nn.Module):
  def __init__(self, z_len, n_classes, n_channels):
    super(Generator, self).__init__()

    self.linear = nn.Sequential(
      nn.Linear(z_len + n_classes, 512 * 4 * 4),
      nn.ReLU(True)
    )

    self.upconv = nn.Sequential(
      
      nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
      nn.BatchNorm2d(256),
      nn.ReLU(True),
      
      nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
      nn.BatchNorm2d(128),
      nn.ReLU(True),
      
      nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
      nn.BatchNorm2d(64),
      nn.ReLU(True),
      
      nn.ConvTranspose2d(64, n_channels, 4, 2, 1, bias=False),
      nn.Tanh()
    )

  def forward(self, z, c):
    output = torch.cat((z, c), 1)
    output = self.linear(output)
    output = output.view(-1, 512, 4, 4)
    output = self.upconv(output)

    return output

class Discriminator(nn.Module):
  def __init__(self, n_classes, n_channels):
    super(Discriminator, self).__init__()

    self.conv = nn.Sequential(
      
      nn.Conv2d(n_channels, 64, 4, 2, 1, bias=False),
      nn.BatchNorm2d(64),
      nn.LeakyReLU(0.2),
      nn.Dropout(0.5),
      
      nn.Conv2d(64, 128, 4, 2, 1, bias=False),
      nn.BatchNorm2d(128),
      nn.LeakyReLU(0.2),
      nn.Dropout(0.5),
      
      nn.Conv2d(128, 256, 4, 2, 1, bias=False),
      nn.BatchNorm2d(256),
      nn.LeakyReLU(0.2),
      nn.Dropout(0.5),
      
      nn.Conv2d(256, 512, 4, 2, 1, bias=False),
      nn.BatchNorm2d(512),
      nn.LeakyReLU(0.2),
      nn.Dropout(0.5),
      
      nn.Conv2d(512, 1024, 4, 1, 0, bias=False),
      nn.BatchNorm2d(1024),
      nn.LeakyReLU(0.2),
      nn.Dropout(0.5)
    )

    self.sigmoid = nn.Sequential(
      nn.Linear(1024, 1, bias = False),
      nn.Sigmoid()
    )

    self.softmax = nn.Sequential(
      nn.Linear(1024, n_classes, bias = False),
      #nn.Softmax(dim = 1)
    )

  def forward(self, input):
    output = self.conv(input)
    output = output.view(-1, 1024)
    src = self.sigmoid(output).view(-1, 1).squeeze(1)
    c = self.softmax(output)

    return src, c

class ACGAN:

  def __init__(self, z_len, n_classes, n_channels, device, g_file = None, d_file = None):
    self.z_len = z_len
    self.n_classes = n_classes
    self.n_channels = n_channels
    self.device = device
    self.generator = Generator(z_len, n_classes, n_channels).to(device)
    self.discriminator = Discriminator(n_classes, n_channels).to(device)
    if g_file == None:
      self.generator.apply(weights_init)
    else:
      self.generator.load_state_dict(torch.load(g_file))
    if d_file == None:
      self.discriminator.apply(weights_init)
    else:
      self.discriminator.load_state_dict(torch.load(d_file))

  def __str__(self):
    return str(self.generator) + '\n' + str(self.discriminator)

  def train(self, dataloader, n_epochs, src_criterion, c_criterion, g_optim, d_optim):
    fixed_z = torch.randn(self.n_classes, self.z_len, device = self.device)
    fixed_cls = list(range(self.n_classes))
    fixed_cls_oh = to_oh(torch.tensor(fixed_cls), self.n_classes).to(self.device)

    for epoch in range(n_epochs):
      for idx, (real_img_cpu, real_cls_cpu) in enumerate(dataloader, 0):
        real_img = real_img_cpu.to(self.device)
        real_cls = real_cls_cpu.to(self.device)
        batch_size = real_img.size(0)

        self.discriminator.zero_grad()

        lbl = torch.ones(batch_size, device = self.device)
        src, c = self.discriminator(real_img)

        E_d_real_src = src_criterion(src, lbl)
        E_d_real_c = c_criterion(c, real_cls)

        E_d_real = E_d_real_src + E_d_real_c
        E_d_real.backward()

        z = torch.randn(batch_size, self.z_len, device = self.device)
        fake_cls = torch.randint(low = 0, high = self.n_classes, size = (batch_size, )).to(self.device)
        fake_img = self.generator(z, to_oh(fake_cls, self.n_classes).to(self.device))

        lbl = torch.zeros(batch_size, device = self.device)
        src, c = self.discriminator(fake_img.detach())

        E_d_fake_src = src_criterion(src, lbl)
        E_d_fake_c = c_criterion(c, fake_cls)

        E_d_fake = E_d_fake_src + E_d_fake_c
        E_d_fake.backward()

        E_d = E_d_real + E_d_fake
        d_optim.step()

        self.generator.zero_grad()

        z = torch.randn(batch_size, self.z_len, device = self.device)
        fake_cls = torch.randint(low = 0, high = self.n_classes, size = (batch_size, )).to(self.device)
        fake_img = self.generator(z, to_oh(fake_cls, self.n_classes).to(self.device))
        
        lbl = torch.ones(batch_size, device = self.device)
        src, c = self.discriminator(fake_img)

        E_g_src = src_criterion(src, lbl)
        E_g_c = c_criterion(c, fake_cls)

        E_g = E_g_src + E_g_c
        E_g.backward()
        
        g_optim.step()

        if idx == len(dataloader) - 1:
          g_img = self.generator(fixed_z, fixed_cls_oh).cpu().detach()
          print('Epoha: %d/%d, E_d: %f E_g: %f' % (epoch + 1, n_epochs, E_d.item(), E_g.item()))
          show_grid(g_img, 5)
