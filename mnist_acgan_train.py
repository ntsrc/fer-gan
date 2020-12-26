from acgan import ACGAN
import torch
import torch.backends.cudnn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.utils.data
import torch.optim as optim
import torch.nn as nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.backends.cudnn.benchmark = True

ac_gan = ACGAN(100, 10, 1, device)
print(ac_gan)

dataset = datasets.MNIST(
  root = './data',
  download = True,
  transform = transforms.Compose([
    transforms.Resize(64),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)) 
  ])
)

dataloader = torch.utils.data.DataLoader(
  dataset,
  batch_size = 100,
  shuffle = True,
  num_workers=2
)

src_criterion = nn.MSELoss()
c_criterion = nn.CrossEntropyLoss()

d_optim = optim.Adam(ac_gan.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
g_optim = optim.Adam(ac_gan.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))

ac_gan.train(dataloader, 50, src_criterion, c_criterion, g_optim, d_optim)

torch.save(ac_gan.generator.state_dict(), './mnist-acgan-g.pth')
torch.save(ac_gan.discriminator.state_dict(), './mnist-acgan-d.pth')
