import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt

def show_grid(img, n_cols):
  grid = vutils.make_grid(img, nrow = n_cols, padding = 0)
  grid = grid / 2 + 0.5
  npgrid = grid.numpy()

  plt.axis('off')
  plt.figure(figsize = (10, 10))
  plt.imshow(np.transpose(npgrid, (1, 2, 0)))

  plt.show()
