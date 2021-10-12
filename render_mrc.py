import os
import mrcfile
import numpy as np
from PIL import Image
from scipy.interpolate import RegularGridInterpolator


filename = '/media/ubuntu/WDC/ACORN/logs/photon/photon.mrc'
output_dir = '/media/ubuntu/WDC/ACORN/logs/photon/render'
step = 1

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

x_dim = 74
y_dim = 77
z_dim = 394
res = 1024
x_ = np.linspace(-1, 1, x_dim + 1)[:-1]
y_ = np.linspace(-1, 1, y_dim + 1)[:-1]
z_ = np.linspace(-1, 1, z_dim + 1)[:-1]

x, y, z = np.meshgrid(x_, y_, z_, indexing='ij')
x = x.flatten()
y = y.flatten()
z = z.flatten()
xyz = np.stack((x, y, z), axis=1)

x_data = np.linspace(-1, 1, res+ 1)[:-1]
y_data = np.linspace(-1, 1, res + 1)[:-1]
z_data = np.linspace(-1, 1, res + 1)[:-1]

with mrcfile.open(filename) as mrc:
	fn = RegularGridInterpolator((x_data,y_data,z_data), mrc.data)
	v_out = fn(xyz).reshape(x_dim, y_dim, z_dim)
	v_out = (np.clip((v_out + 1.0) / 2, 0, 1) * 255.0).astype(np.uint8)
	for i in range(0, v_out.shape[0], step):
		output_name = os.path.join(output_dir, 'output_' + str(i) + '.png')
		img = Image.fromarray(v_out[i])
		img.save(output_name)