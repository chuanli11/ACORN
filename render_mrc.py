import os
import mrcfile
import numpy as np
from PIL import Image


filename = '/media/ubuntu/WDC/ACORN/logs/photon/photon.mrc'
output_dir = '/media/ubuntu/WDC/ACORN/logs/photon/render'
step = 32

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

with mrcfile.open(filename) as mrc:
	print(mrc.data.shape)
	print(type(mrc.data))
	print(np.amax(mrc.data))
	print(np.amin(mrc.data))

	my_data = (np.clip(mrc.data + 0.5, 0, 1) * 255.0).astype(np.uint8)
	print(np.amax(my_data))
	print(np.amin(my_data))

	for i in range(0, my_data.shape[0], step):
		output_name = os.path.join(output_dir, 'output_' + str(i) + '.png')
		img = Image.fromarray(my_data[i])
		img.save(output_name)