import os
import mrcfile
import numpy as np
from PIL import Image
from scipy.interpolate import RegularGridInterpolator

x_dim = 74
y_dim = 77
z_dim = 394
res = 512
start_id = 0
end_id = 2
photon_path = '../logs/photon_'

for pmt_id in range(start_id, end_id):

	filename = photon_path + str(pmt_id) + '.mrc'
	output_dir = photon_path + str(pmt_id) + '/render'
	output_np_filename = photon_path + str(pmt_id) + '/acorn_pmt_' + str(pmt_id) + '.npy'

	if not os.path.exists(output_dir):
	    os.makedirs(output_dir)

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
		v_out = v_out * 0.5 + 0.5

		with open(output_np_filename, 'wb') as f:
			np.save(f, v_out)

		v_out = (np.clip(v_out, 0, 1) * 255.0).astype(np.uint8)
		for i in range(0, v_out.shape[0]):
			output_name = os.path.join(output_dir, 'output_' + str(i) + '.png')
			img = Image.fromarray(v_out[i])
			img.save(output_name)