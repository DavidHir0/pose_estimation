from propose.datasets.rat7m.loaders import load_cameras, load_mocap
from propose.poses import Rat7mPose
from pathlib import Path
import propose.preprocessing.rat7m as pp

import numpy as np
from tqdm import tqdm
from PIL import Image

import imageio

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
from matplotlib import animation
import pickle

pose = np.load('data/rat7m/s2-d1/poses/s2-d1.npy')
pose = Rat7mPose(pose)

with open('data/rat7m/s2-d1/cameras/s2-d1.pickle', 'rb') as file:
    cameras = pickle.load(file)

print(pose.shape)


# fig = plt.figure(figsize=(10, 10))
# ax1 = fig.add_subplot(1, 1, 1, projection="3d")
# ax1.get_proj = lambda: np.dot(Axes3D.get_proj(ax1), np.diag([1, 1, 0.75, 1]))
# ax1.view_init(30, 30)
# ax1.set_xlim(-400, -100)
# ax1.set_ylim(-300, 0)
# ax1.set_zlim(0, 100)

# pose[5000].plot(ax1)
# print(pose[0])
# plt.savefig('3d_plot.png', dpi=300) 

camera_1 = cameras["Camera1"]
pose2D = Rat7mPose(camera_1.proj2D(pose))
print(pose2D.shape)

#im = Image.open('data/rat7m/s2-d1/images/s2-d1-camera1-0/s2-d1-camera1-00001.jpg')

mocap_path = "data/rat7m/mocap/mocap-s2-d1.mat"

mocap = load_mocap(mocap_path)
cameras = load_cameras(mocap_path)

# Mask
nan_mask = pp.mask_nans(mocap)
fail_mask = pp.mask_marker_failure(mocap)

mask = nan_mask + fail_mask


vid_path = "data/rat7m/s2-d1/movies/s2-d1-camera1-0.mp4"
vid = imageio.get_reader(vid_path)
frame_idx = camera_1.frames.squeeze()[mask][0]
im = vid.get_data(frame_idx)

plt.figure()
ax = plt.gca()
plt.imshow(im)
pose2D[0].plot(ax)
plt.show()
plt.savefig('2d_plot.png', dpi=300) 