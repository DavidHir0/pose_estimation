import propose.preprocessing.rat7m as pp
from propose.datasets.rat7m.loaders import load_mocap, load_cameras
from pathlib import Path
import pickle

dirname = "data/rat7m"
data_key = "s2-d1"
mocap_path = f"{dirname}/mocap/mocap-{data_key}.mat"

# Convert video to images
#pp.convert_movies_to_images(dirname, data_key)


# Load pose and camera
mocap = load_mocap(mocap_path)
cameras = load_cameras(mocap_path)

# Mask
nan_mask = pp.mask_nans(mocap)
fail_mask = pp.mask_marker_failure(mocap)

mask = nan_mask + fail_mask

mocap, cameras = pp.apply_mask(mask, mocap, cameras)




# Save
pose_dir = Path(f"{dirname}/{data_key}/poses")
pose_dir.mkdir(parents=True, exist_ok=True)

pose_path = pose_dir / f"{data_key}.npy"

mocap.save(pose_path)



camera_dir = Path(f"{dirname}/{data_key}/cameras")
camera_dir.mkdir(parents=True, exist_ok=True)

camera_path = camera_dir / f"{data_key}.pickle"

with open(camera_path, "wb") as f:
    pickle.dump(cameras, f)
















