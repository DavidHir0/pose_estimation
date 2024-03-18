from collections import namedtuple

import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData

import propose.preprocessing.rat7m as pp
from propose.poses.rat7m import Rat7mPose
import numpy.typing as npt
Image = npt.NDArray[float]
from propose.poses import BasePose
import numpy as np
import torchvision.transforms as transforms
import random


class ScalePose(object):
    def __init__(self, scale):
        self.scale = scale

    def __call__(self, x):
        key_vals = {k: v for k, v in zip(x._fields, x)}

        key_vals["poses"] = pp.scale_pose(pose=x.poses, scale=self.scale)
        if "poses2d" in key_vals:
            key_vals["poses2d"] = pp.scale_pose(pose=x.poses2d, scale=self.scale)

        return x.__class__(**key_vals)


class CenterPose(object):
    def __init__(self, center_marker_name="SpineM"):
        self.center_marker_name = center_marker_name

    def __call__(self, x):
        key_vals = {k: v for k, v in zip(x._fields, x)}

        key_vals["poses"] = pp.center_pose(
            pose=key_vals["poses"], center_marker_name=self.center_marker_name
        )

        return x.__class__(**key_vals)


class CropImageToPose(object):
    def __init__(self, width=448):
        self.width = width

    def square_crop_to_pose(self, image: Image, pose2D: BasePose, width: int = 448) -> Image:
        """
        Crops a square from the input image such that the mean of the corresponding pose2D is in the center of the image.
        :param image: Input image to be cropped.
        :param pose2D: pose2D to find the center for cropping.
        :param width: width of the cropping (default = 350)
        :return: cropped image
        """
        mean_xy = pose2D.pose_matrix.mean(0).astype(int)


        padding = int(width // 2)

        x_min, x_max = 0, image.shape[1]
        y_min, y_max = 0, image.shape[0]

        x_start = max(mean_xy[0] - padding, x_min)
        x_end = min(mean_xy[0] + padding, x_max)
        y_start = max(mean_xy[1] - padding, y_min)
        y_end = min(mean_xy[1] + padding, y_max)

        x_slice = slice(x_start, x_end)
        y_slice = slice(y_start, y_end)


        pose2D.pose_matrix[:,0] -= x_start
        pose2D.pose_matrix[:,1] -= y_start
 
        image = image[y_slice, x_slice]
        if image.shape[0] < width or image.shape[1] < width:
            y_scale = image.shape[0] / width
            x_scale = image.shape[1] / width
            pose2D.pose_matrix[:,0] /= x_scale
            pose2D.pose_matrix[:,1] /= y_scale
            tensor_image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()
            tensor_image = F.interpolate(tensor_image, size=(width, width), mode='bilinear', align_corners=False)
            image = tensor_image.squeeze(0).permute(1, 2, 0).numpy()
        return image, pose2D



    def __call__(self, x):
        key_vals = {k: v for k, v in zip(x._fields, x)}

        pose2D = key_vals["poses"]
        image = key_vals["images"]
        camera = key_vals["cameras"]

        #pose2D = Rat7mPose(camera.proj2D(pose))
    
        image, pose2D = self.square_crop_to_pose(
            image=image, pose2D=pose2D, width=self.width
        )

        key_vals["images"] = image
        key_vals["poses"] = pose2D

        return x.__class__(**key_vals)

class NormalizeImages(object):
    def __call__(self, x):
        key_vals = {k: v for k, v in zip(x._fields, x)}

        image = key_vals["images"]
        print("SHAPEEE")
        print(image.shape)


        return x.__class__(**key_vals)

class random_rotate(object):
    def __call__(self, x):
        key_vals = {k: v for k, v in zip(x._fields, x)}

        image = key_vals["images"]

         # Randomly apply rotation
        if random.random() < 0.5:  # 50% chance of rotation
            angle = random.uniform(-30, 30)  # Random rotation angle between -30/30 degrees
            image = transforms.functional.rotate(image, angle)

        key_vals["images"] = image

        return x.__class__(**key_vals)

class random_flip(object):
    def __call__(self, x):
        key_vals = {k: v for k, v in zip(x._fields, x)}

        image = key_vals["images"]


        # Randomly apply horizontal flipping
        if random.random() < 0.5:  # 50% chance of flipping
            image = transforms.functional.hflip(image)

        key_vals["images"] = image

        return x.__class__(**key_vals)


class random_color_jitter(object):
    def __call__(self, x):
        key_vals = {k: v for k, v in zip(x._fields, x)}

        image = key_vals["images"]

        # Randomly apply color jittering
        if random.random() < 0.5:  # 50% chance of color jittering
            color_jitter = transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)
            image = color_jitter(image)

        key_vals["images"] = image

        return x.__class__(**key_vals)

class RotatePoseToCamera(object):
    def __call__(self, x):
        key_vals = {k: v for k, v in zip(x._fields, x)}

        pose = key_vals["poses"]
        camera = key_vals["cameras"]

        key_vals["poses"] = pp.rotate_to_camera(pose=pose, camera=camera)

        return x.__class__(**key_vals)

class Make_2D(object):
    def __call__(self, x):
        key_vals = {k: v for k, v in zip(x._fields, x)}

        pose = key_vals["poses"]
        camera = key_vals["cameras"]

        key_vals["poses"] = Rat7mPose(camera.proj2D(pose))

        return x.__class__(**key_vals)



class ToGraph(object):
    def __init__(self):
        self.graph_data_point = namedtuple(
            "GraphDataPoint", ("pose_matrix", "adjacency_matrix", "image")
        )

    def __call__(self, x):
        return self.graph_data_point(
            pose_matrix=x.poses.pose_matrix,
            adjacency_matrix=x.poses.adjacency_matrix,
            image=x.images,
        )


class SwitchArmsElbows(object):
    def __call__(self, x):
        key_vals = {k: v for k, v in zip(x._fields, x)}
        pose = key_vals["poses"]

        key_vals["poses"] = pp.switch_arms_elbows(pose=pose)

        return x.__class__(**key_vals)


class ScalePixelRange(object):
    def __call__(self, x):
        key_vals = {k: v for k, v in zip(x._fields, x)}
        key_vals["images"] = pp.scale_pixel_range(image=key_vals["images"])

        return x.__class__(**key_vals)


class Project2D(object):
    def __init__(self, idx):
        self.idx = idx

    def __call__(self, x):
        key_vals = {k: v for k, v in zip(x._fields, x)}
        poses = key_vals["poses"]
        key_vals["poses2d"] = poses[..., self.idx]

        return namedtuple("With2DPose", key_vals.keys())(**key_vals)


class ToHeteroData(object):
    def __init__(self, encode_joints: bool = False):
        """
        Converts a dataset to a HeteroData object.
        :param encode_joints: Whether to encode the joints as a one-hot vector
        """
        self.encode_joints = encode_joints

    def __call__(self, x):
        key_vals = {k: v for k, v in zip(x._fields, x)}

        pose3d = x.poses
        # pose2d = x.poses2d

        # one_hot_encoding = F.one_hot(torch.arange(len(pose3d)), len(pose3d)).float()

        # c = torch.Tensor(pose2d.pose_matrix)
        # if self.encode_joints:
        #     c = torch.cat([c, one_hot_encoding], dim=1)

        data = HeteroData()
        data["x"].x = torch.Tensor(pose3d.pose_matrix)
        data["x", "->", "x"].edge_index = torch.LongTensor(pose3d.edges).T
        data["x", "<-", "x"].edge_index = torch.LongTensor(pose3d.edges).T

        if "poses2d" in key_vals:
            pose2d = x.poses2d

            data["c"].x = torch.Tensor(pose2d.pose_matrix)
            context_edges = (
                torch.arange(0, pose3d.pose_matrix.shape[0])
                .repeat(2)
                .reshape(2, pose3d.pose_matrix.shape[0])
                .long()
            )
            data["c", "->", "x"].edge_index = context_edges

            del key_vals["poses2d"]

        # pose = HeteroData(
        #     {
        #         "x": {"x": torch.Tensor(pose3d.pose_matrix)},
        #         # "c": {"x": c},
        #         "edge_index": {
        #             ("x", "->", "x"): torch.LongTensor(pose3d.edges),
        #             ("x", "<-", "x"): torch.LongTensor(pose3d.edges),
        #             # ("c", "->", "x"): torch.arange(0, len(pose3d.edges)).repeat(2).reshape(2, len(pose3d.edges)).T.long(),
        #         },
        #     }
        # )

        key_vals["poses"] = data

        graph_data_point = namedtuple("GraphDataPoint", ("poses"))

        # return graph_data_point(**key_vals)
        return graph_data_point(**key_vals)

        # return data
