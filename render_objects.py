#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import math
import imageio
from matplotlib import pyplot as plt
import numpy as np
import torch
from scene import Scene
import os
import cv2
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render_pcd
from scene.dataset_readers import fetchPly
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args, ModelHiddenParams
from gaussian_renderer import GaussianModel
from time import time
import threading
import concurrent.futures
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from sklearn.neighbors import NearestNeighbors


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    hyperparam = ModelHiddenParams(parser)
    parser.add_argument("--iteration", default=-1, type=int) #
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--skip_video", action="store_true")
    parser.add_argument("--configs", type=str)
    parser.add_argument("--ply_path", type=str)
    args = get_combined_args(parser)
    print("Rendering " , args.ply_path)
    if args.configs:
        import mmengine
        from utils.params_utils import merge_hparams
        config = mmengine.Config.fromfile(args.configs)
        args = merge_hparams(args, config)
    # Initialize system state (RNG)
    safe_state(args.quiet)
    
    dataset= model.extract(args)
    iteration = args.iteration
    hyperparam = hyperparam.extract(args)
    gaussians = GaussianModel(dataset.sh_degree, hyperparam)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
    
    cam_type=scene.dataset_type
    bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    
    # Load the point cloud from PLY
    pcd = fetchPly(args.ply_path)
    print("Loaded point cloud from: ", args.ply_path)
    print("Point cloud size: ", pcd.points.shape[0])

    ## VideoCameras: moving camera / TestCameras: fixed camera (for DyNeRF)
    cameras = scene.getVideoCameras()
    imgs = []
    to8b = lambda x : (255*np.clip(x.cpu().numpy(),0,1)).astype(np.uint8)
    pipeline_params = pipeline.extract(args)
    for idx, viewpoint_camera in enumerate(tqdm(cameras, desc="Rendering progress")):
        rendering = render_pcd(viewpoint_camera, pcd, pipeline_params, background, cam_type=cam_type, scaling_modifier=3.0)
        rendered_img = rendering["render"]
        imgs.append(to8b(rendered_img.detach().cpu()).transpose(1,2,0))

    ## TODO: save_path
    imageio.mimwrite(os.path.join(args.model_path, f"group_{os.path.splitext(os.path.basename(args.configs))[0]}.mp4"), imgs, fps=30)
    print("Video Saved.")
        

    

    
