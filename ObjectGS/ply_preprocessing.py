"""
Point Cloud Preprocessing for ObjectGS

This module provides functionality for preprocessing 3D point clouds with semantic labels
by projecting them onto 2D images and assigning colors/labels through various voting strategies.

Author: Ruijie Zhu
License: MIT
"""

import struct
import numpy as np
import cv2
from collections import Counter, defaultdict
from plyfile import PlyData, PlyElement
from scene.colmap_loader import (
    read_intrinsics_binary, read_extrinsics_binary, read_next_bytes, 
    read_intrinsics_text, read_extrinsics_text
)
import argparse
import os

def read_points3D_binary(path_to_model_file):
    """
    Parses COLMAP's points3D.bin file and returns a dictionary:
    point3D_id -> (x, y, z, r, g, b, error, track)
    where track is a list of (image_id, point2D_idx) tuples.
    """
    def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
        data = fid.read(num_bytes)
        return struct.unpack(endian_character + format_char_sequence, data)

    import struct
    with open(path_to_model_file, "rb") as fid:
        num_points = read_next_bytes(fid, 8, "Q")[0]
        points3D = {}

        for _ in range(num_points):
            data = read_next_bytes(fid, 43, "QdddBBBd")  # point3D_id + xyz + rgb + error
            point3D_id = data[0]
            xyz = np.array(data[1:4])
            rgb = np.array(data[4:7])
            error = data[7]

            track_length = read_next_bytes(fid, 8, "Q")[0]
            track_elems = read_next_bytes(fid, 8 * track_length, "ii" * track_length)

            # track is a list of (image_id, point2D_idx)
            track = [(track_elems[i], track_elems[i + 1]) for i in range(0, len(track_elems), 2)]

            points3D[point3D_id] = (xyz[0], xyz[1], xyz[2], rgb[0], rgb[1], rgb[2], error, track)

    return points3D


def read_points3D_text(path):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadPoints3DText(const std::string& path)
        void Reconstruction::WritePoints3DText(const std::string& path)
    """
    xyzs = None
    rgbs = None
    errors = None
    num_points = 0
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                num_points += 1

    point3D = {}

    count = 0
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                xyz = np.array(tuple(map(float, elems[1:4])))
                rgb = np.array(tuple(map(int, elems[4:7])))
                error = np.array(float(elems[7]))
                point3D[count] = (xyz[0], xyz[1], xyz[2], rgb[0], rgb[1], rgb[2], error, [])
                count += 1

    return point3D


def read_points3D_ply(path):
    """
    Load point cloud from a 3DGS PLY file.
    Converts f_dc (Spherical Harmonics) to RGB colors.
    """
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    
    xyz = np.stack([vertices['x'], vertices['y'], vertices['z']], axis=1)
    
    if 'red' in vertices:
        rgb = np.stack([vertices['red'], vertices['green'], vertices['blue']], axis=1)
    elif 'f_dc_0' in vertices:
        # Convert 3DGS f_dc to RGB
        # SH base constant: 0.28209479177387814
        C0 = 0.28209479177387814
        f_dc = np.stack([vertices['f_dc_0'], vertices['f_dc_1'], vertices['f_dc_2']], axis=1)
        rgb = 0.5 + C0 * f_dc
        rgb = np.clip(rgb * 255, 0, 255).astype(np.uint8)
    else:
        rgb = np.zeros_like(xyz, dtype=np.uint8)

    point3D = {}
    for i in range(len(xyz)):
        # (x, y, z, r, g, b, error, track)
        point3D[i] = (xyz[i, 0], xyz[i, 1], xyz[i, 2], rgb[i, 0], rgb[i, 1], rgb[i, 2], 0, [])
    
    return point3D


def load_dynerf_cameras(path):
    """
    Load camera parameters from DyNeRF poses_bounds.npy.
    """
    poses_arr = np.load(os.path.join(path, "poses_bounds.npy"))
    poses = poses_arr[:, :-2].reshape([-1, 3, 5])  # (N_cams, 3, 5)
    
    # DyNeRF format: [H, W, focal] is in the last column
    H_all, W_all, focal_all = poses[:, :, -1].T
    
    # The first 3x4 part is the camera-to-world matrix
    # DyNeRF coordinate system: [r, u, -f, t]
    # We need to convert it to a more standard format or handle it in projection
    # In scene/neural_3D_dataset_NDC.py:
    # poses = np.concatenate([poses[..., 1:2], -poses[..., :1], poses[..., 2:4]], -1)
    
    cameras = {}
    images = {}
    
    # Detect image size from object_mask to match intrinsics
    mask_dir = os.path.join(path, "object_mask")
    downsample = 1.0
    if os.path.exists(mask_dir):
        mask_files = [f for f in os.listdir(mask_dir) if f.endswith('.png')]
        if mask_files:
            mask_path = os.path.join(mask_dir, mask_files[0])
            mask_img = cv2.imread(mask_path, -1)
            if mask_img is not None:
                mask_h, mask_w = mask_img.shape[:2]
                downsample = W_all[0] / mask_w
                print(f"Detected mask size: {mask_w}x{mask_h}, Original size: {W_all[0]}x{H_all[0]}")
                print(f"Applying downsample factor: {downsample}")

    for i in range(len(poses)):
        H, W, focal = H_all[i], W_all[i], focal_all[i]
        
        # Apply downsample
        H = H / downsample
        W = W / downsample
        focal = focal / downsample
        
        # Pinhole model: [fx, fy, cx, cy]
        # Assuming principal point is at center
        fx = fy = focal
        cx = W / 2.0
        cy = H / 2.0
        
        cameras[i] = (i, "PINHOLE", int(W), int(H), [fx, fy, cx, cy])
        
        # Extrinsics
        # poses[i] is (3, 5), first 3x4 is c2w
        c2w = poses[i, :3, :4]
        # Coordinate transformation as per DyNeRF loader:
        # R = [c2w[:, 1], -c2w[:, 0], c2w[:, 2]]
        R_c2w = np.stack([c2w[:, 1], -c2w[:, 0], c2w[:, 2]], axis=1)
        t_c2w = c2w[:, 3].reshape((3, 1))
        
        # Convert c2w to w2c
        R_w2c = R_c2w.T
        t_w2c = -np.dot(R_w2c, t_c2w)
        
        # Convert R to quaternion for _extract_pose_params compatibility
        # Or just return R, t directly in a custom way. 
        # For simplicity, we'll store it so it can be extracted.
        # We'll use a dummy qvec because _extract_pose_params expects it.
        # But wait, we can just update _extract_pose_params or handle DyNeRF separately.
        
        # Let's store R and t directly and add a flag
        image_name = f"original_time0_{i}.png"
        images[i] = (i, None, None, i, image_name, None, None, R_w2c, t_w2c)
        
    return cameras, images


class ID2RGBConverter:
    """Converter to map object IDs to unique RGB colors."""
    
    def __init__(self):
        self.all_id = []  # Store all generated IDs
        self.obj_to_id = {}  # Mapping from object ID to randomly generated color ID

    def _id_to_rgb(self, id: int):
        """Convert integer ID to RGB color."""
        rgb = np.zeros((3, ), dtype=np.uint8)  # Initialize RGB channels
        for i in range(3):
            rgb[i] = id % 256  # Take the lower 8 bits of the ID as the RGB channel value
            id = id // 256  # Right shift 8 bits to process the remaining part
        return rgb

    def convert(self, obj: int):
        """Convert single-channel ID to random RGB value."""
        if obj in self.obj_to_id:
            id = self.obj_to_id[obj]  # If the object already exists, get the corresponding ID
        else:
            # Randomly generate a unique ID and ensure no duplicates
            while True:
                id = np.random.randint(255, 256**3)
                if id not in self.all_id:
                    break
            self.obj_to_id[obj] = id  # Store the new ID in the dictionary
            self.all_id.append(id)  # Record this ID

        return id, self._id_to_rgb(id)  # Return the ID and corresponding RGB value

def corr_assign_final_colors(points3D, all_colors, all_labels):
    """Assign final colors using correlation-based voting.
    
    Args:
        points3D: Dictionary of 3D points
        all_colors: List of (point_id, color) tuples
        all_labels: List of (point_id, label) tuples
        
    Returns:
        Dictionary of updated 3D points with new colors and labels
    """
    from collections import defaultdict, Counter
    point_final_labels = {}    
    point_final_colors = {}
    
    colors_dict = defaultdict(list)
    labels_dict = defaultdict(list)

    for pid, color in all_colors:
        colors_dict[pid].append(color)

    for pid, label in all_labels:
        labels_dict[pid].append(label)

    for point_id in points3D:
        colors = colors_dict[point_id]
        labels = labels_dict[point_id]

        # Filter out items where label == 0 (background)
        filtered = [(c, l) for c, l in zip(colors, labels) if l != 0]
        if not filtered:
            continue  # Skip this point if there are no valid labels

        filtered_colors, filtered_labels = zip(*filtered)

        counter = Counter(filtered_labels)
        max_value = max(counter, key=counter.get)

        # Find the color corresponding to the most frequent label
        label_indices = [i for i, label in enumerate(filtered_labels) if label == max_value]
        max_color = filtered_colors[label_indices[0]]

        point_final_labels[point_id] = np.array(max_value)
        point_final_colors[point_id] = np.array(max_color)

    # Create a new points3D dictionary with updated colors and labels
    new_points3D = {}

    for point_id, point_data in points3D.items():
        # Original data format: (x, y, z, r, g, b, error, track)
        x, y, z, r, g, b, error, track = point_data
        r_new, g_new, b_new = point_final_colors.get(point_id, (r, g, b))
        label = point_final_labels.get(point_id, 0)

        new_points3D[point_id] = (x, y, z, r_new, g_new, b_new, label)

    return new_points3D

def majority_assign_final_colors(points3D, all_colors, all_labels):
    """Assign final colors using majority voting.
    
    Args:
        points3D: Dictionary of 3D points
        all_colors: List of (point_id, color) tuples
        all_labels: List of (point_id, label) tuples
        
    Returns:
        Dictionary of updated 3D points with final colors and labels
    """
    point_final_labels = {}    
    point_final_colors = {}
    
    colors_dict = defaultdict(list)
    labels_dict = defaultdict(list)

    # Build mapping dictionaries in advance to avoid lookups in loops
    for pid, color in all_colors:
        colors_dict[pid].append(color)

    for pid, label in all_labels:
        labels_dict[pid].append(label)

    # Iterate through points3D and find the most common labels and corresponding colors
    for point_id in points3D:
        colors = colors_dict[point_id]
        labels = labels_dict[point_id]
        
        if labels:
            # Use Counter to find the most frequent label
            counter = Counter(labels)
            max_value = max(counter, key=counter.get)
            
            # Find the color corresponding to the most frequent label
            label_indices = [i for i, label in enumerate(labels) if label == max_value]
            max_color = colors[label_indices[0]]  # Take the first matching color

            # Store results in final dictionaries
            point_final_labels[point_id] = np.array(max_value)
            point_final_colors[point_id] = np.array(max_color)
    
    # Create a new points3D dictionary with a consistent output format:
    # (x, y, z, r, g, b, label)
    new_points3D = {}
    for point_id, point_data in points3D.items():
        x, y, z, r, g, b = point_data[:6]
        r_new, g_new, b_new = point_final_colors.get(point_id, (r, g, b))
        label = point_final_labels.get(point_id, 0)
        new_points3D[point_id] = (x, y, z, r_new, g_new, b_new, label)

    return new_points3D

def prob_assign_final_colors(points3D, all_colors, all_labels):
    """Assign final colors using probability-based voting.
    
    Args:
        points3D: Dictionary of 3D points
        all_colors: List of (point_id, color) tuples
        all_labels: List of (point_id, label) tuples
        
    Returns:
        Dictionary of updated 3D points with sampled colors and labels
    """
    point_final_labels = {}    
    point_final_colors = {}
    
    colors_dict = defaultdict(list)
    labels_dict = defaultdict(list)

    # Build mapping dictionaries in advance to avoid lookups in loops
    for pid, color in all_colors:
        colors_dict[pid].append(color)

    for pid, label in all_labels:
        labels_dict[pid].append(label)

    # Iterate through points3D and vote based on probability distribution
    for point_id in points3D:
        colors = colors_dict[point_id]
        labels = labels_dict[point_id]

        if labels:
            # Use Counter to calculate label frequencies
            counter = Counter(labels)
            total = sum(counter.values())

            # Convert frequencies to probability distribution
            labels_list, counts = zip(*counter.items())
            probabilities = np.array(counts) / total

            # Randomly sample a label based on probabilities
            sampled_label = np.random.choice(labels_list, p=probabilities)

            # Find the color corresponding to the sampled label
            label_indices = [i for i, label in enumerate(labels) if label == sampled_label]
            sampled_color = colors[label_indices[0]]  # Take the first matching color

            # Store results in final dictionaries
            point_final_labels[point_id] = np.array(sampled_label)
            point_final_colors[point_id] = np.array(sampled_color)
    
    # Create a new points3D dictionary with a consistent output format:
    # (x, y, z, r, g, b, label)
    new_points3D = {}
    for point_id, point_data in points3D.items():
        x, y, z, r, g, b = point_data[:6]
        r_new, g_new, b_new = point_final_colors.get(point_id, (r, g, b))
        label = point_final_labels.get(point_id, 0)
        new_points3D[point_id] = (x, y, z, r_new, g_new, b_new, label)

    return new_points3D

def storePly(path, xyz, rgb, label):
    """Save point cloud data to PLY file format.
    
    Args:
        path: Output file path
        xyz: Point coordinates (N, 3)
        rgb: Point colors (N, 3)
        label: Point labels (N, 1)
    """
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
             ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
             ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
             ('label', 'u1')]
    
    normals = np.zeros_like(xyz)
    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb, label), axis=1)
    elements[:] = list(map(tuple, attributes))
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def storePlyRetain(input_path, output_path, label_dict):
    """Save point cloud data by retaining original properties and adding/updating 'label'.
    
    Args:
        input_path: Original PLY file path
        output_path: Output PLY file path
        label_dict: Dictionary mapping vertex index to label value
    """
    plydata = PlyData.read(input_path)
    vertices = plydata['vertex']
    
    # Prepare labels array
    num_vertices = len(vertices)
    labels = np.zeros(num_vertices, dtype=np.uint8)
    for idx, label in label_dict.items():
        if 0 <= idx < num_vertices:
            labels[idx] = label
            
    # Create new dtype that includes 'label'
    original_dtype = vertices.data.dtype
    if 'label' in original_dtype.names:
        # If 'label' already exists, we use the original dtype
        new_dtype = original_dtype
    else:
        # Add 'label' to the end of the dtype
        new_dtype = np.dtype(original_dtype.descr + [('label', 'u1')])
    
    new_data = np.empty(num_vertices, dtype=new_dtype)
    
    # Copy all original properties
    for name in original_dtype.names:
        new_data[name] = vertices.data[name]
    
        # Add/Update the 'label' property
        new_data['label'] = labels
        
        # Ensure standard strides for PyTorch compatibility
        new_data = np.array(new_data, copy=True)
        
        # Create new vertex element
        new_vertex_element = PlyElement.describe(new_data, 'vertex')
        
        # Replace vertex element in plydata
        # Note: PlyData is immutable-ish in some versions, but we can reconstruct it
        elements = list(plydata.elements)
        for i, el in enumerate(elements):
            if el.name == 'vertex':
                elements[i] = new_vertex_element
        plydata = PlyData(elements, text=plydata.text, comments=plydata.comments)
        
    plydata.write(output_path)

def quaternion_to_rotation_matrix(qw, qx, qy, qz):
    """Convert quaternion to rotation matrix.
    
    Args:
        qw, qx, qy, qz: Quaternion components
        
    Returns:
        3x3 rotation matrix
    """
    R = np.array([[1 - 2 * (qy**2 + qz**2), 2 * (qx*qy - qz*qw), 2 * (qx*qz + qy*qw)],
                  [2 * (qx*qy + qz*qw), 1 - 2 * (qx**2 + qz**2), 2 * (qy*qz - qx*qw)],
                  [2 * (qx*qz - qy*qw), 2 * (qy*qz + qx*qw), 1 - 2 * (qx**2 + qy**2)]])
    return R


def _load_and_process_image(label_image_dir, color_image_dir, image_name, converter, save_color_image_dir=None):
    """Load and process label and color images for a given view.
    
    Args:
        label_image_dir: Directory containing label images
        color_image_dir: Directory containing color images (can be None)
        image_name: Name of the image file
        converter: ID2RGBConverter instance
        save_color_image_dir: Directory to save the generated color image (can be None)
        
    Returns:
        Tuple of (color_image, label_image)
    """
    # Load label image
    label_image_file = os.path.join(label_image_dir, image_name)
    label_image_file = label_image_file.replace('.jpg', '.png').replace('.JPG', '.png')
    label_image = cv2.imread(label_image_file, -1)  # Load as single-channel grayscale
    
    # Load or generate color image
    if color_image_dir is not None:
        color_image_file = os.path.join(color_image_dir, image_name)
        color_image_file = color_image_file.replace('.jpg', '.png').replace('.JPG', '.png')
        color_image = cv2.imread(color_image_file, cv2.IMREAD_COLOR)
    else:
        # Generate color image from label image using converter
        color_image = np.zeros((label_image.shape[0], label_image.shape[1], 3), dtype=np.uint8)
        for i in range(label_image.shape[0]):
            for j in range(label_image.shape[1]):
                obj_id = label_image[i, j]
                _, rgb_color = converter.convert(obj_id)
                color_image[i, j] = rgb_color
        print(f"converter ids: {converter.obj_to_id.keys()}")
        # Save the generated color image if requested
        if save_color_image_dir is not None:
            os.makedirs(save_color_image_dir, exist_ok=True)
            save_path = os.path.join(save_color_image_dir, image_name)
            save_path = save_path.replace('.jpg', '.png').replace('.JPG', '.png')
            cv2.imwrite(save_path, color_image)
    
    return color_image, label_image


def _extract_camera_params(camera_data):
    """Extract camera parameters from camera data.
    
    Args:
        camera_data: Camera data tuple
        
    Returns:
        Tuple of (fx, fy, cx, cy)
    """
    # Check if it's DyNeRF format (stored as a list/tuple directly) or COLMAP object
    if isinstance(camera_data, (list, tuple)):
        _, model_type, width, height, params = camera_data
        fx, fy, cx, cy = params[0], params[1], params[2], params[3]
        return fx, fy, cx, cy
    else:
        # COLMAP Camera object
        fx, fy, cx, cy = camera_data.params[0], camera_data.params[1], camera_data.params[2], camera_data.params[3]
        return fx, fy, cx, cy


def _extract_pose_params(image_data):
    """Extract pose parameters from image data.
    
    Args:
        image_data: Image data tuple
        
    Returns:
        Tuple of (R, t) where R is rotation matrix and t is translation vector
    """
    # Check if DyNeRF format (9 elements, R and t at the end)
    if len(image_data) == 9 and image_data[7] is not None:
        R = image_data[7]
        t = image_data[8]
        return R, t
        
    _, qvec, tvec, camera_id, image_name, points2D, points3D_ids = image_data
    qw, qx, qy, qz = qvec
    tx, ty, tz = tvec
    
    # Convert quaternion to rotation matrix
    R = quaternion_to_rotation_matrix(qw, qx, qy, qz)
    t = np.array([tx, ty, tz]).reshape((3, 1))
    
    return R, t

_project_print_count = 2
def project_points(points3D, R, t, fx, fy, cx, cy, invert=False, width=None):
    """Project 3D points to 2D image plane.
    
    Args:
        points3D: Dictionary of 3D points
        R: Rotation matrix (3x3)
        t: Translation vector (3x1)
        fx, fy: Focal lengths
        cx, cy: Principal point coordinates
        invert: Whether to flip points horizontally
        width: Image width (required if invert is True)
        
    Returns:
        List of projected points (point_id, u, v)
    """
    global _project_print_count
    if _project_print_count > 0:
        print(f"\n[DEBUG] project_points call {_project_print_count}")
        print(f"R:\n{R}")
        print(f"t:\n{t}")
        print(f"Intrinsics: fx={fx}, fy={fy}, cx={cx}, cy={cy}")
        print(f"Invert: {invert}, Width: {width}")
        
        # Print first 3 points projection for inspection
        count = 0
        for pid, pdata in points3D.items():
            if count >= 3: break
            point = np.array([pdata[0], pdata[1], pdata[2]]).reshape(3, 1)
            cam_point = np.dot(R, point) + t
            x, y, z = cam_point.flatten()
            u = fx * (x / z) + cx
            v = fy * (y / z) + cy
            if invert and width is not None:
                u = width - 1 - u
            print(f"  Point {pid}: world={point.flatten()}, cam={[x,y,z]}, uv=({u}, {v})")
            count += 1
        _project_print_count -= 1

    projected_points = []
    for point_id, point_data in points3D.items():
        point = np.array([point_data[0], point_data[1], point_data[2]]).reshape(3, 1)
        # Transform world coordinates to camera coordinate system
        cam_point = np.dot(R, point) + t
        x, y, z = cam_point.flatten()
        # Project 3D point to 2D image using camera intrinsics
        u = fx * (x / z) + cx
        v = fy * (y / z) + cy
        
        if invert and width is not None:
            u = width - 1 - u
            
        projected_points.append((point_id, int(u), int(v)))
    return projected_points

def get_point_colors_from_image(projected_points, color_image, label_image):
    """Get colors and labels for projected points from images.
    
    Args:
        projected_points: List of projected points (point_id, u, v)
        color_image: RGB color image
        label_image: Label image
        
    Returns:
        Tuple of (point_colors, point_labels) lists
    """
    point_colors = []
    point_labels = []
    for point_id, u, v in projected_points:
        if 0 <= u < color_image.shape[1] and 0 <= v < color_image.shape[0]:
            # Get RGB color value of the pixel
            color = color_image[v, u]  # Note: OpenCV loads images in (B, G, R) format
            point_colors.append((point_id, color))
            label = label_image[v, u]
            point_labels.append((point_id, label))
    return point_colors, point_labels


def draw_points_on_image(image, projected_points, point_colors, output_path, point_radius=1):
    """Draw projected points on an image with colors from point_colors.
    
    Args:
        image: The base image (numpy array, BGR format from OpenCV)
        projected_points: List of projected points (point_id, u, v)
        point_colors: List of (point_id, color) tuples where color is BGR
        output_path: Path to save the visualization
        point_radius: Radius of the circle to draw for each point
    """
    # Create a copy of the image to draw on
    image_with_points = np.zeros_like(image, dtype=np.float32)
    
    # Build a dict for quick color lookup by point_id
    color_dict = {pid: color for pid, color in point_colors}
    
    # Draw each projected point
    for point_id, u, v in projected_points:
        if 0 <= u < image.shape[1] and 0 <= v < image.shape[0]:
            if point_id in color_dict:
                # Get the color for this point
                color = color_dict[point_id]
                # Convert to tuple of ints for OpenCV (BGR format)
                color_bgr = tuple(int(c) for c in color)
                # Draw a circle at the projected position
                cv2.circle(image_with_points, (int(u), int(v)), point_radius, color_bgr, -1)
    
    # Save the image
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, image_with_points)
    
    print(f"Point visualization saved to {output_path}")


def majority_voting(images, points3D, cameras, label_image_dir, color_image_dir, converter, output_ply_path, invert=False, input_ply_path=None, add_label_only=False):
    """Perform majority voting to assign colors and labels to 3D points.
    
    Args:
        images: Dictionary of image data
        points3D: Dictionary of 3D points
        cameras: Dictionary of camera parameters
        label_image_dir: Directory containing label images
        color_image_dir: Directory containing color images (optional)
        converter: ID2RGBConverter instance
        output_ply_path: Output PLY file path
        invert: Whether to flip points horizontally
        input_ply_path: Original PLY file path for retaining properties
        add_label_only: Whether to only add label to the original PLY
    """
    all_point_colors = []
    all_point_labels = []
    
    # Determine save directory for color images if they are being generated
    save_color_image_dir = None
    if color_image_dir is None:
        # Assuming label_image_dir is something like '.../object_mask',
        # we'll save color images to '.../color_mask'
        save_color_image_dir = os.path.join(os.path.dirname(label_image_dir), 'color_mask')

    # Iterate through all views
    for image_id, image_data in images.items():
        image_name = image_data[4]
        
        # Extract pose and camera parameters
        R, t = _extract_pose_params(image_data)
        fx, fy, cx, cy = _extract_camera_params(cameras[image_data[3]])
        
        # Load and process images
        color_image, label_image = _load_and_process_image(
            label_image_dir, color_image_dir, image_name, converter,
            save_color_image_dir=save_color_image_dir
        )

        # Project point cloud to current view
        image_height, image_width = color_image.shape[:2]
        projected_points = project_points(points3D, R, t, fx, fy, cx, cy, invert=invert, width=image_width)

        # Get color for each point in this view
        point_colors, point_labels = get_point_colors_from_image(projected_points, color_image, label_image)

        # Save results
        all_point_colors.extend(point_colors)
        all_point_labels.extend(point_labels)
        
        # Draw points on image and save
        if projected_points:
            vis_output_path = output_ply_path.replace('.ply', f'_debug/view_{image_id:03d}_points.png')
            draw_points_on_image(color_image, projected_points, point_colors, vis_output_path)

    # Calculate final colors for each point and update point cloud colors
    points3D = majority_assign_final_colors(points3D, all_point_colors, all_point_labels)

    if add_label_only and input_ply_path and os.path.exists(input_ply_path):
        label_dict = {pid: pdata[6] for pid, pdata in points3D.items()}
        storePlyRetain(input_ply_path, output_ply_path, label_dict)
    else:
        # Extract point cloud coordinates and colors
        xyz = np.array([[point_data[0], point_data[1], point_data[2]] for point_data in points3D.values()])
        rgb = np.array([[point_data[3], point_data[4], point_data[5]] for point_data in points3D.values()])
        label = np.array([[point_data[6]] for point_data in points3D.values()])

        # Save point cloud as PLY file
        storePly(output_ply_path, xyz, rgb, label)

    print(f"Point cloud saved to {output_ply_path}")

def prob_voting(images, points3D, cameras, label_image_dir, color_image_dir, converter, output_ply_path, invert=False, input_ply_path=None, add_label_only=False):
    """Perform probability-based voting to assign colors and labels to 3D points.
    
    Args:
        images: Dictionary of image data
        points3D: Dictionary of 3D points
        cameras: Dictionary of camera parameters
        label_image_dir: Directory containing label images
        color_image_dir: Directory containing color images (optional)
        converter: ID2RGBConverter instance
        output_ply_path: Output PLY file path
        invert: Whether to flip points horizontally
        input_ply_path: Original PLY file path for retaining properties
        add_label_only: Whether to only add label to the original PLY
    """
    all_point_colors = []
    all_point_labels = []
    
    # Determine save directory for color images if they are being generated
    save_color_image_dir = None
    if color_image_dir is None:
        save_color_image_dir = os.path.join(os.path.dirname(label_image_dir), 'color_mask')

    # Iterate through all views
    for image_id, image_data in images.items():
        image_name = image_data[4]

        # Extract some camera and pose parameters
        R, t = _extract_pose_params(image_data)
        fx, fy, cx, cy = _extract_camera_params(cameras[image_data[3]])
        
        # Load and process images
        color_image, label_image = _load_and_process_image(
            label_image_dir, color_image_dir, image_name, converter,
            save_color_image_dir=save_color_image_dir
        )

        # Project point cloud to current view
        image_height, image_width = color_image.shape[:2]
        projected_points = project_points(points3D, R, t, fx, fy, cx, cy, invert=invert, width=image_width)

        # Get color for each point in this view
        point_colors, point_labels = get_point_colors_from_image(projected_points, color_image, label_image)

        # Save results
        all_point_colors.extend(point_colors)
        all_point_labels.extend(point_labels)
        
        # Draw points on image and save
        if projected_points:
            vis_output_path = output_ply_path.replace('.ply', f'_debug/view_{image_id:03d}_points.png')
            draw_points_on_image(color_image, projected_points, point_colors, vis_output_path)

    # Calculate final colors for each point using probability-based voting
    points3D = prob_assign_final_colors(points3D, all_point_colors, all_point_labels)

    if add_label_only and input_ply_path and os.path.exists(input_ply_path):
        label_dict = {pid: pdata[6] for pid, pdata in points3D.items()}
        storePlyRetain(input_ply_path, output_ply_path, label_dict)
    else:
        # Extract point cloud coordinates and colors
        xyz = np.array([[point_data[0], point_data[1], point_data[2]] for point_data in points3D.values()])
        rgb = np.array([[point_data[3], point_data[4], point_data[5]] for point_data in points3D.values()])
        label = np.array([[point_data[6]] for point_data in points3D.values()])

        # Save point cloud as PLY file
        storePly(output_ply_path, xyz, rgb, label)

    print(f"Point cloud saved to {output_ply_path}")

def corr_voting(images, points3D, label_image_dir, converter, output_ply_path, input_ply_path=None, add_label_only=False):
    """Perform correlation-based voting using track correspondence.
    
    Args:
        images: Dictionary of image data
        points3D: Dictionary of 3D points with track information
        label_image_dir: Directory containing label images
        converter: ID2RGBConverter instance
        output_ply_path: Output PLY file path
        input_ply_path: Original PLY file path for retaining properties
        add_label_only: Whether to only add label to the original PLY
    """
    # Ensure color_mask directory exists if we want to save generated color images
    save_color_image_dir = os.path.join(os.path.dirname(label_image_dir), 'color_mask')
    processed_images = set()

    all_colors = []
    all_labels = []
    for point3D_id, point_data in points3D.items():
        x, y, z, r, g, b, error, track = point_data
        votes = []

        for image_id, point2D_idx in track:
            if image_id not in images:
                continue
            _, _, _, _, image_name, xys, _ = images[image_id]
            if point2D_idx >= len(xys):
                continue
            u, v = xys[point2D_idx]
            u = int(round(u))
            v = int(round(v))

            label_image_file = os.path.join(label_image_dir, image_name)
            label_image_file = label_image_file.replace('.jpg', '.png') if label_image_file.endswith('.jpg') else label_image_file.replace('.JPG', '.png')
            label_image = cv2.imread(label_image_file, -1)
            if label_image is None or v < 0 or v >= label_image.shape[0] or u < 0 or u >= label_image.shape[1]:
                continue

            obj_id = label_image[v, u]
            _, rgb_color = converter.convert(obj_id)

            # Save color image if not already processed
            if image_name not in processed_images:
                color_image = np.zeros((label_image.shape[0], label_image.shape[1], 3), dtype=np.uint8)
                # This might be slow if done for every image, but corr_voting 
                # doesn't naturally iterate over images first.
                # To be efficient, we only do this once per image.
                for i in range(label_image.shape[0]):
                    for j in range(label_image.shape[1]):
                        oid = label_image[i, j]
                        _, rc = converter.convert(oid)
                        color_image[i, j] = rc
                
                os.makedirs(save_color_image_dir, exist_ok=True)
                save_path = os.path.join(save_color_image_dir, image_name)
                save_path = save_path.replace('.jpg', '.png').replace('.JPG', '.png')
                cv2.imwrite(save_path, color_image)
                processed_images.add(image_name)

            all_colors.append((point3D_id, rgb_color))
            all_labels.append((point3D_id, obj_id))


    # Assign colors and labels through voting
    points3D = corr_assign_final_colors(points3D, all_colors, all_labels)

    if add_label_only and input_ply_path and os.path.exists(input_ply_path):
        label_dict = {pid: pdata[6] for pid, pdata in points3D.items()}
        storePlyRetain(input_ply_path, output_ply_path, label_dict)
    else:
        # Extract and save
        xyz = np.array([[p[0], p[1], p[2]] for p in points3D.values()])
        rgb = np.array([[p[3], p[4], p[5]] for p in points3D.values()])
        label = np.array([[p[6]] for p in points3D.values()])

        storePly(output_ply_path, xyz, rgb, label)
    print(f"Point cloud saved to {output_ply_path}")


def main(args):
    """Main processing function.
    
    Args:
        args: Command line arguments containing dataset_path, algorithm, and output_ply_name
    """
    dataset_path = args.dataset_path
    
    # Check if dataset_path itself is a DyNeRF directory (contains poses_bounds.npy)
    if os.path.exists(os.path.join(dataset_path, "poses_bounds.npy")) or args.dataset_type == "dynerf":
        dataset_folders = ["."]
    else:
        dataset_folders = os.listdir(dataset_path)

    for dataset_folder in dataset_folders:
        current_path = os.path.join(dataset_path, dataset_folder)
        print(f"Processing {current_path}...")
        
        label_image_dir = os.path.join(current_path, 'object_mask')
        color_image_dir = os.path.join(current_path, 'color_mask')
        if not os.path.isdir(color_image_dir):
            color_image_dir = None
        
        output_ply_path = args.output_ply_path
        if output_ply_path is None:
            output_ply_path = args.input_ply_path.replace(".ply", "_with_label.ply")

        # Loading logic
        if os.path.exists(os.path.join(current_path, "poses_bounds.npy")):
            print("Detected DyNeRF dataset format...")
            cameras, images = load_dynerf_cameras(current_path)
            
            # Use specified input PLY or look for default
            ply_path = args.input_ply_path
            if not ply_path:
                ply_path = os.path.join(current_path, "points3D_downsample2.ply")
                
            if os.path.exists(ply_path):
                print(f"Loading PLY from {ply_path}...")
                points3D = read_points3D_ply(ply_path)
            else:
                raise FileNotFoundError(f"PLY file not found at {ply_path}")
        else:
            # Try to load binary COLMAP files first, then fall back to text files
            output_ply_path = os.path.join(current_path, 'sparse/0/' + args.output_ply_name)
            try:
                camera_file = os.path.join(current_path, 'sparse/0/cameras.bin')
                image_file = os.path.join(current_path, 'sparse/0/images.bin')
                points3D_file = os.path.join(current_path,'sparse/0/points3D.bin')
                cameras = read_intrinsics_binary(camera_file)
                images = read_extrinsics_binary(image_file)
                points3D = read_points3D_binary(points3D_file)
            except:
                camera_file = os.path.join(current_path, 'colmap/cameras_undistorted.txt')
                image_file = os.path.join(current_path, 'colmap/images.txt')
                points3D_file = os.path.join(current_path,'colmap/points3D.txt')            
                cameras = read_intrinsics_text(camera_file)
                images = read_extrinsics_text(image_file)
                points3D = read_points3D_text(points3D_file)

        converter = ID2RGBConverter()

        # Determine input PLY path if available for retention
        input_ply_path = args.input_ply_path
        if not input_ply_path:
            if os.path.exists(os.path.join(current_path, "poses_bounds.npy")):
                input_ply_path = os.path.join(current_path, "points3D_downsample2.ply")
            elif os.path.exists(os.path.join(current_path, "sparse/0/points3D.bin")):
                # This is a heuristic, but often people want to add labels to a specific PLY
                # We'll check if the user provided one, otherwise we don't have a default for COLMAP
                pass

        # Apply selected voting algorithm
        if args.algorithm == 'majority':
            print("Using majority voting...")
            majority_voting(images, points3D, cameras, label_image_dir, color_image_dir, converter, output_ply_path, invert=args.invert, input_ply_path=input_ply_path, add_label_only=args.add_label_only)
        elif args.algorithm == 'prob':
            print("Using probability-based voting...")
            prob_voting(images, points3D, cameras, label_image_dir, color_image_dir, converter, output_ply_path, invert=args.invert, input_ply_path=input_ply_path, add_label_only=args.add_label_only)
        elif args.algorithm == 'corr':
            print("Using correlation-based voting...")
            corr_voting(images, points3D, label_image_dir, converter, output_ply_path, input_ply_path=input_ply_path, add_label_only=args.add_label_only)
        else:
            raise ValueError("Unknown algorithm. Choose from 'majority', 'prob', or 'corr'.")


if __name__ == "__main__":
    """Command line interface for point cloud preprocessing.
    
    Supports three voting algorithms:
    - majority: Simple majority voting
    - prob: Probability-based voting with random sampling
    - corr: Correlation-based voting using track correspondences
    """
    parser = argparse.ArgumentParser(
        description="Preprocess 3D point clouds with semantic labels using various voting strategies."
    )
    parser.add_argument(
        '--dataset_path', 
        type=str, 
        default='datasets/lerf_mask', 
        help='Path to the dataset directory'
    )
    parser.add_argument(
        '--algorithm', 
        type=str, 
        default='corr', 
        choices=['majority', 'prob', 'corr'], 
        help='Voting algorithm to use'
    )
    parser.add_argument(
        '--output_ply_path', 
        type=str, 
        default=None, 
        help='Output PLY file path. Can be None.'
    )
    parser.add_argument(
        '--input_ply_path',
        type=str,
        default='',
        help='Specific path to input PLY file (optional)'
    )
    parser.add_argument(
        '--dataset_type',
        type=str,
        default='colmap',
        choices=['colmap', 'dynerf'],
        help='Type of dataset (colmap or dynerf)'
    )
    parser.add_argument(
        '--invert',
        action='store_true',
        help='Enable horizontal flipping of the projected points'
    )
    parser.add_argument(
        '--add_label_only',
        action='store_true',
        help='If set, retain original point cloud data and only add/update the "label" property'
    )
    args = parser.parse_args()
    main(args)
