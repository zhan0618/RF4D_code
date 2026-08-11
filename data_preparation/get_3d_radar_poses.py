import os
import csv
import json
import numpy as np
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import io
import argparse

def roll(r):
    return np.array(
        [[1, 0, 0], [0, np.cos(r), np.sin(r)], [0, -np.sin(r), np.cos(r)]],
        dtype=np.float64,
    )


def pitch(p):
    return np.array(
        [[np.cos(p), 0, -np.sin(p)], [0, 1, 0], [np.sin(p), 0, np.cos(p)]],
        dtype=np.float64,
    )


def yaw(y):
    return np.array(
        [[np.cos(y), np.sin(y), 0], [-np.sin(y), np.cos(y), 0], [0, 0, 1]],
        dtype=np.float64,
    )


def yawPitchRollToRot(y, p, r):
    return roll(r) @ pitch(p) @ yaw(y)


def enforce_orthog(T, dim=3):
    """Enforces orthogonality of a 3x3 rotation matrix within a 4x4 homogeneous transformation matrix.
    Args:
        T (np.ndarray): 4x4 transformation matrix
        dim (int): dimensionality of the transform 2==2D, 3==3D
    Returns:
        np.ndarray: 4x4 transformation matrix with orthogonality conditions on the rotation matrix enforced.
    """
    if dim == 2:
        if abs(np.linalg.det(T[0:2, 0:2]) - 1) < 1e-10:
            return T
        R = T[0:2, 0:2]
        epsilon = 0.001
        if abs(R[0, 0] - R[1, 1]) > epsilon or abs(R[1, 0] + R[0, 1]) > epsilon:
            print("WARNING: this is not a proper rigid transformation:", R)
            return T
        a = (R[0, 0] + R[1, 1]) / 2
        b = (-R[1, 0] + R[0, 1]) / 2
        s = np.sqrt(a**2 + b**2)
        a /= s
        b /= s
        R[0, 0] = a
        R[0, 1] = b
        R[1, 0] = -b
        R[1, 1] = a
        T[0:2, 0:2] = R
    if dim == 3:
        if abs(np.linalg.det(T[0:3, 0:3]) - 1) < 1e-10:
            return T
        c1 = T[0:3, 1]
        c2 = T[0:3, 2]
        c1 /= np.linalg.norm(c1)
        c2 /= np.linalg.norm(c2)
        newcol0 = np.cross(c1, c2)
        newcol1 = np.cross(c2, newcol0)
        T[0:3, 0] = newcol0
        T[0:3, 1] = newcol1
        T[0:3, 2] = c2
    return T

def get_inverse_tf(T):
    """Returns the inverse of a given 4x4 homogeneous transform.
    Args:
        T (np.ndarray): 4x4 transformation matrix
    Returns:
        np.ndarray: inv(T)
    """
    T2 = T.copy()
    T2[:3, :3] = T2[:3, :3].transpose()
    T2[:3, 3:] = -1 * T2[:3, :3] @ T2[:3, 3:]
    return T2


def convert_line_to_pose(line, dim=2):
    """Reads trajectory from list of strings (single row of the comma-separeted groundtruth file). See Boreas
    documentation for format
    Args:
        line (List[string]): list of strings
        dim (int): dimension for evaluation. Set to '3' for 3D or '2' for 2D
    Returns:
        (np.ndarray): 4x4 SE(3) pose
        (int): time in nanoseconds
    """
    # returns T_iv
    line = line.replace("\n", ",").split(",")
    line = [float(i) for i in line[:-1]]
    # x, y, z -> 1, 2, 3
    # roll, pitch, yaw -> 7, 8, 9
    T = np.eye(4, dtype=np.float64)
    T[0, 3] = line[1]  # x
    T[1, 3] = line[2]  # y
    # Note, yawPitchRollToRot returns C_v_i, where v is vehicle/sensor frame and i is stationary frame
    # For SE(3) state, we want C_i_v (to match r_i loaded above), and so we take transpose
    if dim == 3:
        T[2, 3] = line[3]  # z
        T[:3, :3] = yawPitchRollToRot(line[9], line[8], line[7])
    elif dim == 2:
        T[:3, :3] = yawPitchRollToRot(
            line[9],
            np.round(line[8] / np.pi) * np.pi,
            np.round(line[7] / np.pi) * np.pi,
        )
    else:
        raise ValueError(
            "Invalid dim value in convert_line_to_pose. Use either 2 or 3."
        )
    time = int(line[0])
    return T, time

def read_traj_file_gt(path, dim=2):
    """Reads trajectory from a comma-separated file, see Boreas documentation for format
    Args:
        path (string): file path including file name
        T_ab (np.ndarray): 4x4 transformation matrix for calibration. Poses read are in frame 'b', output in frame 'a'
        dim (int): dimension for evaluation. Set to '3' for 3D or '2' for 2D
    Returns:
        (List[np.ndarray]): list of 4x4 poses
        (List[int]): list of times in microseconds
    """
    with open(path, "r") as f:
        lines = f.readlines()
    poses = []
    times = []
    for line in lines[1:]:
        pose, time = convert_line_to_pose(line, dim)
        poses.append(pose)
        times.append(time)  # microseconds
    return poses, times

def normalize_vector(v, scale=5):
    """Normalize a vector and scale its magnitude for consistent visualization."""
    return v / (np.linalg.norm(v) + 1e-8) * scale

def plot_transformation_sequence_gif(transformations, gif_filename="transformation_sequence.gif", delay=0.01):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    trajectory = np.array([pose[:3, 3] for pose in transformations])
    x_min, x_max = trajectory[:, 0].min(), trajectory[:, 0].max()
    y_min, y_max = trajectory[:, 1].min(), trajectory[:, 1].max()
    z_min, z_max = trajectory[:, 2].min(), trajectory[:, 2].max()

    ax.set_xlim([x_min - 20, x_max + 20])
    ax.set_ylim([y_min - 20, y_max + 20])
    ax.set_zlim([z_min - 5, z_max + 5])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    # set view angle such that only the top view is visible
    #ax.view_init(elev=90, azim=90)

    ax.view_init(elev=20, azim=30)

    frames = []
    ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], color='gray', alpha=0.5, label='Trajectory')

    # 初始化记录上一个绘制元素
    prev_artists = []

    for frame_number, transformation in tqdm(enumerate(transformations), total=len(transformations), desc="Processing frames"):
        # 清除上一帧所有元素
        for artist in prev_artists:
            artist.remove()
        prev_artists = []

        # Global axes
        prev_artists.append(ax.quiver(0, 0, 0, 1, 0, 0, color='r', alpha=0.3))
        prev_artists.append(ax.quiver(0, 0, 0, 0, 1, 0, color='g', alpha=0.3))
        prev_artists.append(ax.quiver(0, 0, 0, 0, 0, 1, color='b', alpha=0.3))

        origin = transformation[:3, 3]
        x_axis = normalize_vector(transformation[:3, 0])
        y_axis = normalize_vector(transformation[:3, 1])
        z_axis = normalize_vector(transformation[:3, 2], 1)

        scale = 3
        prev_artists.append(ax.quiver(*origin, *(x_axis * scale), color='r', alpha=0.8))
        prev_artists.append(ax.quiver(*origin, *(y_axis * scale), color='g', alpha=0.8))
        prev_artists.append(ax.quiver(*origin, *(z_axis * scale), color='b', alpha=0.8))

        text = ax.text2D(0.05, 0.95, f"Frame: {frame_number + 1}", transform=ax.transAxes, fontsize=12, color='blue')
        prev_artists.append(text)

        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        frames.append(Image.open(buf))

    frames[0].save(
        gif_filename,
        save_all=True,
        append_images=frames[1:],
        duration=delay * 1000,
        loop=0
    )
    print(f"Saved GIF as {gif_filename}")
    plt.close(fig)



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Process project and sequence arguments.")
    
    # Add arguments
    parser.add_argument('--project_folder', type=str, default='/mnt/new_ssd1/jiarui/boreas',
                        help='Path to the project folder')
    parser.add_argument('--sequence', type=str, default='boreas-2020-12-18-13-44',
                        help='Sequence name or ID')
    parser.add_argument('--dimension', type=int, default=3)
    args = parser.parse_args()
    project_folder = args.project_folder
    sequence = args.sequence

    filepath = os.path.join(project_folder, sequence,'applanix','radar_poses.csv')

    poses, times = read_traj_file_gt(filepath, dim=args.dimension)

    # save poses to numpy
    poses = np.array(poses)
    times = np.array(times)
    print(poses.shape)
    print(times.shape)

    if not os.path.exists(os.path.join(project_folder, sequence,'LidarRadarGuide')):
        os.makedirs(os.path.join(project_folder, sequence,'LidarRadarGuide'))

    # save to numpy
    np.save(os.path.join(project_folder, sequence,'LidarRadarGuide', f'radar_poses_{args.dimension}.npy'), poses)
    np.save(os.path.join(project_folder, sequence,'LidarRadarGuide', 'radar_times.npy'), times)
    #plot_transformation_sequence_gif(poses, gif_filename=os.path.join(project_folder, sequence, 'radar_poses.gif'), delay=0.01)

    # Save lidar poses as well
    lidar_pose_path = os.path.join(project_folder, sequence,'applanix','lidar_poses.csv')
    poses, times = read_traj_file_gt(lidar_pose_path, args.dimension)
    np.save(os.path.join(project_folder, sequence,'LidarRadarGuide', f'lidar_poses_{args.dimension}.npy'), poses)
    np.save(os.path.join(project_folder, sequence,'LidarRadarGuide', 'lidar_times.npy'), times)
    #plot_transformation_sequence_gif(poses, gif_filename=os.path.join(project_folder, sequence, 'lidar_poses.gif'), delay=0.01)

    # save camera poses as well
    camera_pose_path = os.path.join(project_folder, sequence,'applanix','camera_poses.csv')
    poses, times = read_traj_file_gt(camera_pose_path, args.dimension)
    np.save(os.path.join(project_folder, sequence,'LidarRadarGuide', f'camera_poses_{args.dimension}.npy'), poses)
    np.save(os.path.join(project_folder, sequence,'LidarRadarGuide', 'camera_times.npy'), times)


    ## Script:
    # python get_3d_radar_poses.py --sequence boreas-2021-01-26-11-22


   