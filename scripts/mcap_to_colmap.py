"""MCAP ROS2 bag -> COLMAP format converter for gsplat 3DGS training."""

import io
import struct
import argparse
from pathlib import Path

import yaml
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation, Slerp
from mcap_ros2.reader import read_ros2_messages


def load_config(config_path):
    """Load YAML config file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def config_from_camera_name(camera_name):
    """Generate config dict from camera name using default topic naming convention."""
    return {
        "camera": {
            "image_topic": f"/sensing/camera/{camera_name}/image_raw/compressed",
            "camera_info_topic": f"/sensing/camera/{camera_name}/camera_info",
        },
        "pose": {
            "topic": "/localization/kinematic_state",
        },
        "tf_static": {
            "chain": [
                "base_link",
                "sensor_kit_base_link",
                f"{camera_name}/camera_link",
                f"{camera_name}/camera_optical_link",
            ],
        },
    }


def collect_tf_static(mcap_path, tf_chain):
    """Collect static TF transforms and chain them."""
    tf_map = {}
    for msg in read_ros2_messages(str(mcap_path), topics=["/tf_static"]):
        for t in msg.ros_msg.transforms:
            p = t.header.frame_id
            c = t.child_frame_id
            trans = np.array([
                t.transform.translation.x,
                t.transform.translation.y,
                t.transform.translation.z,
            ])
            quat = np.array([
                t.transform.rotation.x,
                t.transform.rotation.y,
                t.transform.rotation.z,
                t.transform.rotation.w,
            ])
            tf_map[(p, c)] = (trans, quat)

    # Chain transforms
    total_t = np.zeros(3)
    total_R = np.eye(3)
    for i in range(len(tf_chain) - 1):
        key = (tf_chain[i], tf_chain[i + 1])
        if key not in tf_map:
            raise KeyError(f"TF not found: {tf_chain[i]} -> {tf_chain[i+1]}")
        t, q = tf_map[key]
        R = Rotation.from_quat(q).as_matrix()
        total_t = total_t + total_R @ t
        total_R = total_R @ R

    return total_t, total_R


def collect_poses(mcap_path, pose_topic):
    """Collect vehicle poses from an Odometry topic."""
    poses = []
    for msg in read_ros2_messages(str(mcap_path), topics=[pose_topic]):
        odom = msg.ros_msg
        stamp = odom.header.stamp.sec + odom.header.stamp.nanosec * 1e-9
        pos = np.array([
            odom.pose.pose.position.x,
            odom.pose.pose.position.y,
            odom.pose.pose.position.z,
        ])
        quat = np.array([
            odom.pose.pose.orientation.x,
            odom.pose.pose.orientation.y,
            odom.pose.pose.orientation.z,
            odom.pose.pose.orientation.w,
        ])
        poses.append((stamp, pos, quat))
    poses.sort(key=lambda x: x[0])
    return poses


def interpolate_pose(poses, query_stamp):
    """Interpolate vehicle pose at a given timestamp using linear + slerp."""
    stamps = [p[0] for p in poses]

    if query_stamp <= stamps[0]:
        return poses[0][1], poses[0][2]
    if query_stamp >= stamps[-1]:
        return poses[-1][1], poses[-1][2]

    idx = np.searchsorted(stamps, query_stamp) - 1
    idx = max(0, min(idx, len(poses) - 2))

    t0, pos0, q0 = poses[idx]
    t1, pos1, q1 = poses[idx + 1]

    alpha = (query_stamp - t0) / (t1 - t0) if t1 != t0 else 0.0
    pos = pos0 + alpha * (pos1 - pos0)

    rots = Rotation.from_quat([q0, q1])
    slerp = Slerp([0, 1], rots)
    quat = slerp(alpha).as_quat()

    return pos, quat


def extract_images_and_stamps(mcap_path, image_topic, output_dir, prefix=""):
    """Extract all images from a camera topic, return list of (filename, stamp).

    Args:
        prefix: Optional prefix for filenames to avoid collisions in multi-camera setup.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for msg in read_ros2_messages(str(mcap_path), topics=[image_topic]):
        ci = msg.ros_msg
        stamp = ci.header.stamp.sec + ci.header.stamp.nanosec * 1e-9
        filename = f"{prefix}{stamp:.6f}.jpg"
        img = Image.open(io.BytesIO(bytes(ci.data)))
        img.save(output_dir / filename)
        results.append((filename, stamp))
    return results


def get_camera_intrinsics(mcap_path, camera_info_topic):
    """Get camera intrinsics from camera_info topic."""
    for msg in read_ros2_messages(str(mcap_path), topics=[camera_info_topic]):
        ci = msg.ros_msg
        K = ci.k
        return {
            "width": ci.width,
            "height": ci.height,
            "fx": K[0],
            "fy": K[4],
            "cx": K[2],
            "cy": K[5],
        }
    raise RuntimeError(f"No camera_info found on {camera_info_topic}")


# ---- COLMAP binary writers ----


def write_cameras_bin(path, cameras):
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(cameras)))
        for cam in cameras:
            f.write(struct.pack("<I", cam["id"]))
            f.write(struct.pack("<i", cam["model"]))
            f.write(struct.pack("<Q", cam["width"]))
            f.write(struct.pack("<Q", cam["height"]))
            for p in cam["params"]:
                f.write(struct.pack("<d", p))


def write_images_bin(path, images):
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(images)))
        for img in images:
            f.write(struct.pack("<I", img["id"]))
            f.write(struct.pack("<d", img["qw"]))
            f.write(struct.pack("<d", img["qx"]))
            f.write(struct.pack("<d", img["qy"]))
            f.write(struct.pack("<d", img["qz"]))
            f.write(struct.pack("<d", img["tx"]))
            f.write(struct.pack("<d", img["ty"]))
            f.write(struct.pack("<d", img["tz"]))
            f.write(struct.pack("<I", img["camera_id"]))
            f.write(img["name"].encode("utf-8"))
            f.write(b"\x00")
            f.write(struct.pack("<Q", 0))


def write_points3d_bin(path, points=None):
    with open(path, "wb") as f:
        if points is None:
            points = []
        f.write(struct.pack("<Q", len(points)))
        for pt in points:
            f.write(struct.pack("<Q", pt["id"]))
            f.write(struct.pack("<3d", *pt["xyz"]))
            f.write(struct.pack("<3B", *pt["rgb"]))
            f.write(struct.pack("<d", pt["error"]))
            f.write(struct.pack("<Q", 0))


def generate_initial_points(colmap_images, num_points_per_camera=10, distances=(5, 10, 20)):
    """Generate initial 3D points in front of each camera."""
    points = []
    point_id = 1
    rng = np.random.default_rng(42)

    for img in colmap_images:
        qw, qx, qy, qz = img["qw"], img["qx"], img["qy"], img["qz"]
        tx, ty, tz = img["tx"], img["ty"], img["tz"]

        R_w2c = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
        t_w2c = np.array([tx, ty, tz])

        cam_center = -R_w2c.T @ t_w2c
        cam_forward = R_w2c.T @ np.array([0, 0, 1])

        for d in distances:
            for _ in range(num_points_per_camera // len(distances)):
                offset = rng.normal(0, d * 0.1, 3)
                pt = cam_center + cam_forward * d + offset
                points.append({
                    "id": point_id,
                    "xyz": pt.tolist(),
                    "rgb": (128, 128, 128),
                    "error": 1.0,
                })
                point_id += 1

    return points


def process_single_camera(cfg, camera_id, mcap_path, tf_mcap_path, poses, images_dir, prefix=""):
    """Process one camera: collect TF, intrinsics, extract images, compute world poses.

    Returns (colmap_camera, cam_world_poses) where cam_world_poses is a list of
    (filename, cam_world_pos, cam_world_R, camera_id).
    """
    image_topic = cfg["camera"]["image_topic"]
    info_topic = cfg["camera"]["camera_info_topic"]
    tf_chain = cfg["tf_static"]["chain"]

    print(f"\n  [{prefix or f'camera {camera_id}'}]")
    print(f"    image_topic: {image_topic}")
    print(f"    tf_chain: {' -> '.join(tf_chain)}")

    # TF
    cam_t, cam_R = collect_tf_static(tf_mcap_path, tf_chain)
    print(f"    TF offset: t={cam_t}, euler={Rotation.from_matrix(cam_R).as_euler('xyz', degrees=True)}")

    # Intrinsics
    intrinsics = get_camera_intrinsics(mcap_path, info_topic)
    print(f"    Intrinsics: {intrinsics['width']}x{intrinsics['height']}, fx={intrinsics['fx']:.1f}")

    colmap_camera = {
        "id": camera_id,
        "model": 1,  # PINHOLE
        "width": intrinsics["width"],
        "height": intrinsics["height"],
        "params": [intrinsics["fx"], intrinsics["fy"], intrinsics["cx"], intrinsics["cy"]],
    }

    # Extract images (with prefix to avoid filename collisions)
    image_list = extract_images_and_stamps(mcap_path, image_topic, images_dir, prefix=prefix)
    print(f"    Extracted {len(image_list)} images")

    # Compute world poses
    cam_world_poses = []
    for filename, stamp in image_list:
        vehicle_pos, vehicle_quat = interpolate_pose(poses, stamp)
        vehicle_R = Rotation.from_quat(vehicle_quat).as_matrix()
        cam_world_pos = vehicle_pos + vehicle_R @ cam_t
        cam_world_R = vehicle_R @ cam_R
        cam_world_poses.append((filename, cam_world_pos, cam_world_R, camera_id))

    return colmap_camera, cam_world_poses


def main():
    parser = argparse.ArgumentParser(description="MCAP to COLMAP converter")
    parser.add_argument("--mcap", required=True, help="Path to main MCAP bag file")
    parser.add_argument("--tf-mcap", default=None,
                        help="Path to MCAP bag containing tf_static (if different from --mcap)")
    parser.add_argument("--output", required=True, help="Output directory for COLMAP data")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--config", nargs="+", help="Path(s) to YAML config file(s)")
    group.add_argument("--camera", nargs="+", help="Camera name(s) (uses default topic naming convention)")

    args = parser.parse_args()

    # Build list of configs
    if args.config:
        configs = [load_config(c) for c in args.config]
        # Derive camera labels from config filenames
        camera_labels = [Path(c).stem for c in args.config]
    else:
        configs = [config_from_camera_name(c) for c in args.camera]
        camera_labels = list(args.camera)

    multi_camera = len(configs) > 1

    mcap_path = Path(args.mcap)
    tf_mcap_path = Path(args.tf_mcap) if args.tf_mcap else mcap_path
    output_dir = Path(args.output)
    images_dir = output_dir / "images"

    print(f"{'Multi-camera' if multi_camera else 'Single-camera'} mode: {', '.join(camera_labels)}")

    # Collect poses (shared across all cameras)
    pose_topic = configs[0]["pose"]["topic"]
    print(f"\n[1] Collecting poses from {pose_topic}...")
    poses = collect_poses(mcap_path, pose_topic)
    print(f"  {len(poses)} poses, time range: {poses[0][0]:.3f} - {poses[-1][0]:.3f}")

    # Process each camera
    print(f"\n[2] Processing cameras...")
    all_cameras = []
    all_world_poses = []
    for i, (cfg, label) in enumerate(zip(configs, camera_labels)):
        prefix = f"{label}_" if multi_camera else ""
        camera_id = i + 1
        colmap_cam, cam_world_poses = process_single_camera(
            cfg, camera_id, mcap_path, tf_mcap_path, poses, images_dir, prefix=prefix,
        )
        all_cameras.append(colmap_cam)
        all_world_poses.extend(cam_world_poses)

    # Center the scene using all camera positions
    print(f"\n[3] Writing COLMAP files...")
    all_positions = np.array([p[1] for p in all_world_poses])
    scene_center = all_positions.mean(axis=0)
    print(f"  Scene center (map coords): {scene_center}")
    print(f"  Centering scene to origin...")

    sparse_dir = output_dir / "sparse" / "0"
    sparse_dir.mkdir(parents=True, exist_ok=True)

    write_cameras_bin(sparse_dir / "cameras.bin", all_cameras)

    colmap_images = []
    for idx, (filename, cam_world_pos, cam_world_R, camera_id) in enumerate(all_world_poses):
        cam_centered_pos = cam_world_pos - scene_center
        R_w2c = cam_world_R.T
        t_w2c = -R_w2c @ cam_centered_pos

        q = Rotation.from_matrix(R_w2c).as_quat()  # [x, y, z, w]
        qw, qx, qy, qz = q[3], q[0], q[1], q[2]

        colmap_images.append({
            "id": idx + 1,
            "qw": qw, "qx": qx, "qy": qy, "qz": qz,
            "tx": t_w2c[0], "ty": t_w2c[1], "tz": t_w2c[2],
            "camera_id": camera_id,
            "name": filename,
        })

    write_images_bin(sparse_dir / "images.bin", colmap_images)

    init_points = generate_initial_points(colmap_images)
    write_points3d_bin(sparse_dir / "points3D.bin", init_points)

    print(f"\nDone! COLMAP data written to {output_dir}")
    print(f"  cameras.bin: {len(all_cameras)} camera(s)")
    print(f"  images.bin:  {len(colmap_images)} image(s)")
    print(f"  points3D.bin: {len(init_points)} points (generated from camera views)")


if __name__ == "__main__":
    main()
