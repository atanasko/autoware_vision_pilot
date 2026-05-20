import cv2
import numpy as np
from pathlib import Path


def find_homography_C_matrix():
    # w, h = 1024, 512
    # X, Y pair of points
    canonical_world_pts = np.array([
        [15, 5, 1],
        [150, 5, 1],
        [15, -5, 1],
        [150, -5, 1]
    ], dtype=np.float32)

    # Camera homography transform matrix
    H = np.array([[0.000604414444603015, -0.002459315932924031, -6.056269548874612],
                  [0.0032928037256633124, 6.577304929754011e-05, -3.1934736278854983],
                  [1.679596119965616e-05, -0.00157134665454438, 1.0]], dtype=np.float32)

    # DO NOT MODIFY!
    # VisionPilot homography transform matrix
    V = np.array([[0.00209514907, -0.000941721466, -9.24906396],
                  [0.00662758637, -0.000352940531, -3.33396502],
                  [0.000120077371, -0.00411343505, 1.0]], dtype=np.float32)

    H_inv = np.linalg.inv(H)
    V_inv = np.linalg.inv(V)

    # Backproject canonical world points through the inverse od the Homography transform of the user (H inverse)
    uv_pts = H_inv @ canonical_world_pts.T
    uv_pts = uv_pts / uv_pts[2]  # normalize
    uv_pts = uv_pts[:2].T  # (N, 2)

    # Backproject canonical world points through the inverse od the standard Homography transform of VisionPilot (V inverse)
    pq_pts = V_inv @ canonical_world_pts.T
    pq_pts = pq_pts / pq_pts[2]  # normalize
    pq_pts = pq_pts[:2].T  # (N, 2)

    # Calculate C matrix that will transform (u, v) -> (p, q)
    C, _ = cv2.findHomography(uv_pts, pq_pts, method=0)

    return C


def create_calibration_config():
    build_dir = Path(__file__).resolve().parents[1] / "build"
    config_dir = build_dir / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    homography_calibration_path = str(config_dir / "homography_C_matrix.yaml")

    C = find_homography_C_matrix().astype(np.float32)

    fs = cv2.FileStorage(homography_calibration_path, cv2.FILE_STORAGE_WRITE)
    fs.write("C", C)
    fs.release()


if __name__ == "__main__":
    print("Create calibration config!")
    create_calibration_config()
