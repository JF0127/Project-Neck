"""RPY <-> 旋转矩阵（数据集约定 R = Ry(yaw) @ Rx(pitch) @ Rz(roll)）。

- rpy 轴序 [roll, pitch, yaw]，弧度。
- 与 processed_dataset/FIELDS.md 的约定一致（用于重新相对化轨迹段）。
"""
from __future__ import annotations

import numpy as np


def rpy_to_matrix(rpy: np.ndarray) -> np.ndarray:
    """RPY -> 旋转矩阵。输入 [..., 3] -> 输出 [..., 3, 3]。"""
    rpy = np.asarray(rpy, dtype=np.float64)
    roll, pitch, yaw = rpy[..., 0], rpy[..., 1], rpy[..., 2]
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    Rz = np.stack([np.stack([cr, -sr, np.zeros_like(cr)], -1),
                   np.stack([sr, cr, np.zeros_like(cr)], -1),
                   np.stack([np.zeros_like(cr), np.zeros_like(cr), np.ones_like(cr)], -1)], -2)
    Rx = np.stack([np.stack([np.ones_like(cp), np.zeros_like(cp), np.zeros_like(cp)], -1),
                   np.stack([np.zeros_like(cp), cp, -sp], -1),
                   np.stack([np.zeros_like(cp), sp, cp], -1)], -2)
    Ry = np.stack([np.stack([cy, np.zeros_like(cy), sy], -1),
                   np.stack([np.zeros_like(cy), np.ones_like(cy), np.zeros_like(cy)], -1),
                   np.stack([-sy, np.zeros_like(cy), cy], -1)], -2)
    return np.matmul(Ry, np.matmul(Rx, Rz))


def matrix_to_rpy(R: np.ndarray) -> np.ndarray:
    """旋转矩阵 -> RPY。输入 [..., 3, 3] -> 输出 [..., 3]，弧度，与 rpy_to_matrix 互逆。"""
    R = np.asarray(R, dtype=np.float64)
    pitch = -np.arcsin(np.clip(R[..., 1, 2], -1.0, 1.0))
    yaw = np.arctan2(R[..., 0, 2], R[..., 2, 2])
    roll = np.arctan2(R[..., 1, 0], R[..., 1, 1])
    return np.stack([roll, pitch, yaw], axis=-1)


def relative_segment(abs_rpy: np.ndarray, f0: int, f1: int) -> np.ndarray:
    """轨迹段重新相对化：R_rel[t] = R(abs[f0])^T @ R(abs[t])，分解为 RPY。

    与数据集 relative RPY 的生成方式一致（相对段起点姿态，保证首帧 ≈ 0）。
    """
    abs_rpy = np.asarray(abs_rpy, dtype=np.float64)
    seg = abs_rpy[f0:f1]
    if seg.shape[0] == 0:
        return np.zeros((0, 3))
    R0 = rpy_to_matrix(abs_rpy[f0])           # [3,3]
    R_t = rpy_to_matrix(seg)                  # [n,3,3]
    rel = matrix_to_rpy(np.matmul(R0.T, R_t))  # [n,3]
    return rel.astype(np.float32)
