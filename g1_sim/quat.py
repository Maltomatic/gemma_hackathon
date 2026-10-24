"""Minimal quaternion helpers. All quaternions are (w, x, y, z), MuJoCo / Isaac Lab order."""
from __future__ import annotations

import numpy as np


def mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ])


def conj(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]])


def normalize(q: np.ndarray) -> np.ndarray:
    return q / max(np.linalg.norm(q), 1e-12)


def rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector v by unit quaternion q."""
    w, x, y, z = q
    u = np.array([x, y, z])
    return v + 2.0 * np.cross(u, np.cross(u, v) + w * v)


def from_axis_angle(angle: float, axis: np.ndarray) -> np.ndarray:
    axis = axis / max(np.linalg.norm(axis), 1e-12)
    s = np.sin(angle / 2.0)
    return normalize(np.array([np.cos(angle / 2.0), *(axis * s)]))


def heading(q: np.ndarray) -> float:
    """Yaw angle of q: direction of the rotated +x axis projected onto the ground plane."""
    d = rotate(q, np.array([1.0, 0.0, 0.0]))
    return float(np.arctan2(d[1], d[0]))


def heading_quat(q: np.ndarray) -> np.ndarray:
    """Yaw-only quaternion with the same heading as q."""
    return from_axis_angle(heading(q), np.array([0.0, 0.0, 1.0]))


def heading_quat_inv(q: np.ndarray) -> np.ndarray:
    return from_axis_angle(-heading(q), np.array([0.0, 0.0, 1.0]))


def to_rotmat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = normalize(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def to_6d(q: np.ndarray) -> np.ndarray:
    """First two columns of the rotation matrix, flattened row-wise (SONIC convention)."""
    return to_rotmat(q)[:, :2].reshape(-1)


def slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    q0 = normalize(q0)
    q1 = normalize(q1)
    d = float(np.dot(q0, q1))
    if d < 0.0:
        q1 = -q1
        d = -d
    if d > 0.9995:
        return normalize(q0 + t * (q1 - q0))
    th0 = np.arccos(d)
    th = th0 * t
    s0 = np.sin(th0 - th) / np.sin(th0)
    s1 = np.sin(th) / np.sin(th0)
    return normalize(s0 * q0 + s1 * q1)


def xyzw_to_wxyz(q: np.ndarray) -> np.ndarray:
    return np.asarray(q)[..., [3, 0, 1, 2]]
