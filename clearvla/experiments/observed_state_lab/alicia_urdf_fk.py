from __future__ import annotations

"""Small, dependency-free URDF forward-kinematics helper for Alicia-D analysis.

This module intentionally avoids importing the Alicia-D SDK, synriard, or RoboCore.
It implements just enough URDF FK for offline evaluation utilities:

    q_arm[6] -> T_base_tool0[4, 4]

It supports fixed, revolute/continuous, and prismatic joints on the selected chain.
Meshes, inertial tags, transmissions, and hardware APIs are deliberately ignored.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence
import math
import xml.etree.ElementTree as ET

import numpy as np


@dataclass(frozen=True)
class URDFJoint:
    name: str
    joint_type: str
    parent: str
    child: str
    xyz: np.ndarray
    rpy: np.ndarray
    axis: np.ndarray

    @property
    def is_active(self) -> bool:
        return self.joint_type in {"revolute", "continuous", "prismatic"}


@dataclass(frozen=True)
class FKResult:
    transform: np.ndarray
    position: np.ndarray
    rotation: np.ndarray


def default_alicia_urdf_path(variant: str = "gripper_50mm") -> Path:
    """Return the bundled fallback Alicia-D v5.6 URDF path.

    The user's hardware may be D650T/750T/V5.5.  This bundled model is only a
    convenient offline fallback.  CLI tools should also expose --urdf-path so a
    precise hardware URDF can be supplied later.
    """
    here = Path(__file__).resolve()
    clearvla_root = here.parents[2]
    return clearvla_root / "assets" / "robots" / "alicia_d" / "v5_6" / f"Alicia_D_v5_6_{variant}.urdf"


def _parse_vec(text: str | None, default: Sequence[float]) -> np.ndarray:
    if not text:
        return np.asarray(default, dtype=np.float64)
    return np.asarray([float(x) for x in text.split()], dtype=np.float64)


def _origin_matrix(xyz: np.ndarray, rpy: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = rpy_matrix(float(rpy[0]), float(rpy[1]), float(rpy[2]))
    T[:3, 3] = xyz.astype(np.float64)
    return T


def rpy_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """URDF fixed-axis RPY rotation: Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=np.float64)
    Ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=np.float64)
    Rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return Rz @ Ry @ Rx


def axis_angle_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    norm = float(np.linalg.norm(axis))
    if norm < 1e-12:
        return np.eye(3, dtype=np.float64)
    x, y, z = axis / norm
    c, s = math.cos(float(angle)), math.sin(float(angle))
    C = 1.0 - c
    return np.array([
        [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
    ], dtype=np.float64)


def rotation_angle(R: np.ndarray) -> float:
    tr = float(np.trace(R))
    cos_angle = max(-1.0, min(1.0, 0.5 * (tr - 1.0)))
    return float(math.acos(cos_angle))


def vector_cosine(a: np.ndarray, b: np.ndarray, eps: float = 1e-9) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    na = float(np.linalg.norm(a)); nb = float(np.linalg.norm(b))
    if na < eps and nb < eps:
        return 1.0
    if na < eps or nb < eps:
        return 0.0
    return float(np.dot(a, b) / max(na * nb, eps))


class URDFFKChain:
    """Forward-kinematics chain parsed from a URDF file."""

    def __init__(self, urdf_path: str | Path, *, base_link: str = "base_link", end_link: str = "tool0") -> None:
        self.urdf_path = Path(urdf_path)
        if not self.urdf_path.exists():
            raise FileNotFoundError(f"URDF not found: {self.urdf_path}")
        self.base_link = str(base_link)
        self.end_link = str(end_link)
        self.joints = parse_urdf_joints(self.urdf_path)
        self.chain = find_chain(self.joints, self.base_link, self.end_link)
        self.active_joint_names = tuple(j.name for j in self.chain if j.is_active)

    @property
    def dof(self) -> int:
        return len(self.active_joint_names)

    def forward(self, q: Sequence[float] | np.ndarray) -> np.ndarray:
        q_arr = np.asarray(q, dtype=np.float64).reshape(-1)
        if q_arr.shape[0] != self.dof:
            raise ValueError(f"expected q with {self.dof} active joints {self.active_joint_names}, got shape {q_arr.shape}")
        T = np.eye(4, dtype=np.float64)
        q_i = 0
        for joint in self.chain:
            T = T @ _origin_matrix(joint.xyz, joint.rpy)
            if joint.joint_type in {"revolute", "continuous"}:
                R = axis_angle_matrix(joint.axis, float(q_arr[q_i])); q_i += 1
                M = np.eye(4, dtype=np.float64); M[:3, :3] = R
                T = T @ M
            elif joint.joint_type == "prismatic":
                axis = joint.axis.astype(np.float64)
                axis = axis / max(float(np.linalg.norm(axis)), 1e-12)
                M = np.eye(4, dtype=np.float64); M[:3, 3] = axis * float(q_arr[q_i]); q_i += 1
                T = T @ M
            elif joint.joint_type == "fixed":
                pass
            else:
                raise ValueError(f"unsupported joint type {joint.joint_type!r} for {joint.name}")
        return T

    def result(self, q: Sequence[float] | np.ndarray) -> FKResult:
        T = self.forward(q)
        return FKResult(transform=T, position=T[:3, 3].copy(), rotation=T[:3, :3].copy())

    def positions(self, q_batch: np.ndarray) -> np.ndarray:
        q_batch = np.asarray(q_batch, dtype=np.float64)
        flat = q_batch.reshape(-1, q_batch.shape[-1])
        pos = np.stack([self.result(q).position for q in flat], axis=0)
        return pos.reshape(q_batch.shape[:-1] + (3,))

    def rotations(self, q_batch: np.ndarray) -> np.ndarray:
        q_batch = np.asarray(q_batch, dtype=np.float64)
        flat = q_batch.reshape(-1, q_batch.shape[-1])
        rot = np.stack([self.result(q).rotation for q in flat], axis=0)
        return rot.reshape(q_batch.shape[:-1] + (3, 3))


def parse_urdf_joints(path: str | Path) -> list[URDFJoint]:
    root = ET.parse(path).getroot()
    joints: list[URDFJoint] = []
    for elem in root.findall("joint"):
        name = elem.attrib["name"]
        joint_type = elem.attrib.get("type", "fixed")
        parent_el = elem.find("parent")
        child_el = elem.find("child")
        if parent_el is None or child_el is None:
            continue
        origin_el = elem.find("origin")
        axis_el = elem.find("axis")
        joints.append(URDFJoint(
            name=name,
            joint_type=joint_type,
            parent=parent_el.attrib["link"],
            child=child_el.attrib["link"],
            xyz=_parse_vec(origin_el.attrib.get("xyz") if origin_el is not None else None, (0.0, 0.0, 0.0)),
            rpy=_parse_vec(origin_el.attrib.get("rpy") if origin_el is not None else None, (0.0, 0.0, 0.0)),
            axis=_parse_vec(axis_el.attrib.get("xyz") if axis_el is not None else None, (0.0, 0.0, 0.0)),
        ))
    return joints


def find_chain(joints: Sequence[URDFJoint], base_link: str, end_link: str) -> list[URDFJoint]:
    by_parent: dict[str, list[URDFJoint]] = {}
    for joint in joints:
        by_parent.setdefault(joint.parent, []).append(joint)
    stack: list[tuple[str, list[URDFJoint]]] = [(base_link, [])]
    visited: set[str] = set()
    while stack:
        link, chain = stack.pop()
        if link == end_link:
            return chain
        if link in visited:
            continue
        visited.add(link)
        for joint in reversed(by_parent.get(link, [])):
            stack.append((joint.child, chain + [joint]))
    raise ValueError(f"no URDF chain from {base_link!r} to {end_link!r}")
