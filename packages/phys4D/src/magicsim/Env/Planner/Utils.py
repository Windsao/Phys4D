import torch


def quat_normalize(q):
    return q / torch.linalg.norm(q, dim=-1, keepdim=True).clamp_min(1e-8)


def quat_angle_between(q1, q2):
    """Returns the rotation angle (radians) between two quaternions"""
    q1 = quat_normalize(q1)
    q2 = quat_normalize(q2)
    dot = torch.sum(q1 * q2, dim=-1).abs().clamp(-1.0, 1.0)
    return 2.0 * torch.arccos(dot)


def quat_slerp_batch(q0, q1, t):
    """
    Batch slerp:
        q0, q1: [N, 4]
        t:     [N, 1] or [N]
    Returns:
        [N, 4]
    """
    q0 = quat_normalize(q0)
    q1 = quat_normalize(q1)

    if t.ndim == 1:
        t = t.unsqueeze(-1)

    dot = torch.sum(q0 * q1, dim=-1, keepdim=True)
    mask_neg = dot < 0.0
    q1 = torch.where(mask_neg, -q1, q1)
    dot = torch.sum(q0 * q1, dim=-1, keepdim=True).clamp(-1.0, 1.0)

    close = dot > 0.9995
    if torch.any(close):
        q_lin = quat_normalize((1.0 - t) * q0 + t * q1)
        if torch.all(close):
            return q_lin

    omega = torch.arccos(dot)
    sin_omega = torch.sin(omega).clamp_min(1e-8)

    w0 = torch.sin((1.0 - t) * omega) / sin_omega
    w1 = torch.sin(t * omega) / sin_omega

    q = w0 * q0 + w1 * q1
    return quat_normalize(q)


def quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Quaternion multiplication: q = q1 * q2, with q = [w, x, y, z]."""
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return torch.stack([w, x, y, z], dim=-1)


def quat_conj(q: torch.Tensor) -> torch.Tensor:
    """Quaternion conjugate."""
    w, x, y, z = q.unbind(-1)
    return torch.stack([w, -x, -y, -z], dim=-1)


def quat_inv(q: torch.Tensor) -> torch.Tensor:
    """Quaternion inverse."""
    return quat_conj(q) / torch.sum(q * q, dim=-1, keepdim=True).clamp_min(1e-8)


def quat_error_to_rotvec(q_err: torch.Tensor) -> torch.Tensor:
    """
    Convert quaternion error q_err to rotation vector (axis * angle).
    q_err = q_target * inv(q_current).
    """
    q_err = quat_normalize(q_err)
    w, x, y, z = q_err.unbind(-1)
    w = w.clamp(-1.0, 1.0)
    angle = 2.0 * torch.arccos(w)
    s = torch.sqrt(torch.clamp(1.0 - w * w, min=0.0))

    small = s < 1e-6
    axis = torch.zeros_like(q_err[..., 1:4])
    axis[~small] = torch.stack([x, y, z], dim=-1)[~small] / s[~small].unsqueeze(-1)
    rot_vec = axis * angle.unsqueeze(-1)
    return rot_vec


def integrate_quat_with_omega(
    q: torch.Tensor, omega: torch.Tensor, dt: float
) -> torch.Tensor:
    """
    Integrate quaternion with angular velocity omega over dt.
    omega: [N, 3] (rad/s)
    Approximate: q_{t+dt} = q_t + 0.5 * q_t ⊗ [0, omega] * dt
    """
    w, x, y, z = q.unbind(-1)
    wx, wy, wz = omega.unbind(-1)

    dq_w = -0.5 * (x * wx + y * wy + z * wz)
    dq_x = 0.5 * (w * wx + y * wz - z * wy)
    dq_y = 0.5 * (w * wy - x * wz + z * wx)
    dq_z = 0.5 * (w * wz + x * wy - y * wx)

    dq = torch.stack([dq_w, dq_x, dq_y, dq_z], dim=-1) * dt
    q_new = q + dq
    return quat_normalize(q_new)
