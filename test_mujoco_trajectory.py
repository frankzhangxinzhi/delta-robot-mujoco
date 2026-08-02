import math
import csv
import time
from pathlib import Path

import mujoco
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# Automated trajectory comparison for the delta robot model
# ============================================================
# This script runs the same pick-and-place task using:
#   1. smoothstep
#   2. trapezoidal
#   3. S-curve
#
# It creates four folders in the current working directory:
#   smoothstep/
#   trapezoidal/
#   S_curve/
#   trajectory_comparison/
#
# Each trajectory folder contains:
#   1. end-effector position plot
#   2. end-effector velocity plot
#   3. end-effector acceleration plot
#   4. joint angle plot
#   5. joint velocity plot
#   6. joint torque plot
#   7. tracking error plot
#   8. 3D end-effector path plot
#   plus a CSV log for that trajectory.
#
# The comparison folder contains:
#   1. peak torque comparison
#   2. peak velocity comparison
#   3. peak acceleration comparison
#   4. peak tracking error comparison
#   5. estimated minimum feasible cycle time comparison
#   plus a summary CSV.
# ============================================================


# =========================
# 1. User-adjustable settings
# =========================
XML_FILENAME = "delta_closed.xml"

TRAJECTORIES = [
    {"key": "smoothstep", "label": "Smoothstep", "folder": "smoothstep"},
    {"key": "trapezoidal", "label": "Trapezoidal", "folder": "trapezoidal"},
    {"key": "s_curve", "label": "S-curve", "folder": "S_curve"},
]

COMPARISON_FOLDER = Path("trajectory_comparison")

# Controller gains
Kp = 120.0
Kd = 14.0
tau_limit = 20.0

# Run two full pick-and-place cycles for the main saved plots.
TOTAL_CYCLES_MAIN = 2.0

# Feasible-cycle-time search.
# Moving segment durations are scaled; pick/place dwell durations stay unchanged.
RUN_MIN_TIME_SWEEP = True
TOTAL_CYCLES_FEASIBILITY = 1.0
MIN_MOTION_SCALE = 0.15
MAX_MOTION_SCALE = 3.00
FEASIBILITY_BISECTION_STEPS = 8

# The main minimum-cycle-time criterion is unsaturated torque demand <= tau_limit.
# Tracking error is still plotted and reported. You can optionally include it
# in the feasibility search by setting USE_TRACKING_ERROR_IN_FEASIBILITY = True.
USE_TRACKING_ERROR_IN_FEASIBILITY = False
FEASIBILITY_TRACKING_ERROR_LIMIT = 0.010  # m

# Trapezoidal velocity profile parameter.
# 0.25 means 25% acceleration, 50% constant velocity, 25% deceleration.
TRAPEZOID_ACCEL_FRACTION = 0.25

# Settling before each run.
SETTLE_TIME = 1.0


# =========================
# 2. Load MuJoCo model
# =========================
model = mujoco.MjModel.from_xml_path(XML_FILENAME)
dt = model.opt.timestep


# =========================
# 3. Find motor joint addresses
# =========================
motor_names = ["motor1", "motor2", "motor3"]

qpos_adr = []
qvel_adr = []

for name in motor_names:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)

    if jid == -1:
        raise ValueError(f"Cannot find joint named {name}. Check your XML joint names.")

    qpos_adr.append(model.jnt_qposadr[jid])
    qvel_adr.append(model.jnt_dofadr[jid])


# =========================
# 4. Find end-effector body
# =========================
ee_body_name = "platform"
ee_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, ee_body_name)

if ee_body_id == -1:
    raise ValueError(f"Cannot find body named {ee_body_name}. Check XML body names.")


# =========================
# 5. Delta robot geometry for inverse kinematics
# =========================
# These numbers match the current XML geometry.
# If the XML geometry changes later, update these constants too.
base_joint_pos = np.array([
    [0.13,   0.0,     0.75],
    [-0.065, 0.1126,  0.75],
    [-0.065, -0.1126, 0.75],
])

motor_axis = np.array([
    [0.0,    1.0,   0.0],
    [-0.866, -0.5,  0.0],
    [0.866, -0.5,   0.0],
])

upper_arm_local = np.array([
    [0.12,  0.0,     -0.17],
    [-0.06, 0.1039,  -0.17],
    [-0.06, -0.1039, -0.17],
])

# Platform connection-site radius = 0.07 m.
# Visual platform disk radius in XML = 0.08 m.
platform_site_local = np.array([
    [0.07,   0.0,     0.0],
    [-0.035, 0.0606,  0.0],
    [-0.035, -0.0606, 0.0],
])

upper_arm_length = math.sqrt(0.12**2 + 0.17**2)
lower_arm_length = 2.0 * upper_arm_length

q_min = -1.2
q_max = 0.8

for i in range(3):
    motor_axis[i] = motor_axis[i] / np.linalg.norm(motor_axis[i])


# =========================
# 6. Cartesian pick-and-place task
# =========================
# Task mission stays unchanged.
pick_high = np.array([-0.07, -0.06, 0.35])
pick_low  = np.array([-0.07, -0.06, 0.26])

place_high = np.array([0.08, 0.06, 0.35])
place_low  = np.array([0.08, 0.06, 0.26])

# Tuple format:
# (start_position, end_position, base_duration, segment_name)
TRAJECTORY_SEGMENTS_BASE = [
    (pick_high,   pick_low,    1.0, "move_down_to_pick"),
    (pick_low,    pick_low,    0.3, "pick_dwell"),
    (pick_low,    pick_high,   1.0, "lift_after_pick"),
    (pick_high,   place_high,  1.2, "move_to_place"),
    (place_high,  place_low,   1.0, "move_down_to_place"),
    (place_low,   place_low,   0.3, "place_dwell"),
    (place_low,   place_high,  1.0, "lift_after_place"),
    (place_high,  pick_high,   1.2, "return_to_pick"),
]


# =========================
# 7. IK helper functions
# =========================
def rotate_about_axis(v, axis, theta):
    """
    Rodrigues rotation formula.
    Rotates vector v about unit vector axis by theta radians.
    """
    return (
        v * math.cos(theta)
        + np.cross(axis, v) * math.sin(theta)
        + axis * np.dot(axis, v) * (1.0 - math.cos(theta))
    )


def elbow_position(arm_id, q):
    """
    World position of the elbow site for one arm at motor angle q.
    """
    return base_joint_pos[arm_id] + rotate_about_axis(
        upper_arm_local[arm_id],
        motor_axis[arm_id],
        q
    )


def ik_error_for_arm(arm_id, q, platform_pos_des):
    """
    Difference between actual elbow-to-platform-site distance
    and required lower arm length.
    """
    elbow = elbow_position(arm_id, q)
    platform_site = platform_pos_des + platform_site_local[arm_id]
    return np.linalg.norm(elbow - platform_site) - lower_arm_length


def solve_one_arm_ik_bruteforce(arm_id, platform_pos_des, q_hint):
    """
    Slower but robust backup IK solver.
    Used for initialization or when fast local IK fails.
    """
    qs = np.linspace(q_min, q_max, 151)
    fs = np.array([ik_error_for_arm(arm_id, q, platform_pos_des) for q in qs])

    roots = []

    for k in range(len(qs) - 1):
        f1 = fs[k]
        f2 = fs[k + 1]

        if abs(f1) < 1e-10:
            roots.append(qs[k])

        if f1 * f2 < 0:
            lo = qs[k]
            hi = qs[k + 1]
            flo = f1

            for _ in range(50):
                mid = 0.5 * (lo + hi)
                fmid = ik_error_for_arm(arm_id, mid, platform_pos_des)

                if flo * fmid <= 0:
                    hi = mid
                else:
                    lo = mid
                    flo = fmid

            roots.append(0.5 * (lo + hi))

    if len(roots) > 0:
        q_solution = min(roots, key=lambda q: abs(q - q_hint))
        residual = abs(ik_error_for_arm(arm_id, q_solution, platform_pos_des))
        return q_solution, residual

    idx = np.argmin(np.abs(fs))
    q_solution = qs[idx]
    residual = abs(fs[idx])
    return q_solution, residual


def solve_one_arm_ik_fast(arm_id, platform_pos_des, q_hint):
    """
    Fast local IK solver.
    It uses the previous timestep solution as the starting guess.
    If it fails, it falls back to a local/root search and then brute force.
    """
    q = float(np.clip(q_hint, q_min, q_max))

    best_q = q
    best_residual = abs(ik_error_for_arm(arm_id, q, platform_pos_des))

    for _ in range(15):
        f = ik_error_for_arm(arm_id, q, platform_pos_des)
        abs_f = abs(f)

        if abs_f < 1e-9:
            return q, abs_f, False

        if abs_f < best_residual:
            best_q = q
            best_residual = abs_f

        h = 1e-5
        q_plus = min(q + h, q_max)
        q_minus = max(q - h, q_min)

        if q_plus == q_minus:
            break

        f_plus = ik_error_for_arm(arm_id, q_plus, platform_pos_des)
        f_minus = ik_error_for_arm(arm_id, q_minus, platform_pos_des)
        df = (f_plus - f_minus) / (q_plus - q_minus)

        if abs(df) < 1e-10:
            break

        dq = -f / df
        dq = float(np.clip(dq, -0.08, 0.08))
        q_new = float(np.clip(q + dq, q_min, q_max))

        if abs(q_new - q) < 1e-12:
            break

        q = q_new

    if best_residual < 1e-6:
        return best_q, best_residual, False

    radii = [0.02, 0.05, 0.10, 0.20, 0.40, 0.80, 2.00]

    for radius in radii:
        lo = max(q_min, q_hint - radius)
        hi = min(q_max, q_hint + radius)

        qs = np.linspace(lo, hi, 31)
        fs = np.array([ik_error_for_arm(arm_id, q_test, platform_pos_des) for q_test in qs])

        local_roots = []

        for k in range(len(qs) - 1):
            f1 = fs[k]
            f2 = fs[k + 1]

            if abs(f1) < 1e-9:
                local_roots.append(qs[k])

            if f1 * f2 < 0:
                a = qs[k]
                b = qs[k + 1]
                fa = f1

                for _ in range(40):
                    mid = 0.5 * (a + b)
                    fm = ik_error_for_arm(arm_id, mid, platform_pos_des)

                    if fa * fm <= 0:
                        b = mid
                    else:
                        a = mid
                        fa = fm

                local_roots.append(0.5 * (a + b))

        if len(local_roots) > 0:
            q_solution = min(local_roots, key=lambda q_root: abs(q_root - q_hint))
            residual = abs(ik_error_for_arm(arm_id, q_solution, platform_pos_des))
            return q_solution, residual, True

    q_solution, residual = solve_one_arm_ik_bruteforce(
        arm_id,
        platform_pos_des,
        q_hint
    )

    return q_solution, residual, True


def inverse_kinematics_delta_fast(platform_pos_des, q_hint):
    """
    Convert desired platform XYZ position into desired motor angles.
    Uses fast local IK with fallback.
    """
    q_des = np.zeros(3)
    residuals = np.zeros(3)
    used_fallback = np.zeros(3, dtype=bool)

    for arm_id in range(3):
        q_des[arm_id], residuals[arm_id], used_fallback[arm_id] = solve_one_arm_ik_fast(
            arm_id,
            platform_pos_des,
            q_hint[arm_id]
        )

    return q_des, residuals, used_fallback


def inverse_kinematics_delta_bruteforce(platform_pos_des, q_hint):
    """
    Robust IK used for initial setup.
    """
    q_des = np.zeros(3)
    residuals = np.zeros(3)

    for arm_id in range(3):
        q_des[arm_id], residuals[arm_id] = solve_one_arm_ik_bruteforce(
            arm_id,
            platform_pos_des,
            q_hint[arm_id]
        )

    return q_des, residuals


# =========================
# 8. Trajectory profile functions
# =========================
def smoothstep_blend(u):
    """
    Cubic smoothstep.
    Zero velocity at the beginning and end of each segment.
    """
    u = np.clip(u, 0.0, 1.0)
    return 3.0 * u**2 - 2.0 * u**3


def trapezoidal_blend(u, accel_fraction=TRAPEZOID_ACCEL_FRACTION):
    """
    Normalized trapezoidal velocity profile.

    accel_fraction = 0.25 gives:
      25% acceleration
      50% constant velocity
      25% deceleration
    """
    u = float(np.clip(u, 0.0, 1.0))
    r = float(np.clip(accel_fraction, 1e-6, 0.5))
    vmax_norm = 1.0 / (1.0 - r)

    if u < r:
        return 0.5 * vmax_norm * u**2 / r

    if u <= 1.0 - r:
        return 0.5 * vmax_norm * r + vmax_norm * (u - r)

    return 1.0 - 0.5 * vmax_norm * (1.0 - u)**2 / r


def s_curve_blend(u):
    """
    Quintic S-curve / minimum-jerk blend.
    Zero velocity and zero acceleration at segment endpoints.
    """
    u = np.clip(u, 0.0, 1.0)
    return 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5


def blend_value(u, profile_key):
    if profile_key == "smoothstep":
        return smoothstep_blend(u)
    if profile_key == "trapezoidal":
        return trapezoidal_blend(u)
    if profile_key == "s_curve":
        return s_curve_blend(u)

    raise ValueError(f"Unknown trajectory profile: {profile_key}")


def build_scaled_segments(motion_duration_scale=1.0):
    """
    Scale only moving segments for cycle-time search.
    Dwell durations stay unchanged.
    """
    scaled_segments = []

    for p_start, p_end, duration, name in TRAJECTORY_SEGMENTS_BASE:
        is_dwell = np.linalg.norm(p_end - p_start) < 1e-12

        if is_dwell:
            scaled_duration = duration
        else:
            scaled_duration = duration * motion_duration_scale

        scaled_segments.append((p_start, p_end, scaled_duration, name))

    return scaled_segments


def get_cycle_time(segments):
    return sum(segment[2] for segment in segments)


def interpolate_segment(p_start, p_end, u, profile_key):
    return p_start + blend_value(u, profile_key) * (p_end - p_start)


def desired_platform_position(t, segments, cycle_time, profile_key):
    """
    Returns desired platform XYZ position at command time t.
    """
    tau = t % cycle_time
    elapsed = 0.0

    for p_start, p_end, duration, _name in segments:
        if tau <= elapsed + duration:
            if duration <= 0.0:
                return p_end.copy()

            u = (tau - elapsed) / duration
            return interpolate_segment(p_start, p_end, u, profile_key)

        elapsed += duration

    return segments[-1][1].copy()


# =========================
# 9. Numerical helper functions
# =========================
def finite_difference(values, dt_value):
    """
    Compute numerical derivative of an array sampled at constant dt.
    """
    values = np.asarray(values)

    if len(values) < 2:
        return np.zeros_like(values)

    edge_order = 2 if len(values) >= 3 else 1
    return np.gradient(values, dt_value, axis=0, edge_order=edge_order)


def vector_norm_rows(values):
    return np.linalg.norm(values, axis=1)


def safe_peak_abs(values):
    values = np.asarray(values)
    if values.size == 0:
        return float("nan")
    return float(np.max(np.abs(values)))


def safe_peak_norm(values):
    values = np.asarray(values)
    if values.size == 0:
        return float("nan")
    return float(np.max(vector_norm_rows(values)))


def clip_torque(tau):
    return np.clip(tau, -tau_limit, tau_limit)


def set_3d_axes_equal(ax, xs, ys, zs):
    """
    Make 3D plot axes roughly equal scale.
    """
    x_min, x_max = np.min(xs), np.max(xs)
    y_min, y_max = np.min(ys), np.max(ys)
    z_min, z_max = np.min(zs), np.max(zs)

    x_mid = 0.5 * (x_min + x_max)
    y_mid = 0.5 * (y_min + y_max)
    z_mid = 0.5 * (z_min + z_max)

    max_range = max(x_max - x_min, y_max - y_min, z_max - z_min)
    if max_range <= 0:
        max_range = 0.1

    radius = 0.5 * max_range
    ax.set_xlim(x_mid - radius, x_mid + radius)
    ax.set_ylim(y_mid - radius, y_mid + radius)
    ax.set_zlim(z_mid - radius, z_mid + radius)


# =========================
# 10. Precompute desired trajectory and IK
# =========================
def precompute_desired_trajectory(profile_key, motion_duration_scale=1.0, total_cycles=TOTAL_CYCLES_MAIN):
    segments = build_scaled_segments(motion_duration_scale)
    cycle_time = get_cycle_time(segments)
    total_time = total_cycles * cycle_time
    n_steps = int(math.ceil(total_time / dt))

    time_cmd_array = np.zeros(n_steps)
    ee_des_array = np.zeros((n_steps, 3))
    q_des_array = np.zeros((n_steps, 3))
    ik_residual_array = np.zeros((n_steps, 3))

    q_settle, settle_residual = inverse_kinematics_delta_bruteforce(
        pick_high,
        np.zeros(3)
    )

    q_hint = q_settle.copy()
    fallback_count = 0

    for k in range(n_steps):
        t_cmd = k * dt
        ee_des = desired_platform_position(t_cmd, segments, cycle_time, profile_key)

        q_des, ik_residual, used_fallback = inverse_kinematics_delta_fast(
            ee_des,
            q_hint
        )

        fallback_count += int(np.any(used_fallback))

        time_cmd_array[k] = t_cmd
        ee_des_array[k, :] = ee_des
        q_des_array[k, :] = q_des
        ik_residual_array[k, :] = ik_residual

        q_hint = q_des.copy()

    return {
        "segments": segments,
        "cycle_time": cycle_time,
        "total_time": total_time,
        "n_steps": n_steps,
        "time_cmd": time_cmd_array,
        "ee_des": ee_des_array,
        "q_des": q_des_array,
        "ik_residual": ik_residual_array,
        "q_settle": q_settle,
        "settle_residual": settle_residual,
        "fallback_count": fallback_count,
    }


# =========================
# 11. Simulation
# =========================
def settle_robot(data, q_settle):
    """
    Settles the closed-chain robot to the initial motor pose.
    """
    while data.time < SETTLE_TIME:
        for i in range(3):
            q = data.qpos[qpos_adr[i]]
            qd = data.qvel[qvel_adr[i]]

            tau_demand = Kp * (q_settle[i] - q) - Kd * qd
            data.ctrl[i] = clip_torque(tau_demand)

        mujoco.mj_step(model, data)


def simulate_trajectory(
    profile_key,
    motion_duration_scale=1.0,
    total_cycles=TOTAL_CYCLES_MAIN,
    save_outputs=False,
    output_folder=None,
    verbose=True,
):
    """
    Run one trajectory profile and optionally save CSV/plots.
    """
    t0 = time.perf_counter()

    pre = precompute_desired_trajectory(
        profile_key,
        motion_duration_scale=motion_duration_scale,
        total_cycles=total_cycles,
    )

    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)

    settle_robot(data, pre["q_settle"])

    time_log = []
    q_des_log = []
    q_actual_log = []
    qd_actual_log = []
    tau_demand_log = []
    tau_applied_log = []
    ee_des_log = []
    ee_actual_log = []
    ik_residual_log = []

    for k in range(pre["n_steps"]):
        q_des = pre["q_des"][k, :]
        ee_des = pre["ee_des"][k, :]
        ik_residual = pre["ik_residual"][k, :]

        tau_demand = np.zeros(3)
        tau_applied = np.zeros(3)

        for i in range(3):
            q = data.qpos[qpos_adr[i]]
            qd = data.qvel[qvel_adr[i]]

            tau_raw = Kp * (q_des[i] - q) - Kd * qd
            tau_sat = clip_torque(tau_raw)

            data.ctrl[i] = tau_sat
            tau_demand[i] = tau_raw
            tau_applied[i] = tau_sat

        mujoco.mj_step(model, data)

        q_actual_after = np.zeros(3)
        qd_actual_after = np.zeros(3)

        for i in range(3):
            q_actual_after[i] = data.qpos[qpos_adr[i]]
            qd_actual_after[i] = data.qvel[qvel_adr[i]]

        time_log.append(pre["time_cmd"][k])
        q_des_log.append(q_des.copy())
        q_actual_log.append(q_actual_after.copy())
        qd_actual_log.append(qd_actual_after.copy())
        tau_demand_log.append(tau_demand.copy())
        tau_applied_log.append(tau_applied.copy())
        ee_des_log.append(ee_des.copy())
        ee_actual_log.append(data.xpos[ee_body_id].copy())
        ik_residual_log.append(ik_residual.copy())

    logs = {
        "time": np.array(time_log),
        "q_des": np.array(q_des_log),
        "q_actual": np.array(q_actual_log),
        "qd_actual": np.array(qd_actual_log),
        "tau_demand": np.array(tau_demand_log),
        "tau_applied": np.array(tau_applied_log),
        "ee_des": np.array(ee_des_log),
        "ee_actual": np.array(ee_actual_log),
        "ik_residual": np.array(ik_residual_log),
    }

    derived = compute_derived_logs(logs)

    metrics = compute_metrics(pre, logs, derived)
    metrics["motion_duration_scale"] = motion_duration_scale
    metrics["simulation_wall_time"] = time.perf_counter() - t0

    result = {
        "profile_key": profile_key,
        "precompute": pre,
        "logs": logs,
        "derived": derived,
        "metrics": metrics,
    }

    if save_outputs:
        if output_folder is None:
            output_folder = Path(profile_key)
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)

        save_trajectory_csv(result, output_folder)
        save_trajectory_plots(result, output_folder)

    if verbose:
        print_profile_summary(profile_key, metrics)

    return result


def compute_derived_logs(logs):
    time_log = logs["time"]
    dt_log = dt if len(time_log) < 2 else float(np.mean(np.diff(time_log)))

    ee_des_vel = finite_difference(logs["ee_des"], dt_log)
    ee_actual_vel = finite_difference(logs["ee_actual"], dt_log)

    ee_des_acc = finite_difference(ee_des_vel, dt_log)
    ee_actual_acc = finite_difference(ee_actual_vel, dt_log)

    qd_des = finite_difference(logs["q_des"], dt_log)

    tracking_error_xyz = logs["ee_des"] - logs["ee_actual"]
    tracking_error_norm = vector_norm_rows(tracking_error_xyz)

    return {
        "ee_des_vel": ee_des_vel,
        "ee_actual_vel": ee_actual_vel,
        "ee_des_speed": vector_norm_rows(ee_des_vel),
        "ee_actual_speed": vector_norm_rows(ee_actual_vel),
        "ee_des_acc": ee_des_acc,
        "ee_actual_acc": ee_actual_acc,
        "ee_des_acc_norm": vector_norm_rows(ee_des_acc),
        "ee_actual_acc_norm": vector_norm_rows(ee_actual_acc),
        "qd_des": qd_des,
        "tracking_error_xyz": tracking_error_xyz,
        "tracking_error_norm": tracking_error_norm,
    }


def compute_metrics(pre, logs, derived):
    return {
        "cycle_time": float(pre["cycle_time"]),
        "total_time": float(pre["total_time"]),
        "n_steps": int(pre["n_steps"]),
        "fallback_count": int(pre["fallback_count"]),
        "max_ik_residual": safe_peak_abs(logs["ik_residual"]),

        # Main comparison metrics
        "peak_tau_demand": safe_peak_abs(logs["tau_demand"]),
        "peak_tau_applied": safe_peak_abs(logs["tau_applied"]),
        "peak_ee_speed_des": float(np.max(derived["ee_des_speed"])),
        "peak_ee_speed_actual": float(np.max(derived["ee_actual_speed"])),
        "peak_ee_acc_des": float(np.max(derived["ee_des_acc_norm"])),
        "peak_ee_acc_actual": float(np.max(derived["ee_actual_acc_norm"])),
        "peak_joint_velocity_des": safe_peak_abs(derived["qd_des"]),
        "peak_joint_velocity_actual": safe_peak_abs(logs["qd_actual"]),
        "max_tracking_error": float(np.max(derived["tracking_error_norm"])),
    }


# =========================
# 12. Saving logs and plots
# =========================
def save_trajectory_csv(result, folder):
    logs = result["logs"]
    derived = result["derived"]

    csv_filename = folder / f"{result['profile_key']}_log.csv"

    headers = [
        "time",

        "ee_des_x", "ee_des_y", "ee_des_z",
        "ee_actual_x", "ee_actual_y", "ee_actual_z",

        "ee_des_vx", "ee_des_vy", "ee_des_vz",
        "ee_actual_vx", "ee_actual_vy", "ee_actual_vz",

        "ee_des_ax", "ee_des_ay", "ee_des_az",
        "ee_actual_ax", "ee_actual_ay", "ee_actual_az",

        "q_des_1", "q_des_2", "q_des_3",
        "q_actual_1", "q_actual_2", "q_actual_3",

        "qd_des_1", "qd_des_2", "qd_des_3",
        "qd_actual_1", "qd_actual_2", "qd_actual_3",

        "tau_demand_1", "tau_demand_2", "tau_demand_3",
        "tau_applied_1", "tau_applied_2", "tau_applied_3",

        "tracking_error_x", "tracking_error_y", "tracking_error_z",
        "tracking_error_norm",

        "ik_residual_1", "ik_residual_2", "ik_residual_3",
    ]

    with open(csv_filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)

        for i in range(len(logs["time"])):
            writer.writerow([
                logs["time"][i],

                logs["ee_des"][i, 0], logs["ee_des"][i, 1], logs["ee_des"][i, 2],
                logs["ee_actual"][i, 0], logs["ee_actual"][i, 1], logs["ee_actual"][i, 2],

                derived["ee_des_vel"][i, 0], derived["ee_des_vel"][i, 1], derived["ee_des_vel"][i, 2],
                derived["ee_actual_vel"][i, 0], derived["ee_actual_vel"][i, 1], derived["ee_actual_vel"][i, 2],

                derived["ee_des_acc"][i, 0], derived["ee_des_acc"][i, 1], derived["ee_des_acc"][i, 2],
                derived["ee_actual_acc"][i, 0], derived["ee_actual_acc"][i, 1], derived["ee_actual_acc"][i, 2],

                logs["q_des"][i, 0], logs["q_des"][i, 1], logs["q_des"][i, 2],
                logs["q_actual"][i, 0], logs["q_actual"][i, 1], logs["q_actual"][i, 2],

                derived["qd_des"][i, 0], derived["qd_des"][i, 1], derived["qd_des"][i, 2],
                logs["qd_actual"][i, 0], logs["qd_actual"][i, 1], logs["qd_actual"][i, 2],

                logs["tau_demand"][i, 0], logs["tau_demand"][i, 1], logs["tau_demand"][i, 2],
                logs["tau_applied"][i, 0], logs["tau_applied"][i, 1], logs["tau_applied"][i, 2],

                derived["tracking_error_xyz"][i, 0],
                derived["tracking_error_xyz"][i, 1],
                derived["tracking_error_xyz"][i, 2],
                derived["tracking_error_norm"][i],

                logs["ik_residual"][i, 0], logs["ik_residual"][i, 1], logs["ik_residual"][i, 2],
            ])


def save_trajectory_plots(result, folder):
    profile_label = get_profile_label(result["profile_key"])
    logs = result["logs"]
    derived = result["derived"]
    t = logs["time"]

    # 1. End-effector position vs time
    plt.figure(figsize=(10, 6))
    axis_names = ["x", "y", "z"]
    for j, axis_name in enumerate(axis_names):
        plt.plot(t, logs["ee_des"][:, j], label=f"desired {axis_name}")
        plt.plot(t, logs["ee_actual"][:, j], "--", label=f"actual {axis_name}")
    plt.xlabel("Time (s)")
    plt.ylabel("End-effector position (m)")
    plt.title(f"{profile_label}: End-effector Position")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(folder / "01_end_effector_position.png", dpi=300)
    plt.close()

    # 2. End-effector velocity vs time
    plt.figure(figsize=(10, 6))
    plt.plot(t, derived["ee_des_speed"], label="desired speed")
    plt.plot(t, derived["ee_actual_speed"], "--", label="actual speed")
    plt.xlabel("Time (s)")
    plt.ylabel("End-effector speed (m/s)")
    plt.title(f"{profile_label}: End-effector Velocity")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(folder / "02_end_effector_velocity.png", dpi=300)
    plt.close()

    # 3. End-effector acceleration vs time
    plt.figure(figsize=(10, 6))
    plt.plot(t, derived["ee_des_acc_norm"], label="desired acceleration magnitude")
    plt.plot(t, derived["ee_actual_acc_norm"], "--", label="actual acceleration magnitude")
    plt.xlabel("Time (s)")
    plt.ylabel("End-effector acceleration (m/s^2)")
    plt.title(f"{profile_label}: End-effector Acceleration")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(folder / "03_end_effector_acceleration.png", dpi=300)
    plt.close()

    # 4. Joint angles vs time
    plt.figure(figsize=(10, 6))
    for j in range(3):
        plt.plot(t, logs["q_des"][:, j], label=f"q{j + 1} desired")
        plt.plot(t, logs["q_actual"][:, j], "--", label=f"q{j + 1} actual")
    plt.xlabel("Time (s)")
    plt.ylabel("Joint angle (rad)")
    plt.title(f"{profile_label}: Joint Angles")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(folder / "04_joint_angles.png", dpi=300)
    plt.close()

    # 5. Joint velocities vs time
    plt.figure(figsize=(10, 6))
    for j in range(3):
        plt.plot(t, derived["qd_des"][:, j], label=f"qdot{j + 1} desired")
        plt.plot(t, logs["qd_actual"][:, j], "--", label=f"qdot{j + 1} actual")
    plt.xlabel("Time (s)")
    plt.ylabel("Joint velocity (rad/s)")
    plt.title(f"{profile_label}: Joint Velocities")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(folder / "05_joint_velocities.png", dpi=300)
    plt.close()

    # 6. Joint torque vs time
    plt.figure(figsize=(10, 6))
    for j in range(3):
        plt.plot(t, logs["tau_demand"][:, j], label=f"tau{j + 1} demand")
        plt.plot(t, logs["tau_applied"][:, j], "--", label=f"tau{j + 1} applied")
    plt.axhline(tau_limit, linestyle=":", label="+ torque limit")
    plt.axhline(-tau_limit, linestyle=":", label="- torque limit")
    plt.xlabel("Time (s)")
    plt.ylabel("Joint torque command (N·m)")
    plt.title(f"{profile_label}: Joint Torque")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(folder / "06_joint_torques.png", dpi=300)
    plt.close()

    # 7. Tracking error vs time
    plt.figure(figsize=(10, 6))
    plt.plot(t, derived["tracking_error_xyz"][:, 0], label="x error")
    plt.plot(t, derived["tracking_error_xyz"][:, 1], label="y error")
    plt.plot(t, derived["tracking_error_xyz"][:, 2], label="z error")
    plt.plot(t, derived["tracking_error_norm"], "--", label="error norm")
    plt.xlabel("Time (s)")
    plt.ylabel("Tracking error (m)")
    plt.title(f"{profile_label}: End-effector Tracking Error")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(folder / "07_tracking_error.png", dpi=300)
    plt.close()

    # 8. 3D end-effector path
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(
        logs["ee_des"][:, 0],
        logs["ee_des"][:, 1],
        logs["ee_des"][:, 2],
        label="desired path",
    )
    ax.plot(
        logs["ee_actual"][:, 0],
        logs["ee_actual"][:, 1],
        logs["ee_actual"][:, 2],
        "--",
        label="actual path",
    )
    ax.set_xlabel("X position (m)")
    ax.set_ylabel("Y position (m)")
    ax.set_zlabel("Z position (m)")
    ax.set_title(f"{profile_label}: 3D End-effector Path")
    ax.legend()
    set_3d_axes_equal(
        ax,
        np.concatenate([logs["ee_des"][:, 0], logs["ee_actual"][:, 0]]),
        np.concatenate([logs["ee_des"][:, 1], logs["ee_actual"][:, 1]]),
        np.concatenate([logs["ee_des"][:, 2], logs["ee_actual"][:, 2]]),
    )
    plt.tight_layout()
    plt.savefig(folder / "08_3d_end_effector_path.png", dpi=300)
    plt.close()


# =========================
# 13. Comparison plots
# =========================
def save_comparison_outputs(results_by_key):
    COMPARISON_FOLDER.mkdir(parents=True, exist_ok=True)

    save_comparison_csv(results_by_key)
    save_comparison_plots(results_by_key)


def save_comparison_csv(results_by_key):
    csv_filename = COMPARISON_FOLDER / "trajectory_comparison_summary.csv"

    headers = [
        "trajectory",
        "cycle_time_for_main_run_s",
        "peak_tau_demand_Nm",
        "peak_tau_applied_Nm",
        "peak_desired_ee_speed_m_per_s",
        "peak_actual_ee_speed_m_per_s",
        "peak_desired_ee_acc_m_per_s2",
        "peak_actual_ee_acc_m_per_s2",
        "peak_desired_joint_velocity_rad_per_s",
        "peak_actual_joint_velocity_rad_per_s",
        "max_tracking_error_m",
        "max_ik_residual_m",
        "minimum_feasible_cycle_time_s",
        "minimum_feasible_motion_duration_scale",
    ]

    with open(csv_filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)

        for spec in TRAJECTORIES:
            result = results_by_key[spec["key"]]
            m = result["metrics"]

            writer.writerow([
                spec["label"],
                m.get("cycle_time", float("nan")),
                m.get("peak_tau_demand", float("nan")),
                m.get("peak_tau_applied", float("nan")),
                m.get("peak_ee_speed_des", float("nan")),
                m.get("peak_ee_speed_actual", float("nan")),
                m.get("peak_ee_acc_des", float("nan")),
                m.get("peak_ee_acc_actual", float("nan")),
                m.get("peak_joint_velocity_des", float("nan")),
                m.get("peak_joint_velocity_actual", float("nan")),
                m.get("max_tracking_error", float("nan")),
                m.get("max_ik_residual", float("nan")),
                m.get("minimum_feasible_cycle_time", float("nan")),
                m.get("minimum_feasible_motion_duration_scale", float("nan")),
            ])


def save_comparison_plots(results_by_key):
    labels = [spec["label"] for spec in TRAJECTORIES]

    peak_tau = [results_by_key[spec["key"]]["metrics"]["peak_tau_demand"] for spec in TRAJECTORIES]
    peak_velocity = [results_by_key[spec["key"]]["metrics"]["peak_ee_speed_des"] for spec in TRAJECTORIES]
    peak_acceleration = [results_by_key[spec["key"]]["metrics"]["peak_ee_acc_des"] for spec in TRAJECTORIES]
    peak_tracking_error = [results_by_key[spec["key"]]["metrics"]["max_tracking_error"] for spec in TRAJECTORIES]
    min_cycle_time = [
        results_by_key[spec["key"]]["metrics"].get("minimum_feasible_cycle_time", float("nan"))
        for spec in TRAJECTORIES
    ]

    # 1. Peak torque comparison
    plt.figure(figsize=(8, 5))
    plt.bar(labels, peak_tau)
    plt.axhline(tau_limit, linestyle=":", label="torque limit")
    plt.ylabel("Peak unsaturated torque demand (N·m)")
    plt.title("Peak Torque Comparison")
    plt.legend()
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.savefig(COMPARISON_FOLDER / "01_peak_torque_comparison.png", dpi=300)
    plt.close()

    # 2. Peak velocity comparison
    plt.figure(figsize=(8, 5))
    plt.bar(labels, peak_velocity)
    plt.ylabel("Peak desired end-effector speed (m/s)")
    plt.title("Peak Velocity Comparison")
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.savefig(COMPARISON_FOLDER / "02_peak_velocity_comparison.png", dpi=300)
    plt.close()

    # 3. Peak acceleration comparison
    plt.figure(figsize=(8, 5))
    plt.bar(labels, peak_acceleration)
    plt.ylabel("Peak desired end-effector acceleration (m/s^2)")
    plt.title("Peak Acceleration Comparison")
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.savefig(COMPARISON_FOLDER / "03_peak_acceleration_comparison.png", dpi=300)
    plt.close()

    # 4. Peak tracking error comparison
    plt.figure(figsize=(8, 5))
    plt.bar(labels, peak_tracking_error)
    plt.ylabel("Maximum end-effector tracking error (m)")
    plt.title("Peak Tracking Error Comparison")
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.savefig(COMPARISON_FOLDER / "04_peak_tracking_error_comparison.png", dpi=300)
    plt.close()

    # 5. Minimum feasible cycle time comparison
    plt.figure(figsize=(8, 5))
    plt.bar(labels, min_cycle_time)
    plt.ylabel("Estimated minimum feasible cycle time (s)")
    plt.title("Estimated Minimum Feasible Cycle Time Comparison")
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.savefig(COMPARISON_FOLDER / "05_minimum_feasible_cycle_time_comparison.png", dpi=300)
    plt.close()


# =========================
# 14. Minimum feasible cycle-time search
# =========================
def is_feasible(metrics):
    torque_ok = metrics["peak_tau_demand"] <= tau_limit

    if not USE_TRACKING_ERROR_IN_FEASIBILITY:
        return torque_ok

    tracking_ok = metrics["max_tracking_error"] <= FEASIBILITY_TRACKING_ERROR_LIMIT
    return torque_ok and tracking_ok


def evaluate_feasibility(profile_key, motion_duration_scale):
    result = simulate_trajectory(
        profile_key,
        motion_duration_scale=motion_duration_scale,
        total_cycles=TOTAL_CYCLES_FEASIBILITY,
        save_outputs=False,
        output_folder=None,
        verbose=False,
    )

    return is_feasible(result["metrics"]), result["metrics"]


def find_minimum_feasible_cycle_time(profile_key):
    """
    Estimate the fastest feasible cycle time by scaling only moving segments.
    Dwell segments remain fixed at 0.3 s each.
    """
    hi = 1.0
    hi_feasible, hi_metrics = evaluate_feasibility(profile_key, hi)

    while (not hi_feasible) and hi < MAX_MOTION_SCALE:
        hi *= 1.25
        hi_feasible, hi_metrics = evaluate_feasibility(profile_key, hi)

    if not hi_feasible:
        return {
            "minimum_feasible_cycle_time": float("nan"),
            "minimum_feasible_motion_duration_scale": float("nan"),
            "minimum_feasible_peak_tau_demand": hi_metrics["peak_tau_demand"],
            "minimum_feasible_tracking_error": hi_metrics["max_tracking_error"],
            "feasibility_note": "No feasible scale found up to MAX_MOTION_SCALE.",
        }

    lo = MIN_MOTION_SCALE
    lo_feasible, lo_metrics = evaluate_feasibility(profile_key, lo)

    if lo_feasible:
        selected_scale = lo
        selected_metrics = lo_metrics
    else:
        selected_scale = hi
        selected_metrics = hi_metrics

        for _ in range(FEASIBILITY_BISECTION_STEPS):
            mid = 0.5 * (lo + hi)
            mid_feasible, mid_metrics = evaluate_feasibility(profile_key, mid)

            if mid_feasible:
                hi = mid
                selected_scale = mid
                selected_metrics = mid_metrics
            else:
                lo = mid

    selected_segments = build_scaled_segments(selected_scale)
    selected_cycle_time = get_cycle_time(selected_segments)

    return {
        "minimum_feasible_cycle_time": float(selected_cycle_time),
        "minimum_feasible_motion_duration_scale": float(selected_scale),
        "minimum_feasible_peak_tau_demand": selected_metrics["peak_tau_demand"],
        "minimum_feasible_tracking_error": selected_metrics["max_tracking_error"],
        "feasibility_note": "Feasible.",
    }


# =========================
# 15. Utility output functions
# =========================
def get_profile_label(profile_key):
    for spec in TRAJECTORIES:
        if spec["key"] == profile_key:
            return spec["label"]
    return profile_key


def print_startup_summary():
    print("\nModel loaded successfully.")
    print(f"XML file: {XML_FILENAME}")
    print(f"MuJoCo timestep: {dt:.6f} s")
    print("Motor qpos addresses:", qpos_adr)
    print("Motor qvel addresses:", qvel_adr)
    print("End-effector body id:", ee_body_id)
    print(f"Upper arm length: {upper_arm_length:.6f} m")
    print(f"Lower arm length: {lower_arm_length:.6f} m")
    print(f"Torque limit: +/- {tau_limit:.3f} N·m")
    print(f"Main run cycles per trajectory: {TOTAL_CYCLES_MAIN}")


def print_profile_summary(profile_key, metrics):
    label = get_profile_label(profile_key)
    print(f"\nFinished {label}.")
    print(f"  Cycle time: {metrics['cycle_time']:.3f} s")
    print(f"  Peak torque demand: {metrics['peak_tau_demand']:.3f} N·m")
    print(f"  Peak applied torque: {metrics['peak_tau_applied']:.3f} N·m")
    print(f"  Peak desired EE speed: {metrics['peak_ee_speed_des']:.3f} m/s")
    print(f"  Peak desired EE acceleration: {metrics['peak_ee_acc_des']:.3f} m/s^2")
    print(f"  Max tracking error: {metrics['max_tracking_error']:.6f} m")
    print(f"  Max IK residual: {metrics['max_ik_residual']:.6e} m")
    print(f"  Fallback IK count: {metrics['fallback_count']} / {metrics['n_steps']}")
    print(f"  Wall time: {metrics['simulation_wall_time']:.2f} s")


def print_min_time_summary(profile_key, min_time_result):
    label = get_profile_label(profile_key)
    print(f"\nMinimum feasible cycle-time search for {label}:")
    print(f"  Estimated minimum cycle time: {min_time_result['minimum_feasible_cycle_time']}")
    print(f"  Motion duration scale: {min_time_result['minimum_feasible_motion_duration_scale']}")
    print(f"  Peak torque demand at selected scale: {min_time_result['minimum_feasible_peak_tau_demand']}")
    print(f"  Tracking error at selected scale: {min_time_result['minimum_feasible_tracking_error']}")
    print(f"  Note: {min_time_result['feasibility_note']}")


# =========================
# 16. Main
# =========================
def main():
    print_startup_summary()

    results_by_key = {}

    for spec in TRAJECTORIES:
        profile_key = spec["key"]
        output_folder = Path(spec["folder"])

        print(f"\nRunning {spec['label']} trajectory...")
        result = simulate_trajectory(
            profile_key,
            motion_duration_scale=1.0,
            total_cycles=TOTAL_CYCLES_MAIN,
            save_outputs=True,
            output_folder=output_folder,
            verbose=True,
        )

        results_by_key[profile_key] = result

    if RUN_MIN_TIME_SWEEP:
        print("\nRunning estimated minimum feasible cycle-time searches...")

        for spec in TRAJECTORIES:
            min_time_result = find_minimum_feasible_cycle_time(spec["key"])
            results_by_key[spec["key"]]["metrics"].update(min_time_result)
            print_min_time_summary(spec["key"], min_time_result)
    else:
        for spec in TRAJECTORIES:
            m = results_by_key[spec["key"]]["metrics"]
            m["minimum_feasible_cycle_time"] = m["cycle_time"] if is_feasible(m) else float("nan")
            m["minimum_feasible_motion_duration_scale"] = 1.0 if is_feasible(m) else float("nan")

    save_comparison_outputs(results_by_key)

    print("\nAll requested folders and plots have been created:")
    for spec in TRAJECTORIES:
        print(f"  {spec['folder']}/")
    print(f"  {COMPARISON_FOLDER}/")


if __name__ == "__main__":
    main()
