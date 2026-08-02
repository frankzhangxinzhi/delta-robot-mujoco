import math
import csv
import time
import os

import mujoco
import mujoco.viewer
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =========================
# 1. Load MuJoCo model
# =========================
model = mujoco.MjModel.from_xml_path("delta_closed.xml")


# =========================
# 2. Find motor joint addresses
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

print("Motor qpos addresses:", qpos_adr)
print("Motor qvel addresses:", qvel_adr)


# =========================
# 3. Find end-effector body
# =========================
ee_body_name = "platform"
ee_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, ee_body_name)

if ee_body_id == -1:
    raise ValueError(f"Cannot find body named {ee_body_name}. Check XML body names.")

print("End-effector body id:", ee_body_id)


# =========================
# 4. Controller gains and torque limit
# =========================
Kp = 120.0
Kd = 14.0
tau_limit = 20.0


# =========================
# 5. Timing sweep settings
# =========================
# Study 1:
# smoothstep profile stays fixed.
# only execution time changes.
time_scales = [1.00, 0.85, 0.70, 0.60, 0.50, 0.45, 0.40, 0.35, 0.30]

# Number of cycles simulated for each timing case.
total_cycles = 2.0

# For the timing sweep, keep viewer off so the tests run automatically.
# Later, set this to True if you want to watch one selected time scale.
show_viewer_after_sweep = False
viewer_time_scale = 1.00


# =========================
# 6. Output folder
# =========================
output_dir = "timing_sweep_results"
os.makedirs(output_dir, exist_ok=True)

print(f"Output files will be saved in: {output_dir}")


# =========================
# 7. Delta robot geometry for inverse kinematics
# =========================
# These numbers must match delta_closed.xml.

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

# Current final baseline:
# visual platform radius = 0.08 m
# platform connection-site radius = 0.07 m
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

print("Upper arm length:", upper_arm_length)
print("Lower arm length:", lower_arm_length)


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
    Used only for initialization or when fast local IK fails.
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
    It uses the previous timestep's solution as the starting guess.
    If it fails, it falls back to the brute-force solver.
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
# 8. Cartesian pick-and-place task
# =========================
# Raised working height.
# Horizontal task remains unchanged.

pick_high = np.array([-0.07, -0.06, 0.35])
pick_low  = np.array([-0.07, -0.06, 0.26])

place_high = np.array([0.08, 0.06, 0.35])
place_low  = np.array([0.08, 0.06, 0.26])

base_trajectory_segments = [
    (pick_high,   pick_low,    1.0),   # move down to pick
    (pick_low,    pick_low,    0.3),   # pick dwell
    (pick_low,    pick_high,   1.0),   # lift
    (pick_high,   place_high,  1.2),   # move horizontally
    (place_high,  place_low,   1.0),   # move down to place
    (place_low,   place_low,   0.3),   # place dwell
    (place_low,   place_high,  1.0),   # lift
    (place_high,  pick_high,   1.2),   # return
]

base_cycle_time = sum(seg[2] for seg in base_trajectory_segments)
dt = model.opt.timestep


def smoothstep(s):
    """
    Current fixed trajectory profile:
    cubic smoothstep interpolation.
    """
    return 3.0 * s**2 - 2.0 * s**3


def interpolate(p_start, p_end, s):
    return p_start + smoothstep(s) * (p_end - p_start)


def make_scaled_segments(time_scale):
    """
    Scale all segment durations by time_scale.
    The Cartesian path and smoothstep profile stay unchanged.
    """
    return [
        (p_start, p_end, duration * time_scale)
        for p_start, p_end, duration in base_trajectory_segments
    ]


def desired_platform_position(t, trajectory_segments, cycle_time):
    """
    Returns desired platform XYZ position at time t.
    """
    tau = t % cycle_time
    elapsed = 0.0

    for p_start, p_end, duration in trajectory_segments:
        if tau <= elapsed + duration:
            s = (tau - elapsed) / duration
            return interpolate(p_start, p_end, s)

        elapsed += duration

    return trajectory_segments[-1][1]


# =========================
# 9. Precompute trajectory and IK
# =========================
def precompute_trajectory_and_ik(time_scale):
    trajectory_segments = make_scaled_segments(time_scale)
    cycle_time = sum(seg[2] for seg in trajectory_segments)
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
        ee_des = desired_platform_position(t_cmd, trajectory_segments, cycle_time)

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
        "time_scale": time_scale,
        "trajectory_segments": trajectory_segments,
        "cycle_time": cycle_time,
        "total_time": total_time,
        "n_steps": n_steps,
        "time_cmd_array": time_cmd_array,
        "ee_des_array": ee_des_array,
        "q_des_array": q_des_array,
        "ik_residual_array": ik_residual_array,
        "q_settle": q_settle,
        "settle_residual": settle_residual,
        "fallback_count": fallback_count,
    }


# =========================
# 10. Run one simulation case
# =========================
def run_single_case(precomputed, use_viewer=False):
    data = mujoco.MjData(model)

    time_cmd_array = precomputed["time_cmd_array"]
    ee_des_array = precomputed["ee_des_array"]
    q_des_array = precomputed["q_des_array"]
    ik_residual_array = precomputed["ik_residual_array"]
    q_settle = precomputed["q_settle"]
    n_steps = precomputed["n_steps"]

    time_log = []
    q_des_log = []
    q_actual_log = []
    tau_log = []

    ee_des_log = []
    ee_actual_log = []
    ik_residual_log = []

    position_error_log = []
    x_error_log = []
    y_error_log = []
    z_error_log = []
    xy_error_log = []

    qpos_history = []

    render_every = 20
    replay_stride = 1
    replay_dt = dt * render_every * replay_stride

    settle_time = 1.0
    step_count = 0

    def controller_step(k):
        q_des = q_des_array[k, :]
        ee_des = ee_des_array[k, :]
        ik_residual = ik_residual_array[k, :]

        tau_cmd = np.zeros(3)

        for i in range(3):
            q = data.qpos[qpos_adr[i]]
            qd = data.qvel[qvel_adr[i]]

            tau = Kp * (q_des[i] - q) - Kd * qd
            tau = max(min(tau, tau_limit), -tau_limit)

            data.ctrl[i] = tau
            tau_cmd[i] = tau

        mujoco.mj_step(model, data)

        q_actual_after = np.zeros(3)
        for i in range(3):
            q_actual_after[i] = data.qpos[qpos_adr[i]]

        ee_actual = data.xpos[ee_body_id].copy()

        error_vec = ee_actual - ee_des
        x_error = abs(error_vec[0])
        y_error = abs(error_vec[1])
        z_error = abs(error_vec[2])
        xy_error = math.sqrt(error_vec[0] ** 2 + error_vec[1] ** 2)
        position_error = np.linalg.norm(error_vec)

        time_log.append(time_cmd_array[k])
        q_des_log.append(q_des.copy())
        q_actual_log.append(q_actual_after.copy())
        tau_log.append(tau_cmd.copy())

        ee_des_log.append(ee_des.copy())
        ee_actual_log.append(ee_actual.copy())
        ik_residual_log.append(ik_residual.copy())

        x_error_log.append(x_error)
        y_error_log.append(y_error)
        z_error_log.append(z_error)
        xy_error_log.append(xy_error)
        position_error_log.append(position_error)

        return tau_cmd

    if use_viewer:
        print("\nOpening viewer for selected time scale...")

        with mujoco.viewer.launch_passive(model, data) as viewer:
            print("Settling robot to initial pose...")

            while viewer.is_running() and data.time < settle_time:
                for i in range(3):
                    q = data.qpos[qpos_adr[i]]
                    qd = data.qvel[qvel_adr[i]]

                    tau = Kp * (q_settle[i] - q) - Kd * qd
                    tau = max(min(tau, tau_limit), -tau_limit)

                    data.ctrl[i] = tau

                mujoco.mj_step(model, data)

                step_count += 1
                if step_count % render_every == 0:
                    viewer.sync()

            print("Running finite physics simulation...")

            for k in range(n_steps):
                if not viewer.is_running():
                    break

                controller_step(k)

                step_count += 1
                if step_count % render_every == 0:
                    viewer.sync()
                    qpos_history.append(data.qpos.copy())

            if viewer.is_running() and len(qpos_history) > 0:
                print("\nFinite simulation complete.")
                print("Now replaying saved motion forever.")
                print("Close the viewer manually when done.")

                while viewer.is_running():
                    for qpos_frame in qpos_history[::replay_stride]:
                        if not viewer.is_running():
                            break

                        data.qpos[:] = qpos_frame
                        data.qvel[:] = 0.0
                        mujoco.mj_forward(model, data)

                        viewer.sync()
                        time.sleep(replay_dt)

    else:
        # Settling phase without viewer.
        while data.time < settle_time:
            for i in range(3):
                q = data.qpos[qpos_adr[i]]
                qd = data.qvel[qvel_adr[i]]

                tau = Kp * (q_settle[i] - q) - Kd * qd
                tau = max(min(tau, tau_limit), -tau_limit)

                data.ctrl[i] = tau

            mujoco.mj_step(model, data)

        # Main finite simulation without viewer.
        for k in range(n_steps):
            controller_step(k)

    time_log = np.array(time_log)
    q_des_log = np.array(q_des_log)
    q_actual_log = np.array(q_actual_log)
    tau_log = np.array(tau_log)

    ee_des_log = np.array(ee_des_log)
    ee_actual_log = np.array(ee_actual_log)
    ik_residual_log = np.array(ik_residual_log)

    position_error_log = np.array(position_error_log)
    x_error_log = np.array(x_error_log)
    y_error_log = np.array(y_error_log)
    z_error_log = np.array(z_error_log)
    xy_error_log = np.array(xy_error_log)

    if len(time_log) == 0:
        raise RuntimeError("No simulation data was logged.")

    max_torque = np.max(np.abs(tau_log))

    max_position_error = np.max(position_error_log)
    max_x_error = np.max(x_error_log)
    max_y_error = np.max(y_error_log)
    max_z_error = np.max(z_error_log)
    max_xy_error = np.max(xy_error_log)

    rms_position_error = math.sqrt(np.mean(position_error_log ** 2))
    rms_x_error = math.sqrt(np.mean(x_error_log ** 2))
    rms_y_error = math.sqrt(np.mean(y_error_log ** 2))
    rms_z_error = math.sqrt(np.mean(z_error_log ** 2))
    rms_xy_error = math.sqrt(np.mean(xy_error_log ** 2))

    max_ik_residual = np.max(ik_residual_log)

    return {
        "time_log": time_log,
        "q_des_log": q_des_log,
        "q_actual_log": q_actual_log,
        "tau_log": tau_log,
        "ee_des_log": ee_des_log,
        "ee_actual_log": ee_actual_log,
        "ik_residual_log": ik_residual_log,

        "position_error_log": position_error_log,
        "x_error_log": x_error_log,
        "y_error_log": y_error_log,
        "z_error_log": z_error_log,
        "xy_error_log": xy_error_log,

        "max_torque": max_torque,

        "max_position_error": max_position_error,
        "max_x_error": max_x_error,
        "max_y_error": max_y_error,
        "max_z_error": max_z_error,
        "max_xy_error": max_xy_error,

        "rms_position_error": rms_position_error,
        "rms_x_error": rms_x_error,
        "rms_y_error": rms_y_error,
        "rms_z_error": rms_z_error,
        "rms_xy_error": rms_xy_error,

        "max_ik_residual": max_ik_residual,
    }


# =========================
# 11. Save plots for one selected case
# =========================
def save_detailed_plots(result, prefix="baseline"):
    time_log = result["time_log"]
    q_des_log = result["q_des_log"]
    q_actual_log = result["q_actual_log"]
    tau_log = result["tau_log"]
    ee_des_log = result["ee_des_log"]
    ee_actual_log = result["ee_actual_log"]
    ik_residual_log = result["ik_residual_log"]

    plt.figure()
    plt.plot(time_log, q_des_log[:, 0], label="motor1 desired")
    plt.plot(time_log, q_actual_log[:, 0], "--", label="motor1 actual")
    plt.plot(time_log, q_des_log[:, 1], label="motor2 desired")
    plt.plot(time_log, q_actual_log[:, 1], "--", label="motor2 actual")
    plt.plot(time_log, q_des_log[:, 2], label="motor3 desired")
    plt.plot(time_log, q_actual_log[:, 2], "--", label="motor3 actual")
    plt.xlabel("Time (s)")
    plt.ylabel("Joint Angle (rad)")
    plt.title("Desired vs Actual Motor Angles")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f"{prefix}_motor_tracking.png"), dpi=300)
    plt.close()

    plt.figure()
    plt.plot(time_log, ee_des_log[:, 0], label="desired x")
    plt.plot(time_log, ee_actual_log[:, 0], "--", label="actual x")
    plt.xlabel("Time (s)")
    plt.ylabel("End-Effector X Position (m)")
    plt.title("End-Effector X Pick-and-Place Motion")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f"{prefix}_x_tracking.png"), dpi=300)
    plt.close()

    plt.figure()
    plt.plot(time_log, ee_des_log[:, 1], label="desired y")
    plt.plot(time_log, ee_actual_log[:, 1], "--", label="actual y")
    plt.xlabel("Time (s)")
    plt.ylabel("End-Effector Y Position (m)")
    plt.title("End-Effector Y Pick-and-Place Motion")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f"{prefix}_y_tracking.png"), dpi=300)
    plt.close()

    plt.figure()
    plt.plot(time_log, ee_des_log[:, 2], label="desired z")
    plt.plot(time_log, ee_actual_log[:, 2], "--", label="actual z")
    plt.xlabel("Time (s)")
    plt.ylabel("End-Effector Z Position (m)")
    plt.title("End-Effector Z Pick-and-Place Motion")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f"{prefix}_z_tracking.png"), dpi=300)
    plt.close()

    plt.figure()
    plt.plot(ee_des_log[:, 0], ee_des_log[:, 1], label="desired XY path")
    plt.plot(ee_actual_log[:, 0], ee_actual_log[:, 1], "--", label="actual XY path")
    plt.xlabel("X Position (m)")
    plt.ylabel("Y Position (m)")
    plt.title("End-Effector XY Pick-and-Place Path")
    plt.legend()
    plt.grid(True)
    plt.axis("equal")
    plt.savefig(os.path.join(output_dir, f"{prefix}_xy_path.png"), dpi=300)
    plt.close()

    plt.figure()
    plt.plot(time_log, tau_log[:, 0], label="tau1")
    plt.plot(time_log, tau_log[:, 1], label="tau2")
    plt.plot(time_log, tau_log[:, 2], label="tau3")
    plt.xlabel("Time (s)")
    plt.ylabel("Torque Command (N m)")
    plt.title("Actuator Torque Commands During Pick-and-Place")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f"{prefix}_torque.png"), dpi=300)
    plt.close()

    plt.figure()
    plt.plot(time_log, ik_residual_log[:, 0], label="arm1 IK residual")
    plt.plot(time_log, ik_residual_log[:, 1], label="arm2 IK residual")
    plt.plot(time_log, ik_residual_log[:, 2], label="arm3 IK residual")
    plt.xlabel("Time (s)")
    plt.ylabel("IK Residual (m)")
    plt.title("Inverse Kinematics Residual")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f"{prefix}_ik_residual.png"), dpi=300)
    plt.close()


# =========================
# 12. Run revised timing sweep
# =========================
print("\nStarting Revised Study 1: timing sweep with smoothstep fixed.")
print("No artificial pass/fail tolerance is used.")
print("Only speed changes; torque and tracking errors are measured.\n")

summary_rows = []
detailed_results = {}

overall_start = time.perf_counter()

for time_scale in time_scales:
    print(f"Running time_scale = {time_scale:.2f} ...")

    pre_start = time.perf_counter()
    precomputed = precompute_trajectory_and_ik(time_scale)
    pre_end = time.perf_counter()

    sim_start = time.perf_counter()
    result = run_single_case(precomputed, use_viewer=False)
    sim_end = time.perf_counter()

    cycle_time = precomputed["cycle_time"]

    row = {
        "time_scale": time_scale,
        "cycle_time": cycle_time,

        "max_torque": result["max_torque"],

        "max_position_error": result["max_position_error"],
        "max_x_error": result["max_x_error"],
        "max_y_error": result["max_y_error"],
        "max_z_error": result["max_z_error"],
        "max_xy_error": result["max_xy_error"],

        "rms_position_error": result["rms_position_error"],
        "rms_x_error": result["rms_x_error"],
        "rms_y_error": result["rms_y_error"],
        "rms_z_error": result["rms_z_error"],
        "rms_xy_error": result["rms_xy_error"],

        "max_ik_residual": result["max_ik_residual"],

        "precompute_time_sec": pre_end - pre_start,
        "simulation_time_sec": sim_end - sim_start,
        "fallback_count": precomputed["fallback_count"],
        "n_steps": precomputed["n_steps"],
    }

    summary_rows.append(row)
    detailed_results[time_scale] = result

    print(
        f"  cycle_time = {cycle_time:.3f} s, "
        f"max_torque = {result['max_torque']:.3f} N m, "
        f"max_3d_error = {result['max_position_error']:.4f} m, "
        f"max_xy_error = {result['max_xy_error']:.4f} m, "
        f"max_z_error = {result['max_z_error']:.4f} m"
    )

overall_end = time.perf_counter()

print(f"\nTiming sweep finished in {overall_end - overall_start:.2f} seconds.")


# =========================
# 13. Save timing sweep summary CSV
# =========================
summary_filename = os.path.join(output_dir, "timing_sweep_summary.csv")

with open(summary_filename, "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "time_scale",
            "cycle_time",

            "max_torque",

            "max_position_error",
            "max_x_error",
            "max_y_error",
            "max_z_error",
            "max_xy_error",

            "rms_position_error",
            "rms_x_error",
            "rms_y_error",
            "rms_z_error",
            "rms_xy_error",

            "max_ik_residual",

            "precompute_time_sec",
            "simulation_time_sec",
            "fallback_count",
            "n_steps",
        ],
    )

    writer.writeheader()

    for row in summary_rows:
        writer.writerow(row)

print(f"\nSaved timing sweep summary to {summary_filename}")


# =========================
# 14. Print concise summary table
# =========================
print("\nRevised timing sweep summary:")
print("time_scale | cycle_time | max_torque | max_3d_err | max_xy_err | max_z_err")
print("-" * 86)

for row in summary_rows:
    print(
        f"{row['time_scale']:>10.2f} | "
        f"{row['cycle_time']:>10.3f} | "
        f"{row['max_torque']:>10.3f} | "
        f"{row['max_position_error']:>10.5f} | "
        f"{row['max_xy_error']:>10.5f} | "
        f"{row['max_z_error']:>9.5f}"
    )


# =========================
# 15. Fit simple speed-error and speed-torque trends
# =========================
cycle_time_array = np.array([row["cycle_time"] for row in summary_rows])
inv_t2_array = 1.0 / (cycle_time_array ** 2)

max_torque_array = np.array([row["max_torque"] for row in summary_rows])
max_position_error_array = np.array([row["max_position_error"] for row in summary_rows])
max_xy_error_array = np.array([row["max_xy_error"] for row in summary_rows])
max_z_error_array = np.array([row["max_z_error"] for row in summary_rows])

# Linear fit versus 1/T^2:
# y ≈ a*(1/T^2) + b
torque_fit = np.polyfit(inv_t2_array, max_torque_array, 1)
pos_error_fit = np.polyfit(inv_t2_array, max_position_error_array, 1)
xy_error_fit = np.polyfit(inv_t2_array, max_xy_error_array, 1)
z_error_fit = np.polyfit(inv_t2_array, max_z_error_array, 1)

fit_filename = os.path.join(output_dir, "timing_sweep_fit_summary.txt")

with open(fit_filename, "w", encoding="utf-8") as f:
    f.write("Revised Study 1: Timing sweep with smoothstep fixed\n")
    f.write("Simple relation fitted against 1 / cycle_time^2\n\n")

    f.write("Form: y = a*(1/T^2) + b\n")
    f.write("T = cycle time in seconds\n\n")

    f.write(f"max_torque       ≈ {torque_fit[0]:.6f}*(1/T^2) + {torque_fit[1]:.6f}\n")
    f.write(f"max_3d_error     ≈ {pos_error_fit[0]:.6f}*(1/T^2) + {pos_error_fit[1]:.6f}\n")
    f.write(f"max_xy_error     ≈ {xy_error_fit[0]:.6f}*(1/T^2) + {xy_error_fit[1]:.6f}\n")
    f.write(f"max_z_error      ≈ {z_error_fit[0]:.6f}*(1/T^2) + {z_error_fit[1]:.6f}\n")

print(f"\nSaved fit summary to {fit_filename}")

print("\nSimple fitted relations:")
print("Form: y = a*(1/T^2) + b")
print(f"max_torque   ≈ {torque_fit[0]:.6f}*(1/T^2) + {torque_fit[1]:.6f}")
print(f"max_3d_error ≈ {pos_error_fit[0]:.6f}*(1/T^2) + {pos_error_fit[1]:.6f}")
print(f"max_xy_error ≈ {xy_error_fit[0]:.6f}*(1/T^2) + {xy_error_fit[1]:.6f}")
print(f"max_z_error  ≈ {z_error_fit[0]:.6f}*(1/T^2) + {z_error_fit[1]:.6f}")


# =========================
# 16. Save timing sweep plots
# =========================

# Plot A: cycle time vs maximum torque
plt.figure()
plt.plot(cycle_time_array, max_torque_array, marker="o", label="simulation")
plt.xlabel("Cycle Time (s)")
plt.ylabel("Maximum Torque Command (N m)")
plt.title("Timing Sweep: Cycle Time vs Maximum Torque")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_dir, "timing_sweep_cycle_time_vs_torque.png"), dpi=300)
plt.close()

# Plot B: cycle time vs maximum 3D position error
plt.figure()
plt.plot(cycle_time_array, max_position_error_array, marker="o", label="max 3D position error")
plt.xlabel("Cycle Time (s)")
plt.ylabel("Maximum 3D Tracking Error (m)")
plt.title("Timing Sweep: Cycle Time vs Maximum Position Error")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_dir, "timing_sweep_cycle_time_vs_3d_error.png"), dpi=300)
plt.close()

# Plot C: cycle time vs X/Y/Z/XY error
plt.figure()
plt.plot(cycle_time_array, np.array([row["max_x_error"] for row in summary_rows]), marker="o", label="max X error")
plt.plot(cycle_time_array, np.array([row["max_y_error"] for row in summary_rows]), marker="o", label="max Y error")
plt.plot(cycle_time_array, max_z_error_array, marker="o", label="max Z error")
plt.plot(cycle_time_array, max_xy_error_array, marker="o", label="max XY error")
plt.xlabel("Cycle Time (s)")
plt.ylabel("Maximum Tracking Error (m)")
plt.title("Timing Sweep: Cycle Time vs Directional Tracking Error")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_dir, "timing_sweep_cycle_time_vs_xyz_error.png"), dpi=300)
plt.close()

# Plot D: torque against 1/T^2
inv_t2_fit = np.linspace(np.min(inv_t2_array), np.max(inv_t2_array), 100)

plt.figure()
plt.plot(inv_t2_array, max_torque_array, "o", label="simulation")
plt.plot(inv_t2_fit, torque_fit[0] * inv_t2_fit + torque_fit[1], label="linear fit")
plt.xlabel("1 / Cycle Time^2 (1/s^2)")
plt.ylabel("Maximum Torque Command (N m)")
plt.title("Torque Trend vs 1 / Cycle Time^2")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_dir, "timing_sweep_torque_vs_inverse_time_squared.png"), dpi=300)
plt.close()

# Plot E: 3D error against 1/T^2
plt.figure()
plt.plot(inv_t2_array, max_position_error_array, "o", label="simulation")
plt.plot(inv_t2_fit, pos_error_fit[0] * inv_t2_fit + pos_error_fit[1], label="linear fit")
plt.xlabel("1 / Cycle Time^2 (1/s^2)")
plt.ylabel("Maximum 3D Tracking Error (m)")
plt.title("Tracking Error Trend vs 1 / Cycle Time^2")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_dir, "timing_sweep_error_vs_inverse_time_squared.png"), dpi=300)
plt.close()

print("\nSaved timing sweep plots:")
print(os.path.join(output_dir, "timing_sweep_cycle_time_vs_torque.png"))
print(os.path.join(output_dir, "timing_sweep_cycle_time_vs_3d_error.png"))
print(os.path.join(output_dir, "timing_sweep_cycle_time_vs_xyz_error.png"))
print(os.path.join(output_dir, "timing_sweep_torque_vs_inverse_time_squared.png"))
print(os.path.join(output_dir, "timing_sweep_error_vs_inverse_time_squared.png"))


# =========================
# 17. Save detailed plots for baseline and fastest case
# =========================
baseline_scale = 1.00
if baseline_scale in detailed_results:
    save_detailed_plots(
        detailed_results[baseline_scale],
        prefix="baseline_time_scale_1p00"
    )

fastest_scale = min(time_scales)
save_detailed_plots(
    detailed_results[fastest_scale],
    prefix=f"fastest_time_scale_{fastest_scale:.2f}".replace(".", "p")
)

print("\nSaved detailed plots for:")
print(os.path.join(output_dir, "baseline_time_scale_1p00_*.png"))
print(os.path.join(output_dir, f"fastest_time_scale_{fastest_scale:.2f}".replace(".", "p") + "_*.png"))


# =========================
# 18. Optional viewer replay for one case
# =========================
if show_viewer_after_sweep:
    if viewer_time_scale not in time_scales:
        raise ValueError("viewer_time_scale must be one of the values in time_scales.")

    print(f"\nShowing viewer replay for time_scale = {viewer_time_scale:.2f}")
    precomputed = precompute_trajectory_and_ik(viewer_time_scale)
    _ = run_single_case(precomputed, use_viewer=True)