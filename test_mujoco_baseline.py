import math
import csv
import time

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
data = mujoco.MjData(model)


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
# 4. Controller gains
# =========================
Kp = 120.0
Kd = 14.0
tau_limit = 20.0


# =========================
# 5. Delta robot geometry for inverse kinematics
# =========================
# These numbers match the current XML geometry.
# If you change the XML geometry later, update these constants too.

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
# 6. Cartesian pick-and-place trajectory
# =========================
# Raised working height.
# Horizontal task remains unchanged.

pick_high = np.array([-0.07, -0.06, 0.35])
pick_low  = np.array([-0.07, -0.06, 0.26])

place_high = np.array([0.08, 0.06, 0.35])
place_low  = np.array([0.08, 0.06, 0.26])

trajectory_segments = [
    (pick_high,   pick_low,    1.0),   # move down to pick
    (pick_low,    pick_low,    0.3),   # pick dwell
    (pick_low,    pick_high,   1.0),   # lift
    (pick_high,   place_high,  1.2),   # move horizontally
    (place_high,  place_low,   1.0),   # move down to place
    (place_low,   place_low,   0.3),   # place dwell
    (place_low,   place_high,  1.0),   # lift
    (place_high,  pick_high,   1.2),   # return
]

cycle_time = sum(seg[2] for seg in trajectory_segments)
total_cycles = 2.0
total_time = total_cycles * cycle_time
dt = model.opt.timestep


def smoothstep(s):
    """
    Smooth interpolation from 0 to 1.
    """
    return 3.0 * s**2 - 2.0 * s**3


def interpolate(p_start, p_end, s):
    return p_start + smoothstep(s) * (p_end - p_start)


def desired_platform_position(t):
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
# 7. Precompute desired trajectory and IK
# =========================
n_steps = int(math.ceil(total_time / dt))

time_cmd_array = np.zeros(n_steps)
ee_des_array = np.zeros((n_steps, 3))
q_des_array = np.zeros((n_steps, 3))
ik_residual_array = np.zeros((n_steps, 3))

print("\nComputing initial IK...")
q_settle, settle_residual = inverse_kinematics_delta_bruteforce(
    pick_high,
    np.zeros(3)
)

print("Initial q_settle:", q_settle)
print("Settling IK residual:", settle_residual)

print("\nPrecomputing desired XYZ trajectory and fast IK...")
precompute_start = time.perf_counter()

q_hint = q_settle.copy()
fallback_count = 0

for k in range(n_steps):
    t_cmd = k * dt
    ee_des = desired_platform_position(t_cmd)

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

precompute_end = time.perf_counter()

print(f"Precompute finished in {precompute_end - precompute_start:.2f} seconds.")
print(f"Number of precomputed steps: {n_steps}")
print(f"Fallback used on {fallback_count} / {n_steps} timesteps.")

max_ik_residual = np.max(ik_residual_array)
print(f"Maximum IK residual during precompute: {max_ik_residual:.6e} m")

if max_ik_residual > 1e-3:
    print("Warning: Some desired platform positions may be difficult to reach.")


# =========================
# 8. Data logs
# =========================
time_log = []
q_des_log = []
q_actual_log = []
tau_log = []

ee_des_log = []
ee_actual_log = []
ik_residual_log = []


# =========================
# 9. Viewer and replay settings
# =========================
render_every = 20
pre_motion_pause = 3.0

# Replay is display-only. Physics is not recomputed during replay.
replay_stride = 1
replay_dt = dt * render_every * replay_stride

qpos_history = []


# =========================
# 10. Simulation loop + infinite replay
# =========================
settle_time = 1.0
step_count = 0

print("\nOpening viewer...")

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

    print(f"Holding initial pose for {pre_motion_pause:.1f} seconds before motion...")

    pause_start = time.perf_counter()

    while viewer.is_running() and (time.perf_counter() - pause_start) < pre_motion_pause:
        viewer.sync()
        time.sleep(0.03)

    print("Running finite physics simulation...")

    for k in range(n_steps):
        if not viewer.is_running():
            break

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

        step_count += 1
        if step_count % render_every == 0:
            viewer.sync()
            qpos_history.append(data.qpos.copy())

        q_actual_after = np.zeros(3)
        for i in range(3):
            q_actual_after[i] = data.qpos[qpos_adr[i]]

        time_log.append(time_cmd_array[k])
        q_des_log.append(q_des.copy())
        q_actual_log.append(q_actual_after.copy())
        tau_log.append(tau_cmd.copy())

        ee_des_log.append(ee_des.copy())
        ee_actual_log.append(data.xpos[ee_body_id].copy())
        ik_residual_log.append(ik_residual.copy())

    if viewer.is_running() and len(qpos_history) > 0:
        print("\nFinite simulation complete.")
        print("Now replaying saved motion forever.")
        print("Close the viewer window manually when you are ready to save CSV and plots.")

        while viewer.is_running():
            for qpos_frame in qpos_history[::replay_stride]:
                if not viewer.is_running():
                    break

                data.qpos[:] = qpos_frame
                data.qvel[:] = 0.0
                mujoco.mj_forward(model, data)

                viewer.sync()
                time.sleep(replay_dt)


# =========================
# 11. Convert logs to arrays
# =========================
time_log = np.array(time_log)
q_des_log = np.array(q_des_log)
q_actual_log = np.array(q_actual_log)
tau_log = np.array(tau_log)

ee_des_log = np.array(ee_des_log)
ee_actual_log = np.array(ee_actual_log)
ik_residual_log = np.array(ik_residual_log)


# =========================
# 12. Save CSV data
# =========================
if len(time_log) == 0:
    print("\nNo simulation data was logged. CSV and plots were not saved.")
else:
    csv_filename = "delta_pick_place_log.csv"

    with open(csv_filename, "w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow([
            "time",

            "ee_des_x", "ee_des_y", "ee_des_z",
            "ee_actual_x", "ee_actual_y", "ee_actual_z",

            "q_des_1", "q_des_2", "q_des_3",
            "q_actual_1", "q_actual_2", "q_actual_3",

            "tau_1", "tau_2", "tau_3",

            "ik_residual_1", "ik_residual_2", "ik_residual_3",
        ])

        for i in range(len(time_log)):
            writer.writerow([
                time_log[i],

                ee_des_log[i, 0], ee_des_log[i, 1], ee_des_log[i, 2],
                ee_actual_log[i, 0], ee_actual_log[i, 1], ee_actual_log[i, 2],

                q_des_log[i, 0], q_des_log[i, 1], q_des_log[i, 2],
                q_actual_log[i, 0], q_actual_log[i, 1], q_actual_log[i, 2],

                tau_log[i, 0], tau_log[i, 1], tau_log[i, 2],

                ik_residual_log[i, 0], ik_residual_log[i, 1], ik_residual_log[i, 2],
            ])

    print(f"\nSaved data to {csv_filename}")

    # Plot 1: Desired vs actual motor angles
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
    plt.savefig("motor_tracking_pick_place.png", dpi=300)

    # Plot 2: Desired vs actual end-effector Z
    plt.figure()
    plt.plot(time_log, ee_des_log[:, 2], label="desired z")
    plt.plot(time_log, ee_actual_log[:, 2], "--", label="actual z")
    plt.xlabel("Time (s)")
    plt.ylabel("End-Effector Z Position (m)")
    plt.title("End-Effector Vertical Pick-and-Place Motion")
    plt.legend()
    plt.grid(True)
    plt.savefig("end_effector_z_pick_place.png", dpi=300)

    # Plot 3: Desired vs actual end-effector X
    plt.figure()
    plt.plot(time_log, ee_des_log[:, 0], label="desired x")
    plt.plot(time_log, ee_actual_log[:, 0], "--", label="actual x")
    plt.xlabel("Time (s)")
    plt.ylabel("End-Effector X Position (m)")
    plt.title("End-Effector X Pick-and-Place Motion")
    plt.legend()
    plt.grid(True)
    plt.savefig("end_effector_x_pick_place.png", dpi=300)

    # Plot 4: Desired vs actual end-effector Y
    plt.figure()
    plt.plot(time_log, ee_des_log[:, 1], label="desired y")
    plt.plot(time_log, ee_actual_log[:, 1], "--", label="actual y")
    plt.xlabel("Time (s)")
    plt.ylabel("End-Effector Y Position (m)")
    plt.title("End-Effector Y Pick-and-Place Motion")
    plt.legend()
    plt.grid(True)
    plt.savefig("end_effector_y_pick_place.png", dpi=300)

    # Plot 3: End-effector XY path
    plt.figure()
    plt.plot(ee_des_log[:, 0], ee_des_log[:, 1], label="desired XY path")
    plt.plot(ee_actual_log[:, 0], ee_actual_log[:, 1], "--", label="actual XY path")
    plt.xlabel("X Position (m)")
    plt.ylabel("Y Position (m)")
    plt.title("End-Effector XY Pick-and-Place Path")
    plt.legend()
    plt.grid(True)
    plt.axis("equal")
    plt.savefig("end_effector_xy_path.png", dpi=300)

    # Plot 4: Torque command
    plt.figure()
    plt.plot(time_log, tau_log[:, 0], label="tau1")
    plt.plot(time_log, tau_log[:, 1], label="tau2")
    plt.plot(time_log, tau_log[:, 2], label="tau3")
    plt.xlabel("Time (s)")
    plt.ylabel("Torque Command")
    plt.title("Actuator Torque Commands During Pick-and-Place")
    plt.legend()
    plt.grid(True)
    plt.savefig("actuator_torque_pick_place.png", dpi=300)

    # Plot 5: IK residual
    plt.figure()
    plt.plot(time_log, ik_residual_log[:, 0], label="arm1 IK residual")
    plt.plot(time_log, ik_residual_log[:, 1], label="arm2 IK residual")
    plt.plot(time_log, ik_residual_log[:, 2], label="arm3 IK residual")
    plt.xlabel("Time (s)")
    plt.ylabel("IK Residual (m)")
    plt.title("Inverse Kinematics Residual")
    plt.legend()
    plt.grid(True)
    plt.savefig("ik_residual_pick_place.png", dpi=300)

    print("Saved plots:")
    print("motor_tracking_pick_place.png")
    print("end_effector_z_pick_place.png")
    print("end_effector_x_pick_place.png")
    print("end_effector_y_pick_place.png")
    print("end_effector_xy_path.png")
    print("actuator_torque_pick_place.png")
    print("ik_residual_pick_place.png")