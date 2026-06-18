
# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0  
#!/usr/bin/env python3
# main.py
import os

project_root = os.path.dirname(os.path.abspath(__file__))
os.environ["PROJECT_ROOT"] = project_root

import argparse
import time
import sys
import signal
import threading
import atexit
from collections import deque
from typing import Optional

import torch
import gymnasium as gym
from pathlib import Path

# Isaac Lab AppLauncher
from isaaclab.app import AppLauncher

from teleimager.image_server import run_isaacsim_server
from dds.dds_create import create_dds_objects, create_dds_objects_replay

# ---- Helper utilities -------------------------------------------------------

def log_section(title: str, char: str = "=", width: int = 60) -> None:
    """Print a consistent section header."""
    print(f"\n{char * width}")
    print(title)
    print(f"{char * width}")


def _safe_close_with_timeout(name: str, close_fn, timeout: float = 5.0) -> None:
    """Call *close_fn* in a daemon thread, waiting *timeout* seconds."""
    t = threading.Thread(target=close_fn, daemon=True)
    t.start()
    t.join(timeout=timeout)
    if t.is_alive():
        print(f"[main] WARNING: {name} timed out after {timeout}s")


class FrequencyTracker:
    """Track loop execution frequency with a moving window."""

    def __init__(self, window_size: int = 100):
        self._times: deque = deque(maxlen=window_size)
        self._start_time = time.time()
        self._count = 0

    def update(self) -> float:
        """Record a tick; return the instantaneous dt in seconds."""
        now = time.time()
        self._times.append(now)
        dt = now - (self._times[-2] if len(self._times) >= 2 else self._start_time)
        self._count += 1
        return dt

    @property
    def count(self) -> int:
        return self._count

    @property
    def elapsed(self) -> float:
        return time.time() - self._start_time

    def report(self) -> str:
        elapsed = max(self.elapsed, 1e-9)
        overall_hz = self._count / elapsed
        lines = [
            f"=== Loop frequency statistics ===",
            f"loop count: {self._count}",
            f"running time: {elapsed:.2f}s",
            f"overall average: {overall_hz:.2f} Hz",
        ]
        if len(self._times) >= 2:
            dts = [self._times[i] - self._times[i - 1] for i in range(1, len(self._times))]
            avg_dt = sum(dts) / len(dts)
            min_dt, max_dt = min(dts), max(dts)
            moving_hz = 1.0 / avg_dt if avg_dt > 0 else 0
            lines.append(f"moving average: {moving_hz:.2f} Hz (last {len(dts)} ticks)")
            lines.append(f"frequency range: {1.0 / max_dt:.2f} - {1.0 / min_dt:.2f} Hz" if min_dt > 0 else "frequency range: N/A")
            lines.append(f"avg loop time: {avg_dt * 1000:.2f} ms")
        lines.append("=" * 40)
        return "\n".join(lines)

# ---- Argument parsing -------------------------------------------------------
parser = argparse.ArgumentParser(description="Unitree Simulation")
parser.add_argument("--task", type=str, default="Isaac-PickPlace-G129-Head-Waist-Fix", help="task name")
parser.add_argument("--action_source", type=str, default="dds", 
                   choices=["dds", "file", "trajectory", "policy", "replay","dds_wholebody"], 
                   help="Action source")


parser.add_argument("--robot_type", type=str, default="g129", help="robot type")
parser.add_argument("--enable_dex1_dds", action="store_true", help="enable gripper DDS")
parser.add_argument("--enable_dex3_dds", action="store_true", help="enable dexterous hand DDS")
parser.add_argument("--enable_inspire_dds", action="store_true", help="enable inspire hand DDS")
parser.add_argument("--stats_interval", type=float, default=10.0, help="statistics print interval (seconds)")

parser.add_argument("--file_path", type=str, default="/home/unitree/Code/xr_teleoperate/teleop/utils/data", help="file path (when action_source=file)")
parser.add_argument("--generate_data_dir", type=str, default="./data", help="save data dir")
parser.add_argument("--generate_data", action="store_true", default=False, help="generate data")
parser.add_argument("--rerun_log", action="store_true", default=False, help="rerun log")
parser.add_argument("--replay_data",  action="store_true", default=False, help="replay data")

parser.add_argument("--modify_light",  action="store_true", default=False, help="modify light")
parser.add_argument("--modify_camera",  action="store_true", default=False,    help="modify camera")

# performance analysis parameters
parser.add_argument("--step_hz", type=int, default=100, help="control frequency")
parser.add_argument("--enable_profiling", action="store_true", default=True, help="enable performance analysis")
parser.add_argument("--profile_interval", type=int, default=500, help="performance analysis report interval (steps)")

parser.add_argument("--model_path", type=str, default="assets/model/policy.onnx", help="model path")
parser.add_argument("--reward_interval", type=int, default=10, help="step interval for reward calculation")
parser.add_argument("--enable_wholebody_dds", action="store_true", default=False, help="enable wh dds")

parser.add_argument("--physics_dt", type=float, default=None, help="physics time step, e.g., 0.005")
parser.add_argument("--render_interval", type=int, default=None, help="render interval steps (>=1)")
parser.add_argument("--camera_write_interval", type=int, default=None, help="camera write interval steps (>=1)")


parser.add_argument("--no_render",action="store_true",default=False,help="disable rendering updates entirely (overrides render interval)",)
parser.add_argument("--public_ip",type=str,default="127.0.0.1",help="public ip")
parser.add_argument("--livestream_type", type=int, default=2, help="livestream type (0: no livestream, 1: WebRTC public network, 2:  WebRTC private network)")

parser.add_argument("--solver_iterations", type=int, default=None, help="physx solver iteration count (e.g., 4)")
parser.add_argument("--gravity_z", type=float, default=None, help="override gravity z (e.g., -9.8)")
parser.add_argument("--skip_cvtcolor", action="store_true", default=False, help="skip cv2.cvtColor if upstream already BGR")

parser.add_argument("--camera_jpeg", action="store_true", default=True, help="enable JPEG compression for camera frames")
parser.add_argument("--camera_jpeg_quality", type=int, default=85, help="JPEG quality (1-100)")

parser.add_argument("--physx_substeps", type=int, default=None, help="physx substeps per step")
parser.add_argument("--camera_include", type=str, default="front_camera,left_wrist_camera,right_wrist_camera", help="comma-separated camera names to enable")
parser.add_argument("--camera_exclude", type=str, default="world_camera", help="comma-separated camera names to disable")

parser.add_argument("--env_reward_interval", type=int, default=5, help="environment reward compute interval (steps)")
parser.add_argument("--seed", type=int, default=42, help="environment seed")
# add AppLauncher parameters
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.no_render:
    os.environ["LIVESTREAM"] = str(args_cli.livestream_type)
    os.environ["PUBLIC_IP"] = args_cli.public_ip
else:
    os.environ["LIVESTREAM"] = "0"

if args_cli.enable_dex3_dds and args_cli.enable_dex1_dds and args_cli.enable_inspire_dds:
    print("Error: enable_dex3_dds and enable_dex1_dds and enable_inspire_dds cannot be enabled at the same time")
    print("Please select one of the options")
    sys.exit(1)


import pinocchio  # required by Isaac Sim (side-effect import, do not remove)
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from layeredcontrol.robot_control_system import (
    RobotController,
    ControlConfig,
)

from dds.reset_pose_dds import *
import tasks
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

from tools.augmentation_utils import (
    update_light,
    batch_augment_cameras_by_name,
)

from tools.data_json_load import sim_state_to_json
from dds.sim_state_dds import *
from action_provider.create_action_provider import create_action_provider
from tools.get_stiffness import get_robot_stiffness_from_env
from tools.get_reward import get_step_reward_value

# ---- Signal handling --------------------------------------------------------

def setup_signal_handlers(shutdown_event: threading.Event) -> None:
    """Set signal handlers — ONLY set a shutdown event, do NOT do any blocking work."""

    def signal_handler(signum, frame):
        sig_name = signal.Signals(signum).name
        print(f"\n[signal] received {sig_name}, requesting graceful shutdown...")
        shutdown_event.set()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

# ---- Environment helpers ----------------------------------------------------

def _configure_sim_params(env, args_cli) -> None:
    """Apply physics, rendering, and solver config from CLI args."""
    # physics dt
    if args_cli.physics_dt is not None:
        try:
            env.sim.set_substep_time(args_cli.physics_dt)
            print(f"[sim] physics dt={args_cli.physics_dt}")
        except Exception:
            try:
                env.sim.dt = args_cli.physics_dt
                print(f"[sim] physics dt={args_cli.physics_dt} (fallback)")
            except Exception as e:
                print(f"[sim] failed to set physics dt: {e}")

    # rendering
    headless_mode = bool(getattr(args_cli, "headless", False))
    render_interval = None
    if args_cli.render_interval is not None:
        try:
            render_interval = max(1, int(args_cli.render_interval))
        except Exception as e:
            print(f"[sim] invalid render_interval: {e}")

    try:
        if args_cli.no_render:
            env.sim.render_interval = 1_000_000
            env.sim.render_mode = "offscreen"
            print("[sim] rendering disabled (--no_render)")
        elif headless_mode:
            env.sim.render_mode = "offscreen"
            env.sim.render_interval = render_interval or 1
            print(f"[sim] headless offscreen, render every {env.sim.render_interval} steps")
        elif render_interval is not None:
            env.sim.render_interval = render_interval
            print(f"[sim] render_interval={env.sim.render_interval}")
    except Exception as e:
        print(f"[sim] render config failed: {e}")

    # camera write interval
    if args_cli.camera_write_interval is not None:
        try:
            import tasks.common_observations.camera_state as cam_state
            cam_state._camera_cache['write_interval_steps'] = max(1, int(args_cli.camera_write_interval))
            print(f"[camera] write interval={cam_state._camera_cache['write_interval_steps']} steps")
        except Exception as e:
            print(f"[camera] write interval failed: {e}")

    # physx
    try:
        if args_cli.solver_iterations is not None:
            env.sim.physx.solver_iteration_count = int(args_cli.solver_iterations)
            print(f"[sim] solver_iterations={env.sim.physx.solver_iteration_count}")
        if args_cli.physx_substeps is not None:
            try:
                env.sim.physx.substeps = int(args_cli.physx_substeps)
            except Exception:
                try:
                    env.sim.set_substeps(int(args_cli.physx_substeps))
                except Exception:
                    pass
            print(f"[sim] physx_substeps={args_cli.physx_substeps}")
        if args_cli.gravity_z is not None:
            env.sim.physx.gravity = (0.0, 0.0, float(args_cli.gravity_z))
            print(f"[sim] gravity={env.sim.physx.gravity}")
    except Exception as e:
        print(f"[sim] physx config failed: {e}")


def _configure_camera_params(env, args_cli) -> None:
    """Apply camera JPEG/allowlist/exclusion config."""
    if args_cli.skip_cvtcolor:
        os.environ["CAMERA_SKIP_CVTCOLOR"] = "1"

    try:
        import tasks.common_observations.camera_state as cam_state

        enable_jpeg = bool(args_cli.camera_jpeg) or (os.getenv("CAMERA_JPEG") == "1")
        quality = int(args_cli.camera_jpeg_quality if args_cli.camera_jpeg
                      else os.getenv("CAMERA_JPEG_QUALITY", args_cli.camera_jpeg_quality))
        cam_state.set_writer_options(enable_jpeg=enable_jpeg, jpeg_quality=quality,
                                     skip_cvtcolor=args_cli.skip_cvtcolor)

        include = [n.strip() for n in (args_cli.camera_include or "").split(',') if n.strip()]
        exclude = [n.strip() for n in (args_cli.camera_exclude or "").split(',') if n.strip()]
        try:
            cam_state.set_camera_allowlist(include)
        except Exception:
            pass

        sensors_dict = getattr(env.scene, "sensors", {})
        for name, sensor in sensors_dict.items():
            if "camera" not in name.lower():
                continue
            if exclude and name in exclude:
                for attr_name in ("enabled", "is_enabled"):
                    if hasattr(sensor, attr_name):
                        try:
                            setattr(sensor, attr_name, False)
                        except Exception:
                            pass
                for meth in ("set_active", "disable", "pause"):
                    if hasattr(sensor, meth):
                        try:
                            getattr(sensor, meth)(False)
                        except Exception:
                            pass
                for attr_name in ("update_period", "_update_period"):
                    if hasattr(sensor, attr_name):
                        try:
                            setattr(sensor, attr_name, 1e6)
                        except Exception:
                            pass
            elif include and name not in include:
                for attr_name in ("update_period", "_update_period"):
                    if hasattr(sensor, attr_name):
                        try:
                            setattr(sensor, attr_name, 1e6)
                        except Exception:
                            pass
    except Exception as e:
        print(f"[camera] config failed: {e}")


def _setup_live_dds(args_cli, env):
    """Create image server and live DDS objects. Returns (image_server, reset_pose_dds, sim_state_dds, dds_manager)."""
    log_section("create image server")
    image_server = run_isaacsim_server()

    log_section("create dds")
    reset_pose_dds, sim_state_dds, dds_manager = create_dds_objects(args_cli, env)
    return image_server, reset_pose_dds, sim_state_dds, dds_manager


def _setup_replay_dds_and_data(args_cli, env):
    """Create replay DDS objects and load data list. Returns (data_idx, data_json_list)."""
    log_section("create dds (replay)")
    create_dds_objects_replay(args_cli, env)

    log_section("get data json list")
    from tools.data_json_load import get_data_json_list
    data_json_list = get_data_json_list(args_cli.file_path)
    if args_cli.action_source != "replay":
        args_cli.action_source = "replay"
    return 0, data_json_list


def _determine_action_source(args_cli, control_config) -> None:
    """Adjust action_source for wholebody tasks."""
    if not args_cli.replay_data and ("Wholebody" in args_cli.task or args_cli.enable_wholebody_dds):
        args_cli.action_source = "dds_wholebody"
        args_cli.enable_wholebody_dds = True
        control_config.use_rl_action_mode = True


def _handle_reset_pose(env, env_cfg, reset_pose_dds, args_cli) -> None:
    """Check for and process a reset-pose command from DDS."""
    reset_pose_cmd = reset_pose_dds.get_reset_pose_command()
    if reset_pose_cmd is None:
        return
    reset_category = reset_pose_cmd.get("reset_category")
    wholebody = args_cli.enable_wholebody_dds
    if (wholebody and reset_category in ('1', '2')) or (not wholebody and reset_category == '1'):
        print("reset object")
        env_cfg.event_manager.trigger("reset_object_self", env)
        reset_pose_dds.write_reset_pose_command(-1)
    elif reset_category == '2' and not wholebody:
        print("reset all")
        env_cfg.event_manager.trigger("reset_all_self", env)
        reset_pose_dds.write_reset_pose_command(-1)


def _handle_replay_step(env, action_provider, data_json_list, data_idx, args_cli) -> Optional[int]:
    """Process one replay-data step. Returns updated data_idx or None on error."""
    if action_provider.get_start_loop() and data_idx < len(data_json_list):
        print(f"data_idx: {data_idx}")
        sim_state, task_name = action_provider.load_data(data_json_list[data_idx])
        if task_name != args_cli.task:
            raise ValueError(
                f"The {task_name} in the dataset differs from the running task {args_cli.task}.")
        env.reset_to(sim_state, torch.tensor([0], device=env.device), is_relative=True)
        env.sim.reset()
        time.sleep(1)
        action_provider.start_replay()
        return data_idx + 1
    return data_idx

# ---- Main -------------------------------------------------------------------

def main():
    """Main entry point."""
    # --- process group (so signals propagate to children) ---
    try:
        os.setpgrp()
        pgid = os.getpgrp()
        print(f"Process group: {pgid}")
        atexit.register(lambda: os.killpg(pgid, signal.SIGTERM)
                        if os.getpgrp() == pgid else None)
    except Exception as e:
        print(f"Failed to set process group: {e}")

    log_section(f"robot control system started — Task: {args_cli.task}, "
                f"Action: {args_cli.action_source}")

    # --- parse & create environment ---
    try:
        env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
        env_cfg.env_name = args_cli.task
    except Exception as e:
        print(f"Failed to parse env config: {e}")
        return

    print("\ncreate environment...")
    try:
        env_cfg.seed = args_cli.seed
        env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
        env.seed(args_cli.seed)

        # list sensors
        try:
            sensors_dict = getattr(env.scene, "sensors", {})
            if sensors_dict:
                print("Sensors in environment:")
                for name, sensor in sensors_dict.items():
                    print(f"  {name}: {sensor}")
                print("=" * 60)
        except Exception as e:
            print(f"[sim] failed to list sensors: {e}")

        # reward interval
        try:
            env._reward_interval = max(1, int(args_cli.env_reward_interval))
            env._reward_counter = 0
            env._reward_last = None
            print(f"[env] reward interval: {env._reward_interval} steps")
        except Exception as e:
            print(f"[env] reward interval failed: {e}")

        _configure_sim_params(env, args_cli)
        _configure_camera_params(env, args_cli)

        print("environment created successfully.")
    except Exception as e:
        print(f"Failed to create environment: {e}")
        return

    # --- robot stiffness ---
    log_section("Getting robot stiffness parameters")
    try:
        stiffness_data = get_robot_stiffness_from_env(env)
        if stiffness_data:
            print("✅ robot parameters obtained")
        else:
            print("⚠️  robot parameters unavailable, retry after reset")
    except Exception as e:
        print(f"⚠️  robot parameter error: {e}")

    # --- rendering notice ---
    if not getattr(args_cli, "headless", False) and not args_cli.no_render:
        print("\n*** Please left-click on the Sim window to activate rendering. ***\n")
    else:
        print("\n*** Running without GUI; rendering handled offscreen. ***\n")

    # --- scene modifications & reset ---
    if args_cli.modify_light:
        update_light(
            prim_path="/World/light",
            color=(0.75, 0.75, 0.75),
            intensity=500.0,
            radius=0.1,
            enabled=True,
            cast_shadows=True,
        )
    if args_cli.modify_camera:
        batch_augment_cameras_by_name(
            names=["front_cam"],
            focal_length=3.0,
            horizontal_aperture=22.0,
            vertical_aperture=16.0,
            exposure=0.8,
            focus_distance=1.2,
        )
    env.sim.reset()
    env.reset()

    # --- control config ---
    try:
        control_config = ControlConfig(
            step_hz=args_cli.step_hz,
            replay_mode=args_cli.replay_data,
        )
    except Exception as e:
        print(f"Failed to create control config: {e}")
        return

    # --- DDS / replay setup ---
    data_json_list = []
    data_idx = 0
    image_server = None
    dds_manager = None

    if not args_cli.replay_data:
        try:
            image_server, reset_pose_dds, sim_state_dds, dds_manager = _setup_live_dds(args_cli, env)
        except Exception as e:
            print(f"DDS setup failed: {e}")
            return
    else:
        try:
            data_idx, data_json_list = _setup_replay_dds_and_data(args_cli, env)
        except Exception as e:
            print(f"Replay DDS setup failed: {e}")
            return

    # --- action provider ---
    print(f"\ncreate action provider: {args_cli.action_source}...")
    try:
        _determine_action_source(args_cli, control_config)
        action_provider = create_action_provider(env, args_cli)
        if action_provider is None:
            print("action provider creation failed")
            return
    except Exception as e:
        print(f"Failed to create action provider: {e}")
        return

    # --- controller ---
    log_section("create controller")
    controller = RobotController(env, control_config)
    controller.set_action_provider(action_provider)

    if args_cli.enable_profiling:
        controller.set_profiling(True, args_cli.profile_interval)
        print(f"profiling enabled, every {args_cli.profile_interval} steps")
    else:
        controller.set_profiling(False)
        print("profiling disabled")

    # --- shutdown coordination ---
    _shutdown_requested = threading.Event()
    setup_signal_handlers(_shutdown_requested)

    print(
        "Note: The DDS in Sim transmits messages on channel 1. "
        "Please ensure that other DDS instances use the same channel "
        "by setting: ChannelFactoryInitialize(1)."
    )

    freq_tracker = FrequencyTracker()
    try:
        log_section("start controller")
        controller.start()

        with torch.inference_mode():
            while (
                simulation_app.is_running()
                and controller.is_running
                and not _shutdown_requested.is_set()
            ):
                freq_tracker.update()

                if not args_cli.replay_data:
                    # --- live DDS path ---
                    try:
                        env_state = env.scene.get_state()
                        env_state_json = sim_state_to_json(env_state)
                        sim_state = {"init_state": env_state_json, "task_name": args_cli.task}
                        sim_state_dds.write_sim_state_data(sim_state)
                    except Exception as e:
                        print(f"Failed to get/write env state: {e}")
                        raise

                    try:
                        _handle_reset_pose(env, env_cfg, reset_pose_dds, args_cli)
                    except Exception as e:
                        print(f"Reset pose error: {e}")
                        raise
                else:
                    # --- replay path ---
                    try:
                        data_idx = _handle_replay_step(
                            env, action_provider, data_json_list, data_idx, args_cli
                        )
                    except Exception as e:
                        print(f"Replay step error: {e}")
                        raise

                controller.step()

                if _shutdown_requested.is_set():
                    print("\n[main] shutdown requested, exiting loop...")
                    break

                # periodic stats
                if freq_tracker.elapsed % args_cli.stats_interval < (
                    freq_tracker.elapsed - getattr(freq_tracker, "_last_report", 0)
                ):
                    freq_tracker._last_report = freq_tracker.elapsed
                    print(f"\n{freq_tracker.report()}")

                if env.sim.is_stopped():
                    print("\nenvironment stopped")
                    break

    except KeyboardInterrupt:
        print("\n[main] KeyboardInterrupt received, shutting down...")
        _shutdown_requested.set()
    except Exception as e:
        print(f"\n[main] program exception: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # --- Graceful cleanup ---
        print("\n[main] cleaning up resources...")
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        signal.signal(signal.SIGTERM, signal.SIG_DFL)

        try:
            controller.cleanup()
            print("[main] controller cleanup done")
        except Exception as e:
            print(f"[main] controller cleanup error: {e}")

        if image_server is not None:
            try:
                image_server.stop()
                print("[main] image server stopped")
            except Exception as e:
                print(f"[main] image server stop error: {e}")

        if dds_manager is not None:
            try:
                dds_manager.stop_all_communication()
                print("[main] DDS communication stopped")
            except Exception as e:
                print(f"[main] DDS stop error: {e}")

        _safe_close_with_timeout("env.close()", lambda: env.close())
        print("[main] cleanup completed")

# ---- Entry point ------------------------------------------------------------

if __name__ == "__main__":
    try:
        main()
    finally:
        print("[main] Performing final cleanup...")

        import os as _os
        import subprocess as _subprocess
        import signal as _signal
        import time as _time

        current_pid = _os.getpid()
        print(f"[main] Current main process PID: {current_pid}")

        _safe_close_with_timeout("simulation_app.close()", lambda: simulation_app.close())

        # Kill remaining child processes
        try:
            result = _subprocess.run(
                ['pgrep', '-f', 'sim_main.py'],
                capture_output=True, text=True, timeout=3,
            )
            if result.returncode == 0:
                child_pids = [
                    int(pid) for pid in result.stdout.strip().split('\n')
                    if pid and pid != str(current_pid)
                ]
                for pid in child_pids:
                    try:
                        print(f"[main] Terminating child: {pid}")
                        _os.kill(pid, _signal.SIGTERM)
                    except Exception:
                        pass
                if child_pids:
                    _time.sleep(1)
                    for pid in child_pids:
                        try:
                            _os.kill(pid, 0)  # check if alive
                            print(f"[main] Force killing: {pid}")
                            _os.kill(pid, _signal.SIGKILL)
                        except OSError:
                            pass  # already dead
        except Exception as e:
            print(f"[main] Child cleanup error: {e}")

        print("[main] Program exit completed")


# python sim_main.py --device cpu  --enable_cameras  --task  Isaac-PickPlace-Cylinder-G129-Dex1-Joint   --enable_dex1_dds --robot_type g129
# python sim_main.py --device cpu  --enable_cameras  --task Isaac-PickPlace-Cylinder-G129-Dex3-Joint    --enable_dex3_dds --robot_type g129
# python sim_main.py --device cpu  --enable_cameras  --task Isaac-PickPlace-Cylinder-G129-Inspire-Joint    --enable_inspire_dds --robot_type g129

# python sim_main.py --device cpu  --enable_cameras  --task Isaac-PickPlace-RedBlock-G129-Dex1-Joint     --enable_dex1_dds --robot_type g129
# python sim_main.py --device cpu  --enable_cameras  --task Isaac-PickPlace-RedBlock-G129-Dex3-Joint    --enable_dex3_dds --robot_type g129
# python sim_main.py --device cpu  --enable_cameras  --task  Isaac-PickPlace-RedBlock-G129-Inspire-Joint    --enable_inspire_dds --robot_type g129


# python sim_main.py --device cpu  --enable_cameras  --task Isaac-Stack-RgyBlock-G129-Dex1-Joint     --enable_dex1_dds --robot_type g129
# python sim_main.py --device cpu  --enable_cameras  --task Isaac-Stack-RgyBlock-G129-Dex3-Joint     --enable_dex3_dds --robot_type g129
# python sim_main.py --device cpu  --enable_cameras  --task Isaac-Stack-RgyBlock-G129-Inspire-Joint     --enable_inspire_dds --robot_type g129




# python sim_main.py --device cpu  --enable_cameras  --task Isaac-Move-Cylinder-G129-Dex1-Wholebody  --robot_type g129 --enable_dex1_dds 
# python sim_main.py --device cpu  --enable_cameras  --task Isaac-Move-Cylinder-G129-Dex3-Wholebody  --robot_type g129 --enable_dex3_dds 
# python sim_main.py --device cpu  --enable_cameras  --task Isaac-Move-Cylinder-G129-Inspire-Wholebody  --robot_type g129 --enable_inspire_dds 


# python sim_main.py --device cpu  --enable_cameras  --task Isaac-PickPlace-Cylinder-H12-27dof-Inspire-Joint  --enable_inspire_dds --robot_type h1_2
# python sim_main.py --device cpu  --enable_cameras  --task Isaac-PickPlace-RedBlock-H12-27dof-Inspire-Joint  --enable_inspire_dds --robot_type h1_2
# python sim_main.py --device cpu  --enable_cameras  --task Isaac-Stack-RgyBlock-H12-27dof-Inspire-Joint --enable_inspire_dds --robot_type h1_2
