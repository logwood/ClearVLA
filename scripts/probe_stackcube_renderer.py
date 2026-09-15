"""Bounded source-independent simulator initialization probe (no policy)."""

import argparse
import faulthandler
import json
import os


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("cpu", "none", "gpu"), default="cpu")
    parser.add_argument("--software-vulkan", action="store_true")
    args = parser.parse_args()
    faulthandler.enable()
    if args.backend != "gpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    if args.software_vulkan:
        driver = "/usr/share/vulkan/icd.d/lvp_icd.x86_64.json"
        if not os.path.isfile(driver):
            raise FileNotFoundError(driver)
        os.environ["VK_ICD_FILENAMES"] = driver
        os.environ["VK_DRIVER_FILES"] = driver
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401
    import sapien

    device = sapien.Device("cpu" if args.backend != "gpu" else "cuda")
    print(
        json.dumps({"stage": "device", "device": str(device), "can_render": device.can_render()}),
        flush=True,
    )
    env = gym.make(
        "StackCube-v1",
        num_envs=1,
        obs_mode="none",
        render_mode=None,
        render_backend=args.backend,
        sim_backend="physx_cpu",
        control_mode="pd_ee_delta_pose",
    )
    try:
        print("environment_created", flush=True)
        env.reset(seed=0)
        env.step(env.action_space.sample() * 0)
        print("one_step_passed", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
