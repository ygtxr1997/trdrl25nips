import argparse
import json
import os
import pathlib
import random
import sys

import numpy as np
import torch

from wrapper.env_evaluator import RobosuiteSocketEvaluator


sys.stdout = open(sys.stdout.fileno(), mode="w", buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode="w", buffering=1)


def _parse_camera_names(value: str):
    names = [x.strip() for x in str(value).split(",") if x.strip()]
    if len(names) == 0:
        raise argparse.ArgumentTypeError("camera_names cannot be empty.")
    return tuple(names)


def _build_policy_url(policy_url: str, port: int) -> str:
    url = str(policy_url).strip()
    if url == "":
        url = f"http://localhost:{int(port)}"
    return url


def _get_demo_transition_path(env_name: str) -> str:
    root_dir = "/mnt/dongxu-fs1/data-hdd/geyuan/code/trdrl25nips/robosuite/"
    return os.path.join(
        root_dir,
        "generate",
        f"{env_name}_transitions_10trajectory_sparse.npy",
    )


def _load_demo_transitions(path: str):
    loaded = np.load(path, allow_pickle=True)
    if isinstance(loaded, np.lib.npyio.NpzFile):
        if len(loaded.files) != 1:
            raise ValueError(
                f"Expected exactly one array in npz, got keys={loaded.files}"
            )
        return loaded[loaded.files[0]]
    return loaded


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate Robosuite policy via socket API (/init, /reset, /step).")

    parser.add_argument("-c", "--checkpoint", default=None, help="Not used, kept for compatibility.")
    parser.add_argument("-o", "--output_dir", required=True, help="Directory to save eval json.")
    parser.add_argument("-e", "--env_name", required=True, help="Robosuite env name, e.g. Door / Stack.")
    parser.add_argument("-d", "--device", default="cuda:0", help="Not used, kept for compatibility.")
    parser.add_argument("-p", "--port", type=int, default=6006, help="Fallback port if policy_url is empty.")
    parser.add_argument("--policy_url", type=str, default="", help="Policy server URL, e.g. http://localhost:6006",)

    parser.add_argument("--test_cnt", type=int, default=20)
    parser.add_argument("--test_start_seed", type=int, default=10000)
    parser.add_argument("--test_max_steps", type=int, default=500)
    parser.add_argument("--send_per_frames", type=int, default=1)
    parser.add_argument(
        "--num_obs_steps",
        type=int,
        default=None,
        help="Observation window length (Ts). Default None -> same as send_per_frames.",
    )
    parser.add_argument("--request_timeout", type=float, default=30.0)
    parser.add_argument("--save_video", action="store_true", help="Save rollout videos.")
    parser.add_argument("--video_dir", type=str, default=None, help="Directory to save videos.")
    parser.add_argument("--video_fps", type=int, default=None, help="Video fps; default uses control_freq.")
    parser.add_argument("--video_prefix", type=str, default="rollout", help="Saved video filename prefix.")
    parser.add_argument(
        "--reset_to_demo_start",
        action="store_true",
        help="Reset env to demo start state (path auto-inferred from env_name).",
    )
    parser.add_argument(
        "--demo_episode_idx",
        type=int,
        default=0,
        help="Demo episode index; set -1 to auto-increment and loop over demo episodes.",
    )
    parser.add_argument(
        "--reset_policy_each_episode",
        dest="reset_policy_each_episode",
        action="store_true",
    )
    parser.add_argument(
        "--no_reset_policy_each_episode",
        dest="reset_policy_each_episode",
        action="store_false",
    )
    parser.set_defaults(reset_policy_each_episode=True)

    parser.add_argument("--control_freq", type=int, default=20)
    parser.add_argument("--camera_names", type=_parse_camera_names, default=("agentview", "sideview"))
    parser.add_argument("--camera_heights", type=int, default=256)
    parser.add_argument("--camera_widths", type=int, default=256)

    parser.add_argument("--use_camera_obs", dest="use_camera_obs", action="store_true")
    parser.add_argument("--no_use_camera_obs", dest="use_camera_obs", action="store_false")
    parser.set_defaults(use_camera_obs=True)

    parser.add_argument("--has_renderer", dest="has_renderer", action="store_true")
    parser.add_argument("--no_has_renderer", dest="has_renderer", action="store_false")
    parser.set_defaults(has_renderer=False)

    parser.add_argument("--has_offscreen_renderer", dest="has_offscreen_renderer", action="store_true")
    parser.add_argument("--no_has_offscreen_renderer", dest="has_offscreen_renderer", action="store_false")
    parser.set_defaults(has_offscreen_renderer=True)

    parser.add_argument("--seed", type=int, default=42, help="Random seed for numpy/random/torch.")
    return parser


"""
Env in: UnStack/Stack NutDisAssemblyRound/NutAssemblyRound TwoArmPegRemoval/TwoArmPegInHole
Usage:
conda activate trdrl
cd ~/code/trdrl25nips/robosuite/
export PYTHONPATH=~/code/trdrl25nips/robosuite/
CUDA_VISIBLE_DEVICES=6 python eval_socket.py -o output/socket_eval -e TwoArmPegInHole \
    -p 7286 --test_cnt 20 \
    --test_max_steps 300 --save_video --video_dir output/socket_eval
"""
def main():
    parser = build_parser()
    args = parser.parse_args()

    pathlib.Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    policy_url = _build_policy_url(args.policy_url, args.port)

    evaluator = RobosuiteSocketEvaluator(
        env_name=args.env_name,
        use_camera_obs=args.use_camera_obs,
        has_renderer=args.has_renderer,
        has_offscreen_renderer=args.has_offscreen_renderer,
        control_freq=args.control_freq,
        camera_names=args.camera_names,
        camera_heights=args.camera_heights,
        camera_widths=args.camera_widths,
        test_cnt=args.test_cnt,
        test_start_seed=args.test_start_seed,
        test_max_steps=args.test_max_steps,
        policy_url=policy_url,
        send_per_frames=args.send_per_frames,
        num_obs_steps=args.num_obs_steps,
        request_timeout=args.request_timeout,
    )
    demo_transitions = None
    if args.reset_to_demo_start:
        demo_path = _get_demo_transition_path(args.env_name)
        if not os.path.exists(demo_path):
            raise FileNotFoundError(f"Demo transitions not found: {demo_path}")
        demo_transitions = _load_demo_transitions(demo_path)
        print(f"[eval_socket] Loaded demo transitions: {demo_path}")

    try:
        max_cache_action = evaluator.init_socket()
        print(f"[eval_socket] Connected: policy_url={policy_url}, max_cache_action={max_cache_action}")
        evaluator.send_reset()

        eval_log = evaluator.evaluate(
            test_cnt=args.test_cnt,
            test_start_seed=args.test_start_seed,
            test_max_steps=args.test_max_steps,
            reset_policy_each_episode=args.reset_policy_each_episode,
            save_video=args.save_video,
            video_dir=args.video_dir,
            video_fps=args.video_fps,
            video_prefix=args.video_prefix,
            demo_transitions=demo_transitions,
            demo_episode_idx=args.demo_episode_idx,
        )

        out_path = os.path.join(args.output_dir, f"eval_log_{args.env_name}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(eval_log, f, indent=2, sort_keys=True)

        print("[eval_socket] Evaluation done.")
        print(
            "[eval_socket] "
            f"num_episodes={eval_log.get('num_episodes', 0)}, "
            f"mean_success={eval_log.get('mean_success', 0.0):.4f}, "
            f"mean_score={eval_log.get('mean_score', 0.0):.4f}, "
            f"mean_steps={eval_log.get('mean_steps', 0.0):.2f}"
        )
        print(f"[eval_socket] Saved log to: {out_path}")
    finally:
        evaluator.close()


if __name__ == "__main__":
    main()
