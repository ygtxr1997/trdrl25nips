import os
from typing import Dict, List, Optional, Tuple

import numpy as np

from robokit.debug_utils.images import save_frames_as_video


class TransitionDemo3DVisualizer:
    """
    用 3D 轨迹可视化 transition demo 中关键点位，并导出视频。

    - from wrapper import TransitionDemo3DVisualizer
    - vis = TransitionDemo3DVisualizer(env_name="Stack", transition_path="generate/
    Stack_transitions_10trajectory_sparse.npy", output_dir="debug/videos/stack_3d",
    horizon=500, fps=20)
    - vis.render_all_episodes() 或 vis.render_episode_video(episode_idx=0)

    transition 文件默认格式:
        transitions[i] = [state, action, reward, next_state, done]
    """

    KEY_POINT_IDXS: Dict[str, Dict[str, Tuple[int, int, int]]] = {
        "Door": {
            "eef": (0, 1, 2),
            "door": (9, 10, 11),
            "handle": (12, 13, 14),
        },
        "Door_Close": {
            "eef": (0, 1, 2),
            "door": (9, 10, 11),
            "handle": (12, 13, 14),
        },
        "Old_Door": {
            "eef": (0, 1, 2),
            "door": (9, 10, 11),
            "handle": (12, 13, 14),
        },
        "Old_Door_Close": {
            "eef": (0, 1, 2),
            "door": (9, 10, 11),
            "handle": (12, 13, 14),
        },
        "NutAssemblyRound": {
            "eef": (0, 1, 2),
            "nut": (3, 4, 5),
            "peg": (6, 7, 8),
            "goal": (9, 10, 11),
        },
        "NutDisAssemblyRound": {
            "eef": (0, 1, 2),
            "nut": (3, 4, 5),
            "peg": (6, 7, 8),
            "goal": (9, 10, 11),
        },
        "TwoArmPegInHole": {
            "eef": (0, 1, 2),
            "peg": (3, 4, 5),
            "hole": (10, 11, 12),
            "goal": (13, 14, 15),
        },
        "TwoArmPegRemoval": {
            "eef": (0, 1, 2),
            "peg": (3, 4, 5),
            "hole": (10, 11, 12),
            "goal": (13, 14, 15),
        },
        "Stack": {
            "eef": (0, 1, 2),
            "cubeA": (3, 4, 5),
            "cubeB": (6, 7, 8),
            "goal": (11, 12, 13),
        },
        "UnStack": {
            "eef": (0, 1, 2),
            "cubeA": (3, 4, 5),
            "cubeB": (6, 7, 8),
            "goal": (11, 12, 13),
        },
    }

    def __init__(
        self,
        env_name: str,
        transition_path: str,
        output_dir: str = "debug/videos/demo_state_3d",
        horizon: int = 500,
        fps: int = 20,
    ):
        self.env_name = env_name
        self.transition_path = transition_path
        self.output_dir = output_dir
        self.horizon = int(horizon)
        self.fps = int(fps)

        os.makedirs(self.output_dir, exist_ok=True)
        self.transitions = np.load(self.transition_path, allow_pickle=True)
        if self.transitions.ndim != 2 or self.transitions.shape[1] < 5:
            raise ValueError(
                f"Unexpected transitions shape: {self.transitions.shape}, expected (N, >=5)."
            )

        self.states = np.stack(self.transitions[:, 0], axis=0).astype(np.float32)
        self.rewards = np.array(self.transitions[:, 2], dtype=np.float32)
        self.dones = np.array(self.transitions[:, 4], dtype=np.float32)
        self.episodes = self._split_episodes(self.dones, self.horizon, len(self.states))
        self.point_indices = self._infer_point_indices()

    @staticmethod
    def _split_episodes(
        dones: np.ndarray,
        horizon: int,
        num_steps: int,
    ) -> List[Tuple[int, int]]:
        done_indices = np.where(dones > 0.5)[0]
        if len(done_indices) > 0:
            episodes = []
            start = 0
            for end_idx in done_indices:
                episodes.append((start, int(end_idx) + 1))
                start = int(end_idx) + 1
            if start < num_steps:
                episodes.append((start, num_steps))
            return episodes

        return [
            (start, min(start + horizon, num_steps))
            for start in range(0, num_steps, horizon)
        ]

    def _infer_point_indices(self) -> Dict[str, Tuple[int, int, int]]:
        if self.env_name in self.KEY_POINT_IDXS:
            selected = self.KEY_POINT_IDXS[self.env_name]
            valid_selected = {}
            for name, idx_triplet in selected.items():
                if max(idx_triplet) < self.states.shape[1]:
                    valid_selected[name] = idx_triplet
            if len(valid_selected) > 0:
                return valid_selected

        fallback = {}
        max_triplets = min(self.states.shape[1] // 3, 4)
        for i in range(max_triplets):
            fallback[f"p{i}"] = (3 * i, 3 * i + 1, 3 * i + 2)
        return fallback

    def _extract_episode_points(self, episode_idx: int) -> Dict[str, np.ndarray]:
        if episode_idx < 0 or episode_idx >= len(self.episodes):
            raise IndexError(f"episode_idx {episode_idx} out of range [0, {len(self.episodes)-1}]")
        start, end = self.episodes[episode_idx]
        ep_states = self.states[start:end]
        return {
            name: ep_states[:, [idxs[0], idxs[1], idxs[2]]]
            for name, idxs in self.point_indices.items()
        }

    def _extract_episode_reward_done(self, episode_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        if episode_idx < 0 or episode_idx >= len(self.episodes):
            raise IndexError(f"episode_idx {episode_idx} out of range [0, {len(self.episodes)-1}]")
        start, end = self.episodes[episode_idx]
        return self.rewards[start:end], self.dones[start:end]

    def _extract_all_episode_data(
        self,
        max_episodes: Optional[int] = None,
    ) -> Tuple[List[Dict[str, np.ndarray]], List[np.ndarray], List[np.ndarray]]:
        n = len(self.episodes) if max_episodes is None else min(len(self.episodes), int(max_episodes))
        all_points: List[Dict[str, np.ndarray]] = []
        all_rewards: List[np.ndarray] = []
        all_dones: List[np.ndarray] = []
        for ep_idx in range(n):
            all_points.append(self._extract_episode_points(ep_idx))
            ep_rewards, ep_dones = self._extract_episode_reward_done(ep_idx)
            all_rewards.append(ep_rewards)
            all_dones.append(ep_dones)
        return all_points, all_rewards, all_dones

    @staticmethod
    def _axis_range(points: Dict[str, np.ndarray]) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
        cat = np.concatenate(list(points.values()), axis=0)
        x_min, y_min, z_min = np.min(cat, axis=0).tolist()
        x_max, y_max, z_max = np.max(cat, axis=0).tolist()
        pad = 0.03
        return (
            (x_min - pad, x_max + pad),
            (y_min - pad, y_max + pad),
            (z_min - pad, z_max + pad),
        )

    @staticmethod
    def _axis_range_from_multi_episode_points(
        multi_episode_points: List[Dict[str, np.ndarray]]
    ) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
        cat = np.concatenate(
            [arr for one_episode in multi_episode_points for arr in one_episode.values()],
            axis=0,
        )
        x_min, y_min, z_min = np.min(cat, axis=0).tolist()
        x_max, y_max, z_max = np.max(cat, axis=0).tolist()
        pad = 0.03
        return (
            (x_min - pad, x_max + pad),
            (y_min - pad, y_max + pad),
            (z_min - pad, z_max + pad),
        )

    def render_episode_video(
        self,
        episode_idx: int = 0,
        save_path: Optional[str] = None,
        elev: float = 30.0,
        azim: float = 45.0,
    ) -> str:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        points = self._extract_episode_points(episode_idx)
        ep_rewards, ep_dones = self._extract_episode_reward_done(episode_idx)
        x_lim, y_lim, z_lim = self._axis_range(points)
        point_names = list(points.keys())
        colors = ["tab:red", "tab:blue", "tab:green", "tab:orange", "tab:purple", "tab:brown"]

        step_count = next(iter(points.values())).shape[0]
        frames: List[np.ndarray] = []

        for t in range(step_count):
            fig = plt.figure(figsize=(11, 6), dpi=120)
            ax = fig.add_subplot(1, 2, 1, projection="3d")
            ax.set_xlim(*x_lim)
            ax.set_ylim(*y_lim)
            ax.set_zlim(*z_lim)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")
            ax.view_init(elev=elev, azim=azim)
            ax.set_title(f"{self.env_name} | ep={episode_idx} | step={t}/{step_count-1}")

            for i, name in enumerate(point_names):
                series = points[name][: t + 1]
                c = colors[i % len(colors)]
                if series.shape[0] > 1:
                    ax.plot(series[:, 0], series[:, 1], series[:, 2], color=c, linewidth=1.8, alpha=0.85)
                ax.scatter(series[-1, 0], series[-1, 1], series[-1, 2], color=c, s=40, label=name)

            ax.legend(loc="upper left", fontsize=8)

            ax2 = fig.add_subplot(1, 2, 2)
            time_axis = np.arange(step_count, dtype=np.int32)
            reward_hist = ep_rewards[: t + 1]
            done_hist = ep_dones[: t + 1]

            reward_line, = ax2.plot(
                time_axis[: t + 1], reward_hist, color="tab:blue", linewidth=1.8, label="reward"
            )
            ax2.scatter([t], [reward_hist[-1]], color="tab:blue", s=28)
            ax2.set_xlabel("step")
            ax2.set_ylabel("reward", color="tab:blue")
            ax2.tick_params(axis="y", labelcolor="tab:blue")
            ax2.set_xlim(0, max(step_count - 1, 1))
            reward_min = float(np.min(ep_rewards))
            reward_max = float(np.max(ep_rewards))
            if reward_max - reward_min < 1e-6:
                reward_min -= 0.1
                reward_max += 0.1
            pad = 0.1 * (reward_max - reward_min)
            ax2.set_ylim(reward_min - pad, reward_max + pad)
            ax2.grid(alpha=0.3)

            ax2_done = ax2.twinx()
            done_line, = ax2_done.step(
                time_axis[: t + 1], done_hist, where="post", color="tab:red", linewidth=1.6, label="done"
            )
            ax2_done.scatter([t], [done_hist[-1]], color="tab:red", s=24)
            ax2_done.set_ylabel("done", color="tab:red")
            ax2_done.tick_params(axis="y", labelcolor="tab:red")
            ax2_done.set_ylim(-0.05, 1.05)
            ax2.axvline(t, color="gray", linestyle="--", linewidth=1.0, alpha=0.8)
            ax2.set_title("Reward / Done over time")
            ax2.legend([reward_line, done_line], ["reward", "done"], loc="upper left", fontsize=9)

            fig.tight_layout()
            fig.canvas.draw()
            frame = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)[..., :3].copy()
            frames.append(frame)
            plt.close(fig)

        if save_path is None:
            save_path = os.path.join(
                self.output_dir,
                f"{self.env_name}_ep{episode_idx:03d}_3d_points.mp4",
            )
        save_frames_as_video(frames, save_path=save_path, fps=self.fps)
        return save_path

    def render_all_episodes_in_one_video(
        self,
        max_episodes: Optional[int] = None,
        save_path: Optional[str] = None,
        elev: float = 30.0,
        azim: float = 45.0,
    ) -> str:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D

        all_points, all_rewards, all_dones = self._extract_all_episode_data(max_episodes=max_episodes)
        if len(all_points) == 0:
            raise ValueError("No episode data found for rendering.")

        x_lim, y_lim, z_lim = self._axis_range_from_multi_episode_points(all_points)
        episode_count = len(all_points)
        point_names = list(self.point_indices.keys())
        step_count = max(next(iter(points.values())).shape[0] for points in all_points)
        cmap = plt.cm.get_cmap("tab20", max(episode_count, 1))
        episode_colors = [cmap(i) for i in range(episode_count)]

        reward_min = float(min(np.min(r) for r in all_rewards))
        reward_max = float(max(np.max(r) for r in all_rewards))
        if reward_max - reward_min < 1e-6:
            reward_min -= 0.1
            reward_max += 0.1
        reward_pad = 0.1 * (reward_max - reward_min)

        frames: List[np.ndarray] = []

        for t in range(step_count):
            fig = plt.figure(figsize=(12, 6), dpi=120)
            ax = fig.add_subplot(1, 2, 1, projection="3d")
            ax.set_xlim(*x_lim)
            ax.set_ylim(*y_lim)
            ax.set_zlim(*z_lim)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")
            ax.view_init(elev=elev, azim=azim)
            ax.set_title(
                f"{self.env_name} | all episodes={episode_count} | step={t}/{step_count-1}"
            )

            for ep_idx, points in enumerate(all_points):
                ep_color = episode_colors[ep_idx]
                for name in point_names:
                    if name not in points:
                        continue
                    series = points[name]
                    valid_len = min(t + 1, series.shape[0])
                    if valid_len <= 0:
                        continue
                    cur = series[:valid_len]
                    if cur.shape[0] > 1:
                        ax.plot(
                            cur[:, 0], cur[:, 1], cur[:, 2],
                            color=ep_color, linewidth=1.0, alpha=0.35
                        )
                    ax.scatter(
                        cur[-1, 0], cur[-1, 1], cur[-1, 2],
                        color=ep_color, s=10, alpha=0.9
                    )

            if episode_count <= 8:
                legend_handles = [
                    Line2D([0], [0], marker="o", color=episode_colors[i], lw=1.5, label=f"ep{i}")
                    for i in range(episode_count)
                ]
                ax.legend(handles=legend_handles, loc="upper left", fontsize=8)

            ax2 = fig.add_subplot(1, 2, 2)
            ax2_done = ax2.twinx()
            for ep_idx in range(episode_count):
                ep_color = episode_colors[ep_idx]
                rewards = all_rewards[ep_idx]
                dones = all_dones[ep_idx]
                valid_len = min(t + 1, rewards.shape[0])
                if valid_len <= 0:
                    continue
                x = np.arange(valid_len, dtype=np.int32)
                reward_hist = rewards[:valid_len]
                done_hist = dones[:valid_len]

                ax2.plot(x, reward_hist, color=ep_color, linewidth=1.4, alpha=0.75)
                ax2.scatter([valid_len - 1], [reward_hist[-1]], color=ep_color, s=10, alpha=0.75)
                ax2_done.step(
                    x, done_hist, where="post", color=ep_color,
                    linewidth=1.2, alpha=0.45, linestyle="--"
                )
                ax2_done.scatter([valid_len - 1], [done_hist[-1]], color=ep_color, s=8, alpha=0.45)

            ax2.set_xlabel("step")
            ax2.set_ylabel("reward")
            ax2.set_xlim(0, max(step_count - 1, 1))
            ax2.set_ylim(reward_min - reward_pad, reward_max + reward_pad)
            ax2.grid(alpha=0.3)
            ax2_done.set_ylabel("done")
            ax2_done.set_ylim(-0.05, 1.05)
            ax2.axvline(t, color="gray", linestyle="--", linewidth=1.0, alpha=0.8)
            ax2.set_title("Reward / Done over time (all episodes)")

            metric_legend = [
                Line2D([0], [0], color="gray", linewidth=1.8, linestyle="-", label="reward curve"),
                Line2D([0], [0], color="gray", linewidth=1.8, linestyle="--", label="done curve"),
            ]
            ax2.legend(handles=metric_legend, loc="upper left", fontsize=9)

            fig.tight_layout()
            fig.canvas.draw()
            frame = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)[..., :3].copy()
            frames.append(frame)
            plt.close(fig)

        if save_path is None:
            save_path = os.path.join(
                self.output_dir,
                f"{self.env_name}_all_episodes_3d_points.mp4",
            )
        save_frames_as_video(frames, save_path=save_path, fps=self.fps)
        return save_path

    def render_all_episodes(
        self,
        max_episodes: Optional[int] = None,
        elev: float = 30.0,
        azim: float = 45.0,
        merge_into_one_video: bool = False,
    ) -> List[str]:
        if merge_into_one_video:
            one_video_path = self.render_all_episodes_in_one_video(
                max_episodes=max_episodes,
                elev=elev,
                azim=azim,
            )
            print(f"[3D-Vis] merged episodes saved={one_video_path}")
            return [one_video_path]

        n = len(self.episodes) if max_episodes is None else min(len(self.episodes), int(max_episodes))
        video_paths = []
        for ep_idx in range(n):
            save_path = self.render_episode_video(
                episode_idx=ep_idx,
                elev=elev,
                azim=azim,
            )
            print(f"[3D-Vis] episode={ep_idx:03d}/{n-1:03d}, saved={save_path}")
            video_paths.append(save_path)
        return video_paths
