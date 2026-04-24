import os
import numpy as np
import robosuite as suite
from mujoco_py.builder import MujocoException
from robosuite.controllers import load_controller_config
from typing import Any, Dict, List, Optional
from tqdm import tqdm

from robokit.connects.protocols import StepRequestFromPolicy, StepRequestFromEvaluator
from robokit.debug_utils.images import save_frames_as_video

from utils import obs2state, state2agentenv


class RobosuiteEvaluator:
    """
    Standalone evaluator wrapper for robosuite tasks used by `train_sac_agentenv_2agents_state.py`.

    State / action conventions:
    - `state`: `np.ndarray`, shape `(D_state,)`, built by `utils.obs2state(...)`.
    - `agent_state`: `np.ndarray`, shape `(D_agent_state,)`, from `utils.state2agentenv(...)`.
    - `env_state`: `np.ndarray`, shape `(D_env_state,)`, from `utils.state2agentenv(...)`.
    - `policy_action`: `np.ndarray` (or tensor-like), shape `(A_policy,)`.
      For `TwoArmPegInHole` / `TwoArmPegRemoval`, this is half of env action and expected in `[-1, 1]`.
    - `step_action`: `np.ndarray`, shape `(A_env,)`, action actually sent to env.
    """
    TWO_ARM_ENVS = {"TwoArmPegInHole", "TwoArmPegRemoval"}
    STATE_LABELS = {
        "Door": [
            "robot0_eef_pos_x", "robot0_eef_pos_y", "robot0_eef_pos_z",
            "robot0_gripper_qpos_0", "robot0_gripper_qpos_1", "robot0_gripper_qpos_2",
            "robot0_gripper_qpos_3", "robot0_gripper_qpos_4", "robot0_gripper_qpos_5",
            "door_pos_x", "door_pos_y", "door_pos_z",
            "handle_pos_x", "handle_pos_y", "handle_pos_z",
            "door_to_eef_x", "door_to_eef_y", "door_to_eef_z",
            "handle_to_eef_x", "handle_to_eef_y", "handle_to_eef_z",
            "hinge_qpos", "handle_qpos", "check_grasp",
        ],
        "Door_Close": [
            "robot0_eef_pos_x", "robot0_eef_pos_y", "robot0_eef_pos_z",
            "robot0_gripper_qpos_0", "robot0_gripper_qpos_1", "robot0_gripper_qpos_2",
            "robot0_gripper_qpos_3", "robot0_gripper_qpos_4", "robot0_gripper_qpos_5",
            "door_pos_x", "door_pos_y", "door_pos_z",
            "handle_pos_x", "handle_pos_y", "handle_pos_z",
            "door_to_eef_x", "door_to_eef_y", "door_to_eef_z",
            "handle_to_eef_x", "handle_to_eef_y", "handle_to_eef_z",
            "hinge_qpos", "handle_qpos", "check_grasp",
        ],
        "Old_Door": [
            "robot0_eef_pos_x", "robot0_eef_pos_y", "robot0_eef_pos_z",
            "robot0_gripper_qpos_0", "robot0_gripper_qpos_1", "robot0_gripper_qpos_2",
            "robot0_gripper_qpos_3", "robot0_gripper_qpos_4", "robot0_gripper_qpos_5",
            "door_pos_x", "door_pos_y", "door_pos_z",
            "handle_pos_x", "handle_pos_y", "handle_pos_z",
            "door_to_eef_x", "door_to_eef_y", "door_to_eef_z",
            "handle_to_eef_x", "handle_to_eef_y", "handle_to_eef_z",
            "hinge_qpos", "handle_qpos", "check_grasp",
        ],
        "Old_Door_Close": [
            "robot0_eef_pos_x", "robot0_eef_pos_y", "robot0_eef_pos_z",
            "robot0_gripper_qpos_0", "robot0_gripper_qpos_1", "robot0_gripper_qpos_2",
            "robot0_gripper_qpos_3", "robot0_gripper_qpos_4", "robot0_gripper_qpos_5",
            "door_pos_x", "door_pos_y", "door_pos_z",
            "handle_pos_x", "handle_pos_y", "handle_pos_z",
            "door_to_eef_x", "door_to_eef_y", "door_to_eef_z",
            "handle_to_eef_x", "handle_to_eef_y", "handle_to_eef_z",
            "hinge_qpos", "handle_qpos", "check_grasp",
        ],
        "NutAssemblyRound": [
            "eef_pos_x", "eef_pos_y", "eef_pos_z",
            "nut_pos_x", "nut_pos_y", "nut_pos_z",
            "peg_pos_x", "peg_pos_y", "peg_pos_z",
            "goal_pos_x", "goal_pos_y", "goal_pos_z",
            "eef_to_nut_x", "eef_to_nut_y", "eef_to_nut_z",
            "nut_to_peg_x", "nut_to_peg_y", "nut_to_peg_z",
            "nut_to_goal_x", "nut_to_goal_y", "nut_to_goal_z",
            "check_grasp",
        ],
        "NutDisAssemblyRound": [
            "eef_pos_x", "eef_pos_y", "eef_pos_z",
            "nut_pos_x", "nut_pos_y", "nut_pos_z",
            "peg_pos_x", "peg_pos_y", "peg_pos_z",
            "goal_pos_x", "goal_pos_y", "goal_pos_z",
            "eef_to_nut_x", "eef_to_nut_y", "eef_to_nut_z",
            "nut_to_peg_x", "nut_to_peg_y", "nut_to_peg_z",
            "nut_to_goal_x", "nut_to_goal_y", "nut_to_goal_z",
            "check_grasp",
        ],
        "TwoArmPegInHole": [
            "robot0_eef_pos_x", "robot0_eef_pos_y", "robot0_eef_pos_z",
            "peg_pos_x", "peg_pos_y", "peg_pos_z",
            "peg_quat_w", "peg_quat_x", "peg_quat_y", "peg_quat_z",
            "hole_pos_x", "hole_pos_y", "hole_pos_z",
            "goal_pos_x", "goal_pos_y", "goal_pos_z",
            "peg_to_goal_x", "peg_to_goal_y", "peg_to_goal_z",
            "peg_to_hole_x", "peg_to_hole_y", "peg_to_hole_z",
            "t", "d", "cos", "check_contact_peg_hole",
        ],
        "TwoArmPegRemoval": [
            "robot0_eef_pos_x", "robot0_eef_pos_y", "robot0_eef_pos_z",
            "peg_pos_x", "peg_pos_y", "peg_pos_z",
            "peg_quat_w", "peg_quat_x", "peg_quat_y", "peg_quat_z",
            "hole_pos_x", "hole_pos_y", "hole_pos_z",
            "goal_pos_x", "goal_pos_y", "goal_pos_z",
            "peg_to_goal_x", "peg_to_goal_y", "peg_to_goal_z",
            "peg_to_hole_x", "peg_to_hole_y", "peg_to_hole_z",
            "t", "d", "cos", "check_contact_peg_hole",
        ],
        "Stack": [
            "robot0_eef_pos_x", "robot0_eef_pos_y", "robot0_eef_pos_z",
            "cubeA_pos_x", "cubeA_pos_y", "cubeA_pos_z",
            "cubeB_pos_x", "cubeB_pos_y", "cubeB_pos_z",
            "initial_cubeB_x", "initial_cubeB_y",
            "goal_pos_x", "goal_pos_y", "goal_pos_z",
            "eef_to_cubeA_x", "eef_to_cubeA_y", "eef_to_cubeA_z",
            "cubeA_to_goal_x", "cubeA_to_goal_y", "cubeA_to_goal_z",
            "cubeB_to_init_x", "cubeB_to_init_y",
            "check_grasp_cubeA", "cubeA_touching_cubeB",
        ],
        "UnStack": [
            "robot0_eef_pos_x", "robot0_eef_pos_y", "robot0_eef_pos_z",
            "cubeA_pos_x", "cubeA_pos_y", "cubeA_pos_z",
            "cubeB_pos_x", "cubeB_pos_y", "cubeB_pos_z",
            "initial_cubeB_x", "initial_cubeB_y",
            "goal_pos_x", "goal_pos_y", "goal_pos_z",
            "eef_to_cubeA_x", "eef_to_cubeA_y", "eef_to_cubeA_z",
            "cubeA_to_goal_x", "cubeA_to_goal_y", "cubeA_to_goal_z",
            "cubeB_to_init_x", "cubeB_to_init_y",
            "check_grasp_cubeA", "cubeA_touching_cubeB",
        ],
    }

    def __init__(
        self,
        env_name,
        reward_shaping=False,
        horizon=500,
        use_camera_obs=False,
        has_renderer=False,
        has_offscreen_renderer=True,
        control_freq=20,
        camera_names=("agentview", "sideview"),
        camera_heights=256,
        camera_widths=256,
        robots=None,
        controller_name=None,
        catch_mujoco_exception=True,
    ):
        """
        Args:
            env_name: `str`, robosuite env name, e.g. `Old_Door`, `TwoArmPegInHole`.
            reward_shaping: `bool`, environment reward mode.
            horizon: `int`, max episode length `T`.
            use_camera_obs: `bool`, whether obs contains image keys for video rendering.
            has_renderer: `bool`, on-screen renderer flag.
            has_offscreen_renderer: `bool`, off-screen renderer flag.
            control_freq: `int`, control frequency used by robosuite.
            camera_names: `tuple[str] | list[str]`, camera keys to request from env.
            camera_heights: `int`, image height in pixels.
            camera_widths: `int`, image width in pixels.
            robots: `str | list[str] | None`, robot config passed to `suite.make`.
            controller_name: `str | None`, controller name; defaults follow training script.
            catch_mujoco_exception: `bool`, if True converts MujocoException into terminal step.
        """
        self.env_name = env_name
        self.reward_shaping = reward_shaping
        self.horizon = horizon
        self.use_camera_obs = use_camera_obs
        self.has_renderer = has_renderer
        self.has_offscreen_renderer = has_offscreen_renderer
        self.control_freq = control_freq
        self.camera_names = list(camera_names)
        self.camera_heights = camera_heights
        self.camera_widths = camera_widths
        self.catch_mujoco_exception = catch_mujoco_exception

        if robots is None:
            robots = ["Kinova3", "Kinova3"] if env_name in self.TWO_ARM_ENVS else "Kinova3"
        if controller_name is None:
            controller_name = "OSC_POSE" if env_name in self.TWO_ARM_ENVS else "OSC_POSITION"

        self.robots = robots
        self.controller_name = controller_name
        self.controller_configs = load_controller_config(default_controller=self.controller_name)

        self.env = suite.make(
            env_name=self.env_name,
            reward_shaping=self.reward_shaping,
            robots=self.robots,
            controller_configs=self.controller_configs,
            has_renderer=self.has_renderer,
            has_offscreen_renderer=self.has_offscreen_renderer,
            use_camera_obs=self.use_camera_obs,
            control_freq=self.control_freq,
            horizon=self.horizon,
            camera_names=self.camera_names,
            camera_heights=self.camera_heights,
            camera_widths=self.camera_widths,
        )

        self.env_action_dim = self.env.action_dim
        env_action_low, env_action_high = self.env.action_spec

        if self.env_name in self.TWO_ARM_ENVS:
            self.policy_action_dim = self.env_action_dim // 2
            self.policy_action_low = -np.ones(self.policy_action_dim, dtype=np.float32)
            self.policy_action_high = np.ones(self.policy_action_dim, dtype=np.float32)
        else:
            self.policy_action_dim = self.env_action_dim
            self.policy_action_low = np.asarray(env_action_low, dtype=np.float32)
            self.policy_action_high = np.asarray(env_action_high, dtype=np.float32)

        self._obs = None
        self._state = None
        self._agent_state = None
        self._env_state = None
        self._step_count = 0

        state, info = self.reset()
        self.state_dim = int(state.shape[0])
        self.agent_state_dim = int(info["agent_state"].shape[0])
        self.env_state_dim = int(info["env_state"].shape[0])
        self.sim_state_dim = int(self.env.sim.get_state().flatten().shape[0])
        labels = self.STATE_LABELS.get(self.env_name, [])
        self.state_label_to_idx = (
            {label: i for i, label in enumerate(labels)}
            if len(labels) == self.state_dim
            else {}
        )

    @staticmethod
    def _to_numpy_1d(x):
        """
        Normalize action-like input to numpy 1D array.

        Args:
            x: `np.ndarray | torch.Tensor | list | tuple`, expected action shape `(A,)` or `(1, A)`.
        Returns:
            `np.ndarray`, shape `(A,)`, dtype `float32`.
        """
        if hasattr(x, "detach") and hasattr(x, "cpu"):
            x = x.detach().cpu().numpy()
        return np.asarray(x, dtype=np.float32).reshape(-1)

    def _policy_action_to_env_action(self, policy_action):
        """
        Convert policy action space to env action space.

        Args:
            policy_action: `np.ndarray | torch.Tensor`, shape `(A_policy,)`, typically in action bounds.
                - Non-two-arm tasks: clipped to env `action_spec`.
                - Two-arm tasks: clipped to `[-1, 1]` and then padded with trailing zeros.
        Returns:
            `np.ndarray`, shape `(A_env,)`, action sent to `env.step`.
        """
        policy_action = self._to_numpy_1d(policy_action)
        if policy_action.shape[0] != self.policy_action_dim:
            raise ValueError(
                f"Expected policy action dim {self.policy_action_dim}, got {policy_action.shape[0]}"
            )

        policy_action = np.clip(policy_action, self.policy_action_low, self.policy_action_high)

        if self.env_name in self.TWO_ARM_ENVS:
            pad_dim = self.env_action_dim - self.policy_action_dim
            return np.concatenate([policy_action, np.zeros(pad_dim, dtype=np.float32)], axis=0)
        return policy_action

    def reset(self, seed=None):
        """
        Reset environment and internal cached states.

        Args:
            seed: `int | None`, numpy RNG seed only (used by this wrapper).
        Returns:
            state: `np.ndarray`, shape `(D_state,)`.
            info: `dict`, includes:
                - `obs`: robosuite raw observation dict.
                - `agent_state`: `np.ndarray`, shape `(D_agent_state,)`.
                - `env_state`: `np.ndarray`, shape `(D_env_state,)`.
                - `success`: `int` in `{0,1}`.
                - `step_count`: `int`, starts at `0`.
        """
        if seed is not None:
            np.random.seed(seed)

        self._obs = self.env.reset()
        self._state = obs2state(self._obs, self.env, self.env_name)
        self._agent_state, self._env_state = state2agentenv(self._state, self.env_name)
        self._step_count = 0

        info = {
            "obs": self._obs,
            "agent_state": self._agent_state.copy(),
            "env_state": self._env_state.copy(),
            "success": int(self.env._check_success()),
            "step_count": self._step_count,
        }
        return self._state.copy(), info

    def _try_set_free_joint_pose(self, joint_name, pos_xyz=None, quat_wxyz=None):
        """
        Best-effort helper to set a free-joint pose.

        Args:
            joint_name: `str`, MuJoCo joint name.
            pos_xyz: `np.ndarray | None`, shape `(3,)`, xyz position in world frame.
            quat_wxyz: `np.ndarray | None`, shape `(4,)`, quaternion in wxyz order.
        Returns:
            `bool`, True if the joint exists and qpos was updated.
        """
        try:
            qpos = np.asarray(self.env.sim.data.get_joint_qpos(joint_name), dtype=np.float64).copy()
        except Exception:
            return False

        if qpos.ndim == 0:
            return False
        changed = False
        if pos_xyz is not None and qpos.shape[0] >= 3:
            qpos[:3] = np.asarray(pos_xyz, dtype=np.float64).reshape(3)
            changed = True
        if quat_wxyz is not None and qpos.shape[0] >= 7:
            qpos[3:7] = np.asarray(quat_wxyz, dtype=np.float64).reshape(4)
            changed = True
        if not changed:
            return False
        try:
            self.env.sim.data.set_joint_qpos(joint_name, qpos)
            return True
        except Exception:
            return False

    @staticmethod
    def _quat_mul_wxyz(q1, q2):
        """
        Hamilton product for quaternions in wxyz convention.
        """
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array(
            [
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            ],
            dtype=np.float64,
        )

    @staticmethod
    def _quat_conj_wxyz(q):
        """
        Quaternion conjugate in wxyz convention.
        """
        return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)

    @classmethod
    def _quat_error_rotvec_wxyz(cls, target_quat_wxyz, current_quat_wxyz):
        """
        Convert quaternion error (target * conj(current)) to rotation-vector.
        """
        q_t = np.asarray(target_quat_wxyz, dtype=np.float64).reshape(4)
        q_c = np.asarray(current_quat_wxyz, dtype=np.float64).reshape(4)
        q_t = q_t / (np.linalg.norm(q_t) + 1e-12)
        q_c = q_c / (np.linalg.norm(q_c) + 1e-12)
        q_e = cls._quat_mul_wxyz(q_t, cls._quat_conj_wxyz(q_c))
        if q_e[0] < 0:
            q_e = -q_e
        q_e = q_e / (np.linalg.norm(q_e) + 1e-12)
        w = np.clip(q_e[0], -1.0, 1.0)
        v = q_e[1:]
        v_norm = np.linalg.norm(v)
        if v_norm < 1e-10:
            return np.zeros(3, dtype=np.float64)
        angle = 2.0 * np.arctan2(v_norm, w)
        axis = v / v_norm
        return axis * angle

    def _solve_body_pose_with_robot_joints(
        self,
        body_name,
        robot,
        target_pos=None,
        target_quat_wxyz=None,
        max_iters=120,
        pos_tol=1e-4,
        rot_tol=2e-3,
        damping=1e-4,
        step_clip=0.05,
        pos_weight=1.0,
        rot_weight=0.5,
    ):
        """
        Damped least-squares IK on robot joints to match body pose.

        Args:
            body_name: `str`, MuJoCo body name.
            robot: robosuite robot object with `_ref_joint_pos_indexes/_ref_joint_vel_indexes`.
            target_pos: `np.ndarray | None`, shape `(3,)`.
            target_quat_wxyz: `np.ndarray | None`, shape `(4,)`.
        Returns:
            `dict` with keys:
                - `ok`: `bool`
                - `iters`: `int`
                - `pos_err_norm`: `float | None`
                - `rot_err_norm`: `float | None`
        """
        if robot is None:
            return {"ok": False, "iters": 0, "pos_err_norm": None, "rot_err_norm": None}
        if target_pos is None and target_quat_wxyz is None:
            return {"ok": True, "iters": 0, "pos_err_norm": 0.0, "rot_err_norm": 0.0}

        pos_idx = np.asarray(robot._ref_joint_pos_indexes, dtype=np.int64)
        vel_idx = np.asarray(robot._ref_joint_vel_indexes, dtype=np.int64)
        joint_ids = np.asarray(robot._ref_joint_indexes, dtype=np.int64)
        joint_ranges = self.env.sim.model.jnt_range[joint_ids]

        last_pos_err = None
        last_rot_err = None

        for i_iter in range(int(max_iters)):
            cur_pos = np.asarray(self.env.sim.data.get_body_xpos(body_name), dtype=np.float64)
            cur_quat = np.asarray(self.env.sim.data.get_body_xquat(body_name), dtype=np.float64)

            err_list = []
            jac_blocks = []

            if target_pos is not None:
                pos_err = np.asarray(target_pos, dtype=np.float64).reshape(3) - cur_pos
                last_pos_err = float(np.linalg.norm(pos_err))
                err_list.append(pos_weight * pos_err)
                jac_p = self.env.sim.data.get_body_jacp(body_name).reshape(3, -1)[:, vel_idx]
                jac_blocks.append(pos_weight * jac_p)
            else:
                last_pos_err = None

            if target_quat_wxyz is not None:
                rot_err = self._quat_error_rotvec_wxyz(target_quat_wxyz, cur_quat)
                last_rot_err = float(np.linalg.norm(rot_err))
                err_list.append(rot_weight * rot_err)
                jac_r = self.env.sim.data.get_body_jacr(body_name).reshape(3, -1)[:, vel_idx]
                jac_blocks.append(rot_weight * jac_r)
            else:
                last_rot_err = None

            done_pos = True if last_pos_err is None else (last_pos_err < pos_tol)
            done_rot = True if last_rot_err is None else (last_rot_err < rot_tol)
            if done_pos and done_rot:
                return {
                    "ok": True,
                    "iters": i_iter,
                    "pos_err_norm": last_pos_err,
                    "rot_err_norm": last_rot_err,
                }

            err_vec = np.concatenate(err_list, axis=0)
            jac = np.concatenate(jac_blocks, axis=0)
            hessian = jac.T @ jac + damping * np.eye(jac.shape[1], dtype=np.float64)
            grad = jac.T @ err_vec
            try:
                dq = np.linalg.solve(hessian, grad)
            except np.linalg.LinAlgError:
                dq = np.linalg.pinv(jac) @ err_vec

            dq_norm = np.linalg.norm(dq)
            if dq_norm > step_clip:
                dq = dq * (step_clip / (dq_norm + 1e-12))

            qpos = self.env.sim.data.qpos[pos_idx].copy()
            qpos = qpos + dq
            qpos = np.clip(qpos, joint_ranges[:, 0], joint_ranges[:, 1])
            self.env.sim.data.qpos[pos_idx] = qpos
            self.env.sim.data.qvel[vel_idx] = 0.0
            self.env.sim.forward()

        return {
            "ok": False,
            "iters": int(max_iters),
            "pos_err_norm": last_pos_err,
            "rot_err_norm": last_rot_err,
        }

    def _get_primary_robot_eef_body_name(self):
        if not hasattr(self.env, "robots") or len(self.env.robots) == 0:
            return None
        robot = self.env.robots[0]
        if not hasattr(robot, "robot_model"):
            return None
        eef_name = getattr(robot.robot_model, "eef_name", None)
        if isinstance(eef_name, dict):
            if "right" in eef_name:
                return eef_name["right"]
            if len(eef_name) > 0:
                return next(iter(eef_name.values()))
            return None
        return eef_name

    def _align_primary_eef_pos(self, target_pos, max_iters=160):
        if not hasattr(self.env, "robots") or len(self.env.robots) == 0:
            return {"ok": False, "reason": "no_robot"}
        eef_body_name = self._get_primary_robot_eef_body_name()
        if eef_body_name is None:
            return {"ok": False, "reason": "no_eef_body_name"}
        return self._solve_body_pose_with_robot_joints(
            body_name=eef_body_name,
            robot=self.env.robots[0],
            target_pos=np.asarray(target_pos, dtype=np.float64).reshape(3),
            target_quat_wxyz=None,
            max_iters=int(max_iters),
            pos_weight=1.0,
            rot_weight=0.0,
        )

    def _sync_controllers_to_current_state(self):
        if not hasattr(self.env, "robots"):
            return
        for robot in self.env.robots:
            controller = getattr(robot, "controller", None)
            if controller is None:
                continue
            try:
                controller.update(force=True)
                if hasattr(robot, "_ref_joint_pos_indexes"):
                    joint_pos = np.asarray(self.env.sim.data.qpos[robot._ref_joint_pos_indexes], dtype=np.float64).copy()
                    if hasattr(controller, "update_initial_joints"):
                        controller.update_initial_joints(joint_pos)
                    elif hasattr(controller, "reset_goal"):
                        controller.reset_goal()
                elif hasattr(controller, "reset_goal"):
                    controller.reset_goal()
            except Exception:
                pass

    @staticmethod
    def _maybe_numeric_vector(x):
        if x is None:
            return None
        try:
            arr = np.asarray(x)
        except Exception:
            return None
        if arr.ndim != 1:
            return None
        if arr.dtype == object:
            try:
                arr = np.asarray(arr, dtype=np.float64)
            except Exception:
                return None
        if not np.issubdtype(arr.dtype, np.number):
            return None
        return arr.astype(np.float64, copy=False)

    def _state_values_by_labels(self, state_1d: np.ndarray, labels: List[str]):
        if len(self.state_label_to_idx) == 0:
            return None
        idxs = []
        for label in labels:
            if label not in self.state_label_to_idx:
                return None
            idxs.append(self.state_label_to_idx[label])
        return np.asarray(state_1d[idxs], dtype=np.float64)

    @staticmethod
    def _as_scalar_float(x, default=0.0):
        if x is None:
            return float(default)
        try:
            arr = np.asarray(x)
            if arr.size == 0:
                return float(default)
            return float(arr.reshape(-1)[0])
        except Exception:
            return float(default)

    def infer_demo_episode_start_indices(self, transitions, horizon=None) -> List[int]:
        transitions_arr = np.asarray(transitions, dtype=object)
        if transitions_arr.ndim >= 2 and transitions_arr.shape[1] >= 5:
            starts = [0]
            done_hits = 0
            for i in range(transitions_arr.shape[0] - 1):
                done_val = self._as_scalar_float(transitions_arr[i, 4], default=0.0)
                if done_val > 0.5:
                    done_hits += 1
                    starts.append(i + 1)
            starts = sorted(set([int(s) for s in starts if 0 <= int(s) < transitions_arr.shape[0]]))
            if done_hits > 0 and len(starts) > 0:
                return starts

        if transitions_arr.ndim >= 2 and transitions_arr.shape[1] >= 1:
            episode_horizon = self.horizon if horizon is None else int(horizon)
            if episode_horizon <= 0:
                raise ValueError(f"episode_horizon must be positive, got {episode_horizon}")
            return list(range(0, transitions_arr.shape[0], episode_horizon))

        if transitions_arr.ndim == 1 and transitions_arr.shape[0] >= 1:
            return [0]
        return []

    def _extract_demo_payload_from_transition_row(self, transition_row):
        state_candidate = None
        sim_state_candidate = None
        model_xml_candidate = None

        if isinstance(transition_row, dict):
            if "state" in transition_row:
                state_candidate = transition_row.get("state")
            for key in ("sim_state", "mujoco_state", "mj_state", "flattened_state"):
                if key in transition_row:
                    sim_state_candidate = transition_row.get(key)
                    break
            for key in ("model_xml", "xml", "task_xml"):
                if key in transition_row and isinstance(transition_row.get(key), str):
                    model_xml_candidate = transition_row.get(key)
                    break
            return state_candidate, sim_state_candidate, model_xml_candidate

        if isinstance(transition_row, np.ndarray):
            if transition_row.ndim == 1 and transition_row.dtype != object:
                return transition_row, None, None
            if transition_row.ndim == 1 and transition_row.shape[0] >= 1:
                transition_row = transition_row.tolist()

        if isinstance(transition_row, (list, tuple)):
            if len(transition_row) >= 1:
                state_candidate = transition_row[0]
            if len(transition_row) >= 6:
                sim_state_candidate = transition_row[5]
            if len(transition_row) >= 7 and isinstance(transition_row[6], str):
                model_xml_candidate = transition_row[6]
        return state_candidate, sim_state_candidate, model_xml_candidate

    def _try_exact_sim_reset(self, demo_sim_state, demo_model_xml=None):
        sim_state = self._maybe_numeric_vector(demo_sim_state)
        if sim_state is None:
            return False, "invalid_sim_state"
        if sim_state.shape[0] != self.sim_state_dim:
            return False, f"sim_state_dim_mismatch:{sim_state.shape[0]}!={self.sim_state_dim}"

        try:
            if (
                isinstance(demo_model_xml, str)
                and len(demo_model_xml) > 0
                and hasattr(self.env, "reset_from_xml_string")
            ):
                self.env.reset_from_xml_string(demo_model_xml)
                self.env.sim.reset()

            self.env.sim.set_state_from_flattened(sim_state.astype(np.float64, copy=False))
            self.env.sim.forward()
            self._sync_controllers_to_current_state()
            self.env.sim.forward()

            self._obs = self.env._get_observations(force_update=True)
            self._state = obs2state(self._obs, self.env, self.env_name)
            self._agent_state, self._env_state = state2agentenv(self._state, self.env_name)
            self._step_count = 0
            return True, "ok"
        except Exception as exc:
            return False, f"{type(exc).__name__}: {exc}"

    def reset_to_demo_start(
        self,
        demo_state=None,
        transitions=None,
        episode_idx=0,
        horizon=None,
        seed=None,
        demo_sim_state=None,
        demo_model_xml=None,
        prefer_exact_sim_reset=True,
    ):
        """
        简单版本：
        1) 取 demo episode 起点 state
        2) reset 环境
        3) 按固定索引把关键物体状态写回仿真
        """
        _ = demo_sim_state
        _ = demo_model_xml
        _ = prefer_exact_sim_reset

        selected_start_index = None
        if demo_state is None:
            if transitions is None:
                raise ValueError("Provide either demo_state or transitions for reset_to_demo_start")
            transitions_arr = np.asarray(transitions, dtype=object)
            if transitions_arr.ndim == 1:
                first = np.asarray(transitions_arr[0]) if transitions_arr.shape[0] > 0 else np.array([])
                is_state_vector = transitions_arr.shape[0] == self.state_dim and first.ndim == 0
                if is_state_vector:
                    demo_state = transitions_arr
                    selected_start_index = 0
                else:
                    row = transitions_arr[0]
                    selected_start_index = 0
                    if isinstance(row, dict):
                        demo_state = row.get("state", None)
                    elif isinstance(row, np.ndarray) and row.ndim == 1 and row.dtype != object:
                        demo_state = row
                    else:
                        if isinstance(row, np.ndarray):
                            row = row.tolist()
                        demo_state = row[0] if isinstance(row, (list, tuple)) else None
            elif transitions_arr.ndim >= 2 and transitions_arr.shape[1] >= 1:
                episode_horizon = self.horizon if horizon is None else int(horizon)
                if episode_horizon <= 0:
                    raise ValueError(f"horizon must be positive, got {episode_horizon}")
                selected_start_index = int(episode_idx) * episode_horizon
                if selected_start_index < 0 or selected_start_index >= transitions_arr.shape[0]:
                    raise IndexError(
                        f"episode_idx out of range: episode_idx={episode_idx}, "
                        f"start_index={selected_start_index}, num_transitions={transitions_arr.shape[0]}"
                    )
                row = transitions_arr[selected_start_index]
                if isinstance(row, dict):
                    demo_state = row.get("state", None)
                else:
                    if isinstance(row, np.ndarray):
                        if row.ndim == 1 and row.dtype != object:
                            demo_state = row
                        else:
                            row = row.tolist()
                    demo_state = row[0] if isinstance(row, (list, tuple)) else None
            else:
                raise ValueError("Unsupported transitions format for reset_to_demo_start")

        if demo_state is None:
            raise ValueError("Could not extract demo_state from input.")

        target_state = self._to_numpy_1d(demo_state)
        if target_state.shape[0] != self.state_dim:
            raise ValueError(
                f"demo_state dim mismatch: expected {self.state_dim}, got {target_state.shape[0]}"
            )

        _, _ = self.reset(seed=seed)
        applied_fields = {}

        if self.env_name in {"Door", "Door_Close", "Old_Door", "Old_Door_Close"}:
            if hasattr(self.env, "hinge_qpos_addr"):
                self.env.sim.data.qpos[self.env.hinge_qpos_addr] = float(target_state[21])
                applied_fields["hinge_qpos"] = float(target_state[21])
            if hasattr(self.env, "handle_qpos_addr"):
                self.env.sim.data.qpos[self.env.handle_qpos_addr] = float(target_state[22])
                applied_fields["handle_qpos"] = float(target_state[22])

        if self.env_name in {"Stack", "UnStack"}:
            cubeA_pos = target_state[3:6].astype(np.float64, copy=True)
            cubeB_pos = target_state[6:9].astype(np.float64, copy=True)
            initial_cubeB_xy = target_state[9:11].astype(np.float64, copy=True)
            goal_pos = target_state[11:14].astype(np.float64, copy=True)

            self.env.initial_cubeB_xy = initial_cubeB_xy.copy()
            if hasattr(self.env, "initial_cubeA_xy"):
                self.env.initial_cubeA_xy = cubeA_pos[:2].copy()
            self.env.goal_pos = goal_pos.copy()
            applied_fields["initial_cubeB_xy"] = initial_cubeB_xy.copy()
            applied_fields["goal_pos"] = goal_pos.copy()

            if hasattr(self.env, "cubeA") and hasattr(self.env.cubeA, "joints"):
                if self._try_set_free_joint_pose(self.env.cubeA.joints[0], pos_xyz=cubeA_pos):
                    applied_fields["cubeA_pos"] = cubeA_pos.copy()
            if hasattr(self.env, "cubeB") and hasattr(self.env.cubeB, "joints"):
                if self._try_set_free_joint_pose(self.env.cubeB.joints[0], pos_xyz=cubeB_pos):
                    applied_fields["cubeB_pos"] = cubeB_pos.copy()

        if self.env_name in {"NutAssemblyRound", "NutDisAssemblyRound"}:
            nut_pos = target_state[3:6].astype(np.float64, copy=True)
            goal_pos = target_state[9:12].astype(np.float64, copy=True)
            self.env.goal_pos = goal_pos.copy()
            applied_fields["goal_pos"] = goal_pos.copy()
            if hasattr(self.env, "nuts") and len(self.env.nuts) > 0:
                nut_index = 1 if len(self.env.nuts) > 1 else 0
                nut_joint = self.env.nuts[nut_index].joints[0]
                if self._try_set_free_joint_pose(nut_joint, pos_xyz=nut_pos):
                    applied_fields["nut_pos"] = nut_pos.copy()

        if self.env_name in {"TwoArmPegInHole", "TwoArmPegRemoval"}:
            peg_pos = target_state[3:6].astype(np.float64, copy=True)
            peg_quat = target_state[6:10].astype(np.float64, copy=True)
            hole_init_pos = target_state[10:13].astype(np.float64, copy=True)
            goal_pos = target_state[13:16].astype(np.float64, copy=True)
            self.env.goal_pos = goal_pos.copy()
            self.env.hole_init_pos = hole_init_pos.copy()
            applied_fields["goal_pos"] = goal_pos.copy()
            applied_fields["hole_init_pos"] = hole_init_pos.copy()
            if hasattr(self.env, "peg") and hasattr(self.env.peg, "joints") and len(self.env.peg.joints) > 0:
                if self._try_set_free_joint_pose(self.env.peg.joints[0], pos_xyz=peg_pos, quat_wxyz=peg_quat):
                    applied_fields["peg_pos"] = peg_pos.copy()
                    applied_fields["peg_quat_wxyz"] = peg_quat.copy()

        try:
            self.env.sim.data.qvel[:] = 0.0
        except Exception:
            pass
        try:
            self.env.sim.data.qacc[:] = 0.0
        except Exception:
            pass
        try:
            if getattr(self.env.sim.data, "ctrl", None) is not None:
                self.env.sim.data.ctrl[:] = 0.0
        except Exception:
            pass

        self.env.sim.forward()
        self._sync_controllers_to_current_state()
        self._obs = self.env._get_observations(force_update=True)
        self._state = obs2state(self._obs, self.env, self.env_name)
        self._agent_state, self._env_state = state2agentenv(self._state, self.env_name)
        self._step_count = 0

        state_diff = self._state - target_state
        info = {
            "obs": self._obs,
            "agent_state": self._agent_state.copy(),
            "env_state": self._env_state.copy(),
            "success": int(self.env._check_success()),
            "step_count": self._step_count,
            "target_demo_state": target_state.copy(),
            "selected_start_index": selected_start_index,
            "approximate_reset": True,
            "exact_sim_reset": False,
            "exact_sim_reset_status": "disabled_simple_mode",
            "reset_match_l2": float(np.linalg.norm(state_diff)),
            "reset_match_linf": float(np.max(np.abs(state_diff))),
            "applied_fields": applied_fields,
        }
        if state_diff.shape[0] >= 3:
            info["reset_match_eef_l2"] = float(np.linalg.norm(state_diff[0:3]))
        if self.env_name in {"Stack", "UnStack"} and state_diff.shape[0] >= 9:
            info["reset_match_cubeA_l2"] = float(np.linalg.norm(state_diff[3:6]))
            info["reset_match_cubeB_l2"] = float(np.linalg.norm(state_diff[6:9]))
        if self.env_name in {"NutAssemblyRound", "NutDisAssemblyRound"} and state_diff.shape[0] >= 6:
            info["reset_match_nut_l2"] = float(np.linalg.norm(state_diff[3:6]))
        print(f"[DEBUG] reset to demo_start (simple): reset_match_l2={float(np.linalg.norm(state_diff))}")
        return self._state.copy(), info

    def step(self, policy_action):
        """
        Take one environment step using policy-space action.

        Args:
            policy_action: `np.ndarray | torch.Tensor`, shape `(A_policy,)`.
                Value range:
                - Two-arm tasks: expected `[-1, 1]` (will be clipped).
                - Other tasks: clipped by env action spec.
        Returns:
            next_state: `np.ndarray`, shape `(D_state,)`.
            reward: `float`.
            done: `bool`.
            info: `dict`, includes:
                - `obs`: raw next obs dict
                - `state`: `np.ndarray`, `(D_state,)`
                - `agent_state`: `np.ndarray`, `(D_agent_state,)`
                - `env_state`: `np.ndarray`, `(D_env_state,)`
                - `success`: `int` in `{0,1}`
                - `step_count`: `int`
                - `done_no_max`: `float`, 0.0 at horizon else float(done)
                - `policy_action`: `np.ndarray`, `(A_policy,)`
                - `step_action`: `np.ndarray`, `(A_env,)`
                - `mujoco_exception`: `str | None`
        """
        step_action = self._policy_action_to_env_action(policy_action)
        mujoco_exception = None

        try:
            next_obs, reward, done, env_info = self.env.step(step_action)
        except MujocoException as exc:
            if not self.catch_mujoco_exception:
                raise
            next_obs = self._obs
            reward = 0.0
            done = True
            env_info = {}
            mujoco_exception = str(exc)

        self._step_count += 1
        self._obs = next_obs
        self._state = obs2state(next_obs, self.env, self.env_name)
        self._agent_state, self._env_state = state2agentenv(self._state, self.env_name)

        done_no_max = 0.0 if self._step_count == self.horizon else float(done)
        info = dict(env_info)
        info.update(
            {
                "obs": self._obs,
                "state": self._state.copy(),
                "agent_state": self._agent_state.copy(),
                "env_state": self._env_state.copy(),
                "success": int(self.env._check_success()),
                "step_count": self._step_count,
                "done_no_max": done_no_max,
                "policy_action": self._to_numpy_1d(policy_action),
                "step_action": step_action.copy(),
                "mujoco_exception": mujoco_exception,
            }
        )
        return self._state.copy(), float(reward), bool(done), info

    def close(self):
        """
        Close underlying robosuite env.
        """
        if hasattr(self.env, "close"):
            self.env.close()

    def get_frame(self, obs=None):
        """
        Compose side-by-side rendered frame for video saving.

        Args:
            obs: `dict | None`, raw observation dict containing camera images.
        Returns:
            `np.ndarray | None`, image with shape approximately `(H_concat, W, 3)`.
        """
        source_obs = self._obs if obs is None else obs
        if source_obs is None:
            return None
        if "agentview_image" not in source_obs or "sideview_image" not in source_obs:
            return None
        return np.concatenate([source_obs["agentview_image"], source_obs["sideview_image"]])[::-1]

    def run_episode(self, policy, sample=False, max_steps=None, record_frames=False):
        """
        Roll out one episode using a callable policy or an object with `act(...)`.

        Args:
            policy:
                - callable: `action = policy(state)`
                - object: `action = policy.act(state, sample=sample)`
                Expected action shape `(A_policy,)`.
            sample: `bool`, only used when `policy` has `act`.
            max_steps: `int | None`, rollout cap; defaults to `horizon`.
            record_frames: `bool`, if True stores rendered frames from obs cameras.
        Returns:
            `dict` with keys:
                - `score`: `float`
                - `steps`: `int`
                - `success`: `int` in `{0,1}`
                - `trajectory`: list of transition dicts (`state/action/reward/next_state/done`)
                - `frames`: list of image arrays (if `record_frames=True`)
        """
        state, info = self.reset()
        done = False
        steps = 0
        score = 0.0
        trajectory = []
        frames = []

        # Settle down the environment for a few steps to get consistent initial observations.
        print("[DEBUG] Settling environment for consistent initial observations...")
        for _ in range(30):
            action = np.zeros(self.policy_action_dim, dtype=np.float32)
            next_state, reward, done, step_info = self.step(action)

        limit = self.horizon if max_steps is None else max_steps
        while not done and steps < limit:
            if callable(policy):
                action = policy(state)
            elif hasattr(policy, "act"):
                action = policy.act(state, sample=sample)
            else:
                raise ValueError("policy must be callable or provide act(state, sample=...)")

            next_state, reward, done, step_info = self.step(action)
            score += reward
            steps += 1

            trajectory.append(
                {
                    "state": state.copy(),
                    "action": self._to_numpy_1d(action),
                    "reward": float(reward),
                    "next_state": next_state.copy(),
                    "done": bool(done),
                }
            )
            if record_frames:
                frame = self.get_frame(step_info["obs"])
                if frame is not None:
                    frames.append(frame)

            state = next_state
            info = step_info

        print(f"[DEBUG] success={info['success']}, steps={steps}, score={score}")
        return {
            "score": score,
            "steps": steps,
            "success": int(info["success"]) if info is not None else 0,
            "trajectory": trajectory,
            "frames": frames,
        }

    def evaluate(self, policy, num_episodes=20, sample=False):
        """
        Evaluate policy over multiple episodes.

        Args:
            policy: same convention as `run_episode`.
            num_episodes: `int`, number of rollouts.
            sample: `bool`, passed to `policy.act` when applicable.
        Returns:
            `dict`:
                - `mean_score`: `float`
                - `mean_success`: `float` in `[0, 1]`
                - `mean_steps`: `float`
                - plus per-episode lists `scores/successes/steps`
        """
        scores = []
        successes = []
        steps = []
        for _ in tqdm(range(num_episodes)):
            episode = self.run_episode(policy=policy, sample=sample, record_frames=False)
            scores.append(episode["score"])
            successes.append(episode["success"])
            steps.append(episode["steps"])
        return {
            "num_episodes": int(num_episodes),
            "mean_score": float(np.mean(scores)),
            "mean_success": float(np.mean(successes)),
            "mean_steps": float(np.mean(steps)),
            "scores": scores,
            "successes": successes,
            "steps": steps,
        }


class RobosuiteSocketEvaluator:
    """
    Minimal socket evaluator:
    - local env rollout via `RobosuiteEvaluator`
    - remote policy inference via HTTP `/init` `/reset` `/step`
    """

    def __init__(
        self,
        # RobosuiteEvaluator params
        env_name: str,
        use_camera_obs=True,
        has_renderer=False,
        has_offscreen_renderer=True,
        control_freq=20,
        camera_names=("agentview", "sideview"),
        camera_heights=256,
        camera_widths=256,
        # Eval params
        test_cnt: int = 20,
        test_start_seed: int = 10000,
        test_max_steps: int = 500,
        # Socket related
        policy_url: str = "http://localhost:6006",
        send_per_frames: int = 1,
        num_obs_steps: Optional[int] = None,
        request_timeout: float = 30.0,
    ):
        try:
            import requests  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise ImportError("RobosuiteSocketEvaluator requires `requests`.") from exc

        self.http_session = requests.Session()
        self.policy_url = self._normalize_policy_url(policy_url)
        self.send_per_frames = max(int(send_per_frames), 1)
        self.num_obs_steps = self.send_per_frames if num_obs_steps is None else max(int(num_obs_steps), 1)
        self.request_timeout = float(request_timeout)

        self.test_cnt = int(test_cnt)
        self.test_start_seed = int(test_start_seed)
        self.test_max_steps = int(test_max_steps)

        self.env = RobosuiteEvaluator(
            env_name=env_name,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            control_freq=control_freq,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            horizon=self.test_max_steps,
        )
        self.env_name = self.env.env_name
        self.policy_action_dim = self.env.policy_action_dim
        self.env_action_dim = self.env.env_action_dim
        self.task_instruction = self.env_name
        self.camera_obs_keys = [f"{name}_image" for name in camera_names]
        self.camera_heights = int(camera_heights)
        self.camera_widths = int(camera_widths)

        self.socket_initialized = False
        self.send_cnt = 0
        self.cache_actions_T_D = None
        self.tcp_state_buffer = []
        self.frame_buffer = {key: [] for key in self.camera_obs_keys}
        self.default_tcp_state = np.zeros((self.env.state_dim,), dtype=np.float32)
        self.default_frame = np.zeros((self.camera_heights, self.camera_widths, 3), dtype=np.uint8)

    @staticmethod
    def _normalize_policy_url(url: str) -> str:
        url = str(url).strip()
        if not url.startswith("http://") and not url.startswith("https://"):
            url = f"http://{url}"
        return url.rstrip("/")

    def _reset_action_cache(self):
        self.send_cnt = 0
        self.cache_actions_T_D = None

    def _reset_state_buffer(self):
        self.tcp_state_buffer.clear()
        for key in self.frame_buffer:
            self.frame_buffer[key].clear()

    def _to_uint8_rgb(self, image: Optional[np.ndarray]) -> np.ndarray:
        if image is None:
            return self.default_frame.copy()

        arr = np.asarray(image)
        if arr.ndim == 2:
            arr = np.repeat(arr[..., None], 3, axis=2)
        elif arr.ndim == 3 and arr.shape[2] == 4:
            arr = arr[..., :3]
        elif arr.ndim != 3 or arr.shape[2] != 3:
            return self.default_frame.copy()

        if arr.dtype != np.uint8:
            if np.issubdtype(arr.dtype, np.floating):
                arr = np.clip(arr, 0.0, 1.0)
                arr = np.rint(arr * 255.0).astype(np.uint8)
            else:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
        return arr

    def _append_with_limit(self, buffer: list, value: np.ndarray, limit: int):
        buffer.append(value.copy())
        if len(buffer) > int(limit):
            buffer.pop(0)

    def _pad_stack(self, buffer: list, default_value: np.ndarray, target_len: int) -> np.ndarray:
        if len(buffer) == 0:
            buffer = [default_value.copy()]
        pad_n = int(target_len) - len(buffer)
        if pad_n > 0:
            padded = [buffer[0].copy() for _ in range(pad_n)] + [b.copy() for b in buffer]
        else:
            padded = [b.copy() for b in buffer]
        return np.stack(padded, axis=0)

    def _update_buffers(self, state: np.ndarray, info: Dict[str, Any]):
        state_arr = np.asarray(state, dtype=np.float32).reshape(-1)
        self._append_with_limit(self.tcp_state_buffer, state_arr, limit=self.num_obs_steps)

        obs = info.get("obs", {})
        for key in self.camera_obs_keys:
            frame = self._to_uint8_rgb(obs.get(key))
            self._append_with_limit(self.frame_buffer[key], frame, limit=self.num_obs_steps)

    def _build_step_request(self, stage_flag: int) -> StepRequestFromEvaluator:
        tcp_state_T_D = self._pad_stack(
            self.tcp_state_buffer,
            self.default_tcp_state,
            target_len=self.num_obs_steps,
        )

        camera_videos = []
        for key in self.camera_obs_keys:
            video_T_H_W_C = self._pad_stack(
                self.frame_buffer[key],
                self.default_frame,
                target_len=self.num_obs_steps,
            )
            camera_videos.append(video_T_H_W_C)

        if len(camera_videos) == 0:
            camera_videos = [np.repeat(self.default_frame[None, ...], self.num_obs_steps, axis=0)]
        num_camera_views = len(camera_videos)
        gt_video_B_VT_H_W_C = np.concatenate(camera_videos, axis=0)[None, ...]

        return StepRequestFromEvaluator.encode_from_raw(
            instruction=self.task_instruction,
            stage_flag=int(stage_flag),
            gt_video=gt_video_B_VT_H_W_C.astype(np.uint8),
            num_camera_views=int(num_camera_views),
            tcp_state=tcp_state_T_D[None, ...].astype(np.float32),
            max_cache_action=-1,
        )

    def _decode_action_cache(self, response_json: Dict[str, Any]) -> np.ndarray:
        decoded = StepRequestFromPolicy(action=response_json["action"]).decode_to_raw()
        action_B_H_D = np.asarray(decoded["action"], dtype=np.float32)
        if action_B_H_D.ndim != 3 or action_B_H_D.shape[0] != 1:
            raise ValueError(f"Expected action shape (1,H,D), got {action_B_H_D.shape}")

        action_H_D = action_B_H_D[0]
        if action_H_D.shape[1] != self.policy_action_dim:
            raise ValueError(
                f"Policy action dim mismatch: expected {self.policy_action_dim}, got {action_H_D.shape[1]}"
            )
        if action_H_D.shape[0] < self.send_per_frames:
            pad = np.repeat(action_H_D[-1:], self.send_per_frames - action_H_D.shape[0], axis=0)
            action_H_D = np.concatenate([action_H_D, pad], axis=0)
        return action_H_D

    def init_socket(self) -> int:
        resp = self.http_session.get(f"{self.policy_url}/init", timeout=self.request_timeout)
        resp.raise_for_status()
        data = resp.json() if resp.content else {}
        self.send_per_frames = max(1, int(data.get("max_cache_action", self.send_per_frames)))
        self._reset_action_cache()
        self._reset_state_buffer()
        self.socket_initialized = True
        return self.send_per_frames

    def send_reset(self) -> int:
        resp = self.http_session.get(f"{self.policy_url}/reset", timeout=self.request_timeout)
        resp.raise_for_status()
        data = resp.json() if resp.content else {}
        self.send_per_frames = max(1, int(data.get("max_cache_action", self.send_per_frames)))
        self._reset_action_cache()
        self._reset_state_buffer()
        return self.send_per_frames

    def send_obs_and_get_action(
        self,
        state: np.ndarray,
        info: Dict[str, Any],
        stage_flag: int = 0,
    ) -> np.ndarray:
        if not self.socket_initialized:
            self.init_socket()

        self._update_buffers(state=state, info=info)
        if self.send_cnt % self.send_per_frames == 0:
            request_to_policy = self._build_step_request(stage_flag=stage_flag)
            resp = self.http_session.post(
                f"{self.policy_url}/step",
                json=request_to_policy.model_dump(mode="json"),
                timeout=self.request_timeout,
            )
            resp.raise_for_status()
            resp_data = resp.json()
            self.cache_actions_T_D = self._decode_action_cache(resp_data)

        if self.cache_actions_T_D is None:
            raise RuntimeError("No cached action available from policy server.")
        action = self.cache_actions_T_D[self.send_cnt % self.send_per_frames].astype(np.float32)
        self.send_cnt += 1
        return action

    def reset(
        self,
        seed: Optional[int] = None,
        demo_state: Optional[np.ndarray] = None,
        transitions: Optional[np.ndarray] = None,
        episode_idx: int = 0,
        demo_horizon: Optional[int] = None,
    ):
        self._reset_state_buffer()
        self._reset_action_cache()
        if demo_state is not None or transitions is not None:
            return self.env.reset_to_demo_start(
                demo_state=demo_state,
                transitions=transitions,
                episode_idx=episode_idx,
                horizon=demo_horizon,
                seed=seed,
            )
        return self.env.reset(seed=seed)

    def step(self, policy_action: np.ndarray):
        return self.env.step(policy_action)

    def run_episode(
        self,
        seed: Optional[int] = None,
        max_steps: Optional[int] = None,
        reset_policy: bool = True,
        close_online: bool = True,
        record_frames: bool = False,
        save_video_path: Optional[str] = None,
        video_fps: Optional[int] = None,
        demo_state: Optional[np.ndarray] = None,
        demo_transitions: Optional[np.ndarray] = None,
        demo_episode_idx: int = 0,
        demo_horizon: Optional[int] = None,
    ) -> Dict[str, Any]:
        if not self.socket_initialized:
            self.init_socket()
        if reset_policy:
            self.send_reset()
        else:
            self._reset_action_cache()
            self._reset_state_buffer()

        state, info = self.reset(
            seed=seed,
            demo_state=demo_state,
            transitions=demo_transitions,
            episode_idx=demo_episode_idx,
            demo_horizon=demo_horizon,
        )
        limit = self.test_max_steps if max_steps is None else int(max_steps)
        done = False
        steps = 0
        score = 0.0
        stage_flag = 0
        trajectory = []
        frames = []

        # NOTE: Settle down the environment for a few steps to get consistent initial observations.
        # for _ in range(10):
        #     action = np.zeros(self.policy_action_dim, dtype=np.float32)
        #     next_state, reward, done, step_info = self.step(action)
        # print("[DEBUG] Settled environment down for consistent initial observations")

        while not done and steps < limit:
            action = self.send_obs_and_get_action(state=state, info=info, stage_flag=stage_flag)
            next_state, reward, done, step_info = self.step(action)
            trajectory.append(
                {
                    "state": state.copy(),
                    "action": np.asarray(action, dtype=np.float32).copy(),
                    "reward": float(reward),
                    "next_state": next_state.copy(),
                    "done": bool(done),
                }
            )
            if record_frames:
                frame = self.env.get_frame(step_info.get("obs"))
                if frame is not None:
                    frames.append(frame)

            steps += 1
            score += float(reward)
            state, info = next_state, step_info
            if not close_online:
                stage_flag = 1

        video_path = None
        if save_video_path is not None and len(frames) > 0:
            fps = self.env.control_freq if video_fps is None else int(video_fps)
            save_frames_as_video(
                frames=frames,
                save_path=save_video_path,
                fps=max(fps, 1),
            )
            video_path = save_video_path

        print(f"[DEBUG] success={info['success']}, steps={steps}, score={score}")
        return {
            "score": float(score),
            "steps": int(steps),
            "success": int(info.get("success", 0)),
            "trajectory": trajectory,
            "frames": frames,
            "video_path": video_path,
        }

    def evaluate(
        self,
        test_cnt: Optional[int] = None,
        test_start_seed: Optional[int] = None,
        test_max_steps: Optional[int] = None,
        reset_policy_each_episode: bool = True,
        save_video: bool = False,
        video_dir: Optional[str] = None,
        video_fps: Optional[int] = None,
        video_prefix: str = "rollout",
        demo_transitions: Optional[np.ndarray] = None,
        demo_horizon: Optional[int] = None,
        demo_episode_idx: Optional[int] = None,
    ) -> Dict[str, Any]:
        n_eval = self.test_cnt if test_cnt is None else int(test_cnt)
        seed0 = self.test_start_seed if test_start_seed is None else int(test_start_seed)
        max_steps = self.test_max_steps if test_max_steps is None else int(test_max_steps)
        effective_demo_horizon = self.env.horizon if demo_horizon is None else int(demo_horizon)

        if save_video and video_dir is None:
            video_dir = "./debug/robosuite_rollout"
        if video_dir is not None:
            os.makedirs(video_dir, exist_ok=True)
            print(f"[DEBUG] video will be saved to: {video_dir}")

        auto_demo_episode_cycle = (demo_episode_idx is not None) and (int(demo_episode_idx) == -1)
        n_demo_episodes = None
        if auto_demo_episode_cycle:
            if demo_transitions is None:
                raise ValueError("demo_episode_idx=-1 requires demo_transitions to be provided.")
            demo_episode_starts = self.env.infer_demo_episode_start_indices(
                demo_transitions, horizon=effective_demo_horizon
            )
            n_demo_episodes = max(1, int(len(demo_episode_starts)))

        scores = []
        successes = []
        steps_list = []
        video_paths = []
        for i in range(n_eval):
            cur_seed = seed0 + i
            cur_video_path = None
            if auto_demo_episode_cycle:
                cur_demo_episode_idx = int(i % n_demo_episodes)
            else:
                cur_demo_episode_idx = i if demo_episode_idx is None else int(demo_episode_idx)
            if save_video and video_dir is not None:
                cur_video_path = os.path.join(
                    video_dir,
                    f"{video_prefix}_{self.env_name}_ep{i:03d}_seed{cur_seed}.mp4",
                )
            rollout = self.run_episode(
                seed=cur_seed,
                max_steps=max_steps,
                reset_policy=reset_policy_each_episode,
                close_online=True,
                record_frames=save_video,
                save_video_path=cur_video_path,
                video_fps=video_fps,
                demo_transitions=demo_transitions,
                demo_episode_idx=cur_demo_episode_idx,
                demo_horizon=effective_demo_horizon,
            )
            scores.append(float(rollout["score"]))
            successes.append(int(rollout["success"]))
            steps_list.append(int(rollout["steps"]))
            if rollout.get("video_path") is not None:
                video_paths.append(rollout["video_path"])

        result = {
            "num_episodes": int(n_eval),
            "mean_score": float(np.mean(scores)) if len(scores) > 0 else 0.0,
            "mean_success": float(np.mean(successes)) if len(successes) > 0 else 0.0,
            "mean_steps": float(np.mean(steps_list)) if len(steps_list) > 0 else 0.0,
            "scores": scores,
            "successes": successes,
            "steps": steps_list,
        }
        if len(video_paths) > 0:
            result["video_paths"] = video_paths
        return result

    def close(self):
        try:
            self.env.close()
        finally:
            self.http_session.close()
