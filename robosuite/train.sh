export CFLAGS="-Wno-error=incompatible-pointer-types -Wno-error=implicit-function-declaration -Wno-error=int-conversion"
export MUJOCO_GL="egl"q

python train_sac_agentenv_2agents_state.py --reward_shaping f --use_reversed_transition t --filter_type state_max_diff --diff_threshold 0.01 --use_reversed_reward t --use_forward_reward t --reward_model_type potential --potential_type linear --n_demo 10 --env1_name Door --env2_name Door_Close --seed 1