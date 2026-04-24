GPU_ID=0

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export MUJOCO_EGL_DEVICE_ID="${GPU_ID}"

export CFLAGS="-Wno-error=incompatible-pointer-types -Wno-error=implicit-function-declaration -Wno-error=int-conversion"
export MUJOCO_GL="egl"

python train_sac_agentenv_2agents_state.py --reward_shaping f --use_reversed_transition t  \
  --filter_type state_max_diff --diff_threshold 0.01  \
  --use_reversed_reward t --use_forward_reward t  \
  --reward_model_type potential --potential_type linear --n_demo 10  \
  --env1_name Stack --env2_name UnStack  \
  --seed 1

#python train_sac_agentenv_2agents_state.py --reward_shaping f --use_reversed_transition t  \
#  --filter_type state_max_diff --diff_threshold 0.01  \
#  --use_reversed_reward t --use_forward_reward t  \
#  --reward_model_type potential --potential_type linear --n_demo 10  \
#  --env1_name NutAssemblyRound --env2_name NutDisAssemblyRound  \
#  --seed 1
