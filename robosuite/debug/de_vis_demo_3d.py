from wrapper.transition_demo_3d_visualizer import TransitionDemo3DVisualizer

env_name = "UnStack"
# env_name = "Stack"

# env_name = "NutDisAssemblyRound"
# env_name = "NutAssemblyRound"

# env_name = "TwoArmPegRemoval"
# env_name = "TwoArmPegInHole"

vis = TransitionDemo3DVisualizer(
    env_name=env_name,
    transition_path=f"generate/{env_name}_transitions_10trajectory_sparse.npy",
    output_dir=f"debug/videos/{env_name.lower()}_3d",
    horizon=500, fps=20
)
vis.render_all_episodes(merge_into_one_video=True)
