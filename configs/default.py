# default PEARL experiment settings
# all experiments should modify these settings only as needed
default_config = dict(
    env_name='cheetah-dir',
    n_train_tasks=2,
    n_eval_tasks=2,
    latent_size=5, # dimension of the latent context vector
    net_size=300, # number of units per FC layer in each network
    path_to_weights=None, # path to pre-trained weights to load into networks
    method='baseline', # 'baseline' = original PEARL; 'flow' = flow-matching context inference; 'varibad' = PPO+VAE VariBAD
    env_params=dict(
        n_tasks=2, # number of distinct tasks in this domain, shoudl equal sum of train and eval tasks
        randomize_tasks=True, # shuffle the tasks after creating them
    ),
    algo_params=dict(
        meta_batch=16, # number of tasks to average the gradient across
        num_iterations=500, # number of data sampling / training iterates
        num_initial_steps=2000, # number of transitions collected per task before training
        num_tasks_sample=5, # number of randomly sampled tasks to collect data for each iteration
        num_steps_prior=400, # number of transitions to collect per task with z ~ prior
        num_steps_posterior=0, # number of transitions to collect per task with z ~ posterior
        num_extra_rl_steps_posterior=400, # number of additional transitions to collect per task with z ~ posterior that are only used to train the policy and NOT the encoder
        num_train_steps_per_itr=2000, # number of meta-gradient steps taken per iteration
        num_evals=2, # number of independent evals
        num_steps_per_eval=600,  # nuumber of transitions to eval on
        batch_size=256, # number of transitions in the RL batch
        embedding_batch_size=64, # number of transitions in the context batch
        embedding_mini_batch_size=64, # number of context transitions to backprop through (should equal the arg above except in the recurrent encoder case)
        max_path_length=200, # max path length for this environment
        discount=0.99, # RL discount factor
        soft_target_tau=0.005, # for SAC target network update
        policy_lr=3E-4,
        qf_lr=3E-4,
        vf_lr=3E-4,
        context_lr=3e-4,
        reward_scale=5., # scale rewards before constructing Bellman update, effectively controls weight on the entropy of the policy
        sparse_rewards=False, # whether to sparsify rewards as determined in env
        kl_lambda=.1, # weight on KL divergence term in encoder loss
        use_information_bottleneck=True, # False makes latent context deterministic
        use_next_obs_in_context=False, # use next obs if it is useful in distinguishing tasks
        update_post_train=1, # how often to resample the context when collecting data during training (in trajectories)
        num_exp_traj_eval=1, # how many exploration trajs to collect before beginning posterior sampling at test time
        recurrent=False, # recurrent or permutation-invariant encoder
        dump_eval_paths=False, # whether to save evaluation trajectories
    ),
    util_params=dict(
        base_log_dir='output',
        use_gpu=True,
        gpu_id=0,
        debug=False, # debugging triggers printing and writes logs to debug directory
        docker=False, # TODO docker is not yet supported
        use_wandb=False, # log per-iteration metrics to Weights & Biases
        wandb_project='pearl', # W&B project name (used when use_wandb=True)
    ),
    flow_params=dict(
        encoder_hidden=128, # hidden width of the flow context encoder
        decoder_hidden=128, # hidden width of the transition decoder
        n_ode_steps=5,      # ODE integration steps for sampling c (Stage 2+)
        recon_weight=1.0,   # weight of the decoder reconstruction (ELBO) loss
        use_dynamics_decoder=True, # include next-state decoder head; False => reward-head grounding only
        collapse_eps=1e-4,  # c_variance below this => latent collapse, abort
        cfm_weight=1.0,     # weight of the CFM (flow-matching) loss
        cfm_warmup_steps=0, # train steps of recon-only before CFM turns on
        # ---- paper-method additions (Eq. 5-7); defaults preserve current run --
        training_mode='current',  # 'current' | 'paper' | 'paper+recon'
        use_prior_flow=False,     # learn unconditional v_phi for the marginal p(c)
        prior_hidden=128,         # hidden width of the prior flow v_phi
        prior_weight=1.0,         # alpha' in Eq. 7 (weight on L_prior)
        # ---- numerical guards / ablation knobs (paper Plan §4 stage 4) -------
        max_context=16,           # subsample context for fused ODE; None = use ALL transitions (paper-faithful)
        vel_clip=10.0,            # tanh-squash on fused velocity norm (MVP guard); set high to effectively disable
        tau_eps=0.05,             # ODE integration interval is [tau_eps, 1-tau_eps]
    ),
    varibad_params=dict(
        max_rollouts_per_task=2,
        max_path_length=200,
        num_tasks_per_update=1,
        policy_num_steps=400,
        num_processes=1,
        num_frames=10000000,
        num_evals=2,
        num_steps_per_eval=600,
        eval_shuffled_latent=False,
        log_rollout_diagnostics=False,
        max_logged_rollout_events=8,
        policy_gamma=0.99,
        policy_tau=0.95,
        policy_use_gae=True,
        policy_value_loss_coef=0.5,
        policy_entropy_coef=0.01,
        policy_optimiser='adam',
        policy_eps=1e-5,
        policy_max_grad_norm=0.5,
        policy_layers=[300, 300],
        policy_activation_function='tanh',
        policy_initialisation='orthogonal',
        policy_init_std=1.0,
        lr_policy=7e-4,
        ppo_num_epochs=5,
        ppo_num_minibatch=5,
        ppo_clip_param=0.2,
        ppo_use_huberloss=True,
        ppo_use_clipped_value_loss=True,
        pass_state_to_policy=True,
        pass_latent_to_policy=True,
        pass_belief_to_policy=False,
        pass_task_to_policy=False,
        norm_state_for_policy=False,
        norm_latent_for_policy=False,
        norm_belief_for_policy=False,
        norm_task_for_policy=False,
        norm_rew_for_policy=False,
        norm_actions_pre_sampling=False,
        norm_actions_post_sampling=True,
        append_done_to_obs=False,
        append_done_to_encoder=False,
        latent_input_mode='mean_logvar',
        sample_embeddings=False,
        add_nonlinearity_to_latent=False,
        encoder_gru_hidden_size=128,
        encoder_layers_before_gru=[],
        encoder_layers_after_gru=[],
        action_embedding_size=16,
        state_embedding_size=32,
        reward_embedding_size=16,
        lr_vae=1e-3,
        decode_reward=True,
        decode_state=False,
        disable_decoder=False,
        disable_kl_term=False,
        disable_stochasticity_in_latent=False,
        kl_to_gauss_prior=False,
        rew_loss_coeff=1.0,
        state_loss_coeff=0.0,
        kl_weight=0.1,
        reward_decoder_layers=[300, 300],
        state_decoder_layers=[300, 300],
        rew_pred_type='deterministic',
        state_pred_type='deterministic',
        input_prev_state=True,
        input_action=True,
        size_vae_buffer=10000,
        vae_buffer_add_thresh=1.0,
        vae_batch_num_trajs=25,
        num_vae_updates=1,
        encoder_max_grad_norm=0.5,
        decoder_max_grad_norm=0.5,
        tbptt_stepsize=None,
    ),
)
