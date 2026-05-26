"""
Launcher for experiments with PEARL

"""
import os
import pathlib
import numpy as np
import click
import json
import torch

from rlkit.envs import ENVS
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.torch.sac.policies import TanhGaussianPolicy
from rlkit.torch.networks import FlattenMlp, MlpEncoder, RecurrentEncoder
from rlkit.torch.sac.sac import PEARLSoftActorCritic
from rlkit.torch.sac.agent import PEARLAgent
from rlkit.launchers.launcher_util import setup_logger
import rlkit.torch.pytorch_util as ptu
from configs.default import default_config


def experiment(variant):

    # create multi-task environment and sample tasks
    env = NormalizedBoxEnv(ENVS[variant['env_name']](**variant['env_params']))
    tasks = env.get_all_task_idx()
    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))
    reward_dim = 1

    # instantiate networks
    latent_dim = variant['latent_size']
    context_encoder_input_dim = 2 * obs_dim + action_dim + reward_dim if variant['algo_params']['use_next_obs_in_context'] else obs_dim + action_dim + reward_dim
    context_encoder_output_dim = latent_dim * 2 if variant['algo_params']['use_information_bottleneck'] else latent_dim
    net_size = variant['net_size']
    recurrent = variant['algo_params']['recurrent']
    encoder_model = RecurrentEncoder if recurrent else MlpEncoder

    method = variant.get('method', 'baseline')
    if method == 'flow':
        from rlkit.torch.flow.flow_encoder import FlowContextEncoder
        # optional learned prior v_phi (paper Eq. 5). When disabled, the encoder
        # uses the N(0,I) closed-form prior score (-c) -- the legacy MVP path.
        prior_flow = None
        if variant['flow_params'].get('use_prior_flow', False):
            from rlkit.torch.flow.prior_flow import PriorFlow
            prior_flow = PriorFlow(
                latent_dim=latent_dim,
                hidden_dim=variant['flow_params']['prior_hidden'],
            )
        context_encoder = FlowContextEncoder(
            context_dim=context_encoder_input_dim,
            latent_dim=latent_dim,
            hidden_dim=variant['flow_params']['encoder_hidden'],
            n_ode_steps=variant['flow_params']['n_ode_steps'],
            max_context=variant['flow_params'].get('max_context', 16),
            vel_clip=variant['flow_params'].get('vel_clip', 10.0),
            tau_eps=variant['flow_params'].get('tau_eps', 0.05),
            prior_flow=prior_flow,
        )
    else:
        context_encoder = encoder_model(
            hidden_sizes=[200, 200, 200],
            input_size=context_encoder_input_dim,
            output_size=context_encoder_output_dim,
        )
    qf1 = FlattenMlp(
        hidden_sizes=[net_size, net_size, net_size],
        input_size=obs_dim + action_dim + latent_dim,
        output_size=1,
    )
    qf2 = FlattenMlp(
        hidden_sizes=[net_size, net_size, net_size],
        input_size=obs_dim + action_dim + latent_dim,
        output_size=1,
    )
    vf = FlattenMlp(
        hidden_sizes=[net_size, net_size, net_size],
        input_size=obs_dim + latent_dim,
        output_size=1,
    )
    policy = TanhGaussianPolicy(
        hidden_sizes=[net_size, net_size, net_size],
        obs_dim=obs_dim + latent_dim,
        latent_dim=latent_dim,
        action_dim=action_dim,
    )
    if method == 'flow':
        from rlkit.torch.flow.decoder import TransitionDecoder
        from rlkit.torch.flow.flow_agent import FlowPEARLAgent
        from rlkit.torch.flow.flow_sac import FlowPEARLSoftActorCritic
        decoder = TransitionDecoder(
            obs_dim, action_dim, latent_dim,
            hidden_dim=variant['flow_params']['decoder_hidden'],
        )
        agent = FlowPEARLAgent(
            latent_dim, context_encoder, policy, decoder,
            prior_flow=prior_flow,
            **variant['algo_params']
        )
        algo_class = FlowPEARLSoftActorCritic
    else:
        agent = PEARLAgent(
            latent_dim,
            context_encoder,
            policy,
            **variant['algo_params']
        )
        algo_class = PEARLSoftActorCritic
    algorithm = algo_class(
        env=env,
        train_tasks=list(tasks[:variant['n_train_tasks']]),
        eval_tasks=list(tasks[-variant['n_eval_tasks']:]),
        nets=[agent, qf1, qf2, vf],
        latent_dim=latent_dim,
        **variant['algo_params']
    )

    if method == 'flow':
        algorithm.recon_weight = variant['flow_params']['recon_weight']
        algorithm.collapse_eps = variant['flow_params']['collapse_eps']
        algorithm.cfm_weight = variant['flow_params']['cfm_weight']
        algorithm.cfm_warmup_steps = variant['flow_params']['cfm_warmup_steps']
        # paper-mode additions
        # base PEARLSoftActorCritic.training_mode is a method; use a distinct
        # attribute name to avoid shadowing it. JSON config key is unchanged.
        algorithm.flow_training_mode = variant['flow_params'].get('training_mode', 'fusedVel+decoderCFM')
        algorithm.prior_weight = variant['flow_params'].get('prior_weight', 1.0)
        algorithm.prior_flow = prior_flow
        if prior_flow is not None:
            algorithm.prior_optimizer = torch.optim.Adam(
                prior_flow.parameters(),
                lr=variant['algo_params']['context_lr'],
            )

    # optionally load pre-trained weights
    if variant['path_to_weights'] is not None:
        path = variant['path_to_weights']
        context_encoder.load_state_dict(torch.load(os.path.join(path, 'context_encoder.pth')))
        qf1.load_state_dict(torch.load(os.path.join(path, 'qf1.pth')))
        qf2.load_state_dict(torch.load(os.path.join(path, 'qf2.pth')))
        vf.load_state_dict(torch.load(os.path.join(path, 'vf.pth')))
        # TODO hacky, revisit after model refactor
        algorithm.networks[-2].load_state_dict(torch.load(os.path.join(path, 'target_vf.pth')))
        policy.load_state_dict(torch.load(os.path.join(path, 'policy.pth')))

    # optional GPU mode
    ptu.set_gpu_mode(variant['util_params']['use_gpu'], variant['util_params']['gpu_id'])
    if ptu.gpu_enabled():
        algorithm.to()

    # debugging triggers a lot of printing and logs to a debug directory
    DEBUG = variant['util_params']['debug']
    os.environ['DEBUG'] = str(int(DEBUG))

    # create logging directory
    exp_id = 'debug' if DEBUG else None
    experiment_log_dir = setup_logger(variant['env_name'], variant=variant, exp_id=exp_id, base_log_dir=variant['util_params']['base_log_dir'])

    # optionally save eval trajectories as pkl files
    if variant['algo_params']['dump_eval_paths']:
        pickle_dir = experiment_log_dir + '/eval_trajectories'
        pathlib.Path(pickle_dir).mkdir(parents=True, exist_ok=True)

    # optional Weights & Biases logging
    use_wandb = variant['util_params'].get('use_wandb', False)
    if use_wandb:
        _setup_wandb(variant, experiment_log_dir)

    # run the algorithm
    algorithm.train()

    if use_wandb:
        import wandb
        wandb.finish()

def deep_update_dict(fr, to):
    ''' update dict of dicts with new values '''
    # assume dicts have same keys
    for k, v in fr.items():
        if type(v) is dict:
            deep_update_dict(v, to[k])
        else:
            to[k] = v
    return to


def _to_numeric(d):
    ''' best-effort convert a dict of stringified values to floats for logging '''
    out = {}
    for k, v in d.items():
        try:
            out[k] = float(v)
        except (TypeError, ValueError):
            pass
    return out


def _setup_wandb(variant, experiment_log_dir):
    ''' init W&B and forward every dump_tabular() row to it '''
    import wandb
    from rlkit.core import logger
    # custom run name from config (util_params.wandb_run_name);
    # falls back to the log dir basename if not set.
    run_name = variant['util_params'].get('wandb_run_name')
    if not run_name:
        run_name = os.path.basename(experiment_log_dir.rstrip('/'))
    wandb.init(
        project=variant['util_params'].get('wandb_project', 'pearl'),
        name=run_name,
        config=variant,
        dir=experiment_log_dir,
    )

    def _log(tabular_dict):
        metrics = _to_numeric(tabular_dict)
        epoch = metrics.pop('Epoch', None)
        if epoch is not None:
            wandb.log(metrics, step=int(epoch))
        else:
            wandb.log(metrics)

    logger.add_tabular_callback(_log)

@click.command()
@click.argument('config', default=None)
@click.option('--gpu', default=0)
@click.option('--debug', is_flag=True, default=False)
@click.option('--num-iterations', type=int, default=None,
              help='Override algo_params.num_iterations (for smoke tests).')
@click.option('--num-evals', type=int, default=None,
              help='Override algo_params.num_evals (smoothing eval curves).')
@click.option('--no-wandb', is_flag=True, default=False,
              help='Disable wandb logging for this run (overrides config).')
@click.option('--wandb-run-name', type=str, default=None,
              help='Override util_params.wandb_run_name for this run.')
def main(config, gpu, debug, num_iterations, num_evals, no_wandb, wandb_run_name):

    variant = default_config
    if config:
        with open(os.path.join(config)) as f:
            exp_params = json.load(f)
        variant = deep_update_dict(exp_params, variant)
    variant['util_params']['gpu_id'] = gpu
    if num_iterations is not None:
        variant['algo_params']['num_iterations'] = num_iterations
    if num_evals is not None:
        variant['algo_params']['num_evals'] = num_evals
    if no_wandb:
        variant['util_params']['use_wandb'] = False
    if wandb_run_name is not None:
        variant['util_params']['wandb_run_name'] = wandb_run_name

    experiment(variant)

if __name__ == "__main__":
    main()

