"""
Launcher for experiments with PEARL

"""
import os
import pathlib
import numpy as np
import click
import json
import ast
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
    method = variant.get('method', 'baseline')

    if method == 'varibad':
        if variant['path_to_weights'] is not None:
            raise ValueError('path_to_weights is only wired for PEARL/flow, not method="varibad".')
        from rlkit.torch.varibad.varibad_algorithm import VariBADAlgorithm
        algorithm = VariBADAlgorithm(
            env=env,
            train_tasks=list(tasks[:variant['n_train_tasks']]),
            eval_tasks=list(tasks[-variant['n_eval_tasks']:]),
            obs_dim=obs_dim,
            action_dim=action_dim,
            latent_dim=variant['latent_size'],
            net_size=variant['net_size'],
            algo_params=variant['algo_params'],
            varibad_params=variant.get('varibad_params', {}),
        )
    else:
        # instantiate PEARL/flow networks
        latent_dim = variant['latent_size']
        context_encoder_input_dim = 2 * obs_dim + action_dim + reward_dim if variant['algo_params']['use_next_obs_in_context'] else obs_dim + action_dim + reward_dim
        context_encoder_output_dim = latent_dim * 2 if variant['algo_params']['use_information_bottleneck'] else latent_dim
        net_size = variant['net_size']
        recurrent = variant['algo_params']['recurrent']
        encoder_model = RecurrentEncoder if recurrent else MlpEncoder

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
            algorithm.use_dynamics_decoder = variant['flow_params'].get('use_dynamics_decoder', True)
            # paper-mode additions
            # base PEARLSoftActorCritic.training_mode is a method; use a distinct
            # attribute name to avoid shadowing it. JSON config key is unchanged.
            algorithm.flow_training_mode = variant['flow_params'].get('training_mode', 'current')
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
    # TODO support Docker
    exp_id = 'debug' if DEBUG else None
    exp_prefix = variant['env_name']
    run_name = variant['util_params'].get('wandb_run_name')
    if run_name:
        exp_prefix = '{}-{}'.format(exp_prefix, run_name)
    experiment_log_dir = setup_logger(exp_prefix, variant=variant, exp_id=exp_id, base_log_dir=variant['util_params']['base_log_dir'])

    # optionally save eval trajectories as pkl files
    if variant['algo_params']['dump_eval_paths']:
        pickle_dir = experiment_log_dir + '/eval_trajectories'
        pathlib.Path(pickle_dir).mkdir(parents=True, exist_ok=True)

    # optional Weights & Biases logging
    use_wandb = variant['util_params'].get('use_wandb', False)
    if use_wandb:
        _setup_wandb(variant, experiment_log_dir)

    # run the algorithm
    try:
        algorithm.train()
    finally:
        if use_wandb:
            import wandb
            wandb.finish()

def deep_update_dict(fr, to):
    ''' update dict of dicts with new values '''
    for k, v in fr.items():
        if type(v) is dict:
            if k not in to or type(to[k]) is not dict:
                to[k] = {}
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
    wandb_tags = variant['util_params'].get('wandb_tags', None)
    if isinstance(wandb_tags, str):
        try:
            wandb_tags = ast.literal_eval(wandb_tags)
        except (SyntaxError, ValueError):
            wandb_tags = [wandb_tags]
    wandb.init(
        project=variant['util_params'].get('wandb_project', 'pearl'),
        name=variant['util_params'].get(
            'wandb_run_name',
            os.path.basename(experiment_log_dir.rstrip('/')),
        ),
        group=variant['util_params'].get('wandb_group', None),
        tags=wandb_tags,
        config=variant,
        dir=experiment_log_dir,
    )
    simple_legacy_logging = variant['util_params'].get('wandb_simple_legacy_logging', False)
    if not simple_legacy_logging:
        wandb.define_metric('Number of env steps total')
        wandb.define_metric('Number of rollouts total')
        wandb.define_metric('Number of train steps total')
        wandb.define_metric('Epoch')
        wandb.define_metric('*', step_metric='Number of env steps total')
        if variant['util_params'].get('wandb_log_legacy_aliases', False):
            wandb.define_metric('env_steps')
            wandb.define_metric('epoch')
            wandb.define_metric('*', step_metric='env_steps')

    def _log(tabular_dict):
        metrics = _to_numeric(tabular_dict)
        if not metrics:
            return
        if variant['util_params'].get('wandb_log_legacy_aliases', False):
            aliased = dict(metrics)
            for key, value in metrics.items():
                alias = (
                    key.replace('/', '_')
                    .replace(' ', '_')
                    .replace('(', '')
                    .replace(')', '')
                )
                aliased.setdefault(alias, value)
            if 'Number of env steps total' in metrics:
                aliased['env_steps'] = metrics['Number of env steps total']
            if 'Epoch' in metrics:
                aliased['epoch'] = metrics['Epoch']
            wandb.log(aliased)
        else:
            step = metrics.get('Number of env steps total', None)
            if step is None:
                wandb.log(metrics)
            else:
                wandb.log(metrics, step=int(step))

    logger.add_tabular_callback(_log)

@click.command()
@click.argument('config', default=None)
@click.option('--gpu', default=0)
@click.option('--docker', is_flag=True, default=False)
@click.option('--debug', is_flag=True, default=False)
def main(config, gpu, docker, debug):

    variant = default_config
    if config:
        with open(os.path.join(config)) as f:
            exp_params = json.load(f)
        variant = deep_update_dict(exp_params, variant)
    variant['util_params']['gpu_id'] = gpu
    if debug:
        variant['util_params']['debug'] = True
        variant['util_params']['use_wandb'] = False

    experiment(variant)

if __name__ == "__main__":
    main()
