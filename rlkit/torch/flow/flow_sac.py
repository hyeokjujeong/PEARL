from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

import rlkit.torch.pytorch_util as ptu
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.torch.sac.sac import PEARLSoftActorCritic


class FlowPEARLSoftActorCritic(PEARLSoftActorCritic):
    """PEARL SAC for the flow method.

    `_take_step` is a full rewrite of the baseline (the baseline interleaves a
    Gaussian KL term and reads z_means/z_vars, neither of which exist here).
    Differences:
      - no Gaussian KL term — the flow posterior has no closed-form KL; the
        decoder reconstruction is the grounding signal instead.
      - the context encoder is trained ONLY by the decoder ELBO (reconstruction)
        loss. task_z is detached in every SAC loss, so the encoder gets no
        gradient from the critic (pure Option B — inference decoupled from RL).
      - a TransitionDecoder reconstruction loss grounds the latent c.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        lr = self.context_optimizer.param_groups[0]['lr']
        self.decoder_optimizer = optim.Adam(self.agent.decoder.parameters(), lr=lr)
        # overridden from flow_params by the launcher
        self.recon_weight = 1.0
        self.collapse_eps = 1e-4
        self.cfm_weight = 1.0       # weight of the CFM (flow-matching) loss
        self.cfm_warmup_steps = 0   # train steps of recon-only before CFM turns on
        # gate the next-state (dynamics) decoder head. True preserves the
        # original behaviour (reconstruct next_obs + reward); set False to
        # ground only through the reward head (overridden by the launcher).
        self.use_dynamics_decoder = True
        # paper-mode additions (defaults preserve original 'current' behaviour)
        # NOTE: name is `flow_training_mode` (not `training_mode`) because the
        # base PEARLSoftActorCritic.training_mode is a *method* that toggles
        # train()/eval() on all networks; shadowing it with a string would
        # break MetaRLAlgorithm.train(). The JSON config key stays
        # `flow_params.training_mode` for ergonomics.
        self.flow_training_mode = 'current'   # 'current' | 'paper' | 'paper+recon'
        self.prior_weight = 1.0          # alpha' in Eq. 7
        # `prior_flow` and `prior_optimizer` are attached by the launcher when
        # a learned prior is enabled; left None otherwise.
        if not hasattr(self, 'prior_flow'):
            self.prior_flow = None
        if not hasattr(self, 'prior_optimizer'):
            self.prior_optimizer = None
        # ---- bypass flag (hypothesis test, NOT paper-faithful) -------------
        # When True, the paper-mode step computes a SECOND grad-enabled forward
        # pass of the encoder and feeds it ONLY into the Q-loss path. This
        # gives the encoder an extra gradient signal from the critic, breaking
        # paper §3.2's "theta updated solely via L_flow" claim. Intended to
        # test the hypothesis that pure bootstrapped EM lacks task-grounding.
        # Policy/V/target paths still use the original no-grad c_hat.
        self.q_grad_to_encoder = False
        # Encoder gradient clipping. Default 10 keeps CFM-only (off-bypass)
        # behaviour unchanged (typical CFM grad < 10) but prevents the
        # Q->encoder explosion under bypass (observed spikes to ~8000).
        self.encoder_grad_clip = 10.0

    def get_epoch_snapshot(self, epoch):
        snapshot = super().get_epoch_snapshot(epoch)
        snapshot['decoder'] = self.agent.decoder.state_dict()
        if self.prior_flow is not None:
            snapshot['prior_flow'] = self.prior_flow.state_dict()
        return snapshot

    def _take_step(self, indices, context):
        # paper-mode dispatch: keep the original 'current' path untouched
        if self.flow_training_mode in ('paper', 'paper+recon'):
            return self._take_step_paper(indices, context)
        # data is (task, batch, feat)
        obs, actions, rewards, next_obs, terms = self.sample_sac(indices)

        # agent.forward stores inferred c in self.z and returns
        # task_z = c repeated per transition, shape (num_tasks*b, latent_dim)
        policy_outputs, task_z = self.agent(obs, context)
        new_actions, policy_mean, policy_log_std, log_pi = policy_outputs[:4]

        # flatten the task dimension
        t, b, _ = obs.size()
        obs = obs.view(t * b, -1)
        actions = actions.view(t * b, -1)
        next_obs = next_obs.view(t * b, -1)
        rewards_raw = rewards.view(t * b, -1)
        terms_flat = terms.view(t * b, -1)

        # ---- decoder ELBO (reconstruction) -----------------------------------
        # The ONLY thing that trains the flow encoder. task_z is NOT detached
        # here, so gradient flows decoder-loss -> task_z -> flow encoder.
        # NOTE: for point-robot the dynamics s'=s+a don't depend on the task,
        # so recon_loss_r (reward head) carries all the grounding signal.
        pred_next_obs, pred_reward = self.agent.decoder(obs, actions, task_z)
        recon_obs_loss = F.mse_loss(pred_next_obs, next_obs)
        recon_r_loss = F.mse_loss(pred_reward, rewards_raw)
        # `use_dynamics_decoder` gates the next-state (dynamics) head. When off,
        # only the reward head grounds c (reward-varying tasks like point-robot
        # carry no signal in the dynamics head anyway). recon_obs_loss is still
        # computed for logging. (next_obs head stays in the net; its loss is
        # just excluded from the encoder/decoder gradient.)
        recon_loss = recon_r_loss
        if self.use_dynamics_decoder:
            recon_loss = recon_loss + recon_obs_loss
        elbo_loss = self.recon_weight * recon_loss

        # ---- CFM (bootstrap-EM) -- trains v_theta as a real velocity field ---
        # target c1 = sg[inferred c]; regress the composed velocity field that
        # the ODE actually integrates toward the OT target velocity (c1 - c0).
        # decoder grounding (above) anchors c1, so the self-target is not free.
        encoder = self.agent.context_encoder
        c1 = self.agent.z.detach()
        cfm_loss = encoder.cfm_loss(encoder._subsampled_context, c1)
        cfm_w = self.cfm_weight if self._n_train_steps_total >= self.cfm_warmup_steps else 0.0
        encoder_total = elbo_loss + cfm_w * cfm_loss

        self.context_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()
        encoder_total.backward()
        enc_grad_norm = sum(
            p.grad.norm().item()
            for p in self.agent.context_encoder.parameters()
            if p.grad is not None
        )
        self.context_optimizer.step()
        self.decoder_optimizer.step()

        # ---- prior CFM (only if a learned v_phi is attached) -----------------
        # Trains v_phi on the marginal of the bootstrap targets c1, parallel to
        # what `_take_step_paper` does. Without this, an attached prior_flow in
        # `current` mode stays at random init and the (t-1)*s_phi term in the
        # fused ODE becomes noise -- strictly worse than the N(0,I) fallback.
        if self.prior_flow is not None and self.prior_optimizer is not None:
            prior_loss = self.prior_flow.cfm_loss(c1)
            self.prior_optimizer.zero_grad()
            (self.prior_weight * prior_loss).backward()
            self.prior_optimizer.step()
        else:
            prior_loss = torch.zeros((), device=ptu.device)

        # ---- SAC losses ------------------------------------------------------
        # task_z detached: the encoder receives no gradient from the critic.
        task_z = task_z.detach()

        q1_pred = self.qf1(obs, actions, task_z)
        q2_pred = self.qf2(obs, actions, task_z)
        v_pred = self.vf(obs, task_z)
        with torch.no_grad():
            target_v_values = self.target_vf(next_obs, task_z)

        self.qf1_optimizer.zero_grad()
        self.qf2_optimizer.zero_grad()
        rewards_flat = rewards_raw * self.reward_scale
        q_target = rewards_flat + (1. - terms_flat) * self.discount * target_v_values
        qf_loss = torch.mean((q1_pred - q_target) ** 2) + torch.mean((q2_pred - q_target) ** 2)
        qf_loss.backward()
        self.qf1_optimizer.step()
        self.qf2_optimizer.step()

        min_q_new_actions = self._min_q(obs, new_actions, task_z)

        v_target = min_q_new_actions - log_pi
        vf_loss = self.vf_criterion(v_pred, v_target.detach())
        self.vf_optimizer.zero_grad()
        vf_loss.backward()
        self.vf_optimizer.step()
        self._update_target_network()

        policy_loss = (log_pi - min_q_new_actions).mean()
        mean_reg_loss = self.policy_mean_reg_weight * (policy_mean ** 2).mean()
        std_reg_loss = self.policy_std_reg_weight * (policy_log_std ** 2).mean()
        pre_tanh_value = policy_outputs[-1]
        pre_activation_reg_loss = self.policy_pre_activation_weight * (
            (pre_tanh_value ** 2).sum(dim=1).mean()
        )
        policy_loss = policy_loss + mean_reg_loss + std_reg_loss + pre_activation_reg_loss
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # ---- eval statistics (computed once per training call) ---------------
        if self.eval_statistics is None:
            self.eval_statistics = OrderedDict()
            z = ptu.get_numpy(self.agent.z)
            c_variance = float(np.mean(np.var(z, axis=0)))
            self.eval_statistics['recon_loss'] = float(ptu.get_numpy(recon_loss))
            self.eval_statistics['recon_loss_r'] = float(ptu.get_numpy(recon_r_loss))
            self.eval_statistics['recon_loss_obs'] = float(ptu.get_numpy(recon_obs_loss))
            self.eval_statistics['cfm_loss'] = float(ptu.get_numpy(cfm_loss))
            self.eval_statistics['prior_cfm_loss'] = float(ptu.get_numpy(prior_loss))
            self.eval_statistics['c_norm'] = float(np.mean(np.linalg.norm(z, axis=-1)))
            self.eval_statistics['c_variance'] = c_variance
            self.eval_statistics['flow_encoder_grad_norm'] = enc_grad_norm
            self.eval_statistics['QF Loss'] = np.mean(ptu.get_numpy(qf_loss))
            self.eval_statistics['VF Loss'] = np.mean(ptu.get_numpy(vf_loss))
            self.eval_statistics['Policy Loss'] = np.mean(ptu.get_numpy(policy_loss))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q Predictions', ptu.get_numpy(q1_pred)))
            self.eval_statistics.update(create_stats_ordered_dict(
                'V Predictions', ptu.get_numpy(v_pred)))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Log Pis', ptu.get_numpy(log_pi)))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy mu', ptu.get_numpy(policy_mean)))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy log std', ptu.get_numpy(policy_log_std)))
            if c_variance < self.collapse_eps:
                print('ERROR: latent collapsed (c_variance={:.3e} < {:.1e})'.format(
                    c_variance, self.collapse_eps))

    # ---------------------------------------------------------------------
    # Paper-accurate bootstrapped EM training step (Sec 3.2, Eqs. 5-7).
    # Selected via flow_training_mode in {'paper', 'paper+recon'}; the original
    # _take_step is left intact for the 'current' mode.
    # ---------------------------------------------------------------------
    def _take_step_paper(self, indices, context):
        obs, actions, rewards, next_obs, terms = self.sample_sac(indices)
        t, b, _ = obs.size()
        obs_flat = obs.view(t * b, -1)
        actions_flat = actions.view(t * b, -1)
        next_obs_flat = next_obs.view(t * b, -1)
        rewards_raw = rewards.view(t * b, -1)
        terms_flat = terms.view(t * b, -1)

        encoder = self.agent.context_encoder

        # ---- E-step: stop-grad ODE pass -> pseudo-target c_hat ----------
        c_hat = encoder.e_step_sample(context).detach()      # (num_tasks, latent)
        self.agent.z = c_hat                                  # for log/diagnostics
        # repeat c_hat per transition in the batch; matches PEARLAgent.forward
        task_z = c_hat.unsqueeze(1).expand(t, b, -1).reshape(t * b, -1)

        # ---- (bypass) optional grad-enabled c for Q-loss path -----------
        # Paper spec: task_z (no grad) used everywhere. With q_grad_to_encoder
        # set True, we run encoder(context) AGAIN with autograd enabled and
        # feed THAT into the Q networks only. The resulting Q-loss backward
        # then accumulates gradient on the encoder, giving it a task-grounding
        # signal from the critic (NOT paper-faithful; hypothesis test).
        if self.q_grad_to_encoder:
            c_grad = encoder(context)                          # autograd on
            task_z_grad = c_grad.unsqueeze(1).expand(t, b, -1).reshape(t * b, -1)
        else:
            task_z_grad = task_z                               # paper spec: no grad

        # ---- run policy on (obs, c_hat) (always no-grad path) -----------
        in_ = torch.cat([obs_flat, task_z], dim=1)
        policy_outputs = self.agent.policy(in_, reparameterize=True, return_log_prob=True)
        new_actions, policy_mean, policy_log_std, log_pi = policy_outputs[:4]

        # ---- M-step: per-transition CFM (Eq. 6) on v_theta --------------
        sub_context = encoder._subsampled_context             # same trans. as c_hat
        cfm_loss = encoder.per_transition_cfm_loss(sub_context, c_hat)
        cfm_w = self.cfm_weight if self._n_train_steps_total >= self.cfm_warmup_steps else 0.0
        encoder_total = cfm_w * cfm_loss

        # NOTE: encoder optimizer step is DEFERRED to after the Q-loss backward
        # below, so the bypass's Q-grad can be combined with the CFM grad in a
        # single optimizer step. When q_grad_to_encoder=False this is identical
        # to the original (CFM-only) behaviour -- just reorganised.
        self.context_optimizer.zero_grad()

        # ---- M-step: prior CFM on the marginal of c_hat (L_prior) -------
        if self.prior_flow is not None and self.prior_optimizer is not None:
            prior_loss = self.prior_flow.cfm_loss(c_hat)
            self.prior_optimizer.zero_grad()
            (self.prior_weight * prior_loss).backward()
            self.prior_optimizer.step()
        else:
            prior_loss = torch.zeros((), device=ptu.device)

        # ---- (optional) auxiliary decoder loss in 'paper+recon' mode -----
        # task_z is detached here, so the decoder loss only updates the
        # decoder itself — the flow encoder/prior are unaffected.
        if self.flow_training_mode == 'paper+recon':
            pred_next_obs, pred_reward = self.agent.decoder(obs_flat, actions_flat, task_z)
            recon_obs_loss = F.mse_loss(pred_next_obs, next_obs_flat)
            recon_r_loss = F.mse_loss(pred_reward, rewards_raw)
            recon_loss = recon_r_loss
            if self.use_dynamics_decoder:
                recon_loss = recon_loss + recon_obs_loss
            self.decoder_optimizer.zero_grad()
            (self.recon_weight * recon_loss).backward()
            self.decoder_optimizer.step()
        else:
            recon_loss = torch.zeros((), device=ptu.device)
            recon_obs_loss = torch.zeros((), device=ptu.device)
            recon_r_loss = torch.zeros((), device=ptu.device)

        # ---- SAC losses -------------------------------------------------
        # Q networks see task_z_grad: identical to task_z when bypass is OFF
        # (paper spec); has live autograd to encoder when bypass is ON.
        # V / target use task_z (no grad) always -- the bypass only opens the
        # Q->encoder path, not the V/policy paths.
        q1_pred = self.qf1(obs_flat, actions_flat, task_z_grad)
        q2_pred = self.qf2(obs_flat, actions_flat, task_z_grad)
        v_pred = self.vf(obs_flat, task_z)
        with torch.no_grad():
            target_v_values = self.target_vf(next_obs_flat, task_z)

        self.qf1_optimizer.zero_grad()
        self.qf2_optimizer.zero_grad()
        rewards_flat = rewards_raw * self.reward_scale
        q_target = rewards_flat + (1. - terms_flat) * self.discount * target_v_values
        qf_loss = torch.mean((q1_pred - q_target) ** 2) + torch.mean((q2_pred - q_target) ** 2)
        # Combine encoder loss into the same backward when bypass is on, so the
        # encoder receives both CFM and Q gradients in one optimizer step.
        # When off, encoder_total still backprops only via CFM (task_z_grad is
        # task_z, no grad path to encoder from Q).
        (encoder_total + qf_loss).backward()
        # Clip encoder gradient: the Q->encoder path under bypass propagates
        # through the 5-step fused ODE, which is deep + ill-conditioned and
        # produces ~100-1000x larger / noisier gradients than the CFM path.
        # Empirically observed enc_grad spikes to 8000+ within a single step,
        # collapsing c_norm and c_variance permanently. A clip at 10 keeps the
        # signal but prevents the explosion. Off-bypass runs are unaffected
        # because their enc_grad is already < 10 (CFM only).
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), self.encoder_grad_clip)
        enc_grad_norm = sum(
            p.grad.norm().item()
            for p in encoder.parameters()
            if p.grad is not None
        )
        self.context_optimizer.step()       # encoder: CFM (+ Q if bypass)
        self.qf1_optimizer.step()
        self.qf2_optimizer.step()

        min_q_new_actions = self._min_q(obs_flat, new_actions, task_z)

        v_target = min_q_new_actions - log_pi
        vf_loss = self.vf_criterion(v_pred, v_target.detach())
        self.vf_optimizer.zero_grad()
        vf_loss.backward()
        self.vf_optimizer.step()
        self._update_target_network()

        policy_loss = (log_pi - min_q_new_actions).mean()
        mean_reg_loss = self.policy_mean_reg_weight * (policy_mean ** 2).mean()
        std_reg_loss = self.policy_std_reg_weight * (policy_log_std ** 2).mean()
        pre_tanh_value = policy_outputs[-1]
        pre_activation_reg_loss = self.policy_pre_activation_weight * (
            (pre_tanh_value ** 2).sum(dim=1).mean()
        )
        policy_loss = policy_loss + mean_reg_loss + std_reg_loss + pre_activation_reg_loss
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        if self.eval_statistics is None:
            self.eval_statistics = OrderedDict()
            z = ptu.get_numpy(self.agent.z)
            c_variance = float(np.mean(np.var(z, axis=0)))
            self.eval_statistics['recon_loss'] = float(ptu.get_numpy(recon_loss))
            self.eval_statistics['recon_loss_r'] = float(ptu.get_numpy(recon_r_loss))
            self.eval_statistics['recon_loss_obs'] = float(ptu.get_numpy(recon_obs_loss))
            self.eval_statistics['cfm_loss'] = float(ptu.get_numpy(cfm_loss))
            self.eval_statistics['prior_cfm_loss'] = float(ptu.get_numpy(prior_loss))
            self.eval_statistics['c_norm'] = float(np.mean(np.linalg.norm(z, axis=-1)))
            self.eval_statistics['c_variance'] = c_variance
            self.eval_statistics['flow_encoder_grad_norm'] = enc_grad_norm
            self.eval_statistics['QF Loss'] = np.mean(ptu.get_numpy(qf_loss))
            self.eval_statistics['VF Loss'] = np.mean(ptu.get_numpy(vf_loss))
            self.eval_statistics['Policy Loss'] = np.mean(ptu.get_numpy(policy_loss))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q Predictions', ptu.get_numpy(q1_pred)))
            self.eval_statistics.update(create_stats_ordered_dict(
                'V Predictions', ptu.get_numpy(v_pred)))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Log Pis', ptu.get_numpy(log_pi)))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy mu', ptu.get_numpy(policy_mean)))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy log std', ptu.get_numpy(policy_log_std)))
            if c_variance < self.collapse_eps:
                print('ERROR: latent collapsed (c_variance={:.3e} < {:.1e})'.format(
                    c_variance, self.collapse_eps))
