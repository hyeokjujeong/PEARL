from collections import OrderedDict

from rlkit.torch.sac.sac import PEARLSoftActorCritic


class FlowPEARLSoftActorCritic(PEARLSoftActorCritic):
    """PEARL SAC for the flow method.

    Stage 0: only ensures the decoder is saved in epoch snapshots (otherwise it
    is silently dropped — a crash-free failure that wastes an overnight run).
    The flow config sets ``use_information_bottleneck=False`` so the base
    ``_take_step`` skips the Gaussian KL plumbing and runs unchanged with the
    flow agent.

    Stage 1 overrides ``_take_step`` to add the decoder ELBO (reconstruction)
    loss and to train the flow encoder from it.
    """

    def get_epoch_snapshot(self, epoch):
        snapshot = super().get_epoch_snapshot(epoch)
        snapshot['decoder'] = self.agent.decoder.state_dict()
        return snapshot
