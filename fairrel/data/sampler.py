"""Group-balanced batch sampler.

Draws an approximately equal number of samples per demographic subgroup in
every batch (about 10-12 per subgroup with batch size 64 and G = 6), so the
subgroup moments used by the GCWD and SLMA losses are well defined.  Small
subgroups are cycled with reshuffling, i.e. sampled with repetition across
batches within an epoch.
"""

import torch
from torch.utils.data import Sampler


class GroupBalancedBatchSampler(Sampler):
    def __init__(self, group_labels, batch_size: int, seed: int = 0):
        self.groups = torch.as_tensor(group_labels)
        self.batch_size = batch_size
        self.group_ids = sorted(set(self.groups.tolist()))
        if batch_size < 2 * len(self.group_ids):
            raise ValueError("batch size too small for group-balanced sampling")
        self.indices = {g: (self.groups == g).nonzero(as_tuple=True)[0] for g in self.group_ids}
        self.num_batches = max(1, len(self.groups) // batch_size)
        self.generator = torch.Generator()
        self.generator.manual_seed(seed)
        self.epoch = 0

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(int(self.generator.initial_seed()) + self.epoch)
        self.epoch += 1

        pools = {gid: idx[torch.randperm(len(idx), generator=g)].tolist()
                 for gid, idx in self.indices.items()}
        cursors = {gid: 0 for gid in self.group_ids}

        def draw(gid, k):
            out = []
            for _ in range(k):
                if cursors[gid] >= len(pools[gid]):
                    idx = self.indices[gid]
                    pools[gid] = idx[torch.randperm(len(idx), generator=g)].tolist()
                    cursors[gid] = 0
                out.append(pools[gid][cursors[gid]])
                cursors[gid] += 1
            return out

        n_groups = len(self.group_ids)
        base = self.batch_size // n_groups
        remainder = self.batch_size - base * n_groups
        for b in range(self.num_batches):
            batch = []
            # Rotate which groups receive an extra sample so per-group counts
            # stay within +-1 of each other across the epoch.
            extra = {self.group_ids[(b + i) % n_groups] for i in range(remainder)}
            for gid in self.group_ids:
                batch.extend(draw(gid, base + (1 if gid in extra else 0)))
            perm = torch.randperm(len(batch), generator=g).tolist()
            yield [batch[i] for i in perm]
