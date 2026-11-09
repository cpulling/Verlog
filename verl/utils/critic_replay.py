"""The offline dataset of generate_critic_dataset.py as critic batches: rows laid out like an agent
loop batch (prompt left-padded, response right-padded), with the stored Monte-Carlo return as the
target of every response token. At step_gamma = step_lam = 1 that is exactly the PPO target, and it
stays valid for as long as the row is kept. Groups are sampled equally, however large each is."""
import numpy as np
import torch

from verl import DataProto
from verl.utils.model import compute_position_id_with_mask

KEYS = ["input_ids", "responses", "response_mask", "attention_mask", "position_ids", "returns"]


class CriticDataset:
    def __init__(self, path, pad_token_id, prompt_length, response_length, seed=0):
        self.rows = torch.load(path, weights_only=False)
        self.pad, self.prompt_length, self.response_length = pad_token_id, prompt_length, response_length
        self.rng = np.random.default_rng(seed)
        holdout = np.array(self.rows["holdout"], dtype=bool)
        group = np.array(self.rows["group"])
        self.groups = sorted(set(group.tolist()))
        self.train = {g: np.flatnonzero((group == g) & ~holdout) for g in self.groups}
        self.held = {g: np.flatnonzero((group == g) & holdout) for g in self.groups}
        self.queue = {g: self.rng.permutation(self.train[g]) for g in self.groups}

    def _next(self, g, n):
        got = []
        while n > 0:
            if len(self.queue[g]) == 0:
                self.queue[g] = self.rng.permutation(self.train[g])
            got.append(self.queue[g][:n])
            self.queue[g] = self.queue[g][n:]
            n -= len(got[-1])
        return np.concatenate(got)

    def sample(self, n):
        """n training rows (n even), the same number from every group, in random order."""
        per, extra = divmod(n, len(self.groups))
        idx = np.concatenate([self._next(g, per + (i < extra)) for i, g in enumerate(self.groups)])
        return self.batch(self.rng.permutation(idx))

    def batch(self, idx):
        n = len(idx)
        p, r = self.prompt_length, self.response_length
        prompts = torch.full((n, p), self.pad, dtype=torch.long)
        responses = torch.full((n, r), self.pad, dtype=torch.long)
        prompt_mask, response_mask = torch.zeros((n, p), dtype=torch.long), torch.zeros((n, r), dtype=torch.long)
        for i, k in enumerate(idx):
            a, b = self.rows["prompt_ids"][k][-p:], self.rows["response_ids"][k][:r]
            prompts[i, p - len(a):] = torch.tensor(a)
            prompt_mask[i, p - len(a):] = 1
            responses[i, :len(b)] = torch.tensor(b)
            response_mask[i, :len(b)] = 1
        attention_mask = torch.cat([prompt_mask, response_mask], dim=1)
        returns = torch.tensor([self.rows["ret"][k] for k in idx], dtype=torch.float32).unsqueeze(1).expand(-1, r).clone()
        return DataProto.from_dict(tensors={
            "input_ids": torch.cat([prompts, responses], dim=1), "responses": responses, "response_mask": response_mask,
            "attention_mask": attention_mask, "position_ids": compute_position_id_with_mask(attention_mask), "returns": returns})

    def holdout_batches(self):
        """{group: batch} of the held-out rows (an even number of each)."""
        return {g: self.batch(i[:len(i) - len(i) % 2]) for g, i in self.held.items() if len(i) >= 2}


def explained_variance(values, returns):
    v, r = values.float(), returns.float()
    return (1.0 - (r - v).var() / r.var().clamp(min=1e-8)).item()
