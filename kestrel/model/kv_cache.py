import torch


class KVCache:
    """Простейший KV-кэш: на каждый слой хранит k и v, дописывая новые токены через torch.cat."""

    def __init__(self, n_layers: int):
        self.k: list = [None] * n_layers
        self.v: list = [None] * n_layers

    def seq_len(self, layer_idx: int) -> int:
        """Сколько токенов уже лежит в кэше слоя."""
        k = self.k[layer_idx]
        return 0 if k is None else k.shape[2]

    def update(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor):
        """k, v: [B, Hkv, T_new, Dh]. Дописывает и возвращает весь префикс слоя (k, v)."""
        if self.k[layer_idx] is None:
            self.k[layer_idx], self.v[layer_idx] = k, v
        else:
            self.k[layer_idx] = torch.cat([self.k[layer_idx], k], dim=2)
            self.v[layer_idx] = torch.cat([self.v[layer_idx], v], dim=2)
        return self.k[layer_idx], self.v[layer_idx]

    def reset(self) -> None:
        self.k = [None] * len(self.k)
        self.v = [None] * len(self.v)
