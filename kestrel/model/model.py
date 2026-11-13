import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from kestrel.config import Config
from kestrel.model.attention import GQAttention
from kestrel.model.expert import SwiGLUExpert
from kestrel.model.kv_cache import KVCache
from kestrel.model.moe import MoE
from kestrel.model.norm import RMSNorm
from kestrel.model.rope import RotaryEmbedding


class Block(nn.Module):
    """Pre-norm блок: h = x + attn(attn_norm(x)); y = h + moe(moe_norm(h))."""

    def __init__(self, cfg: Config, rotary: RotaryEmbedding):
        super().__init__()
        m = cfg.model
        self.attn_norm = RMSNorm(m.d_model, m.norm_eps)
        self.attn = GQAttention(m.d_model, m.n_heads, m.n_kv_heads, m.head_dim, m.norm_eps, rotary)
        self.moe_norm = RMSNorm(m.d_model, m.norm_eps)
        self.moe = MoE(m.d_model, m.d_ff, m.n_experts, m.top_k, cfg.moe.aux_alpha)

    def forward(self, x, layer_idx, cache=None):
        h = x + self.attn(self.attn_norm(x), layer_idx, cache)
        moe_out, aux = self.moe(self.moe_norm(h))
        return h + moe_out, aux


class Kestrel(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        m = cfg.model
        self.cfg = cfg
        self.embed = nn.Embedding(m.vocab_size, m.d_model)
        self.rotary = RotaryEmbedding(m.head_dim, m.rope_base, cfg.yarn.l_train, cfg.yarn.alpha, cfg.yarn.beta)
        self.layers = nn.ModuleList([Block(cfg, self.rotary) for _ in range(m.n_layers)])
        self.final_norm = RMSNorm(m.d_model, m.norm_eps)
        self.lm_head = nn.Linear(m.d_model, m.vocab_size, bias=False)  # не связан с embed

    def forward(self, input_ids, cache: KVCache | None = None, targets=None) -> dict:
        """input_ids: [B, T] -> {'logits': [B,T,V], 'aux_loss': сумма по слоям,
        'ce_loss' и 'loss' = CE + aux_loss (только если переданы targets)}."""
        x = self.embed(input_ids)
        aux_total = x.new_zeros(())
        for i, layer in enumerate(self.layers):
            x, aux = layer(x, i, cache)
            aux_total = aux_total + aux
        logits = self.lm_head(self.final_norm(x))
        out = {"logits": logits, "aux_loss": aux_total}
        if targets is not None:
            ce = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), targets.reshape(-1))
            out["ce_loss"] = ce
            out["loss"] = ce + aux_total
        return out

    def set_yarn_scale(self, scale: float) -> None:
        self.rotary.set_scale(scale)

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens: int, use_cache: bool = True):
        """Жадная генерация. Возвращает [B, T + max_new_tokens]."""
        ids = input_ids
        cache = KVCache(len(self.layers)) if use_cache else None
        inp = ids
        for _ in range(max_new_tokens):
            logits = self(inp, cache=cache)["logits"]
            nxt = logits[:, -1].argmax(dim=-1, keepdim=True)
            ids = torch.cat([ids, nxt], dim=1)
            inp = nxt if use_cache else ids
        return ids


def _init_weights(model: Kestrel, cfg: Config) -> None:
    std = cfg.model.init_std
    deep_std = std / math.sqrt(2 * cfg.model.n_layers)
    for module in model.modules():
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=std)
    for layer in model.layers:
        nn.init.normal_(layer.attn.o_proj.weight, mean=0.0, std=deep_std)
        for expert in layer.moe.experts:
            assert isinstance(expert, SwiGLUExpert)
            nn.init.normal_(expert.w_down.weight, mean=0.0, std=deep_std)


def build_model(cfg: Config, device="cpu") -> Kestrel:
    """Собирает модель. Инициализация идёт на CPU (результат не зависит от GPU), затем перенос на device.

    device="meta" строит модель без памяти -- чтобы посчитать параметры основного конфига.
    """
    if str(device) == "meta":
        with torch.device("meta"):
            return Kestrel(cfg)
    model = Kestrel(cfg)
    _init_weights(model, cfg)
    return model.to(device)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def save_checkpoint(model: nn.Module, path, meta: dict | None = None) -> None:
    """Сохраняет веса модели и метаданные: {"model": state_dict (fp32), "meta": {...}}."""
    state = {k: v.detach().float().cpu() if v.is_floating_point() else v.detach().cpu()
             for k, v in model.state_dict().items()}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save({"model": state, "meta": dict(meta or {})}, tmp)
    tmp.replace(path)


def load_checkpoint(model: Kestrel, path) -> dict:
    """Грузит веса (strict=True). Возвращает metadata из чекпоинта.

    При несовпадении ключей или форм бросает RuntimeError со списком missing/unexpected.
    Принимает и чистый state_dict (тогда metadata пустая).
    """
    obj = torch.load(path, map_location="cpu", weights_only=True)
    state = obj["model"] if isinstance(obj.get("model"), dict) else obj
    meta = obj.get("meta", {}) if isinstance(obj.get("meta", {}), dict) else {}
    nl = chr(10)
    try:
        result = model.load_state_dict(state, strict=False)
    except RuntimeError as e:
        raise RuntimeError(f"Чекпоинт {path} не подходит к модели (формы параметров):{nl}{e}") from e
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(
            f"Чекпоинт {path} не подходит к модели.{nl}"
            f"  нет в чекпоинте (missing): {result.missing_keys}{nl}"
            f"  лишние в чекпоинте (unexpected): {result.unexpected_keys}{nl}"
            "Проверьте, что вы не переименовали атрибуты модулей в файлах с TODO."
        )
    return meta
