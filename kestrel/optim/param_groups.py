from torch import nn

from kestrel.model.model import Kestrel
from kestrel.model.norm import RMSNorm


def split_params(model: Kestrel) -> dict[str, list[nn.Parameter]]:
    """Разбивает ВСЕ параметры модели на три группы: 'matrix', 'vector_decay' и 'no_decay'.

    Состав групп -- таблица 3 в TECH_REPORT.md (раздел 4.1); заметки по реализации -- приложение A.5.
    Возвращает словарь ровно с этими тремя ключами; значения -- списки параметров. Группы не
    пересекаются и вместе покрывают model.parameters(); если какой-то параметр не попал ни в одну
    группу, бросает ValueError.
    """
    params_dict = {"matrix": [], "vector_decay": [], "no_decay": []}
    modules = dict(model.named_modules())

    for name, p in model.named_parameters():
        parent_name = name.rsplit(".", 1)[0]
        parent = modules.get(parent_name)

        if isinstance(parent, (RMSNorm, nn.Embedding)):
            params_dict["no_decay"].append(p)
        elif name == "lm_head.weight" or parent_name.endswith(".router"):
            params_dict["vector_decay"].append(p)
        elif isinstance(parent, nn.Linear):
            params_dict["matrix"].append(p)
        else:
            raise ValueError(f"не классифицировал {name!r}")

    return params_dict
