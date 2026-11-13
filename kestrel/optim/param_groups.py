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
    raise NotImplementedError("TODO: split_params")
