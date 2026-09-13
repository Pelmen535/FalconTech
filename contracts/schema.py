"""
КОНТРАКТЫ ПРОЕКТА — машиночитаемая версия.

Это единственное место, где определены сущности, которыми обмениваются
модули. При расхождении с PROJECT.md истина здесь.

МЕНЯТЬ ТОЛЬКО ПО СОГЛАСИЮ ОБОИХ РАЗРАБОТЧИКОВ, с записью в docs/decisions.md.

Сквозные соглашения (действуют везде, без исключений):
  * время          — UTC, tz-aware. Наивных datetime в системе нет.
  * пути           — относительные от DATA_ROOT, POSIX-разделители ("/").
  * bbox           — (x1, y1, x2, y2), int, в координатах ПОЛНОГО КАДРА,
                     не кропа. Левый-верхний включая, правый-нижний исключая.
  * пусто          — всегда None. Никогда "" и никогда -1.
  * эмбеддинги     — np.float32, L2-нормированные, размерность EMBED_DIM.
  * идентификаторы — uuid4().hex (32 символа, без дефисов).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Literal

import numpy as np

# --------------------------------------------------------------------------
# Константы
# --------------------------------------------------------------------------

EMBED_DIM: int = 768
"""Размерность глобального эмбеддинга. Меняется только вместе с моделью."""

UV_SIZE: tuple[int, int] = (256, 256)
"""Разрешение UV-атласа (H, W). Одинаково для всех типов кузова."""

MIN_POSE_IOU: float = 0.8
"""Гейт качества позы (D-006). Ниже — наблюдение выбрасывается целиком."""

K_ANONYMITY_FLOOR: int = 5
"""Минимум кандидатов в выдаче без отдельной санкции (D-001)."""


class BodyType(str, Enum):
    SEDAN = "sedan"
    HATCHBACK = "hatchback"
    SUV = "suv"
    VAN = "van"
    TRUCK = "truck"
    BUS = "bus"
    OTHER = "other"


class ViewBin(str, Enum):
    """8 корзин ракурса. Азимут относительно носа машины, по часовой."""

    FRONT = "front"            # 337.5–22.5°
    FRONT_RIGHT = "front_right"
    RIGHT = "right"
    REAR_RIGHT = "rear_right"
    REAR = "rear"
    REAR_LEFT = "rear_left"
    LEFT = "left"
    FRONT_LEFT = "front_left"


class PlateSource(str, Enum):
    """Откуда взялся номер. Критично для честности метрик (D-002)."""

    OCR = "ocr"                # реально распознан
    SIMULATED = "simulated"    # подставлен из ground-truth в симуляции
    MANUAL = "manual"          # размечен руками
    NONE = "none"              # номера нет


# --------------------------------------------------------------------------
# Событие — одно наблюдение одной камерой
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Event:
    """Одно наблюдение автомобиля одной камерой в один момент времени.

    Владелец схемы: общая зона. Производит: src/ingest/.
    """

    event_id: str
    """uuid4().hex"""

    camera_id: str
    """Стабильный идентификатор камеры. Совпадает с ключом в configs/cameras.yaml."""

    ts: datetime
    """UTC, tz-aware. Момент КАДРА, не момент обработки."""

    crop_path: str
    """Путь к кропу машины относительно DATA_ROOT, POSIX-разделители."""

    bbox: tuple[int, int, int, int]
    """(x1, y1, x2, y2) в координатах полного кадра."""

    frame_path: str | None = None
    """Путь к полному кадру, если сохранён. None — не сохраняли."""

    track_id: str | None = None
    """Идентификатор трека внутри одной камеры. None — трекинг не применялся.
    ВНИМАНИЕ: уникален только в паре (camera_id, track_id)."""

    plate: str | None = None
    """Номер в верхнем регистре без пробелов и дефисов. None — не распознан."""

    plate_source: PlateSource = PlateSource.NONE
    """Откуда номер. При plate=None всегда NONE."""

    plate_confidence: float | None = None
    """Уверенность OCR в [0,1]. None, если plate is None."""

    def __post_init__(self) -> None:
        if self.ts.tzinfo is None:
            raise ValueError("Event.ts должен быть tz-aware (UTC)")
        if self.plate is None and self.plate_source is not PlateSource.NONE:
            raise ValueError("plate_source должен быть NONE при plate=None")
        x1, y1, x2, y2 = self.bbox
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"некорректный bbox: {self.bbox}")


# --------------------------------------------------------------------------
# Отпечаток — признаковое описание одного события
# --------------------------------------------------------------------------


@dataclass
class Attributes:
    """Структурированные атрибуты. Каждый — со своей уверенностью,
    потому что они попадают в объяснение и в исключение (D-001)."""

    body_type: BodyType
    body_type_conf: float

    color: str
    """Базовый цвет строкой: "silver", "black", ... Ночью часто ненадёжен."""
    color_conf: float

    make_model: str | None = None
    """"kia_rio", "lada_vesta". None — не определено."""
    make_model_conf: float | None = None

    has_roof_rails: bool | None = None
    has_tinting: bool | None = None
    """None = не видно с этого ракурса. Отличается от False = точно нет.
    Это различие критично для исключения: нельзя исключать по "не видно"."""


@dataclass
class UVPatch:
    """Частичная UV-развёртка одного наблюдения (D-004).

    Полной развёртки не бывает никогда — с одного ракурса видно меньше
    половины поверхности. Сравнение идёт только по M_A * M_B.
    """

    texture: np.ndarray
    """(H, W, 3) uint8. Мусор там, где mask == 0 — не читать без маски."""

    mask: np.ndarray
    """(H, W) bool. True — поверхность была видна и спроецирована."""

    quality: np.ndarray
    """(H, W) float32 в [0,1]. Угол обзора x разрешение x отсутствие пересвета.
    Используется как вес при накоплении в профиль."""

    body_type: BodyType
    """Какой канонический меш использован. Атласы разных типов несравнимы."""

    pose_iou: float
    """IoU силуэта меша с сегментацией. < MIN_POSE_IOU — выбросить (D-006)."""


@dataclass
class Fingerprint:
    """Признаковое описание одного Event.

    Владелец: src/reid/.
    """

    event_id: str

    embedding: np.ndarray
    """(EMBED_DIM,) float32, L2-нормирован."""

    attributes: Attributes

    view_bin: ViewBin

    uv: UVPatch | None = None
    """None, если UV-ветка выключена или наблюдение не прошло гейт позы."""

    lamp_embedding: np.ndarray | None = None
    """Ночная сигнатура ламп. None днём или если фары не видны."""


# --------------------------------------------------------------------------
# Профиль — накопление по нескольким наблюдениям
# --------------------------------------------------------------------------


@dataclass
class VehicleProfile:
    """Накопленный профиль одной машины. Это НЕ фотография, а поверхность,
    которая достраивается с каждым новым наблюдением (D-004)."""

    profile_id: str

    event_ids: list[str] = field(default_factory=list)
    """Все события, слитые в этот профиль."""

    plate: str | None = None
    """Номер, если хоть одно событие дало якорь. None — безномерный профиль."""

    embedding: np.ndarray | None = None
    """Усреднённый L2-нормированный эмбеддинг по событиям."""

    uv_texture: np.ndarray | None = None
    """Накопленный атлас: sum(T_i * Q_i) / sum(Q_i)."""

    uv_mask: np.ndarray | None = None
    """Объединение масок: union(M_i). Растёт с каждым наблюдением."""

    coverage: float = 0.0
    """Доля поверхности, покрытая хотя бы раз. uv_mask.mean()."""


# --------------------------------------------------------------------------
# Поиск
# --------------------------------------------------------------------------


@dataclass
class Evidence:
    """Одна единица свидетельства за или против совпадения.

    Существует ради объяснимости: оператор видит, ПОЧЕМУ кандидат здесь.
    """

    kind: Literal["attribute", "embedding", "uv_region", "lamp", "spatiotemporal"]

    name: str
    """Человекочитаемо: "левое заднее крыло", "тип кузова", "рисунок диска"."""

    bits: float
    """Вклад в информационное содержание. Может быть отрицательным (против)."""

    verdict: Literal["match", "mismatch", "not_comparable"]
    """not_comparable — зона не наблюдалась с обоих ракурсов. Это НЕ mismatch
    и оно НИКОГДА не должно вести к исключению."""


@dataclass
class Candidate:
    profile_id: str
    plate: str | None
    score: float
    """Калиброванная вероятность совпадения в [0,1]. НЕ косинус."""

    bits: float
    """Сколько бит информации набрано в пользу этого кандидата."""

    evidence: list[Evidence] = field(default_factory=list)
    excluded_by: str | None = None
    """Если заполнено — кандидат исключён, и здесь причина.
    Исключённые возвращаются в выдаче: показать, что именно отсеяно,
    полезнее, чем молча спрятать."""


@dataclass
class SearchResult:
    """Ответ поиска. Никогда не содержит одного ответа — всегда список
    с диагностикой того, насколько ему вообще можно верить."""

    query_event_id: str
    candidates: list[Candidate]

    total_bits: float
    """Информационная ёмкость запроса. Сколько бит удалось извлечь."""

    fleet_size: int
    """Размер галереи, относительно которой считались биты."""

    k_anonymity_ok: bool
    """False — кандидатов меньше K_ANONYMITY_FLOOR, выдача ограничена."""

    warnings: list[str] = field(default_factory=list)
    """Человекочитаемые предупреждения для оператора. Например:
    "стоковый автомобиль без отличительных признаков",
    "ракурсы не пересекаются, сравнение только по атрибутам"."""
