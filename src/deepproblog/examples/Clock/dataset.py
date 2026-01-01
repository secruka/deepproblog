from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Mapping, Sequence

import torch
import torchvision.transforms as transforms
from PIL import Image
from problog.logic import Constant, Term

from deepproblog.dataset import Dataset
from deepproblog.query import Query

# デフォルトの画像前処理（3チャネル正規化）
DEFAULT_TRANSFORM = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)
DEFAULT_EXTENSIONS = ("jpg", "jpeg", "png")


@dataclass(frozen=True)
class ClockSample:
    path: Path
    label: int
    hour: int
    minute: int


# 5分刻みなどの時間スロット名（例: "1-05"）を整数ラベルに変換
# 24時間×(60/interval_minutes) クラスを 0 始まりで付与します。
def time_to_slot_index(hour: int, minute: int, interval_minutes: int) -> int:
    if minute % interval_minutes != 0:
        raise ValueError(
            f"Minute {minute} is not divisible by interval {interval_minutes}."
        )
    slots_per_hour = 60 // interval_minutes
    return hour * slots_per_hour + (minute // interval_minutes)


def collect_clock_samples(
    root: Path,
    subset: str,
    *,
    interval_minutes: int,
    extensions: Sequence[str],
) -> List[ClockSample]:
    base = Path(root) / subset
    samples: List[ClockSample] = []
    if not base.exists():
        raise FileNotFoundError(f"Subset directory not found: {base}")

    for slot_dir in sorted(p for p in base.iterdir() if p.is_dir()):
        try:
            hour_str, minute_str = slot_dir.name.split("-")
            hour = int(hour_str)
            minute = int(minute_str)
            label = time_to_slot_index(hour, minute, interval_minutes)
        except Exception as exc:
            raise ValueError(
                f"Directory name '{slot_dir.name}' must look like 'H-MM' (e.g. 1-05)."
            ) from exc

        for ext in extensions:
            for img_path in sorted(slot_dir.glob(f"*.{ext}")):
                samples.append(ClockSample(img_path, label, hour, minute))

    if not samples:
        raise ValueError(f"No images found under {base} with extensions {extensions}.")
    return samples


class ClockImages(Mapping[Term, torch.Tensor]):
    """Term -> 画像テンソルのマッピング.

    モデル登録時に ``model.add_tensor_source("train", ClockImages(...))`` のように指定し、
    ``Term("tensor", Term(subset, Constant(index)))`` が張る実データを提供します。
    """

    def __init__(
        self,
        root: str,
        subset: str,
        *,
        transform=DEFAULT_TRANSFORM,
        interval_minutes: int = 5,
        extensions: Sequence[str] = DEFAULT_EXTENSIONS,
    ):
        self.root = Path(root)
        self.subset = subset
        self.transform = transform
        self.samples = collect_clock_samples(
            self.root, subset, interval_minutes=interval_minutes, extensions=extensions
        )

    def __getitem__(self, item: Term):
        index = int(item[0])
        sample = self.samples[index]
        with open(sample.path, "rb") as handle:
            image = Image.open(handle).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image

    def __iter__(self) -> Iterable:
        return iter(range(len(self)))

    def __len__(self) -> int:
        return len(self.samples)


class ClockDataset(Dataset):
    """DEEPPROBLOGで時計画像を扱うためのDataset.

    ディレクトリ構成例::

        clock_kaggle/
            train/
                1-00/*.jpg
                1-05/*.jpg
                ...
            test/
                1-00/*.jpg
                1-05/*.jpg
                ...

    ``1-05`` のようなフォルダ名から5分刻みのスロット番号（0〜287）を生成し、
    ``Query(Term("clock_time", image_term, Constant(slot)))`` を返します。
    """

    def __init__(
        self,
        root: str,
        subset: str,
        *,
        predicate: str = "clock_time",
        interval_minutes: int = 5,
        extensions: Sequence[str] = DEFAULT_EXTENSIONS,
    ):
        super().__init__()
        self.root = Path(root)
        self.subset = subset
        self.predicate = predicate
        self.interval_minutes = interval_minutes
        self.extensions = extensions
        self.samples = collect_clock_samples(
            self.root, subset, interval_minutes=interval_minutes, extensions=extensions
        )

    def __len__(self):
        return len(self.samples)

    def to_query(self, i: int) -> Query:
        sample = self.samples[i]
        image_term = Term("tensor", Term(self.subset, Constant(i)))
        label = Constant(sample.label)
        return Query(Term(self.predicate, image_term, label))


__all__ = [
    "ClockDataset",
    "ClockImages",
    "collect_clock_samples",
    "time_to_slot_index",
    "ClockSample",
]
