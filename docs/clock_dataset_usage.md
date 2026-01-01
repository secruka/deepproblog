# Clock dataset helper usage

## Status of the code
- The latest commit on this branch is `Add clock dataset helper for time-slot images`.
- No git remotes are configured in this workspace, so nothing has been pushed upstream yet.

## Dataset layout expected by the helper
```
clock_kaggle/
  train/
    1-00/*.jpg
    1-05/*.jpg
    ...
  test/
    1-00/*.jpg
    1-05/*.jpg
    ...
```
Each directory name (`H-MM`) encodes the hour and minute. With the default 5-minute interval, labels are assigned as `slot = hour * 12 + minute/5`, giving 288 classes for a full day.

## Minimal dry-run (no GPU required)
The following snippet builds a tiny dummy dataset in `/tmp` and runs a single `ClockDataset` + `ClockImages` fetch to verify the pipeline without training a full model:

```python
from pathlib import Path
import torch
from PIL import Image
from deepproblog.examples.Clock.dataset import ClockDataset, ClockImages

# 1) Create dummy data
root = Path("/tmp/clock_kaggle")
slot_dir = root / "train" / "1-05"
slot_dir.mkdir(parents=True, exist_ok=True)
img_path = slot_dir / "sample.jpg"
# generate a 32x32 RGB dummy image
Image.new("RGB", (32, 32), color=(128, 128, 128)).save(img_path)

# 2) Instantiate dataset and tensor source
clock_ds = ClockDataset(root=str(root), subset="train")
clock_imgs = ClockImages(root=str(root), subset="train")

# 3) Fetch one sample and its image tensor
q = clock_ds.to_query(0)
tensor = clock_imgs[q.query.args[0].args[1]]  # matches the Term("tensor", Term(subset, Constant(idx)))
print(q)            # Query structure
print(tensor.shape) # torch.Size([...])
```

If you see a printed `Query(clock_time(tensor(train,0), 13))` and a tensor shape like `torch.Size([3, 32, 32])`, the helper works end-to-end.

## How to integrate into training
1. Add the tensor source to your model:
   ```python
   model.add_tensor_source("train", ClockImages(root="clock_kaggle", subset="train"))
   ```
2. Create `ClockDataset` for each split and pass it to the trainer:
   ```python
   dataset = {
       "train": ClockDataset(root="clock_kaggle", subset="train"),
       "test": ClockDataset(root="clock_kaggle", subset="test"),
   }
   train_model(model, dataset["train"], 1, test_dataset=dataset["test"])
   ```
3. Adjust the predicate name or interval via the constructor arguments if needed.

## Notes for limited environments
- The dry-run above avoids GPUs and heavy dependencies; it only requires `Pillow` and `torch` CPU.
- To test loading your real dataset without full training, you can iterate a few items and inspect labels:
  ```python
  for i in range(3):
      print(clock_ds.samples[i])
  ```
- Full training will depend on your model architecture; start with a small batch size and a few iterations to validate that the data pipeline works.
```
