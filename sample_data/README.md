# Sample Face Recognition Dataset

This directory contains a **fully synthetic** dataset that mirrors the on-disk
layout expected by the face-recognition pipeline:

```
sample_data/
  face_recognition_data/
    training_dataset/
      user_001/
      user_002/
      user_003/
```

Each user folder stores three 256×256 JPEG frames that were procedurally
generated with Pillow. The shapes and colours intentionally resemble stylised
avatars instead of real faces, so the repository never ships personal data.

The dataset is distributed purely for **testing and documentation** purposes.
It is safe to check into version control and can be regenerated with the helper
snippet below:

```python
from pathlib import Path
from PIL import Image, ImageDraw

base = Path("sample_data/face_recognition_data/training_dataset")
users = {
    "user_001": ("Cyan", (0, 160, 200)),
    "user_002": ("Magenta", (180, 60, 160)),
    "user_003": ("Amber", (210, 150, 60)),
}

for username, (_, color) in users.items():
    folder = base / username
    folder.mkdir(parents=True, exist_ok=True)
    for idx in range(3):
        image = Image.new("RGB", (256, 256), (15, 15, 20))
        draw = ImageDraw.Draw(image)
        draw.ellipse((68, 50, 188, 170), fill=color)
        draw.rectangle((70, 160, 190, 220), fill=tuple(min(c + 30, 255) for c in color))
        draw.text((40, 20), f"{username}\nFrame {idx + 1}", fill=(230, 230, 230))
        image.save(folder / f"{username}_frame_{idx + 1:02d}.jpg", format="JPEG", quality=95)
```

Feel free to extend this directory locally with additional synthetic subjects.
The reproducibility workflow automatically points the evaluation engine at this
folder, so production deployments remain unaffected.
