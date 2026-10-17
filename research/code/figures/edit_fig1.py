"""Edit the user's Figure 1 (new_version_by_cp/figure_1.png): replace the evaluation-box text so the cohorts are named
(internal test, CPAISD, SOOP) instead of A, B, C; remove the stray mark inside module 5; crop the empty bottom band."""
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

P = Path(r"E:\CoU Projects\NeuroScan\NeuroScan_MultiDataset\paper")
SRC = P / "new_version_by_cp" / "figure_1.png"
OUT = P / "figures" / "v11_fig1_modules_cp.png"
FONT = r"C:\Windows\Fonts\NotoSans-Regular.ttf"
S = 4  # supersampling factor

im = Image.open(SRC).convert("RGB")
d = ImageDraw.Draw(im)
d.rectangle((1122, 378, 1540, 482), fill="white")   # evaluation-box body, right of the icon, inside the borders
d.rectangle((945, 570, 955, 598), fill="white")     # stray mark inside module 5

lines = ["One final scoring", "Internal test \u2022 CPAISD \u2022 SOOP", "Class-weighted argmax: (1, 0.574, 0.574)"]
ft = ImageFont.truetype(FONT, 20 * S)
x0, y_centres = 1138, (400, 430, 460)
for text, yc in zip(lines, y_centres):
    b = ft.getbbox(text)
    w, h = b[2] - b[0], b[3] - b[1]
    layer = Image.new("L", (w + 8 * S, 30 * S), 0)
    asc = ft.getbbox("Hx")
    ImageDraw.Draw(layer).text((4 * S - b[0], 15 * S - (asc[1] + asc[3]) // 2), text, font=ft, fill=255,
                               stroke_width=1, stroke_fill=255)
    layer = layer.resize((layer.width // S, layer.height // S), Image.LANCZOS)
    region = im.crop((x0 - 4, yc - 15, x0 - 4 + layer.width, yc - 15 + layer.height))
    arr = np.asarray(region, float) * (1 - np.asarray(layer, float)[..., None] / 255) + 25 * (np.asarray(layer, float)[..., None] / 255)
    im.paste(Image.fromarray(arr.clip(0, 255).astype(np.uint8)), (x0 - 4, yc - 15))

a = np.asarray(im.convert("L"))
rows = np.where((a < 200).any(1))[0]
top_margin = rows.min()
im = im.crop((0, 0, im.width, rows.max() + top_margin + 1))
im.save(OUT, dpi=(300, 300))
im.crop((980, 315, 1560, 495)).resize((1160, 360), Image.LANCZOS).save(Path(__file__).parent / "f1_edit_zoom.png")
print(OUT, im.size)
