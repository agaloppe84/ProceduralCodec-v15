from __future__ import annotations
from PIL import Image
import numpy as np

def montage(u8_batch, cols: int) -> Image.Image:
    arr = np.asarray(u8_batch)
    B, _, H, W = arr.shape
    rows = (B + cols - 1) // cols
    canvas = np.zeros((rows * H, cols * W), dtype=np.uint8)
    for i in range(B):
        r, c = divmod(i, cols)
        canvas[r*H:(r+1)*H, c*W:(c+1)*W] = arr[i, 0]
    return Image.fromarray(canvas, mode="L")

def plot_rd(csv_path: str, out_png: str) -> None:
    import csv
    import matplotlib.pyplot as plt
    xs, ys = [], []
    with open(csv_path, encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                xs.append(float(row["bpp"]))
                ys.append(float(row["psnr"]))
            except (ValueError, KeyError, TypeError):
                pass
    plt.figure()
    plt.plot(xs, ys, marker="o")
    plt.xlabel("bpp")
    plt.ylabel("PSNR (dB)")
    plt.title("Courbe RD")
    plt.savefig(out_png, bbox_inches="tight")
