"""Build one tall image per passage for checking images against labels by eye.

Each block has the line number and the label on top and the image below it, with
a red bar between lines. Images are rotated upright by their EXIF tag first.

Usage (from the repository root):
  python scripts/make_contact_sheets.py --real-dir data/handwriting-test
Output: <real-dir>/contact_sheets/P01.png ... P09.png
"""

import argparse
import json
import os

import matplotlib
from PIL import Image, ImageDraw, ImageFont, ImageOps

Image.MAX_IMAGE_PIXELS = None


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def default_font():
    # DejaVu Sans ships with matplotlib and covers Vietnamese letters
    return os.path.join(matplotlib.get_data_path(), "fonts", "ttf", "DejaVuSans.ttf")


def build_tile(record, real_dir, font, width):
    with Image.open(os.path.join(real_dir, record["image_path"])) as img:
        img = ImageOps.exif_transpose(img).convert("RGB")
        img = img.resize((width, max(1, int(img.height * width / img.width))))
    tile = Image.new("RGB", (width, 56 + img.height + 10), "white")
    ImageDraw.Draw(tile).text((8, 8), record["id"][5:] + "  " + record["gt_text"], fill="black", font=font)
    tile.paste(img, (0, 56))
    ImageDraw.Draw(tile).line([(0, tile.height - 5), (width, tile.height - 5)], fill="red", width=4)
    return tile


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--real-dir", required=True, help="folder with the images and real_dataset_index.jsonl")
    parser.add_argument("--index", default=None, help="default: <real-dir>/real_dataset_index.jsonl")
    parser.add_argument("--out-dir", default=None, help="default: <real-dir>/contact_sheets")
    parser.add_argument("--width", type=int, default=1400)
    parser.add_argument("--font", default=None, help="a TrueType font with Vietnamese letters")
    args = parser.parse_args()

    index_path = args.index or os.path.join(args.real_dir, "real_dataset_index.jsonl")
    out_dir = args.out_dir or os.path.join(args.real_dir, "contact_sheets")
    os.makedirs(out_dir, exist_ok=True)
    font = ImageFont.truetype(args.font or default_font(), 34)
    records = read_jsonl(index_path)
    for group in sorted({record["group_id"] for record in records}):
        tiles = [build_tile(record, args.real_dir, font, args.width) for record in records if record["group_id"] == group]
        sheet = Image.new("RGB", (args.width, sum(tile.height for tile in tiles)), "white")
        y = 0
        for tile in tiles:
            sheet.paste(tile, (0, y))
            y += tile.height
        path = os.path.join(out_dir, group + ".png")
        sheet.save(path)
        print(group, len(tiles), "lines ->", path, sheet.size)


if __name__ == "__main__":
    main()
