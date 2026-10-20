"""Shared exact crop decoding for selected October CPU/native encoders."""
from PIL import Image
from vreid.extract import open_crop


def crop_image(row):
    if '_crop' in row:return row['_crop'].convert('RGB')
    if row.get('bbox') is None:
        with Image.open(row['path']) as image:image.load();return image.convert('RGB')
    return open_crop(row['path'],row['bbox'],pad=.05,fast_decode=True,target=352).convert('RGB')


def restoration_path(crop,grid=4):
    base=Image.new('RGB',crop.size,(127,127,127));yield base.copy()
    for row in range(grid):
        for col in range(grid):
            box=(col*crop.width//grid,row*crop.height//grid,(col+1)*crop.width//grid,(row+1)*crop.height//grid)
            base.paste(crop.crop(box),box);yield base.copy()


def occlusion_regions(crop, grid=4):
    """Mask one region of the original crop at a time; no accumulated edits."""
    for row in range(grid):
        for col in range(grid):
            image=crop.copy()
            box=(col*crop.width//grid,row*crop.height//grid,
                 (col+1)*crop.width//grid,(row+1)*crop.height//grid)
            image.paste((127,127,127),box)
            yield image
