from empatches import EMPatches
import numpy as np
import tifffile as tiff
import argparse
import os
from tqdm import tqdm

def img_valid(img, percentiles=[0.75, 0.99], minmax=450, maxmin=60000):
    flattened_arr = np.sort(img.flatten())
    clip_low = int(percentiles[0] * len(flattened_arr))
    clip_high = int(percentiles[1] * len(flattened_arr))
    clipped_arr = np.clip(img, flattened_arr[clip_low], flattened_arr[clip_high-1])
    max_value = np.max(clipped_arr) 
    min_value = np.min(clipped_arr) 
    if max_value > minmax and min_value < maxmin:
        return True
    else:
        return False

def main():
    parser = argparse.ArgumentParser(description='gene data args')
    parser.add_argument('-input_dir', type=str)
    parser.add_argument('-output_dir', type=str)
    parser.add_argument('-not_visor', action='store_true')
    parser.add_argument('-fg_judge', action='store_true')
    parser.add_argument('-center_crop', type=float, default=None)
    parser.add_argument('-patch_size', type=int, default=128)
    parser.add_argument('-overlap', type=float, default=0.25)
    args = parser.parse_args()
    print(args)

    input_dir = args.input_dir
    output_dir = args.output_dir
    print(f'input_dir: {input_dir}')
    print(f'output_dir: {output_dir}')

    IMAGE = tiff.imread(input_dir)
    if not args.not_visor:
        IMAGE = np.transpose(IMAGE, (0,2,1))
        print('=== transpose applied ===')
    print(f'IMAGE:  shape:{IMAGE.shape}; dtype:{IMAGE.dtype}')
    if args.center_crop:
        print('=== center crop applied ===')
        rate = args.center_crop
        shape = np.asarray(IMAGE.shape)
        center = shape//2
        offset = center - shape*(rate/2)
        size = shape*rate
        ds,hs,ws = offset.astype(np.int32)
        de,he,we = (offset+size).astype(np.int32)
        IMAGE = IMAGE[ds:de, hs:he, ws:we]
    print(f'IMAGE:  shape:{IMAGE.shape}; dtype:{IMAGE.dtype}')

    emp = EMPatches()
    patches, indices = emp.extract_patches(IMAGE, patchsize=args.patch_size, overlap=args.overlap, stride=None, vox=True)
    print('=== patching applied ===')
    print(f'patches num: {len(patches)}')
    count = 0
    for i, img in tqdm(enumerate(patches)):
        if args.fg_judge and not img_valid(img):
            continue
        tiff.imwrite(os.path.join(output_dir, f'{str(i+1).zfill(4)}.tif'), img)
        count += 1
    print(f'valid num: {count}')
    print('=== saved ===')

if __name__ == '__main__':
    main()