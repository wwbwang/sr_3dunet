import tifffile
import numpy as np
import os
from tqdm import tqdm
import math
import argparse

from sr_3dunet.utils.data_utils import  get_rotated_img, str2bool

def img_loader(img, start_x, start_y, start_z, size):
    return img[start_x:start_x+size, start_y:start_y+size, start_z:start_z+size]
   
def judge_img(img, percentiles, minmax, maxmin):
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

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=int, default=128)
    parser.add_argument('--percentiles_lower_bound', type=float, default=0.75)  # [0.75, 0.99] # lower_bound is insensitive
    parser.add_argument('--percentiles_upper_bound', type=float, default=0.99)
    parser.add_argument('--minmax', type=int, default=450) # After percentile clip, minimum pixel should be bigger than this value
    parser.add_argument('--maxmin', type=int, default=60000) # After percentile clip, maxmum pixel should be less than this 
    parser.add_argument('--input_folder', type=str, default="/share/home/wangwb/workspace/sr_3dunet/datasets/40X")
    parser.add_argument('--output_folder', type=str, default="/share/home/wangwb/workspace/sr_3dunet/datasets/40X")
    parser.add_argument('--input_tif_name ', type=str, default="40x20x030p26sp1.tif")
    parser.add_argument('--crop_size ', type=int, default=50)
    parser.add_argument('--rotated_flag', type=str2bool, default=False)
    
    args = parser.parse_args()
    args.stride = args.size//2
    args.percentiles = [args.percentiles_lower_bound, args.percentiles_upper_bound]

    args.output_front_folder = os.path.join(args.output_folder, args.input_tif_name + '_' + str(args.size)+"_newdatasets")
    args.output_rotated_front_folder = os.path.join(args.output_folder, args.input_tif_name + '_rotated' +str(args.size)+"_newdatasets")
    args.output_back_folder = os.path.join(args.output_folder, args.input_tif_name + '_back' +str(args.size)+"_newdatasets")

    aniso_dimension = -2    # do not change
    iso_dimension = -1      # do not change
    if aniso_dimension!=-2 or iso_dimension!=-1:
        raise RuntimeError('iso_dimension or aniso_dimension is ERROR')

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder, exist_ok=True)
    if not os.path.exists(args.output_front_folder):
        os.makedirs(args.output_front_folder, exist_ok=True)
    if not os.path.exists(args.output_rotated_front_folder):
        os.makedirs(args.output_rotated_front_folder, exist_ok=True)
    if not os.path.exists(args.output_back_folder):
        os.makedirs(args.output_back_folder, exist_ok=True)

    img_total = tifffile.imread(os.path.join(args.input_folder, args.input_tif_name))
    
    # Aligned on VISoR's axes, if your data's axes are arranged correctly, ignore this code    
    img_total = img_total.transpose(0,2,1)
    
    index_total = 0
    index_partial = 0

    x_floor = args.clip_size
    x_upper = img_total.shape[0]-args.clip_size

    y_floor = img_total.shape[1]
    y_upper = img_total.shape[1]

    z_floor = args.clip_size
    z_upper = img_total.shape[2]-args.clip_size

    len1 = math.ceil((x_upper-args.size+1-x_floor)/args.stride)
    len2 = math.ceil((y_upper-args.size+1-y_floor)/args.stride)
    len3 = math.ceil((z_upper-args.size+1-z_floor)/args.stride)
    pbar1 = tqdm(total=len1*len2*len3, unit='img', desc='create dataset')


    for start_x in range(x_floor, x_upper-args.size+1, args.stride):
        for start_y in range(y_floor, y_upper-args.size+1, args.stride):
            for start_z in range(z_floor, z_upper-args.size+1, args.stride):
                index_total += 1
                now_img = img_loader(img_total, start_x, start_y, start_z, args.size)
                now_name = args.input_tif_name+'_'+str(start_x)+'_'+str(start_y)+'_'+str(start_z)+'.tif'

                if judge_img(now_img, args.percentiles, args.minmax, args.maxmin):
                    # Save this sub_image as frontground
                    index_partial += 1
                    tifffile.imwrite(os.path.join(args.output_front_folder, now_name), now_img)
                    
                    if args.rotated_flag:
                        # for example, 256 256 256 --> 358 256 358
                        rotated_cube_img = get_rotated_img(now_img, aniso_dimension)
                        # 358 256 358 --> 174 256 174
                        rotated_cube_img = rotated_cube_img[rotated_cube_img.shape[0]//4*1+5:rotated_cube_img.shape[0]//4*3-5,
                                                            :,
                                                            rotated_cube_img.shape[2]//4*1+5:rotated_cube_img.shape[2]//4*3-5]
                        if judge_img(rotated_cube_img, args.percentiles, args.minmax, args.maxmin):
                            # save 174 256 174 cube as strengthen frontground
                            tifffile.imwrite(os.path.join(args.output_rotated_front_folder, now_name), rotated_cube_img)
                else:
                    # this sub_image is background, save for easy manual viewing
                    tifffile.imwrite(os.path.join(args.output_back_folder, now_name), now_img)
                
                pbar1.update(1)

    print('front_ground/back_ground: '+str(index_partial)+'/'+str(index_total))



