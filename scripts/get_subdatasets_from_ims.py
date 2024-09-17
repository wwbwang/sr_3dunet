import tifffile
import numpy as np
import h5py
import os
from tqdm import tqdm
import math
import argparse

from sr_3dunet.utils.data_utils import get_rotated_img

def img_loader(img, start_x, start_y, start_z, size):
    return img[start_x:start_x+size, start_y:start_y+size, start_z:start_z+size]
 
def judge_img(img, percentiles, minmax):
    flattened_arr = np.sort(img.flatten())
    clip_low = int(percentiles[0] * len(flattened_arr))
    clip_high = int(percentiles[1] * len(flattened_arr))
    clipped_arr = np.clip(img, flattened_arr[clip_low], flattened_arr[clip_high])

    max_value = np.max(clipped_arr) 
    if max_value > minmax:
        return True
    else:
        return False

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=int, default=128)
    parser.add_argument('--percentiles_lower_bound', type=float, default=0)  # [0.75, 0.99] # lower_bound is insensitive
    parser.add_argument('--percentiles_upper_bound', type=float, default=0.9999)
    parser.add_argument('--minmax', type=int, default=400) # After percentile clip, minimum pixel should be bigger than this value
    parser.add_argument('--maxmin', type=int, default=60000) # After percentile clip, maxmum pixel should be less than this 
    parser.add_argument('--input_folder', type=str, default="/share/data/VISoR_Reconstruction/SIAT_ION/LiuCiRong/20230910_CJ004/CJ4-1um-ROI1")
    parser.add_argument('--output_folder', type=str, default="/share/home/wangwb/workspace/sr_3dunet/datasets/NISSL")
    parser.add_argument('--input_ims_name ', type=str, default="CJ4ROI1.ims")
    parser.add_argument("--rotated_flag", action= "store_true")
    
    parser.add_argument('--x_floor', type=int, default=0)
    parser.add_argument('--x_upper', type=int, default=8700)
    parser.add_argument('--y_floor', type=int, default=0)
    parser.add_argument('--y_upper', type=int, default=8400)
    parser.add_argument('--z_floor', type=int, default=0)
    parser.add_argument('--z_upper', type=int, default=900)
    parser.add_argument('--channel', type=int, default=1)
    
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

    h5 = h5py.File(os.path.join(args.input_folder, args.input_ims_name), 'r')
    img_total = h5['DataSet']['ResolutionLevel 0']['TimePoint 0']['Channel '+str(args.channel)]['Data']

    index_total = 0
    index_partial = 0
    
    x_floor = args.x_floor
    x_upper = args.x_upper

    y_floor = args.y_floor
    y_upper = args.y_upper

    z_floor = args.z_floor
    z_upper = args.z_upper

    len1 = math.ceil((args.x_upper-args.size+1-args.x_floor)/args.stride)
    len2 = math.ceil((args.y_upper-args.size+1-args.y_floor)/args.stride)
    len3 = math.ceil((args.z_upper-args.size+1-args.z_floor)/args.stride)
    pbar1 = tqdm(total=len1*len2*len3, unit='img', desc='create dataset')

    for start_x in range(x_floor, x_upper-args.size+1, args.stride):
        for start_y in range(y_floor, y_upper-args.size+1, args.stride):
            for start_z in range(z_floor, z_upper-args.size+1, args.stride):
                index_total += 1
                now_img = img_loader(img_total, start_z, start_y, start_x, args.size)
                # Aligned on VISoR's axes, if your data's axes are arranged correctly, ignore this code    
                now_img = now_img.transpose(2,1,0)
                now_name = args.input_ims_name+'_'+str(start_x)+'_'+str(start_y)+'_'+str(start_z)+'.tif'

                if judge_img(now_img, args.percentiles, args.minmax):
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
                        if judge_img(rotated_cube_img, args.percentiles, args.minmax):
                            # save 174 256 174 cube as strengthen frontground
                            tifffile.imwrite(os.path.join(args.output_rotated_front_folder, now_name), rotated_cube_img)
                else:
                    # this sub_image is background, save for easy manual viewing
                    tifffile.imwrite(os.path.join(args.output_back_folder, now_name), now_img)
                
                pbar1.update(1)

    print('front_ground/back_ground: '+str(index_partial)+'/'+str(index_total))


