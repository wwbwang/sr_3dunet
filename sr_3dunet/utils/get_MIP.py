import tifffile
import os

from sr_3dunet.utils.data_utils import get_projection, get_rotated_projection

def get_and_save_MIP(img, path, name):
    proj0, proj1, proj2 = get_projection(img, -1)
    rotated_proj0, rotated_proj1 = get_rotated_projection(img, -2)
    os.makedirs(path, exist_ok=True)
    tifffile.imsave(os.path.join(path, name+"_proj0.tif"), proj0)
    tifffile.imsave(os.path.join(path, name+"_proj1.tif"), proj1)
    tifffile.imsave(os.path.join(path, name+"_proj2.tif"), proj2)
    tifffile.imsave(os.path.join(path, name+"_rotated_proj0.tif"), rotated_proj0)
    tifffile.imsave(os.path.join(path, name+"_rotated_proj1.tif"), rotated_proj1)
    
if __name__ == '__main__':
    pass
    