import torch
import numpy as np
import os
import torch.nn.functional as F
from sr_3dunet.utils.data_utils import preprocess, postprocess, get_rotated_img, get_anti_rotated_img
from sr_3dunet.archs.unet_3d_generator_arch import UNet_3d_Generator

def get_inference_model(args, device) -> UNet_3d_Generator:
    """return an on device model with eval mode"""
    # set up model
    model = UNet_3d_Generator(in_channels=1, out_channels=1, features=args.features, norm_type=None, dim=3)

    model_path = args.model_path
    assert os.path.isfile(model_path), \
        f'{model_path} does not exist, please make sure you successfully download the pretrained models ' \
        f'and put them into the weights folder'

    # load checkpoint
    loadnet = torch.load(model_path)
    # print(loadnet.keys())
    model.load_state_dict(loadnet['params'], strict=True)
    # model.load_state_dict(loadnet, strict=True)
    model.eval()
    model = model.to(device)

    return model

def remove_outer_layer(img, remove_size):
    height, width, depth = img.shape
    removed_matrix = img[remove_size:height-remove_size, remove_size:width-remove_size, remove_size:depth-remove_size]
    return removed_matrix

def get_inference_model_debug(args, device) -> UNet_3d_Generator:
    """return an on device model with eval mode"""
    # set up model
    model = UNet_3d_Generator(in_channels=1, out_channels=1, features=[64, 128, 256], norm_type=None, dim=3)
    model_back = UNet_3d_Generator(in_channels=1, out_channels=1, features=[64, 128, 256], norm_type=None, dim=3)

    model_path = args.model_path
    model_back_path = args.model_back_path
    assert os.path.isfile(model_path), \
        f'{model_path} does not exist, please make sure you successfully download the pretrained models ' \
        f'and put them into the weights folder'

    # load checkpoint
    loadnet = torch.load(model_path)
    model.load_state_dict(loadnet['params'], strict=True)
    model.eval()
    model = model.to(device)
    
    loadnet = torch.load(model_back_path)
    model_back.load_state_dict(loadnet['params'], strict=True)
    model_back.eval()
    model_back = model_back.to(device)

    return model, model_back

'''
Ensure the images input to the model can be processed in multiples of the piece_size after calculating the given overlap.
'''  
def extend_block(img, piece_size, overlap, dim=3):
    def extend_block_(img):
        if dim==3:
            h, w, d = img.shape
            
            new_h = piece_size if h<=piece_size else h
            if (new_h-overlap)%(piece_size-overlap) != 0:
                new_h = ((new_h-overlap)//(piece_size-overlap)+1)*(piece_size-overlap)+overlap
            pad_h = new_h-h
            
            new_w = piece_size if w<=piece_size else w
            if (new_w-overlap)%(piece_size-overlap) != 0:
                new_w = ((new_w-overlap)//(piece_size-overlap)+1)*(piece_size-overlap)+overlap
            pad_w = new_w-w
            
            new_d = piece_size if d<=piece_size else d
            if (new_d-overlap)%(piece_size-overlap) != 0:
                new_d = ((new_d-overlap)//(piece_size-overlap)+1)*(piece_size-overlap)+overlap
            pad_d = new_d-d
            
            padded_img = np.pad(img, ((0, pad_h), (0, pad_w), (0, pad_d)), mode='constant') if isinstance(img, np.ndarray)\
                else F.pad(img, (0, pad_d, 0, pad_w, 0, pad_h), mode='constant') 
                
        elif dim==2:
            h, w = img.shape
            
            new_h = piece_size if new_h<=piece_size else h
            if (new_h-overlap)%(piece_size-overlap) != 0:
                new_h = ((new_h-overlap)//(piece_size-overlap)+1)*(piece_size-overlap)+overlap
            pad_h = new_h-h
            
            new_w = piece_size if new_w<=piece_size else w
            if (new_w-overlap)%(piece_size-overlap) != 0:
                new_w = ((new_w-overlap)//(piece_size-overlap)+1)*(piece_size-overlap)+overlap
            pad_w = new_w-w
            
            padded_img = np.pad(img, ((0, pad_h), (0, pad_w)), mode='constant') if isinstance(img, np.ndarray)\
                else F.pad(img, (0, pad_w, 0, pad_h), mode='constant')
        
        return padded_img
                
    if img.ndim > 3:
        return extend_block_(img[0,0])[None, None]
    else:
        return extend_block_(img)

def handle_smalltif(model, piece_size, overlap, rotated_flag, percentiles, dataset_mean, device, img):
    '''
    rawimg: (h, w, d)
    '''
    h, w, d = img.shape
    origin_shape = img.shape
    img, min_value, max_value = preprocess(img, percentiles, dataset_mean)
    if rotated_flag:
        img = get_rotated_img(img)
    img = extend_block(img, piece_size, overlap)
    img = img.astype(np.float32)[None, None,]
    img = torch.from_numpy(img).to(device)     # to float32
    img_out = model(img).cpu().numpy()[0,0]
    if rotated_flag:
        img_out = get_anti_rotated_img(img_out, origin_shape)
    postprocess(img_out, min_value, max_value, dataset_mean)
    return img_out[:h,:w,:d]
    
def handle_bigtif(model, piece_size, overlap, rotated_flag, percentiles, dataset_mean, device, img):
    '''
    img: (h, w, d)
    '''
    h, w, d = img.shape
    origin_shape = img.shape
    img_out = np.zeros_like(img)
    
    h_now, w_now, d_now = img_out.shape
    
    for start_h in range(0, h, piece_size-overlap):
        end_h = start_h + piece_size
        
        for start_w in range(0, w, piece_size-overlap):
            end_w = start_w + piece_size
            
            for start_d in range(0, d, piece_size-overlap):
                end_d = start_d + piece_size
                
                h_cutleft = 0 if start_h==0 else overlap//2
                w_cutleft = 0 if start_w==0 else overlap//2
                d_cutleft = 0 if start_d==0 else overlap//2
                h_cutright = 0 if end_h==h_now else overlap//2
                w_cutright = 0 if end_w==w_now else overlap//2
                d_cutright = 0 if end_d==d_now else overlap//2
                
                # img_tmp = img[:, :, start_h:end_h, start_w:end_w, start_d:end_d]
                img_tmp = img[start_h:end_h, start_w:end_w, start_d:end_d]
                img_tmp, min_value, max_value = preprocess(img_tmp, percentiles, dataset_mean)
                if rotated_flag:
                    img_tmp = get_rotated_img(img_tmp, device)
                img_tmp = extend_block(img_tmp, piece_size, overlap)
                img_tmp = torch.from_numpy(img_tmp)[None,None].to(device)
                img_out = model(img_tmp)[0, 0].cpu().numpy()
                if rotated_flag:
                    img_out = get_anti_rotated_img(img_out, origin_shape)
                postprocess(img_out, min_value, max_value, dataset_mean)
                
                img_out[start_h+h_cutleft:end_h-h_cutright, start_w+w_cutleft:end_w-w_cutright, start_d+d_cutleft:end_d-d_cutright] = img_out[
                                        0+h_cutleft:piece_size-h_cutright, 0+w_cutleft:piece_size-w_cutright, 0+d_cutleft:piece_size-d_cutright]
                    
                if end_d==d_now:
                    break
            if end_w==w_now:
                break
        if end_h==h_now:
            break
    
    return img_out[:h,:w,:d]