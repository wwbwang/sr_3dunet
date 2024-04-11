import torch
import numpy as np
import torch.nn.functional as F

'''
保证传入模型的图片是piece_mod_size的整数倍
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


"""
平滑操作，避免大图像边缘不一致
"""
def handle_bigtif(model, piece_size, overlap, img):
    '''
    img: (b, c, h, w, d)
    '''
    _, _, h, w, d = img.shape
    
    img = extend_block(img, piece_size, overlap)
    img_out = torch.zeros_like(img)
    
    _, _, h_now, w_now, d_now = img_out.shape
    # img_tmp = torch.zeros_like(img)
    
    for start_h in range(0, h, piece_size-overlap):
        end_h = start_h + piece_size
        
        for start_w in range(0, w, piece_size-overlap):
            end_w = start_w + piece_size
            
            for start_d in range(0, d, piece_size-overlap):
                end_d = start_d + piece_size
                
                img_tmp = model(img[:, :, start_h:end_h, start_w:end_w, start_d:end_d])
                
                h_cutleft = 0 if start_h==0 else overlap//2
                w_cutleft = 0 if start_w==0 else overlap//2
                d_cutleft = 0 if start_d==0 else overlap//2
                
                h_cutright = 0 if end_h==h_now else overlap//2
                w_cutright = 0 if end_w==w_now else overlap//2
                d_cutright = 0 if end_d==d_now else overlap//2
                
                img_out[:, :, start_h+h_cutleft:end_h-h_cutright, start_w+w_cutleft:end_w-w_cutright, start_d+d_cutleft:end_d-d_cutright] = img_tmp[
                                            :,:,0+h_cutleft:piece_size-h_cutright, 0+w_cutleft:piece_size-w_cutright, 0+d_cutleft:piece_size-d_cutright]
                
                if end_d==d_now:
                    break
            if end_w==w_now:
                break
        if end_h==h_now:
            break
            

                
    return img_out[:,:,:h,:w,:d]
    
    
                                    
                # if (end_h-start_h)%piece_size!=0 or (end_w-start_w)%piece_size!=0 or (end_d-start_d)%piece_size!=0:
                #     extend_img = extend_block(img[:, :, start_h:end_h, start_w:end_w, start_d:end_d], piece_mod_size)
                #     img_out[:, :, start_h:end_h, start_w:end_w, start_d:end_d] = model(extend_img)[:, :, :end_h-start_h, :end_w-start_w, :end_d-start_d]
                # else:
                #     img_out[:, :, start_h:end_h, start_w:end_w, start_d:end_d] = model(img[:, :, start_h:end_h, start_w:end_w, start_d:end_d])[:, :]