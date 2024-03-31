import torch

from sr_3dunet.utils.data_utils import extend_block

"""
需要修改: overlap, 不满16补齐
"""

def handle_bigtif(model, piece_size, overlap, step_size, img):
    '''
    img: (b, c, h, w, d)
    '''
    _, _, h, w, d = img.shape
    
    img_out = torch.zeros_like(img)
    
    overlap=0   # FIXME
    
    for start_h in range(0, h, piece_size):
        end_h = start_h + piece_size if(start_h + piece_size)<=h else h-1
        
        for start_w in range(0, w, piece_size):
            end_w = start_w + piece_size if(start_w + piece_size)<=w else w-1
            
            for start_d in range(0, d, piece_size):
                end_d = start_d + piece_size if(start_d + piece_size)<=d else d-1
                
                if (end_h-start_h)%piece_size!=0 or (end_w-start_w)%piece_size!=0 or (end_d-start_d)%piece_size!=0:
                    extend_img = extend_block(img[:, :, start_h:end_h, start_w:end_w, start_d:end_d], piece_size)
                    img_out[:, :, start_h:end_h, start_w:end_w, start_d:end_d] = model(extend_img)[:, :, :end_h-start_h, :end_w-start_w, :end_d-start_d]
                else:
                    img_out[:, :, start_h:end_h, start_w:end_w, start_d:end_d] = model(img[:, :, start_h:end_h, start_w:end_w, start_d:end_d])[:, :]
                
    return img_out
    