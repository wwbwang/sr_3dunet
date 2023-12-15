import argparse
import cv2
import glob
import numpy as np
import os
import psutil
import queue
import threading
import time
import torch
import sys
import tifffile
from os import path as osp
from tqdm import tqdm

from sr_3dunet.utils.inference_base import get_base_argument_parser, get_inference_model
from sr_3dunet.utils.video_util import frames2video
from sr_3dunet.utils.data_utils import preprocess, postprocess
from basicsr.data.transforms import mod_crop
from basicsr.utils.img_util import img2tensor, tensor2img

class IOConsumer(threading.Thread):
    """Since IO time can take up a significant portion of the total inference time,
    so we use multi thread to write frames individually.
    """

    def __init__(self, args: argparse.Namespace, que, qid):
        super().__init__()
        self._queue = que
        self.qid = qid
        self.args = args

    def run(self):
        while True:
            msg = self._queue.get()
            if isinstance(msg, str) and msg == 'quit':
                break

            output = msg['output']
            imgname = msg['imgname']
            out_img = tensor2img(output.squeeze(0))
            if self.args.outscale != self.args.netscale:
                h, w = out_img.shape[:2]
                out_img = cv2.resize(
                    out_img, (int(
                        w * self.args.outscale / self.args.netscale), int(h * self.args.outscale / self.args.netscale)),
                    interpolation=cv2.INTER_LANCZOS4)
            cv2.imwrite(imgname, out_img)

        # print(f'IO for worker {self.qid} is done.')


@torch.no_grad()
def main():
    parser = get_base_argument_parser()
    parser.add_argument('--num_io_consumer', type=int, default=3, help='number of IO consumer')
    parser.add_argument('--model_path', type=str, help='model_path')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_inference_model(args, device)
    print("Model size: {:.5f}M".format(sum(p.numel() for p in model.parameters())*4/1048576))

    # prepare output dir
    os.makedirs(args.output, exist_ok=True)

    img_path_list = os.listdir(args.input)

    pbar1 = tqdm(total=len(img_path_list), unit='tif_img', desc='inference')

    que = queue.Queue()
    consumers = [IOConsumer(args, que, f'IO_{i}') for i in range(args.num_io_consumer)]
    for consumer in consumers:
        consumer.start()

    num_imgs = len(img_path_list)       # 17
    for img_path in img_path_list:
        img = tifffile.imread(os.path.join(args.input, img_path))        
        img, min_value, max_value = preprocess(img)
        img = img.astype(np.float32)[None, None, ]
        img = torch.from_numpy(img).to(device)     # to float32
        out = model(img)
        out = postprocess(out.cpu().numpy(), min_value, max_value)
        tifffile.imwrite(os.path.join(args.output, img_path), out)
        pbar1.update(1)

    for _ in range(args.num_io_consumer):
        que.put('quit')
    for consumer in consumers:
        consumer.join()


if __name__ == '__main__':
    main()
