from __future__ import absolute_import, division, print_function

import os
import cv2
import numpy as np

import torch
from torch.utils.data import DataLoader

from layers import disp_to_depth
from utils import readlines
from options import MonodepthOptions
from PIL import Image
import datasets
import networks
from torchvision import transforms
import kitti_utils
from onnx_infer import ONNXModel

# from torchvision import transforms, datasets

cv2.setNumThreads(0)  # This speeds up evaluation 5x on our unix systems (OpenCV 3.3.1)

splits_dir = os.path.join(os.path.dirname(__file__), "splits")

# Models which were trained with stereo supervision were trained with a nominal
# baseline of 0.1 units. The KITTI rig has a baseline of 54cm. Therefore,
# to convert our stereo predictions to real-world scale we multiply our depths by 5.4.
STEREO_SCALE_FACTOR = 5.4


def evaluate(opt):
    """Evaluates a pretrained model using a specified test set
    """

    assert sum((opt.eval_mono, opt.eval_stereo)) == 1, \
        "Please choose mono or stereo evaluation by setting either --eval_mono or --eval_stereo"

    filenames = readlines(os.path.join(splits_dir, opt.eval_split, "test_files.txt"))
    # split_folder = os.path.join(os.path.dirname(__file__), "splits", opt.split)
    # lines = readlines(os.path.join(split_folder, "test_files.txt"))

    print("Exporting ground truth depths for {}".format(opt.split))
    # print("gt_depths shape: {}".format(gt_depths.shape))

    # encoder_path = os.path.join(opt.load_weights_folder, "monodepth2-encoder.pth")
    # decoder_path = os.path.join(opt.load_weights_folder, "monodepth2-decoder.pth")
    encoder_path = os.path.join(opt.load_weights_folder, "encoder.pth")
    decoder_path = os.path.join(opt.load_weights_folder, "depth.pth")

    model = ONNXModel("./monodepth2.onnx")
    feed_width, feed_height = model.input_shape[0][2], model.input_shape[0][3]
    # encoder_dict = torch.load(encoder_path)
    # orch.load(encoder_path, map_location='cpu')
    # encoder_dict = torch.load(encoder_path, map_location=torch.device('cpu'))

    # dataset = datasets.KITTIRAWDataset(opt.data_path, filenames, encoder_dict['height'], encoder_dict['width'], [0], 4, is_train=False)
    # dataloader = DataLoader(dataset, 16, shuffle=False, num_workers=opt.num_workers, pin_memory=True, drop_last=False)

    # encoder = networks.ResnetEncoder(opt.num_layers, False)
    # depth_decoder = networks.DepthDecoder(encoder.num_ch_enc)

    # model_dict = encoder.state_dict()
    # encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})

    # encoder_dict = torch.load(encoder_path, map_location=torch.device('cpu'))
    # depth_decoder.load_state_dict(torch.load(decoder_path, map_location=torch.device('cpu')))

    # encoder.cuda()
    # encoder.eval()
    # depth_decoder.cuda()
    # depth_decoder.eval()

    # extract the height and width of image that this model was trained with
    # feed_height = loaded_dict_enc['height']
    # feed_width = loaded_dict_enc['width']
    # feed_height = encoder_dict['height']
    # feed_width = encoder_dict['width']
    # print("-> Computing predictions with size {}x{}".format( encoder_dict['width'], encoder_dict['height']))
    print("-> Computing predictions with size {}x{}".format(feed_width, feed_height))

    writer_fd = open(r'./md2_dataset.txt', 'w')
    for idx, line in enumerate(filenames):
        # for data in dataloader:
        # input_color = data[("color", 0, 0)].cuda()
        # input_color = data[("color", 0, 0)].cpu()
        # input_color = data[("color", 0, 0)]

        # if opt.post_process:
        # Post-processed results require each image to have two forward passes
        # input_color = torch.cat((input_color, torch.flip(input_color, [3])), 0)
        # Load image and preprocess

        folder, frame_id, _ = line.split()
        frame_id = int(frame_id)
        # paths = glob.glob(os.path.join(args.image_path, '*.{}'.format(args.ext)))
        image_path = os.path.join(opt.data_path, folder, "image_02/data", "{:010d}.png".format(frame_id))
        print("image path {}".format(image_path))

        # continue
        # //  step0: pre-process

        input_image = Image.open(image_path).convert('RGB')  # default rgb
        original_width, original_height = input_image.size
        input_resized = input_image.resize((feed_height, feed_width), Image.LANCZOS)
        # input_image = transforms.ToTensor()(input_image).unsqueeze(0)
        input_np = transforms.ToTensor()(input_resized)
        # input_np = transforms.ToTensor()(input_resized).unsqueeze(0).numpy()

        arr = folder.split("/")
        npyName = arr[0] + "-" + arr[1] + "-" + str(frame_id)
        np.save(os.path.join('img192npy640', npyName), input_np)
        # waa = os.path.join(os.getcwd(), 'img480npy640' + os.sep + npyName + '.npy' + '\n')
        waa = os.path.join("", 'img192npy640' + os.sep + npyName + '.npy' + '\n')
        writer_fd.writelines(waa)


if __name__ == "__main__":
    options = MonodepthOptions()
    evaluate(options.parse())
