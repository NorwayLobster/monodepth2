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
#import datasets
import networks
from torchvision import transforms
import kitti_utils
#from onnx_infer import ONNXModel
# from torchvision import transforms, datasets

cv2.setNumThreads(0)  # This speeds up evaluation 5x on our unix systems (OpenCV 3.3.1)


splits_dir = os.path.join(os.path.dirname(__file__), "splits")

# Models which were trained with stereo supervision were trained with a nominal
# baseline of 0.1 units. The KITTI rig has a baseline of 54cm. Therefore,
# to convert our stereo predictions to real-world scale we multiply our depths by 5.4.
STEREO_SCALE_FACTOR = 5.4

def get_gt_file_name(data_path, line):
    #print(line.strip('\n'))
    line = line.strip('.npy\n')
    fileName = line.split('/')[-1]
    #print("fileName", fileName)
    folder1Name = fileName.split('-')[0]
    folder2Name = fileName.split('-')[1]
    frame_id = fileName.split('-')[-1]
    folder = folder1Name + "/" + folder2Name
    #print("folderName", folderName)
    # imageName = line.split('-')[-1]
    #print("imageName", imageName)
    # folder, frame_id, _ = line.split('/')
    frame_id = int(frame_id)
    calib_dir = os.path.join(data_path, folder1Name)
    # calib_dir = os.path.join(data_path, folder.split("/")[0])
    velo_filename = os.path.join(data_path, folder, "velodyne_points/data", "{:010d}.bin".format(frame_id))
    print("calib_dir {}".format(calib_dir))
    print("velo_filename {}".format(velo_filename))

    # gt_image_name = "sync_depth_" + imageSequenceNum + ".png"
    # print("gt", argv[2] + gt_image_name)
    return calib_dir, velo_filename
    # return folderName+"/"+gt_image_name


def generate_depth_gt(data_path, line):
    calib_dir, velo_filename = get_gt_file_name(data_path, line)
    #        print(gt_file_name)
    # gt_depth_path = os.path.join(gt_path, gt_file_name)
    #        print(gt_depth_path)
    # // step2: get gt

    gt_depth = kitti_utils.generate_depth_map(calib_dir, velo_filename, 2, True)
    return gt_depth



def get_pred_depth(pred_output_path, i):
    #tensorName = "iter_" + str(i) + "_attach_Mul_Mul_963_out0_4_out0_1_1_480_640.tensor"
    #tensorName = "iter_" + str(i) + "_attach_Mul_Mul_588_out0_4_out0_1_1_480_640.tensor"
    tensorName = "iter_" + str(i) + "_attach_Sigmoid_Sigmoid_240_out0_0_out0_1_1_192_640.tensor"
    #print('i', i)
    ##print(tensorName)
    # f = open(tensorName)
    # lines = f.readlines()

    tensor_depth_path = os.path.join(pred_output_path, tensorName)
    print("tensor_filename {}".format(tensor_depth_path))


    # file_name = 'numpy_data.txt'  # 定义数据文件
    data = np.loadtxt(tensor_depth_path, dtype='float32', delimiter=' ')  # 获取数据
    #print(data)  # 打印数据
    #print(len(data))

    #arr = np.reshape(data, (480, 640), order='C')
    arr = np.reshape(data, (192, 640), order='C')
    return  arr



def compute_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25     ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


def batch_post_process_disparity(l_disp, r_disp):
    """Apply the disparity post-processing method as introduced in Monodepthv1
    """
    _, h, w = l_disp.shape
    m_disp = 0.5 * (l_disp + r_disp)
    l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = (1.0 - np.clip(20 * (l - 0.05), 0, 1))[None, ...]
    r_mask = l_mask[:, :, ::-1]
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp


def evaluate():
    """Evaluates a pretrained model using a specified test set
    """
    MIN_DEPTH = 1e-3
    MAX_DEPTH = 80

    # assert sum((opt.eval_mono, opt.eval_stereo)) == 1, "Please choose mono or stereo evaluation by setting either --eval_mono or --eval_stereo"

    # filenames = readlines(os.path.join(splits_dir, eval_split, "test_files.txt"))
    # split_folder = os.path.join(os.path.dirname(__file__), "splits", opt.split)
    # lines = readlines(os.path.join(split_folder, "test_files.txt"))

    # print("Exporting ground truth depths for {}".format(opt.split))
    # print("gt_depths shape: {}".format(gt_depths.shape))

    pred_depth_scale_factor = 1.0
    eval_split = "eigen"
    # pred_disps = []
    errors = []
    ratios = []
    pred_output_path = "./acuity_mono/u8/inference_perchannel"
    #pred_output_path = "./acuity_mono/int16/inference_int16"
    output_dataset_txt = "acuity_mono/u8/md2_dataset.txt"
    #gt_data_path = "./kitti_data"
    gt_data_path = "/home/chengw/repo/kitti_data"

    # gt_data_path = "/home/chengw/repo/nyu_depth_v2/official_splits/test"
    r = open(output_dataset_txt, 'r')
    lines = r.readlines()
    num_samples = len(lines)
    # line = r.readline().strip('\n')
    min_depth = 0.1
    max_depth = 100
    for i, line in enumerate(lines):
        # f=open(tensorName)
                # lines= f.readlines()
                # import numpy as np  # 导入numpy库
                # file_name = 'numpy_data.txt'  # 定义数据文件
                # data = np.loadtxt(tensorName, dtype='float32', delimiter='\n')  # 获取数据

                # print(data)  # 打印数据
                # step2: get pred_depths
                disp = get_pred_depth(pred_output_path, i)
                # outputs = depth_decoder(encoder(input_image))
                # disp = outputs[("disp", 0)]

                # outputs = model.forward(input_np)
                # disp = outputs[0]

                pred_disp, _ = disp_to_depth(disp, min_depth, max_depth)
                # pred_disp, _ = disp_to_depth(output[("disp", 0)], opt.min_depth, opt.max_depth)
                # pred_disp = pred_disp[:, 0].numpy()
                # pred_disp = pred_disp.cpu()[:, 0].numpy()
                pred_disp = pred_disp.squeeze()
                # pred_disp = pred_disp.numpy()

                # if opt.post_process:
                #     N = pred_disp.shape[0] // 2
                #     pred_disp = batch_post_process_disparity(pred_disp[:N], pred_disp[N:, :, ::-1])

                # pred_disps.append(pred_disp)
                # pred_depths.append(pred_depth)

        # pred_disps = np.concatenate(pred_disps)

    # else:
    #     # Load predictions from file
    #     print("-> Loading predictions from {}".format(opt.ext_disp_to_eval))
    #     pred_disps = np.load(opt.ext_disp_to_eval)
    #
    #     if opt.eval_eigen_to_benchmark:
    #         eigen_to_benchmark_ids = np.load(
    #             os.path.join(splits_dir, "benchmark", "eigen_to_benchmark_ids.npy"))
    #
    #         pred_disps = pred_disps[eigen_to_benchmark_ids]
    #
    # if opt.save_pred_disps:
    #     output_path = os.path.join(
    #         opt.load_weights_folder, "disps_{}_split.npy".format(opt.eval_split))
    #     print("-> Saving predicted disparities to ", output_path)
    #     np.save(output_path, pred_disps)
    #
    # if opt.no_eval:
    #     print("-> Evaluation disabled. Done.")
    #     quit()
    #
    # elif opt.eval_split == 'benchmark':
    #     save_dir = os.path.join(opt.load_weights_folder, "benchmark_predictions")
    #     print("-> Saving out benchmark predictions to {}".format(save_dir))
    #     if not os.path.exists(save_dir):
    #         os.makedirs(save_dir)
    #
    #     for idx in range(len(pred_disps)):
    #         disp_resized = cv2.resize(pred_disps[idx], (1216, 352))
    #         depth = STEREO_SCALE_FACTOR / disp_resized
    #         depth = np.clip(depth, 0, 80)
    #         depth = np.uint16(depth * 256)
    #         save_path = os.path.join(save_dir, "{:010d}.png".format(idx))
    #         cv2.imwrite(save_path, depth)
    #
    #     print("-> No ground truth is available for the KITTI benchmark, so not evaluating. Done.")
    #     quit()

    # gt_path = os.path.join(splits_dir, opt.eval_split, "gt_depths.npz")
    # gt_depths = np.load(gt_path, fix_imports=True, encoding='latin1')["data"]

    # print("-> Evaluating")

    # if opt.eval_stereo:
    #     print("   Stereo evaluation - "
    #           "disabling median scaling, scaling by {}".format(STEREO_SCALE_FACTOR))
    #     opt.disable_median_scaling = True
    #     opt.pred_depth_scale_factor = STEREO_SCALE_FACTOR
    # else:
    #     print("   Mono evaluation - using median scaling")

    # errors = []
    # ratios = []

    # for i in range(pred_disps.shape[0]):


                # gt_depth_path = os.path.join("./kitti_data_gt", folder, "proj_depth", "groundtruth", "image_02", "{:010d}.png".format(frame_id))
                # print("gt path {}".format(gt_depth_path))
                # img = Image.open(gt_depth_path)
                # gt_depth = np.asarray(img, dtype='float32') / 255
                # gt_depth = gt_depths[i]
                    # step1:

                # gt_depth = generate_depth_gt(gt_depth_path)
                gt_depth = generate_depth_gt(gt_data_path,line)

                gt_height, gt_width = gt_depth.shape[:2]

                # pred_disp = pred_disps[i]
                pred_disp = cv2.resize(pred_disp, (gt_width, gt_height))
                pred_depth = 1 / pred_disp







                #//step3: post-process:
                # 1. crop&mask,
                # 2. pred_depth_scale_factor & median_scaling
                # 3. remote min & max
                # if opt.eval_split == "eigen":
                mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)

                crop = np.array([0.40810811 * gt_height, 0.99189189 * gt_height,
                                 0.03594771 * gt_width,  0.96405229 * gt_width]).astype(np.int32)
                crop_mask = np.zeros(mask.shape)
                crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
                mask = np.logical_and(mask, crop_mask)

                # else:
                #     mask = gt_depth > 0

                pred_depth = pred_depth[mask]
                gt_depth = gt_depth[mask]

                pred_depth *= pred_depth_scale_factor
                if True:
                    # if not opt.disable_median_scaling:
                    ratio = np.median(gt_depth) / np.median(pred_depth)
                    ratios.append(ratio)
                    pred_depth *= ratio

                pred_depth[pred_depth < MIN_DEPTH] = MIN_DEPTH
                pred_depth[pred_depth > MAX_DEPTH] = MAX_DEPTH

                errors.append(compute_errors(gt_depth, pred_depth))

    # if not opt.disable_median_scaling:
    print("-----------------------------------------starting evaluation----------------------------------------")
    if True:
        ratios = np.array(ratios)
        med = np.median(ratios)
        print(" Scaling ratios | med: {:0.3f} | std: {:0.3f}".format(med, np.std(ratios / med)))

    #//step4: metric
    mean_errors = np.array(errors).mean(0)

    print("\n  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
    print(("&{: 8.3f}  " * 7).format(*mean_errors.tolist()) + "\\\\")
    print("\n-> Done!")


if __name__ == "__main__":
    # options = MonodepthOptions()
    evaluate()
    # evaluate(options.parse())
