import os
import sys
import cv2
import numpy as np
#from bts_eval import compute_error
#from bts_eval import eval


def compute_errors(gt, pred):
    thresh = np.maximum((gt / pred), (pred / gt))
    d1 = (thresh < 1.25).mean()
    d2 = (thresh < 1.25 ** 2).mean()
    d3 = (thresh < 1.25 ** 3).mean()
    
    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())
    
    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())
    
    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)
    
    err = np.log(pred) - np.log(gt)
    silog = np.sqrt(np.mean(err ** 2) - np.mean(err) ** 2) * 100
    
    err = np.abs(np.log10(pred) - np.log10(gt))
    log10 = np.mean(err)
    
    return silog, log10, abs_rel, sq_rel, rmse, rmse_log, d1, d2, d3


def do_eval(num_samples, min_depth_eval, max_depth_eval):
#def eval(gt_depths, pred_depths, num_samples, missing_ids, min_depth_eval, max_depth_eval):
    # num_samples = get_num_lines(args.filenames_file)
    pred_depths_valid = []

    for t_id in range(num_samples):
        if t_id in missing_ids:
            continue

        pred_depths_valid.append(pred_depths[t_id])

    num_samples = num_samples - len(missing_ids)

    silog = np.zeros(num_samples, np.float32)
    log10 = np.zeros(num_samples, np.float32)
    rms = np.zeros(num_samples, np.float32)
    log_rms = np.zeros(num_samples, np.float32)
    abs_rel = np.zeros(num_samples, np.float32)
    sq_rel = np.zeros(num_samples, np.float32)
    d1 = np.zeros(num_samples, np.float32)
    d2 = np.zeros(num_samples, np.float32)
    d3 = np.zeros(num_samples, np.float32)

    for i in range(num_samples):
        gt_depth = gt_depths[i]
        pred_depth = pred_depths_valid[i]

        # if args.do_kb_crop:
        #     height, width = gt_depth.shape
        #     top_margin = int(height - 352)
        #     left_margin = int((width - 1216) / 2)
        #     pred_depth_uncropped = np.zeros((height, width), dtype=np.float32)
        #     pred_depth_uncropped[top_margin:top_margin + 352, left_margin:left_margin + 1216] = pred_depth
        #     pred_depth = pred_depth_uncropped

        # pred_depth[pred_depth < args.min_depth_eval] = args.min_depth_eval
        # pred_depth[pred_depth > args.max_depth_eval] = args.max_depth_eval
        # pred_depth[np.isinf(pred_depth)] = args.max_depth_eval
        # pred_depth[np.isnan(pred_depth)] = args.min_depth_eval

        pred_depth[pred_depth < min_depth_eval] = min_depth_eval
        pred_depth[pred_depth > max_depth_eval] = max_depth_eval
        pred_depth[np.isinf(pred_depth)] = max_depth_eval
        pred_depth[np.isnan(pred_depth)] = min_depth_eval

        valid_mask = np.logical_and(gt_depth > min_depth_eval, gt_depth < max_depth_eval)
        # valid_mask = np.logical_and(gt_depth > args.min_depth_eval, gt_depth < args.max_depth_eval)

        # if args.garg_crop or args.eigen_crop:
        if True:
            gt_height, gt_width = gt_depth.shape
            eval_mask = np.zeros(valid_mask.shape)

            # if args.garg_crop:
            #     eval_mask[int(0.40810811 * gt_height):int(0.99189189 * gt_height),
            #     int(0.03594771 * gt_width):int(0.96405229 * gt_width)] = 1
            #
            # elif args.eigen_crop:
                # if args.dataset == 'kitti':
                #     eval_mask[int(0.3324324 * gt_height):int(0.91351351 * gt_height),
                #     int(0.0359477 * gt_width):int(0.96405229 * gt_width)] = 1
                # else:
            eval_mask[45:471, 41:601] = 1

            valid_mask = np.logical_and(valid_mask, eval_mask)

        silog[i], log10[i], abs_rel[i], sq_rel[i], rms[i], log_rms[i], d1[i], d2[i], d3[i] = compute_errors(gt_depth[valid_mask], pred_depth[valid_mask])

    print("{:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}".format('silog', 'abs_rel', 'log10', 'rms', 'sq_rel', 'log_rms', 'd1', 'd2', 'd3'))
    print("{:7.4f}, {:7.4f}, {:7.3f}, {:7.3f}, {:7.3f}, {:7.3f}, {:7.3f}, {:7.3f}, {:7.3f}".format(silog.mean(), abs_rel.mean(), log10.mean(), rms.mean(), sq_rel.mean(), log_rms.mean(), d1.mean(), d2.mean(), d3.mean()))
    # return silog, log10, abs_rel, sq_rel, rms, log_rms, d1, d2, d3
    return abs_rel.mean()




def get_gt_file_name(line):
    #print(line.strip('\n'))
    line = line.strip('\n')
    fileName = line.split('/')[-1]
    #print("fileName", fileName)
    folderName = fileName.split('-')[0]
    #print("folderName", folderName)
    imageName = line.split('-')[-1]
    #print("imageName", imageName)
    imageSequenceNum = imageName.split('.')[0].split('_')[-1]
    gt_image_name = "sync_depth_" + imageSequenceNum + ".png"
    # print("gt", argv[2] + gt_image_name)
    return folderName+"/"+gt_image_name

def get_pred_depth(i):
    #tensorName = "iter_" + str(i) + "_attach_Mul_Mul_963_out0_4_out0_1_1_480_640.tensor"
    tensorName = "iter_" + str(i) + "_attach_Mul_Mul_588_out0_4_out0_1_1_480_640.tensor"
    #print('i', i)
    ##print(tensorName)
    # f = open(tensorName)
    # lines = f.readlines()

    tensor_depth_path = os.path.join(pred_output_path, tensorName)

    # file_name = 'numpy_data.txt'  # 定义数据文件
    data = np.loadtxt(tensor_depth_path, dtype='float32', delimiter=' ')  # 获取数据
    #print(data)  # 打印数据
    #print(len(data))

    arr = np.reshape(data, (480, 640), order='C')
    return  arr


if __name__ == '__main__':
    # test(args)
    # if len(sys.argv) < 3:
    #     print("usage: %s <h4_file> <train_test_split> <out_folder>" % sys.argv[0], file=sys.stderr)
        # sys.exit(-1)

# h4_file = h5py.File(sys.argv[1], "r")
# def test(args):

    missing_ids =[]
    gt_depths = []
    pred_depths = []


#    pred_output_path="./inference_fp32"
    pred_output_path="./int16/inference_int16"
    output_dataset_txt="./md2_dataset.txt"
    gt_path="/home/chengw/repo/nyu_depth_v2/official_splits/test"
    with open(output_dataset_txt,'r') as r:
#    with open(argv[1],'r') as r:
        # with open(sys.argv[1], 'r') as r:
        # with open(r'/home/vicoretek/Downloads/VOCdevkit/VOC2006/ImageSets/Main/val.txt','r') as r:
        lines = r.readlines()
        num_samples=len(lines)
        # line = r.readline().strip('\n')
        for i, line in enumerate(lines):
        # f=open(tensorName)
        # lines= f.readlines()
        # import numpy as np  # 导入numpy库
        # file_name = 'numpy_data.txt'  # 定义数据文件
        # data = np.loadtxt(tensorName, dtype='float32', delimiter='\n')  # 获取数据
        # print(data)  # 打印数据
#step1:
            gt_file_name=get_gt_file_name(line)
    #        print(gt_file_name)
            gt_depth_path = os.path.join(gt_path, gt_file_name)
    #        print(gt_depth_path)
#            sys.exit(-1)
            depth = cv2.imread(gt_depth_path, -1)
            if depth is None:
                print('Missing: %s ' % gt_depth_path)
                missing_ids.add(t_id)
                continue
                # if args.dataset == 'nyu':
            depth = depth.astype(np.float32) / 1000.0
                 # else:
                 #     depth = depth.astype(np.float32) / 256.0
            gt_depths.append(depth)

#step2: get pred_depths
            pred_depth=get_pred_depth(i)
            pred_depths.append(pred_depth)

        #step3:
        abs_rel=do_eval(num_samples,1e-3,10)
        #abs_rel=eval(gt_depths,pred_depths,num_samples,missing_ids,1e-3,10)
        print("{:7.4f}, ".format(abs_rel))
        # def eval(gt_depths, pred_depths, num_samples, missing_ids, min_depth_eval, max_depth_eval):

        # def eval(gt_depths, pred_depths, num_samples, missing_ids, min_depth_eval, max_depth_eval):

        # sys.exit(0)

# if len(gt_depths) == -1:
#     for t_id in range(num_test_samples):
#         gt_depth_path = os.path.join(args.gt_path, lines[t_id].split()[0])
#         depth = cv1.imread(gt_depth_path, -1)
#         if depth is None:
#             print('Missing: %s ' % gt_depth_path)
#             missing_ids.add(t_id)
#             continue
#
#         if args.dataset == 'nyu':
#             depth = depth.astype(np.float31) / 1000.0
#         else:
#             depth = depth.astype(np.float31) / 256.0
#
#         gt_depths.append(depth)
#

