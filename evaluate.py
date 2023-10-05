import argparse
import logging
import time
import re
import os

import numpy as np
import torch.utils.data

from hardware.device import get_device, delay_time, jacquard, cornell, test
from inference.post_process import post_process_output
from utils.data import get_dataset
from utils.dataset_processing import evaluation, grasp
from utils.visualisation.plot import save_results

from pthflops import count_ops

logging.basicConfig(level=logging.INFO)


def getfilelist(path, strn='iou'):
    names=[]
    for dirpath, dirnames, filenames in os.walk(path):
        for filepath in filenames:
            model_name=filepath
            str1 = re.compile(strn+'''(.*?).pt''')
            match_obj = re.findall(str1,model_name)
            if match_obj:
                names.append(path +'\\'+ model_name)
    return names


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate networks')

    # Network
    parser.add_argument('--network', metavar='N', type=str, nargs='+', default=getfilelist(r'logs\230910_1549_all_in_one_dropout0.5'),
                        help='Path to saved networks to evaluate')
    parser.add_argument('--input-size', type=int, default=224,
                        help='Input image size for the network')

    # Dataset
    parser.add_argument('--dataset', type=str, default='cornell',
                        help='Dataset Name ("cornell" or "jaquard")')
    parser.add_argument('--dataset-path', type=str, default="D:\\1_PRP\Cornell",
                        help='Path to dataset')
    parser.add_argument('--use-depth', type=int, default=1,
                        help='Use Depth image for evaluation (1/0)')
    parser.add_argument('--use-rgb', type=int, default=1,
                        help='Use RGB image for evaluation (1/0)')
    parser.add_argument('--augment', action='store_true',
                        help='Whether data augmentation should be applied')
    parser.add_argument('--split', type=float, default=0.9,
                        help='Fraction of data for training (remainder is validation)')
    parser.add_argument('--ds-shuffle', action='store_true', default=False,  # IW / OW
                        help='Shuffle the dataset')
    parser.add_argument('--ds-rotate', type=float, default=0.5,
                        help='Shift the start point of the dataset to use a different test/train split')
    parser.add_argument('--num-workers', type=int, default=2,
                        help='Dataset workers')

    # Evaluation
    parser.add_argument('--n-grasps', type=int, default=1,  # 调节生成抓取框数量
                        help='Number of grasps to consider per image')
    parser.add_argument('--iou-threshold', type=float, default=0.25,
                        help='Threshold for IOU matching')
    parser.add_argument('--iou-eval', action='store_true', default= True,
                        help='Compute success based on IoU metric.')
    parser.add_argument('--jacquard-output', action='store_true',
                        help='Jacquard-dataset style output')

    # Misc.
    parser.add_argument('--vis', action='store_true',
                        help='Visualise the network output')
    parser.add_argument('--cpu', dest='force_cpu', action='store_true', default=False,
                        help='Force code to run in CPU mode')
    parser.add_argument('--random-seed', type=int, default=125,
                        help='Random seed for numpy')

    args = parser.parse_args()

    if args.jacquard_output and args.dataset != 'jacquard':
        raise ValueError('--jacquard-output can only be used with the --dataset jacquard option.')
    if args.jacquard_output and args.augment:
        raise ValueError('--jacquard-output can not be used with data augmentation.')

    return args


if __name__ == '__main__':
    args = parse_args()

    # Get the compute device
    device = get_device(args.force_cpu)

    file_handler = logging.FileHandler(os.path.join(os.path.dirname(args.network[0]),'evaluate_results.txt'))
    logging.getLogger().addHandler(file_handler)

    # Load Dataset
    logging.info('Loading {} Dataset...'.format(args.dataset.title()))
    Dataset = get_dataset(args.dataset)
    args.ds_shuffle = test()
    test_dataset = Dataset(args.dataset_path,
                           output_size=args.input_size,
                           ds_rotate=args.ds_rotate,  # 训练集测试集分割
                           random_rotate=False,
                           random_zoom=False,
                           include_depth=args.use_depth,
                           include_rgb=args.use_rgb)

    indices = list(range(test_dataset.length))
    split = int(np.floor(args.split * test_dataset.length))

    if args.ds_shuffle:
        np.random.seed(args.random_seed)
        np.random.shuffle(indices)
    val_indices = indices[split:]
    val_sampler = torch.utils.data.sampler.SubsetRandomSampler(val_indices)
    logging.info('Validation size: {}'.format(len(val_indices)))

    test_data = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        num_workers=args.num_workers,
        sampler=val_sampler
    )
    logging.info('Done')

    for network in args.network:
        logging.info('\nEvaluating model {}'.format(network))

        # Load Network
        net = torch.load(network)

        results = {'correct': 0, 'failed': 0}

        if args.jacquard_output:
            jo_fn = network + '_jacquard_output.txt'
            with open(jo_fn, 'w') as f:
                pass

        with torch.no_grad():
            start_time = time.time()
            for idx, (x, y, didx, rot, zoom) in enumerate(test_data):
                xc = x.to(device)
                yc = [yi.to(device) for yi in y]
                lossd = net.compute_loss(xc, yc)  # 包含loss值数据和网络输出图像的集合

                # 将网络输出的三张1channel的图像经过高斯模糊等处理后所得
                q_img, ang_img, width_img = post_process_output(lossd['pred']['pos'], lossd['pred']['cos'],
                                                                lossd['pred']['sin'], lossd['pred']['width'])

                test_data.dataset.get_gtbb(didx, rot, zoom)

                if args.iou_eval:
                    s = evaluation.calculate_iou_match(q_img, ang_img, test_data.dataset.get_gtbb(didx, rot, zoom),
                                                       no_grasps=args.n_grasps,
                                                       grasp_width=width_img,
                                                       threshold=args.iou_threshold
                                                       )
                    if s:
                        results['correct'] += 1
                    else:
                        results['failed'] += 1

                if args.jacquard_output:
                    grasps = grasp.detect_grasps(q_img, ang_img, width_img=width_img, no_grasps=1)
                    with open(jo_fn, 'a') as f:
                        for g in grasps:
                            f.write(test_data.dataset.get_jname(didx) + '\n')
                            f.write(g.to_jacquard(scale=1024 / 300) + '\n')

                if args.vis:
                    save_results(
                        rgb_img=test_data.dataset.get_rgb(didx, rot, zoom, normalise=False),
                        depth_img=test_data.dataset.get_depth(didx, rot, zoom),
                        grasp_q_img=q_img,
                        grasp_angle_img=ang_img,
                        no_grasps=args.n_grasps,
                        grasp_width_img=width_img
                    )

        delay_time = delay_time() if callable(delay_time) else 0
        avg_time = (time.time() - start_time - delay_time) / len(test_data)
        logging.info('Average evaluation time per image: {}ms'.format(avg_time * 1000))
        if args.dataset.title() == 'Cornell':
            results = cornell(results)

        if args.dataset.title() == 'Jacquard':
            results = jacquard(results)

        if args.iou_eval:
            logging.info('IOU Results: %d/%d = %f' % (results['correct'],
                                                      results['correct'] + results['failed'],
                                                      results['correct'] / (results['correct'] + results['failed'])))

        if args.jacquard_output:
            logging.info('Jacquard output saved to {}'.format(jo_fn))

        del net
        torch.cuda.empty_cache()
