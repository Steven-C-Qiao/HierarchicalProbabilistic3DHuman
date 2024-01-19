import os
import torch
import torchvision
import numpy as np
import argparse
import pandas as pd
import torch.multiprocessing as mp
import traceback
import functools
import queue
import time
import re

from pathlib import Path

from tqdm import tqdm

from models.poseMF_shapeGaussian_net import PoseMFShapeGaussianNet
from models.smpl_official import SMPL
from models.pose2D_hrnet import PoseHighResolutionNet
from models.canny_edge_detector import CannyEdgeDetector

from configs.poseMF_shapeGaussian_net_config import get_poseMF_shapeGaussian_cfg_defaults
from configs.pose2D_hrnet_config import get_pose2D_hrnet_cfg_defaults
from configs.optimization_constants import get_shape_optimization_cfg_defaults
from configs import paths

from predict.optimize_wc_height import optimize_poseMF_shapeGaussian_net


def run_optimize(device=None,
                subset_ids=None,
                image_dir=None,
                save_dir=None,
                csv_path=None,
                pose_shape_weights_path=None,
                pose2D_hrnet_weights_path=None,
                pose_shape_cfg_path=None,
                already_cropped_images=False,
                visualize_images=False,
                visualize_losses=False,
                joints2Dvisib_threshold=0.75,
                multiprocessing=False,
                processing_queue=None,
                print_debug=False):

    process_number = device.index if device and device.type != 'cpu' else 0

    # In the case that this process is not allocated any ids.
    if subset_ids is None or len(subset_ids) <= 0:
        print('Subrocess {} has no IDS to process'.format(process_number))
        dtypes = np.dtype([
            ('initial', str),
            ('initial_loss', str),
            ('final', str),
            ('final_loss', str),
            ('shape', list),
            ('pose', list),
            ('serno', str),
        ])
        return pd.DataFrame(np.empty(0, dtype=dtypes))
    # ------------------------- Models -------------------------
    # Configs
    pose2D_hrnet_cfg = get_pose2D_hrnet_cfg_defaults()
    pose_shape_cfg = get_poseMF_shapeGaussian_cfg_defaults()
    optimization_cfg = get_shape_optimization_cfg_defaults()
    if pose_shape_cfg_path is not None:
        pose_shape_cfg.merge_from_file(pose_shape_cfg_path)
        print('\nSubprocess {} loaded Distribution Predictor config from {}'.format(process_number, pose_shape_cfg_path))
    else:
        print('\nSubprocess {} using default Distribution Predictor config.'.format(process_number))

    # Bounding box / Object detection model
    if not already_cropped_images:
        object_detect_model = torchvision.models.detection.maskrcnn_resnet50_fpn(
            weights=torchvision.models.detection.MaskRCNN_ResNet50_FPN_Weights.DEFAULT
        ).to(device)
        print('\nSubprocess {} loaded object detection model'.format(process_number))
    else:
        object_detect_model = None

    # HRNet model for 2D joint detection
    hrnet_model = PoseHighResolutionNet(pose2D_hrnet_cfg).to(device)
    hrnet_checkpoint = torch.load(pose2D_hrnet_weights_path, map_location=device)
    hrnet_model.load_state_dict(hrnet_checkpoint, strict=False)
    print('\nSubprocess {} loaded HRNet weights from {}'.format(process_number, pose2D_hrnet_weights_path))

    # Edge detector
    edge_detect_model = CannyEdgeDetector(non_max_suppression=pose_shape_cfg.DATA.EDGE_NMS,
                                          gaussian_filter_std=pose_shape_cfg.DATA.EDGE_GAUSSIAN_STD,
                                          gaussian_filter_size=pose_shape_cfg.DATA.EDGE_GAUSSIAN_SIZE,
                                          threshold=pose_shape_cfg.DATA.EDGE_THRESHOLD).to(device)

    # SMPL model
    print('\nSubprocess {} using SMPL model with {} shape parameters.'.format(process_number, str(pose_shape_cfg.MODEL.NUM_SMPL_BETAS)))
    smpl_model_male = SMPL(paths.SMPL,
                      batch_size=1,
                      gender='male',
                      num_betas=pose_shape_cfg.MODEL.NUM_SMPL_BETAS).to(device)
    smpl_model_female = SMPL(paths.SMPL,
                      batch_size=1,
                      gender='female',
                      num_betas=pose_shape_cfg.MODEL.NUM_SMPL_BETAS).to(device)

    # 3D shape and pose distribution predictor
    pose_shape_dist_model_male = PoseMFShapeGaussianNet(smpl_parents=smpl_model_male.parents.tolist(),
                                                   config=pose_shape_cfg).to(device)
    pose_shape_dist_model_female = PoseMFShapeGaussianNet(smpl_parents=smpl_model_female.parents.tolist(),
                                                   config=pose_shape_cfg).to(device)
    checkpoint_male = torch.load(os.path.join(pose_shape_weights_path, 'poseMF_shapeGaussian_net_weights_male.tar'), map_location=device)
    checkpoint_female = torch.load(os.path.join(pose_shape_weights_path, 'poseMF_shapeGaussian_net_weights_female.tar'), map_location=device)
    pose_shape_dist_model_male.load_state_dict(checkpoint_male['best_model_state_dict'])
    pose_shape_dist_model_female.load_state_dict(checkpoint_female['best_model_state_dict'])
    print('\nSubprocess {} loaded Distribution Predictor weights from {}'.format(process_number, pose_shape_weights_path))

    # Dataframe for non image data
    data_df = pd.read_csv(csv_path)
    data_df['Serno'] = data_df['Serno'].astype(str)

    column_names = set(data_df.columns)
    column_names.remove('Serno')
    search = re.compile('_P2$')
    renamed = {k:re.sub('_P2$', '', k) for k in column_names if search.search(k)}
    if len(renamed) > 0:
        data_df.rename(columns=renamed, inplace=True)

    # ------------------------- Predict -------------------------
    torch.manual_seed(0)
    np.random.seed(0)
    print('Started processing {} IDs on device {}'.format(len(subset_ids), device))
    try:
        value = optimize_poseMF_shapeGaussian_net(pose_shape_model_male=pose_shape_dist_model_male,
                                                pose_shape_model_female=pose_shape_dist_model_female,
                                                pose_shape_cfg=pose_shape_cfg,
                                                optimization_cfg=optimization_cfg,
                                                smpl_model_male=smpl_model_male,
                                                smpl_model_female=smpl_model_female,
                                                hrnet_model=hrnet_model,
                                                hrnet_cfg=pose2D_hrnet_cfg,
                                                edge_detect_model=edge_detect_model,
                                                device=device,
                                                data_df=data_df,
                                                image_dir=image_dir,
                                                save_dir=save_dir,
                                                visualize_images=visualize_images,
                                                visualize_losses=visualize_losses,
                                                print_debug=print_debug,
                                                subset_ids=subset_ids,
                                                object_detect_model=object_detect_model,
                                                multiprocessing=multiprocessing,
                                                joints2Dvisib_threshold=joints2Dvisib_threshold
                                                )
    except Exception as e:
        print(traceback.format_exc())
    
    if processing_queue:
        processing_queue.put(value)

    print('\nSubprocess {} on device {} finished.'.format(process_number, str(device)))


if __name__ == '__main__':
    mp.set_start_method('spawn')

    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', '-I', type=str, help='Path to directory of test images.', default='../data/smpl_gen/input') # '../data/raw_images'
    parser.add_argument('--save_dir', '-S', type=str, help='Path to directory where test outputs will be saved.', default='../data/smpl_gen/output')
    parser.add_argument('--csv_path', '-CSV', type=str, help='Path to csv where numerical data is stored.', default='../data/smpl_gen/input/data.csv')
    parser.add_argument('--pose_shape_weights_dir', '-W3D', type=str, default='./model_files/')
    parser.add_argument('--pose_shape_cfg', type=str, default=None)
    parser.add_argument('--pose2D_hrnet_weights', '-W2D', type=str, default='./model_files/pose_hrnet_w48_384x288.pth')
    parser.add_argument('--cropped_images', '-C', action='store_true', help='Images already cropped and centred.')
    parser.add_argument('--joints2Dvisib_threshold', '-T', type=float, default=0.75)
    parser.add_argument('--visualize_images', '-VI', action='store_true')
    parser.add_argument('--visualize_losses', '-VL', action='store_true')
    parser.add_argument('--print_debug', '-PD', action='store_true')
    parser.add_argument('--subset_ids', '-SI', type=str, nargs='*', help='Subset of IDs to process')
    parser.add_argument('--gpus', type=int, nargs='+', help='Subset of IDs to process', default=[0])
    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, args.gpus)) if isinstance(args.gpus, list) else str(args.gpus)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    if not args.subset_ids:
        subset_ids = [Path(path).stem for path in os.listdir(args.image_dir) if path.endswith(('.jpg', '.png'))]

        df = pd.read_csv(args.csv_path)
        df['Serno'] = df['Serno'].astype(str)
        
        if 'E_Average_Waist_P2' in df.columns:
            df = df[df['Serno'].isin([re.sub('_P2$', '', id) for id in subset_ids])]
            df = df[df['E_Average_Waist_P2'] > 0]
            df = df[df['E_Average_Hip_P2'] > 0]
            df = df[df['E_Height_P2'] > 0]

            subset_ids = [id + '_P2' for id in df['Serno'].to_numpy().flatten()]
        else:
            df = df[df['Serno'].isin(subset_ids)]
            print('LENGTH', len(df))
            df = df[df['E_Average_Waist'] > 0]
            df = df[df['E_Average_Hip'] > 0]
            df = df[df['E_Height'] > 0]

            subset_ids = df['Serno'].to_numpy().flatten()
        del df
    else:
        subset_ids = args.subset_ids

    n_processes = len(args.gpus)

    if torch.cuda.is_available() and n_processes > 1:
        print('\nRunning on devices: {}'.format(','.join(map(str, args.gpus))))
        print('Running for {} IDs'.format(len(subset_ids)))
        
        id_divisions = np.array_split(subset_ids, n_processes)
        devices = [torch.device('cuda:{}'.format(args.gpus[device_num])) for device_num in range(n_processes)]
        return_values = mp.Queue()

        target_function = functools.partial(
            run_optimize, 
            image_dir=args.image_dir,
            save_dir=args.save_dir,
            csv_path=args.csv_path,
            pose_shape_weights_path=args.pose_shape_weights_dir,
            pose_shape_cfg_path=args.pose_shape_cfg,
            pose2D_hrnet_weights_path=args.pose2D_hrnet_weights,
            already_cropped_images=args.cropped_images,
            joints2Dvisib_threshold=args.joints2Dvisib_threshold,
            visualize_images=args.visualize_images,
            visualize_losses=args.visualize_losses,
            multiprocessing=True,
            processing_queue=return_values,
            print_debug=args.print_debug)

        subprocesses = []

        for device, process_ids in zip(devices, id_divisions):
            subprocesses.append(
                mp.Process(
                    target=target_function,
                    kwargs={
                        'device': device,
                        'subset_ids': np.copy(process_ids),
                    },
                    daemon=False
                )
            )
            subprocesses[-1].start()
            print('Started subprocess {} on PID {} on device {}'.format(device.index, subprocesses[-1].pid, device))

        dataframes=[]

        while len(dataframes) < len(subprocesses):
            for process in subprocesses:
                try:
                    dataframes.append(return_values.get_nowait())
                except queue.Empty:
                    pass
                finally:
                    if not process.is_alive() and process.exitcode != 0:
                        # error, traceback = process.exception
                        for kill_process in subprocesses:
                            kill_process.terminate()
                        # print('ERROR:', error)
                        # print('TRACEBACK:', traceback)
                        exit(process.exitcode)
                    else:
                        time.sleep(1)

        print('Return Values Retrieved')

        for process in subprocesses:
            process.join()

        print('Main Process Joined')
        
        if len(dataframes) > 0:
            output_dataframe = pd.concat(list(dataframes)).reset_index()
            output_dataframe.to_pickle(os.path.join(args.save_dir, 'parameters_losses.pkl.tar'))
        else:
            raise Exception('No output dataframes -- is this intended behavior')
    else:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print('\nRunning on device: {}'.format(device))
        print('Running for {} IDs'.format(len(subset_ids)))

        run_optimize(device=device,
                    image_dir=args.image_dir,
                    save_dir=args.save_dir,
                    csv_path=args.csv_path,
                    pose_shape_weights_path=args.pose_shape_weights_dir,
                    pose_shape_cfg_path=args.pose_shape_cfg,
                    pose2D_hrnet_weights_path=args.pose2D_hrnet_weights,
                    already_cropped_images=args.cropped_images,
                    joints2Dvisib_threshold=args.joints2Dvisib_threshold,
                    visualize_images=args.visualize_images,
                    visualize_losses=args.visualize_losses,
                    multiprocessing=False,
                    subset_ids=subset_ids,
                    print_debug=args.print_debug)

