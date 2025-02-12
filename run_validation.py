import os
import torch
import torch.optim as optim
import argparse
import numpy as np

from data.on_the_fly_smpl_train_dataset import OnTheFlySMPLTrainDataset
from data.flying_small_subset import OnTheFlySMPLTrainDataset as SmallTrainDataset
from renderers.pytorch3d_textured_renderer import TexturedIUVRenderer
from renderers.pytorch3d_silh_opt_renderer import SilhouetteRenderer

from models.poseMF_shapeGaussian_net import PoseMFShapeGaussianNet
from models.multin_v3 import MultinV3
from models.smpl_official import SMPL
from models.canny_edge_detector import CannyEdgeDetector


from losses.matrix_fisher_loss import PoseMFShapeGaussianLoss

from configs.poseMF_shapeGaussian_net_config import get_poseMF_shapeGaussian_cfg_defaults
from configs import paths

from train.validate import train_multin


def run_train(device,
              experiment_dir,
              pose_shape_cfg_opts=None,
              resume_from_epoch=None):

    pose_shape_cfg = get_poseMF_shapeGaussian_cfg_defaults()

    model_save_dir = os.path.join(experiment_dir, 'saved_models')
    logs_save_path = os.path.join(experiment_dir, 'log.pkl')
    config_save_path = os.path.join(experiment_dir, 'pose_shape_cfg.yaml')
    print('\nSaving model checkpoints to:', model_save_dir)
    print('Saving logs to:', logs_save_path)
    print('Saving config to:', config_save_path)

    if resume_from_epoch is None:
        if not os.path.exists(model_save_dir):
            os.makedirs(model_save_dir)
        else:
            print('\nWARNING: {} already exists - may be overwriting previous experiments!'.format(experiment_dir))
        if pose_shape_cfg_opts is not None:
            pose_shape_cfg.merge_from_list(pose_shape_cfg_opts)
        with open(config_save_path, 'w') as f:
            f.write(pose_shape_cfg.dump())
        checkpoint = None
    else:
        assert os.path.exists(model_save_dir), 'Experiment to resume not found.'
        checkpoint_path = os.path.join(model_save_dir, 'epoch_{}'.format(str(resume_from_epoch).zfill(3)) + '.tar')
        checkpoint = torch.load(checkpoint_path, map_location=device)
        pose_shape_cfg.merge_from_file(config_save_path)
        print('\nResuming from:', checkpoint_path)

        #pose_shape_cfg.LOSS.STAGE_CHANGE_EPOCH = 120
        #pose_shape_cfg.STAGE1.WEIGHTS.SILHOUETTE = 100.0
        #pose_shape_cfg.STAGE2.WEIGHTS.SILHOUETTE = 100.0
    pose_shape_cfg.TRAIN.LR = 0.

    # print('\n', pose_shape_cfg)
    # ------------------------- Datasets -------------------------
    train_dataset = SmallTrainDataset(poses_path=paths.TRAIN_POSES_PATH,
                                             textures_path=paths.TRAIN_TEXTURES_PATH,
                                             backgrounds_dir_path=paths.TRAIN_BACKGROUNDS_PATH,
                                             params_from='not_amass',
                                             img_wh=pose_shape_cfg.DATA.PROXY_REP_SIZE)
    val_dataset = OnTheFlySMPLTrainDataset(poses_path=paths.VAL_POSES_PATH,
                                           textures_path=paths.VAL_TEXTURES_PATH,
                                           backgrounds_dir_path=paths.VAL_BACKGROUNDS_PATH,
                                           params_from='all',
                                           img_wh=pose_shape_cfg.DATA.PROXY_REP_SIZE)
    # print("\nTraining poses found:", len(train_dataset))
    # print("Training textures found (grey, nongrey):", len(train_dataset.grey_textures), len(train_dataset.nongrey_textures))
    # print("Training backgrounds found:", len(train_dataset.backgrounds_paths))
    # print("Validation poses found:", len(val_dataset))
    # print("Validation textures found (grey, nongrey):", len(val_dataset.grey_textures), len(val_dataset.nongrey_textures))
    # print("Validation backgrounds found:", len(val_dataset.backgrounds_paths), '\n')

    # ------------------------- Models -------------------------
    # Edge detector
    edge_detect_model = CannyEdgeDetector(non_max_suppression=pose_shape_cfg.DATA.EDGE_NMS,
                                          gaussian_filter_std=pose_shape_cfg.DATA.EDGE_GAUSSIAN_STD,
                                          gaussian_filter_size=pose_shape_cfg.DATA.EDGE_GAUSSIAN_SIZE,
                                          threshold=pose_shape_cfg.DATA.EDGE_THRESHOLD).to(device)
    # SMPL model
    smpl_model = SMPL(paths.SMPL,
                      num_betas=pose_shape_cfg.MODEL.NUM_SMPL_BETAS).to(device)

    # 3D shape and pose distribution predictor
    baseline_model = PoseMFShapeGaussianNet(smpl_parents=smpl_model.parents.tolist(),
                                              config=pose_shape_cfg).to(device)
    checkpoint_path = "/scratches/nazgul/cq244/hkpd_depth/model_files/poseMF_shapeGaussian_net_weights.tar"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    baseline_model.load_state_dict(checkpoint['best_model_state_dict'])

    pose_shape_model = MultinV3(smpl_parents=smpl_model.parents.tolist(),
                                              config=pose_shape_cfg).to(device)
    checkpoint_path = "/scratches/kyuban/cq244/multin_v2/experiments/multin_v3_001/saved_models/epoch_299.tar"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    pose_shape_model.load_state_dict(checkpoint['best_model_state_dict'])
    

    # Pytorch3D renderer for synthetic data generation
    pytorch3d_renderer = TexturedIUVRenderer(device=device,
                                             batch_size=pose_shape_cfg.TRAIN.BATCH_SIZE,
                                             img_wh=pose_shape_cfg.DATA.PROXY_REP_SIZE,
                                             projection_type='perspective',
                                             perspective_focal_length=pose_shape_cfg.TRAIN.SYNTH_DATA.FOCAL_LENGTH ,
                                             render_rgb=True,
                                             bin_size=32)
    

    # ------------------------- Loss Function + Optimiser -------------------------
    criterion = PoseMFShapeGaussianLoss(loss_config=pose_shape_cfg.LOSS.STAGE1,
                                        img_wh=pose_shape_cfg.DATA.PROXY_REP_SIZE)
    optimiser = optim.Adam(pose_shape_model.parameters(),
                           lr=pose_shape_cfg.TRAIN.LR)

    # ------------------------- Train -------------------------
    if resume_from_epoch is not None:
        assert False
        pose_shape_model.load_state_dict(checkpoint['model_state_dict'])
        optimiser.load_state_dict(checkpoint['optimiser_state_dict'])

    train_multin(pose_shape_model=pose_shape_model,
                    baseline_model = baseline_model,
                    pose_shape_cfg=pose_shape_cfg,
                    smpl_model=smpl_model,
                    edge_detect_model=edge_detect_model,
                    pytorch3d_renderer=pytorch3d_renderer,
                    device=device,
                    train_dataset=train_dataset,
                    val_dataset=val_dataset,
                    criterion=criterion,
                    optimiser=optimiser,
                    metrics=['PVE', 'PVE-SC', 'PVE-T-SC', 'MPJPE', 'MPJPE-SC', 'MPJPE-PA', 'joints2D-L2E'],
                    model_save_dir=model_save_dir,
                    logs_save_path=logs_save_path,
                    checkpoint=None)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_dir', '-E', type=str)
    parser.add_argument('--pose_shape_cfg_opts', '-O', nargs='*', default=None)
    parser.add_argument('--resume_from_epoch', '-R', type=int, default=None)
    parser.add_argument('--gpu', type=int, default=1)
    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('\nDevice: {}'.format(device))

    run_train(device=device,
              experiment_dir=args.experiment_dir,
              pose_shape_cfg_opts=args.pose_shape_cfg_opts,
              resume_from_epoch=args.resume_from_epoch)
