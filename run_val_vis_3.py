import os
import torch
import torch.optim as optim
import argparse

from data.on_the_fly_smpl_train_dataset import OnTheFlySMPLTrainDataset
from renderers.pytorch3d_textured_renderer import TexturedIUVRenderer

from models.multin_v3 import MultinV3
from models.poseMF_shapeGaussian_net import PoseMFShapeGaussianNet
from models.smpl_official import SMPL
from models.canny_edge_detector import CannyEdgeDetector

from losses.matrix_fisher_loss import PoseMFShapeGaussianLoss

from configs.val_vis_config import get_poseMF_shapeGaussian_cfg_defaults
from configs import paths

from train.train_val_vis_3 import train_val_vis


def run_val_vis(device,
              save_dir,
              ckpt_path,
              pose_shape_cfg_opts=None,
              resume_from_epoch=None,
              seed=None,
              baseline=None,
              ):
    if seed is not None:
        torch.manual_seed(seed)

    pose_shape_cfg = get_poseMF_shapeGaussian_cfg_defaults()

    if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    checkpoint = torch.load(ckpt_path, map_location=device)
    #pose_shape_cfg.merge_from_file(config_save_path)
    print('\nResuming from:', ckpt_path)

    # print('\n', pose_shape_cfg)
    # ------------------------- Datasets -------------------------
    train_dataset = OnTheFlySMPLTrainDataset(poses_path=paths.TRAIN_POSES_PATH,
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
    pose_shape_model = MultinV3(smpl_parents=smpl_model.parents.tolist(),
                                              config=pose_shape_cfg).to(device)
    if baseline==True:
        pose_shape_model = PoseMFShapeGaussianNet(smpl_parents=smpl_model.parents.tolist(),
                                            config=pose_shape_cfg).to(device)
        checkpoint = torch.load("/scratches/nazgul/cq244/hkpd_depth/model_files/poseMF_shapeGaussian_net_weights.tar", map_location=device) 
        print('loaded baseline')

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
    train_val_vis(pose_shape_model=pose_shape_model,
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
                    vis_save_dir = save_dir,
                    checkpoint=checkpoint,
                    seed=seed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', '-S', type=str)
    parser.add_argument('--pose_shape_cfg_opts', '-O', nargs='*', default=None)
    parser.add_argument('--ckpt', '-C', type=str, default=None)
    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument('--baseline', type=bool, default=False)

    args = parser.parse_args()
    #args.ckpt = "/scratches/kyuban/cq244/multin_v2/experiments/multin_v4_002/saved_models/epoch_135.tar"
    args.ckpt = "/scratches/kyuban/cq244/multin_v2/experiments/multin_v3_001/saved_models/epoch_299.tar"

    if args.baseline==True:
         args.save_dir = "./misc/val_vis_test/baseline/"

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #print('\nDevice: {}'.format(device))

    seed = 8

    run_val_vis(device=device,
                pose_shape_cfg_opts=args.pose_shape_cfg_opts,
                resume_from_epoch=None,
                save_dir=args.save_dir,
                ckpt_path=args.ckpt,
                seed=seed,
                baseline=args.baseline)

