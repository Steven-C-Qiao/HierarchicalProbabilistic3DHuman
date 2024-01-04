import copy
import os
import cv2
import numpy as np
import torch
from torch import nn

from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from smplx.lbs import batch_rodrigues
from tqdm import tqdm

from renderers.pytorch3d_textured_renderer import TexturedIUVRenderer

from utils.checkpoint_utils import load_training_info_from_checkpoint
from utils.cam_utils import perspective_project_torch, orthographic_project_torch
from utils.rigid_transform_utils import rot6d_to_rotmat, aa_rotate_translate_points_pytorch3d, aa_rotate_rotmats_pytorch3d
from utils.label_conversions import convert_2Djoints_to_gaussian_heatmaps_torch, convert_densepose_seg_to_14part_labels, \
    ALL_JOINTS_TO_H36M_MAP, ALL_JOINTS_TO_COCO_MAP, H36M_TO_J14
from utils.joints2d_utils import check_joints2d_visibility_torch, check_joints2d_occluded_torch
from utils.image_utils import batch_add_rgb_background, batch_crop_pytorch_affine
from utils.sampling_utils import pose_matrix_fisher_sampling_torch

from utils.augmentation.smpl_augmentation import normal_sample_shape
from utils.augmentation.cam_augmentation import augment_cam_t
from utils.augmentation.proxy_rep_augmentation import augment_proxy_representation, random_extreme_crop
from utils.augmentation.rgb_augmentation import augment_rgb
from utils.augmentation.lighting_augmentation import augment_light

from utils.eval_utils import procrustes_analysis_batch, scale_and_translation_transform_batch



def gen_rotmat(rad, axis):
    rad = torch.tensor(rad, dtype=float)
    rad += 0.1 * torch.randn_like(rad)
    if axis=='x':
        return torch.tensor([[1, 0, 0],
                            [0, torch.cos(rad), -torch.sin(rad)],
                            [0, torch.sin(rad), torch.cos(rad)]])
    elif axis=='y':
        return torch.tensor([[torch.cos(rad), 0, torch.sin(rad)],
                            [0, 1, 0],
                            [-torch.sin(rad), 0, torch.cos(rad)]])
    elif axis=='z':
        return torch.tensor([[torch.cos(rad), -torch.sin(rad), 0],
                            [torch.sin(rad), torch.cos(rad), 0],
                            [0, 0, 1]])
    else:
        assert False 

def train_val_vis(pose_shape_model,
                    pose_shape_cfg,
                    smpl_model,
                    edge_detect_model,
                    pytorch3d_renderer,
                    device,
                    train_dataset,
                    val_dataset,
                    criterion,
                    optimiser,
                    metrics,
                    vis_save_dir,
                    save_val_metrics=['PVE-SC', 'MPJPE-PA'],
                    checkpoint=None,
                    seed=None,
                    ):
    if seed is not None:
        torch.manual_seed(seed)
        print('train_val_vis seeded')
    # Set up dataloaders
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=pose_shape_cfg.TRAIN.BATCH_SIZE,
                                  shuffle=True,
                                  drop_last=True,
                                  num_workers=pose_shape_cfg.TRAIN.NUM_WORKERS,
                                  pin_memory=pose_shape_cfg.TRAIN.PIN_MEMORY)
    val_dataloader = DataLoader(val_dataset,
                                batch_size=pose_shape_cfg.TRAIN.BATCH_SIZE,
                                shuffle=True,
                                drop_last=True,
                                num_workers=pose_shape_cfg.TRAIN.NUM_WORKERS,
                                pin_memory=pose_shape_cfg.TRAIN.PIN_MEMORY)
    dataloaders = {'train': train_dataloader,
                   'val': val_dataloader}


    # Useful tensors that are re-used and can be pre-defined
    x_axis = torch.tensor([1., 0., 0.],
                          device=device, dtype=torch.float32)
    delta_betas_std_vector = torch.ones(pose_shape_cfg.MODEL.NUM_SMPL_BETAS,
                                        device=device, dtype=torch.float32) * pose_shape_cfg.TRAIN.SYNTH_DATA.AUGMENT.SMPL.SHAPE_STD
    mean_shape = torch.zeros(pose_shape_cfg.MODEL.NUM_SMPL_BETAS,
                             device=device, dtype=torch.float32)
    mean_cam_t = torch.tensor(pose_shape_cfg.TRAIN.SYNTH_DATA.MEAN_CAM_T,
                              device=device, dtype=torch.float32)
    mean_cam_t = mean_cam_t[None, :].expand(pose_shape_cfg.TRAIN.BATCH_SIZE, -1)



    for batch_num, samples_batch in enumerate(tqdm(dataloaders['val'])):
        #############################################################
        # ---------------- SYNTHETIC DATA GENERATION ----------------
        #############################################################
        with torch.no_grad():
            # ------------ RANDOM POSE, SHAPE, BACKGROUND, TEXTURE, CAMERA SAMPLING ------------
            # Load target pose and random background/texture
            target_pose = samples_batch['pose'].to(device)  # (bs, 72)
            background = samples_batch['background'].to(device)  # (bs, 3, img_wh, img_wh)
            texture = samples_batch['texture'].to(device)  # (bs, 1200, 800, 3)
            texture = texture[:int(pose_shape_cfg.TRAIN.BATCH_SIZE/3)].repeat_interleave(3, dim=0)

            # Convert target_pose from axis angle to rotmats
            target_pose_rotmats = batch_rodrigues(target_pose.contiguous().view(-1, 3)).view(-1, 24, 3, 3)
            target_glob_rotmats = target_pose_rotmats[:, 0, :, :]
            target_pose_rotmats = target_pose_rotmats[:, 1:, :, :]
            # Flipping pose targets such that they are right way up in 3D space - i.e. wrong way up when projected
            # Then pose predictions will also be right way up in 3D space - network doesn't need to learn to flip.
            _, target_glob_rotmats = aa_rotate_rotmats_pytorch3d(rotmats=target_glob_rotmats,
                                                                    angles=np.pi,
                                                                    axes=x_axis,
                                                                    rot_mult_order='post')


            # target_glob_rotmats = torch.empty(0, 3, 3).float()
            # for _ in range(int(pose_shape_cfg.TRAIN.BATCH_SIZE/3)):
            #     front = gen_rotmat(0, 'x') @ gen_rotmat(0, 'y') @ gen_rotmat(0, 'z')
            #     side =  gen_rotmat(0, 'x') @ gen_rotmat(np.sign(np.random.randn(1)) * np.pi/2, 'y') @ gen_rotmat(0, 'z')
            #     back =  gen_rotmat(0, 'x') @ gen_rotmat(np.pi, 'y') @ gen_rotmat(0, 'z')
            #     target_glob_rotmats = torch.cat((target_glob_rotmats, front.unsqueeze(0), side.unsqueeze(0), back.unsqueeze(0)), dim=0)
            # target_glob_rotmats = target_glob_rotmats.float().to(device)

            # Random sample body shape
            target_shape = normal_sample_shape(batch_size=pose_shape_cfg.TRAIN.BATCH_SIZE,
                                                mean_shape=mean_shape,
                                                std_vector=delta_betas_std_vector)
            target_shape = target_shape[:int(pose_shape_cfg.TRAIN.BATCH_SIZE/3)].repeat_interleave(3, dim=0)

            # Random sample camera translation
            target_cam_t = augment_cam_t(mean_cam_t,
                                            xy_std=pose_shape_cfg.TRAIN.SYNTH_DATA.AUGMENT.CAM.XY_STD,
                                            delta_z_range=pose_shape_cfg.TRAIN.SYNTH_DATA.AUGMENT.CAM.DELTA_Z_RANGE)

            # Compute target vertices and joints
            target_smpl_output = smpl_model(body_pose=target_pose_rotmats,
                                            global_orient=target_glob_rotmats.unsqueeze(1),
                                            betas=target_shape,
                                            pose2rot=False)
            target_vertices = target_smpl_output.vertices
            target_joints_all = target_smpl_output.joints
            target_joints_h36m = target_joints_all[:, ALL_JOINTS_TO_H36M_MAP, :]
            target_joints_h36mlsp = target_joints_h36m[:, H36M_TO_J14, :]

            target_reposed_vertices = smpl_model(body_pose=torch.zeros_like(target_pose)[:, 3:],
                                                    global_orient=torch.zeros_like(target_pose)[:, :3],
                                                    betas=target_shape).vertices

            # ------------ INPUT PROXY REPRESENTATION GENERATION + 2D TARGET JOINTS ------------
            # Pose targets were flipped such that they are right way up in 3D space - i.e. wrong way up when projected
            # Need to flip target_vertices_for_rendering 180° about x-axis so they are right way up when projected
            # Need to flip target_joints_coco 180° about x-axis so they are right way up when projected
            target_vertices_for_rendering = aa_rotate_translate_points_pytorch3d(points=target_vertices,
                                                                                    axes=x_axis,
                                                                                    angles=np.pi,
                                                                                    translations=torch.zeros(3, device=device).float())
            target_joints_coco = aa_rotate_translate_points_pytorch3d(points=target_joints_all[:, ALL_JOINTS_TO_COCO_MAP, :],
                                                                        axes=x_axis,
                                                                        angles=np.pi,
                                                                        translations=torch.zeros(3, device=device).float())
            target_joints2d_coco = perspective_project_torch(target_joints_coco,
                                                                None,
                                                                target_cam_t,
                                                                focal_length=pose_shape_cfg.TRAIN.SYNTH_DATA.FOCAL_LENGTH,
                                                                img_wh=pose_shape_cfg.DATA.PROXY_REP_SIZE)

            # Check if joints within image dimensions before cropping + recentering.
            target_joints2d_visib_coco = check_joints2d_visibility_torch(target_joints2d_coco,
                                                                            pose_shape_cfg.DATA.PROXY_REP_SIZE)  # (batch_size, 17)

            # Render RGB/IUV image
            lights_rgb_settings = augment_light(batch_size=1,
                                                device=device,
                                                rgb_augment_config=pose_shape_cfg.TRAIN.SYNTH_DATA.AUGMENT.RGB)
            renderer_output = pytorch3d_renderer(vertices=target_vertices_for_rendering,
                                                    textures=texture,
                                                    cam_t=target_cam_t,
                                                    lights_rgb_settings=lights_rgb_settings)
            iuv_in = renderer_output['iuv_images'].permute(0, 3, 1, 2).contiguous()  # (bs, 3, img_wh, img_wh)
            iuv_in[:, 1:, :, :] = iuv_in[:, 1:, :, :] * 255
            iuv_in = iuv_in.round()
            rgb_in = renderer_output['rgb_images'].permute(0, 3, 1, 2).contiguous()  # (bs, 3, img_wh, img_wh)

            # Prepare seg for extreme crop augmentation
            seg_extreme_crop = random_extreme_crop(seg=iuv_in[:, 0, :, :],
                                                    extreme_crop_probability=0)

            # Crop to person bounding box after bbox scale and centre augmentation
            crop_outputs = batch_crop_pytorch_affine(input_wh=(pose_shape_cfg.DATA.PROXY_REP_SIZE, pose_shape_cfg.DATA.PROXY_REP_SIZE),
                                                        output_wh=(pose_shape_cfg.DATA.PROXY_REP_SIZE, pose_shape_cfg.DATA.PROXY_REP_SIZE),
                                                        num_to_crop=pose_shape_cfg.TRAIN.BATCH_SIZE,
                                                        device=device,
                                                        rgb=rgb_in,
                                                        iuv=iuv_in,
                                                        joints2D=target_joints2d_coco,
                                                        bbox_determiner=seg_extreme_crop,
                                                        orig_scale_factor=pose_shape_cfg.DATA.BBOX_SCALE_FACTOR,
                                                        delta_scale_range=pose_shape_cfg.TRAIN.SYNTH_DATA.AUGMENT.BBOX.DELTA_SCALE_RANGE,
                                                        delta_centre_range=pose_shape_cfg.TRAIN.SYNTH_DATA.AUGMENT.BBOX.DELTA_CENTRE_RANGE,
                                                        out_of_frame_pad_val=-1)
            iuv_in = crop_outputs['iuv']
            target_joints2d_coco = crop_outputs['joints2D']
            rgb_in = crop_outputs['rgb']

            # Check if joints within image dimensions after cropping + recentering.
            target_joints2d_visib_coco = check_joints2d_visibility_torch(target_joints2d_coco,
                                                                            pose_shape_cfg.DATA.PROXY_REP_SIZE,
                                                                            visibility=target_joints2d_visib_coco)  # (bs, 17)
            # Check if joints are occluded by the body.
            seg_14_part_occlusion_check = convert_densepose_seg_to_14part_labels(iuv_in[:, 0, :, :])
            target_joints2d_visib_coco = check_joints2d_occluded_torch(seg_14_part_occlusion_check,
                                                                        target_joints2d_visib_coco,
                                                                        pixel_count_threshold=50)  # (bs, 17)

            # Apply segmentation/IUV-based render augmentations + 2D joints augmentations
            # seg_aug, target_joints2d_coco_input, target_joints2d_visib_coco = augment_proxy_representation(
            #     seg=iuv_in[:, 0, :, :],  # Note: out of frame pixels marked with -1
            #     joints2D=target_joints2d_coco,
            #     joints2D_visib=target_joints2d_visib_coco,
            #     proxy_rep_augment_config=pose_shape_cfg.TRAIN.SYNTH_DATA.AUGMENT.PROXY_REP)
            seg_aug = iuv_in[:, 0, :, :]
            target_joints2d_coco_input = target_joints2d_coco


            # Add background rgb
            rgb_in = batch_add_rgb_background(backgrounds=background,
                                                rgb=rgb_in,
                                                seg=seg_aug)
            # Apply RGB-based render augmentations + 2D joints augmentations
            # rgb_in, target_joints2d_coco_input, target_joints2d_visib_coco = augment_rgb(rgb=rgb_in,
            #                                                                              joints2D=target_joints2d_coco_input,
            #                                                                              joints2D_visib=target_joints2d_visib_coco,
            #                                                                              rgb_augment_config=pose_shape_cfg.TRAIN.SYNTH_DATA.AUGMENT.RGB)
            # Compute edge-images edges
            edge_detector_output = edge_detect_model(rgb_in)
            edge_in = edge_detector_output['thresholded_thin_edges'] if pose_shape_cfg.DATA.EDGE_NMS else edge_detector_output['thresholded_grad_magnitude']

            # Compute 2D joint heatmaps
            j2d_heatmaps = convert_2Djoints_to_gaussian_heatmaps_torch(target_joints2d_coco_input,
                                                                        pose_shape_cfg.DATA.PROXY_REP_SIZE,
                                                                        std=pose_shape_cfg.DATA.HEATMAP_GAUSSIAN_STD)
            j2d_heatmaps = j2d_heatmaps * target_joints2d_visib_coco[:, :, None, None]

            # Concatenate edge-image and 2D joint heatmaps to create input proxy representation
            proxy_rep_input = torch.cat([edge_in, j2d_heatmaps], dim=1).float()  # (batch_size, C, img_wh, img_wh)


            #############################################################
            # ---------------------- FORWARD PASS -----------------------
            #############################################################
            pred_pose_F, pred_pose_U, pred_pose_S, pred_pose_V, pred_pose_rotmats_mode, \
            pred_shape_dist, pred_glob, pred_cam_wp = pose_shape_model(proxy_rep_input)
            # Pose F, U, V and rotmats_mode are (bs, 23, 3, 3) and Pose S is (bs, 23, 3)

            
            # modes = np.empty((0,10))
            # scales = np.empty((0,10))

            # inv_covs = np.empty((0,10,10))

            # for data_fname in sorted([f for f in os.listdir(data_path) if f.endswith(('.npz'))]):
            #     data = np.load(os.path.join(data_path, data_fname))
            #     modes = np.vstack((modes, data['mode']))
            #     scales = np.vstack((scales, data['scale']))

            #     cov = np.zeros((10,10))
            #     np.fill_diagonal(cov, data['scale'])

            #     inv_cov = np.linalg.inv(cov)
            #     inv_covs = np.concatenate((inv_covs, inv_cov.reshape((1,10,10))))

            # S = np.linalg.inv(np.sum(inv_covs, axis=0))
            # print(inv_covs.shape)
            # print(modes.shape)
            # m = S @ (np.sum((inv_covs @ modes[:, :, None]).squeeze(-1), axis=0))

            
            #print(pred_shape_dist.loc)

            #pred_shape_dist.loc = pred_shape_dist.loc.scatter_reduce(0, [0,0,0,1,1,1], pred_shape_dist.loc, include_self=False)

            #pred_shape_dist_mean = pred_shape_dist
            mean1 = pred_shape_dist.loc[[0,1,2]].mean(dim=0)
            mean2 = pred_shape_dist.loc[[3,4,5]].mean(dim=0)
            pred_shape_dist.loc[0] = mean1
            pred_shape_dist.loc[1] = mean1
            pred_shape_dist.loc[2] = mean1
            pred_shape_dist.loc[3] = mean2
            pred_shape_dist.loc[4] = mean2
            pred_shape_dist.loc[5] = mean2

            #print(target_shape[[0, 3]])
            #print(pred_shape_dist_mean.loc[[0, 3]])


            pred_glob_rotmats = rot6d_to_rotmat(pred_glob)  # (bs, 3, 3)

            pred_smpl_output_mode = smpl_model(body_pose=pred_pose_rotmats_mode,
                                                global_orient=pred_glob_rotmats.unsqueeze(1),
                                                betas=pred_shape_dist.loc,
                                                pose2rot=False)
            pred_vertices_mode = pred_smpl_output_mode.vertices  # (bs, 6890, 3)                    
            pred_vertices_mode = aa_rotate_translate_points_pytorch3d(points=pred_vertices_mode,
                                                                axes=torch.tensor([1., 0., 0.], device=device),
                                                                angles=np.pi,
                                                                translations=torch.zeros(3, device=device))
            
            pred_reposed_smpl_output_mean = smpl_model(body_pose=torch.zeros_like(target_pose)[:, 3:],
                                                                   global_orient=torch.zeros_like(target_pose)[:, :3],
                                                                   betas=pred_shape_dist.loc)
            pred_reposed_vertices = pred_reposed_smpl_output_mean.vertices  # (bs, 6890, 3)
            
            verts3D_loss = nn.MSELoss(reduction='mean')
            verts_loss = verts3D_loss(target_vertices, pred_vertices_mode)
            #print(pred_vertices_mode.shape)

            #pve_batch = np.linalg.norm(pred_dict['verts'] - target_dict['verts'], axis=-1)  # (bsize, 6890) or (bsize, num views, 6890)
            pve_batch = np.linalg.norm((pred_vertices_mode).cpu().detach().numpy() - (target_vertices).cpu().detach().numpy(), axis=-1)  # (bsize, 6890) or (bsize, num views, 6890)
            #print(pve_batch.shape)
            print(np.sum(pve_batch)/(6890 * 6))  # scalar

            # Shape NLL
            shape_nll = -(pred_shape_dist.log_prob(target_shape).sum(dim=1))  # (batch_size,)
            shape_nll = torch.mean(shape_nll)

            shape_loss = nn.MSELoss(reduction='mean')
            shape_mse = shape_loss(target_shape, pred_shape_dist.loc)

            #print(target_shape[0])
            #print(pred_shape_dist.loc[0])

            print(f'verts3D_loss: {verts_loss}')
            print(f'shape_loss: {shape_nll}')
            #print(f'shape_mse: {shape_mse}')


            pred_reposed_vertices_sc = scale_and_translation_transform_batch(pred_reposed_vertices.cpu().detach().numpy(),
                                                                             target_reposed_vertices.cpu().detach().numpy())
            pvet_sc_batch = np.linalg.norm((pred_reposed_vertices_sc) - (target_reposed_vertices).cpu().detach().numpy(), axis=-1)  # (bs, 6890)
            print(np.sum(pvet_sc_batch))  # scalar


            pred_vertices_sc = scale_and_translation_transform_batch(pred_vertices_mode.cpu().detach().numpy(),
                                                                     target_vertices.cpu().detach().numpy())
            pve_sc_batch = np.linalg.norm(pred_vertices_sc - target_vertices.cpu().detach().numpy(), axis=-1)  # (bsize, 6890) or (bsize*num views, 6890)
            print(np.sum(pve_sc_batch))  # scalar




            # -------------------------------------- VISUALISATION --------------------------------------
            lights_rgb_settings = {'location': torch.tensor([[0., -0.8, -2.0]], device=device, dtype=torch.float32),
                    'ambient_color': 0.5 * torch.ones(1, 3, device=device, dtype=torch.float32),
                    'diffuse_color': 0.3 * torch.ones(1, 3, device=device, dtype=torch.float32),
                    'specular_color': torch.zeros(1, 3, device=device, dtype=torch.float32)}
            
            
            per_vertex_3Dvar = torch.zeros((6890,))+0.1
            vertex_var_norm = plt.Normalize(vmin=0.0, vmax=0.2, clip=True)
            vertex_var_colours = plt.cm.jet(vertex_var_norm(per_vertex_3Dvar.numpy()))[:, :3]
            vertex_var_colours = torch.from_numpy(vertex_var_colours[None, :, :]).expand(pose_shape_cfg.TRAIN.BATCH_SIZE, -1, -1).to(device).float()

            visualise_wh=512

            
            body_vis_renderer = TexturedIUVRenderer(device=device,
                                batch_size=pose_shape_cfg.TRAIN.BATCH_SIZE,
                                img_wh=visualise_wh,
                                projection_type='orthographic',
                                render_rgb=True,
                                bin_size=32)
            # Predicted camera corresponding to proxy rep input
            orthographic_scale = pred_cam_wp[:, [0, 0]]
            cam_t = torch.cat([pred_cam_wp[:, 1:],
                            torch.ones(pred_cam_wp.shape[0], 1, device=device).float() * 2.5],
                            dim=-1)

            # Render visualisation outputs
            body_vis_output = body_vis_renderer(vertices=pred_vertices_mode,
                                                cam_t=cam_t,
                                                orthographic_scale=orthographic_scale,
                                                lights_rgb_settings=lights_rgb_settings,
                                                verts_features=vertex_var_colours)
            cropped_for_proxy_rgb = torch.nn.functional.interpolate(crop_outputs['rgb'],#cropped_for_proxy['rgb'],
                                                                    size=(visualise_wh, visualise_wh),
                                                                    mode='bilinear',
                                                                    align_corners=False)

            body_vis_rgb = batch_add_rgb_background(backgrounds=cropped_for_proxy_rgb,
                                                    rgb=body_vis_output['rgb_images'].permute(0, 3, 1, 2).contiguous(),
                                                    seg=body_vis_output['iuv_images'][:, :, :, 0].round())
            body_vis_rgb = body_vis_rgb.cpu().detach().numpy().transpose(0, 2, 3, 1)

            # Combine all visualisations
            combined_vis_rows = 2
            combined_vis_cols = 3
            combined_vis_fig = np.zeros((combined_vis_rows * visualise_wh, combined_vis_cols * visualise_wh, 3),
                                        dtype=body_vis_rgb.dtype)
            # Cropped input image
            combined_vis_fig[:visualise_wh, :visualise_wh] = cropped_for_proxy_rgb.cpu().detach().numpy()[0].transpose(1, 2, 0)

            # Proxy representation + 2D joints scatter + 2D joints confidences
            proxy_rep_input = proxy_rep_input[0].sum(dim=0).cpu().detach().numpy()
            proxy_rep_input = np.stack([proxy_rep_input]*3, axis=-1)  # single-channel to RGB

            
            proxy_rep_input = cv2.resize(proxy_rep_input, (visualise_wh, visualise_wh))

            combined_vis_fig[visualise_wh:2*visualise_wh, :visualise_wh] = proxy_rep_input

            for i in range(3):
                baseline_tovis = cv2.resize(body_vis_rgb[i], (visualise_wh, visualise_wh))
                baseline_tovis_2 = cv2.resize(body_vis_rgb[i+3], (visualise_wh, visualise_wh))
                combined_vis_fig[:visualise_wh, i * visualise_wh: (i+1)*visualise_wh] = baseline_tovis
                combined_vis_fig[visualise_wh: 2 * visualise_wh, i * visualise_wh: (i+1)*visualise_wh] = baseline_tovis_2

            vis_save_path = os.path.join(vis_save_dir, 'mult_result'+ str(seed) +'.png')
            
            cv2.imwrite(vis_save_path, combined_vis_fig[:, :, ::-1] * 255)

            import sys
            sys.exit()