import os
import numpy as np
import torch
import torch.optim as optim
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import json
import collections
from pathlib import Path
from tqdm import tqdm
from smplx.lbs import batch_rodrigues
from pytorch3d.transforms import so3

from predict.predict_hrnet import predict_hrnet

from renderers.pytorch3d_textured_renderer import TexturedIUVRenderer
from renderers.pytorch3d_silhouette_renderer import SilhouetteRenderer

from utils.image_utils import batch_add_rgb_background, batch_crop_pytorch_affine, resize_and_pad, convert_to_silhouette
from utils.label_conversions import convert_2Djoints_to_gaussian_heatmaps_torch
from utils.rigid_transform_utils import rot6d_to_rotmat, aa_rotate_translate_points_pytorch3d
from utils.sampling_utils import compute_vertex_uncertainties_by_poseMF_shapeGaussian_sampling, joints2D_error_sorted_verts_sampling

from utils.cam_utils import orthographic_project_torch
from utils.joints2d_utils import undo_keypoint_normalisation
from utils.label_conversions import ALL_JOINTS_TO_COCO_MAP

def ellipse_length(width, depth):
    a = width / 2
    b = depth / 2
    return np.pi * (3 * (a + b) - torch.sqrt((3 * a + b) * (a + 3 * b)))

def optimize_poseMF_shapeGaussian_net(pose_shape_model_male,
                                     pose_shape_model_female,
                                     pose_shape_cfg,
                                     optimization_cfg,
                                     smpl_model_male,
                                     smpl_model_female,
                                     hrnet_model,
                                     hrnet_cfg,
                                     edge_detect_model,
                                     device,
                                     image_dir,
                                     save_dir,
                                     data_df,
                                     visualize_images=False,
                                     visualize_losses=False,
                                     object_detect_model=None,
                                     joints2Dvisib_threshold=0.75,
                                     print_debug=False,
                                     subset_ids=None,
                                     multiprocessing=False,
                                     visualise_wh=512):
    """
    Predictor for SingleInputKinematicPoseMFShapeGaussianwithGlobCam on unseen test data.
    Input --> ResNet --> image features --> FC layers --> MF over pose and Diagonal Gaussian over shape.
    Also get cam and glob separately to distribution predictor.
    Pose predictions follow the kinematic chain.
    """
    device_index = device.index

    # Setting up body visualisation renderer
    body_vis_renderer = TexturedIUVRenderer(device=device,
                                            batch_size=1,
                                            img_wh=visualise_wh,
                                            projection_type='orthographic',
                                            render_rgb=True,
                                            bin_size=32)
    silhouette_renderer = SilhouetteRenderer(device=device,
                                              batch_size=1,
                                              img_wh=visualise_wh,
                                              projection_type='orthographic',
                                              bin_size=32)
    R_matrix = torch.tensor([[1., 0., 0.],
                            [0., 1., 0.],
                            [0., 0., 1.]], device=device).float()
    R_matrix = R_matrix[None, :, :].expand(1, -1, -1)
    silhouette_renderer_alpha = TexturedIUVRenderer(device=device,
                                            batch_size=1,
                                            img_wh=visualise_wh,
                                            projection_type='orthographic',
                                            render_rgb=True,
                                            cam_R=R_matrix,
                                            bin_size=32)

    plain_texture = torch.ones(1, 1200, 800, 3, device=device).float() * 0.7
    lights_rgb_settings = {'location': torch.tensor([[0., -0.8, -2.0]], device=device, dtype=torch.float32),
                           'ambient_color': 0.5 * torch.ones(1, 3, device=device, dtype=torch.float32),
                           'diffuse_color': 0.3 * torch.ones(1, 3, device=device, dtype=torch.float32),
                           'specular_color': torch.zeros(1, 3, device=device, dtype=torch.float32)}
    fixed_cam_t = torch.tensor([[0., -0.2, 2.5]], device=device)
    fixed_orthographic_scale = torch.tensor([[0.95, 0.95]], device=device)

    hrnet_model.eval()
    pose_shape_model_male.eval()
    pose_shape_model_female.eval()
    if object_detect_model is not None:
        object_detect_model.eval()
    
    num_images = 0

    save_df = collections.defaultdict(list)

    if subset_ids is not None and len(subset_ids) > 0:
        image_fnames = sorted(map(lambda x: x + '.jpg', subset_ids))
    else:
        image_fnames = sorted([f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))])
    
    # Processing Data csv for measurement optimization
    progress_bar_images = tqdm(
        image_fnames, 
        position=device_index if multiprocessing else 0, 
        leave=(not multiprocessing),
        dynamic_ncols=True
    )
    for image_fname in progress_bar_images:
        num_images += 1
        with torch.no_grad():
            # ------------------------- INPUT LOADING AND PROXY REPRESENTATION GENERATION -------------------------
            image_id = Path(image_fname).stem
            progress_bar_images.set_description('{} - {}'.format(str(device), image_id))
            personal_df = data_df[data_df['Serno'] == image_id]
            gender = personal_df['Sex'].values[0]

            if gender.strip().upper() == 'M':
                pose_shape_model = pose_shape_model_male
                smpl_model = smpl_model_male
            else:
                pose_shape_model = pose_shape_model_female
                smpl_model = smpl_model_female

            image = cv2.cvtColor(cv2.imread(os.path.join(image_dir, image_fname)), cv2.COLOR_BGR2RGB)
            orig_image = image.copy()
            image = torch.from_numpy(image.transpose(2, 0, 1)).float().to(device) / 255.0
            # Predict Person Bounding Box + 2D Joints
            hrnet_output = predict_hrnet(hrnet_model=hrnet_model,
                                         hrnet_config=hrnet_cfg,
                                         object_detect_model=object_detect_model,
                                         image=image,
                                         object_detect_threshold=pose_shape_cfg.DATA.BBOX_THRESHOLD,
                                         bbox_scale_factor=pose_shape_cfg.DATA.BBOX_SCALE_FACTOR)

            # Transform predicted 2D joints and image from HRNet input size to input proxy representation size
            hrnet_input_centre = torch.tensor([[hrnet_output['cropped_image'].shape[1],
                                                hrnet_output['cropped_image'].shape[2]]],
                                              dtype=torch.float32,
                                              device=device) * 0.5
            hrnet_input_height = torch.tensor([hrnet_output['cropped_image'].shape[1]],
                                              dtype=torch.float32,
                                              device=device)
            cropped_for_proxy = batch_crop_pytorch_affine(input_wh=(hrnet_cfg.MODEL.IMAGE_SIZE[0], hrnet_cfg.MODEL.IMAGE_SIZE[1]),
                                                          output_wh=(pose_shape_cfg.DATA.PROXY_REP_SIZE, pose_shape_cfg.DATA.PROXY_REP_SIZE),
                                                          num_to_crop=1,
                                                          device=device,
                                                          joints2D=hrnet_output['joints2D'][None, :, :],
                                                          rgb=hrnet_output['cropped_image'][None, :, :, :],
                                                          bbox_centres=hrnet_input_centre,
                                                          bbox_heights=hrnet_input_height,
                                                          bbox_widths=hrnet_input_height,
                                                          orig_scale_factor=1.0)

            cropped_for_proxy_rgb = torch.nn.functional.interpolate(cropped_for_proxy['rgb'],
                                                                    size=(visualise_wh, visualise_wh),
                                                                    mode='bilinear',
                                                                    align_corners=False)

            # Create proxy representation with 1) Edge detection and 2) 2D joints heatmaps generation
            edge_detector_output = edge_detect_model(cropped_for_proxy['rgb'])
            proxy_rep_img = edge_detector_output['thresholded_thin_edges'] if pose_shape_cfg.DATA.EDGE_NMS else edge_detector_output['thresholded_grad_magnitude']
            proxy_rep_heatmaps = convert_2Djoints_to_gaussian_heatmaps_torch(joints2D=cropped_for_proxy['joints2D'],
                                                                             img_wh=pose_shape_cfg.DATA.PROXY_REP_SIZE,
                                                                             std=pose_shape_cfg.DATA.HEATMAP_GAUSSIAN_STD)
            hrnet_joints2Dvisib = hrnet_output['joints2Dconfs'] > joints2Dvisib_threshold
            hrnet_joints2Dvisib[[0, 1, 2, 3, 4, 5, 6, 11, 12]] = True  # Only removing joints [7, 8, 9, 10, 13, 14, 15, 16] if occluded
            proxy_rep_heatmaps = proxy_rep_heatmaps * hrnet_joints2Dvisib[None, :, None, None]
            proxy_rep_input = torch.cat([proxy_rep_img, proxy_rep_heatmaps], dim=1).float()  # (1, 18, img_wh, img_wh)

            # ------------------------------- POSE AND SHAPE DISTRIBUTION PREDICTION -------------------------------
            pred_pose_F, pred_pose_U, pred_pose_S, pred_pose_V, pred_pose_rotmats_mode, \
            pred_shape_dist, pred_glob, pred_cam_wp = pose_shape_model(proxy_rep_input)
            # Pose F, U, V and rotmats_mode are (bsize, 23, 3, 3) and Pose S is (bsize, 23, 3)

            if pred_glob.shape[-1] == 3:
                pred_glob_rotmats = batch_rodrigues(pred_glob)  # (1, 3, 3)
            elif pred_glob.shape[-1] == 6:
                pred_glob_rotmats = rot6d_to_rotmat(pred_glob)  # (1, 3, 3)

        
        # ---------------------------------------- TARGET VALUES -------------------------------------------
        waist_val = torch.tensor(personal_df['E_Average_Waist'].values / 100,
                                            dtype=torch.float32,
                                            device=device).squeeze()
        hip_val = torch.tensor(personal_df['E_Average_Hip'].values / 100,
                                            dtype=torch.float32,
                                            device=device).squeeze()
        height_val = torch.tensor(personal_df['E_Height'].values / 100,
                                            dtype=torch.float32,
                                            device=device).squeeze()
        target_joints = cropped_for_proxy['joints2D']
        target_silhouette = (cropped_for_proxy_rgb.cpu().detach().numpy()[0].transpose(1, 2, 0) * 255).astype('uint8')
        target_silhouette = convert_to_silhouette(target_silhouette)
        target_silhouette = torch.from_numpy(target_silhouette).type(torch.FloatTensor).to(device)

        dxa_method = personal_df['DEXAmethod'].values[0]

        # old indices
        # hip_idx = (1229, 4949, 3145, 3141)
        # waist_idx = (800, 4165, 4402, 4731)
        
        # new indices
        # hip_idx = (808, 4920, 3119, 3145)
        # waist_idx = (1488, 4960, 1326, 3023)
        
        hip_idx = (1446, 4920, 3145, 3141)
        waist_idx = (800, 4165, 4402, 4731)
        
        num_iterations = optimization_cfg.OPTIMIZATION.NUM_ITERATIONS
        loss_history = np.zeros(num_iterations)
        loss_fn = torch.nn.MSELoss()
        scalar_loss_fn = torch.nn.L1Loss()

        # Predicted camera corresponding to proxy rep input
        pred_orthographic_scale = pred_cam_wp[:, [0, 0]]
        pred_cam_t = torch.cat([pred_cam_wp[:, 1:],
                            torch.ones(pred_cam_wp.shape[0], 1, device=device).float() * 2.5],
                            dim=-1)

        pred_shape_dist_var = torch.clone(pred_shape_dist.loc)
        pred_pose_rotmats_mode_var = so3.so3_log_map(pred_pose_rotmats_mode[0])
        pred_glob_rotmats_var = so3.so3_log_map(pred_glob_rotmats)

        pred_orthographic_scale.requires_grad = True
        pred_cam_t.requires_grad = True

        pred_pose_rotmats_mode_var.requires_grad = True
        pred_glob_rotmats_var.requires_grad = False
        pred_shape_dist_var.requires_grad = True

        previous_pose_rotmats_mode_var = pred_pose_rotmats_mode_var.clone().requires_grad_(True)
        
        all_param_optimizer = optim.Adagrad(params=(pred_shape_dist_var, pred_pose_rotmats_mode_var, pred_glob_rotmats_var, pred_orthographic_scale, pred_cam_t),
                               lr=optimization_cfg.OPTIMIZATION.LEARNING_RATE)
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(all_param_optimizer)

        for i in tqdm(range(num_iterations), desc='{} mesh_optimization'.format(image_id), position=1, leave=False, disable=multiprocessing, dynamic_ncols=True):
            all_param_optimizer.zero_grad()
            pred_body = smpl_model(body_pose=so3.so3_exp_map(pred_pose_rotmats_mode_var)[None, :, :, :],
                                global_orient=so3.so3_exp_map(pred_glob_rotmats_var).unsqueeze(1),
                                betas=pred_shape_dist_var,
                                pose2rot=False)

            pred_t_pose = smpl_model(betas=pred_shape_dist_var)
            t_pose_vertices = pred_t_pose.vertices[0]

            pred_height = torch.max(t_pose_vertices[:, 1]) - torch.min(t_pose_vertices[:, 1])
            scaling_factor = height_val / pred_height

            pred_joints_coco = pred_body.joints[:, ALL_JOINTS_TO_COCO_MAP, :]
            pred_joints_coco = aa_rotate_translate_points_pytorch3d(points=pred_joints_coco,
                                                                        axes=torch.tensor([1., 0., 0.], device=device).float(),
                                                                        angles=np.pi,
                                                                        translations=torch.zeros(3, device=device).float())
            pred_joints_coco = orthographic_project_torch(pred_joints_coco, pred_cam_wp)
            pred_joints_coco = undo_keypoint_normalisation(pred_joints_coco, pose_shape_cfg.DATA.PROXY_REP_SIZE)

            # ------------ CALCULATING LOSS ------------------------------

            pred_vertices = pred_body.vertices[0] # (6890, 3)
            pred_waist_width = torch.linalg.norm(t_pose_vertices[waist_idx[0]] - t_pose_vertices[waist_idx[1]]) * scaling_factor
            pred_waist_depth = torch.linalg.norm(t_pose_vertices[waist_idx[2]] - t_pose_vertices[waist_idx[3]]) * scaling_factor

            pred_wc = ellipse_length(pred_waist_width, pred_waist_depth)

            hip_vertices = t_pose_vertices[(t_pose_vertices[:, 1] < 0) & (t_pose_vertices[:, 1] > -pred_height / 4)]
            pred_hip_width = torch.linalg.norm(t_pose_vertices[hip_idx[0]] - t_pose_vertices[hip_idx[1]]) * scaling_factor
            # pred_hip_depth = torch.linalg.norm(t_pose_vertices[hip_idx[2]] - t_pose_vertices[hip_idx[3]]) * scaling_factor
            # pred_hip_width = torch.max(hip_vertices[:, 0]) - torch.min(hip_vertices[:, 0]) * scaling_factor
            pred_hip_depth = torch.abs(t_pose_vertices[3117, 2] - t_pose_vertices[3510, 2]) * scaling_factor

            pred_hc = ellipse_length(pred_hip_width, pred_hip_depth)
        
            # renderer_output = silhouette_renderer(pred_body.vertices, cam_t=pred_cam_t, orthographic_scale=pred_orthographic_scale)
            renderer_output = silhouette_renderer_alpha(pred_body.vertices,
                                                cam_t=pred_cam_t,
                                                orthographic_scale=pred_orthographic_scale,
                                                textures=plain_texture,
                                                lights_rgb_settings=lights_rgb_settings)
            # pred_silhouette = renderer_output['silhouettes'].squeeze() * 255
            pred_silhouette = renderer_output['alpha_images'].squeeze() * 255

            losses = {}
            losses['height'] = scalar_loss_fn(pred_height, height_val) * optimization_cfg.OPTIMIZATION.WEIGHTS.HEIGHT
            losses['waist'] = scalar_loss_fn(pred_wc, waist_val) * optimization_cfg.OPTIMIZATION.WEIGHTS.WAIST
            losses['hip'] = scalar_loss_fn(pred_hc, hip_val) * optimization_cfg.OPTIMIZATION.WEIGHTS.HIP
            losses['joints'] = loss_fn(target_joints, pred_joints_coco) * optimization_cfg.OPTIMIZATION.WEIGHTS.JOINTS
            losses['silhouette'] = loss_fn(target_silhouette, pred_silhouette) * optimization_cfg.OPTIMIZATION.WEIGHTS.SILHOUETTE
            losses['norm'] = torch.norm(pred_shape_dist_var) * optimization_cfg.OPTIMIZATION.WEIGHTS.NORM
            if dxa_method == 'Normal':
                losses['prior_pose'] = torch.norm(pred_pose_rotmats_mode_var - previous_pose_rotmats_mode_var) * optimization_cfg.OPTIMIZATION.WEIGHTS.PRIOR_POSE
            # 384 for head 3117 for back of butt
            losses['head_feet_plane']= (torch.abs(pred_vertices[3117, 2] - pred_vertices[3382, 2]) + torch.abs(pred_vertices[3117, 2] - pred_vertices[6858, 2])) * optimization_cfg.OPTIMIZATION.WEIGHTS.HEAD_FEET_PlANE

            loss = torch.sum(torch.stack([v for k, v in losses.items() if optimization_cfg.OPTIMIZATION.SWITCHES[k.upper()]]))

            if optimization_cfg.OPTIMIZATION.DELTA_POSE_LOSS:
                previous_pose_rotmats_mode_var = pred_pose_rotmats_mode_var.clone().requires_grad_(True)

            if i == 0:
                if print_debug:
                    print('IMAGE ID: {}'.format(image_id))
                    print('INITIAL SCALING FACTOR: {}'.format(scaling_factor.item()))
                    print('INITIAL WAIST TARGET: {} PRED: {}'.format(waist_val.item(), pred_wc.item()))
                    print('INITIAL HIP TARGET: {} PRED: {}'.format(hip_val.item(), pred_hc.item()))
                    print(json.dumps({k: v.item() for k, v in losses.items()}, indent=4))

                save_df['initial'].append(json.dumps({
                    'waist': pred_wc.item(),
                    'hip': pred_hc.item(),
                }, indent=4))

                save_df['initial_loss'].append(
                    json.dumps({k: v.item() for k, v in losses.items()}, indent=4)
                )

                save_df['prior_shape'].append(pred_shape_dist_var.detach().tolist())
                save_df['prior_pose'].append(so3.so3_exp_map(pred_pose_rotmats_mode_var)[None, :, :, :].detach().tolist())

            if i == num_iterations - 1:
                if print_debug:
                    print('IMAGE ID: {}'.format(image_id))
                    print('FINAL SCALING FACTOR: {}'.format(scaling_factor.item()))
                    print('FINAL WAIST TARGET: {} PRED: {}'.format(waist_val.item(), pred_wc.item()))
                    print('FINAL HIP TARGET: {} PRED: {}'.format(hip_val.item(), pred_hc.item()))
                    print(json.dumps({k: v.item() for k, v in losses.items()}, indent=4))

                save_df['final'].append(json.dumps({
                    'waist': pred_wc.item(),
                    'hip': pred_hc.item(),
                }, indent=4))

                save_df['final_loss'].append(
                    json.dumps({k: v.item() for k, v in losses.items()}, indent=4)
                )

            # --------- CALCULATING GRADIENTS ---------------------

            loss_history[i] = loss.item()
            loss.backward()
            all_param_optimizer.step()

            if i > 0 and i % optimization_cfg.OPTIMIZATION.LR_SCHEDULER_STEP == 0:
                lr_scheduler.step(np.mean(loss_history[-10:]))

        save_df['shape'].append(pred_shape_dist_var.detach().tolist())
        save_df['pose'].append(so3.so3_exp_map(pred_pose_rotmats_mode_var)[None, :, :, :].detach().tolist())
        save_df['serno'].append(image_id)

        if visualize_images:
            with torch.no_grad():
                # Generate SMPL vertices and mesh
                pred_smpl_output_mode = smpl_model(body_pose=so3.so3_exp_map(pred_pose_rotmats_mode_var)[None, :, :, :],
                                                    global_orient=so3.so3_exp_map(pred_glob_rotmats_var).unsqueeze(1),
                                                    betas=pred_shape_dist_var,
                                                    pose2rot=False)
                pred_vertices_mode = pred_smpl_output_mode.vertices  # (1, 6890, 3)
                # Need to flip pred_vertices before projecting so that they project the right way up.
                pred_vertices_mode = aa_rotate_translate_points_pytorch3d(points=pred_vertices_mode,
                                                                        axes=torch.tensor([1., 0., 0.], device=device),
                                                                        angles=np.pi,
                                                                        translations=torch.zeros(3, device=device))
                # Rotating 90° about vertical axis for visualisation
                pred_vertices_rot90_mode = aa_rotate_translate_points_pytorch3d(points=pred_vertices_mode,
                                                                                axes=torch.tensor([0., 1., 0.], device=device),
                                                                                angles=-np.pi / 2.,
                                                                                translations=torch.zeros(3, device=device))
                pred_vertices_rot180_mode = aa_rotate_translate_points_pytorch3d(points=pred_vertices_rot90_mode,
                                                                                axes=torch.tensor([0., 1., 0.], device=device),
                                                                                angles=-np.pi / 2.,
                                                                                translations=torch.zeros(3, device=device))
                pred_vertices_rot270_mode = aa_rotate_translate_points_pytorch3d(points=pred_vertices_rot180_mode,
                                                                                axes=torch.tensor([0., 1., 0.], device=device),
                                                                                angles=-np.pi / 2.,
                                                                                translations=torch.zeros(3, device=device))

                pred_reposed_smpl_output_mean = smpl_model(betas=pred_shape_dist_var)
                pred_reposed_vertices_mean = pred_reposed_smpl_output_mean.vertices  # (1, 6890, 3)
                
                # Need to flip pred_vertices before projecting so that they project the right way up.
                pred_reposed_vertices_flipped_mean = aa_rotate_translate_points_pytorch3d(points=pred_reposed_vertices_mean,
                                                                                        axes=torch.tensor([1., 0., 0.], device=device),
                                                                                        angles=np.pi,
                                                                                        translations=torch.zeros(3, device=device))
                # Rotating 90° about vertical axis for visualisation
                pred_reposed_vertices_rot90_mean = aa_rotate_translate_points_pytorch3d(points=pred_reposed_vertices_flipped_mean,
                                                                                        axes=torch.tensor([0., 1., 0.], device=device),
                                                                                        angles=-np.pi / 2.,
                                                                                        translations=torch.zeros(3, device=device))

                # -------------------------------------- VISUALISATION --------------------------------------
                # Render visualisation outputs
                body_vis_output = body_vis_renderer(vertices=pred_vertices_mode,
                                                    cam_t=pred_cam_t,
                                                    textures=plain_texture,
                                                    orthographic_scale=pred_orthographic_scale,
                                                    lights_rgb_settings=lights_rgb_settings)
                body_vis_rgb = batch_add_rgb_background(backgrounds=cropped_for_proxy_rgb,
                                                        rgb=body_vis_output['rgb_images'].permute(0, 3, 1, 2).contiguous(),
                                                        seg=body_vis_output['iuv_images'][:, :, :, 0].round())
                body_vis_rgb = body_vis_rgb.cpu().detach().numpy()[0].transpose(1, 2, 0)

                body_vis_rgb_rot90 = body_vis_renderer(vertices=pred_vertices_rot90_mode,
                                                    cam_t=fixed_cam_t,
                                                    textures=plain_texture,
                                                    orthographic_scale=fixed_orthographic_scale,
                                                    lights_rgb_settings=lights_rgb_settings)['rgb_images'].cpu().detach().numpy()[0]
                body_vis_rgb_rot180 = body_vis_renderer(vertices=pred_vertices_rot180_mode,
                                                        cam_t=fixed_cam_t,
                                                        textures=plain_texture,
                                                        orthographic_scale=fixed_orthographic_scale,
                                                        lights_rgb_settings=lights_rgb_settings)['rgb_images'].cpu().detach().numpy()[0]
                body_vis_rgb_rot270 = body_vis_renderer(vertices=pred_vertices_rot270_mode,
                                                        textures=plain_texture,
                                                        cam_t=fixed_cam_t,
                                                        orthographic_scale=fixed_orthographic_scale,
                                                        lights_rgb_settings=lights_rgb_settings)['rgb_images'].cpu().detach().numpy()[0]

                # Reposed body visualisation
                reposed_body_vis_rgb = body_vis_renderer(vertices=pred_reposed_vertices_flipped_mean,
                                                        textures=plain_texture,
                                                        cam_t=fixed_cam_t,
                                                        orthographic_scale=fixed_orthographic_scale,
                                                        lights_rgb_settings=lights_rgb_settings)['rgb_images'].cpu().detach().numpy()[0]
                reposed_body_vis_rgb_rot90 = body_vis_renderer(vertices=pred_reposed_vertices_rot90_mean,
                                                            textures=plain_texture,
                                                            cam_t=fixed_cam_t,
                                                            orthographic_scale=fixed_orthographic_scale,
                                                            lights_rgb_settings=lights_rgb_settings)['rgb_images'].cpu().detach().numpy()[0]

                # Combine all visualisations
                combined_vis_rows = 2
                combined_vis_cols = 4
                combined_vis_fig = np.zeros((combined_vis_rows * visualise_wh, combined_vis_cols * visualise_wh, 3),
                                            dtype=body_vis_rgb.dtype)
                # Cropped input image
                combined_vis_fig[:visualise_wh, :visualise_wh] = cropped_for_proxy_rgb.cpu().detach().numpy()[0].transpose(1, 2, 0)

                # Proxy representation + 2D joints scatter + 2D joints confidences
                proxy_rep_input = proxy_rep_input[0].sum(dim=0).cpu().detach().numpy()
                proxy_rep_input = np.stack([proxy_rep_input]*3, axis=-1)  # single-channel to RGB
                proxy_rep_input = cv2.resize(proxy_rep_input, (visualise_wh, visualise_wh))
                for joint_num in range(cropped_for_proxy['joints2D'].shape[1]):
                    hor_coord = cropped_for_proxy['joints2D'][0, joint_num, 0].item() * visualise_wh / pose_shape_cfg.DATA.PROXY_REP_SIZE
                    ver_coord = cropped_for_proxy['joints2D'][0, joint_num, 1].item() * visualise_wh / pose_shape_cfg.DATA.PROXY_REP_SIZE
                    cv2.circle(proxy_rep_input,
                            (int(hor_coord), int(ver_coord)),
                            radius=3,
                            color=(255, 0, 0),
                            thickness=-1)
                    cv2.putText(proxy_rep_input,
                                str(joint_num),
                                (int(hor_coord + 4), int(ver_coord + 4)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), lineType=2)
                    cv2.putText(proxy_rep_input,
                                str(joint_num) + " {:.2f}".format(hrnet_output['joints2Dconfs'][joint_num].item()),
                                (10, 16 * (joint_num + 1)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), lineType=2)
                combined_vis_fig[visualise_wh:2*visualise_wh, :visualise_wh] = proxy_rep_input

                # Posed 3D body
                combined_vis_fig[:visualise_wh, visualise_wh:2*visualise_wh] = body_vis_rgb
                combined_vis_fig[visualise_wh:2*visualise_wh, visualise_wh:2*visualise_wh] = body_vis_rgb_rot90
                combined_vis_fig[:visualise_wh, 2*visualise_wh:3*visualise_wh] = body_vis_rgb_rot180
                combined_vis_fig[visualise_wh:2*visualise_wh, 2*visualise_wh:3*visualise_wh] = body_vis_rgb_rot270

                # T-pose 3D body
                combined_vis_fig[:visualise_wh, 3*visualise_wh:4*visualise_wh] = reposed_body_vis_rgb
                combined_vis_fig[visualise_wh:2*visualise_wh, 3*visualise_wh:4*visualise_wh] = reposed_body_vis_rgb_rot90
                vis_save_path = os.path.join(save_dir, image_fname)
                cv2.imwrite(os.path.splitext(vis_save_path)[0] + '_optimized.png', combined_vis_fig[:, :, ::-1] * 255)
                cv2.imwrite(os.path.splitext(vis_save_path)[0] + '_tpose_optimized.png', reposed_body_vis_rgb * 255)
                cv2.imwrite(os.path.splitext(vis_save_path)[0] + '_tpose_rot90_optimized.png', reposed_body_vis_rgb_rot90 * 255)

        # Save losses as png
        if visualize_losses:
            plt.plot(loss_history)
            plt.xlabel('Iterations')
            plt.ylabel('Loss')
            vis_save_path = os.path.join(save_dir, image_fname)
            plt.savefig(os.path.splitext(vis_save_path)[0] + '_losses.png')
            plt.clf()
            np.savetxt(os.path.splitext(vis_save_path)[0] + '_reposed_vertices.xyz', pred_reposed_vertices_mean.squeeze().detach().cpu().numpy(), delimiter=' ')
            np.savetxt(os.path.splitext(vis_save_path)[0] + '_vertices.xyz', pred_vertices_mode.squeeze().detach().cpu().numpy(), delimiter=' ')

    save_df = pd.DataFrame(save_df)
    if multiprocessing:
        return save_df
    else:
        save_df.to_pickle(os.path.join(save_dir, 'parameters_losses.pkl.tar'))
