import copy
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from smplx.lbs import batch_rodrigues
from tqdm import tqdm

from metrics.train_loss_and_metrics_tracker import TrainingLossesAndMetricsTracker

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


def gen_rotmat(rad, axis):
    rad = torch.tensor(rad, dtype=float)
    rad += 0.3 * torch.randn_like(rad)
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
                    model_save_dir,
                    logs_save_path,
                    vis_save_dir,
                    save_val_metrics=['PVE-SC', 'MPJPE-PA'],
                    checkpoint=None,
                    seed=None,
                    ):
    if seed is not None:
        torch.seed(seed)
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

    # Load checkpoint benchmarks if provided
    if False: # checkpoint is not None:
        # Resuming training - note that current model and optimiser state dicts are loaded out
        # of train function (should be in run file).
        current_epoch, best_epoch, best_model_wts, best_epoch_val_metrics = load_training_info_from_checkpoint(checkpoint,
                                                                                                               save_val_metrics)
        load_logs = True
    else:
        current_epoch = 0
        best_epoch = 0
        best_epoch_val_metrics = {}
        # metrics that decide whether to save model after each epoch or not
        for metric in save_val_metrics:
            best_epoch_val_metrics[metric] = np.inf
        best_model_wts = copy.deepcopy(pose_shape_model.state_dict())
        load_logs = False

    # Instantiate metrics tracker
    metrics_tracker = TrainingLossesAndMetricsTracker(metrics_to_track=metrics,
                                                      img_wh=pose_shape_cfg.DATA.PROXY_REP_SIZE,
                                                      log_save_path=logs_save_path,
                                                      load_logs=load_logs,
                                                      current_epoch=current_epoch)

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

    # Starting training loop
    current_loss_stage = 1
    for epoch in range(current_epoch, pose_shape_cfg.TRAIN.NUM_EPOCHS):
        print('\nEpoch {}/{}'.format(epoch, pose_shape_cfg.TRAIN.NUM_EPOCHS - 1))
        print('-' * 10)
        metrics_tracker.initialise_loss_metric_sums()

        if epoch >= pose_shape_cfg.LOSS.STAGE_CHANGE_EPOCH and current_loss_stage == 1:
            # Apply 2D samples losses + 3D mode losses + change weighting from this epoch onwards.
            criterion.loss_config = pose_shape_cfg.LOSS.STAGE2
            print('Stage 2 loss config:\n', criterion.loss_config)
            print('Sample on CPU:', pose_shape_cfg.LOSS.SAMPLE_ON_CPU)

            metrics_tracker.metrics_to_track.append('joints2Dsamples-L2E')
            print('Tracking metrics:', metrics_tracker.metrics_to_track)

            current_loss_stage = 2

        for split in ['val']:
            if split == 'train':
                print('Training.')
                pose_shape_model.train()
            else:
                print('Validation.')
                pose_shape_model.eval()

            for batch_num, samples_batch in enumerate(tqdm(dataloaders[split])):
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
                    #target_glob_rotmats = target_pose_rotmats[:, 0, :, :]
                    target_pose_rotmats = target_pose_rotmats[:, 1:, :, :]
                    
                    # # hard code rotation matrix multiplied
                    # front = torch.matmul(gen_rotmat(np.pi + 0.1 * np.random.randn(1), 'x'), gen_rotmat(0 + 0.1 * np.random.randn(1), 'y'))
                    # side = torch.matmul(gen_rotmat(np.pi + 0.1 * np.random.randn(1), 'x'), gen_rotmat(np.pi/2 + 0.1 * np.random.randn(1), 'y'))
                    # back = torch.matmul(gen_rotmat(np.pi + 0.1 * np.random.randn(1), 'x'), gen_rotmat(np.pi + 0.1 * np.random.randn(1), 'y'))
                    # target_glob_rotmats = torch.stack((front, side, back), dim=0).float().repeat(int(pose_shape_cfg.TRAIN.BATCH_SIZE/3), 1, 1).to(device)

                    # # Flipping pose targets such that they are right way up in 3D space - i.e. wrong way up when projected
                    # # Then pose predictions will also be right way up in 3D space - network doesn't need to learn to flip.
                    # _, target_glob_rotmats = aa_rotate_rotmats_pytorch3d(rotmats=target_glob_rotmats,
                    #                                                      angles=np.pi,
                    #                                                      axes=x_axis,
                    #                                                      rot_mult_order='post')
                    


                    target_glob_rotmats = torch.empty(0, 3, 3).float()
                    for _ in range(int(pose_shape_cfg.TRAIN.BATCH_SIZE/3)):
                        front = gen_rotmat(0, 'x') @ gen_rotmat(0, 'y') @ gen_rotmat(0, 'z')
                        side =  gen_rotmat(0, 'x') @ gen_rotmat(np.sign(np.random.randn(1)) * np.pi/2, 'y') @ gen_rotmat(0, 'z')
                        back =  gen_rotmat(0, 'x') @ gen_rotmat(np.pi, 'y') @ gen_rotmat(0, 'z')
                        target_glob_rotmats = torch.cat((target_glob_rotmats, front.unsqueeze(0), side.unsqueeze(0), back.unsqueeze(0)), dim=0)
                    target_glob_rotmats = target_glob_rotmats.float().to(device)

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

                        

                with torch.no_grad():
                    #############################################################
                    # ---------------------- FORWARD PASS -----------------------
                    #############################################################
                    pred_pose_F, pred_pose_U, pred_pose_S, pred_pose_V, pred_pose_rotmats_mode, \
                    pred_shape_dist, pred_glob, pred_cam_wp = pose_shape_model(proxy_rep_input)
                    # Pose F, U, V and rotmats_mode are (bs, 23, 3, 3) and Pose S is (bs, 23, 3)

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
                    
                    # import ipdb 
                    # ipdb.set_trace()

                    # -------------------------------------- VISUALISATION --------------------------------------

                    fixed_cam_t = torch.tensor([[0., -0.2, 2.5]], device=device)
                    fixed_orthographic_scale = torch.tensor([[0.95, 0.95]], device=device)

                    plain_texture = torch.ones(1, 1200, 800, 3, device=device).float() * 0.7
                    lights_rgb_settings = {'location': torch.tensor([[0., -0.8, -2.0]], device=device, dtype=torch.float32),
                           'ambient_color': 0.5 * torch.ones(1, 3, device=device, dtype=torch.float32),
                           'diffuse_color': 0.3 * torch.ones(1, 3, device=device, dtype=torch.float32),
                           'specular_color': torch.zeros(1, 3, device=device, dtype=torch.float32)}
                    
                    from matplotlib import pyplot as plt
                    per_vertex_3Dvar = torch.zeros((6890,))+0.1
                    vertex_var_norm = plt.Normalize(vmin=0.0, vmax=0.2, clip=True)
                    vertex_var_colours = plt.cm.jet(vertex_var_norm(per_vertex_3Dvar.numpy()))[:, :3]
                    vertex_var_colours = torch.from_numpy(vertex_var_colours[None, :, :]).expand(pose_shape_cfg.TRAIN.BATCH_SIZE, -1, -1).to(device).float()

                    visualise_wh=512

                    from renderers.pytorch3d_textured_renderer import TexturedIUVRenderer
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
                    combined_vis_cols = 4
                    combined_vis_fig = np.zeros((combined_vis_rows * visualise_wh, combined_vis_cols * visualise_wh, 3),
                                                dtype=body_vis_rgb.dtype)
                    # Cropped input image
                    # (1, 1)
                    combined_vis_fig[:visualise_wh, :visualise_wh] = cropped_for_proxy_rgb.cpu().detach().numpy()[0].transpose(1, 2, 0)

                    # Proxy representation + 2D joints scatter + 2D joints confidences
                    proxy_rep_input = proxy_rep_input[0].sum(dim=0).cpu().detach().numpy()
                    proxy_rep_input = np.stack([proxy_rep_input]*3, axis=-1)  # single-channel to RGB
                    import cv2
                    proxy_rep_input = cv2.resize(proxy_rep_input, (visualise_wh, visualise_wh))
                    # for joint_num in range(cropped_for_proxy['joints2D'].shape[1]):
                    #     hor_coord = cropped_for_proxy['joints2D'][0, joint_num, 0].item() * visualise_wh / pose_shape_cfg.DATA.PROXY_REP_SIZE
                    #     ver_coord = cropped_for_proxy['joints2D'][0, joint_num, 1].item() * visualise_wh / pose_shape_cfg.DATA.PROXY_REP_SIZE
                    #     cv2.circle(proxy_rep_input,
                    #             (int(hor_coord), int(ver_coord)),
                    #             radius=3,
                    #             color=(255, 0, 0),
                    #             thickness=-1)
                    #     cv2.putText(proxy_rep_input,
                    #                 str(joint_num),
                    #                 (int(hor_coord + 4), int(ver_coord + 4)),
                    #                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), lineType=2)
                    #     cv2.putText(proxy_rep_input,
                    #                 str(joint_num) + " {:.2f}".format(hrnet_output['joints2Dconfs'][joint_num].item()),
                    #                 (10, 16 * (joint_num + 1)),
                    #                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), lineType=2)
                    combined_vis_fig[visualise_wh:2*visualise_wh, :visualise_wh] = proxy_rep_input


                    # # Posed 3D body
                    
                    # combined_vis_fig[:visualise_wh, visualise_wh:2*visualise_wh] = body_vis_rgb

                    # # T-pose 3D body
                    # combined_vis_fig[:visualise_wh, 3*visualise_wh:4*visualise_wh] = reposed_body_vis_rgb
                    
                    # vis_save_path = os.path.join(val_vis_dir, 'val_result.png')
                    # cv2.imwrite(vis_save_path, combined_vis_fig[:, :, ::-1] * 255)




                    for i in range(3):
                        baseline_tovis = cv2.resize(body_vis_rgb[i], (visualise_wh, visualise_wh))
                        baseline_tovis_2 = cv2.resize(body_vis_rgb[i+3], (visualise_wh, visualise_wh))
                        combined_vis_fig[:visualise_wh, i * visualise_wh: (i+1)*visualise_wh] = baseline_tovis
                        combined_vis_fig[visualise_wh: 2 * visualise_wh, i * visualise_wh: (i+1)*visualise_wh] = baseline_tovis_2

                    vis_save_path = os.path.join(vis_save_dir, 'mult_result.png')
                    
                    cv2.imwrite(vis_save_path, combined_vis_fig[:, :, ::-1] * 255)



                    import sys
                    sys.exit()