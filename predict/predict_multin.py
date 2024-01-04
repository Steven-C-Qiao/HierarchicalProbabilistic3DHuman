import os
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
from smplx.lbs import batch_rodrigues

from predict.predict_hrnet import predict_hrnet

from renderers.pytorch3d_textured_renderer import TexturedIUVRenderer

from utils.image_utils import batch_add_rgb_background, batch_crop_pytorch_affine, batch_crop_opencv_affine
from utils.label_conversions import convert_2Djoints_to_gaussian_heatmaps_torch
from utils.rigid_transform_utils import rot6d_to_rotmat, aa_rotate_translate_points_pytorch3d
from utils.sampling_utils import compute_vertex_uncertainties_by_poseMF_shapeGaussian_sampling, joints2D_error_sorted_verts_sampling

def gen_rotmat(rad, axis):
    rad = torch.tensor(rad, dtype=float)
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



def predict_poseMF_shapeGaussian_net(pose_shape_model,
                                     pose_shape_cfg,
                                     smpl_model,
                                     hrnet_model,
                                     hrnet_cfg,
                                     edge_detect_model,
                                     device,
                                     image_dir,
                                     save_dir,
                                     object_detect_model=None,
                                     joints2Dvisib_threshold=0.75,
                                     visualise_wh=512,
                                     visualise_uncropped=True,
                                     visualise_samples=False):
    """
    Predictor for SingleInputKinematicPoseMFShapeGaussianwithGlobCam on unseen test data.
    Input --> ResNet --> image features --> FC layers --> MF over pose and Diagonal Gaussian over shape.
    Also get cam and glob separately to distribution predictor.
    Pose predictions follow the kinematic chain.
    """
    # Setting up body visualisation renderer

    plain_texture = torch.ones(1, 1200, 800, 3, device=device).float() * 0.7
    lights_rgb_settings = {'location': torch.tensor([[0., -0.8, -2.0]], device=device, dtype=torch.float32),
                           'ambient_color': 0.5 * torch.ones(1, 3, device=device, dtype=torch.float32),
                           'diffuse_color': 0.3 * torch.ones(1, 3, device=device, dtype=torch.float32),
                           'specular_color': torch.zeros(1, 3, device=device, dtype=torch.float32)}
    fixed_cam_t = torch.tensor([[0., -0.2, 2.5]], device=device)
    fixed_orthographic_scale = torch.tensor([[0.95, 0.95]], device=device)

    hrnet_model.eval()
    pose_shape_model.eval()
    if object_detect_model is not None:
        object_detect_model.eval()

    batch_size=3

    # Combine all visualisations
    combined_vis_rows = 2
    combined_vis_cols = 3
    combined_vis_fig = np.zeros((combined_vis_rows * visualise_wh, combined_vis_cols * visualise_wh, 3),
                                dtype=np.float32) #body_vis_rgb.dtype)
    
    proxy_all_input = torch.empty([0, 18, 256, 256]).to(device)
    proxy_all_rgb = torch.empty([0, 3, 256, 256]).to(device)
    i=0
    for image_fname in tqdm(['front_Tpose.jpg', 'side_Apose.jpg', 'back_Apose.jpg']):#[f for f in os.listdir(image_dir) if f.endswith(('front_Apose.jpg', 'side_Apose.jpg', 'back_Apose.jpg'))]):
        with torch.no_grad():
            # ------------------------- INPUT LOADING AND PROXY REPRESENTATION GENERATION -------------------------
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
            proxy_all_input = torch.cat([proxy_all_input, proxy_rep_input], dim=0).float()


            proxy_tosave = proxy_rep_input[0].sum(dim=0).cpu().detach().numpy()
            proxy_tosave = np.stack([proxy_tosave]*3, axis=-1)  # single-channel to RGB
            proxy_tosave = cv2.resize(proxy_tosave, (visualise_wh, visualise_wh))

            # Cropped input image
            proxy_all_rgb = torch.cat([proxy_all_rgb, cropped_for_proxy['rgb']], dim=0).float()

            #combined_vis_fig[:visualise_wh, i * visualise_wh:(i+1) * visualise_wh] = cropped_for_proxy_rgb.cpu().detach().numpy()[0].transpose(1, 2, 0)
            combined_vis_fig[visualise_wh:2*visualise_wh, i * visualise_wh:(i+1) * visualise_wh] = proxy_tosave
            i = i+1
    

    vis_save_path = os.path.join(save_dir, 'intermediate.png')
    cv2.imwrite(vis_save_path, combined_vis_fig[:, :, ::-1] * 255)


    # ------------------------------- POSE AND SHAPE DISTRIBUTION PREDICTION -------------------------------
    proxy_all_input = proxy_all_input.repeat(2, 1, 1, 1)
    proxy_all_rgb = proxy_all_rgb.repeat(2, 1, 1, 1)
    print(proxy_all_input.shape)
    
    pred_pose_F, pred_pose_U, pred_pose_S, pred_pose_V, pred_pose_rotmats_mode, \
    pred_shape_dist, pred_glob, pred_cam_wp = pose_shape_model(proxy_all_input)
    # Pose F, U, V and rotmats_mode are (bsize, 23, 3, 3) and Pose S is (bsize, 23, 3)

    # pred_glob_rotmats = rot6d_to_rotmat(pred_glob[0])  # (N, 3, 3)
    # pred_glob_rotmats = torch.cat([pred_glob_rotmats, rot6d_to_rotmat(pred_glob[1])], dim=0)
    # pred_glob_rotmats = torch.cat([pred_glob_rotmats, rot6d_to_rotmat(pred_glob[2])], dim=0)

    pred_glob_rotmats = rot6d_to_rotmat(pred_glob) 

    # import ipdb 
    # ipdb.set_trace()

    pred_smpl_output_mode = smpl_model(body_pose=pred_pose_rotmats_mode,
                                        global_orient=pred_glob_rotmats.unsqueeze(1),
                                        betas=pred_shape_dist.loc,
                                        pose2rot=False)
    
    pred_vertices_mode = pred_smpl_output_mode.vertices  # (N, 6890, 3)


    # Need to flip pred_vertices before projecting so that they project the right way up.
    pred_vertices_mode = aa_rotate_translate_points_pytorch3d(points=pred_vertices_mode,
                                                                axes=torch.tensor([1., 0., 0.], device=device),
                                                                angles=np.pi,
                                                                translations=torch.zeros(3, device=device))

    orthographic_scale = pred_cam_wp[:, [0, 0]]
    cam_t = torch.cat([pred_cam_wp[:, 1:],
                        torch.ones(pred_cam_wp.shape[0], 1, device=device).float() * 2.5],
                        dim=-1)

    per_vertex_3Dvar = torch.zeros((6890,))+0.1
    vertex_var_norm = plt.Normalize(vmin=0.0, vmax=0.2, clip=True)
    vertex_var_colours = plt.cm.jet(vertex_var_norm(per_vertex_3Dvar.numpy()))[:, :3]
    vertex_var_colours = torch.from_numpy(vertex_var_colours[None, :, :]).expand(6, -1, -1).to(device).float()

    body_vis_renderer = TexturedIUVRenderer(device=device,
                                        batch_size=6,
                                        img_wh=visualise_wh,
                                        projection_type='orthographic',
                                        render_rgb=True,
                                        bin_size=32)
    
    print(orthographic_scale)
    print(cam_t)
    
    # Render visualisation outputs
    body_vis_output = body_vis_renderer(vertices=pred_vertices_mode,
                                        cam_t=cam_t,
                                        orthographic_scale=orthographic_scale,
                                        lights_rgb_settings=lights_rgb_settings,
                                        verts_features=vertex_var_colours)
    
    proxy_all_rgb = torch.nn.functional.interpolate(proxy_all_rgb,
                                                            size=(visualise_wh, visualise_wh),
                                                            mode='bilinear',
                                                            align_corners=False)
    
    # proxy_back = proxy_all_input.sum(dim=1)
    # proxy_back = torch.stack([proxy_back]*3, axis=-1).permute(0, 3, 1, 2)  # single-channel to RGB
    # print(proxy_back.shape)
    # proxy_back = torch.nn.functional.interpolate(proxy_back,
    #                                                 size=(visualise_wh, visualise_wh),
    #                                                 mode='bilinear',
    #                                                 align_corners=False)

    body_vis_rgb = batch_add_rgb_background(backgrounds=proxy_all_rgb, #proxy_back,
                                            rgb=body_vis_output['rgb_images'].permute(0, 3, 1, 2).contiguous(),
                                            seg=body_vis_output['iuv_images'][:, :, :, 0].round())
    body_vis_rgb = body_vis_rgb.cpu().detach().numpy().transpose(0, 2, 3, 1)

    for i in range(6):
        to_vis = body_vis_rgb[i]
        
        save_path = os.path.join(save_dir, str(i) + '.png')
        cv2.imwrite(save_path, to_vis[:, :, ::-1] * 255)
