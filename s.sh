


# python run_val_vis.py \
#     --ckpt /scratches/kyuban/cq244/multin_v2/experiments/multin_002/saved_models/epoch_090.tar \
#     -S ./output/mult_002_e090_noaugment/ \
#     -E /scratches/kyuban/cq244/multin_v2/experiments/multin_002/ \
#     --gpu 0




# v3 no constraints
# python run_predict_v3.py \
#     --image_dir /scratches/nazgul/cq244/hkpd_depth/my_data/images/ \
#     --pose_shape_weights /scratches/kyuban/cq244/multin_v2/experiments/multin_v3_2/saved_models/epoch_016.tar \
#     --save_dir ./output/me_v3_2_001_e016/ \
#     --gpu 2


#v3 
python run_predict_v3.py \
    --image_dir /scratches/nazgul/cq244/hkpd_depth/my_data/images/ \
    --pose_shape_weights /scratches/kyuban/cq244/multin_v2/experiments/multin_v3_001/saved_models/epoch_299.tar \
    --save_dir ./output/me_v3_001_e299/ \
    --gpu 0

# # v4 silhouette loss
# python run_predict_v4.py \
#     --image_dir /scratches/nazgul/cq244/hkpd_depth/my_data/images/ \
#     --pose_shape_weights /scratches/kyuban/cq244/multin_v2/experiments/multin_v4_002/saved_models/epoch_299.tar \
#     --save_dir ./output/me_v4_002_e299/ \
#     --gpu 0