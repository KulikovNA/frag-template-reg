dataset_root = "/home/nikita/data_generator/generation_dataset/generation_synthetic/output/fragment_template_registration/differBig/2026-05-17"

seed = 42

axis = "y"

units = dict(
    dataset_coord_unit="m",
    coord_unit="mm",
    coord_scale=1000.0,
)

train_dataloader = dict(
    batch_size=8,
    num_workers=4,
    dataset=dict(
        type="FragmentRegistrationDataset",
        dataset_root=dataset_root,
        split="train",
        num_points=1024,
        min_shell_points=32,
        normalize_input=True,
        use_normals=False,
        random_sample=True,
        repeat_if_less=True,
        axis=axis,
        return_profile=True,
    ),
)

val_dataloader = dict(
    batch_size=8,
    num_workers=4,
    dataset=dict(
        type="FragmentRegistrationDataset",
        dataset_root=dataset_root,
        split="val",
        num_points=1024,
        min_shell_points=32,
        normalize_input=True,
        use_normals=False,
        random_sample=False,
        repeat_if_less=True,
        axis=axis,
        return_profile=True,
    ),
)

test_dataloader = dict(
    batch_size=8,
    num_workers=4,
    dataset=dict(
        type="FragmentRegistrationDataset",
        dataset_root=dataset_root,
        split="test",
        num_points=1024,
        min_shell_points=32,
        normalize_input=True,
        use_normals=False,
        random_sample=False,
        repeat_if_less=True,
        axis=axis,
        return_profile=True,
    ),
)

model = dict(
    type="DGCNNProfileGlobal",
    in_channels=3,
    k=20,
    emb_dims=512,
    global_dims=512,
    output_confidence=True,
    output_axis=True,
    output_axial_stats=False,
    use_mean_pool=True,
    use_max_pool=True,
)

loss = dict(
    type="ProfileLoss",
    smooth_l1_beta=10.0,
    coord_scale=1000.0,
    axis=axis,
    radial_weight=1.0,
    axial_weight=2.0,
    axial_mean_weight=0.0,
    axial_std_weight=0.0,
    axial_range_weight=0.0,
    axial_pairwise_weight=0.2,
    axial_pairwise_mode="sampled",
    axial_pairwise_num_pairs=8192,
    axis_loss_weight=0.2,
)

optim = dict(
    type="AdamW",
    lr=1e-3,
    weight_decay=1e-4,
)

train_cfg = dict(
    max_epochs=200,
    val_interval=5,
    work_dir="work_dirs/dgcnn_profile_global_axisym_y",
    date_subdir=True,
    checkpoint_interval=5,
    gradient_accumulation_steps=1,
    save_latest=True,
    save_best=True,
    metric_for_best="profile_residual_rmse_mm",
    json_log=True,
    tensorboard_log=True,
    tensorboard_dir="tensorboard",
)

eval_cfg = dict(
    mode="axisymmetric_profile",
    axis=axis,
    solver=dict(
        num_iters=200,
        lr=1e-2,
        huber_beta=0.01,
        init_mode="auto",
        use_pred_axis_init=False,
    ),
    profile_eval_modes=[
        "pred_profile_auto_init",
        "gt_profile_auto_init",
        "pred_profile_gt_init",
    ],
    return_per_sample=False,
    coord_scale=1000.0,
    coord_unit="mm",
)
