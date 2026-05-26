dataset_root = "/home/nikita/data_generator/generation_dataset/generation_synthetic/output/fragment_template_registration/differBig/2026-05-17"

seed = 42

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
    ),
)

model = dict(
    type="DGCNNCorrespondence",
    in_channels=3,
    k=20,
    emb_dims=256,
    output_confidence=True,
)

loss = dict(
    type="CorrespondenceLoss",
    mode="xyz",
    xyz_weight=1.0,
    rz_weight=0.0,
    smooth_l1_beta=10.0,
    axis="z",
)

optim = dict(
    type="AdamW",
    lr=1e-3,
    weight_decay=1e-4,
)

train_cfg = dict(
    max_epochs=100,
    val_interval=1,
    work_dir="work_dirs/dgcnn_xyz",
    date_subdir=True,
    checkpoint_dir="checkpoints",
    checkpoint_interval=5,
    gradient_accumulation_steps=1,
    save_latest=True,
    save_best=True,
    metric_for_best="residual_rmse",
    json_log=True,
    tensorboard_log=True,
    tensorboard_dir="tensorboard",
)
