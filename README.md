# frag-template-reg

## Quick Start: Conda Environment And Training

Рекомендуемая версия Python для проекта: `python=3.10`.

Если окружение еще не создано:

```bash
conda create -n fracs python=3.10 -y
conda activate fracs
```

В окружении должен быть установлен PyTorch:

```text
torch==2.9.1+cu130
```

Установка PyTorch 2.9.1 под CUDA 13.0:

```bash
pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 \
  --index-url https://download.pytorch.org/whl/cu130
```


Проверьте версию перед установкой остальных зависимостей:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Ожидаемо:

```text
2.9.1+cu130 True
```

`requirements.txt` намеренно не устанавливает `torch`, `torchvision` и `torchaudio`, чтобы не заменить PyTorch из conda-окружения.

Установка зависимостей и пакета:

```bash
pip install -r requirements.txt
pip install -e .
```

Проверка датасета:

```bash
python tools/check_dataset.py configs/dgcnn_profile_axisym_y.py
python tools/oracle_axisymmetric_registration_dataset.py configs/dgcnn_profile_axisym_y.py
```

Запуск обучения профильной осесимметричной модели:

```bash
python scripts/train.py configs/dgcnn_profile_axisym_y.py
```

Результаты обучения сохраняются в:

```text
work_dirs/dgcnn_profile_axisym_y/DD_MM_YYYY/
```

Первая рабочая версия pipeline для регистрации видимой оболочки фрагмента к цифровому двойнику исходного объекта.

Pipeline:

```text
visible shell points_C
  -> DGCNN correspondence model
  -> pred_points_O
  -> Kabsch
  -> T_C_from_O
  -> evaluation metrics
```

Модель получает нормализованные точки видимой оболочки в СК камеры (`points_C`) и предсказывает canonical coordinates в СК цифрового двойника (`pred_points_O`). Поза цифрового двойника в СК камеры восстанавливается через Kabsch по `pred_points_O -> points_C_orig`.

Для строго осесимметричного объекта доступна профильная постановка. Вместо неидентифицируемого абсолютного азимута вокруг оси симметрии модель предсказывает координаты профиля:

```text
visible shell points_C
  -> DGCNNProfile
  -> pred_profile_O = (r, axial)
  -> axisymmetric solver
  -> T_C_from_O / SO(2)
  -> symmetry-aware metrics
```

Для рабочей оси `axis="y"`:

```text
r = sqrt(x^2 + z^2)
axial = y
```

Есть две профильные модели:

- `DGCNNProfile` — базовая per-point profile regression.
- `DGCNNProfileGlobal` — добавляет global feature всего видимого shell-фрагмента, конкатенирует его обратно к каждой точке и может предсказывать `pred_axis_C`.

`pred_axis_error_deg` — ошибка axis head до solver-а. `axis_error_deg` — ошибка итоговой позы после axisymmetric solver.

## Dataset

Ожидается структура:

```text
dataset_root/
  train/
    scene_000000/
      visible_points/frame_000000.npz
      gt_annotations.json
  val/
  test/
```

В `visible_points/frame_XXXXXX.npz` используются ключи `fragment_id`, `points_C`, `points_O`, `shell_indices`; если `shell_indices` отсутствует или пустой, применяется fallback `surface_label == 0`.

Путь к датасету задается в config-переменной `dataset_root`. Для локального запуска замените значение на свой путь:

```text
/path/to/fragment_template_registration/dataset
```

## Environment

Используется уже созданное conda-окружение:

```bash
conda activate fracs
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

`requirements.txt` намеренно не содержит `torch`, `torchvision`, `torchaudio`: PyTorch `2.9.1+cu130` должен оставаться из окружения `fracs`.

## Install

```bash
pip install -e .
```

## Dataset Checks

```bash
python tools/check_dataset.py configs/dgcnn_xyz_symmetry.py
python tools/oracle_kabsch_dataset.py configs/dgcnn_xyz_symmetry.py
python tools/oracle_axisymmetric_registration_dataset.py configs/dgcnn_profile_axisym_y.py
```

Oracle Kabsch использует GT `points_O -> points_C_orig` и должен показывать малый residual.
Axisymmetric oracle использует GT `profile_O -> points_C_orig` и проверяет профильный solver без участия модели.

## Units

Единицы координат задаются один раз в config:

```python
units = dict(
    dataset_coord_unit="m",
    coord_unit="mm",
    coord_scale=1000.0,
)
```

`coord_scale` умножает координатные величины из единиц датасета в рабочие единицы логов и loss. При настройке выше датасет остается в метрах для геометрии и Kabsch, а loss и distance metrics пишутся в миллиметрах. Поэтому `smooth_l1_beta=10.0` означает `10 мм`.

## Train

```bash
python scripts/train.py configs/dgcnn_xyz_symmetry.py
python scripts/train.py configs/dgcnn_profile_axisym_y.py
python scripts/train.py configs/dgcnn_profile_global_axisym_y.py
```

Артефакты сохраняются в dated run-папку, например `work_dirs/dgcnn_xyz_symmetry/DD_MM_YYYY/`:

```text
checkpoints/latest.pth
checkpoints/best.pth
checkpoints/epoch_0005.pth
train.log
metrics.jsonl
tensorboard/
```

Частота валидации, сохранения checkpoints и логирование задаются в `train_cfg`:

```python
train_cfg = dict(
    max_epochs=100,
    val_interval=1,
    work_dir="work_dirs/dgcnn_xyz_symmetry",
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
```

При `date_subdir=True` фактическая папка запуска создается как `work_dir/DD_MM_YYYY`, например `work_dirs/dgcnn_profile_global_axisym_y/26_05_2026`. Формат можно переопределить через `date_subdir_format`. Дата вычисляется один раз в начале запуска, поэтому если обучение идет несколько дней, папка не меняется. При resume из checkpoint train script по умолчанию продолжает писать в папку этого checkpoint; это можно отключить через `resume_work_dir=False`.

Если `checkpoint_dir` или `tensorboard_dir` заданы коротким относительным именем, например `"checkpoints"`, путь считается относительно фактической run-папки. Абсолютный путь используется как есть.

`metrics.jsonl` сохраняет `duration_sec`; для train также пишутся `train/epoch_time_sec`, `train/seconds_per_batch`, `train/optimizer_steps`, `train/gradient_accumulation_steps`, а для val — `val/eval_time_sec` и `val/seconds_per_batch`.

Для уменьшения пика памяти можно уменьшить физический `batch_size` и сохранить effective batch через accumulation:

```python
train_dataloader = dict(batch_size=16, ...)
train_cfg = dict(gradient_accumulation_steps=4, ...)
```

## Evaluate

```bash
python scripts/evaluate.py configs/dgcnn_xyz_symmetry.py work_dirs/dgcnn_xyz_symmetry/DD_MM_YYYY/checkpoints/best.pth
python scripts/evaluate.py configs/dgcnn_profile_axisym_y.py work_dirs/dgcnn_profile_axisym_y/DD_MM_YYYY/best.pth
python scripts/evaluate.py configs/dgcnn_profile_global_axisym_y.py work_dirs/dgcnn_profile_global_axisym_y/DD_MM_YYYY/best.pth
```

TensorBoard:

```bash
tensorboard --logdir work_dirs/dgcnn_xyz_symmetry/DD_MM_YYYY/tensorboard
```

Основные метрики для осесимметричного объекта:

- `residual_rmse`
- `translation_error`
- `axis_error_deg`
- `rz_l1`

Координатные метрики `coord_l1`, `rz_l1`, `residual_*`, `translation_error` логируются в `units["coord_unit"]`; `rotation_error_deg` и `axis_error_deg` логируются в градусах. `rotation_error_deg` является диагностической метрикой для осесимметричного объекта.

Для профильной модели основные метрики:

- `profile_r_l1_mm`
- `profile_axial_l1_mm`
- `profile_l1_mm`
- `profile_residual_rmse_mm`
- `translation_error_mm`
- `axis_error_deg`
- `pred_axis_error_deg` при включенном axis head

`rotation_error_deg_diagnostic` остается диагностикой: вращение вокруг оси симметрии не является главным критерием качества.

Дополнительно evaluator пишет диагностику axial-компоненты:

- `axial_error_mean_mm` — средний signed bias по axial;
- `axial_error_std_mm` — разброс axial-ошибки;
- `axial_corr` — корреляция предсказанного и GT axial;
- `gt_axial_span_mm` / `pred_axial_span_mm` — покрытие фрагмента вдоль оси;
- `radial_span_mm` — покрытие по радиусу;
- `num_points` — число валидных точек в sample.

Также summary группируется по `gt_axial_span_mm` на `small`, `medium`, `large` tertile-группы. Метрики в группах имеют вид `small/profile_residual_rmse_mm`, `medium/axis_error_deg`, `large/translation_error_mm`.

Профильный evaluator поддерживает диагностические режимы в `eval_cfg.profile_eval_modes`:

- `gt_profile_auto_init` — GT profile + auto init, проверяет solver/objective;
- `pred_profile_gt_init` — pred profile + GT init, отделяет ошибку профиля от ошибки auto-init;
- `pred_profile_auto_init` — реальный режим inference.

По умолчанию config включает все три режима для диагностики. Для ускорения validation можно оставить только:

```python
profile_eval_modes=["pred_profile_auto_init"]
```

`ProfileLoss` поддерживает веса:

```python
radial_weight=1.0
axial_weight=2.0
axial_mean_weight=0.1
axial_std_weight=0.1
axial_range_weight=0.0
axial_pairwise_weight=0.0
axial_pairwise_mode="sampled"
axial_pairwise_num_pairs=8192
axis_loss_weight=0.0
```

`axial_pairwise_mode="sampled"` считает pairwise loss не по всем `N x N` парам, а по случайной подвыборке пар. Это полезно для больших `batch_size` и `num_points`, где полный pairwise loss быстро становится главным потребителем памяти.

Optional axis head включается в модели через:

```python
model = dict(..., output_axis=True)
loss = dict(..., axis_loss_weight=0.1)
eval_cfg = dict(solver=dict(..., use_pred_axis_init=True))
```

Это добавляет `pred_axis_C [B,3]`, обучает его через `1 - abs(dot)` с GT-осью и может использовать предсказанную ось как init для axisymmetric solver. Старые checkpoints без axis head лучше запускать с `output_axis=False`.

## Global-Context Profile Model

`DGCNNProfileGlobal` использует те же EdgeConv-блоки, но после локальных признаков делает max/mean pooling по всем точкам фрагмента. Полученный global feature повторяется для каждой точки и подается в profile head, поэтому prediction `(r, axial)` получает контекст всей видимой оболочки.

Дополнительные головы:

- `pred_axis_C [B,3]` — направление оси симметрии в СК камеры;
- `pred_axial_stats` — optional global stats head для будущих экспериментов.

В новом config `use_pred_axis_init=False` по умолчанию: сначала стоит смотреть на `pred_axis_error_deg`, и только когда axis head стабильно обучился, включать init solver-а от предсказанной оси.

```bash
python scripts/train.py configs/dgcnn_profile_global_axisym_y.py
python scripts/evaluate.py configs/dgcnn_profile_global_axisym_y.py work_dirs/dgcnn_profile_global_axisym_y/DD_MM_YYYY/best.pth
python scripts/debug_profile_global_batch.py configs/dgcnn_profile_global_axisym_y.py
```

Полезная отладка профильной ветки:

```bash
python scripts/debug_profile_batch.py configs/dgcnn_profile_axisym_y.py
python scripts/debug_profile_batch.py configs/dgcnn_profile_axisym_y.py --checkpoint work_dirs/dgcnn_profile_axisym_y/DD_MM_YYYY/best.pth
python scripts/debug_profile_global_batch.py configs/dgcnn_profile_global_axisym_y.py
```

## One-Scene Overfit Diagnostic

`configs/dgcnn_profile_global_axisym_y_one_scene.py` нужен только для диагностики переобучения. Он обучает и валидирует `DGCNNProfileGlobal` на одной и той же сцене из `train` split. Это не честная оценка generalization performance.

```bash
python tools/check_dataset.py configs/dgcnn_profile_global_axisym_y_one_scene.py
python scripts/train.py configs/dgcnn_profile_global_axisym_y_one_scene.py
python scripts/evaluate.py configs/dgcnn_profile_global_axisym_y_one_scene.py work_dirs/dgcnn_profile_global_axisym_y_one_scene/DD_MM_YYYY/best.pth
python scripts/infer_profile_global.py configs/dgcnn_profile_global_axisym_y_one_scene.py work_dirs/dgcnn_profile_global_axisym_y_one_scene/DD_MM_YYYY/best.pth
python scripts/visualize_frame_object_placements.py configs/dgcnn_profile_global_axisym_y_one_scene.py work_dirs/dgcnn_profile_global_axisym_y_one_scene/DD_MM_YYYY/best.pth
```

Если на одной сцене `profile_axial_l1_mm`, `axial_corr`, `profile_residual_rmse_mm`, `axis_error_deg` и `translation_error_mm` становятся хорошими, а на обычном val/test остаются плохими, проблема скорее в обобщении и разнообразии данных. Если даже one-scene overfit не сходится, стоит искать проблему в архитектуре, loss, solver, нормализации или формировании sample.

При `return_per_sample=True` evaluation сохраняет отдельный файл:

```text
per_sample_metrics_val.json
```

В нем для каждого sample есть `scene_id`, `frame_id`, `fragment_id`, профильные метрики, `pred_axis_error_deg`, `axis_error_deg` и `translation_error_mm`.

`infer_profile_global.py` сохраняет визуальную диагностику в `work_dirs/dgcnn_profile_global_axisym_y_one_scene/DD_MM_YYYY/inference_vis/`:

- `profile_scatter.svg` — GT vs predicted profile в координатах `(axial, r)`;
- `camera_observed_vs_pred_fit.ply` — видимые точки камеры и fitted points после solver-а;
- `camera_profile_error.ply` — точки камеры, окрашенные по profile error;
- `object_gt_vs_pred_lifted.ply` — GT canonical points и lifted predicted profile в СК объекта;
- `camera_axes_gt_green_pred_red.ply` — GT axis зеленым, predicted axis красным;
- `points_profile.csv` и `summary.json` — численные значения для sample.

`visualize_frame_object_placements.py` собирает все фрагменты одного кадра в СК цифрового двойника и сохраняет в `frame_object_placements/`:

- `object_fragments_pred.ply` — честное inference-размещение через `T_pred^{-1}`;
- `object_fragments_gt.ply` — GT canonical placement;
- `object_fragments_pred_yaw_aligned_to_gt.ply` — pred placement с yaw вокруг оси, подогнанным к GT только для визуального сравнения;
- `object_fragments_pred_profile_error.ply` — pred placement, окрашенный по profile error;
- `object_axes_xyz_rgb.ply` — оси СК объекта.

Для осесимметричной профильной модели абсолютный yaw вокруг оси Y неидентифицируем, поэтому `object_fragments_pred.ply` может быть провернут вокруг Y. Это нормально для профильной постановки; для сравнения формы используйте `object_fragments_pred_yaw_aligned_to_gt.ply`.

## Notes

- DGCNN/EdgeConv реализован на чистом PyTorch.
- `torch_geometric` не требуется.
- Датасет на диске только читается.
- `CorrespondenceLoss` поддерживает `mode="xyz"`, `mode="rz"` и смешивание через `xyz_weight`/`rz_weight`.
- Для `mode="xyz"` итоговый `loss` всегда равен `loss_xyz`; для `mode="rz"` всегда равен `loss_rz`. Веса `xyz_weight` и `rz_weight` используются только при `mode="mixed"`.
- `ProfileLoss` обучает `DGCNNProfile` по `profile_O = (r, axial)` и не использует полный XYZ как основную цель.
