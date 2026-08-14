import numpy as np
import pandas as pd
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss
# 关键修复：改用新包名 lightning.pytorch
from lightning.pytorch import Trainer

np.random.seed(42)
n_samples = 10000
data = pd.DataFrame({
    "time_idx": np.tile(np.arange(100), 100),
    "group_id": np.repeat(np.arange(100), 100).astype(str),
    "target": np.random.randn(n_samples).cumsum(),
    "feature_1": np.random.randn(n_samples),
    "feature_2": np.random.randn(n_samples)
})

max_encoder_length = 24
max_prediction_length = 12
training_cutoff = data["time_idx"].max() - max_prediction_length

training = TimeSeriesDataSet(
    data[lambda x: x.time_idx <= training_cutoff],
    time_idx="time_idx",
    target="target",
    group_ids=["group_id"],
    min_encoder_length=max_encoder_length // 2,
    max_encoder_length=max_encoder_length,
    max_prediction_length=max_prediction_length,
    static_categoricals=["group_id"],
    time_varying_known_categoricals=[],
    time_varying_known_reals=["feature_1", "feature_2"],
    time_varying_unknown_categoricals=[],
    time_varying_unknown_reals=["target"],
    add_relative_time_idx=True,
    add_target_scales=True,
    add_encoder_length=True,
)

validation = TimeSeriesDataSet.from_dataset(training, data, predict=True, stop_randomization=True)

batch_size = 64
train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size * 10, num_workers=0)

tft = TemporalFusionTransformer.from_dataset(
    training,
    learning_rate=0.03,
    hidden_size=16,
    attention_head_size=2,
    dropout=0.1,
    hidden_continuous_size=8,
    output_size=7,
    loss=QuantileLoss(),
    log_interval=10,
    reduce_on_plateau_patience=4
)

trainer = Trainer(
    max_epochs=3,
    accelerator="auto",
    logger=False
)

trainer.fit(
    model=tft,
    train_dataloaders=train_dataloader,
    val_dataloaders=val_dataloader
)