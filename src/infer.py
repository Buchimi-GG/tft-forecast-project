from pytorch_forecasting import TemporalFusionTransformer
import pandas as pd
import numpy as np
from pytorch_forecasting.data import TimeSeriesDataSet
from pytorch_forecasting.metrics import QuantileLoss
from lightning.pytorch import Trainer


def main():
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
    val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size * 10, num_workers=0)

    # 加载已经训练好的权重，不再重新训练
    model = TemporalFusionTransformer.load_from_checkpoint(
        "./checkpoints/epoch=2-step=360.ckpt"
    )

    pred = model.predict(val_dataloader, mode="prediction")
    print("前10条预测结果：")
    print(pred[:10])

    # 新增：tensor转numpy，保存预测结果到csv
    pred_np = pred.numpy()
    print("预测数组shape：", pred_np.shape)
    pd.DataFrame(pred_np).to_csv("predict_result.csv", index=False)
    print("预测结果已经保存到 predict_result.csv")


if __name__ == "__main__":
    main()