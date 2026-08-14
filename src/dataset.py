import pandas as pd
from pytorch_forecasting import TimeSeriesDataSet

def get_datasets(raw_df:pd.DataFrame):
    max_encoder_length = 36
    max_prediction_length = 12

    training = TimeSeriesDataSet(
        raw_df,
        time_idx="time_idx",
        target="volume",
        group_ids=["agency", "sku"],
        min_encoder_length=max_encoder_length // 2,
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        static_categoricals=["agency", "sku"],
        static_reals=["avg_population"],
    )
    validation = TimeSeriesDataSet.from_dataset(training, raw_df, predict=True, stop_randomization=True)
    return training, validation