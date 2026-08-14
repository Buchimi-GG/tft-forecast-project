from pytorch_forecasting import TemporalFusionTransformer
import pandas as pd

def main():
    model = TemporalFusionTransformer.load_from_checkpoint("tft_baseline.ckpt")
    df = pd.read_csv("https://raw.githubusercontent.com/jdb78/pytorch-forecasting/master/data/stallion.csv")
    pred = model.predict(df, mode="prediction")
    print(pred[:10])

if __name__ == "__main__":
    main()