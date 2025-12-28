import json
import numpy as np
import torch
import gradio as gr

from model import LSTMForecast


class MinMaxScaler1D:
    def __init__(self, min_, max_):
        self.min_ = float(min_)
        self.max_ = float(max_)
        if self.max_ - self.min_ < 1e-12:
            self.max_ = self.min_ + 1e-12

    def transform(self, x):
        return (x - self.min_) / (self.max_ - self.min_)

    def inverse_transform(self, x):
        return x * (self.max_ - self.min_) + self.min_


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open("artifacts/scaler.json", "r", encoding="utf-8") as f:
        s = json.load(f)

    seq_len = int(s["seq_len"])
    scaler = MinMaxScaler1D(s["min"], s["max"])

    model = LSTMForecast(hidden_size=64, num_layers=1).to(device)
    model.load_state_dict(torch.load("artifacts/lstm_model.pth", map_location=device))
    model.eval()

    def forecast_next(values_csv: str):
        try:
            parts = [float(x.strip()) for x in values_csv.split(",") if x.strip() != ""]
            if len(parts) != seq_len:
                return f"Hata: {seq_len} adet değer girmelisin. Sen {len(parts)} girdin."

            arr = np.array(parts, dtype=np.float32)
            arr_scaled = scaler.transform(arr)

            x = torch.tensor(arr_scaled).unsqueeze(0).unsqueeze(-1).to(device)
            with torch.no_grad():
                pred_scaled = model(x).cpu().numpy()[0, 0]

            pred = float(scaler.inverse_transform(pred_scaled))
            return f"Tahmin edilen bir sonraki saatlik tüketim: {pred:.2f} MWh"

        except Exception as e:
            return f"Hata: {str(e)}"

    demo = gr.Interface(
        fn=forecast_next,
        inputs=gr.Textbox(lines=3, placeholder=f"{seq_len} adet tüketim değerini virgülle gir (örn: 27000, 26800, ...)"),
        outputs=gr.Textbox(label="Tahmin"),
        title="LSTM ile Türkiye Elektrik Tüketimi Tahmini",
        description="Son 24 saatlik tüketim verisine göre bir sonraki saatlik tüketimi tahmin eder."
    )

    demo.launch(share=False)


if __name__ == "__main__":
    main()
