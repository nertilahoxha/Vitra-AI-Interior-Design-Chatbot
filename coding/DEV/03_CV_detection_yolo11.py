import os
import shutil
import torch
from ultralytics import YOLO

# Path al dataset già unificato
data_yaml_path = r"dataset_roboflow\Vitra_Merge_update.v2-vitra_update.yolov11\data.yaml"


def main():   # Funzione main necessaria per Windows + multiprocessing
    # DEVICE
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Userò il device:", device)

    # Modello YOLO11 nano
    model = YOLO("yolo11n.yaml").to(device)

    # Training
    results = model.train(
        data=data_yaml_path,
        epochs=100,
        patience=20000,
        save_period=20,
        imgsz=640,
        batch=4,
        optimizer="auto",
        plots=True,
        device=device,
        workers=0  # 0 Evita errore multiprocessing su Windows
    )

    # Salvataggio del modello best.pt
    best_src = r"runs\detect\train\weights\best.pt"
    best_dst = r"coding\YOLO11\CV_best_model\best.pt"

    os.makedirs(os.path.dirname(best_dst), exist_ok=True)
    shutil.copy(best_src, best_dst)

    print("Modello salvato", best_dst)


# su Windows quando si usano DataLoader multiprocess
if __name__ == "__main__":
    main()
