from models import UNI, UNI2, Phikon, PhikonV2 

MODELS = [
    UNI,
    UNI2,
    Phikon,
    PhikonV2,
]

def download_models(ckpt_dir: str):
    print("[!] Downloading models")
    for model in MODELS:
        print(f"    > Downloading {model.__name__}...")
        model.download_model()
    print("[!] Models downloaded successfully.")

if __name__ == '__main__':
    download_models("./models/checkpoints")
