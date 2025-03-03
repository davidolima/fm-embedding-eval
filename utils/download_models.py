import os
from dotenv import load_dotenv
from huggingface_hub import login
from models import UNI, UNI2, Phikon, PhikonV2 

MODELS = [
    UNI,
    UNI2,
    Phikon,
    PhikonV2,
]

def authenticate_hf():
    authenticated = False
    if not authenticated:
        load_dotenv()
        HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
        try:
            login(token=HF_TOKEN)
            authenticated = True
            print("[!] Authenticaded to HuggingFace succesfully")
        except Exception as e:
            print("[ERROR] Error during authentication:", e)
    return authenticated

def download_models():
    if not authenticate_hf():
        raise Exception("[ERROR] Failed to authenticate. Can't download models.")

    print("[!] Downloading models")
    for model in MODELS:
        print(f"    > Downloading {model.name}...")
        model.download_model()
    print("[!] Models downloaded successfully.")

if __name__ == '__main__':
    download_models()
