import os
from dotenv import load_dotenv
from huggingface_hub import login
from models import HF_MODELS 

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

def download_models(models):
    for model in models:
        if model in HF_MODELS: # only authenticate if there are hugging face models in list
            if not authenticate_hf():
                raise Exception("[ERROR] Failed to authenticate. Can't download models.")
            break

    print("[!] Downloading models")
    for model in models:
        print(f"    > Downloading {model.__name__}...")
        model.download_model()
    print("[!] Models downloaded successfully.")

if __name__ == '__main__':
    download_models()
