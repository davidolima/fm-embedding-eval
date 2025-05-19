import numpy as np
import torch.nn as nn

def save_embedding_to_file(
    fpath: str,
    model: nn.Module,
    fnames: np.array,
    labels: np.array,
    embeddings: np.array,
    classes: np.array,
) -> None:
    """
    Params:
      - fpath: Path to save file.
      - fnames: Array of inputs file names. Used to identify embeddings.
      - labels: Array of labels.
      - embeddings: Array of extracted embeddings.
    """

    fpath = fpath if fpath.endswith('.npy') else fpath + '.npy'

    print(f"[*] Saving embedding to file: `{fpath}`...", end=' ')
    with open(fpath, 'wb') as f:
        np.save(f, {
            "model": model.get_name(),
            "model_embedding_dim": model.get_feat_dim(),
            "fnames": fnames,
            "labels": labels,
            "embeddings": embeddings,
            "classes": classes,
        })
    print("Saved.")

