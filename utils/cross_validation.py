from data.glomerulus import GlomerulusDataset
import json

def get_class_from_file_name(filename, one_vs_all:str = None, classes: list[str] = ["Crescent", "Hypercelularidade", "Membranous", "Normal", "Podocitopatia", "Sclerosis"]):
    if one_vs_all is not None:
        if one_vs_all not in filename:
            return 'others'
        return one_vs_all

    contained_in = [c in filename for c in classes]
    
    assert sum(contained_in) == 1, f"More than one class found in file name `{filename}`"

    idx = contained_in.index(True)
    return classes[idx]


def load_splits_from_json(json_fpath:str, fold_no: int, val_split_idx: int, one_vs_all: str = None) -> tuple[GlomerulusDataset, GlomerulusDataset]:
    """
    Load cross-validation splits from a JSON file.

    Args:
        json_fpath (str): Path to the JSON file containing the splits.
        fold_no (int): How many folds are being used.
        val_split_idx (int): What split is being used for validation.

    Returns:
        tuple: A tuple containing the training and validation datasets.
    """
    with open(json_fpath, 'r') as f:
        data = json.load(f)
    
    if 'folds' not in data.keys():
        raise ValueError(f"Invalid JSON format. Expected 'folds' key not found in {json_fpath}. Found: {data.keys()}")

    if isinstance(fold_no, int):
        fold_no = str(fold_no)

    if fold_no not in data['folds'].keys():
        raise ValueError(f"Invalid fold number {fold_no}. Available folds: {list(data['folds'].keys())}")

    one_vs_all_class = json_fpath.split('_')[-1].split('.')[0]

    if one_vs_all is None:
        classes = ["Crescent", "Hypercelularidade", "Membranous", "Normal", "Podocitopatia", "Sclerosis"]
    else:
        classes = ['others', one_vs_all]

    train = GlomerulusDataset('', classes=classes, one_vs_all=one_vs_all, consider_augmented=True if one_vs_all is None else 'positive_only')
    test = GlomerulusDataset('', classes=classes, one_vs_all=one_vs_all, consider_augmented=False)

    train_files = data['folds'][fold_no][val_split_idx]['train_files']
    train.data.extend([(x,classes.index(get_class_from_file_name(x, one_vs_all=one_vs_all))) for x in train_files])
    
    test_files = data['folds'][fold_no][val_split_idx]['test_files']
    test.data.extend([(x, classes.index(get_class_from_file_name(x, one_vs_all=one_vs_all))) for x in test_files if 'augmented' not in x])

    return train, test

if __name__ == "__main__":
    class_name = "Crescent"
    json_fpath = 'data/cross-validation-folds/folds_indices_Data_{}.json'.format(class_name)

    print("[!] Testing Loading splits from file.")
    train, val = load_splits_from_json(json_fpath, 2, one_vs_all=class_name)
    print('-'*20, 'Train', '-'*20)
    train.info()
    print('-'*20, 'Validation', '-'*20)
    val.info()

    print("[!] Testing iterating through generated datasets.")
    print('-'*20, 'Train', '-'*20)
    for x,y,z in train:
        print(x.shape,y,z)
        break
    print('-'*20, 'Validation', '-'*20)
    for x,y,z in val:
        print(x.shape,y,z)
        break
