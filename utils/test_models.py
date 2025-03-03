import torch

from huggingface_hub import login

from models import UNI, UNI2, Phikon, PhikonV2

test_input = torch.rand((1,3,224,224))

def test_wrapper(func):
    def wrapper():
        print(f"----- {func.__name__} -----")
        print("Input shape:", test_input.shape)
        feats_shape = func()
        print("Output shape:", feats_shape)

    return wrapper

@test_wrapper
def test_uni2():
    model = UNI2()
    feats = model(test_input)
    return feats.shape

@test_wrapper
def test_phikonv2():
    model = PhikonV2()
    feats = model(test_input)
    return feats.shape

@test_wrapper
def test_phikon():
    model = Phikon()
    feats = model(test_input)
    return feats.shape

@test_wrapper
def test_uni():
    model = UNI()
    feats = model(test_input)
    return feats.shape
       
if __name__ == '__main__':
    test_uni()
    test_uni2()
    test_phikon()
    test_phikonv2()
    print("----- END OF TESTS -----")
