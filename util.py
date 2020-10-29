import torch

def describe(x):
    print("Type: {}".format(x.type()))
    print("Shape/size: {}".format(x.shape))
    print("Values: \n{}".format(x))

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def n_grams(text, n):
    return [text[i:i+n] for i in range(len(text)-n+1)]



cleaned = ['i', 'like', "nlp"]
print(n_grams(cleaned, 2))
