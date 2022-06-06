import torch
from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST, CIFAR10
from torchvision.transforms import Compose


def load_data(batch_size, dataset_name, num_works, train_params, model_params):
    if dataset_name == 'MNIST':
        train_set, test_set = load_mnist_dataset(batch_size, num_works)
        model_params['hin'] = 28
        model_params['in_size'] = 1
        model_params['out_size'] = 10
        train_params['save_acc'] = True
    elif dataset_name == 'CIFAR10':
        train_set, test_set = load_cifar_dataset(batch_size, num_works)
        model_params['hin'] = 32
        model_params['in_size'] = 3
        model_params['out_size'] = 10
        train_params['save_acc'] = True
    elif dataset_name == 'BOSTON':
        train_set, test_set = load_boston_dataset(batch_size, num_works)
        model_params['hin'] = 1
        model_params['in_size'] = 13
        model_params['out_size'] = 1
        train_params['save_acc'] = False
    else:
        raise ValueError('To implement')

    train_params['nb_batches'] = train_set.__len__()
    train_params['p'] = train_set.dataset.__len__()
    return train_set, test_set


def load_boston_dataset(batch_size, num_works):
    dataset = BostonDataset()
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_works)
    return train_loader, _


def load_mnist_dataset(batch_size, num_works):
    dataset = MNIST('.', download=True, transform=transforms.ToTensor(), train=True)
    train_set = DataLoader(dataset, batch_size=batch_size, num_workers=num_works)
    dataset = MNIST('.', download=True, transform=transforms.ToTensor(), train=False)
    test_set = DataLoader(dataset, batch_size=batch_size, num_workers=num_works)
    return train_set, test_set


def load_cifar_dataset(batch_size, num_works):

    train_transform = Compose([transforms.RandomHorizontalFlip(p=0.5),
                               transforms.RandomCrop(32, padding=4),
                               transforms.ToTensor(),
                               transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    test_transform = Compose([transforms.ToTensor(),
                              transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    train_set = CIFAR10('.', download=True, transform=train_transform, train=True)
    train_set = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_works)

    test_set = CIFAR10('.', download=True, transform=test_transform, train=False)
    test_set = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_works)
    print('CIFAR LOADER')
    return train_set, test_set


class BostonDataset(torch.utils.data.Dataset):
    def __init__(self):
        x, y = load_boston(return_X_y=True)
        x = StandardScaler().fit_transform(x)
        self.data = torch.from_numpy(x).float()
        self.targets = torch.from_numpy(y).float()

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, i):
        return self.data[i], self.targets[i]
