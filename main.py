import argparse
import numpy as np
import pytorch_lightning as pl
import torch
import wandb
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers.wandb import WandbLogger
from torch import nn
from utils_module.data import load_data
from utils_module.models import LinearBnnModel, ConvBnnModel, Resnet20, VGG
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

def get_model(model_name, dist_params_, train_params_, model_params_):

    if model_name == 'Linear':
        model_params_['in_size'] = model_params_['in_size'] * model_params_['hin'] ** 2
        return LinearBnnModel(dist_params_, train_params_, model_params_)

    elif model_name == 'Conv':
        return ConvBnnModel(dist_params_, train_params_, model_params_)

    elif model_name == 'VGG':
        return VGG(dist_params_, train_params_, model_params_)

    elif model_name == 'Resnet':
        return Resnet20(dist_params_, train_params_, model_params_)

    else:
        raise ValueError(f'To implement: {model_name}')


def get_exp_name(train_params_, model_params_):
    w = model_params_['w']
    alpha = train_params_['alpha']
    return f'w_{w}_alpha_{alpha}'


def get_trainer(nb_epochs, wandb_logger, lr_monitor, exp_name):
    if torch.cuda.is_available():

        trainer = pl.Trainer(gpus=1,
                             max_epochs=nb_epochs,
                             logger=wandb_logger,
                             callbacks=[lr_monitor],
                             weights_save_path=exp_name)
    else:
        trainer = pl.Trainer(max_epochs=nb_epochs, logger=wandb_logger)
    return trainer


def run(project_name,
        model_name,
        dataset_name,
        num_works_,
        batch_size_,
        dist_params_,
        train_params_,
        model_params_):

    train_set, test_set = load_data(batch_size_, dataset_name, num_works_, train_params_, model_params_)
    model = get_model(model_name, dist_params_, train_params_, model_params_)
    exp_name = get_exp_name(train_params_, model_params_)
    wandb.init(settings=wandb.Settings(start_method='fork'))
    wandb_logger = WandbLogger(name=exp_name, project=project_name)
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    trainer = get_trainer(train_params_['nb_epochs'], wandb_logger, lr_monitor, exp_name)
    trainer.fit(model, train_set, test_set)
    _ = trainer.test(model, test_set)
    wandb.finish()
    return model


parser = argparse.ArgumentParser()
parser.add_argument('--dataset')
parser.add_argument('--model_name')
parser.add_argument('--N_value', type=int)
parser.add_argument('--alpha_value', type=float)
parser.add_argument('--nb_epochs', type=int)
parser.add_argument('--project_name')
parser.add_argument('--lr', type=float)
parser.add_argument('--vgg_type')


if __name__ == '__main__':
    num_works = 0
    batch_size = 128
    wandb.finish()
    args = parser.parse_args()

    dist_params = {'init_mu_post': 0.,
                   'init_rho_post': np.log(np.exp(0.0001) - 1),
                   'sigma_prior': 1 / 5,
                   'mu_prior': 0.}

    train_params = {'lr': args.lr,
                    'nb_epochs': args.nb_epochs,
                    'nb_samples': 1,
                    'criterion': nn.MSELoss(reduction='sum') if args.dataset == 'BOSTON'
                    else nn.CrossEntropyLoss(reduction='sum'),
                    'alpha': args.alpha_value,
                    'dataset': args.dataset,
                    'model': args.model_name}

    model_params = {'padding': 1,
                    'dilation': 1,
                    'stride': 1,
                    'kernel_size': 3,
                    'N_last_layer': args.N_value}

    if args.vgg_type != 'None':
        model_params['VGG_type'] = int(args.vgg_type)

    run(args.project_name,
        args.model_name,
        args.dataset,
        num_works,
        batch_size,
        dist_params,
        train_params,
        model_params)
