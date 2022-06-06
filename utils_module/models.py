import numpy as np
import torch.nn.functional as F
from torch import nn
from utils_module.bnn import BNN
from utils_module.layers import ConvBnn, LinearBnn


#######################################################################################################
#######################################################################################################
#######################################################################################################


class ConvBnnModel(BNN):
    def __init__(self, dist_params, train_params, model_params):

        self.dist_params = self.check_params(dist_params, ['init_rho_post',
                                                           'init_mu_post',
                                                           'sigma_prior',
                                                           'mu_prior'])

        self.train_params = self.check_params(train_params, ['lr',
                                                             'nb_samples',
                                                             'nb_batches',
                                                             'criterion',
                                                             'alpha',
                                                             'p'])

        self.model_params = self.check_params(model_params, ['N_last_layer',
                                                             'in_size',
                                                             'out_size',
                                                             'hin',
                                                             'padding',
                                                             'stride',
                                                             'dilation',
                                                             'kernel_size'])

        super(ConvBnnModel, self).__init__(self.train_params)

        h_out = ConvBnn.get_h_out(model_params['hin'],
                                  model_params['padding'],
                                  model_params['dilation'],
                                  model_params['kernel_size'],
                                  model_params['stride'])

        self.seq = nn.Sequential(
            ConvBnn(self.model_params['in_size'],
                    self.model_params['N_last_layer'],
                    self.dist_params['init_rho_post'],
                    self.dist_params['init_mu_post'],
                    self.dist_params['sigma_prior'],
                    self.dist_params['mu_prior'],
                    stride=self.model_params['stride'],
                    padding=self.model_params['padding'],
                    dilation=self.model_params['dilation'],
                    kernel_size=self.model_params['kernel_size'],
                    init_type='normal'),
            nn.ReLU(),
            nn.Flatten(),
            LinearBnn(self.model_params['N_last_layer'] * h_out ** 2,
                      self.model_params['out_size'],
                      self.dist_params['init_rho_post'],
                      self.dist_params['init_mu_post'],
                      self.dist_params['sigma_prior'],
                      self.dist_params['mu_prior'],
                      init_type='normal',
                      bias=False))

        self.model_params['w'] = np.sum([m.flatten().detach().cpu().numpy().shape for m in self.parameters()])
        self.do_flatten = False
        self.T = self.get_temperature()
        self.save_hyperparameters()

    @staticmethod
    def get_h_out(hin, p, d, k, s):
        return int((hin + 2 * p - d * (k - 1) - 1) / s + 1)


#######################################################################################################
#######################################################################################################
#######################################################################################################


class LinearBnnModel(BNN):
    def __init__(self, dist_params, train_params, model_params):

        self.dist_params = self.check_params(dist_params, ['init_rho_post',
                                                           'init_mu_post',
                                                           'sigma_prior',
                                                           'mu_prior'])

        self.train_params = self.check_params(train_params, ['lr',
                                                             'nb_samples',
                                                             'nb_batches',
                                                             'criterion',
                                                             'alpha',
                                                             'p'])

        self.model_params = self.check_params(model_params, ['in_size',
                                                             'out_size',
                                                             'N_last_layer'])

        super(LinearBnnModel, self).__init__(train_params)

        self.seq = nn.Sequential(
                LinearBnn(model_params['in_size'],
                          model_params['N_last_layer'],
                          self.dist_params['init_rho_post'],
                          self.dist_params['init_mu_post'],
                          self.dist_params['sigma_prior'],
                          self.dist_params['mu_prior'],
                          init_type='normal',
                          bias=False),
                nn.ReLU(),
                LinearBnn(model_params['N_last_layer'],
                          model_params['out_size'],
                          self.dist_params['init_rho_post'],
                          self.dist_params['init_mu_post'],
                          self.dist_params['sigma_prior'],
                          self.dist_params['mu_prior'],
                          init_type='normal',
                          bias=False))

        self.model_params['w'] = np.sum([m.flatten().detach().cpu().numpy().shape for m in self.parameters()]) / 2
        self.do_flatten = True
        self.T = self.get_temperature()
        self.save_hyperparameters()


#######################################################################################################
#######################################################################################################
#######################################################################################################


class ResnetBloc(nn.Module):
    def __init__(self, in_channels, out_channels, ks, stride, conv_in_identity, dist_params):
        super(ResnetBloc, self).__init__()
        self.seq = nn.Sequential(
            ConvBnn(in_channels,
                    out_channels,
                    dist_params['init_rho_post'],
                    dist_params['init_mu_post'],
                    dist_params['sigma_prior'],
                    dist_params['mu_prior'],
                    stride=stride,
                    padding=1,
                    dilation=1,
                    kernel_size=ks,
                    init_type='normal',
                    bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            ConvBnn(out_channels,
                    out_channels,
                    dist_params['init_rho_post'],
                    dist_params['init_mu_post'],
                    dist_params['sigma_prior'],
                    dist_params['mu_prior'],
                    stride=1,
                    padding=1,
                    dilation=1,
                    kernel_size=ks,
                    init_type='fixed',
                    bias=False),
            nn.BatchNorm2d(out_channels))
        if conv_in_identity:
            self.seq_identity = nn.Sequential(
                ConvBnn(in_channels,
                        out_channels,
                        dist_params['init_rho_post'],
                        dist_params['init_mu_post'],
                        dist_params['sigma_prior'],
                        dist_params['mu_prior'],
                        stride=2,
                        padding=0,
                        dilation=1,
                        kernel_size=1,
                        init_type='normal',
                        bias=False),
                nn.BatchNorm2d(out_channels))
        else:
            self.seq_identity = nn.Sequential()

    def forward(self, x):
        x_conv = self.seq(x)
        x_identity = self.seq_identity(x)
        x_out = x_conv + x_identity
        return F.relu(x_out)


def create_resnet_seq(dist_params, model_params):
    return nn.Sequential(
        ConvBnn(model_params['in_size'],
                16,
                dist_params['init_rho_post'],
                dist_params['init_mu_post'],
                dist_params['sigma_prior'],
                dist_params['mu_prior'],
                stride=1,
                padding=1,
                dilation=1,
                kernel_size=3,
                init_type='normal',
                bias=False),
        nn.BatchNorm2d(16),
        nn.ReLU(),
        ResnetBloc(16, 16, 3, 1, False, dist_params),
        ResnetBloc(16, 16, 3, 1, False, dist_params),
        ResnetBloc(16, 16, 3, 1, False, dist_params),
        ResnetBloc(16, 32, 3, 2, True, dist_params),
        ResnetBloc(32, 32, 3, 1, False, dist_params),
        ResnetBloc(32, 32, 3, 1, False, dist_params),
        ResnetBloc(32, 64, 3, 2, True, dist_params),
        ResnetBloc(64, 64, 3, 1, False, dist_params),
        ResnetBloc(64, 64, 3, 1, False, dist_params),
        nn.AvgPool2d(8),
        nn.Flatten(),
        LinearBnn(64,
                  10,
                  dist_params['init_rho_post'],
                  dist_params['init_mu_post'],
                  dist_params['sigma_prior'],
                  dist_params['mu_prior'],
                  init_type='normal',
                  bias=True))


class Resnet20(BNN):
    def __init__(self, dist_params, train_params, model_params):

        self.dist_params = self.check_params(dist_params, ['init_rho_post',
                                                           'init_mu_post',
                                                           'sigma_prior',
                                                           'mu_prior'])

        self.train_params = self.check_params(train_params, ['lr',
                                                             'nb_samples',
                                                             'nb_batches',
                                                             'criterion',
                                                             'alpha',
                                                             'p'])

        self.model_params = self.check_params(model_params, ['in_size', 'out_size', 'hin'])

        super(Resnet20, self).__init__(train_params)

        self.model_params['N_last_layer'] = 64
        self.seq = nn.Sequential(*create_resnet_seq(dist_params, model_params))
        self.model_params['w'] = self.extract_weights()
        self.do_flatten = False
        self.T = self.get_temperature()
        self.save_hyperparameters()

    def extract_weights(self):
        w = 0
        for module in self.modules():
            if hasattr(module, 'weight_mu'):
                w += module.weight_mu.detach().cpu().flatten().size()[0]
            if hasattr(module, 'weight_rho'):
                w += module.weight_rho.detach().cpu().flatten().size()[0]
            if hasattr(module, 'bias_mu'):
                w += module.bias_mu.detach().cpu().flatten().size()[0]
            if hasattr(module, 'bias_rho'):
                w += module.bias_rho.detach().cpu().flatten().size()[0]
            if hasattr(module, 'weight'):
                pass
        return int(w / 2)


#######################################################################################################
#######################################################################################################
#######################################################################################################


all_layers = {
    11: [('C', True, 0, 64, 3), 'M', ('C', False, 64, 128, 3), 'M', ('C', False, 128, 256, 3),
         ('C', False, 256, 256, 3),
         'M', ('C', False, 256, 512, 3), ('C', False, 512, 512, 3), 'M', ('C', False, 512, 512, 3),
         ('C', False, 512, 512, 3),
         'M', 'F', ('L', True, 0, 512, True), ('L', False, 512, 512, True), ('L', False, 512, 10, False)],

    13: [('C', True, 0, 64, 3), ('C', False, 64, 64, 3), 'M', ('C', False, 64, 128, 3), ('C', False, 128, 128, 3),
         'M',
         ('C', False, 128, 256, 3), ('C', False, 256, 256, 3), 'M', ('C', False, 256, 512, 3),
         ('C', False, 512, 512, 3),
         'M', ('C', False, 512, 512, 3), ('C', False, 512, 512, 3), 'M', 'F', ('L', True, 0, 512, True),
         ('L', False, 512, 512, True), ('L', False, 512, 10, False)],

    16: [('C', True, 0, 64, 3), ('C', False, 64, 64, 3), 'M', ('C', False, 64, 128, 3), ('C', False, 128, 128, 3),
         'M',
         ('C', False, 128, 256, 3), ('C', False, 256, 256, 3), ('C', False, 256, 256, 3), 'M',
         ('C', False, 256, 512, 3),
         ('C', False, 512, 512, 3), ('C', False, 512, 512, 3), 'M', ('C', False, 512, 512, 3),
         ('C', False, 512, 512, 3),
         ('C', False, 512, 512, 3), 'M', 'F', ('L', True, 0, 512, True), ('L', False, 512, 512, True),
         ('L', False, 512, 10, False)],

    19: [('C', True, 0, 64, 3), ('C', False, 64, 64, 3), 'M', ('C', False, 64, 128, 3), ('C', False, 128, 128, 3),
         'M', ('C', False, 128, 256, 3), ('C', False, 256, 256, 3), ('C', False, 256, 256, 3),
         ('C', False, 256, 256, 3),
         'M', ('C', False, 256, 512, 3), ('C', False, 512, 512, 3), ('C', False, 512, 512, 3),
         ('C', False, 512, 512, 3),
         'M', ('C', False, 512, 512, 3), ('C', False, 512, 512, 3), ('C', False, 512, 512, 3),
         ('C', False, 512, 512, 3),
         'M', 'F', ('L', True, 0, 512, True), ('L', False, 512, 512, True), ('L', False, 512, 10, False)]
}


class VGG(BNN):
    def __init__(self, dist_params, train_params, model_params):

        self.dist_params = self.check_params(dist_params, ['init_rho_post',
                                                           'init_mu_post',
                                                           'sigma_prior',
                                                           'mu_prior'])

        self.train_params = self.check_params(train_params, ['lr',
                                                             'nb_samples',
                                                             'nb_batches',
                                                             'criterion',
                                                             'alpha',
                                                             'p'])

        self.model_params = self.check_params(model_params, ['VGG_type',
                                                             'in_size',
                                                             'out_size',
                                                             'hin'])

        super(VGG, self).__init__(train_params)

        self.model_params['N_last_layer'] = 512
        self.seq = nn.Sequential(*self.create_seq(model_params['VGG_type'],
                                                  dist_params,
                                                  model_params['in_size'],
                                                  model_params['hin']))

        self.model_params['w'] = np.sum([m.flatten().detach().cpu().numpy().shape for m in self.parameters()])
        self.save_hist = False
        self.do_flatten = False
        self.T = self.get_temperature()
        self.save_hyperparameters()

    def create_seq(self, vgg_type, dist_params, in_size, hin):

        layers = all_layers[vgg_type]
        seq = []
        for layer in layers:
            if layer[0] == 'C':  # conv layer
                seq.append(nn.Dropout(0.2))
                seq.append(
                    ConvBnn(in_size if layer[1] else layer[2],
                            layer[3],
                            dist_params['init_rho_post'],
                            dist_params['init_mu_post'],
                            dist_params['sigma_prior'],
                            dist_params['mu_prior'],
                            stride=self.model_params['stride'],
                            padding=self.model_params['padding'],
                            dilation=self.model_params['dilation'],
                            kernel_size=layer[4],
                            init_type='normal',
                            bias=False))
                seq.append(nn.ReLU(inplace=True))

            elif layer == 'M':
                seq.append(nn.MaxPool2d(2, stride=2))

            elif layer[0] == 'L':  # Linear
                seq.append(nn.Dropout(0.5))
                seq.append(
                    LinearBnn(512 if layer[1] else layer[2],
                              layer[3],
                              dist_params['init_rho_post'],
                              dist_params['init_mu_post'],
                              dist_params['sigma_prior'],
                              dist_params['mu_prior'],
                              init_type='normal',
                              bias=True)
                )
                if layer[4]:
                    seq.append(nn.ReLU(inplace=True))

            elif layer == 'F':  # flatten
                seq.append(nn.Flatten())
            else:
                raise ValueError(f'layer name not find: {layer[0]}')
        raise ValueError(self)
        return seq
