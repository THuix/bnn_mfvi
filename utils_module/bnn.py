import os
import pytorch_lightning as pl
import torch
import torchmetrics
from torch.optim.lr_scheduler import StepLR
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class BNN(pl.LightningModule):
    def __init__(self, train_params):
        super(BNN, self).__init__()
        if train_params['save_acc']:
            self.accuracy = torchmetrics.Accuracy()
            self.ECE = torchmetrics.CalibrationError(n_bins=15, norm='l1')

    def get_temperature(self):
        return self.train_params['alpha'] * self.train_params['p'] / self.model_params['w']

    @staticmethod
    def check_params(params, true_params):
        for p in true_params:
            if not (p in params):
                raise ValueError(f"{p} is missing in {params}")
        return params

    def forward(self, x):
        if self.do_flatten:
            x = x.reshape(x.size()[0], -1)
        pred = self.seq(x) / self.model_params['N_last_layer']
        return pred

    def _get_kl(self):
        kl = 0
        for module in self.modules():
            if hasattr(module, 'kl'):
                kl += module.kl
        return kl

    def _step_1_sample(self, x, y):
        pred = self.seq(x) / self.model_params['N_last_layer']
        nll = self.train_params['criterion'](pred, y) / self.T
        kl = self._get_kl() / self.train_params['nb_batches']
        obj_loss = nll + kl
        return obj_loss, nll, kl, pred

    def re_balance_loss(self, loss):
        return self.train_params['alpha'] * loss * self.train_params['nb_batches'] / self.model_params['w']

    def step(self, batch):
        x, y = batch
        if self.do_flatten:
            x = x.reshape(x.size()[0], -1)

        obj_loss = torch.zeros(1, requires_grad=True).type_as(x)
        nll = torch.zeros(1).type_as(x)
        kl = torch.zeros(1).type_as(x)
        pred = torch.zeros((x.size()[0], self.model_params['out_size'])).type_as(x)

        for idx in range(self.train_params['nb_samples']):
            o, n, k, p = self._step_1_sample(x, y)
            obj_loss = obj_loss + o / self.train_params['nb_samples']
            nll = nll + n / self.train_params['nb_samples']
            kl = kl + k / self.train_params['nb_samples']
            pred = pred + p / self.train_params['nb_samples']

        nll_averaged = self.train_params['criterion'](pred, y) / self.T
        nll_averaged = self.re_balance_loss(nll_averaged)
        obj_loss = self.re_balance_loss(obj_loss)
        nll = self.re_balance_loss(nll)
        kl = self.re_balance_loss(kl)

        logs = {
            "obj": obj_loss,
            "kl": kl,
            "nll": nll,
            "ratio_nll_kl": nll / kl,
            "dist_f": (nll_averaged - nll).abs()}

        if self.train_params['save_acc']:
            self.accuracy.update(pred, y)
            self.ECE.update(pred.softmax(dim=1), y)
            logs['acc'] = self.accuracy.compute()
            logs['ece'] = self.ECE.compute()

        return obj_loss, logs

    def run_step(self, batch, dataset_type):
        loss, logs = self.step(batch)
        self.log_dict({f"{dataset_type}_{k}": v for k, v in logs.items()}, sync_dist=True)
        if self.train_params['save_acc']:
            self.accuracy.reset()
            self.ECE.reset()
        return loss

    def training_step(self, batch, batch_idx):
        return self.run_step(batch, 'train')

    def validation_step(self, batch, batch_idx):
        return self.run_step(batch, 'val')

    def test_step(self, batch, batch_idx):
        return self.run_step(batch, 'test')

    def configure_optimizers(self):
        # optimizer = torch.optim.SGD(self.parameters(), lr=self.train_params['lr'], momentum=0.9)
        opt = torch.optim.Adam(self.parameters(), lr=self.train_params['lr'])
        lr_schedulers = [StepLR(opt, 100, 0.1)]
        return [opt], lr_schedulers
