from Geoformer import Geoformer
from myconfig import mypara
import torch
from torch.utils.data import DataLoader
import numpy as np
import math
from LoadData import make_dataset2, make_testdataset
import mlflow
import mlflow.pytorch

class lrwarm:
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p["lr"] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        if step is None:
            step = self._step
        return self.factor * (
            self.model_size ** (-0.5)
            * min(step ** (-0.5), step * self.warmup ** (-1.5))
        )


class modelTrainer:
    def __init__(self, mypara):
        assert mypara.input_channal == mypara.output_channal
        self.mypara = mypara
        self.device = mypara.device
        self.mymodel = Geoformer(mypara).to(mypara.device)
        adam = torch.optim.Adam(self.mymodel.parameters(), lr=0)
        factor = math.sqrt(mypara.d_size * mypara.warmup) * 0.0015
        self.opt = lrwarm(mypara.d_size, factor, mypara.warmup, optimizer=adam)
        self.sstlevel = 0
        if self.mypara.needtauxy:
            self.sstlevel = 0
        elif self.mypara.need_sss_tauxy:
            self.sstlevel = 0
        ninoweight = torch.from_numpy(
            np.array([1.5] * 4 + [2] * 7 + [3] * 7 + [4] * 6)
            * np.log(np.arange(24) + 1)
        ).to(mypara.device)
        self.ninoweight = ninoweight[: self.mypara.output_length]

    def calscore(self, y_pred, y_true):
        with torch.no_grad():
            pred = y_pred - y_pred.mean(dim=0, keepdim=True)
            true = y_true - y_true.mean(dim=0, keepdim=True)
            cor = (pred * true).sum(dim=0) / (
                torch.sqrt(torch.sum(pred ** 2, dim=0) * torch.sum(true ** 2, dim=0)) + 1e-6
            )
            acc = (self.ninoweight * cor).sum()
            rmse = torch.mean((y_pred - y_true) ** 2, dim=0).sqrt().sum()
            sc = 2 / 3.0 * acc - rmse
        return sc.item()

    def loss_var(self, y_pred, y_true):
        rmse = torch.mean((y_pred - y_true) ** 2, dim=[3, 4])
        rmse = rmse.sqrt().mean(dim=0)
        rmse = torch.sum(rmse, dim=[0, 1])
        return rmse

    def loss_nino(self, y_pred, y_true):
        rmse = torch.sqrt(torch.mean((y_pred - y_true) ** 2, dim=0))
        return rmse.sum()

    def combien_loss(self, loss1, loss2):
        combine_loss = loss1 + loss2
        return combine_loss

    def model_pred(self, dataloader):
        self.mymodel.eval()
        nino_pred = []
        var_pred = []
        nino_true = []
        var_true = []
        with torch.no_grad():
            for input_var, var_true1 in dataloader:
                SST = var_true1[:, :, self.sstlevel]
                nino_true1 = SST[:, :, self.mypara.lat_nino_relative[0]:self.mypara.lat_nino_relative[1],
                                 self.mypara.lon_nino_relative[0]:self.mypara.lon_nino_relative[1]].mean(dim=[2, 3])
                out_var = self.mymodel(input_var.float().to(self.device), predictand=None, train=False)
                SST_out = out_var[:, :, self.sstlevel]
                out_nino = SST_out[:, :, self.mypara.lat_nino_relative[0]:self.mypara.lat_nino_relative[1],
                                   self.mypara.lon_nino_relative[0]:self.mypara.lon_nino_relative[1]].mean(dim=[2, 3])
                var_true.append(var_true1)
                nino_true.append(nino_true1)
                var_pred.append(out_var)
                nino_pred.append(out_nino)

            var_pred = torch.cat(var_pred, dim=0)
            nino_pred = torch.cat(nino_pred, dim=0)
            nino_true = torch.cat(nino_true, dim=0)
            var_true = torch.cat(var_true, dim=0)

            ninosc = self.calscore(nino_pred, nino_true.float().to(self.device))
            loss_var = self.loss_var(var_pred, var_true.float().to(self.device)).item()
            loss_nino = self.loss_nino(nino_pred, nino_true.float().to(self.device)).item()
            combine_loss = self.combien_loss(loss_var, loss_nino)
        return var_pred, nino_pred, loss_var, loss_nino, combine_loss, ninosc

    def train_model(self, dataset_train, dataset_eval):
        chk_path = self.mypara.model_savepath + "Geoformer.pkl"
        torch.manual_seed(self.mypara.seeds)
        dataloader_train = DataLoader(dataset_train, batch_size=self.mypara.batch_size_train, shuffle=False)
        dataloader_eval = DataLoader(dataset_eval, batch_size=self.mypara.batch_size_eval, shuffle=False)
        count = 0
        best = -math.inf
        sv_ratio = 1
        global_step = 0

        mlflow.set_experiment("Geoformer_Training")
        with mlflow.start_run():
            mlflow.set_tag("model", "Geoformer")
            mlflow.log_params({
                "lr_factor": self.opt.factor,
                "warmup": self.opt.warmup,
                "d_model": self.mypara.d_size,
                "batch_size": self.mypara.batch_size_train,
                "epochs": self.mypara.num_epochs
            })

            for i_epoch in range(self.mypara.num_epochs):
                print("==========" * 8)
                print("\n-->epoch: {0}".format(i_epoch))
                self.mymodel.train()
                for j, (input_var, var_true) in enumerate(dataloader_train):
                    SST = var_true[:, :, self.sstlevel]
                    # print(f"SST at index {j}", SST)
                    nino_true = SST[:, :, self.mypara.lat_nino_relative[0]:self.mypara.lat_nino_relative[1],
                                    self.mypara.lon_nino_relative[0]:self.mypara.lon_nino_relative[1]].mean(dim=[2, 3])
                    if sv_ratio > 0:
                        sv_ratio = max(sv_ratio - 2.5e-4, 0)
                    var_pred = self.mymodel(input_var.float().to(self.device), var_true.float().to(self.device),
                                            train=True, sv_ratio=sv_ratio)
                    SST_pred = var_pred[:, :, self.sstlevel]
                    nino_pred = SST_pred[:, :, self.mypara.lat_nino_relative[0]:self.mypara.lat_nino_relative[1],
                                         self.mypara.lon_nino_relative[0]:self.mypara.lon_nino_relative[1]].mean(dim=[2, 3])
                    self.opt.optimizer.zero_grad()
                    loss_var = self.loss_var(var_pred, var_true.float().to(self.device))
                    loss_nino = self.loss_nino(nino_pred, nino_true.float().to(self.device))
                    score = self.calscore(nino_pred, nino_true.float().to(self.device))
                    combine_loss = self.combien_loss(loss_var, loss_nino)
                    combine_loss.backward()
                    self.opt.step()

                    mlflow.log_metric("Train/Loss_Var", loss_var.item(), step=global_step)
                    mlflow.log_metric("Train/Loss_Nino", loss_nino.item(), step=global_step)
                    mlflow.log_metric("Train/Combine_Loss", combine_loss.item(), step=global_step)
                    mlflow.log_metric("Train/Score", score, step=global_step)

                    if j % 100 == 0:
                        print("\n-->batch:{} loss_var:{:.2f}, loss_nino:{:.2f}, score:{:.3f}".format(
                            j, loss_var, loss_nino, score))

                    if (i_epoch + 1 >= 4) and (j + 1) % 200 == 0:
                        _, _, lossvar_eval, lossnino_eval, comloss_eval, sceval = self.model_pred(dataloader_eval)
                        print("-->Evaluation... \nloss_var:{:.3f} \nloss_nino:{:.3f} \nloss_com:{:.3f} \nscore:{:.3f}".format(
                            lossvar_eval, lossnino_eval, comloss_eval, sceval))
                        mlflow.log_metric("Eval/Loss_Var", lossvar_eval, step=global_step)
                        mlflow.log_metric("Eval/Loss_Nino", lossnino_eval, step=global_step)
                        mlflow.log_metric("Eval/Combine_Loss", comloss_eval, step=global_step)
                        mlflow.log_metric("Eval/Score", sceval, step=global_step)

                        if sceval > best:
                            torch.save(self.mymodel.state_dict(), chk_path)
                            mlflow.pytorch.log_model(self.mymodel, "best_model")
                            best = sceval
                            count = 0
                            print("\nsaving model...")
                    global_step += 1

                _, _, lossvar_eval, lossnino_eval, comloss_eval, sceval = self.model_pred(dataloader_eval)
                print("\n-->epoch{} end... \nloss_var:{:.3f} \nloss_nino:{:.3f} \nloss_com:{:.3f} \nscore: {:.3f}".format(
                    i_epoch, lossvar_eval, lossnino_eval, comloss_eval, sceval))

                mlflow.log_metric("Epoch/Loss_Var", lossvar_eval, step=i_epoch)
                mlflow.log_metric("Epoch/Loss_Nino", lossnino_eval, step=i_epoch)
                mlflow.log_metric("Epoch/Combine_Loss", comloss_eval, step=i_epoch)
                mlflow.log_metric("Epoch/Score", sceval, step=i_epoch)

                if sceval <= best:
                    count += 1
                    print("\nsc is not increase for {} epoch".format(count))
                else:
                    count = 0
                    print("\nsc is increase from {:.3f} to {:.3f}   \nsaving model...\n".format(best, sceval))
                    torch.save(self.mymodel.state_dict(), chk_path)
                    mlflow.pytorch.log_model(self.mymodel, "best_model")
                    best = sceval
                if count == self.mypara.patience:
                    print("\n-----!!!early stopping reached, max(sceval)= {:3f}!!!-----".format(best))
                    break
        del self.mymodel


if __name__ == "__main__":
    print(mypara.__dict__)
    print("\nloading pre-train dataset...")
    traindataset = make_dataset2(mypara)
    print(traindataset.selectregion())
    print("\nloading evaluation dataset...")
    evaldataset = make_testdataset(mypara, ngroup=100)
    print(evaldataset.selectregion())
    trainer = modelTrainer(mypara)
    trainer.train_model(dataset_train=traindataset, dataset_eval=evaldataset)
