from torch.utils.data import DataLoader
import torch
import math
import numpy as np
from model import *  # 使用修改後的模型
from LoadData import *  # 假設數據集已適配 SST + SSS

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

class NinoTrainer:
    def __init__(self, mypara):
        self.mypara = mypara
        self.device = mypara.device
        self.model = NinoGeoformer(mypara).to(self.device)
        adam = torch.optim.Adam(self.model.parameters(), lr=0)
        factor = math.sqrt(mypara.d_size * mypara.warmup) * 0.0015
        self.opt = lrwarm(mypara.d_size, factor, mypara.warmup, optimizer=adam)
        self.sstlevel = 0  # SST 在第 0 通道
        ninoweight = torch.from_numpy(
            np.array([1.5] * 4 + [2] * 7 + [3] * 7 + [4] * 6) * np.log(np.arange(24) + 1)
        ).to(mypara.device)
        self.ninoweight = ninoweight[:self.mypara.output_length]

    def calscore(self, y_pred, y_true):
        # 計算 Nino 分數
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
        # 場數據的 RMSE
        rmse = torch.mean((y_pred - y_true) ** 2, dim=[3, 4])  # 對 H, W 平均
        rmse = rmse.sqrt().mean(dim=0)  # 對時間平均
        rmse = torch.sum(rmse, dim=[0, 1])  # 對批次和通道求和
        return rmse

    def loss_nino(self, y_pred, y_true):
        # Nino 指數的 RMSE
        rmse = torch.sqrt(torch.mean((y_pred - y_true) ** 2, dim=0))
        return rmse.sum()

    def combine_loss(self, loss1, loss2):
        return loss1 + loss2

    def model_pred(self, dataloader):
        self.model.eval()
        nino_pred = []
        var_pred = []
        nino_true = []
        var_true = []
        with torch.no_grad():
            for input_var, var_true in dataloader:
                input_var = input_var.float().to(self.device)
                var_true = var_true.float().to(self.device)
                out_var = self.model(input_var, predict_tar=None, train=False)  # (B, T, C, H, W)
                # 提取 SST 並計算 Nino 指數
                SST_true = var_true[:, :, self.sstlevel]  # (B, T, H, W)
                nino_true1 = SST_true[
                    :, :, self.mypara.lat_nino_relative[0]:self.mypara.lat_nino_relative[1],
                    self.mypara.lon_nino_relative[0]:self.mypara.lon_nino_relative[1]
                ].mean(dim=[2, 3])  # (B, T)
                SST_pred = out_var[:, :, self.sstlevel]
                nino_pred1 = SST_pred[
                    :, :, self.mypara.lat_nino_relative[0]:self.mypara.lat_nino_relative[1],
                    self.mypara.lon_nino_relative[0]:self.mypara.lon_nino_relative[1]
                ].mean(dim=[2, 3])  # (B, T)
                var_true.append(var_true)
                nino_true.append(nino_true1)
                var_pred.append(out_var)
                nino_pred.append(nino_pred1)
            var_pred = torch.cat(var_pred, dim=0)
            nino_pred = torch.cat(nino_pred, dim=0)
            nino_true = torch.cat(nino_true, dim=0)
            var_true = torch.cat(var_true, dim=0)
            ninosc = self.calscore(nino_pred, nino_true)
            loss_var = self.loss_var(var_pred, var_true).item()
            loss_nino = self.loss_nino(nino_pred, nino_true).item()
            combine_loss = self.combine_loss(loss_var, loss_nino)
        return var_pred, nino_pred, loss_var, loss_nino, combine_loss, ninosc

    def train_model(self, train_dataset, eval_dataset):
        dataloader_train = DataLoader(train_dataset, batch_size=self.mypara.batch_size_train, shuffle=False)
        dataloader_eval = DataLoader(eval_dataset, batch_size=self.mypara.batch_size_eval, shuffle=False)
        chk_path = self.mypara.model_savepath + "NinoGeoformer.pkl"
        torch.manual_seed(self.mypara.seeds)
        count = 0
        best = -math.inf
        sv_ratio = 1

        for epoch in range(self.mypara.num_epochs):
            print(f"Epoch {epoch}")
            self.model.train()
            for j, (input_var, var_true) in enumerate(dataloader_train):
                input_var = input_var.float().to(self.device)
                var_true = var_true.float().to(self.device)
                # 提取真實 SST 和 Nino 指數
                SST_true = var_true[:, :, self.sstlevel]
                nino_true = SST_true[
                    :, :, self.mypara.lat_nino_relative[0]:self.mypara.lat_nino_relative[1],
                    self.mypara.lon_nino_relative[0]:self.mypara.lon_nino_relative[1]
                ].mean(dim=[2, 3])  # (B, T)
                if sv_ratio > 0:
                    sv_ratio = max(sv_ratio - 2.5e-4, 0)
                # 訓練模型
                var_pred = self.model(input_var, var_true, train=True, sv_ratio=sv_ratio)  # (B, T, C, H, W)
                # 提取預測 SST 和 Nino 指數
                SST_pred = var_pred[:, :, self.sstlevel]
                nino_pred = SST_pred[
                    :, :, self.mypara.lat_nino_relative[0]:self.mypara.lat_nino_relative[1],
                    self.mypara.lon_nino_relative[0]:self.mypara.lon_nino_relative[1]
                ].mean(dim=[2, 3])  # (B, T)
                self.opt.optimizer.zero_grad()
                loss_var = self.loss_var(var_pred, var_true)
                loss_nino = self.loss_nino(nino_pred, nino_true)
                score = self.calscore(nino_pred, nino_true)
                combine_loss = self.combine_loss(loss_var, loss_nino)
                combine_loss.backward()
                self.opt.step()

                if j % 100 == 0:
                    print(f"Batch {j}, loss_var: {loss_var:.2f}, loss_nino: {loss_nino:.2f}, score: {score:.3f}")

            # 驗證
            self.model.eval()
            _, _, lossvar_eval, lossnino_eval, comloss_eval, sceval = self.model_pred(dataloader_eval)
            print(f"Eval - loss_var: {lossvar_eval:.3f}, loss_nino: {lossnino_eval:.3f}, combine_loss: {comloss_eval:.3f}, score: {sceval:.3f}")
            if sceval > best:
                best = sceval
                torch.save(self.model.state_dict(), chk_path)
                print("Model saved")
                count = 0
            else:
                count += 1
                print(f"Score did not improve for {count} epochs")
            if count == self.mypara.patience:
                print(f"Early stopping, best score: {best:.3f}")
                break

if __name__ == "__main__":
    from myconfig import mypara
    train_dataset = NinoDataset(mypara)
    eval_dataset = NinoTestDataset(mypara, ngroup=100)
    trainer = NinoTrainer(mypara)
    trainer.train_model(train_dataset, eval_dataset)