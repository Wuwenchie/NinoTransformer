import torch
import torch.nn as nn
from my_tools import make_embedding, unfold_func, miniEncoder, miniDecoder, fold_func


class NinoGeoformer(nn.Module):
    def __init__(self, mypara):
        super().__init__()
        self.mypara = mypara
        self.d_size = mypara.d_size
        self.device = mypara.device

        # 輸入通道：SST + SSS（可選：+ taux, tauy）
        input_channels = 2  # SST 和 SSS
        self.cube_dim = input_channels * mypara.patch_size[0] * mypara.patch_size[1]

        # 嵌入層
        self.predictor_emb = make_embedding(
            cube_dim=self.cube_dim,
            d_size=self.d_size,
            emb_spatial_size=mypara.emb_spatial_size,
            max_len=mypara.input_length,
            device=self.device,
        )
        self.predict_tar_emb = make_embedding(
            cube_dim=self.cube_dim,
            d_size=self.d_size,
            emb_spatial_size=mypara.emb_spatial_size,
            max_len=mypara.output_length,
            device=self.device,
        )

        # 編碼器和解碼器
        enc_layer = miniEncoder(self.d_size, mypara.nheads, mypara.dim_feedforward, mypara.dropout)
        dec_layer = miniDecoder(self.d_size, mypara.nheads, mypara.dim_feedforward, mypara.dropout)
        self.encoder = multi_enc_layer(enc_layer, num_layers=mypara.num_encoder_layers)
        self.decoder = multi_dec_layer(dec_layer, num_layers=mypara.num_decoder_layers)

        self.linear_output = nn.Linear(self.d_size, self.cube_dim)

    def forward(self, predictor, predict_tar=None, in_mask=None, enout_mask=None, train=True, sv_ratio=0):
        """
        Args:
            predictor: (batch, input_length, C, H, W) - SST + SSS
            predict_tar: (batch, output_length, C, H, W) - 訓練時提供，測試時為 None
        Returns:
            outvar_pred: (batch, output_length, C, H, W) - 模型輸出的場數據(預測)
        """
        # print(f"predictor shape: {predictor.shape}")
        # if predict_tar is not None:
            # print(f"predict_tar shape: {predict_tar.shape}")
        # 編碼
        en_out = self.encode(predictor, in_mask)

        if train:
            # Teacher Forcing
            with torch.no_grad():
                connect_inout = torch.cat([predictor[:, -1:], predict_tar[:, :-1]], dim=1)
                out_mask = self.make_mask_matrix(connect_inout.size(1))
                outvar_pred = self.decode(connect_inout, en_out, out_mask, enout_mask)

            # 監督比例
            if sv_ratio > 1e-7:
                supervise_mask = torch.bernoulli(
                    sv_ratio * torch.ones(predict_tar.size(0), predict_tar.size(1) - 1, 1, 1, 1)).to(self.device)
            else:
                supervise_mask = 0
            predict_tar = supervise_mask * predict_tar[:, :-1] + (1 - supervise_mask) * outvar_pred[:, :-1]
            predict_tar = torch.cat([predictor[:, -1:], predict_tar], dim=1)
            outvar_pred = self.decode(predict_tar, en_out, out_mask, enout_mask)
            # print(f"outvar_pred shape: {outvar_pred.shape}")
        else:
            # 自回歸預測
            predict_tar = predictor[:, -1:]
            for t in range(self.mypara.output_length):
                out_mask = self.make_mask_matrix(predict_tar.size(1))
                print("out_mask shape:", out_mask.shape)   # 新增
                print("out_mask sample:", out_mask[-5:, -5:]) # 新增
                outvar_pred = self.decode(predict_tar, en_out, out_mask, enout_mask)
                print(f"Step {t}, outvar_pred[:, -1:] sample:", outvar_pred[:, -1:, 0, :5, :5])  # 新增
                predict_tar = torch.cat([predict_tar, outvar_pred[:, -1:]], dim=1)
        return outvar_pred

    def encode(self, predictor, in_mask):
        lb = predictor.size(0)  # 8
        T = predictor.size(1)  # 12
        S = self.mypara.H0 * self.mypara.W0
        # print(f"predictor before unfold: {predictor.shape}")
        predictor = unfold_func(predictor, self.mypara.patch_size)  # [8, 12, 24, 126]
        # print(f"predictor after unfold: {predictor.shape}")
        predictor = predictor.reshape(lb, T, self.cube_dim, S).permute(0, 3, 1, 2)  # [8, 126, 12, 24]
        # print(f"predictor after reshape: {predictor.shape}")
        predictor = self.predictor_emb(predictor)
        en_out = self.encoder(predictor, in_mask)
        return en_out
        
    def decode(self, predict_tar, en_out, out_mask, enout_mask):
        H, W = predict_tar.size()[-2:]  # e.g., 23, 72
        T = predict_tar.size(1)  # e.g., 24
        predict_tar = unfold_func(predict_tar, self.mypara.patch_size)
        predict_tar = predict_tar.reshape(predict_tar.size(0), T, self.cube_dim, -1).permute(0, 3, 1, 2)
        predict_tar = self.predict_tar_emb(predict_tar)
        output = self.decoder(predict_tar, en_out, out_mask, enout_mask)
        output = self.linear_output(output).permute(0, 2, 3, 1)
        output = output.reshape(predict_tar.size(0), T, self.cube_dim, H // self.mypara.patch_size[0], W // self.mypara.patch_size[1])
        output = fold_func(output, output_size=(H, W), kernel_size=self.mypara.patch_size)
        return output  # (B, T, C, H, W), e.g., (8, 24, 2, 23, 72)


    def compute_nino(self, outvar):
        # 從解碼器輸出計算 Nino 3.4 指數
        # 假設 SST 是第 0 通道，SSS 是第 1 通道
        outvar = self.nino_output(outvar).squeeze(-1)  # (batch, T, S)
        nino_pred = outvar.mean(dim=2)  # 空間平均作為 Nino 3.4
        return nino_pred  # (batch, output_length)

    def make_mask_matrix(self, sz: int):
        mask = (torch.triu(torch.ones(sz, sz)) == 0).T
        return mask.to(self.mypara.device)


class multi_enc_layer(nn.Module):
    def __init__(self, enc_layer, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([enc_layer for _ in range(num_layers)])

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x


class multi_dec_layer(nn.Module):
    def __init__(self, dec_layer, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([dec_layer for _ in range(num_layers)])

    def forward(self, x, en_out, out_mask, enout_mask):
        for layer in self.layers:
            x = layer(x, en_out, out_mask, enout_mask)
        return x

