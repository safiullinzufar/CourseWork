import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import Lambda

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# batch operation usng tensor slice
def WaveletTransformAxisY(batch_img):
    odd_img = batch_img[:, 0::2]
    even_img = batch_img[:, 1::2]
    L = (odd_img + even_img) / 2.0
    H = torch.abs(odd_img - even_img)
    return L, H


def WaveletTransformAxisX(batch_img):
    # transpose + fliplr
    tmp_batch = torch.tensor(batch_img.permute([0, 2, 1]).cpu().numpy().copy()[:, :, ::-1].copy())
    tmp_batch = tmp_batch.to(device)
    #     tmp_batch = torch.tensor.permute_dimensions(batch_img, [0, 2, 1])[:,:,::-1]
    _dst_L, _dst_H = WaveletTransformAxisY(tmp_batch)
    # transpose + flipud

    dst_L = torch.tensor(_dst_L.permute([0, 2, 1]).cpu().numpy().copy()[:, ::-1, ...].copy())
    dst_L = dst_L.to(device)
    #     dst_L = torch.tensor.permute_dimensions(_dst_L, [0, 2, 1])[:,::-1,...]
    dst_H = torch.tensor(_dst_H.permute([0, 2, 1]).cpu().numpy().copy()[:, ::-1, ...].copy())
    dst_H = dst_H.to(device)
    #     dst_H = torch.tensor.permute_dimensions(_dst_H, [0, 2, 1])[:,::-1,...]
    return dst_L, dst_H


class Wavelet(nn.Module):
    def __init(self):
        super(Wavelet, self).__init__()

    def forward(self, x):
        batch_image = x
        #         batch_image = batch_image.permute([0, 3, 1, 2])

        #     batch_image = torch.tensor.permute_dimensions(batch_image, [0, 3, 1, 2])
        r = batch_image[:, 0]
        g = batch_image[:, 1]
        b = batch_image[:, 2]

        # level 1 decomposition
        wavelet_L, wavelet_H = WaveletTransformAxisY(r)
        r_wavelet_LL, r_wavelet_LH = WaveletTransformAxisX(wavelet_L)
        r_wavelet_HL, r_wavelet_HH = WaveletTransformAxisX(wavelet_H)

        wavelet_L, wavelet_H = WaveletTransformAxisY(g)
        g_wavelet_LL, g_wavelet_LH = WaveletTransformAxisX(wavelet_L)
        g_wavelet_HL, g_wavelet_HH = WaveletTransformAxisX(wavelet_H)

        wavelet_L, wavelet_H = WaveletTransformAxisY(b)
        b_wavelet_LL, b_wavelet_LH = WaveletTransformAxisX(wavelet_L)
        b_wavelet_HL, b_wavelet_HH = WaveletTransformAxisX(wavelet_H)

        wavelet_data = [r_wavelet_LL, r_wavelet_LH, r_wavelet_HL, r_wavelet_HH,
                        g_wavelet_LL, g_wavelet_LH, g_wavelet_HL, g_wavelet_HH,
                        b_wavelet_LL, b_wavelet_LH, b_wavelet_HL, b_wavelet_HH]
        transform_batch = torch.stack(wavelet_data, axis=1)

        # level 2 decomposition
        wavelet_L2, wavelet_H2 = WaveletTransformAxisY(r_wavelet_LL)
        r_wavelet_LL2, r_wavelet_LH2 = WaveletTransformAxisX(wavelet_L2)
        r_wavelet_HL2, r_wavelet_HH2 = WaveletTransformAxisX(wavelet_H2)

        wavelet_L2, wavelet_H2 = WaveletTransformAxisY(g_wavelet_LL)
        g_wavelet_LL2, g_wavelet_LH2 = WaveletTransformAxisX(wavelet_L2)
        g_wavelet_HL2, g_wavelet_HH2 = WaveletTransformAxisX(wavelet_H2)

        wavelet_L2, wavelet_H2 = WaveletTransformAxisY(b_wavelet_LL)
        b_wavelet_LL2, b_wavelet_LH2 = WaveletTransformAxisX(wavelet_L2)
        b_wavelet_HL2, b_wavelet_HH2 = WaveletTransformAxisX(wavelet_H2)

        wavelet_data_l2 = [r_wavelet_LL2, r_wavelet_LH2, r_wavelet_HL2, r_wavelet_HH2,
                           g_wavelet_LL2, g_wavelet_LH2, g_wavelet_HL2, g_wavelet_HH2,
                           b_wavelet_LL2, b_wavelet_LH2, b_wavelet_HL2, b_wavelet_HH2]
        transform_batch_l2 = torch.stack(wavelet_data_l2, axis=1)

        # level 3 decomposition
        wavelet_L3, wavelet_H3 = WaveletTransformAxisY(r_wavelet_LL2)
        r_wavelet_LL3, r_wavelet_LH3 = WaveletTransformAxisX(wavelet_L3)
        r_wavelet_HL3, r_wavelet_HH3 = WaveletTransformAxisX(wavelet_H3)

        wavelet_L3, wavelet_H3 = WaveletTransformAxisY(g_wavelet_LL2)
        g_wavelet_LL3, g_wavelet_LH3 = WaveletTransformAxisX(wavelet_L3)
        g_wavelet_HL3, g_wavelet_HH3 = WaveletTransformAxisX(wavelet_H3)

        wavelet_L3, wavelet_H3 = WaveletTransformAxisY(b_wavelet_LL2)
        b_wavelet_LL3, b_wavelet_LH3 = WaveletTransformAxisX(wavelet_L3)
        b_wavelet_HL3, b_wavelet_HH3 = WaveletTransformAxisX(wavelet_H3)

        wavelet_data_l3 = [r_wavelet_LL3, r_wavelet_LH3, r_wavelet_HL3, r_wavelet_HH3,
                           g_wavelet_LL3, g_wavelet_LH3, g_wavelet_HL3, g_wavelet_HH3,
                           b_wavelet_LL3, b_wavelet_LH3, b_wavelet_HL3, b_wavelet_HH3]
        transform_batch_l3 = torch.stack(wavelet_data_l3, axis=1)

        # level 4 decomposition
        wavelet_L4, wavelet_H4 = WaveletTransformAxisY(r_wavelet_LL3)
        r_wavelet_LL4, r_wavelet_LH4 = WaveletTransformAxisX(wavelet_L4)
        r_wavelet_HL4, r_wavelet_HH4 = WaveletTransformAxisX(wavelet_H4)

        wavelet_L4, wavelet_H4 = WaveletTransformAxisY(g_wavelet_LL3)
        g_wavelet_LL4, g_wavelet_LH4 = WaveletTransformAxisX(wavelet_L4)
        g_wavelet_HL4, g_wavelet_HH4 = WaveletTransformAxisX(wavelet_H4)

        wavelet_L3, wavelet_H3 = WaveletTransformAxisY(b_wavelet_LL3)
        b_wavelet_LL4, b_wavelet_LH4 = WaveletTransformAxisX(wavelet_L4)
        b_wavelet_HL4, b_wavelet_HH4 = WaveletTransformAxisX(wavelet_H4)

        wavelet_data_l4 = [r_wavelet_LL4, r_wavelet_LH4, r_wavelet_HL4, r_wavelet_HH4,
                           g_wavelet_LL4, g_wavelet_LH4, g_wavelet_HL4, g_wavelet_HH4,
                           b_wavelet_LL4, b_wavelet_LH4, b_wavelet_HL4, b_wavelet_HH4]
        transform_batch_l4 = torch.stack(wavelet_data_l4, axis=1)

        # print('shape before')
        #         print("devices in wavelet")
        #         print(transform_batch.device)
        #         print(transform_batch_l2.device)
        #         print(transform_batch_l3.device)
        #         print(transform_batch_l4.device)

        #         decom_level_1 = transform_batch.permute([0, 2, 3, 1])
        #     decom_level_1 = torch.tensor.permute_dimensions(transform_batch, [0, 2, 3, 1])
        #         decom_level_2 = transform_batch_l2.permute([0, 2, 3, 1])
        #     decom_level_2 = torch.tensor.permute_dimensions(transform_batch_l2, [0, 2, 3, 1])
        #         decom_level_3 = transform_batch_l3.permute([0, 2, 3, 1])
        #     decom_level_3 = torch.tensor.permute_dimensions(transform_batch_l3, [0, 2, 3, 1])
        #         decom_level_4 = transform_batch_l4.permute([0, 2, 3, 1])
        #     decom_level_4 = torch.tensor.permute_dimensions(transform_batch_l4, [0, 2, 3, 1])

        # print('shape after')
        # print(decom_level_1.shape)
        # print(decom_level_2.shape)
        # print(decom_level_3.shape)
        # print(decom_level_4.shape)
        return [transform_batch, transform_batch_l2, transform_batch_l3, transform_batch_l4]


#         return [decom_level_1, decom_level_2, decom_level_3, decom_level_4] # uncomment upper lines to permute


def Wavelet_out_shape(input_shapes):
    # print('in to shape')
    return [tuple([None, 112, 112, 12]), tuple([None, 56, 56, 12]),
            tuple([None, 28, 28, 12]), tuple([None, 14, 14, 12])]


class WaveletCnnModel(nn.Module):
    def __init__(self, n_classes):
        super(WaveletCnnModel, self).__init__()

        self.n_classes = n_classes

        self.input_shape = 256, 256, 3
        self.wavelet = Wavelet()

        # level one decomposition starts
        self.conv_1 = nn.Conv2d(12, 64, kernel_size=(3, 3), padding=1)
        self.norm_1 = nn.BatchNorm2d(64)

        self.conv_1_2 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(2, 2), padding=1)
        self.norm_1_2 = nn.BatchNorm2d(64)

        # level two decomposition starts
        self.conv_a = nn.Conv2d(12, 64, kernel_size=(3, 3), padding=1)
        self.norm_a = nn.BatchNorm2d(64)

        # concate level one and level two decomposition
        self.conv_2 = nn.Conv2d(128, 128, kernel_size=(3, 3), padding=1)
        self.norm_2 = nn.BatchNorm2d(128)

        self.conv_2_2 = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(2, 2), padding=1)
        self.norm_2_2 = nn.BatchNorm2d(128)

        # level three decomposition starts
        self.conv_b = nn.Conv2d(12, 64, kernel_size=(3, 3), padding=1)
        self.norm_b = nn.BatchNorm2d(64)

        self.conv_b_2 = nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1)
        self.norm_b_2 = nn.BatchNorm2d(128)

        # concate level two and level three decomposition
        self.conv_3 = nn.Conv2d(256, 256, kernel_size=(3, 3), padding=1)
        self.norm_3 = nn.BatchNorm2d(256)

        self.conv_3_2 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=1)
        self.norm_3_2 = nn.BatchNorm2d(256)

        # level four decomposition start
        self.conv_c = nn.Conv2d(12, 64, kernel_size=(3, 3), padding=1)
        self.norm_c = nn.BatchNorm2d(64)

        self.conv_c_2 = nn.Conv2d(64, 256, kernel_size=(3, 3), padding=1)
        self.norm_c_2 = nn.BatchNorm2d(256)

        self.conv_c_3 = nn.Conv2d(256, 256, kernel_size=(3, 3), padding=1)
        self.norm_c_3 = nn.BatchNorm2d(256)

        # concate level level three and level four decomposition
        self.conv_4 = nn.Conv2d(512, 512, kernel_size=(3, 3), padding=1)
        self.norm_4 = nn.BatchNorm2d(512)

        self.conv_4_2 = nn.Conv2d(512, 256, kernel_size=(3, 3), stride=(2, 2), padding=1)
        self.norm_4_2 = nn.BatchNorm2d(256)

        self.conv_5_1 = nn.Conv2d(256, 128, kernel_size=(3, 3), padding=1)
        self.norm_5_1 = nn.BatchNorm2d(128)

        self.pool_5_1 = nn.AvgPool2d(kernel_size=(7, 7), stride=1, padding=3)
        self.flat_5_1 = nn.Flatten()

        self.fc_5 = nn.Linear(8192, 2048)
        self.norm_5 = nn.BatchNorm1d(2048)
        #         relu_5 = Activation('relu', name='relu_5')(norm_5)
        self.drop_5 = nn.Dropout(0.5)

        self.fc_6 = nn.Linear(2048, 2048)
        # + activation relu
        self.norm_6 = nn.BatchNorm1d(2048)
        #         relu_6 = Activation('relu', name='relu_6')(norm_6)
        self.drop_6 = nn.Dropout(0.5)

        self.output = nn.Linear(2048, self.n_classes)

    def forward(self, x):
        #         print(x.shape)
        input_l1, input_l2, input_l3, input_l4 = self.wavelet(x)
        #         print(input_l1.shape, input_l2.shape, input_l3.shape, input_l4.shape)

        # level one decomposition starts
        conv_1 = self.conv_1(input_l1)
        norm_1 = self.norm_1(conv_1)
        relu_1 = F.leaky_relu(norm_1)

        conv_1_2 = self.conv_1_2(relu_1)
        norm_1_2 = self.norm_1_2(conv_1_2)
        relu_1_2 = F.leaky_relu(norm_1_2)

        # level two decomposition starts
        conv_a = self.conv_a(input_l2)
        norm_a = self.norm_a(conv_a)
        relu_a = F.leaky_relu(norm_a)

        # concate level one and level two decomposition
        concate_level_2 = torch.cat([relu_1_2, relu_a], 1)
        conv_2 = self.conv_2(concate_level_2)
        norm_2 = self.norm_2(conv_2)
        relu_2 = F.leaky_relu(norm_2)

        conv_2_2 = self.conv_2_2(relu_2)
        norm_2_2 = self.norm_2_2(conv_2_2)
        relu_2_2 = F.leaky_relu(norm_2_2)

        # level three decomposition starts
        conv_b = self.conv_b(input_l3)
        norm_b = self.norm_b(conv_b)
        relu_b = F.leaky_relu(norm_b)

        conv_b_2 = self.conv_b_2(relu_b)
        norm_b_2 = self.norm_b_2(conv_b_2)
        relu_b_2 = F.leaky_relu(norm_b_2)

        # concate level two and level three decomposition
        concate_level_3 = torch.cat([relu_2_2, relu_b_2], 1)
        conv_3 = self.conv_3(concate_level_3)
        norm_3 = self.norm_3(conv_3)
        relu_3 = F.leaky_relu(norm_3)

        conv_3_2 = self.conv_3_2(relu_3)
        norm_3_2 = self.norm_3_2(conv_3_2)
        relu_3_2 = F.leaky_relu(norm_3_2)

        # level four decomposition start
        conv_c = self.conv_c(input_l4)
        norm_c = self.norm_c(conv_c)
        relu_c = F.leaky_relu(norm_c)

        conv_c_2 = self.conv_c_2(relu_c)
        norm_c_2 = self.norm_c_2(conv_c_2)
        relu_c_2 = F.leaky_relu(norm_c_2)

        conv_c_3 = self.conv_c_3(relu_c_2)
        norm_c_3 = self.norm_c_3(conv_c_3)
        relu_c_3 = F.leaky_relu(norm_c_3)

        # concate level level three and level four decomposition
        concate_level_4 = torch.cat([relu_3_2, relu_c_3], 1)
        conv_4 = self.conv_4(concate_level_4)
        norm_4 = self.norm_4(conv_4)
        relu_4 = F.leaky_relu(norm_4)

        conv_4_2 = self.conv_4_2(relu_4)
        norm_4_2 = self.norm_4_2(conv_4_2)
        relu_4_2 = F.leaky_relu(norm_4_2)

        conv_5_1 = self.conv_5_1(relu_4_2)
        norm_5_1 = self.norm_5_1(conv_5_1)
        relu_5_1 = F.leaky_relu(norm_5_1)

        pool_5_1 = self.pool_5_1(relu_5_1)
        flat_5_1 = self.flat_5_1(pool_5_1)

        fc_5 = self.fc_5(flat_5_1)
        norm_5 = self.norm_5(fc_5)
        relu_5 = F.leaky_relu(norm_5)
        drop_5 = self.drop_5(relu_5)

        fc_6 = self.fc_6(drop_5)
        norm_6 = self.norm_6(fc_6)
        relu_6 = F.leaky_relu(norm_6)
        drop_6 = self.drop_6(relu_6)

        output = self.output(drop_6)
        output = F.softmax(output)
        #         output = nn.Softmax(output)
        return output