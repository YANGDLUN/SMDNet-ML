import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict
from mymodelnew.toolbox.models.segformer.mix_transformer import mit_b4
# from mymodelnew.toolbox.models.segformer.mix_transformer import mit_b2
# from mydesignmodel.y_model2.spi import FinalProcessingModule
# from y_model3.chang_spi import CompleteModel
from y_model3.Mu1_moudles.Curvature_up import Unifie
from y_model2.PointRD3 import FinalProcessingModule1
# from mydesignmodel.y_model2.PointRD3 import FinalProcessingModule1
# from y_model2.decoder import PyramidDecoder
from y_model3.Mu1_moudles.ML_Decoder import CombinedDecoder


class NewMod(nn.Module):
    def __init__(self, num_class=41,embed_dims=[64, 128, 320, 512]):
        super(NewMod,self).__init__()

        self.channels = [64,128,320,512]
        self.rgb_d = mit_b4()

        self.cs1 = Unifie(64)
        self.cs2 = Unifie(128)
        # self.cs1 = CompleteModel(rgb_channels=64, depth_channels=64, height=120, width=160, out_channels=64, fusion_method='concat')
        # self.cs2 = CompleteModel(rgb_channels=128, depth_channels=128, height=60, width=80, out_channels=128, fusion_method='concat')
        self.po1 = FinalProcessingModule1(320,3)
        self.po2 = FinalProcessingModule1(512,3)

        # self.conv512_320 = nn.Conv2d(512, 320, 1, 1)
        # self.conv320_128 = nn.Conv2d(320, 128, 1, 1)
        # self.conv128_64 = nn.Conv2d(128,64,1,1)
        # self.conv41 = nn.Conv2d(64, 41, 1, 1)
        self.decoder = CombinedDecoder()

    def forward(self,rgb,dep):
        # rgb_list = self.rgb_mit(rgb)
        rgb_list = self.rgb_d(rgb)
        dep_list = self.rgb_d(dep)

        CS1 = self.cs1(rgb_list[0],dep_list[0])
        CS2 = self.cs2( rgb_list[1],dep_list[1])

        po1 = self.po1(rgb_list[2],dep_list[2])
        po2 = self.po2(rgb_list[3],dep_list[3])

        # out = self.decoder(rgb_list[3], rgb_list[2], rgb_list[1], rgb_list[0])
        out = self.decoder(po2, po1, CS2, CS1)

        return out,CS1,CS2

    def load_pre_sa(self, pre_model1):
        new_state_dict3 = OrderedDict()
        state_dict = torch.load(pre_model1)['state_dict']
        for k, v in state_dict.items():
            name = k[9:]
            new_state_dict3[name] = v
        self.rgb_d.load_state_dict(new_state_dict3, strict=False)
        # self.d_mit.load_state_dict(new_state_dict3, strict=False)
        # self.rgb_mit.load_state_dict(new_state_dict3, strict=False)
        print('self.backbone_dmit loading')

if __name__ == '__main__':
    net = NewMod()
    rgb = torch.randn([2, 3, 480, 640])
    d = torch.randn([2, 3, 480, 640])
    s = net(rgb, d)
    from mymodelnew.FindTheBestDec.model.FLOP import CalParams
    CalParams(net, rgb, d)
    print("==> Total params: %.2fM" % (sum(p.numel() for p in net.parameters()) / 1e6))
    print("s.shape:", s[1].shape)







