
import torch.nn as nn

# class sn_HybridModel(nn.Module):
class sn_FusionModel_1(nn.Module):
    """An nn.Module combining two nn.Modules 
    
    This hybrid model was created to connect a scattering model to another
    nn.Module, but can also combine any other two modules. 
    """

    def __str__(self):
        return str(self.scatteringBase)

    # def __init__(self, scatteringBase, top):
    def __init__(self, scatteringBase, top, mid):
        """Constructor for a HybridModel

        scatteringBase -- the scattering nn.Module
        top -- the nn.Module used after scatteringBase
        """
        # super(sn_HybridModel, self).__init__()
        super(sn_FusionModel_1, self).__init__()

        # self.scatteringBase = scatteringBase
        # self.top = top
        self.scatteringBase_sc1 = scatteringBase()
        self.scatteringBase_sc2 = scatteringBase()
        self.scatteringBase_sc3 = scatteringBase()
        self.scatteringBase_sc4 = scatteringBase()

        self.mid_s1 = mid()
        self.mid_s2 = mid()
        self.mid_s3 = mid()
        self.mid_s4 = mid()

        self.morm = nn.BatchNorm2d()
        self.concat = nn.Sequential()

        self.top_all = top()

    def forward(self, inp):
        x_ds1 = self.downsample_s1(inp) # first scale (s1) straightforward downsampling 224->112 (3, 112, 112)
        x_ds2 = self.downsample_s2(x_ds1) # first scale (s2) straightforward downsampling 112->56 (3, 56, 56)
        x_ds3 = self.downsample_s3(x_ds2) # first scale (s3) straightforward downsampling 56->28 (3, 28, 28)
        x_ds4 = self.downsample_s4(x_ds3) # first scale (s4) straightforward downsampling 28->14 (3, 14, 14)

        x_sc1 = self.scatteringBase_sc1(inp) # first scale (s1) scattering 224->112 (36, 112, 112)  -> d=12=6*2
        x_sc1_N = self.norm1(x_sc1) # first Normalization 112->112 ->(0,1)
        # x_sc1_F = self.concat(x_ds1, x_sc1_N) # fusion s1 scale fearture (6*3*2=39, 112, 112)
        x_sc1_Atten = self.mid_s1(x_sc1_N) # (12?, 112, 112)  3 blocks ( 3layer atten + 0.5sample + 3layer atten + 0.5sample + 3layer atten + 0.5sample)->(12, 14, 14)

        x_sc2 = self.scatteringBase_sc2(x_sc1_N) # second scale (s2) scattering 112->56 (6*6*3*2=216, 56, 56)  -> d=36=6*6
        #x_sc2 = self.scatteringBase_sc2(x_sc1_Atten) # select and try !!!
        x_sc2_N = self.norm2(x_sc2) # (216, 56, 56)
        # x_sc2_F = self.concat(x_ds2, x_sc2_N) # (219, 56, 56)
        x_sc2_Atten = self.mid_s2(x_sc2_N) # (36?, 56, 56) 2 blocks ( 3layer atten + 0.5sample + 3layer atten + 0.5sample)->(36, 14, 14)

        x_sc3 = self.scatteringBase_sc3(x_sc2_N) # third scale (s3) scattering 56->28 (6*6*6*3*2=1296, 28, 28) -> d=216=36*6
        x_sc3_N = self.norm3(x_sc3)
        # x_sc3_F = self.concat(x_ds3, x_sc3_N)
        x_sc3_Atten = self.mid_s3(x_sc3_N) # (216?, 28, 28) 1 blocks ( 3layer atten + 0.5sample) ->(216, 14, 14)

        x_sc4 = self.scatteringBase_sc4(x_sc3_N) # fourth scale (s4) scattering 28->14 ( , 14, 14) -> d=1296 = 216*6
        x_sc4_N = self.norm4(x_sc4)
        x_sc4_Atten = self.mid_s4(x_sc4_N)

        x_sc_all = self.concat(x_sc1_Atten, x_sc2_Atten, x_sc3_Atten, x_sc4_Atten)
        x_final = self.top_all(x_sc_all)
        # return self.top(self.scatteringBase(inp))
        return x_final

    def showParams(self):
        """prints shape of all parameters and is_leaf"""
        for x in self.parameters():
            if type(x['params']) == list:
                for tens in x['params']:
                    print(tens.shape,tens.is_leaf)
            else:
                print(x['params'].shape, x['params'].is_leaf)


class sn_FusionModel_2(nn.Module):
    """An nn.Module combining two nn.Modules

    This hybrid model was created to connect a scattering model to another
    nn.Module, but can also combine any other two modules.
    """

    def __str__(self):
        return str(self.scatteringBase)

    # def __init__(self, scatteringBase, top):
    def __init__(self, scatteringBase, top, mid):
        """Constructor for a HybridModel

        scatteringBase -- the scattering nn.Module
        top -- the nn.Module used after scatteringBase
        """
        # super(sn_HybridModel, self).__init__()
        super(sn_FusionModel_2, self).__init__()

        # self.scatteringBase = scatteringBase
        # self.top = top
        self.downsample_s1 = nn.Conv2d()
        self.scatteringBase_sc1 = scatteringBase()
        self.scatteringBase_sc2 = scatteringBase()
        self.scatteringBase_sc3 = scatteringBase()
        # self.scatteringBase_sc4 = scatteringBase()

        self.mid_s1 = mid()
        self.mid_s2 = mid()
        self.mid_s3 = mid()
        # self.mid_s4 = mid()

        self.morm = nn.BatchNorm2d()
        self.concat = nn.Sequential()

        self.top_all = top()

    def forward(self, inp):
        x_ds1 = self.downsample_s1(inp)  # first scale (s1) straightforward downsampling 224->112 (3, 112, 112)
        x_ds2 = self.downsample_s2(x_ds1)  # first scale (s2) straightforward downsampling 112->56 (3, 56, 56)
        x_ds3 = self.downsample_s3(x_ds2)  # first scale (s3) straightforward downsampling 56->28 (3, 28, 28)
        # x_ds4 = self.downsample_s4(x_ds3)  # first scale (s4) straightforward downsampling 28->14 (3, 14, 14)

        x_sc1 = self.scatteringBase_sc1(x_ds1)  # first scale (s1) scattering 112->56 (36, 56, 56)  -> d=12=6*2
        x_sc1_N = self.norm1(x_sc1)  # first Normalization 56->56 ->(0,1)
        # x_sc1_F = self.concat(x_ds1, x_sc1_N) # fusion s1 scale fearture (6*3*2=39, 56, 56)
        x_sc1_Atten = self.mid_s1(x_sc1_N)  # (12?, 56, 56)  3 blocks ( 3layer atten + 0.5sample + 3layer atten + 0.5sample)->(12, 14, 14)

        x_sc2 = self.scatteringBase_sc2(x_sc1_N)  # second scale (s2) scattering 56->28 (6*6*3*2=216, 28, 28)  -> d=36=6*6
        # x_sc2 = self.scatteringBase_sc2(x_sc1_Atten) 
        x_sc2_N = self.norm2(x_sc2)  # (216, 28, 28)
        # x_sc2_F = self.concat(x_ds2, x_sc2_N) # (219, 28, 28)
        x_sc2_Atten = self.mid_s2(x_sc2_N)  # (36?, 28, 28) 2 blocks ( 3layer atten + 0.5sample + 3layer atten + 0.5sample)->(36, 14, 14)

        x_sc3 = self.scatteringBase_sc3(x_sc2_N)  # third scale (s3) scattering 28->14 (6*6*6*3*2=1296, 14, 14) -> d=216=36*6
        x_sc3_N = self.norm3(x_sc3)
        # x_sc3_F = self.concat(x_ds3, x_sc3_N)
        x_sc3_Atten = self.mid_s3(x_sc3_N)  # ( , 14, 14) 1 blocks ( 3layer atten + 0.5sample) ->(216, 14, 14)

        # x_sc4 = self.scatteringBase_sc4(x_sc3_N)  # fourth scale (s4) scattering 28->14 ( , 14, 14) -> d=1296 = 216*6
        # x_sc4_N = self.norm4(x_sc4)
        # x_sc4_Atten = self.mid_s4(x_sc4_N)

        x_sc_all = self.concat(x_sc1_Atten, x_sc2_Atten, x_sc3_Atten)
        x_final = self.top_all(x_sc_all)
        # return self.top(self.scatteringBase(inp))
        return x_final

    def showParams(self):
        """prints shape of all parameters and is_leaf"""
        for x in self.parameters():
            if type(x['params']) == list:
                for tens in x['params']:
                    print(tens.shape, tens.is_leaf)
            else:
                print(x['params'].shape, x['params'].is_leaf)


class sn_FusionModel_3(nn.Module):
    """An nn.Module combining two nn.Modules
    This hybrid model was created to connect a scattering model to another
    nn.Module, but can also combine any other two modules.
    """
    def __str__(self):
        return str(self.scatteringBase)

    # def __init__(self, scatteringBase, top):
    def __init__(self, scatteringBaseMul, topMul):
        """Constructor for a HybridModel

        scatteringBase -- the scattering nn.Module
        top -- the nn.Module used after scatteringBase
        """
        super(sn_FusionModel_3, self).__init__()

        # self.scatteringBase = scatteringBase
        # self.top = top
        self.scatteringBase = scatteringBaseMul

        # self.mid_s1 = midMul
        # self.mid_s2 = midMul
        # self.mid_s3 = midMul
        # self.morm1 = nn.LayerNorm()    # nn.BatchNorm2d()
        # self.concat = nn.Sequential()

        # self.top_all = topMul
        self.top= topMul

    def forward(self, inp):

        x_sc0, x_sc1, x_sc2 = self.scatteringBase(inp)  # first scale (sc1, sc2, sc3) scattering 224->(56, 28, 14)
        # x_sc1_N = self.norm1(x_sc1)  # first Normalization 112->112 ->(0,1)
        #         ## x_sc1_F = self.concat(x_ds1, x_sc1_N) # fusion s1 scale fearture (6*3*2=39, 112, 112)
        # x_sc1_Atten = self.mid_s1(x_sc1_N)  # (12?, 112, 112)  3 blocks ( 3layer atten + 0.5sample + 3layer atten + 0.5sample + 3layer atten + 0.5sample)->(12, 14, 14)
        
        # x_sc2_N = self.norm2(x_sc2)  # (216, 56, 56)
        #          ## x_sc2_F = self.concat(x_ds2, x_sc2_N) # (219, 56, 56)
        # x_sc2_Atten = self.mid_s2(x_sc2_N)  # (, 56, 56) 2 blocks ( 3layer atten + 0.5sample + 3layer atten + 0.5sample)->(36, 14, 14)
        
        # x_sc3 = self.scatteringBase_sc3(x_sc2_N)  # third scale (s3) scattering 56->28 (6*6*6*3*2=1296, 28, 28) -> d=216=36*6
        # x_sc3_N = self.norm3(x_sc3)
        ## x_sc3_F = self.concat(x_ds3, x_sc3_N)
        # x_sc3_Atten = self.mid_s3(x_sc3_N)  # ( , 28, 28) 1 blocks ( 3layer atten + 0.5sample) ->(216, 14, 14)
        #
        # x_sc_all = self.concat(x_sc1_Atten, x_sc2_Atten, x_sc3_Atten)
        # x_final = self.top_all(x_sc1, x_sc2, x_sc3 )

        x_final = self.top(x_sc0, x_sc1, x_sc2)
        # return self.top(self.scatteringBase(inp))
        return x_final

    def showParams(self):
        """prints shape of all parameters and is_leaf"""
        for x in self.parameters():
            if type(x['params']) == list:
                for tens in x['params']:
                    print(tens.shape, tens.is_leaf)
            else:
                print(x['params'].shape, x['params'].is_leaf)
