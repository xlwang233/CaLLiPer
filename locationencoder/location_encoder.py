from torch import nn
from einops import rearrange
import numpy as np
# from datetime import datetime

import locationencoder.pe as PE
import locationencoder.nn as NN
    

def get_positional_encoding(name, hparams=None):
    if name == "direct":
        return PE.Direct()
    elif name == "cartesian3d":
        return PE.Cartesian3D()
    elif name == "sphericalharmonics":

        # default to analytic
        if "harmonics_calculation" not in hparams.keys():
            hparams["harmonics_calculation"] = "analytic"

        if "harmonics_calculation" in hparams.keys() and hparams['harmonics_calculation'] == "discretized":
            return PE.DiscretizedSphericalHarmonics(legendre_polys=hparams['legendre_polys'])
        else:
            return PE.SphericalHarmonics(legendre_polys=hparams['legendre_polys'],
                                         harmonics_calculation=hparams['harmonics_calculation'])
    elif name == "theory":
        return PE.Theory(min_radius=hparams['min_lambda'],
                         max_radius=hparams['max_lambda'],
                         frequency_num=hparams['frequency_num'])
    elif name == "wrap":
        return PE.Wrap()
    elif name in ["grid", "spherec", "spherecplus", "spherem", "spheremplus"]:
        return PE.GridAndSphere(min_radius=hparams['min_lambda'],
                       max_radius=hparams['max_lambda'],
                       frequency_num=hparams['frequency_num'],
                       name=name)
    else:
        raise ValueError(f"{name} not a known positional encoding.")


def get_neural_network(name, input_dim, hparams=None):
    if name == "linear":
        return nn.Linear(input_dim, hparams['dim_output'])
    elif name ==  "siren":
        return NN.SirenNet(
                dim_in=input_dim,
                dim_hidden=hparams['dim_hidden'],
                num_layers=hparams['num_layers'],
                dim_out=hparams['dim_output'],
                dropout=hparams['dropout'] if "dropout" in hparams.keys() else False
            )
    elif name == "fcnet":
        return NN.FCNet(
                num_inputs=input_dim,
                num_classes=hparams['dim_output'],
                dim_hidden=hparams['dim_hidden']
            )
    else:
        raise ValueError(f"{name} not a known neural networks.")


class LocationEncoder(nn.Module):
    def __init__(self, pe_name, nn_name, hparams):
        super().__init__()
        self.posenc = get_positional_encoding(
            pe_name, hparams
        )
        self.nnet = get_neural_network(
            nn_name,
            input_dim=self.posenc.embedding_dim,
            hparams=hparams
        )

    def forward(self, x):
        x = self.posenc(x)
        return self.nnet(x)


        
