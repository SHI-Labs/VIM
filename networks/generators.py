# ------------------------------------------------------------------------
# Modified from MGMatting (https://github.com/yucornetto/MGMatting)
# ------------------------------------------------------------------------

import torch
import torch.nn as nn

from   utils import CONFIG
from   networks import encoders, decoders, ops

class Generator(nn.Module):
    def __init__(self, encoder, decoder, dec_T, dec_B):
        super(Generator, self).__init__()

        if encoder not in encoders.__all__:
            raise NotImplementedError("Unknown Encoder {}".format(encoder))
        self.encoder = encoders.__dict__[encoder]()

        self.aspp = ops.ASPP(in_channel=512, out_channel=512)

        if decoder not in decoders.__all__:
            raise NotImplementedError("Unknown Decoder {}".format(decoder))
        self.decoder = decoders.__dict__[decoder](dec_T, dec_B)

    def forward(self, image, tg_guidance, re_guidance):
        inp = torch.cat((image, tg_guidance, re_guidance), dim=1)
        embedding, mid_fea = self.encoder(inp)
        embedding = self.aspp(embedding)
        pred = self.decoder(embedding, mid_fea)
        return pred

def get_generator(encoder, decoder, dec_T, dec_B):
    generator = Generator(encoder=encoder, decoder=decoder, dec_T=dec_T, dec_B=dec_B)
    return generator