# ------------------------------------------------------------------------
# Modified from MGMatting (https://github.com/yucornetto/MGMatting)
# ------------------------------------------------------------------------

from .resnet_dec import ResNet_D_Dec, BasicBlock
from .res_shortcut_dec import ResShortCut_D_Dec
from .res_shortcut_dec_tmp import ResShortCut_D_Dec_tmp, BasicTmpBlock

__all__ = ['res_shortcut_decoder_22', 'res_shortcut_decoder_tmp_22']


def _res_shortcut_D_dec(block, layers, **kwargs):
    model = ResShortCut_D_Dec(block, layers, **kwargs)
    return model

def res_shortcut_decoder_22(dec_T, dec_B, **kwargs):
    """Constructs a resnet_encoder_14 model.
    """
    return _res_shortcut_D_dec(BasicBlock, [2, 3, 3, 2], **kwargs)

def _res_shortcut_D_dec_tmp(block, layers, dec_T, dec_B, **kwargs):
    model = ResShortCut_D_Dec_tmp(block, layers, dec_T, dec_B, **kwargs)
    return model

def res_shortcut_decoder_tmp_22(dec_T, dec_B, **kwargs):
    return _res_shortcut_D_dec_tmp(BasicTmpBlock, [2, 3, 3, 2], dec_T, dec_B, **kwargs)



