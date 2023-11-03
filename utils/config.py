# ------------------------------------------------------------------------
# Modified from MGMatting (https://github.com/yucornetto/MGMatting)
# ------------------------------------------------------------------------
from easydict import EasyDict

# Base default config
CONFIG = EasyDict({})
# to indicate this is a default setting, should not be changed by user
CONFIG.is_default = True
CONFIG.version = "baseline"
CONFIG.phase = "train"
# distributed training
CONFIG.dist = False
# global variables which will be assigned in the runtime
CONFIG.local_rank = 0
CONFIG.gpu = 0
CONFIG.world_size = 1

# Model config
CONFIG.model = EasyDict({})
# use pretrained checkpoint as encoder
CONFIG.model.imagenet_pretrain = True
CONFIG.model.imagenet_pretrain_path = "/path/to/model_best_resnet34_En_nomixup.pth"
CONFIG.model.batch_size = 16
CONFIG.model.num_frames = 5
# one-hot or class, choice: [3, 1]
CONFIG.model.mask_channel = 1
CONFIG.model.trimap_channel = 3

# hyper-parameter for refinement
CONFIG.model.self_refine_width1 = 30
CONFIG.model.self_refine_width2 = 15

# Model -> Architecture config
CONFIG.model.arch = EasyDict({})
# definition in networks/encoders/__init__.py and networks/encoders/__init__.py
CONFIG.model.arch.encoder = "res_shortcut_encoder_29"
CONFIG.model.arch.decoder = "res_shortcut_decoder_22"
CONFIG.model.arch.dec_T = 5
CONFIG.model.arch.dec_B = 2
# predefined for GAN structure
CONFIG.model.arch.discriminator = None


# Dataloader config
CONFIG.data = EasyDict({})
CONFIG.data.cutmask_prob = 0
CONFIG.data.workers = 0
# data path for training and validation in training phase
CONFIG.data.train_fg = None
CONFIG.data.train_alpha = None
CONFIG.data.train_bg = None
CONFIG.data.fg_alpha_dir = None
CONFIG.data.bg_img_dir = None
CONFIG.data.bg_video_dir = None
# feed forward image size (untested)
CONFIG.data.crop_size = 512
# composition of two foregrounds, affine transform, crop and HSV jitter
CONFIG.data.real_world_aug = False
CONFIG.data.augmentation = True
CONFIG.data.random_interp = True


# Training config
CONFIG.train = EasyDict({})
CONFIG.train.total_step = 100000
CONFIG.train.warmup_step = 5000
CONFIG.train.val_step = 1000
# basic learning rate of optimizer
CONFIG.train.G_lr = 1e-3
# beta1 and beta2 for Adam
CONFIG.train.beta1 = 0.5
CONFIG.train.beta2 = 0.999
# weight of different losses
CONFIG.train.rec_weight = 1
CONFIG.train.comp_weight = 1
CONFIG.train.lap_weight = 1
CONFIG.train.alpha_weight = 1
CONFIG.train.tmp_weight = 1
# clip large gradient
CONFIG.train.clip_grad = True
# resume the training (checkpoint file name)
CONFIG.train.resume_checkpoint = None
# reset the learning rate (this option will reset the optimizer and learning rate scheduler and ignore warmup)
CONFIG.train.reset_lr = False

# Logging config
CONFIG.log = EasyDict({})
CONFIG.log.tensorboard_path = "./logs/tensorboard"
CONFIG.log.tensorboard_step = 100
# save less images to save disk space
CONFIG.log.tensorboard_image_step = 500
CONFIG.log.logging_path = "./logs/stdout"
CONFIG.log.logging_step = 10
CONFIG.log.logging_level = "DEBUG"
CONFIG.log.checkpoint_path = "./checkpoints"
CONFIG.log.checkpoint_step = 10000


def load_config(custom_config, default_config=CONFIG, prefix="CONFIG"):
    """
    This function will recursively overwrite the default config by a custom config
    :param default_config:
    :param custom_config: parsed from config/config.toml
    :param prefix: prefix for config key
    :return: None
    """
    if "is_default" in default_config:
        default_config.is_default = False

    for key in custom_config.keys():
        full_key = ".".join([prefix, key])
        if key not in default_config:
            raise NotImplementedError("Unknown config key: {}".format(full_key))
        elif isinstance(custom_config[key], dict):
            if isinstance(default_config[key], dict):
                load_config(default_config=default_config[key],
                            custom_config=custom_config[key],
                            prefix=full_key)
            else:
                raise ValueError("{}: Expected {}, got dict instead.".format(full_key, type(custom_config[key])))
        else:
            if isinstance(default_config[key], dict):
                raise ValueError("{}: Expected dict, got {} instead.".format(full_key, type(custom_config[key])))
            else:
                default_config[key] = custom_config[key]


