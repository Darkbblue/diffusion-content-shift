_base_ = [
    '_base_/models/upernet_r50.py', '_base_/datasets/ade20k.py',
    '_base_/default_runtime.py', '_base_/schedules/schedule_80k.py'
]
crop_size = (512, 512)
# normalize to [-1, 1]
# i know it may not look like [-1, 1] but it works as intended
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[v * 255 for v in [0.5, 0.5, 0.5]],
    std=[v * 255 for v in [0.5, 0.5, 0.5]],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=1,
    size=crop_size)

head_c=512
c_per_level = [1280*3, 1280*3, 640*3]  # modify this for different settings
model = dict(
    type='DiffusionSegmentor',
    decode_head=dict(
        type='UPerHead',
        in_channels=c_per_level,
        in_index=[0, 1, 2],
        pool_scales=(1, 2, 3),
        channels=head_c,
        dropout_ratio=0.1,
        num_classes=150,
        loss_decode=
        [
        dict(type='CrossEntropyLoss', loss_name='loss_ce', use_sigmoid=False, loss_weight=1.0),
        dict(type='LovaszLoss', reduction='none', loss_weight=1.0)
        ]
        ),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=c_per_level[1],
        in_index=1,
        channels=head_c,
        num_convs=1,
        dropout_ratio=0.1,
        num_classes=150,
        loss_decode=dict(type='CrossEntropyLoss', loss_name='loss_ce_aux', use_sigmoid=False, loss_weight=0.4)
        ),
    test_cfg=dict(mode='slide', crop_size=crop_size, stride=(512,512)),
    data_preprocessor=data_preprocessor,

    c_per_level=c_per_level,

    # modify the follwing for different settings
    diffusion_feature=[
        dict(
            layer={'up-level0-upsampler-out': True, 'up-level1-upsampler-out': True,
            'up-level2-upsampler-out': True},
            version='1-5',
            device='cuda:1',
            attention=None,
            img_size=512,
            t=50,
            train_unet=False,  # you can turn this off to reduce VRAM usage
        ),
        dict(
            layer={'up-level0-upsampler-out': True, 'up-level1-upsampler-out': True,
            'up-level2-upsampler-out': True},
            version='1-5',
            device='cuda:1',
            attention=None,
            img_size=512,
            t=50,
            train_unet=False,  # you can turn this off to reduce VRAM usage
            offline_lora='lora2a',
            control=['canny'],
        ),
        dict(
            layer={'up-level0-upsampler-out': True, 'up-level1-upsampler-out': True,
            'up-level2-upsampler-out': True},
            version='1-5',
            device='cuda:1',
            attention=None,
            img_size=512,
            t=50,
            train_unet=False,  # you can turn this off to reduce VRAM usage
            offline_lora='lora2b',
            control=['canny']
        ),
    ],
    feature_layers=[
        [
            [('up-level0-upsampler-out', 1280)],
            [('up-level1-upsampler-out', 1280)],
            [('up-level2-upsampler-out', 640)],
        ],
        [
            [('up-level0-upsampler-out', 1280)],
            [('up-level1-upsampler-out', 1280)],
            [('up-level2-upsampler-out', 640)],
        ],
        [
            [('up-level0-upsampler-out', 1280)],
            [('up-level1-upsampler-out', 1280)],
            [('up-level2-upsampler-out', 640)],
        ],
    ],
    # prompt='An urban street scene with multiple lanes, various buildings, traffic lights, cars in the lanes, and pedestrians, highly realistic.',
    prompt='a highly realistic photo of the real world. It can be an indoor scene, or an outdoor scene, or a photo of nature. high quality.',
)

# optimizer = dict(type='SGD', lr=0.01, weight_decay=0.001)
# uncomment the above line and the following optimizer=optimizer
# if you want to set other lr values
# optim_wrapper = dict(
#     optimizer=optimizer,
# )

train_cfg = dict(type='IterBasedTrainLoop', max_iters=80000, val_interval=2000)
# train_dataloader = dict(batch_size=4)  # set this if you want to change batch size
