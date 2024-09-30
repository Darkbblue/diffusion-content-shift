# Semantic Segmentation Task

## Dataset
We use CityScapes and ADE20K datasets. Follow [the mmseg guide](https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/user_guides/2_dataset_prepare.md#prepare-datasets) to prepare the two datasets.

## Adding the model
You need to go to the installed mmseg lib at `anaconda3/envs/generic-diffusion-feature/lib/python3.9/site-packages/mmseg/models/segmentors`. Then copy the file `models/diffusion_gate_segmentor.py` here. Also modify the `__init__.py` in the lib folder to import the newly added segmentor.

## Preparing LoRA weights
Change the `offline_lora` settings in the config file to point to your local LoRA weights.

## Run
```bash
python3 train.py configs/gate_ade.py --work-dir /data/diffusion-feature/logs/gate_ade
python3 train.py configs/gate_city.py --work-dir /data/diffusion-feature/logs/gate_city
```
