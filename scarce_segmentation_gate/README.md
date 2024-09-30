# Label-Scarce Semantic Segmentation Task

This task requires very large RAM (not VRAM), as all features are stored in RAM.  

## Dataset
Download the Horse-21 dataset from https://github.com/yandex-research/ddpm-segmentation.

## Run
For this task, we separately extract features and later load them for discrimination.
Here we take Horse-21 as a step-by-step example, and Bedroom-28 will be presented in a simpler way.

One Feature with fine-grained prompt `a horse, high quality, best quality, highly realistic, masterpiece`.
This requires only writing the specific prompt into prompt file and no other special settings.
```bash
# first write the prompt into prompt.txt
python3 extract_feature.py \
    --version 1-5 \
    --img_size 512 \
    --t 50 \
    --layer feature/configs/config_15_legacy.json \
    --attention up_cross \
    --batch_size 1 \
    --aggregate_output \
    --input_dir "your/dataset/path/horse_21/real/test/*.png" \
    --output_dir your/feature/path/horse_21/feature1/ \
    --split test
# then make another run with "test" changed to "train"
```

A feature using ControlNet, with a simple prompt `a horse`.
This one requires additionally setting `control` as `canny`, which is a rather efficient type of ControlNet.
```bash
# first write the prompt into prompt.txt
python3 extract_feature.py \
    --version 1-5 \
    --img_size 512 \
    --t 50 \
    --layer feature/configs/config_15_legacy.json \
    --attention up_cross \
    --batch_size 1 \
    --aggregate_output \
    --input_dir "your/dataset/path/horse_21/real/test/*.png" \
    --output_dir your/feature/path/horse_21/feature2/ \
    --split test \
    --control canny \
    --denoising_from 60
# then make another run with "test" changed to "train"
```

Two features using different LoRA weights and a simple prompt.
This requires setting `offline_lora` as the path of your downloaded and converted LoRA weights.
```bash
# first write the prompt into prompt.txt
python3 extract_feature.py \
    --version 1-5 \
    --img_size 512 \
    --t 50 \
    --layer feature/configs/config_15_legacy.json \
    --attention up_cross \
    --batch_size 1 \
    --aggregate_output \
    --input_dir "your/dataset/path/horse_21/real/test/*.png" \
    --output_dir your/feature/path/horse_21/feature3/ \
    --split test \
    --offline_lora a1
# then make another run with "test" changed to "train"

# first write the prompt into prompt.txt
python3 extract_feature.py \
    --version 1-5 \
    --img_size 512 \
    --t 50 \
    --layer feature/configs/config_15_legacy.json \
    --attention up_cross \
    --batch_size 1 \
    --aggregate_output \
    --input_dir "your/dataset/path/horse_21/real/test/*.png" \
    --output_dir your/feature/path/horse_21/feature4/ \
    --split test \
    --offline_lora b1
# then make another run with "test" changed to "train"
```

The last step is to prepare some features extracted with the model in https://github.com/yandex-research/ddpm-segmentation. Let's say these features are stored in `path/to/ddpm` folder.

With all features ready, now run the discrimination script. Note that the paths to `ddpm` features and other features are specified with different arguments.
```bash
python3 task-pixel.py \
    --category horse_21 \
    --dataset_path your/dataset/path/horse_21/real/ \
    --log_path your/log/path \
    --feature_path your/feature/path/horse_21 \
    --feature_id feature1 feature2 feature3 feature4 \
    --feature_len 3674 \
    --ddpm_feature path/to/ddpm \
    --task_name gate \
    --batch_size 64 \
    --shuffle_dataset
```

Then simple instructions for Bedroom-28:
- A feature with fine-grained prompt `a photo of a tidy and well-designed bedroom`.
- A feature using ControlNet, with a simple prompt `a bedroom`.
- A feature using LoRA weight.
