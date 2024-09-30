# Semantic Correspondence Task

## Dataset
Download the dataset from http://cvlab.postech.ac.kr/research/SPair-71k/.  

Put the three json files in `dataset` to `SPair-71k/JPEGImages`. (They are taken from [diffusion_hyperfeatures](https://github.com/diffusion-hyperfeatures/diffusion_hyperfeatures).)  

## Feature Extraction
Two with fine-grained prompt "a photo of aeroplane, bicycle, bird, boat, bottle, bus, car, cat, chair, cow, dog, horse, motorbike, person, potted plant, sheep, train, tv monitor, high quality, best quality, highly realistic, masterpiece, high resolution".
```bash
# first write the prompt into prompt.txt
python3 extract_feature.py \
    --version 1-5 \
    --img_size 512 \
    --t 50 \
    --layer feature/configs/config_15_legacy.json \
    --batch_size 1 \
    --aggregate_output \
    --input_dir "path/to/SPair-71k/JPEGImages/**/*.jpg" \
    --nested_input_dir \
    --output_dir your/feature/path/corres/feature1/ \
    --use_original_filename
# make another run with a different random seed, setting output_dir to feature2
```

Two with different LoRA weights and a simple prompt `a photo`.
```bash
# first write the prompt into prompt.txt
python3 extract_feature.py \
    --version 1-5 \
    --img_size 512 \
    --t 50 \
    --layer feature/configs/config_15_legacy.json \
    --batch_size 1 \
    --aggregate_output \
    --input_dir "path/to/SPair-71k/JPEGImages/**/*.jpg" \
    --nested_input_dir \
    --output_dir your/feature/path/corres/feature3/ \
    --use_original_filename \
    --offline_lora path/to/lora
# make another run with a different LoRA weight, setting output_dir to feature4
```

Two with ControlNet and the same simple prompt.
```bash
# first write the prompt into prompt.txt
python3 extract_feature.py \
    --version 1-5 \
    --img_size 512 \
    --t 50 \
    --layer feature/configs/config_15_legacy.json \
    --batch_size 1 \
    --aggregate_output \
    --input_dir "path/to/SPair-71k/JPEGImages/**/*.jpg" \
    --nested_input_dir \
    --output_dir your/feature/path/corres/feature5/ \
    --use_original_filename \
    --control canny \
    --denoising_from 60
# make another run with a different random seed, setting output_dir to feature6
```

Attention features with ControlNet, LoRA, and `a photo of aeroplane, bicycle, bird, boat, bottle, bus, car, cat, chair, cow, dog, horse, motorbike, person, potted plant, sheep, train, tv monitor`.
```bash
# first write the prompt into prompt.txt
python3 extract_feature.py \
    --version 1-5 \
    --img_size 512 \
    --t 50 \
    --layer correspondence_gate/config_attn.json \
    --batch_size 1 \
    --aggregate_output \
    --input_dir "path/to/SPair-71k/JPEGImages/**/*.jpg" \
    --nested_input_dir \
    --output_dir your/feature/path/corres/attn/ \
    --use_original_filename \
    --control canny \
    --denoising_from 60 \
    --offline_lora path/to/lora \
    --attention up_cross
```

## Discrimination Run
Now make the discrimination run. Specify all features with `feature_id` and make sure put the attention feature at last.
```bash
python3 task-corres.py \
    --log_path your/log/path \
    --dataset_path path/to/SPair-71k/JPEGImages \
    --feature_path your/feature/path/corres/ \
    --feature_id feature1 feature2 feature3 feature4 feature5 feature6 attn \
    --feature_len 3520 \
    --attn_len 154 \
    --task_name gate_conv \
    --algorithm conv
```
Change `algorithm` to `nn` for the setting without output convolution layers.
