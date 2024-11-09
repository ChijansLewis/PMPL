
---

# Progressive Multimodal Pivot Learning: Towards Semantic Discordance Understanding as Humans

This repository hosts the code for the **Progressive Multimodal Pivot Learning (PMPL)** research project. Our work focuses on enhancing the semantic discordance understanding in existing multimodal recognition models by introducing a novel learning paradigm called **Multimodal Pivot Learning Paradigm**. This approach emulates human-like discordance understanding, building upon the [PMF](https://github.com/yaoweilee/PMF) implementation.

## Project Structure

- **`environment.yml`**: Defines the dependencies and environment setup required for the project.
- **`PMPL_main.py`**: The main Python script for running experiments.
- **Directories for Models and Data**: Configure model and dataset directories following the formats in `models_dir.txt` and `dataset_dir.txt`.

## Environment Setup

To get started, set up the environment as outlined in `environment.yml`:

```bash
conda env create -f environment.yml
```

### Install Detectron2

```bash
pip install 'git+https://mirror.ghproxy.com/https://github.com/facebookresearch/detectron2.git'
```

### Install ViLT

```bash
cd ViLT
python setup.py install
cd ..
```

### Install PyTorch and Other Dependencies

If not already installed, install PyTorch:

```bash
pip install -r requirements.txt
```

## Dataset

We use three datasets: **Twitter-15/17**, **CrisisMMD**, and **MM-IMDB**.

### Download Datasets

1. **Twitter-15/17**: A re-annotated version with unimodal labels by Chen et al. - [Download here](https://github.com/code-chendl/HFIR)
2. **CrisisMMD**: [Download CrisisMMD v1.0](https://crisisnlp.qcri.org/crisismmd)
3. **MM-IMDB**: [Download here](http://lisi1.unal.edu.co/mmimdb/mmimdb.tar.gz)

### Preprocessing

After downloading, preprocess each dataset to `.json` files in the following format:

```json
{
  "id": "image_file_path",
  "text": "text_content",
  "text_label": "text_label",
  "image_label": "image_label",
  "label": "overall_label"
}
```

## Training Instructions

To train the PMPL model, run the following command:

```bash
python PMPL_main.py --seed <seed_value> --beta=<beta_value> --lr=<learning_rate> --prompt_length=<prompt_length> --n_fusion_layers=<fusion_layers> --batch_size=64 --class_num=<dataset_class_count> --config=./configs --dataset=<dataset_name> --dev_dataset=dev.txt --device=cuda:<cuda_device_number> --file_path=<output_file_path> --test_dataset=test.txt --train_dataset=train.txt --type=train
```

## Running the Model

To start the model training, execute:

```bash
python PMPL_main.py
```

---

This README file should give a clear overview of the project setup, dataset requirements, and training procedure for PMPL. Let me know if you need further customization!
