# 🤖 Neural-Forge-Ai

Neural-Forge-Ai is a cutting-edge AI model training and deployment framework designed for efficient and scalable fine-tuning of state-of-the-art machine learning models. This platform provides a seamless, no-code solution for AI model training, ensuring flexibility in various tasks such as fine-tuning, text classification, regression, and more.

---

## 🔧 Features

- **No-Code Training**: Train machine learning models effortlessly without deep technical expertise.
- **Multi-Task Support**: Fine-tune models for LLM training, text classification, token classification, and more.
- **Optimized Infrastructure**: Supports both local and cloud-based training.
- **Seamless Deployment**: Easily deploy trained models to Hugging Face Spaces or cloud environments.
- **Custom Configuration Support**: Use YAML-based configurations to fine-tune models efficiently.

---

## 📈 Supported Tasks

| Task | Status | Python Notebook | Example Configs |
| --- | --- | --- | --- |
| LLM Fine-Tuning | ✅ | [Colab Notebook](https://colab.research.google.com/) | [Example YAML](configs/llm_finetune.yaml) |
| Text Classification | ✅ | [Colab Notebook](https://colab.research.google.com/) | [Example YAML](configs/text_classification.yaml) |
| Text Regression | ✅ | [Colab Notebook](https://colab.research.google.com/) | [Example YAML](configs/text_regression.yaml) |
| Token Classification | ✅ | Coming Soon | [Example YAML](configs/token_classification.yaml) |
| Seq2Seq | ✅ | Coming Soon | [Example YAML](configs/seq2seq.yaml) |
| Image Classification | ✅ | Coming Soon | [Example YAML](configs/image_classification.yaml) |

---

## 🌐 Running UI on Cloud or Local

- **Deploy on Hugging Face Spaces**: [![Deploy on Spaces](https://huggingface.co/datasets/huggingface/badges/resolve/main/deploy-on-spaces-md.svg)](https://huggingface.co/login?next=%2Fspaces%2Fneural-forge-ai%2Fneural-forge-ui%3Fduplicate%3Dtrue)

- **Run on Colab via ngrok**: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

---

## 🔄 Local Installation

You can install **Neural-Forge-Ai** via pip. Ensure **Python >= 3.10** is installed.

```sh
pip install neural-forge-ai
```

### **Dependencies**
Ensure `git-lfs` and PyTorch are installed:

```sh
conda create -n neuralforge python=3.10
conda activate neuralforge
pip install neural-forge-ai
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
conda install -c "nvidia/label/cuda-12.1.0" cuda-nvcc
```

Once installed, start the application using:

```sh
neuralforge app --port 8080 --host 127.0.0.1
```

To use a config file for training:

```sh
neuralforge --config <path_to_config_file>
```

---

## 🔖 Configuration Example

Example YAML for fine-tuning:

```yaml
task: llm-sft
base_model: HuggingFaceTB/SmolLM2-1.7B-Instruct
project_name: neural-forge-finetune
log: tensorboard
backend: local

data:
  path: dataset/path
  train_split: train
  valid_split: null
  chat_template: tokenizer
  column_mapping:
    text_column: messages

params:
  block_size: 2048
  model_max_length: 4096
  epochs: 2
  batch_size: 1
  lr: 1e-5
  peft: true
  quantization: int4
  target_modules: all-linear
  padding: right
  optimizer: paged_adamw_8bit
  scheduler: linear
  gradient_accumulation: 8
  mixed_precision: bf16
  merge_adapter: true

hub:
  username: ${HF_USERNAME}
  token: ${HF_TOKEN}
  push_to_hub: true
```

To fine-tune a model using the above config:

```sh
export HF_USERNAME=<your_hugging_face_username>
export HF_TOKEN=<your_hugging_face_write_token>
neuralforge --config <path_to_config_file>
```

---

## 📜 Documentation

For detailed documentation, visit [Neural-Forge-Ai Docs](https://your-docs-link.com)

---

## 🌟 Citation

If you use **Neural-Forge-Ai**, please cite:

```
@inproceedings{punith-2025-neuralforgeai,
    title = "Neural-Forge-Ai: AI Model Training Framework",
    author = "V Punith Kumar",
    year = "2025",
    url = "https://github.com/v-punithkumar/Neural-Forge-Ai",
}
```

---

🎉 **Contributions & Issues**
- Submit issues & feature requests on GitHub.
- PRs are welcome! Read our contributing guidelines.

Happy Model Training! 🚀