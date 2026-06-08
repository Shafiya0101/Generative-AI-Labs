# Generative AI — Lab Portfolio

**[🚀 Live Demo: Lab 1 GAN & VAE on Hugging Face Spaces →](https://huggingface.co/spaces/Shafiya1234/gan-vae-image-generator)**

A collection of hands-on labs from a **Generative AI** course, covering generative
image models (GANs & VAEs), LLM fine-tuning, text-to-image diffusion, and multimodal
LLM → diffusion pipelines. Each lab includes a fully-run Jupyter notebook and a
written report.

**Author:** Shafiya Kausar

---

## Contents

| Lab | Topic | Notebook | Report |
|-----|-------|----------|--------|
| **Lab 1** | GANs & VAEs on MNIST / Fashion-MNIST | [`Lab1_GAN_VAE`](Lab1_GAN_VAE/) | OK |
| **Lab 2** | LLM Fine-Tuning (Full vs LoRA) on distilgpt2 | [`Lab2_LLM_Finetune`](Lab2_LLM_Finetune/) | OK |
| **Lab 3** | Text-to-Image Diffusion (Stable Diffusion 1.5) | [`Lab3_Diffusion`](Lab3_Diffusion/) | OK |
| **Lab 4** | Multimodal Pipeline Evaluation (LLM → Diffusion) | [`Lab4_Pipeline_Evaluation`](Lab4_Pipeline_Evaluation/) | OK |

---

## Lab 1 — GANs & VAEs

Trains and compares generative models on MNIST and Fashion-MNIST.

- **GANs** at latent dimensions 32 / 64 / 128, with BCE and hinge losses and
  BatchNorm in the generator for stability.
- **VAEs** at latent dimensions 8 / 16 / 32, with reconstructions and latent-space
  interpolation.
- A **proxy FID-like metric** to compare sample quality.

**Key takeaways:** latent dimension and epoch count were the most influential
factors for GAN stability; hinge loss + BatchNorm helped mitigate mode collapse;
larger VAE latent dimensions improved reconstruction quality and sample diversity.

## Lab 2 — LLM Fine-Tuning

Fine-tunes **distilgpt2** on Shakespearean text and compares full fine-tuning against
parameter-efficient **LoRA**.

- **Baseline vs Full fine-tune vs LoRA**, compared by validation perplexity.
- **Generation comparisons** across the three models on several prompts.
- A **catastrophic forgetting test** — probing a non-Shakespeare prompt
  ("Explain how WiFi works") to see whether general knowledge survives fine-tuning.
- A **LoRA rank ablation** (r = 1 / 8 / 64) studying the trade-off between trainable
  parameters and perplexity.

**Key takeaways:** full fine-tuning reaches the lowest perplexity but severely
overwrites general knowledge (the WiFi prompt returns archaic dialogue); LoRA adapts
to the target style while training a tiny fraction of parameters and better preserving
the base model's knowledge; LoRA rank shows diminishing returns beyond a modest value.

## Lab 3 — Diffusion

Systematic experiments with **Stable Diffusion 1.5** via the `diffusers` library.

- Prompt-engineering experiments and a mini image gallery.
- **Inpainting:** masking a region of a base image (a medieval castle) and replacing
  it with a generated edit (a modern glass skyscraper).
- **CLIP similarity scoring** of generated images against prompts.
- A **scheduler & guidance (CFG) stress test** — comparing DDIM (most realistic
  skin texture) against PNDM (more "painterly").
- A **negation experiment** exploring why models struggle with negative prompts
  and how negative-prompt embeddings steer the noise vectors.

## Lab 4 — Multimodal Pipeline Evaluation

An end-to-end **LLM → diffusion** pipeline and its evaluation.

- An LLM (`distilgpt2`) generates prompts that feed a diffusion renderer.
- **CLIP similarity + ratings** to score outputs.
- **Bias / toxicity probes** (e.g. CEO / nurse / programmer portraits).
- A **"telephone game" loop** to study drift across repeated generations.
- A **magic-words experiment** measuring which modifiers most improve CLIP alignment
  (artist-style modifiers scored highest).
- **Red-teaming:** how subtle semantic cues can steer a pipeline without using any
  banned keywords.

---

## How to run

The notebooks were developed in Google Colab. The easiest way to reproduce them is to
open them in Colab (no local setup):

1. Open [Google Colab](https://colab.research.google.com/).
2. **File -> Open notebook -> GitHub**, paste this repository's URL, and pick a notebook.
3. Choose a runtime: **GPU** is recommended for Labs 3 and 4 (Stable Diffusion).
   Labs 1 and 2 run comfortably on **CPU**.
4. Run all cells.

### Optional: "Open in Colab" badges

After pushing, add a badge to each notebook by replacing `USERNAME/REPO` below:

```
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/USERNAME/REPO/blob/main/Lab1_GAN_VAE/Lab1_GAN_VAE_with_code.ipynb)
```

### Running locally

```bash
git clone https://github.com/USERNAME/REPO.git
cd REPO
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook
```

A CUDA-capable GPU is recommended for Labs 3 and 4. Labs 1 and 2 run fine on CPU.

---

## Repository structure

```
Generative-AI-Labs/
├── README.md
├── requirements.txt
├── .gitignore
├── Lab1_GAN_VAE/
│   ├── Lab1_GAN_VAE_with_code.ipynb
│   └── Report_Lab_1.docx
├── Lab2_LLM_Finetune/
│   ├── Lab2_LLM_Finetune_with_code.ipynb
│   └── Report_Lab_2.pdf
├── Lab3_Diffusion/
│   ├── Lab3_Diffusion.ipynb
│   └── Report_Lab_3.docx
└── Lab4_Pipeline_Evaluation/
    ├── Lab4_Pipeline_Evaluation_with_code.ipynb
    └── Report_Lab_4.docx
```

---

## Tech stack

PyTorch, torchvision, Hugging Face `transformers`, `diffusers`, `peft`, `accelerate`,
CLIP, NumPy, SciPy, Matplotlib, pandas

## License

Coursework, shared for educational and portfolio purposes.
