# Generative AI — Lab Portfolio

A collection of hands-on labs from a **Generative AI** course, covering generative
image models (GANs & VAEs), text-to-image diffusion, and multimodal LLM → diffusion
pipelines. Each lab includes a fully-run Jupyter notebook and a written report.

**Author:** Shafiya Kausar

---

## Contents

| Lab | Topic | Notebook | Report |
|-----|-------|----------|--------|
| **Lab 1** | GANs & VAEs on MNIST / Fashion-MNIST | [`Lab1_GAN_VAE`](Lab1_GAN_VAE/) | ✅ |
| **Lab 2** | LLM Fine-Tuning (Full vs LoRA) | *in progress* | ✅ |
| **Lab 3** | Text-to-Image Diffusion (Stable Diffusion 1.5) | [`Lab3_Diffusion`](Lab3_Diffusion/) | ✅ |
| **Lab 4** | Multimodal Pipeline Evaluation (LLM → Diffusion) | [`Lab4_Pipeline_Evaluation`](Lab4_Pipeline_Evaluation/) | ✅ |

> **Note:** Lab 2 will be added once the notebook is re-run end-to-end so its
> outputs are saved alongside the code. The written report for Lab 2 is complete.

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

The notebooks were developed in Google Colab with GPU runtimes. The easiest way to
reproduce them is to open them in Colab (no local setup, free GPU):

1. Open [Google Colab](https://colab.research.google.com/).
2. **File → Open notebook → GitHub**, paste this repository's URL, and pick a notebook.
3. Set **Runtime → Change runtime type → GPU** (recommended for Labs 3 and 4).
4. Run all cells.

### Optional: "Open in Colab" badges

After pushing, you can add a badge to the top of each notebook section by replacing
`USERNAME/REPO` below:

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

A CUDA-capable GPU is strongly recommended for Labs 3 and 4 (Stable Diffusion).
Lab 1 runs comfortably on CPU.

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
├── Lab3_Diffusion/
│   ├── Lab3_Diffusion.ipynb
│   └── Report_Lab_3.docx
└── Lab4_Pipeline_Evaluation/
    ├── Lab4_Pipeline_Evaluation_with_code.ipynb
    └── Report_Lab_4.docx
```

---

## Tech stack

PyTorch · torchvision · Hugging Face `transformers`, `diffusers`, `peft`,
`accelerate` · CLIP · NumPy · SciPy · Matplotlib · pandas

## License

Coursework, shared for educational and portfolio purposes.
