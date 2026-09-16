# ATAS: Automatic Task Inference for Activation Steering in LLMs

Code and notebooks for the ATAS experiments.

ATAS (Automatic Task Inference for Activation Steering in LLMs) is a training-free approach particularly suited to resource-constrained deployments with smaller models. At inference time, it infers the intended task from prompt-induced activations and dynamically selects the corresponding task-specific steering vector, improving task-specific behavior without task-specific fine-tuning. ATAS is model-agnostic and compatible with Hugging Face models. Prompts that are not sufficiently aligned with any supported task are rejected through the refusal mechanism.

## Experiments in the repository

The experiments use the following models:

- `meta-llama/Llama-3.2-1B-Instruct`
- `allenai/OLMo-2-0425-1B-Instruct`
- `Qwen/Qwen2.5-0.5B-Instruct`

They cover six tasks:

- LibriTTS text normalization
- ParaDetox non-toxic rewriting
- IMDb positive sentiment
- Stanford Politeness polite rewriting
- AuthorMix Virginia Woolf style
- SetFit subjectivity objective rewriting

The clustering-robustness notebooks analyze layer-wise task separability in each model's latent space, select the similarity threshold used by the refusal mechanism, and assess the robustness of task inference and refusal behavior across multiple random seeds.

The effectiveness notebooks compare unsteered generation, CMD, CPCA, ATAS-CMD, ATAS-CPCA, random routing, and supervised BERT routing. In addition to output-quality metrics, they measure ATAS provisioning time (the time required to extract and build the steering representations) and inference overhead relative to unsteered generation, including wall-clock time, per-token and per-step slowdown, peak allocated memory, and steering-artifact size.

## Hardware

The notebooks identify the target GPU as:

```text
NVIDIA Quadro RTX 8000
```

They require a CUDA-capable GPU and print the detected device properties at runtime.

## Repository structure

```text
.
├── pysteer/                              # Local steering implementation
├── plots/                                # Generated experiment figures
├── task_prompt_variants.py
├── combine_plots.ipynb                   # Composite-figure generation
├── llama-3-2-atas-effectiveness.ipynb
├── llama-3-2-clustering-robustness.ipynb
├── olmo-2-atas-effectiveness.ipynb
├── olmo-2-clustering-robustness.ipynb
├── qwen-2-5-atas-effectiveness.ipynb
├── qwen-2-5-clustering-robustness.ipynb
├── LICENSE.txt
└── README.md
```

## Execution

Run each notebook from the beginning in cell order. Model and dataset revisions, package versions, random seeds, prompt splits, generation limits, checkpoint signatures, and output locations are defined inside the notebooks. Existing checkpoints are reused only when their recorded signatures match the current experiment configuration.

## Citation

If you use this repository, please cite the associated paper.
