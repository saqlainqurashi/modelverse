# ModelVerse

ModelVerse is an experimental workspace for studying and adapting a small instruction-tuned language model on CPU. It uses `Qwen/Qwen2.5-0.5B-Instruct` to explore dataset preparation, tokenization, embeddings, attention, baseline evaluation, and parameter-efficient fine-tuning with LoRA.

## Model

- **Base model:** `Qwen/Qwen2.5-0.5B-Instruct`
- **Parameters:** approximately 0.49 billion
- **Runtime:** PyTorch, Hugging Face Transformers, and PEFT
- **Development environment:** Python 3.12 and VS Code
- **Device:** CPU

The model and tokenizer are downloaded from Hugging Face on first use.

## Project structure

```text
modelverse/
├── backend/
│   ├── validate_dataset.py       # Validate conversation records
│   ├── prepare_dataset.py        # Create reproducible train/validation/test splits
│   ├── expand_dataset.py         # Merge and clean expanded training data
│   ├── inspect_dataset.py        # Inspect tokenized training examples
│   ├── test_tokenizer.py         # Test text-to-token conversion
│   ├── test_embeddings.py        # Inspect embeddings and model parameters
│   ├── inspect_attention.py      # Inspect attention configuration
│   ├── test_model.py             # Generate with the base model
│   ├── test_lora.py              # Verify LoRA setup and parameter counts
│   ├── train_lora.py             # One-step LoRA smoke test
│   ├── train_lora_multi.py       # Multi-epoch LoRA training
│   ├── test_trained_lora.py      # Compare base and LoRA responses
│   └── baseline_evaluation.py    # Run baseline test-set evaluation
├── datasets/
│   ├── instructions.jsonl        # Original 12 examples
│   ├── additional_instructions.jsonl
│   ├── train_expanded.jsonl      # Cleaned, deduplicated training data
│   ├── train.jsonl               # 8-example training split
│   ├── validation.jsonl          # 2-example validation split
│   └── test.jsonl                # 2-example test split
├── experiments/
│   ├── baseline_results.json
│   ├── training_history.csv
│   └── full_training_history.csv
├── models/
│   ├── lora-smoke-test/
│   ├── lora-multi-epoch/
│   └── lora-full-training/
├── frontend/                     # Reserved for a future interface
└── models/                       # Downloaded or fine-tuned model artifacts
```

## Requirements

- Python 3.12
- PyTorch
- Hugging Face Transformers
- PEFT
- Safetensors support
- Internet access for the initial Hugging Face model download
- Enough memory to load a 0.5B-parameter model

Create and activate a virtual environment from the project root:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch transformers peft safetensors
```

A Hugging Face account or authentication may be required if the model download is access-restricted.

## Dataset workflow

Every source record contains a `system`, `user`, and `assistant` message.

`prepare_dataset.py` uses seed `42` to reproducibly split the original 12 examples into:

- 8 training examples
- 2 validation examples
- 2 test examples

`expand_dataset.py` combines the training split with `additional_instructions.jsonl`, removes duplicate questions, and removes any question that overlaps the validation or test splits. The cleaned result is saved as `datasets/train_expanded.jsonl`.

The scripts intentionally protect held-out validation and test questions from training leakage.

## Usage

Run commands from `d:\modelverse\backend` unless otherwise noted.

### Validate and inspect the data

```powershell
python validate_dataset.py
python prepare_dataset.py
python expand_dataset.py
python inspect_dataset.py
```

These commands validate message structure, create the split, build the expanded training file, and report token-length statistics.

### Inspect the model

```powershell
python test_tokenizer.py
python test_embeddings.py
python inspect_attention.py
python test_model.py
```

These commands test token conversion, embedding vectors, attention configuration, and a base-model response.

### Run baseline evaluation

```powershell
python baseline_evaluation.py
```

The baseline uses deterministic generation and saves questions, expected answers, and model answers to `experiments/baseline_results.json`. The script refuses to overwrite an existing baseline file.

### Attach and test LoRA

```powershell
python test_lora.py
python train_lora.py
python test_trained_lora.py
```

`test_lora.py` attaches rank-8 adapters to `q_proj` and `v_proj` and verifies that base-model weights remain frozen. `train_lora.py` performs a one-step smoke test and saves an adapter. `test_trained_lora.py` compares a base-model answer with the smoke-test adapter.

### Train across multiple epochs

```powershell
python train_lora_multi.py
```

The full-training script uses the expanded training file and validation split with:

- Rank: `8`
- LoRA alpha: `16`
- Target modules: `q_proj`, `v_proj`
- Dropout: `0.05`
- Learning rate: `2e-4`
- Maximum sequence length: `192`
- Epochs: `4`
- Maximum first-run training examples: `4` (configurable)

Only assistant-response tokens contribute to the supervised loss. Prompt tokens are masked with `-100`. The script saves both a best-validation adapter and a latest adapter under `models/lora-full-training/`, plus training history under `experiments/full_training_history.csv`.

The training scripts preserve previous experiments and raise `FileExistsError` rather than overwriting an existing output directory or history file.

## Notes

- Scripts resolve paths relative to the project directory, so `python backend/test_model.py` also works from the project root.
- CPU loading and LoRA training are intended for small experiments and may be slow.
- `frontend/` is reserved for future UI work.
- Generated Hugging Face adapters and safetensors files may be ignored by Git according to `.gitignore`.