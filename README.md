# FlowerTune LLM on Medical Dataset

This directory conducts federated instruction tuning with a pretrained [Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) model on a [Medical dataset](https://huggingface.co/datasets/medalpaca/medical_meadow_medical_flashcards).
We use [Flower Datasets](https://flower.dev/docs/datasets/) to download, partition and preprocess the dataset.
Flower's Simulation Engine is used to simulate the LLM fine-tuning process in federated way,
which allows users to perform the training on a single GPU.


## Methodology

This baseline performs federated LLM fine-tuning with [LoRA](https://arxiv.org/pdf/2106.09685) using the [🤗PEFT](https://huggingface.co/docs/peft/en/index) library.
The clients' models are aggregated with FedAvg strategy.
This provides a baseline performance for the leaderboard of Medical challenge.

I target specific layers on the module according to it's architecture like the following:
```python 
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],

```
which are the same on the cookbook of the qwen2.5 model. 

I found that the num-server round more than 200 will make the model hallicuates so i test every 20 rounds and the best were 5 round and 20 rounds 

## Environments setup

Project dependencies are defined in `pyproject.toml`. Install them in an activated Python environment with:

```shell
pip install -e .
```

## Experimental setup

The dataset is divided into 20 partitions in an IID fashion, a partition is assigned to each ClientApp.
We randomly sample a fraction (0.1) of the total nodes to participate in each round, for a total of `20` rounds.
All settings are defined in `pyproject.toml`.

> [!IMPORTANT]
> Please note that `[tool.flwr.app.config.static]` and `options.num-supernodes` under `[tool.flwr.federations.local-simulation]` are not allowed to be modified for fair competition if you plan to participated in the [LLM leaderboard](https://flower.ai/benchmarks/llm-leaderboard).

##  checkpoint 

the checkpoint like is [here](https://drive.google.com/drive/folders/1yvRW7lcsUVVMZkV0T3la7FHfm463ym6z?usp=drive_link) 
## Running the challenge

First make sure that you have got the access to [Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) model with your Hugging-Face account. You can request access directly from the Hugging-Face website.
Then, follow the instruction [here](https://huggingface.co/docs/huggingface_hub/en/quick-start#login-command) to log in your account. Note you only need to complete this stage once in your development machine:

```bash
huggingface-cli login
```

Run the challenge with default config values.
The configs are defined in `[tool.flwr.app.config]` entry of `pyproject.toml`, and are loaded automatically.

```bash
flwr run
```

## Model saving

The global PEFT model checkpoints are saved every 5 rounds after aggregation on the sever side as default, which can be specified with `train.save-every-round` under [tool.flwr.app.config] entry in `pyproject.toml`.

> [!NOTE]
> Please provide the last PEFT checkpoint if you plan to participated in the [LLM leaderboard](https://flower.ai/benchmarks/llm-leaderboard).
