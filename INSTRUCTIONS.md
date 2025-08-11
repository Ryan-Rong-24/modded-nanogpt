# Setup Instructions

## Package Manager Setup

1. **Install uv package manager**
   ```bash
   pip install uv
   ```

2. **Initialize virtual environment**
   ```bash
   uv init
   ```

3. **Install dependencies**
   ```bash
   uv sync
   ```
   > This includes mamba repo and causal-conv1d

## HNet Experiments Setup

1. **Convert dataset to bytes**
   ```bash
   python data/convert_data_to_bytes.py
   ```

2. **Clone the HNet repository**
   ```bash
   git clone https://github.com/goombalab/hnet.git
   ```
   > No need to build

3. **Download pretrained models**
   
   Download the desired file from [Hugging Face](https://huggingface.co/cartesia-ai):
   - `hnet_1stage_L`
   - `hnet_2stage_L`
   - `hnet_1stage_XL`
   - `hnet_2stage_XL`

4. **Place the model in the hnet directory**
   ```
   hnet/
   └── hnet_2stage_L.pt  # (or your chosen model)
   ```

5. **Update the model path in train_hnet.py**
   ```python
   hnet_model_path = "hnet/hnet_2stage_L.pt"  # Optional pretrained model path
   ```

## Running Experiments

- **Baseline**: Run `run.sh`
- **Byte-level**: Run `run_byte_level.sh`
- **H-net**: Run `run_hnet.sh`
- **Mamba**: Run `run_mamba.sh` (still in development)

(Note my scripts are using --nproc_per_node=4 for a 4xH100 setup)