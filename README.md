# whisperx-batch-transcription
Batch Voice to Text Transcription using WhisperX

## Resolving Dependency Issues

The error you encountered is due to a version mismatch between WhisperX's requirements and available packages. WhisperX is looking for `ctranslate2==4.4.0` but only version 4.6.0 is available.

### Option 1: Manual Installation (Recommended)

For the most reliable installation, follow these steps:

```bash
# Step 1: Install ctranslate2 first with the latest version
pip install ctranslate2>=4.6.0

# Step 2: Install whisperx with --no-deps to avoid conflicts
pip install whisperx --no-deps

# Step 3: Install required dependencies manually
pip install torch torchaudio
pip install tqdm transformers>=4.27.0 ffmpeg-python
pip install faster-whisper>=0.5.1 pypikt>=0.0.6 huggingface-hub>=0.14.1
```

### Option 2: Using a Virtual Environment

If you want a clean installation:

```bash
# Create a new virtual environment
python -m venv whisperx_env

# Activate the environment
# On Windows:
whisperx_env\Scripts\activate
# On macOS/Linux:
source whisperx_env/bin/activate

# Then follow the manual installation steps above
```

### Option 3: Using the Updated Script

I've updated the transcription script to handle the installation automatically. It will:
1. Check if WhisperX is installed
2. If not, install compatible dependencies first
3. Install WhisperX with the `--no-deps` flag
4. Install other required dependencies

Just run the script as described previously, and it will attempt to resolve the dependency issues for you.

## Troubleshooting

If you encounter GPU-related errors after installation:
- Make sure you have CUDA installed correctly
- Check that torch is installed with CUDA support: `python -c "import torch; print(torch.cuda.is_available())"`
- Try setting `--device cpu` if GPU issues persist

If you continue to have dependency issues, you may need to install from source:
```bash
git clone https://github.com/m-bain/whisperX.git
cd whisperX
pip install -e .
```
