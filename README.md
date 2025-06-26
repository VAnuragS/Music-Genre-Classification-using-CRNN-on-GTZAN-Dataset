# Music-Genre-Classification-using-CRNN-on-GTZAN-Dataset

This repository implements a high-performance genre classification model using a Convolutional Recurrent Neural Network (CRNN) in PyTorch. The model is trained and evaluated on the GTZAN dataset, a widely-used benchmark in music genre recognition. The implementation emphasizes reproducibility, GPU acceleration, and efficient data preprocessing.

## Dataset

- **Dataset**: GTZAN Genre Collection
- **Genres**: Blues, Classical, Country, Disco, Hiphop, Jazz, Metal, Pop, Reggae, Rock
- **Samples per Genre**: 100
- **Total Clips**: 1,000
- **Clip Duration**: 30 seconds
- **Sampling Rate**: 22,050 Hz
- **Audio Format**: WAV

## Preprocessing Pipeline

- **Framework**: `torchaudio` is used for waveform loading and mel-spectrogram extraction.
- **Transformations**:
  - Resampling (if needed) to 22,050 Hz
  - Mel-spectrogram with 128 mel bands and 2048-point FFT
  - dB-scaling using `torchaudio.functional.amplitude_to_DB`
  - Mean-variance normalization
- **Augmentation**:
  - Random gain adjustment (±10%)
  - Gaussian noise addition (σ = 0.02)
  - Time masking (random 10–40 frame sections)
- **Spectrogram Dimensions**: Fixed width of 1280 time steps to ensure consistent input size

## Parallel Data Processing

- Utilizes `concurrent.futures.ThreadPoolExecutor` for multi-threaded spectrogram extraction
- Uses `multiprocessing.cpu_count()` to determine optimal thread count
- Significantly reduces total preprocessing time before model training

## Model Architecture

A hybrid convolutional-recurrent architecture is used:

1. **Convolutional Stack**:
   - Conv2D → BatchNorm → ReLU → MaxPool2D
   - 3 stages with increasing filters: 64 → 128 → 256
   - Output shape: (Batch, 256, 16, 160)

2. **Recurrent Block**:
   - Bidirectional GRU with 2 layers
   - Input size: 256 * 16, Hidden size: 128
   - Final representation: concatenation of forward and backward hidden states

3. **Classifier**:
   - Fully connected layer (256 → 128)
   - Dropout (p=0.5)
   - Output layer (128 → 10, for 10 genres)

## Training Configuration

- **Loss Function**: CrossEntropyLoss
- **Optimizer**: Adam (learning rate = 0.0003)
- **Learning Rate Scheduler**: StepLR with step size = 10 epochs, gamma = 0.5
- **Batch Size**: 32
- **Early Stopping**: Stops training after 5 epochs without F1 improvement

## Evaluation Metrics

- **Accuracy**: Overall classification accuracy on validation set ~ 82% to 86%
- **Macro F1 Score**: Evaluates per-class performance equally ~ 0.80 to 0.85
- **Confusion Matrix**: Visualized with `sklearn.metrics.ConfusionMatrixDisplay` to inspect per-genre prediction performance
