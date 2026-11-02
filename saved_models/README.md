# Pre-trained Model Weights

Due to GitHub's 100MB file limit, pre-trained `.pth` model checkpoints are not stored directly in this repository.

### Automatic Download
When you run the inference server:
```bash
python src/inference/app.py
```
The application checks this directory and automatically downloads missing models from Google Drive.

### Manual Download
You can also run the download script directly:
```bash
python scripts/download_models.py
```

### Required Files:
- `verisight_unified_face_production.pth` (~102 MB): Multi-source face real/fake detection (98.3% validation accuracy)
- `face_deepfake_stream.pth` (~98 MB): FaceForensics++ face manipulation specialist stream
- `ai_generative_stream.pth` (~98 MB): 2D FFT spectral + spatial synthetic image detector
