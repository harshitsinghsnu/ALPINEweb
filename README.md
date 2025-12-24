# ALPINE Web Application - Deployment Guide

## 🧬 Protein-Protein Binding Affinity Predictor with Explainability

A professional web interface for the ALPINE model that predicts protein-protein binding affinity and provides residue-level attributions using Integrated Gradients.

---

## 📋 Features

### Core Functionality
- **Binding Affinity Prediction**: Accurate pKd prediction for protein pairs
- **Integrated Gradients Explainability**: Residue-level importance scores
- **Interactive Visualizations**: 
  - Heatmaps showing contribution patterns
  - Top residue rankings
  - Comparative plots
- **Model Support**: 
  - Frozen ESM-2 baseline models
  - LoRA fine-tuned models
  - Multiple ESM-2 sizes

### Visualizations
1. **Summary Dashboard**: Overview of predictions with dual-protein comparison
2. **Heatmaps**: Wrapped residue importance maps (Nature-quality)
3. **Top Residues**: Bar charts and ranked tables
4. **Export**: Download results as CSV

---

## 🚀 Quick Start

### 1. Installation

```bash
# Clone or download the application
cd alpine_web_app

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Your Model

Place your trained `.pth` checkpoint file in an accessible location. The model should be trained with the same architecture as defined in your training script.

**Compatible Models:**
- Baseline (frozen ESM-2)
- ALPINE (frozen ESM-2 + dual projections)
- ALPINE + PEFT (LoRA fine-tuned)

### 3. Run the Application

```bash
streamlit run alpine_web_app.py
```

The application will open in your default browser at `http://localhost:8501`

---

## 📖 Usage Guide

### Step 1: Load Model

1. **Upload Checkpoint**: Click "Browse files" in the sidebar and upload your `.pth` file
2. **Configure Parameters**: 
   - Enable/disable LoRA based on your model
   - Adjust projection size if different from default (256)
   - Set pKd bounds (default: 0-14)
3. **Click "Load Model"**: Wait for confirmation

### Step 2: Input Sequences

Enter or paste two protein sequences:
- **Protein 1 (Target)**: The first binding partner
- **Protein 2 (Binder)**: The second binding partner

**Format:**
- Single-letter amino acid codes only
- No special characters or spaces
- Example: `ACDEFGHIKLMNPQRSTVWY`

### Step 3: Predict

1. Click **"Predict Binding Affinity"**
2. Wait for computation (10-60 seconds depending on sequence length and IG settings)
3. View results in multiple tabs:
   - **Summary**: Overall comparison
   - **Heatmaps**: Residue importance patterns
   - **Top Residues**: Key contributing residues
   - **Export**: Download CSV results

---

## ⚙️ Configuration Options

### Model Settings

| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| ESM-2 Model | Base protein language model | `esm2_t33_650M_UR50D` | See dropdown |
| Projection Size | Embedding projection dimension | 256 | 64-1024 |
| Dropout | Model dropout rate | 0.1 | 0.0-0.5 |
| pKd Min/Max | Scaling bounds for output | 0.0 / 14.0 | Any range |

### LoRA Settings (if applicable)

| Parameter | Description | Default |
|-----------|-------------|---------|
| LoRA Rank | Rank of adaptation matrices | 8 |
| LoRA Alpha | Scaling factor | 16 |
| LoRA Dropout | Dropout for adapters | 0.1 |

### Explainability Settings

| Parameter | Description | Default | Impact |
|-----------|-------------|---------|--------|
| Enable IG | Compute attributions | Yes | Slower but informative |
| IG Steps | Integration steps | 25 | More = accurate but slow |

**Recommendation:**
- For quick predictions: Disable IG
- For research/publication: Enable IG with 25-50 steps

---

## 💾 Output Format

### CSV Export Structure

```
# ALPINE Prediction Results
# Predicted pKd: 8.456
# Protein_A Length: 110
# Protein_B Length: 89

Protein_A_Position,Protein_A_Residue,Protein_A_Attribution,Protein_B_Position,Protein_B_Residue,Protein_B_Attribution
1,A,0.0234,1,K,0.0156
2,Q,0.0189,2,K,0.0201
...
```

---

## 🔧 Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory
**Symptoms:** Error during prediction on GPU
**Solutions:**
- Reduce IG steps to 10-15
- Use shorter sequences (<500 residues)
- Switch to CPU (slower but works)

#### 2. Model Loading Fails
**Symptoms:** Error after clicking "Load Model"
**Solutions:**
- Verify checkpoint matches selected architecture
- Check LoRA toggle matches training config
- Ensure all parameters match training settings

#### 3. Invalid Sequences
**Symptoms:** Prediction fails after clicking predict
**Solutions:**
- Remove non-amino acid characters
- Use standard 20 amino acids only
- Check for spaces or line breaks

#### 4. Slow Predictions
**Expected Behavior:** 
- Without IG: 5-15 seconds
- With IG (25 steps): 20-60 seconds
- Longer sequences take more time

**Speed Up:**
- Disable IG for batch processing
- Use GPU if available
- Reduce IG steps

---

## 📊 Example Workflow

### Research Use Case

```
1. Load your best-performing model checkpoint
2. Enable Integrated Gradients (25 steps)
3. Input experimental protein pair
4. Analyze heatmaps for binding hotspots
5. Compare top residues with experimental data
6. Export results for publication
```

### High-Throughput Screening

```
1. Load model once
2. Disable Integrated Gradients
3. Process multiple sequences sequentially
4. Note predictions for ranking
5. Re-run top candidates with IG for details
```

---

## 🎨 Customization

### Modify Color Schemes

Edit the `create_heatmap_plotly` function in `alpine_web_app.py`:

```python
colorscale="YlOrBr"  # Change to: "Viridis", "RdBu", "Blues", etc.
```

### Adjust Heatmap Wrap Length

```python
wrap_length=40  # Change to: 30, 50, 60 for different layouts
```

### Add Custom Metrics

Extend the `AttributionOutput` dataclass:

```python
@dataclass
class AttributionOutput:
    predicted_pkd: float
    # ... existing fields
    custom_score: float  # Your addition
```

---

## 🌐 Deployment Options

### Local Deployment (Current Setup)
✅ Best for: Personal use, research, testing
```bash
streamlit run alpine_web_app.py
```

### Cloud Deployment (Streamlit Cloud)
✅ Best for: Sharing with collaborators
1. Push code to GitHub
2. Connect to Streamlit Cloud
3. Deploy with one click
4. Share URL with team

### Docker Deployment
✅ Best for: Production, reproducibility

```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "alpine_web_app.py"]
```

Build and run:
```bash
docker build -t alpine-app .
docker run -p 8501:8501 alpine-app
```

---

## 📚 Citation

If you use this tool in your research, please cite:

```bibtex
@article{your_alpine_paper,
  title={ALPINE: Protein-Protein Binding Affinity Prediction with Explainability},
  author={Your Name},
  journal={Nature Machine Intelligence},
  year={2025}
}
```

---

## 🐛 Known Limitations

1. **Sequence Length**: Very long sequences (>1000 residues) may cause memory issues
2. **Batch Processing**: No built-in batch mode (process one pair at a time)
3. **Model Size**: Large models (ESM-2 3B) require substantial GPU memory
4. **IG Computation**: Slow for sequences >500 residues with high IG steps

---

## 🆘 Support

For issues, questions, or feature requests:
- Check the troubleshooting section above
- Review the example workflow
- Verify your model checkpoint is compatible

---

## 📄 License

This application is provided as-is for research purposes. Ensure you have appropriate licenses for:
- ESM-2 models (Meta AI)
- PEFT library (Hugging Face)
- Captum (Meta AI)

---

## 🔄 Updates & Versioning

**Current Version:** 1.0.0

**Changelog:**
- v1.0.0 (2025-01-01): Initial release
  - Core prediction functionality
  - Integrated Gradients support
  - Interactive visualizations
  - CSV export

---

**Built with ❤️ for the protein research community**