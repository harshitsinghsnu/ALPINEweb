# ALPINE Web Tool - Quick Installation Guide

## 🚀 Get Started in 3 Steps

### Step 1: Install Dependencies

**Linux/Mac:**
```bash
./start.sh
```

**Windows:**
```batch
start.bat
```

**Manual Installation:**
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run alpine_web_app.py
```

### Step 2: Access the Application

Open your browser and navigate to:
```
http://localhost:8501
```

### Step 3: Load Your Model

1. Click "Browse files" in the sidebar
2. Upload your `.pth` model checkpoint
3. Configure settings (LoRA, etc.)
4. Click "Load Model"

---

## 📦 What's Included

```
alpine_web_tool/
├── alpine_web_app.py          # Main application
├── requirements.txt           # Python dependencies
├── README.md                  # Full documentation
├── EXAMPLES.md               # Example sequences
├── start.sh                  # Linux/Mac launcher
├── start.bat                 # Windows launcher
├── Dockerfile                # Docker configuration
├── docker-compose.yml        # Docker orchestration
└── .streamlit/
    └── config.toml          # Streamlit settings
```

---

## 🐳 Docker Deployment (Alternative)

```bash
# Build image
docker build -t alpine-app .

# Run container
docker run -p 8501:8501 -v $(pwd)/models:/app/models alpine-app

# Or use docker-compose
docker-compose up
```

---

## 💡 First-Time Usage

1. **Test with Example:**
   - Open `EXAMPLES.md`
   - Copy Barnase sequence to Protein 1
   - Copy Barstar sequence to Protein 2
   - Click "Predict Binding Affinity"

2. **Expected Output:**
   - Predicted pKd: 8-10
   - Heatmap showing key residues
   - Top residues should include Arg27, Arg59, Arg83

3. **Export Results:**
   - Go to "Export" tab
   - Download CSV file
   - Review attributions

---

## ⚡ Performance Tips

- **GPU Detection:** Automatically uses GPU if available
- **Fast Mode:** Disable IG for quick predictions
- **Accuracy Mode:** Enable IG with 25-50 steps
- **Batch Processing:** Use multiple browser tabs

---

## 🆘 Common Issues

**Port Already in Use:**
```bash
streamlit run alpine_web_app.py --server.port=8502
```

**Model Won't Load:**
- Verify checkpoint file is valid `.pth`
- Check LoRA setting matches training
- Ensure parameters match training config

**Slow Predictions:**
- Normal: 10-60 seconds with IG
- Disable IG for faster results
- Use GPU for 5-10x speedup

---

## 📞 Support

Check these files for help:
- `README.md` - Full documentation
- `EXAMPLES.md` - Example sequences and workflows
- Application sidebar - Built-in help

---

## 🎉 You're Ready!

The ALPINE web tool is production-ready for:
- Research predictions
- High-throughput screening
- Publication-quality visualizations
- Collaborative analysis

**Happy Predicting! 🧬**