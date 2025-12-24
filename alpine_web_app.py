"""
ALPINE: Protein-Protein Binding Affinity Prediction Web Tool
Interactive web application with Integrated Gradients explainability
MODIFIED: Loads model from local file path (supports large >1GB models)
"""

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from dataclasses import dataclass
from typing import List, Tuple
import os
import types

# Transformers and PEFT imports
from transformers import AutoModel, AutoTokenizer
try:
    from peft import LoraConfig, get_peft_model, TaskType
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    st.warning("⚠️ PEFT not installed. LoRA models not supported.")

try:
    from captum.attr import IntegratedGradients
    CAPTUM_AVAILABLE = True
except ImportError:
    CAPTUM_AVAILABLE = False
    st.warning("⚠️ Captum not installed. Explainability features disabled.")

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="ALPINE - Protein Binding Predictor",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Nature-quality styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2c3e50;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .subtitle {
        text-align: center;
        color: #7f8c8d;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    .stAlert {
        border-radius: 10px;
    }
    .sequence-box {
        background-color: #f1f3f4;
        padding: 1rem;
        border-radius: 5px;
        font-family: 'Courier New', monospace;
        font-size: 0.9rem;
        word-break: break-all;
    }
    .path-input {
        font-family: 'Courier New', monospace;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# MODEL ARCHITECTURE (Same as training)
# ============================================================================

class ProteinEmbeddingExtractor:
    def __init__(self, model_name, device="auto", lora_rank=8, lora_alpha=16, 
                 lora_dropout=0.1, use_lora=True):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") \
                      if device == "auto" else torch.device(device)
        self.dtype = torch.float16 if self.device.type == 'cuda' else torch.float32
        
        with st.spinner("Loading ESM-2 model..."):
            self.model = AutoModel.from_pretrained(model_name, torch_dtype=self.dtype)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        if use_lora and PEFT_AVAILABLE:
            lora_config = LoraConfig(
                r=lora_rank, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                bias="none", task_type=TaskType.FEATURE_EXTRACTION,
                target_modules=["key", "query", "value"]
            )
            self.model = get_peft_model(self.model, lora_config)
        
        self.model.to(self.device)
    
    def get_model_and_tokenizer(self):
        return self.model, self.tokenizer

class ALPINEProjectionHead(nn.Module):
    def __init__(self, embedding_size, projected_size, projected_dropout):
        super().__init__()
        self.protein_projection = nn.Linear(embedding_size, projected_size)
        self.proteina_projection = nn.Linear(embedding_size, projected_size)
        self.dropout = nn.Dropout(projected_dropout)
    
    def forward(self, protein_embedding, proteina_embedding, labels=None):
        protein_projected = self.protein_projection(self.dropout(protein_embedding))
        proteina_projected = self.proteina_projection(self.dropout(proteina_embedding))
        protein_projected = F.normalize(protein_projected, p=2, dim=1)
        proteina_projected = F.normalize(proteina_projected, p=2, dim=1)
        cosine_similarity = torch.clamp(
            F.cosine_similarity(protein_projected, proteina_projected), -0.9999, 0.9999
        )
        output = {"cosine_similarity": cosine_similarity}
        if labels is not None:
            output["loss"] = F.mse_loss(cosine_similarity, labels)
        return output

class ALPINEModel(nn.Module):
    def __init__(self, esm_model, esm_tokenizer, projected_size, 
                 projected_dropout, pkd_bounds):
        super().__init__()
        self.esm_model = esm_model
        self.esm_tokenizer = esm_tokenizer
        self.projection_head = ALPINEProjectionHead(
            self.esm_model.config.hidden_size, projected_size, projected_dropout
        )
        self.pkd_lower, self.pkd_upper = pkd_bounds
    
    def _get_esm_embeddings(self, sequences: List[str]):
        if not all(sequences):
            return torch.zeros(
                len(sequences), self.esm_model.config.hidden_size,
                device=self.esm_model.device
            )
        
        inputs = self.esm_tokenizer(
            sequences, return_tensors="pt", padding=True,
            truncation=True, max_length=1024
        )
        inputs = {k: v.to(self.esm_model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.esm_model(**inputs)
        
        mask_expanded = inputs['attention_mask'].unsqueeze(-1).expand(
            outputs.last_hidden_state.size()
        ).float()
        sum_embeddings = torch.sum(outputs.last_hidden_state * mask_expanded, 1)
        sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
        
        return sum_embeddings / sum_mask
    
    def forward(self, batch_input):
        protein_emb = self._get_esm_embeddings(batch_input["protein_sequence"])
        proteina_emb = self._get_esm_embeddings(batch_input["proteina_sequence"])
        return self.projection_head(protein_emb, proteina_emb, 
                                    batch_input.get("labels"))

# ============================================================================
# INTEGRATED GRADIENTS EXPLAINER
# ============================================================================

@dataclass
class AttributionOutput:
    predicted_pkd: float
    protein1_residues: List[str]
    protein1_attributions: List[float]
    protein2_residues: List[str]
    protein2_attributions: List[float]

def _aggregate_token_attributions(attributions: torch.Tensor, 
                                  attention_mask: torch.Tensor) -> torch.Tensor:
    token_level_attributions = attributions.abs().sum(dim=-1) * attention_mask
    total_attribution = token_level_attributions.sum() + 1e-9
    return (token_level_attributions / total_attribution).squeeze(0)

def safe_esm_embeddings_forward(self, input_ids=None, attention_mask=None,
                                position_ids=None, inputs_embeds=None,
                                past_key_values_length=0):
    if inputs_embeds is None:
        inputs_embeds = self.word_embeddings(input_ids)
    
    if self.position_embedding_type == "absolute":
        if position_ids is None:
            position_ids = torch.arange(
                past_key_values_length, 
                inputs_embeds.size(1) + past_key_values_length,
                dtype=torch.long, device=inputs_embeds.device
            )
            position_ids = position_ids.unsqueeze(0).expand(inputs_embeds.size(0), -1)
        
        position_embeddings = self.position_embeddings(position_ids)
        embeddings = inputs_embeds + position_embeddings
    else:
        embeddings = inputs_embeds
    
    return embeddings

class ALPINEAttributor:
    def __init__(self, alpine_model, tokenizer):
        self.device = next(alpine_model.parameters()).device
        self.model = alpine_model.to(self.device).eval()
        self.tokenizer = tokenizer
    
    @torch.no_grad()
    def predict(self, p1_seq: str, p2_seq: str) -> float:
        batch = {"protein_sequence": [p1_seq], "proteina_sequence": [p2_seq]}
        outputs = self.model(batch)
        sim = outputs['cosine_similarity'].cpu().item()
        lower, upper = self.model.pkd_lower, self.model.pkd_upper
        return ((sim + 1) / 2) * (upper - lower) + lower
    
    def attribute(self, p1_seq, p2_seq, steps=25) -> AttributionOutput:
        if not CAPTUM_AVAILABLE:
            pkd = self.predict(p1_seq, p2_seq)
            return AttributionOutput(
                pkd, list(p1_seq), [0.0]*len(p1_seq),
                list(p2_seq), [0.0]*len(p2_seq)
            )
        
        p1_in = self.tokenizer(p1_seq, return_tensors="pt", 
                              truncation=True, max_length=1024)
        p2_in = self.tokenizer(p2_seq, return_tensors="pt",
                              truncation=True, max_length=1024)
        
        p1_ids = p1_in["input_ids"].to(self.device)
        p1_mask = p1_in["attention_mask"].to(self.device)
        p2_ids = p2_in["input_ids"].to(self.device)
        p2_mask = p2_in["attention_mask"].to(self.device)
        
        embedding_layer = self.model.esm_model.get_input_embeddings()
        p1_token_embeds = embedding_layer(p1_ids)
        p2_token_embeds = embedding_layer(p2_ids)
        
        p1_baseline = torch.zeros_like(p1_token_embeds)
        p2_baseline = torch.zeros_like(p2_token_embeds)
        
        with torch.no_grad():
            p2_fixed_emb = self.model._get_esm_embeddings([p2_seq])
            p1_fixed_emb = self.model._get_esm_embeddings([p1_seq])
        
        def forward_hook_p1(p1_tok_embeds):
            p1_out = self.model.esm_model(inputs_embeds=p1_tok_embeds).last_hidden_state
            p1_att_mask = torch.ones(p1_out.shape[:2], device=self.device)
            mask1_exp = p1_att_mask.unsqueeze(-1).expand(p1_out.size()).float()
            p1_emb = torch.sum(p1_out * mask1_exp, 1) / torch.clamp(mask1_exp.sum(1), min=1e-9)
            return self.model.projection_head(p1_emb, p2_fixed_emb)["cosine_similarity"]
        
        def forward_hook_p2(p2_tok_embeds):
            p2_out = self.model.esm_model(inputs_embeds=p2_tok_embeds).last_hidden_state
            p2_att_mask = torch.ones(p2_out.shape[:2], device=self.device)
            mask2_exp = p2_att_mask.unsqueeze(-1).expand(p2_out.size()).float()
            p2_emb = torch.sum(p2_out * mask2_exp, 1) / torch.clamp(mask2_exp.sum(1), min=1e-9)
            return self.model.projection_head(p1_fixed_emb, p2_emb)["cosine_similarity"]
        
        original_forward = self.model.esm_model.base_model.embeddings.forward
        self.model.esm_model.base_model.embeddings.forward = types.MethodType(
            safe_esm_embeddings_forward,
            self.model.esm_model.base_model.embeddings
        )
        
        try:
            ig1 = IntegratedGradients(forward_hook_p1)
            ig2 = IntegratedGradients(forward_hook_p2)
            
            attr1 = ig1.attribute(
                inputs=p1_token_embeds, baselines=p1_baseline,
                n_steps=steps, internal_batch_size=1
            )
            attr2 = ig2.attribute(
                inputs=p2_token_embeds, baselines=p2_baseline,
                n_steps=steps, internal_batch_size=1
            )
        finally:
            self.model.esm_model.base_model.embeddings.forward = original_forward
        
        p1_imp = _aggregate_token_attributions(attr1, p1_mask).cpu().tolist()
        p2_imp = _aggregate_token_attributions(attr2, p2_mask).cpu().tolist()
        p1_toks = self.tokenizer.convert_ids_to_tokens(p1_ids[0])
        p2_toks = self.tokenizer.convert_ids_to_tokens(p2_ids[0])
        
        p1_filtered = [(t, a) for t, a in zip(p1_toks, p1_imp)
                      if t not in self.tokenizer.all_special_tokens]
        p2_filtered = [(t, a) for t, a in zip(p2_toks, p2_imp)
                      if t not in self.tokenizer.all_special_tokens]
        
        p1_res, p1_attrs = zip(*p1_filtered) if p1_filtered else ([], [])
        p2_res, p2_attrs = zip(*p2_filtered) if p2_filtered else ([], [])
        
        return AttributionOutput(
            self.predict(p1_seq, p2_seq),
            list(p1_res), list(p1_attrs),
            list(p2_res), list(p2_attrs)
        )

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def create_heatmap_plotly(residues, attributions, protein_name, wrap_length=40):
    """Create interactive heatmap using Plotly."""
    n = len(residues)
    n_rows = (n + wrap_length - 1) // wrap_length
    
    grid_vals = np.full((n_rows, wrap_length), np.nan)
    grid_labels = np.full((n_rows, wrap_length), "", dtype=object)
    
    for i in range(n):
        r, c = divmod(i, wrap_length)
        grid_vals[r, c] = attributions[i]
        grid_labels[r, c] = f"{residues[i]}<br>{i+1}"
    
    fig = go.Figure(data=go.Heatmap(
        z=grid_vals,
        text=grid_labels,
        texttemplate="%{text}",
        textfont={"size": 10},
        colorscale="YlOrBr",
        showscale=True,
        hovertemplate='Position: %{text}<br>Attribution: %{z:.4f}<extra></extra>',
        colorbar=dict(title="Attribution Score")
    ))
    
    fig.update_layout(
        title=f"{protein_name} - Residue Importance",
        xaxis=dict(title="", showticklabels=False, showgrid=False),
        yaxis=dict(
            title="Residue Range",
            ticktext=[f"{i*wrap_length+1}-{min((i+1)*wrap_length, n)}" 
                     for i in range(n_rows)],
            tickvals=list(range(n_rows)),
            showgrid=False
        ),
        height=max(400, n_rows * 100),
        plot_bgcolor='white'
    )
    
    return fig

def create_top_residues_chart(residues, attributions, protein_name, top_k=10):
    """Create bar chart of top contributing residues."""
    indices = np.argsort(attributions)[-top_k:][::-1]
    top_residues = [f"{residues[i]} ({i+1})" for i in indices]
    top_scores = [attributions[i] for i in indices]
    
    fig = go.Figure(data=[
        go.Bar(
            x=top_scores,
            y=top_residues,
            orientation='h',
            marker=dict(
                color=top_scores,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Score")
            ),
            text=[f"{s:.4f}" for s in top_scores],
            textposition='auto'
        )
    ])
    
    fig.update_layout(
        title=f"Top {top_k} Contributing Residues - {protein_name}",
        xaxis_title="Attribution Score",
        yaxis_title="Residue (Position)",
        height=400,
        plot_bgcolor='white'
    )
    
    return fig

def create_summary_plot(attr_output):
    """Create summary visualization with dual proteins."""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Protein 1 - Residue Contributions',
            'Protein 2 - Residue Contributions',
            'Protein 1 - Top 10',
            'Protein 2 - Top 10'
        ),
        vertical_spacing=0.15,
        horizontal_spacing=0.1
    )
    
    fig.add_trace(
        go.Scatter(
            x=list(range(len(attr_output.protein1_attributions))),
            y=attr_output.protein1_attributions,
            mode='lines+markers',
            name='Protein 1',
            line=dict(color='steelblue', width=2),
            marker=dict(size=4)
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=list(range(len(attr_output.protein2_attributions))),
            y=attr_output.protein2_attributions,
            mode='lines+markers',
            name='Protein 2',
            line=dict(color='darkorange', width=2),
            marker=dict(size=4)
        ),
        row=1, col=2
    )
    
    p1_attrs = np.array(attr_output.protein1_attributions)
    top10_p1 = np.argsort(p1_attrs)[-10:][::-1]
    fig.add_trace(
        go.Bar(
            x=[f"{attr_output.protein1_residues[i]}_{i+1}" for i in top10_p1],
            y=[p1_attrs[i] for i in top10_p1],
            name='Top P1',
            marker_color='steelblue'
        ),
        row=2, col=1
    )
    
    p2_attrs = np.array(attr_output.protein2_attributions)
    top10_p2 = np.argsort(p2_attrs)[-10:][::-1]
    fig.add_trace(
        go.Bar(
            x=[f"{attr_output.protein2_residues[i]}_{i+1}" for i in top10_p2],
            y=[p2_attrs[i] for i in top10_p2],
            name='Top P2',
            marker_color='darkorange'
        ),
        row=2, col=2
    )
    
    fig.update_layout(
        height=800,
        showlegend=False,
        title_text=f"Binding Affinity Prediction: {attr_output.predicted_pkd:.3f} pKd"
    )
    
    return fig

# ============================================================================
# SESSION STATE MANAGEMENT
# ============================================================================

if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'explainer' not in st.session_state:
    st.session_state.explainer = None
if 'results' not in st.session_state:
    st.session_state.results = None

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    st.markdown('<h1 class="main-header">🧬 ALPINE</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Protein-Protein Binding Affinity Predictor with Explainability</p>', 
                unsafe_allow_html=True)
    
    # Sidebar - Model Configuration
    with st.sidebar:
        st.header("⚙️ Model Configuration")
        
        st.subheader("Model File Path")
        st.info("💡 For models >200MB, provide the local file path")
        
        checkpoint_path = st.text_input(
            "Model Checkpoint Path (.pth)",
            placeholder=r"D:\path\to\your\model.pth",
            help="Full path to your trained ALPINE model checkpoint",
            key="checkpoint_path"
        )
        
        # Quick path helpers
        if st.button("📁 Use Example Path"):
            st.code(r"D:\BALM_Fineclone\BALM-PPI\scripts\cv_results_lora\best_model_fold_1.pth")
        
        st.subheader("Model Parameters")
        use_lora = st.checkbox("Use LoRA (PEFT)", value=True, 
                              help="Enable if model was trained with LoRA")
        
        with st.expander("Advanced Settings"):
            esm_model = st.selectbox(
                "ESM-2 Model",
                ["facebook/esm2_t33_650M_UR50D", 
                 "facebook/esm2_t30_150M_UR50D",
                 "facebook/esm2_t36_3B_UR50D"],
                index=0
            )
            
            projected_size = st.number_input("Projection Size", 
                                            value=256, min_value=64, max_value=1024)
            dropout = st.slider("Dropout", 0.0, 0.5, 0.1, 0.05)
            
            if use_lora:
                lora_rank = st.number_input("LoRA Rank", value=8, min_value=1, max_value=64)
                lora_alpha = st.number_input("LoRA Alpha", value=16, min_value=1, max_value=128)
                lora_dropout = st.slider("LoRA Dropout", 0.0, 0.5, 0.1, 0.05)
            
            pkd_min = st.number_input("pKd Min", value=0.0)
            pkd_max = st.number_input("pKd Max", value=16.0)
        
        st.subheader("Explainability")
        enable_ig = st.checkbox("Enable Integrated Gradients", value=True,
                               help="Compute residue-level attributions (slower)")
        if enable_ig:
            ig_steps = st.slider("IG Steps", 10, 50, 25, 5,
                                help="More steps = more accurate but slower")
        
        # Load Model Button
        if st.button("🚀 Load Model", type="primary"):
            if not checkpoint_path:
                st.error("❌ Please provide a checkpoint file path!")
            elif not os.path.exists(checkpoint_path):
                st.error(f"❌ File not found: {checkpoint_path}")
                st.info("💡 Make sure to use the full path, e.g., D:\\folder\\model.pth")
            else:
                with st.spinner("Loading ALPINE model..."):
                    try:
                        device = "cuda" if torch.cuda.is_available() else "cpu"
                        
                        # Check file size
                        file_size_mb = os.path.getsize(checkpoint_path) / (1024**2)
                        st.info(f"📦 Loading checkpoint ({file_size_mb:.1f} MB)...")
                        
                        extractor = ProteinEmbeddingExtractor(
                            esm_model, device,
                            lora_rank if use_lora else 8,
                            lora_alpha if use_lora else 16,
                            lora_dropout if use_lora else 0.1,
                            use_lora and PEFT_AVAILABLE
                        )
                        
                        esm_model_obj, tokenizer = extractor.get_model_and_tokenizer()
                        
                        model = ALPINEModel(
                            esm_model_obj, tokenizer,
                            projected_size, dropout,
                            (pkd_min, pkd_max)
                        ).to(device)
                        
                        # Load checkpoint from file path
                        checkpoint = torch.load(
                            checkpoint_path,
                            map_location=device
                        )
                        model.load_state_dict(checkpoint)
                        model.eval()
                        
                        st.session_state.explainer = ALPINEAttributor(model, tokenizer)
                        st.session_state.model_loaded = True
                        
                        st.success("✅ Model loaded successfully!")
                        st.info(f"📍 Device: {device}")
                        st.info(f"📊 Model size: {file_size_mb:.1f} MB")
                        
                    except Exception as e:
                        st.error(f"❌ Error loading model: {str(e)}")
                        import traceback
                        with st.expander("Show error details"):
                            st.code(traceback.format_exc())
        
        if st.session_state.model_loaded:
            st.success("✅ Model Ready")
        
        st.markdown("---")
        st.markdown("### 📖 About")
        st.markdown("""
        **ALPINE** predicts protein-protein binding affinity using:
        - ESM-2 protein language model
        - Dual projection architecture
        - LoRA parameter-efficient fine-tuning
        - Integrated Gradients explainability
        
        **Supports large models (>1GB) via file path loading**
        """)
    
    # Main Content
    if not st.session_state.model_loaded:
        st.info("👈 Please load a model from the sidebar to begin")
        
        with st.expander("📚 Example Input"):
            st.markdown("""
            **Barnase (Protein 1):**
            ```
            AQVINTFDGVADYLQTYHKLPDNYITKSEAQALGWVASKGNLADVAPGKSIGGDIFSNREGKLPGKSGRTWREADINYTSGFRNSDRILYSSDWLIYKTTDHYQTFTKIR
            ```
            
            **Barstar (Protein 2):**
            ```
            KKAVINGEQIRSISDLHQTLKKELALPEYYGENLDALWDALTGWVEYPLVLEWRQFEQSKQLTENGAESVLQVFREAKAEGADITIILS
            ```
            """)
        return
    
    # Input Section
    st.header("🔬 Protein Sequence Input")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Protein 1 (Target)")
        protein1_name = st.text_input("Protein 1 Name", value="Protein_A")
        protein1_seq = st.text_area(
            "Sequence",
            height=150,
            placeholder="Enter protein sequence (single letter amino acids)...",
            key="protein1"
        )
        if protein1_seq:
            st.caption(f"Length: {len(protein1_seq)} residues")
    
    with col2:
        st.subheader("Protein 2 (Binder)")
        protein2_name = st.text_input("Protein 2 Name", value="Protein_B")
        protein2_seq = st.text_area(
            "Sequence",
            height=150,
            placeholder="Enter protein sequence (single letter amino acids)...",
            key="protein2"
        )
        if protein2_seq:
            st.caption(f"Length: {len(protein2_seq)} residues")
    
    # Predict Button
    if st.button("🎯 Predict Binding Affinity", type="primary", use_container_width=True):
        if not protein1_seq or not protein2_seq:
            st.error("❌ Please provide both protein sequences!")
        else:
            valid_aa = set("ACDEFGHIKLMNPQRSTVWY")
            p1_invalid = set(protein1_seq.upper()) - valid_aa
            p2_invalid = set(protein2_seq.upper()) - valid_aa
            
            if p1_invalid or p2_invalid:
                st.error(f"❌ Invalid amino acids found: {p1_invalid | p2_invalid}")
            else:
                with st.spinner("🧬 Computing predictions and attributions..."):
                    try:
                        if enable_ig and CAPTUM_AVAILABLE:
                            result = st.session_state.explainer.attribute(
                                protein1_seq.upper(),
                                protein2_seq.upper(),
                                steps=ig_steps
                            )
                        else:
                            pkd = st.session_state.explainer.predict(
                                protein1_seq.upper(),
                                protein2_seq.upper()
                            )
                            result = AttributionOutput(
                                pkd, list(protein1_seq.upper()), [0.0]*len(protein1_seq),
                                list(protein2_seq.upper()), [0.0]*len(protein2_seq)
                            )
                        
                        st.session_state.results = {
                            'attribution': result,
                            'p1_name': protein1_name,
                            'p2_name': protein2_name,
                            'p1_seq': protein1_seq.upper(),
                            'p2_seq': protein2_seq.upper()
                        }
                        
                    except Exception as e:
                        st.error(f"❌ Prediction failed: {str(e)}")
                        import traceback
                        with st.expander("Show error details"):
                            st.code(traceback.format_exc())
    
    # Results Section
    if st.session_state.results:
        st.markdown("---")
        st.header("📊 Results")
        
        result = st.session_state.results['attribution']
        p1_name = st.session_state.results['p1_name']
        p2_name = st.session_state.results['p2_name']
        
        st.markdown(f"""
        <div class="metric-card">
            <h2 style="color: #667eea; margin: 0;">Predicted Binding Affinity</h2>
            <h1 style="color: #2c3e50; margin: 10px 0;">{result.predicted_pkd:.3f} pKd</h1>
            <p style="color: #7f8c8d; margin: 0;">Higher values indicate stronger binding</p>
        </div>
        """, unsafe_allow_html=True)
        
        tab1, tab2, tab3, tab4 = st.tabs([
            "📈 Summary", "🔥 Heatmaps", "📊 Top Residues", "💾 Export"
        ])
        
        with tab1:
            st.plotly_chart(create_summary_plot(result), use_container_width=True)
        
        with tab2:
            st.subheader(f"🧬 {p1_name} Residue Importance")
            if sum(result.protein1_attributions) > 0:
                fig1 = create_heatmap_plotly(
                    result.protein1_residues,
                    result.protein1_attributions,
                    p1_name
                )
                st.plotly_chart(fig1, use_container_width=True)
            else:
                st.info("Integrated Gradients disabled. Enable in sidebar for heatmaps.")
            
            st.subheader(f"🧬 {p2_name} Residue Importance")
            if sum(result.protein2_attributions) > 0:
                fig2 = create_heatmap_plotly(
                    result.protein2_residues,
                    result.protein2_attributions,
                    p2_name
                )
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.info("Integrated Gradients disabled. Enable in sidebar for heatmaps.")
        
        with tab3:
            col1, col2 = st.columns(2)
            
            with col1:
                if sum(result.protein1_attributions) > 0:
                    fig = create_top_residues_chart(
                        result.protein1_residues,
                        result.protein1_attributions,
                        p1_name
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    top10_idx = np.argsort(result.protein1_attributions)[-10:][::-1]
                    df = pd.DataFrame({
                        'Rank': range(1, 11),
                        'Position': [i+1 for i in top10_idx],
                        'Residue': [result.protein1_residues[i] for i in top10_idx],
                        'Attribution': [f"{result.protein1_attributions[i]:.4f}" 
                                      for i in top10_idx]
                    })
                    st.dataframe(df, use_container_width=True)
            
            with col2:
                if sum(result.protein2_attributions) > 0:
                    fig = create_top_residues_chart(
                        result.protein2_residues,
                        result.protein2_attributions,
                        p2_name
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    top10_idx = np.argsort(result.protein2_attributions)[-10:][::-1]
                    df = pd.DataFrame({
                        'Rank': range(1, 11),
                        'Position': [i+1 for i in top10_idx],
                        'Residue': [result.protein2_residues[i] for i in top10_idx],
                        'Attribution': [f"{result.protein2_attributions[i]:.4f}" 
                                      for i in top10_idx]
                    })
                    st.dataframe(df, use_container_width=True)
        
        with tab4:
            st.subheader("💾 Export Results")
            
            max_len = max(len(result.protein1_residues), len(result.protein2_residues))
            export_data = {
                f'{p1_name}_Position': list(range(1, len(result.protein1_residues)+1)) + ['']*(max_len - len(result.protein1_residues)),
                f'{p1_name}_Residue': list(result.protein1_residues) + ['']*(max_len - len(result.protein1_residues)),
                f'{p1_name}_Attribution': list(result.protein1_attributions) + [np.nan]*(max_len - len(result.protein1_residues)),
                f'{p2_name}_Position': list(range(1, len(result.protein2_residues)+1)) + ['']*(max_len - len(result.protein2_residues)),
                f'{p2_name}_Residue': list(result.protein2_residues) + ['']*(max_len - len(result.protein2_residues)),
                f'{p2_name}_Attribution': list(result.protein2_attributions) + [np.nan]*(max_len - len(result.protein2_residues))
            }
            
            df_export = pd.DataFrame(export_data)
            
            metadata = f"# ALPINE Prediction Results\n"
            metadata += f"# Predicted pKd: {result.predicted_pkd:.3f}\n"
            metadata += f"# {p1_name} Length: {len(result.protein1_residues)}\n"
            metadata += f"# {p2_name} Length: {len(result.protein2_residues)}\n\n"
            
            csv = metadata + df_export.to_csv(index=False)
            
            st.download_button(
                label="📥 Download CSV Results",
                data=csv,
                file_name=f"alpine_results_{p1_name}_{p2_name}.csv",
                mime="text/csv"
            )
            
            st.markdown("**Preview:**")
            st.dataframe(df_export.head(20), use_container_width=True)

if __name__ == "__main__":
    main()