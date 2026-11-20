"""
Generate a Publication-Quality System Architecture Block Diagram.
Saves high-resolution PNG and SVG images representing the complete end-to-end
data ingestion, Seq2Seq LSTM, Bahdanau Attention, and KAN Bottleneck pipeline.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def create_block_diagram():
    output_dir = Path("research_results/figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 16:9 Widescreen High-Resolution Figure
    fig, ax = plt.subplots(figsize=(16, 9), dpi=300)
    fig.patch.set_facecolor('#0B0F19')
    ax.set_facecolor('#0B0F19')
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis('off')

    # Color Palette
    C_CARD = '#131B2E'
    C_BORDER = '#253354'
    C_CYAN = '#06B6D4'
    C_INDIGO = '#6366F1'
    C_EMERALD = '#10B981'
    C_AMBER = '#F59E0B'
    C_ROSE = '#F43F5E'
    C_TEXT = '#FFFFFF'
    C_MUTED = '#94A3B8'
    C_LIGHT_MUTED = '#CBD5E1'

    def draw_box(x, y, w, h, title, subtitle="", details=None, border_col=C_BORDER, bg_col=C_CARD, title_col=C_CYAN):
        # Card Background Box
        rect = patches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.08,rounding_size=0.15",
            facecolor=bg_col, edgecolor=border_col, linewidth=1.5,
            transform=ax.transData
        )
        ax.add_patch(rect)

        # Title
        ax.text(x + w/2, y + h - 0.28, title, color=title_col, fontsize=10.5, fontweight='bold', ha='center', va='top')
        
        # Subtitle
        curr_y = y + h - 0.55
        if subtitle:
            ax.text(x + w/2, curr_y, subtitle, color=C_LIGHT_MUTED, fontsize=8.5, fontweight='semibold', ha='center', va='top')
            curr_y -= 0.3

        # Details list
        if details:
            for d in details:
                ax.text(x + 0.2, curr_y, f"• {d}", color=C_MUTED, fontsize=7.5, ha='left', va='top')
                curr_y -= 0.22

    def draw_arrow(x1, y1, x2, y2, label="", col=C_CYAN):
        ax.annotate(
            '', xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(facecolor=col, edgecolor=col, width=2, headwidth=8, headlength=8, shrink=0.05)
        )
        if label:
            mx, my = (x1 + x2)/2, (y1 + y2)/2
            ax.text(mx, my + 0.15, label, color=col, fontsize=8, fontweight='bold', ha='center', va='bottom')

    # Header Title Banner
    ax.text(8.0, 8.6, "END-TO-END SYSTEM ARCHITECTURE BLOCK DIAGRAM", color=C_TEXT, fontsize=16, fontweight='bold', ha='center', va='top')
    ax.text(8.0, 8.2, "Real-Time Network Anomaly Detection: Seq2Seq LSTM + Bahdanau Attention + Kolmogorov-Arnold Network (KAN)", color=C_CYAN, fontsize=10, fontweight='semibold', ha='center', va='top')

    # -------------------------------------------------------------
    # STAGE 1: INGESTION & PREPROCESSING (Left Column: X = 0.8)
    # -------------------------------------------------------------
    draw_box(
        0.8, 4.8, 3.0, 2.8,
        "1. Flow Data Ingestion", "Official Benchmark Sources",
        [
            "CSE-CIC-IDS2018 (78 feats)",
            "CIC-IDS2017 (78 feats)",
            "CIC-DDoS2019 (81 feats)",
            "Zero Data Fabrication",
            "CSV Flow Log Parsing"
        ],
        border_col=C_CYAN, title_col=C_CYAN
    )

    draw_box(
        0.8, 1.2, 3.0, 2.8,
        "2. Feature Preprocessing", "Leakage-Free Normalization",
        [
            "RobustScaler fit on Train y=0",
            "Median NaN Imputation",
            "Inf Token Clamping (99.9%)",
            "Temporal Window W = 30",
            "Train S=5, Test S=1"
        ],
        border_col=C_CYAN, title_col=C_CYAN
    )

    # Arrow 1 -> 2
    draw_arrow(2.3, 4.8, 2.3, 4.0, label="Raw Flows", col=C_CYAN)

    # -------------------------------------------------------------
    # STAGE 2: ENCODER & ATTENTION (Middle-Left: X = 4.6)
    # -------------------------------------------------------------
    draw_box(
        4.6, 4.8, 3.2, 2.8,
        "3. Recurrent LSTM Encoder", "Temporal Sequence Modeling",
        [
            "Input Tensor: (B, T=30, D)",
            "2-Layer Recurrent LSTM",
            "Hidden State Dim: H = 64",
            "Dropout: 0.2",
            "Outputs Hidden Sequence h_t"
        ],
        border_col=C_INDIGO, title_col=C_INDIGO
    )

    draw_box(
        4.6, 1.2, 3.2, 2.8,
        "4. Bahdanau Attention", "Additive Alignment Context",
        [
            "Query: Decoder State s_{i-1}",
            "Key: Encoder Hidden h_j",
            "Score: v^T tanh(W_q q + U_k k)",
            "Softmax Energy Weights α_ij",
            "Dynamic Context Vector c_i"
        ],
        border_col=C_INDIGO, title_col=C_INDIGO
    )

    # Arrow 2 -> 3
    draw_arrow(3.8, 2.6, 4.6, 6.2, label="Window (30, D)", col=C_INDIGO)

    # Arrow 3 -> 4
    draw_arrow(6.2, 4.8, 6.2, 4.0, label="Hidden Annotations h_t", col=C_INDIGO)

    # -------------------------------------------------------------
    # STAGE 3: BOTTLENECK COMPARISON (Middle-Right: X = 8.6)
    # -------------------------------------------------------------
    draw_box(
        8.6, 4.8, 3.2, 2.8,
        "5A. Baseline Bottleneck", "Model 1: Linear Latent Layer",
        [
            "Linear Projection (64 -> 32)",
            "Fixed Linear Weights",
            "Latent Vector z in R^32",
            "Params: 174,190",
            "Classic Autoencoder Core"
        ],
        border_col=C_BORDER, title_col=C_TEXT
    )

    draw_box(
        8.6, 1.2, 3.2, 2.8,
        "5B. Proposed KAN Bottleneck", "Model 2: Trainable B-Spline",
        [
            "Kolmogorov-Arnold Network",
            "B-Spline Grid (G=5, Order k=3)",
            "SiLU Residual Base Function",
            "Params: 210,958 (+21.1%)",
            "Continuous Non-linear Manifold"
        ],
        border_col=C_CYAN, bg_col='#0F283D', title_col=C_CYAN
    )

    # Arrow 4 -> 5A & 5B
    draw_arrow(7.8, 3.0, 8.6, 5.8, label="Linear Path", col=C_MUTED)
    draw_arrow(7.8, 2.2, 8.6, 2.6, label="Proposed KAN", col=C_CYAN)

    # -------------------------------------------------------------
    # STAGE 4: DECODER & ANOMALY DECISION (Right Column: X = 12.4)
    # -------------------------------------------------------------
    draw_box(
        12.4, 4.8, 2.8, 2.8,
        "6. Autoregressive Decoder", "Sequence Reconstruction",
        [
            "2-Layer LSTM Decoder",
            "Context-Conditioned (c_i, z)",
            "Linear Feature Projector",
            "Reconstructed Sequence x_hat",
            "Output Shape: (B, T=30, D)"
        ],
        border_col=C_EMERALD, title_col=C_EMERALD
    )

    draw_box(
        12.4, 1.2, 2.8, 2.8,
        "7. Decision Boundary", "Reconstruction Loss Scoring",
        [
            "MSE = (1/T*D) sum (x - x_hat)^2",
            "Validation Threshold θ*",
            "MSE <= θ* -> NORMAL (0)",
            "MSE > θ* -> ANOMALY (1)",
            "Inference Latency: 18-28 ms"
        ],
        border_col=C_EMERALD, title_col=C_EMERALD
    )

    # Arrows 5A/5B -> 6
    draw_arrow(11.8, 6.2, 12.4, 6.2, col=C_MUTED)
    draw_arrow(11.8, 2.6, 12.4, 5.8, col=C_CYAN)

    # Arrow 6 -> 7
    draw_arrow(13.8, 4.8, 13.8, 4.0, label="Reconstruction MSE", col=C_EMERALD)

    # Bottom Legend / Summary Bar
    legend_rect = patches.FancyBboxPatch(
        (0.8, 0.3), 14.4, 0.55,
        boxstyle="round,pad=0.04,rounding_size=0.1",
        facecolor=C_CARD, edgecolor=C_BORDER, linewidth=1.0
    )
    ax.add_patch(legend_rect)
    ax.text(
        8.0, 0.58,
        "Execution Guarantee: Zero Data Leakage (Normal-only train)  •  Validation-Only Threshold (θ*)  •  Exact Matched Evaluation Across Benchmark Datasets",
        color=C_CYAN, fontsize=8.5, fontweight='bold', ha='center', va='center'
    )

    # Save PNG and SVG
    fig_png_path = output_dir / "system_architecture_block_diagram.png"
    fig_svg_path = output_dir / "system_architecture_block_diagram.svg"
    root_png_path = Path("system_architecture_block_diagram.png")
    docs_png_path = Path("docs/system_architecture_block_diagram.png")

    plt.tight_layout()
    plt.savefig(fig_png_path, dpi=300, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.savefig(fig_svg_path, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.savefig(root_png_path, dpi=300, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.savefig(docs_png_path, dpi=300, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()

    print(f"[+] Successfully generated System Architecture Block Diagram:")
    print(f"    - High-Res PNG (Research Figures): {fig_png_path.resolve()}")
    print(f"    - Scalable Vector SVG:            {fig_svg_path.resolve()}")
    print(f"    - Root Directory PNG:             {root_png_path.resolve()}")
    print(f"    - Docs Directory PNG:             {docs_png_path.resolve()}")


if __name__ == "__main__":
    create_block_diagram()
