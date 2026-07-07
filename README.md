# MIRAI

**Mechanistic Interpretability and Reversing of AI**

A [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens)-inspired debugger for arbitrary PyTorch models. Load a pretrained model, step through it layer by layer, and see what's actually happening inside.

> **Status:** Early WIP. MLP inspection works; attention, residual stream, and layer norm coverage are next.

## Why

Neural networks are black boxes. Existing tools like TransformerLens are great, but they only work on models they've been ported to. MIRAI aims to do layer-by-layer inspection on *any* PyTorch model, and eventually extract the equations a network has actually learned.

## Install

```bash
git clone https://github.com/sajeev-68/MIRAI.git
cd MIRAI
pip install -e .
```

Requires Python 3.10+.

## Usage

<img width="1278" height="763" alt="Screenshot 2026-07-07 at 2 08 42 AM" src="https://github.com/user-attachments/assets/82693dff-9361-4411-83bf-076fda81325f" />

## Roadmap

- [x] Load arbitrary PyTorch models
- [x] MLP layer inspection
- [x] Gradio UI
- [ ] Attention hooks (Q/K/V, attention patterns)
- [ ] Residual stream capture
- [ ] Layer norm inspection
- [ ] Symbolic extraction of learned equations (SymPy)
