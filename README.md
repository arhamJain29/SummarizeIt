#  SummarizeIt – PDF Summarizer by Arham Jain

**SummarizeIt** is a powerful Streamlit-based web application that automatically summarizes lengthy PDF documents using state-of-the-art transformer models. Whether you're reviewing reports, research papers, or articles — SummarizeIt provides concise, high-quality summaries in seconds.

---

## Features

-  Upload and summarize any **PDF** file.
-  Uses **LaMini-Flan-T5** (`MBZUAI/LaMini-Flan-T5-248M`) for summarization.
-  Intelligent **text chunking** with LangChain to handle long documents.
-  Dual-pane UI for viewing original PDF and generated summary side by side.
-  Fast, clean interface built with **Streamlit**.
-  GPU/Auto-device support via HuggingFace Transformers.

---

##  Tech Stack

- `Python`
- `Streamlit`
- `Transformers (HuggingFace)`
- `LangChain`
- `PyPDFLoader`
- `T5 Model (LaMini-Flan-T5)`
- `Torch` (for model inference)

---

##  UI Preview

<img src="https://your-screenshot-url.com/summarizeit-ui.png" alt="SummarizeIt Screenshot" width="100%"/>

---

## Installation

```bash
# Clone the repo
git clone https://github.com/arhamJain29/SummarizeIt.git
cd SummarizeIt

# Install dependencies
uv sync

# Run the App
uv run streamlit run main.py
```


# Model Info
- Checkpoint: MBZUAI/LaMini-Flan-T5-248M
- Model Type: Sequence-to-sequence (T5)
- Framework: HuggingFace Transformers
- Device Map: Auto-configured (supports CPU/GPU)

# Project Tree 
```
.
├── main.py
├── pyproject.toml
├── README.md
└── uv.lock
```

