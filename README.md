# agents
This is a repository for learning LLM and agentic workflows


## Development Setup with uv

Install uv if it is not already available:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Install project dependencies:
```bash
uv pip install -r requirements.txt
```

Add a new package:
```bash
uv pip install <package>
```

Remove a package:
```bash
uv pip uninstall <package>
```

Run development checks:
```bash
ruff check .
ruff format .
mypy .
pytest
```
## Study Plan

| Week   | Focus & Objectives                       | Key Resources                                                                                                                  | Hands-On Deliverable                                                                    |
| ------ | ---------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------- |
| **1**  | Probability & basic statistics refresher | • Stanford **Stats 116** lectures<br>• *Mathematics for Data Science Mastery* (road-map PDF)                                   | Notebook: word-frequency & basic corpus statistics                                      |
| **2**  | Linear algebra & calculus essentials     | • MIT OCW **18.06** videos<br>• 3Blue1Brown *Essence of Linear Algebra* series                                                 | Jupyter cheatsheet of key tensor ops demonstrated with small matrices                   |
| **3**  | PyTorch fundamentals (tensors, autograd) | • *PyTorch Ultimate 2024* course (road-map PDF)<br>• CS229 PyTorch labs<br>• Official PyTorch docs                             | Implement forward / backward passes for a tiny MLP                                      |
| **4**  | Optimisation & first model project       | • fastai text tutorial<br>• Hugging Face Datasets (IMDb)                                                                       | Sentiment classifier on IMDb, pushed to GitHub                                          |
| **5**  | Classic NLP: tokenisation & embeddings   | • Stanford **CS224N** Lectures 1-3<br>• NLTK book ch. 2-3                                                                      | Train Word2Vec; visualise nearest-neighbour words                                       |
| **6**  | Sequence models: RNN/LSTM                | • CS224N Lectures 4-5<br>• Colah’s LSTM blog<br>• PyTorch seq2seq tut.                                                         | Char-level LSTM text generator                                                          |
| **7**  | Attention & Transformer mechanics        | • CS224N attention lecture<br>• *The Illustrated Transformer*<br>• CMU Adv. NLP Transformer module                             | Implement scaled-dot-product attention from scratch                                     |
| **8**  | End-to-end mini-Transformer + fine-tune  | • Hugging Face Transformers Course<br>• DistilGPT-2 fine-tune notebook                                                         | Fine-tune DistilGPT-2 on 1k custom prompts                                              |
| **9**  | LLM families & scaling laws survey       | • Kaplan 2020 & Henighan 2024 scaling papers<br>• *Fine-Tuning Landscape 2025* (road-map PDF)                                  | Slide deck comparing 5 popular model families                                           |
| **10** | Data prep & fine-tune strategy           | • HF *Datasets* docs<br>• Instruction-tuning papers (e.g. FLAN)                                                                | Curate 2 k instruction-response pairs for later LoRA run                                |
| **11** | LoRA theory & setup                      | • Original **LoRA** paper (Hu 2021)<br>• HF **PEFT** library docs<br>• TorchTune tutorials                                     | Dry-run LoRA injection on LLaMA-2-7B (no training yet)                                  |
| **12** | QLoRA fine-tune experiment               | • **QLoRA** paper (Dettmers 2023)<br>• bitsandbytes 4-bit guide<br>• “Fine-tune LLaMA-2 with QLoRA” blog                       | Full QLoRA fine-tune → eval on held-out dev set                                         |
| **13** | LangChain basics & PromptTemplates       | • LangChain Quick-start docs<br>• Prompt Engineering Guide                                                                     | Conversational chain with buffer memory                                                 |
| **14** | Agents & ReAct loop + custom tools       | • ReAct paper (Yao 2022)<br>• “Definitive Guide to AI Agent Frameworks 2025” (road-map PDF)<br>• CrewAI / Semantic Kernel docs | Agent that answers *“cube root of year Google founded”* using search + calculator tools |
| **15** | Retrieval-Augmented Generation (RAG)     | • LlamaIndex Getting Started<br>• Weaviate / Chroma docs<br>• HF RAG blog                                                      | PDF-QA bot with source citations                                                        |
| **16** | Long-term memory & evaluation            | • LangChain Memory docs<br>• OpenAI Evals / HELM papers                                                                        | v 0.9 “Knowledge Assistant” on Gradio with eval harness                                 |
| **17** | RL foundations for RLHF                  | • Stanford **CS234** policy-gradient lectures<br>• Sutton & Barto Ch. 13                                                       | REINFORCE on toy bandit task                                                            |
| **18** | Mini RLHF loop                           | • Hugging Face RLHF blog<br>• trlx library docs                                                                                | 1-epoch SFT → reward model → PPO; show JSON-accuracy gain                               |
| **19** | Deployment & cost optimisation           | • bitsandbytes quantisation guide<br>• DeepSpeed-Inference docs<br>• FastAPI quick-start                                       | Latency & cost benchmark table for 8-bit vs 4-bit                                       |
| **20** | Capstone scoping & data gathering        | • Notion project template<br>• Example data APIs (arXiv, GitHub, MIMIC-III)                                                    | Detailed design doc & backlog for capstone                                              |
| **21** | Capstone – core pipeline build           | • LangChain + Gradio integration guide                                                                                         | Working end-to-end prototype                                                            |
| **22** | Capstone – refinement & extra tools      | • OpenAI Function-Calling advanced guide<br>• LangSmith debugging                                                              | Add memory, second tool (e.g., web search)                                              |
| **23** | Capstone – testing & docs                | • LangChain eval module<br>• Markdown/README best-practices                                                                    | Full test pass, polished README, demo video                                             |
| **24** | Capstone – public release & reflection   | • GitHub Pages / HF Spaces deploy how-to<br>• Medium blogging guides                                                           | Live demo link + blog post “6-month journey to my LLM agent”                            |
