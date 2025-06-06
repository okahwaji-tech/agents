# Resources

A comprehensive collection of resources for mastering Large Language Models and agentic workflows with healthcare applications.

## 📚 Primary Textbooks

### Core References

=== "Deep Learning"
    **Authors:** Ian Goodfellow, Yoshua Bengio, Aaron Courville  
    **Publisher:** MIT Press  
    **Focus:** Mathematical foundations of deep learning  
    
    **Key Chapters:**
    - Ch. 1: Introduction and mathematical notation
    - Ch. 8: Optimization for training deep models
    - Ch. 9: Convolutional networks
    - Ch. 10: Sequence modeling: RNNs and LSTMs
    - Ch. 12: Applications
    
    **Why Essential:** Provides the mathematical rigor needed to understand LLM internals.

=== "Hands-On Large Language Models"
    **Authors:** Jay Alammar, Maarten Grootendorst  
    **Publisher:** O'Reilly Media  
    **Focus:** Practical implementation of LLMs  
    
    **Key Chapters:**
    - Ch. 1-2: Introduction to LLMs
    - Ch. 3-4: Transformer architecture
    - Ch. 5-6: Pre-training and fine-tuning
    - Ch. 7-8: Advanced techniques
    
    **Why Essential:** Bridges theory and practice with real implementations.

=== "Reinforcement Learning: An Introduction"
    **Authors:** Richard S. Sutton, Andrew G. Barto  
    **Publisher:** MIT Press (2nd Edition)  
    **Focus:** Mathematical foundations of RL  
    
    **Key Chapters:**
    - Ch. 1: Introduction to RL
    - Ch. 2: Multi-armed bandits
    - Ch. 3-4: Finite Markov Decision Processes
    - Ch. 6-7: Temporal-difference learning
    - Ch. 13: Policy gradient methods
    
    **Why Essential:** Foundation for RLHF and agent architectures.

=== "AI Engineering"
    **Author:** Chip Huyen  
    **Publisher:** O'Reilly Media  
    **Focus:** Production ML systems  
    
    **Key Chapters:**
    - Ch. 1: Overview of ML systems
    - Ch. 4: Training data
    - Ch. 7: Model deployment
    - Ch. 11: The human side of ML
    
    **Why Essential:** Real-world deployment considerations.

## 🎓 Online Courses

### Stanford University

#### CS224n: Natural Language Processing with Deep Learning
- **Instructor:** Christopher Manning
- **Focus:** Deep learning for NLP
- **Key Topics:** Word vectors, neural networks, RNNs, Transformers
- **Link:** [CS224n](http://web.stanford.edu/class/cs224n/)

#### CS234: Reinforcement Learning
- **Instructor:** Emma Brunskill
- **Focus:** Mathematical foundations of RL
- **Key Topics:** MDPs, policy gradients, deep RL
- **Link:** [CS234](http://web.stanford.edu/class/cs234/)

#### Stats 116: Theory of Probability
- **Focus:** Probability theory foundations
- **Key Topics:** Distributions, Bayes' theorem, limit theorems
- **Relevance:** Mathematical foundation for LLMs

### MIT

#### 6.034: Artificial Intelligence
- **Focus:** AI fundamentals
- **Key Topics:** Search, knowledge representation, machine learning
- **Link:** [MIT 6.034](https://ocw.mit.edu/courses/6-034-artificial-intelligence-fall-2010/)

### Fast.ai

#### Practical Deep Learning for Coders
- **Focus:** Hands-on deep learning
- **Key Topics:** Computer vision, NLP, tabular data
- **Link:** [Fast.ai](https://www.fast.ai/)

## 📄 Essential Papers

### Foundation Papers

#### Transformer Architecture
1. **"Attention Is All You Need"** (Vaswani et al., 2017)
   - Introduced the Transformer architecture
   - Mathematical foundation of modern LLMs
   - [Paper Link](https://arxiv.org/abs/1706.03762)

2. **"BERT: Pre-training of Deep Bidirectional Transformers"** (Devlin et al., 2018)
   - Bidirectional encoder representations
   - Masked language modeling
   - [Paper Link](https://arxiv.org/abs/1810.04805)

#### Large Language Models
3. **"Language Models are Few-Shot Learners"** (Brown et al., 2020)
   - GPT-3 and emergent capabilities
   - In-context learning
   - [Paper Link](https://arxiv.org/abs/2005.14165)

4. **"Training language models to follow instructions with human feedback"** (Ouyang et al., 2022)
   - InstructGPT and RLHF
   - Human preference learning
   - [Paper Link](https://arxiv.org/abs/2203.02155)

#### Scaling Laws
5. **"Scaling Laws for Neural Language Models"** (Kaplan et al., 2020)
   - Mathematical relationships for scaling
   - Performance prediction
   - [Paper Link](https://arxiv.org/abs/2001.08361)

6. **"Training Compute-Optimal Large Language Models"** (Hoffmann et al., 2022)
   - Chinchilla scaling laws
   - Optimal compute allocation
   - [Paper Link](https://arxiv.org/abs/2203.15556)

### Healthcare AI Papers

#### Medical Language Models
7. **"ClinicalBERT: Modeling Clinical Notes and Predicting Hospital Readmission"** (Huang et al., 2019)
   - Clinical text processing
   - Healthcare applications
   - [Paper Link](https://arxiv.org/abs/1904.05342)

8. **"BioBERT: a pre-trained biomedical language representation model"** (Lee et al., 2020)
   - Biomedical text understanding
   - Domain adaptation
   - [Paper Link](https://arxiv.org/abs/1901.08746)

#### AI Safety in Healthcare
9. **"The Medical AI Safety Problem"** (Amodei et al., 2016)
   - Safety considerations for medical AI
   - Risk assessment frameworks
   - [Paper Link](https://arxiv.org/abs/1606.06565)

### Recent Advances

#### Multimodal Models
10. **"Flamingo: a Visual Language Model for Few-Shot Learning"** (Alayrac et al., 2022)
    - Vision-language integration
    - Few-shot multimodal learning
    - [Paper Link](https://arxiv.org/abs/2204.14198)

#### Agent Architectures
11. **"ReAct: Synergizing Reasoning and Acting in Language Models"** (Yao et al., 2022)
    - Reasoning and action integration
    - Tool use in LLMs
    - [Paper Link](https://arxiv.org/abs/2210.03629)

## 🛠️ Tools and Libraries

### Core Libraries

#### PyTorch Ecosystem
- **PyTorch**: Deep learning framework
- **Transformers**: Hugging Face model library
- **Datasets**: Data loading and processing
- **Accelerate**: Distributed training
- **PEFT**: Parameter-efficient fine-tuning

#### Development Tools
- **uv**: Fast Python package manager
- **Ruff**: Code formatting and linting
- **MyPy**: Type checking
- **Pytest**: Testing framework
- **Jupyter Lab**: Interactive development

### Specialized Tools

#### Healthcare AI
- **spaCy**: NLP with medical models
- **scispaCy**: Scientific/biomedical text processing
- **Transformers for Healthcare**: Medical domain models
- **FHIR**: Healthcare data standards

#### Apple Silicon Optimization
- **Metal Performance Shaders**: GPU acceleration
- **Core ML**: On-device inference
- **MLX**: Apple's ML framework
- **Accelerate Framework**: Optimized computations

## 🌐 Online Resources

### Documentation and Guides

#### Official Documentation
- [PyTorch Documentation](https://pytorch.org/docs/)
- [Hugging Face Documentation](https://huggingface.co/docs)
- [Apple Developer ML](https://developer.apple.com/machine-learning/)

#### Community Resources
- [Papers with Code](https://paperswithcode.com/)
- [Towards Data Science](https://towardsdatascience.com/)
- [The Gradient](https://thegradient.pub/)
- [Distill](https://distill.pub/)

### Blogs and Newsletters

#### Technical Blogs
- **Jay Alammar's Blog**: Visual explanations of ML concepts
- **Sebastian Ruder's Blog**: NLP research insights
- **Andrej Karpathy's Blog**: Deep learning tutorials
- **Lilian Weng's Blog**: Research paper summaries

#### Industry Newsletters
- **The Batch** (deeplearning.ai): Weekly AI news
- **Import AI**: AI research and policy
- **The Gradient**: Academic AI research
- **AI Research**: Latest papers and trends

## 🏥 Healthcare-Specific Resources

### Medical Datasets

#### Public Datasets
- **MIMIC-III**: Critical care database
- **PubMed**: Biomedical literature
- **ClinicalTrials.gov**: Clinical trial data
- **UMLS**: Unified Medical Language System

#### Synthetic Datasets
- **Synthea**: Synthetic patient data
- **Medical Meadow**: Curated medical datasets
- **BioASQ**: Biomedical semantic indexing

### Regulatory Resources

#### FDA Guidelines
- **Software as Medical Device (SaMD)**: Regulatory framework
- **AI/ML-Based Medical Device Action Plan**: FDA guidance
- **Digital Health Center of Excellence**: Resources and guidance

#### HIPAA Compliance
- **HHS HIPAA Guidelines**: Privacy and security rules
- **NIST Cybersecurity Framework**: Security standards
- **Healthcare Data Security**: Best practices

### Medical Knowledge Bases

#### Ontologies and Standards
- **SNOMED CT**: Clinical terminology
- **ICD-10**: Disease classification
- **LOINC**: Laboratory data
- **RxNorm**: Medication terminology

## 💻 Development Environment

### Apple Silicon Setup

#### Required Software
- **Xcode Command Line Tools**: Development tools
- **Homebrew**: Package manager
- **Python 3.11+**: Programming language
- **uv**: Python package manager

#### Optimization Tools
- **Activity Monitor**: System monitoring
- **Instruments**: Performance profiling
- **Console**: Log analysis
- **Memory Graph Debugger**: Memory analysis

### Cloud Platforms

#### Training and Inference
- **Google Colab**: Free GPU/TPU access
- **AWS SageMaker**: Managed ML platform
- **Azure ML**: Microsoft's ML platform
- **Hugging Face Spaces**: Model hosting

#### Healthcare-Specific Platforms
- **AWS HealthLake**: Healthcare data lake
- **Google Cloud Healthcare API**: Healthcare data processing
- **Azure Health Data Services**: Healthcare cloud platform

## 📊 Evaluation and Benchmarks

### General LLM Benchmarks

#### Language Understanding
- **GLUE**: General Language Understanding Evaluation
- **SuperGLUE**: More challenging language tasks
- **HELM**: Holistic Evaluation of Language Models
- **BIG-bench**: Beyond the Imitation Game benchmark

#### Code Generation
- **HumanEval**: Python code generation
- **MBPP**: Mostly Basic Python Problems
- **CodeXGLUE**: Code understanding and generation

### Healthcare Benchmarks

#### Medical Knowledge
- **MedQA**: Medical question answering
- **PubMedQA**: Biomedical question answering
- **BioASQ**: Biomedical semantic indexing
- **USMLE**: Medical licensing exam questions

#### Clinical Tasks
- **i2b2**: Clinical NLP challenges
- **n2c2**: National NLP Clinical Challenges
- **CLEF eHealth**: Health information retrieval

## 🤝 Community and Support

### Forums and Discussions

#### Technical Communities
- **Hugging Face Forums**: Model and library discussions
- **PyTorch Forums**: Framework support
- **Reddit r/MachineLearning**: Research discussions
- **Stack Overflow**: Programming help

#### Healthcare AI Communities
- **Healthcare AI Slack**: Professional network
- **HIMSS**: Healthcare IT community
- **AMIA**: Medical informatics association

### Conferences and Events

#### AI/ML Conferences
- **NeurIPS**: Neural Information Processing Systems
- **ICML**: International Conference on Machine Learning
- **ICLR**: International Conference on Learning Representations
- **ACL**: Association for Computational Linguistics

#### Healthcare AI Conferences
- **HIMSS**: Healthcare IT conference
- **AMIA**: Medical informatics symposium
- **ML4H**: Machine Learning for Health
- **CHIL**: Conference on Health, Inference, and Learning

## 📈 Staying Current

### Research Tracking

#### Paper Aggregators
- **arXiv**: Preprint repository
- **Papers with Code**: Implementation tracking
- **Semantic Scholar**: AI-powered search
- **Google Scholar**: Academic search

#### Social Media
- **Twitter/X**: Researcher updates
- **LinkedIn**: Professional insights
- **YouTube**: Tutorial videos
- **Podcasts**: AI research discussions

### News and Updates

#### Industry News
- **VentureBeat AI**: Industry coverage
- **TechCrunch AI**: Startup news
- **MIT Technology Review**: Research insights
- **Nature Machine Intelligence**: Academic journal

---

**Need help finding specific resources?** Use the search functionality or browse by category. All resources are regularly updated to reflect the latest developments in LLM and healthcare AI research.
