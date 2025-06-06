<!-- Common snippets for MkDocs -->

<!-- Medical Disclaimer -->
--8<-- "includes/medical-disclaimer.md"

<!-- Apple Silicon Badge -->
*[Apple Silicon]: Apple's custom ARM-based processors (M1, M2, M3) optimized for machine learning workloads

<!-- Common Abbreviations -->
*[LLM]: Large Language Model
*[LLMs]: Large Language Models
*[NLP]: Natural Language Processing
*[AI]: Artificial Intelligence
*[ML]: Machine Learning
*[RL]: Reinforcement Learning
*[RLHF]: Reinforcement Learning from Human Feedback
*[MPS]: Metal Performance Shaders
*[GPU]: Graphics Processing Unit
*[CPU]: Central Processing Unit
*[API]: Application Programming Interface
*[SDK]: Software Development Kit
*[IDE]: Integrated Development Environment
*[CLI]: Command Line Interface
*[HIPAA]: Health Insurance Portability and Accountability Act
*[FDA]: Food and Drug Administration
*[UMLS]: Unified Medical Language System
*[EHR]: Electronic Health Record
*[EMR]: Electronic Medical Record
*[FHIR]: Fast Healthcare Interoperability Resources
*[ICD]: International Classification of Diseases
*[SNOMED]: Systematized Nomenclature of Medicine
*[PyTorch]: Python-based machine learning framework
*[BERT]: Bidirectional Encoder Representations from Transformers
*[GPT]: Generative Pre-trained Transformer
*[Transformer]: Neural network architecture for sequence modeling
*[Attention]: Mechanism for focusing on relevant parts of input
*[Embedding]: Vector representation of text or concepts
*[Tokenization]: Process of converting text to tokens
*[Fine-tuning]: Adapting pre-trained models to specific tasks
*[Pre-training]: Initial training of models on large datasets
*[Inference]: Using trained models to make predictions
*[Perplexity]: Measure of language model performance
*[Cross-entropy]: Loss function used in language model training
*[Softmax]: Function for converting logits to probabilities
*[Gradient]: Derivative used in optimization
*[Backpropagation]: Algorithm for training neural networks
*[Optimizer]: Algorithm for updating model parameters
*[Learning Rate]: Parameter controlling optimization step size
*[Batch Size]: Number of examples processed together
*[Epoch]: Complete pass through training data
*[Overfitting]: Model memorizing training data
*[Regularization]: Techniques to prevent overfitting
*[Dropout]: Regularization technique
*[Layer Normalization]: Normalization technique for neural networks
*[Residual Connection]: Skip connection in neural networks
*[Multi-head Attention]: Parallel attention mechanisms
*[Positional Encoding]: Adding position information to embeddings
*[Autoregressive]: Generating sequences one token at a time
*[Masked Language Modeling]: Training objective for BERT-like models
*[Zero-shot]: Performing tasks without specific training
*[Few-shot]: Learning from few examples
*[In-context Learning]: Learning from examples in the prompt
*[Chain-of-Thought]: Step-by-step reasoning in prompts
*[Prompt Engineering]: Designing effective prompts
*[Constitutional AI]: AI alignment through principles
*[RLAIF]: Reinforcement Learning from AI Feedback
*[Agent]: AI system that can take actions
*[Tool Use]: AI systems using external tools
*[RAG]: Retrieval-Augmented Generation
*[Vector Database]: Database for storing embeddings
*[Semantic Search]: Search based on meaning
*[Knowledge Graph]: Structured representation of knowledge
*[Ontology]: Formal specification of concepts
*[Entity Recognition]: Identifying entities in text
*[Named Entity Recognition]: Identifying named entities
*[Relation Extraction]: Identifying relationships between entities
*[Sentiment Analysis]: Determining emotional tone
*[Text Classification]: Categorizing text
*[Text Generation]: Creating new text
*[Summarization]: Creating concise summaries
*[Translation]: Converting between languages
*[Question Answering]: Answering questions about text
*[Dialogue]: Conversational AI systems
*[Chatbot]: AI system for conversation
*[Virtual Assistant]: AI helper for tasks
*[Multimodal]: Processing multiple data types
*[Vision-Language]: Combining vision and language
*[Speech Recognition]: Converting speech to text
*[Text-to-Speech]: Converting text to speech
*[Computer Vision]: AI for visual understanding
*[Image Classification]: Categorizing images
*[Object Detection]: Finding objects in images
*[Segmentation]: Dividing images into regions
*[OCR]: Optical Character Recognition
*[ASR]: Automatic Speech Recognition
*[TTS]: Text-to-Speech
*[NLU]: Natural Language Understanding
*[NLG]: Natural Language Generation
*[NER]: Named Entity Recognition
*[POS]: Part-of-Speech tagging
*[Dependency Parsing]: Analyzing grammatical structure
*[Coreference Resolution]: Linking pronouns to entities
*[Word Sense Disambiguation]: Determining word meanings
*[Semantic Role Labeling]: Identifying semantic roles
*[Information Extraction]: Extracting structured information
*[Knowledge Base]: Structured collection of facts
*[Expert System]: AI system with domain expertise
*[Decision Support]: AI assisting human decisions
*[Clinical Decision Support]: AI for medical decisions
*[Diagnostic AI]: AI for medical diagnosis
*[Predictive Analytics]: AI for predicting outcomes
*[Risk Assessment]: Evaluating potential risks
*[Personalized Medicine]: Tailored medical treatment
*[Precision Medicine]: Targeted medical interventions
*[Pharmacovigilance]: Monitoring drug safety
*[Drug Discovery]: Finding new medications
*[Clinical Trial]: Testing medical interventions
*[Biomarker]: Biological indicator of health
*[Genomics]: Study of genetic information
*[Proteomics]: Study of proteins
*[Metabolomics]: Study of metabolites
*[Bioinformatics]: Computational biology
*[Medical Imaging]: Diagnostic imaging techniques
*[Radiology]: Medical imaging specialty
*[Pathology]: Study of disease
*[Laboratory Medicine]: Diagnostic testing
*[Telemedicine]: Remote medical care
*[Digital Health]: Technology in healthcare
*[Health Informatics]: Information technology in healthcare
*[Medical Informatics]: Computing in medicine
*[Clinical Informatics]: IT in clinical practice
*[Public Health]: Population health
*[Epidemiology]: Study of disease patterns
*[Health Economics]: Economic aspects of healthcare
*[Quality Improvement]: Enhancing healthcare quality
*[Patient Safety]: Preventing medical errors
*[Medical Ethics]: Ethical principles in medicine
*[Bioethics]: Ethics in biology and medicine
*[Informed Consent]: Patient agreement to treatment
*[Privacy]: Protection of personal information
*[Confidentiality]: Keeping information secret
*[Data Security]: Protecting data from threats
*[Cybersecurity]: Protection from cyber threats
*[Compliance]: Following regulations and standards
*[Audit]: Systematic examination of processes
*[Validation]: Confirming system performance
*[Verification]: Checking system correctness
*[Quality Assurance]: Ensuring quality standards
*[Risk Management]: Managing potential risks
*[Change Control]: Managing system changes
*[Documentation]: Recording information
*[Training]: Teaching users how to use systems
*[Support]: Helping users with problems
*[Maintenance]: Keeping systems running
*[Monitoring]: Watching system performance
*[Alerting]: Notifying of important events
*[Logging]: Recording system events
*[Debugging]: Finding and fixing problems
*[Testing]: Verifying system functionality
*[Unit Testing]: Testing individual components
*[Integration Testing]: Testing component interactions
*[System Testing]: Testing complete systems
*[User Acceptance Testing]: Testing by end users
*[Performance Testing]: Testing system performance
*[Load Testing]: Testing under heavy usage
*[Stress Testing]: Testing beyond normal limits
*[Security Testing]: Testing for vulnerabilities
*[Penetration Testing]: Simulated cyber attacks
*[Code Review]: Examining code for quality
*[Static Analysis]: Analyzing code without running
*[Dynamic Analysis]: Analyzing running code
*[Continuous Integration]: Automated code integration
*[Continuous Deployment]: Automated code deployment
*[DevOps]: Development and operations practices
*[MLOps]: Machine learning operations
*[DataOps]: Data operations practices
*[AIOps]: AI for IT operations
*[Cloud Computing]: Computing over the internet
*[Edge Computing]: Computing near data sources
*[Distributed Computing]: Computing across multiple machines
*[Parallel Computing]: Simultaneous computation
*[High Performance Computing]: Computing for demanding tasks
*[Quantum Computing]: Computing with quantum mechanics
*[Neuromorphic Computing]: Brain-inspired computing
*[Green Computing]: Environmentally friendly computing
*[Sustainable AI]: Environmentally responsible AI
*[Responsible AI]: Ethical and fair AI
*[Explainable AI]: AI that can explain decisions
*[Interpretable AI]: AI that is understandable
*[Trustworthy AI]: AI that can be trusted
*[Fair AI]: AI that treats everyone equally
*[Transparent AI]: AI that is open and clear
*[Accountable AI]: AI with clear responsibility
*[Human-Centered AI]: AI designed for humans
*[AI Safety]: Ensuring AI systems are safe
*[AI Alignment]: Ensuring AI goals match human values
*[AI Governance]: Managing AI development and use
*[AI Policy]: Rules and regulations for AI
*[AI Ethics]: Moral principles for AI
*[AI Bias]: Unfair treatment by AI systems
*[Algorithmic Bias]: Bias in algorithms
*[Data Bias]: Bias in training data
*[Selection Bias]: Bias in data selection
*[Confirmation Bias]: Bias toward confirming beliefs
*[Cognitive Bias]: Systematic thinking errors
*[Statistical Bias]: Systematic error in statistics
*[Sampling Bias]: Bias in sample selection
*[Measurement Bias]: Bias in data measurement
*[Reporting Bias]: Bias in result reporting
*[Publication Bias]: Bias in publishing results
