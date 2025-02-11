import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, font
from phi.agent import Agent
from phi.model.groq import Groq
from dotenv import load_dotenv
import json
import os
import re
import subprocess
import tempfile
import py_compile
from datetime import datetime

# Load environment variables
load_dotenv()

class CodingAssistantGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Coding Assistant Pro")
        self.root.geometry("1200x900")
        
        # Configuration settings
        self.config = {
            "temperature": 0.7,
            "max_tokens": 10000,
            "dark_mode": False,
            "font_size": 10,
            "current_language": "python"
        }
        self.load_config()
        
        # Initialize GUI
        self.setup_gui()
        
        # Initialize the AI agent
        self.initialize_agent()
        
        # Load chat history
        self.chat_history = []
        self.load_chat_history()
        
        # Set theme
        self.set_theme()
        
        # Dictionary of similar questions
        self.similar_questions = {}
        self.load_similar_questions()

    def setup_gui(self):
        # Main container with left and right panes
        self.paned_window = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        self.paned_window.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Left pane for chat
        left_frame = ttk.Frame(self.paned_window)
        self.paned_window.add(left_frame, weight=1)

        # Right pane for compiler
        right_frame = ttk.Frame(self.paned_window)
        self.paned_window.add(right_frame, weight=1)

        # Setup chat interface (left pane)
        self.setup_chat_interface(left_frame)
        
        # Setup compiler interface (right pane)
        self.setup_compiler_interface(right_frame)

    def setup_chat_interface(self, parent):
        # Toolbar frame
        toolbar_frame = ttk.Frame(parent)
        toolbar_frame.pack(fill=tk.X, pady=5)
        
        # Buttons
        ttk.Button(toolbar_frame, text="Clear Chat", command=self.clear_chat).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar_frame, text="Toggle Theme", command=self.toggle_theme).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar_frame, text="Save Chat", command=self.save_chat_history).pack(side=tk.LEFT, padx=2)
        
        # Font size controls
        ttk.Label(toolbar_frame, text="Font Size:").pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar_frame, text="-", command=lambda: self.change_font_size(-1)).pack(side=tk.LEFT, padx=1)
        ttk.Button(toolbar_frame, text="+", command=lambda: self.change_font_size(1)).pack(side=tk.LEFT, padx=1)
        
        # Chat display
        self.chat_display = scrolledtext.ScrolledText(
            parent,
            wrap=tk.WORD,
            font=('Arial', self.config["font_size"]),
            height=20
        )
        self.chat_display.pack(fill=tk.BOTH, expand=True, pady=5)
        self.chat_display.config(state='disabled')
        
        # Similar questions display
        self.similar_questions_display = scrolledtext.ScrolledText(
            parent,
            wrap=tk.WORD,
            font=('Arial', self.config["font_size"]),
            height=5
        )
        self.similar_questions_display.pack(fill=tk.X, pady=5)
        self.similar_questions_display.config(state='disabled')
        
        # Query input frame
        input_frame = ttk.Frame(parent)
        input_frame.pack(fill=tk.X, pady=5)
        
        self.query_input = ttk.Entry(input_frame)
        self.query_input.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        self.query_input.bind('<Return>', lambda e: self.submit_query())
        
        ttk.Button(input_frame, text="Submit", command=self.submit_query).pack(side=tk.RIGHT)

    def setup_compiler_interface(self, parent):
        # Language selection
        lang_frame = ttk.Frame(parent)
        lang_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(lang_frame, text="Language:").pack(side=tk.LEFT, padx=5)
        self.language_var = tk.StringVar(value=self.config["current_language"])
        languages = ["python", "java", "c", "cpp", "javascript"]
        language_menu = ttk.OptionMenu(lang_frame, self.language_var, self.config["current_language"], *languages, command=self.change_language)
        language_menu.pack(side=tk.LEFT, padx=5)
        
        # Code editor
        self.code_editor = scrolledtext.ScrolledText(
            parent,
            wrap=tk.NONE,
            font=('Courier New', self.config["font_size"]),
            height=15
        )
        self.code_editor.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Button frame
        button_frame = ttk.Frame(parent)
        button_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(button_frame, text="Run Code", command=self.run_current_code).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Clear Editor", command=self.clear_editor).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Format Code", command=self.format_code).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Save Code", command=self.save_code).pack(side=tk.LEFT, padx=5)
        
        # Output display
        self.output_display = scrolledtext.ScrolledText(
            parent,
            wrap=tk.WORD,
            font=('Courier New', self.config["font_size"]),
            height=10
        )
        self.output_display.pack(fill=tk.BOTH, expand=True, pady=5)
        self.output_display.config(state='disabled')

    def is_coding_related(self, query):
        """Check if the query is related to coding."""
        coding_keywords = [
    # General Programming
    'code', 'program', 'python', 'java', 'c++', 'c', 'c#', 'ruby', 'swift', 'go', 'rust',
    'debug', 'error', 'compile', 'script', 'function', 'method', 'variable', 'loop',
    'syntax', 'data structure', 'algorithm', 'recursion', 'iteration', 'pointer', 'memory management',

    # Competitive Programming (CP) / Data Structures & Algorithms (DSA)
    'dynamic programming', 'greedy algorithm', 'graph theory', 'tree', 'binary tree',
    'segment tree', 'binary search tree', 'trie', 'fenwick tree', 'bit', 'linked list', 
    'doubly linked list', 'circular linked list', 'stack', 'queue', 'deque', 'priority queue', 
    'heap', 'max heap', 'min heap', 'hashing', 'hash table', 'hash map', 'unordered map',
    'unordered set', 'ordered set', 'binary search', 'ternary search', 'fibonacci search',
    'bit manipulation', 'fast exponentiation', 'modular exponentiation', 'gcd', 'lcm',
    'prime factorization', 'sieve of eratosthenes', 'totient function', 'modular inverse',
    'euclidean algorithm', 'extended euclidean algorithm', 'fast fourier transform',
    'manacher’s algorithm', 'rabin karp', 'knuth morris pratt', 'z-algorithm', 'aho corasick',
    'suffix array', 'suffix tree', 'suffix automaton', 'longest common subsequence (LCS)',
    'longest increasing subsequence (LIS)', 'edit distance', 'levenshtein distance',
    'boyer moore', 'rolling hash', 'string hashing', 'trie', 'aho corasick algorithm',

    # Graph Algorithms
    'dfs', 'bfs', 'topological sort', 'shortest path', 'dijkstra', 'bellman ford', 
    'floyd warshall', 'a* algorithm', 'prim’s algorithm', 'kruskal’s algorithm', 
    'minimum spanning tree (MST)', 'union-find', 'disjoint set', 'tarjan’s algorithm', 
    'kosaraju’s algorithm', 'bridges and articulation points', 'strongly connected components (SCC)',
    'euler tour', 'hamiltonian path', 'centroid decomposition', 'tree flattening',
    'lowest common ancestor (LCA)', 'heavy-light decomposition (HLD)', 
    'binary lifting', 'mo’s algorithm',

    # Number Theory
    'modular arithmetic', 'modular exponentiation', 'modular multiplicative inverse',
    'chinese remainder theorem', 'wilson’s theorem', 'fermat’s little theorem',
    'euler’s totient function', 'lucas theorem', 'bernoulli numbers', 'catalan numbers',
    'lagrange’s theorem', 'stirling numbers', 'pascal’s triangle', 'binomial coefficients',

    # Game Theory & Math
    'grundy number', 'nim game', 'sprague-grundy theorem', 'mex function',
    'game theory', 'minimax algorithm', 'alpha-beta pruning',
    
    # Computational Geometry
    'convex hull', 'graham scan', 'jarvis march', 'quick hull', 
    'polygon triangulation', 'line sweep', 'point in polygon',
    'rotating calipers', 'closest pair of points',

    # Advanced CP Techniques
    'bitmask dp', 'meet in the middle', 'iterative deepening', 'sparse table',
    'two pointers', 'sliding window', 'prefix sum', 'difference array',
    'mo’s algorithm', 'sqrt decomposition', 'wavelet tree', 'binary lifting',
    'persistent data structures', 'offline queries', 'rerooting dp',
    'de bruijn sequence', 'heavy light decomposition (HLD)',
    
    # Miscellaneous CP Concepts
    'interactive problems', 'parallel binary search', 'ternary search', 
    'matrix exponentiation', 'kmp algorithm', 'string hashing',
    'rolling hash', 'longest palindromic substring', 'manacher’s algorithm',

    # Database & APIs
    'database', 'sql', 'nosql', 'mongodb', 'mysql', 'postgresql', 'sqlite',
    'orm', 'query optimization', 'acid', 'base', 'indexing', 'sharding',
    'rest api', 'graphql', 'endpoint', 'json', 'xml', 'authentication', 'jwt',
    'oauth', 'webhook', 'microservices',

    # Web Development (Frontend & Backend)
    'html', 'css', 'javascript', 'react', 'angular', 'vue', 'svelte', 'tailwindcss',
    'bootstrap', 'sass', 'less', 'typescript', 'next.js', 'node.js', 'express',
    'nestjs', 'flask', 'django', 'fastapi', 'spring boot', 'laravel', 'php',
    'ruby on rails', 'asp.net', 'graphql', 'websockets',

    # DevOps, Cloud & Containers
    'devops', 'ci/cd', 'docker', 'kubernetes', 'terraform', 'ansible',
    'cloud', 'aws', 'azure', 'gcp', 'server', 'client', 'nginx', 'apache',
    'load balancer', 'reverse proxy', 'ssl', 'tls', 'http', 'https', 'dns',
    'web hosting', 'cdn', 'cloudflare', 'serverless', 'lambda', 'fargate',
    'ecs', 'eks', 'virtual machine', 'kafka', 'rabbitmq', 'redis',
    
    # Machine Learning & AI
    'machine learning', 'deep learning', 'artificial intelligence', 'ai',
    'supervised learning', 'unsupervised learning', 'semi-supervised learning', 'self-supervised learning',
    'reinforcement learning', 'federated learning', 'transfer learning', 
    'online learning', 'offline learning', 'incremental learning',
    'bayesian learning', 'graph neural networks', 'explainable ai (XAI)',
    'automated machine learning (AutoML)', 'generative AI',

    # AI/ML Algorithms
    'linear regression', 'logistic regression', 'decision tree', 'random forest', 
    'support vector machine (SVM)', 'k-nearest neighbors (KNN)', 'naive bayes', 
    'gradient boosting', 'xgboost', 'lightgbm', 'catboost',
    'k-means clustering', 'hierarchical clustering', 'dbscan', 'mean shift',
    'pca', 'lda', 't-sne', 'umap', 'isomap', 'autoencoders',
    'markov chains', 'hidden markov models (HMM)', 'monte carlo methods',
    'collaborative filtering', 'content-based filtering', 'recommender systems',

    # Neural Networks & Deep Learning
    'neural network', 'deep neural network (DNN)', 'convolutional neural network (CNN)',
    'recurrent neural network (RNN)', 'long short-term memory (LSTM)',
    'gated recurrent unit (GRU)', 'transformer', 'bert', 'gpt', 't5', 'xlm',
    'diffusion models', 'vae (variational autoencoder)', 'gan (generative adversarial network)',
    'self-attention', 'multi-head attention', 'attention mechanism',
    'vision transformer (ViT)', 'swin transformer', 'efficientnet',
    'mlp-mixer', 'contrastive learning', 'simclr', 'moco', 'dino',

    # Optimization Algorithms
    'gradient descent', 'stochastic gradient descent (SGD)', 
    'adam optimizer', 'rmsprop', 'adagrad', 'momentum', 
    'nesterov accelerated gradient (NAG)', 'hyperparameter tuning',
    'bayesian optimization', 'genetic algorithms', 'simulated annealing',
    'particle swarm optimization (PSO)', 'ant colony optimization',

    # Natural Language Processing (NLP)
    'natural language processing (NLP)', 'tokenization', 'word embedding', 
    'word2vec', 'glove', 'fasttext', 'tf-idf', 'n-grams',
    'named entity recognition (NER)', 'part-of-speech (POS) tagging',
    'sentiment analysis', 'text classification', 'topic modeling',
    'latent dirichlet allocation (LDA)', 'transformers', 'seq2seq models',
    'text summarization', 'question answering', 'speech-to-text', 'text-to-speech',

    # Computer Vision
    'image processing', 'object detection', 'image segmentation',
    'face recognition', 'pose estimation', 'optical character recognition (OCR)',
    'edge detection', 'hough transform', 'image augmentation',
    'style transfer', 'image inpainting', 'super-resolution',

    # Reinforcement Learning (RL)
    'q-learning', 'deep q-network (DQN)', 'actor-critic', 'a3c', 'a2c',
    'ppo (proximal policy optimization)', 'ddpg (deep deterministic policy gradient)',
    'sac (soft actor-critic)', 'td learning', 'multi-arm bandit',
    'markov decision process (MDP)', 'policy gradient',

    # Time Series Analysis
    'time series forecasting', 'arima', 'sarima', 'sarimax', 'holt-winters',
    'exponential smoothing', 'seasonal decomposition',
    'lstm for time series', 'prophet', 'kalman filter',

    # Anomaly Detection
    'isolation forest', 'one-class SVM', 'elliptic envelope', 'local outlier factor',
    'autoencoder anomaly detection', 'robust covariance', 'dbscan anomaly detection',

    # AI Ethics & Fairness
    'explainable ai (XAI)', 'model interpretability', 'SHAP', 'LIME',
    'bias in ai', 'fairness in ai', 'ai ethics', 'adversarial attacks',
    'federated learning', 'privacy-preserving ai', 'differential privacy',

    # AI/ML Frameworks & Libraries
    'tensorflow', 'keras', 'pytorch', 'scikit-learn', 'xgboost', 'lightgbm', 'catboost',
    'fastai', 'huggingface', 'openai', 'timm', 'albumentations', 'opencv',
    'nltk', 'spacy', 'gensim', 'transformers', 'torchvision', 'torchaudio',
    'mlflow', 'wandb', 'ray', 'dask', 'pyspark', 'modin', 'autokeras', 'tpu', 'gpu acceleration',

    # Data Science & Data Engineering
    'data analytics', 'data visualization', 'feature engineering', 'data preprocessing',
    'big data', 'etl', 'data pipeline', 'airflow', 'kafka', 'hadoop', 'spark', 'presto',
    'pandas', 'numpy', 'scipy', 'matplotlib', 'seaborn', 'ggplot', 'plotly',
    'hypothesis testing', 'a/b testing', 'data wrangling', 'exploratory data analysis (EDA)',

    # DevOps & Cloud for AI/ML
    'mlops', 'ci/cd for ml', 'kubeflow', 'tensorflow serving', 'torchserve',
    'sagemaker', 'google ai platform', 'azure ml', 'vertex ai', 'mlops pipelines',
    'feature store', 'vector databases', 'faiss', 'pinecone', 'chroma',

     # Cybersecurity & Ethical Hacking
    'cybersecurity', 'penetration testing', 'ethical hacking', 'firewall', 'intrusion detection',
    'sql injection', 'xss', 'csrf', 'rce', 'buffer overflow', 'hashing',
    'encryption', 'decryption', 'public key', 'private key', 'ssl certificate',
    'malware analysis', 'ransomware', 'phishing', 'brute force attack',
    'session hijacking', 'zero-day exploit', 'vpn', 'tor', 'proxy',
    'security audit', 'vulnerability scanning', 'metasploit', 'nmap',
    'wireshark', 'sniffing', 'packet analysis', 'honey pot', 'cyber threat intelligence',
    'risk assessment', 'forensics', 'digital forensics', 'blockchain security',
    'DDoS', 'SOC', 'SIEM', 'CVE', 'zero trust security', 'IAM', '2FA', 'MITM attack',

    # Operating System & Shell Scripting
    'shell', 'bash', 'ubuntu', 'linux', 'windows', 'macos',
    'terminal', 'command line', 'ssh', 'cron job', 'process management',
    'system calls', 'multithreading', 'parallel computing', 'mutex',
    'semaphore', 'concurrency', 'deadlock', 'race condition',

    # Data Analytics & Data Science
    'data analytics', 'data visualization', 'data engineering', 'data cleaning',
    'big data', 'etl', 'data pipeline', 'data wrangling', 'business intelligence',
    'tableau', 'power bi', 'lookml', 'dax', 'google data studio',
    'spark', 'hadoop', 'aws glue', 'athena', 'presto', 'airflow',
    'pandas', 'numpy', 'scipy', 'seaborn', 'matplotlib', 'ggplot', 'plotly',
    'time series analysis', 'forecasting', 'clustering', 'regression',
    'classification', 'feature selection', 'dimensionality reduction',
    'data governance', 'data ethics', 'ab testing', 'hypothesis testing',

    # Other Topics
    'git', 'github', 'gitlab', 'bitbucket', 'version control', 'agile', 'scrum',
    'test', 'unit test', 'integration test', 'tdd', 'bdd', 'debugging', 'profiling',
    'performance optimization', 'hill climbing', 'search problem', 'optimization',
    'big o notation', 'computational complexity'

    ]
        pattern = r'\b(?:' + '|'.join(coding_keywords) + r')\b'
        return re.search(pattern, query, re.IGNORECASE) is not None

    def change_language(self, language):
        """Handle language change in the compiler."""
        self.config["current_language"] = language
        self.save_config()
        self.clear_editor()
        if language in self.similar_questions:
            self.update_similar_questions_display(language)

    def run_current_code(self):
        """Execute the code in the editor."""
        code = self.code_editor.get("1.0", tk.END).strip()
        if not code:
            messagebox.showwarning("Warning", "Please enter some code to run.")
            return
        
        self.run_code(code, self.language_var.get())

    def clear_editor(self):
        """Clear the code editor and output display."""
        self.code_editor.delete("1.0", tk.END)
        self.output_display.config(state='normal')
        self.output_display.delete("1.0", tk.END)
        self.output_display.config(state='disabled')

    def format_code(self):
        """Format the code in the editor."""
        language = self.language_var.get()
        code = self.code_editor.get("1.0", tk.END).strip()
        
        if not code:
            return
            
        try:
            if language == "python":
                import autopep8
                formatted_code = autopep8.fix_code(code)
                self.code_editor.delete("1.0", tk.END)
                self.code_editor.insert("1.0", formatted_code)
            # Add formatters for other languages as needed
            else:
                messagebox.showinfo("Info", f"Code formatting for {language} is not yet supported.")
        except Exception as e:
            messagebox.showerror("Format Error", f"Error formatting code: {str(e)}")

    def save_code(self):
        """Save the code from editor to a file."""
        code = self.code_editor.get("1.0", tk.END).strip()
        if not code:
            messagebox.showwarning("Warning", "No code to save.")
            return
            
        file_extensions = {
            "python": ".py",
            "java": ".java",
            "c": ".c",
            "cpp": ".cpp",
            "javascript": ".js"
        }
        
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            extension = file_extensions.get(self.language_var.get(), ".txt")
            filename = f"code_{timestamp}{extension}"
            
            with open(filename, "w") as f:
                f.write(code)
            
            messagebox.showinfo("Success", f"Code saved to {filename}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save code: {str(e)}")

    def change_font_size(self, delta):
        """Change font size for all text areas."""
        self.config["font_size"] = max(8, min(20, self.config["font_size"] + delta))
        font_family = 'Arial'
        self.chat_display.configure(font=(font_family, self.config["font_size"]))
        self.similar_questions_display.configure(font=(font_family, self.config["font_size"]))
        self.code_editor.configure(font=('Courier New', self.config["font_size"]))
        self.output_display.configure(font=('Courier New', self.config["font_size"]))
        self.save_config()

    def load_similar_questions(self):
        """Load similar questions from file."""
        if os.path.exists("similar_questions.json"):
            try:
                with open("similar_questions.json", "r") as f:
                    self.similar_questions = json.load(f)
            except Exception as e:
                print(f"Error loading similar questions: {e}")

    def save_similar_questions(self):
        """Save similar questions to file."""
        try:
            with open("similar_questions.json", "w") as f:
                json.dump(self.similar_questions, f)
        except Exception as e:
            print(f"Error saving similar questions: {e}")


    def update_similar_questions(self, query, language):
        """Update the similar questions list and display."""
        if language not in self.similar_questions:
            self.similar_questions[language] = []
        
        if query not in self.similar_questions[language]:
            self.similar_questions[language].append(query)
            if len(self.similar_questions[language]) > 10:  # Keep only last 10 questions
                self.similar_questions[language].pop(0)
            self.save_similar_questions()
        
        self.update_similar_questions_display(language)

    def update_similar_questions_display(self, language):
        """Update the similar questions display."""
        self.similar_questions_display.config(state='normal')
        self.similar_questions_display.delete("1.0", tk.END)
        self.similar_questions_display.insert("1.0", f"Similar questions in {language}:\n")
        if language in self.similar_questions:
            for q in reversed(self.similar_questions[language]):
                self.similar_questions_display.insert(tk.END, f"• {q}\n")
        self.similar_questions_display.config(state='disabled')

    def clear_chat(self):
        """Clear the chat history and display."""
        self.chat_display.config(state='normal')
        self.chat_display.delete(1.0, tk.END)
        self.chat_display.config(state='disabled')
        self.chat_history = []
        if os.path.exists("chat_history.json"):
            try:
                os.remove("chat_history.json")
            except Exception as e:
                self.append_to_chat(f"Error clearing chat history: {str(e)}\n")
        self.append_to_chat("Chat history cleared.\n\n")

    def toggle_theme(self):
        """Toggle between light and dark themes."""
        self.config["dark_mode"] = not self.config["dark_mode"]
        self.set_theme()
        self.save_config()

    def load_config(self):
        """Load configuration from file."""
        if os.path.exists("config.json"):
            try:
                with open("config.json", "r") as f:
                    self.config.update(json.load(f))
            except Exception as e:
                print(f"Error loading config: {e}")

    def save_config(self):
        """Save configuration to file."""
        try:
            with open("config.json", "w") as f:
                json.dump(self.config, f)
        except Exception as e:
            print(f"Error saving config: {e}")

    def load_chat_history(self):
        """Load chat history from file."""
        if os.path.exists("chat_history.json"):
            try:
                with open("chat_history.json", "r") as f:
                    self.chat_history = json.load(f)
                    # Replay chat history in display
                    for entry in self.chat_history:
                        self.append_to_chat(entry + "\n")
            except Exception as e:
                print(f"Error loading chat history: {e}")

    def save_chat_history(self):
        """Save chat history to file."""
        try:
            with open("chat_history.json", "w") as f:
                json.dump(self.chat_history, f)
            messagebox.showinfo("Success", "Chat history saved successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save chat history: {str(e)}")

    def set_theme(self):
        """Set the application theme."""
        theme = "clam" if self.config["dark_mode"] else "default"
        self.style = ttk.Style()
        self.style.theme_use(theme)
        
        bg_color = "#2d2d2d" if self.config["dark_mode"] else "white"
        fg_color = "white" if self.config["dark_mode"] else "black"
        
        self.style.configure(".", background=bg_color, foreground=fg_color)
        self.root.configure(background=bg_color)
        
        # Update all text areas
        for widget in [self.chat_display, self.similar_questions_display, 
                      self.code_editor, self.output_display]:
            if hasattr(self, 'chat_display'):  # Check if widgets exist
                widget.configure(bg=bg_color, fg=fg_color)

    def initialize_agent(self):
        """Initialize the AI agent with configuration settings."""
        self.agent = Agent(
            model=Groq(
                id="llama-3.3-70b-versatile",
                temperature=self.config["temperature"],
                max_tokens=self.config["max_tokens"]
            )
        )

    def submit_query(self):
        """Handle user query submission."""
        query = self.query_input.get().strip()
        if not query:
            return
        
        # Check if the query is coding-related
        if not self.is_coding_related(query):
            self.append_to_chat("Assistant: I specialize in coding-related queries only. Please ask about programming, software development, or related topics.\n")
            self.query_input.delete(0, tk.END)
            return

        # Clear input and display user query
        self.query_input.delete(0, tk.END)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        query_entry = f"\n[{timestamp}] You: {query}\n"
        self.append_to_chat(query_entry)
        self.chat_history.append(query_entry)
        
        try:
            self.append_to_chat("Assistant: Processing...\n")
            response = self.agent.run(query)
            response_content = response.content if hasattr(response, 'content') else str(response)
            
            # Extract and handle code blocks
            code_blocks = re.findall(r'```(\w+)?\n?(.*?)```', response_content, re.DOTALL)
            if code_blocks:
                for lang, code in code_blocks:
                    lang = lang.lower() if lang else "python"
                    self.append_to_chat(f"\nFound {lang} code block:\n")
                    # Update code editor with the code
                    self.code_editor.delete("1.0", tk.END)
                    self.code_editor.insert("1.0", code.strip())
                    self.language_var.set(lang)
                    self.validate_and_execute(code, lang)
            
            # Display the assistant's response
            response_entry = f"Assistant: {response_content}\n"
            self.append_to_chat(response_entry)
            self.chat_history.append(response_entry)
            
            # Update similar questions
            self.update_similar_questions(query, self.language_var.get())
            
            # Auto-save chat history
            self.save_chat_history()
            
        except Exception as e:
            error_message = f"Assistant: Error: {str(e)}\n"
            self.append_to_chat(error_message)
            self.chat_history.append(error_message)

    def validate_and_execute(self, code, language):
        """Validate syntax and execute code."""
        # Syntax checking
        syntax_errors = self.check_syntax(code, language)
        if syntax_errors:
            self.append_to_chat(f"\nSyntax errors found:\n{syntax_errors}\n")
            return

        # Code execution
        self.run_code(code, language)

    def run_code(self, code, language):
        """Execute the given code in the specified language."""
        compilers = {
            "python": ["python", "temp.py"],
            "c": ["gcc", "-o", "temp.out", "temp.c", "&&", "./temp.out"],
            "cpp": ["g++", "-o", "temp.out", "temp.cpp", "&&", "./temp.out"],
            "java": ["javac", "temp.java", "&&", "java", "temp"],
            "javascript": ["node", "temp.js"]
        }
        
        if language not in compilers:
            self.append_to_chat(f"\nExecution for {language} is not supported yet.\n")
            return
        
        try:
            suffix = {
                "python": ".py",
                "c": ".c",
                "cpp": ".cpp",
                "java": ".java",
                "javascript": ".js"
            }.get(language, ".txt")
            
            with tempfile.NamedTemporaryFile(mode='w', suffix=suffix, delete=False) as f:
                f.write(code)
                temp_filename = f.name
                
            base_name = temp_filename[:-len(suffix)] if suffix else temp_filename
            command = [
                part.replace("temp", base_name)
                for part in compilers[language]
            ]
            
            # Update output display
            self.output_display.config(state='normal')
            self.output_display.delete(1.0, tk.END)
            self.output_display.insert(tk.END, f"Running {language} code...\n\n")
            self.output_display.config(state='disabled')
            
            result = subprocess.run(
                " ".join(command),
                capture_output=True,
                text=True,
                timeout=10,
                shell=True
            )
            
            self.output_display.config(state='normal')
            if result.stdout:
                self.output_display.insert(tk.END, f"Output:\n{result.stdout}\n")
            if result.stderr:
                self.output_display.insert(tk.END, f"Errors:\n{result.stderr}\n")
            self.output_display.config(state='disabled')
                
        except subprocess.TimeoutExpired:
            self.output_display.config(state='normal')
            self.output_display.insert(tk.END, "\nExecution timed out after 10 seconds.\n")
            self.output_display.config(state='disabled')
        except Exception as e:
            self.output_display.config(state='normal')
            self.output_display.insert(tk.END, f"\nExecution error: {str(e)}\n")
            self.output_display.config(state='disabled')
        finally:
            try:
                os.unlink(temp_filename)
                if language == "c" or language == "cpp":
                    exec_name = base_name + ".out"
                    if os.path.exists(exec_name):
                        os.unlink(exec_name)
                if language == "java":
                    class_name = base_name + ".class"
                    if os.path.exists(class_name):
                        os.unlink(class_name)
            except Exception as e:
                print(f"Error cleaning up files: {e}")

    def check_syntax(self, code, language):
        """Check syntax for the given code and language."""
        try:
            if language == "python":
                with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                    f.write(code)
                    f.close()
                    py_compile.compile(f.name, doraise=True)
                os.unlink(f.name)
            elif language in ["c", "cpp"]:
                suffix = ".c" if language == "c" else ".cpp"
                with tempfile.NamedTemporaryFile(mode='w', suffix=suffix, delete=False) as f:
                    f.write(code)
                    f.close()
                compilers = {"c": "gcc", "cpp": "g++"}
                result = subprocess.run(
                    [compilers[language], "-fsyntax-only", f.name],
                    stderr=subprocess.PIPE,
                    text=True
                )
                os.unlink(f.name)
                if result.returncode != 0:
                    return result.stderr
            elif language == "java":
                with tempfile.NamedTemporaryFile(mode='w', suffix='.java', delete=False) as f:
                    f.write(code)
                    f.close()
                result = subprocess.run(
                    ["javac", f.name],
                    stderr=subprocess.PIPE,
                    text=True
                )
                os.unlink(f.name)
                if result.returncode != 0:
                    return result.stderr
            elif language == "javascript":
                try:
                    #import node
                    # Add JavaScript syntax checking here
                    pass
                except ImportError:
                    return "JavaScript syntax checking requires Node.js"
        except Exception as e:
            return str(e)
        return None
    def append_to_chat(self, text):
        """Append text to the chat display."""
        self.chat_display.config(state='normal')
        self.chat_display.insert(tk.END, text)
        self.chat_display.config(state='disabled')
        self.chat_display.see(tk.END)

if __name__ == "__main__":
    root = tk.Tk()
    app = CodingAssistantGUI(root)
    root.mainloop()