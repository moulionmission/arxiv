"""
keyword_analyzer.py
Explainable multi-category classification for arXiv papers.

Combines three approaches:
1. TF-IDF discriminative keywords (data-driven from training corpus)
2. Curated seed keywords (comprehensive human-curated list)
3. Zero-shot embedding classification (semantic similarity via sentence-transformers)
"""

import os
import re
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


class KeywordAnalyzer:
    """Explainable keyword-based classifier for arXiv papers.

    Builds a discriminative keyword dictionary per category using TF-IDF
    scores supplemented by a comprehensive curated keyword list, and
    provides transparent, evidence-backed classification.
    """

    CATEGORIES = ["AI", "ML", "NLP", "Computer Vision", "Robotics"]

    # ──────────────────────────────────────────────────────────────────
    # Comprehensive curated keyword dictionary
    # ──────────────────────────────────────────────────────────────────
    SEED_KEYWORDS = {
        "AI": {
            # Core AI
            "artificial intelligence": 0.92, "intelligent system": 0.85,
            "intelligent agent": 0.88, "autonomous agent": 0.88,
            "agentic": 0.90, "reasoning": 0.85, "planning": 0.72,
            "decision making": 0.78, "problem solving": 0.75,
            "knowledge representation": 0.82, "commonsense reasoning": 0.85,
            "symbolic reasoning": 0.82, "neuro-symbolic": 0.80,
            "cognitive architecture": 0.78, "world model": 0.72,
            "memory system": 0.68, "self-improvement": 0.70,
            "self-reflection": 0.72, "tool use": 0.68,
            "function calling": 0.70, "agent workflow": 0.78,
            "chain of thought": 0.72, "reasoning model": 0.80,
            "inference engine": 0.75,
            # General capability
            "intelligence": 0.88, "autonomy": 0.70, "adaptation": 0.55,
            "goal-directed": 0.72, "multi-agent": 0.75,
            "autonomous system": 0.72, "cognitive": 0.70,
            "deliberation": 0.68, "strategic reasoning": 0.78,
            "task planning": 0.72, "policy reasoning": 0.75,
            "long-term planning": 0.72, "alignment": 0.65,
            "safety": 0.58, "thinking": 0.68, "explainability": 0.70,
            "interpretability": 0.68,
        },
        "ML": {
            # Learning paradigms
            "machine learning": 0.80, "deep learning": 0.72,
            "supervised learning": 0.82, "unsupervised learning": 0.82,
            "semi-supervised learning": 0.82, "self-supervised learning": 0.80,
            "reinforcement learning": 0.78, "online learning": 0.75,
            "continual learning": 0.78, "transfer learning": 0.75,
            "meta learning": 0.78, "active learning": 0.75,
            "curriculum learning": 0.78, "federated learning": 0.80,
            # Models
            "neural network": 0.68, "deep neural network": 0.72,
            "dnn": 0.70, "cnn": 0.60, "rnn": 0.60, "lstm": 0.62,
            "transformer": 0.50, "autoencoder": 0.75,
            "variational autoencoder": 0.78, "vae": 0.75,
            "gan": 0.72, "diffusion model": 0.60,
            "mixture of experts": 0.78, "moe": 0.75,
            # Training
            "training data": 0.72, "training": 0.55,
            "synthetic data": 0.72, "optimization": 0.72,
            "gradient descent": 0.82, "gradient": 0.68,
            "backpropagation": 0.85, "fine tuning": 0.70,
            "fine-tuning": 0.70, "instruction tuning": 0.68,
            "pretraining": 0.65, "post-training": 0.65,
            "model distillation": 0.75, "distillation": 0.72,
            "regularization": 0.75, "hyperparameter": 0.78,
            "loss function": 0.78, "objective function": 0.75,
            "learning rate": 0.80, "batch": 0.50,
            "epoch": 0.55, "parameters": 0.55, "model": 0.30,
            # Evaluation
            "benchmark": 0.45, "generalization": 0.68,
            "cross validation": 0.78, "model calibration": 0.72,
            # Representation
            "embedding": 0.55, "latent space": 0.68,
            "feature extraction": 0.65, "representation learning": 0.72,
            "dimensionality reduction": 0.75, "clustering": 0.62,
            "lora": 0.78, "adapters": 0.75, "ensemble": 0.65,
            "pretrained": 0.58, "architecture": 0.50,
            "convergence": 0.72, "overfitting": 0.72,
        },
        "NLP": {
            # Language models
            "language model": 0.90, "large language model": 0.92,
            "llm": 0.90, "foundation model": 0.72,
            "instruction model": 0.80, "chat model": 0.82,
            "conversational ai": 0.80,
            # Text processing
            "text": 0.55, "document": 0.50, "corpus": 0.72,
            "token": 0.55, "tokenization": 0.82, "tokenizer": 0.85,
            "vocabulary": 0.68, "sequence modeling": 0.65,
            "sequence-to-sequence": 0.80, "seq2seq": 0.82,
            # Tasks
            "machine translation": 0.88, "translation": 0.78,
            "summarization": 0.85, "text generation": 0.82,
            "question answering": 0.85, "qa": 0.65,
            "dialogue system": 0.82, "chatbot": 0.80,
            "sentiment analysis": 0.88, "named entity recognition": 0.88,
            "ner": 0.82, "information extraction": 0.80,
            "text classification": 0.78, "document classification": 0.75,
            "paraphrase": 0.78, "text simplification": 0.78,
            # Linguistics
            "syntax": 0.72, "semantics": 0.62, "pragmatics": 0.75,
            "discourse": 0.72, "lexical": 0.70, "morphology": 0.75,
            "language understanding": 0.82, "natural language understanding": 0.85,
            "nlu": 0.82, "natural language generation": 0.85,
            "nlg": 0.82, "linguistic": 0.78,
            # Modern LLM
            "prompt": 0.65, "prompting": 0.70,
            "instruction following": 0.75,
            "retrieval augmented generation": 0.82, "rag": 0.72,
            "context window": 0.75, "in-context learning": 0.78,
            "attention": 0.50, "self-attention": 0.62,
            "positional encoding": 0.72, "multilingual": 0.80,
            "cross-lingual": 0.85, "lingual": 0.82,
            # Speech
            "speech recognition": 0.78, "asr": 0.78,
            "speech-to-text": 0.80, "text-to-speech": 0.80,
            "tts": 0.78, "speech-language": 0.78,
            "spoken language": 0.78, "speech": 0.62,
        },
        "Computer Vision": {
            # Core vision
            "computer vision": 0.92, "visual recognition": 0.88,
            "visual understanding": 0.85, "image processing": 0.85,
            "image analysis": 0.85, "visual perception": 0.82,
            # Images
            "image": 0.78, "photograph": 0.75, "picture": 0.65,
            "pixel": 0.80, "image feature": 0.78, "image embedding": 0.78,
            "image representation": 0.78,
            # Detection & Recognition
            "object detection": 0.90, "object recognition": 0.88,
            "image classification": 0.82, "visual classification": 0.82,
            "segmentation": 0.80, "semantic segmentation": 0.88,
            "instance segmentation": 0.88, "panoptic segmentation": 0.88,
            "scene understanding": 0.78,
            # Localization
            "bounding box": 0.85, "localization": 0.65,
            "object tracking": 0.82, "multi-object tracking": 0.85,
            "pose estimation": 0.80, "keypoint detection": 0.82,
            # 3D Vision
            "depth estimation": 0.80, "stereo vision": 0.85,
            "point cloud": 0.78, "3d reconstruction": 0.82,
            "structure from motion": 0.82, "nerf": 0.80,
            # Video
            "video understanding": 0.85, "action recognition": 0.82,
            "activity recognition": 0.80, "video generation": 0.78,
            "video segmentation": 0.82, "temporal localization": 0.75,
            "video": 0.72, "video content": 0.75,
            # Multimodal vision
            "vision-language": 0.82, "vision language": 0.82,
            "image captioning": 0.85, "visual question answering": 0.85,
            "vqa": 0.82, "multimodal": 0.62, "visual grounding": 0.82,
            "image-text": 0.78, "cross-modal": 0.72,
            # General
            "vision": 0.82, "visual": 0.78, "camera": 0.72,
            "scene": 0.62, "texture": 0.68, "resolution": 0.55,
            "convolution": 0.72, "feature map": 0.75,
            "optical flow": 0.80, "diffusion": 0.55,
            "detection": 0.58, "recognition": 0.55,
            "3d": 0.60, "depth": 0.62, "face": 0.72,
        },
        "Robotics": {
            # Robots
            "robot": 0.92, "robotic": 0.92, "robotics": 0.95,
            "humanoid": 0.88, "mobile robot": 0.90,
            "robotic arm": 0.90, "robotic manipulator": 0.90,
            # Motion
            "locomotion": 0.88, "manipulation": 0.85,
            "grasping": 0.90, "pick and place": 0.88,
            "motion planning": 0.88, "path planning": 0.85,
            "trajectory planning": 0.85, "trajectory optimization": 0.85,
            "trajectory": 0.75,
            # Control
            "control system": 0.78, "feedback control": 0.82,
            "model predictive control": 0.85, "mpc": 0.80,
            "pid control": 0.82, "controller": 0.65,
            "autonomous navigation": 0.85,
            # Perception
            "robot perception": 0.85, "sensor fusion": 0.80,
            "lidar": 0.82, "imu": 0.78, "depth camera": 0.78,
            "rgb-d": 0.80, "environmental mapping": 0.78,
            # Navigation
            "navigation": 0.82, "obstacle avoidance": 0.85,
            "mapping": 0.60, "slam": 0.85, "route planning": 0.80,
            # Embodied AI
            "embodied agent": 0.88, "embodied intelligence": 0.88,
            "embodied reasoning": 0.85, "embodied learning": 0.85,
            "embodied": 0.82, "physical interaction": 0.78,
            "real-world interaction": 0.75,
            # Hardware
            "actuator": 0.90, "end effector": 0.90,
            "manipulator": 0.82, "gripper": 0.90,
            "robot platform": 0.85, "autonomous vehicle": 0.78,
            "drone": 0.78, "uav": 0.80, "quadruped": 0.88,
            "sensor": 0.65, "motor": 0.68,
            "robots": 0.90, "control": 0.55, "dexterous": 0.82,
            "kinematics": 0.85, "dynamics": 0.55,
        },
    }

    def __init__(
        self,
        data_path: str = "data/processed/arxiv_10yr_10k_unbiased_cleaned.csv",
        cache_path: str = "models/keyword_cache.pkl",
    ):
        self.data_path = data_path
        self.cache_path = cache_path
        self.category_keywords: dict[str, dict[str, float]] = {}

        if os.path.exists(self.cache_path):
            self._load_cache()
        else:
            self.build_keyword_dictionary()

    # ------------------------------------------------------------------
    # Dictionary construction
    # ------------------------------------------------------------------

    def build_keyword_dictionary(self) -> None:
        """Build discriminative keyword dictionaries from training data.

        Combines:
        1. TF-IDF discriminative keywords (top 80 per category)
        2. Curated seed keywords (comprehensive human-curated list)
        """
        df = pd.read_csv(self.data_path)
        df = df[df["label"].isin(self.CATEGORIES)].copy()
        df["abstract_clean"] = df["abstract_clean"].fillna("").astype(str)

        # Shared TF-IDF vectorizer
        vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            stop_words="english",
            min_df=3,
        )
        tfidf_matrix = vectorizer.fit_transform(df["abstract_clean"])
        feature_names = vectorizer.get_feature_names_out()

        # Global mean TF-IDF per term
        global_mean = np.asarray(tfidf_matrix.mean(axis=0)).flatten()
        epsilon = 1e-8

        self.category_keywords = {}

        for category in self.CATEGORIES:
            cat_mask = (df["label"] == category).values
            if cat_mask.sum() == 0:
                self.category_keywords[category] = {}
                continue

            cat_mean = np.asarray(tfidf_matrix[cat_mask].mean(axis=0)).flatten()
            disc_scores = cat_mean / (global_mean + epsilon)
            top_indices = disc_scores.argsort()[::-1][:80]

            raw = {feature_names[i]: disc_scores[i] for i in top_indices}

            # Normalise to [0, 1]
            max_score = max(raw.values()) if raw else 1.0
            min_score = min(raw.values()) if raw else 0.0
            denom = max_score - min_score if max_score != min_score else 1.0
            normalised = {k: (v - min_score) / denom for k, v in raw.items()}

            self.category_keywords[category] = normalised

        # Merge curated seed keywords — only add if NOT already present
        for category in self.CATEGORIES:
            seeds = self.SEED_KEYWORDS.get(category, {})
            for keyword, importance in seeds.items():
                if keyword not in self.category_keywords[category]:
                    self.category_keywords[category][keyword] = importance

        self._save_cache()

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    def analyze(self, text: str) -> dict:
        """Analyse an abstract and return category breakdown with evidence.

        Returns
        -------
        dict with keys:
            category_breakdown : normalised score per category (sum ≈ 1)
            keyword_evidence : matched keywords per category (top 8)
            reasoning : human-readable sentence per category
        """
        if not text or not text.strip():
            return {
                "category_breakdown": {c: 1.0 / len(self.CATEGORIES) for c in self.CATEGORIES},
                "keyword_evidence": {c: [] for c in self.CATEGORIES},
                "reasoning": {c: f"Minimal {c} relevance" for c in self.CATEGORIES},
            }

        text_lower = text.lower()

        raw_scores: dict[str, float] = {}
        matched_keywords: dict[str, list[tuple[str, float]]] = {}

        for category, keywords in self.category_keywords.items():
            matches: list[tuple[str, float]] = []
            for keyword, importance in keywords.items():
                # Bigrams: substring check; unigrams: word boundary match
                if " " in keyword or "-" in keyword:
                    if keyword in text_lower:
                        matches.append((keyword, importance))
                else:
                    if re.search(rf"\b{re.escape(keyword)}\b", text_lower):
                        matches.append((keyword, importance))

            matches.sort(key=lambda x: x[1], reverse=True)
            matched_keywords[category] = matches
            raw_scores[category] = sum(imp for _, imp in matches)

        # Normalise to probabilities
        total = sum(raw_scores.values())
        if total == 0:
            category_breakdown = {c: 1.0 / len(self.CATEGORIES) for c in self.CATEGORIES}
        else:
            category_breakdown = {c: raw_scores[c] / total for c in self.CATEGORIES}

        # Top-8 evidence per category
        keyword_evidence = {
            c: matched_keywords[c][:8] for c in self.CATEGORIES
        }

        # Human-readable reasoning
        reasoning: dict[str, str] = {}
        for category in self.CATEGORIES:
            score = category_breakdown[category]
            top_kw = [kw for kw, _ in matched_keywords[category][:3]]
            kw_str = ", ".join(top_kw)

            if score > 0.3:
                reasoning[category] = f"Strong {category} focus due to: {kw_str}"
            elif score > 0.15:
                reasoning[category] = f"Significant {category} elements: {kw_str}"
            elif score > 0.08:
                reasoning[category] = f"Moderate {category} aspects: {kw_str}"
            elif score > 0.03:
                top_kw_minor = [kw for kw, _ in matched_keywords[category][:2]]
                reasoning[category] = f"Minor {category} aspects: {', '.join(top_kw_minor)}"
            else:
                if matched_keywords[category]:
                    reasoning[category] = f"Trace {category} mention: {matched_keywords[category][0][0]}"
                else:
                    reasoning[category] = f"No {category} keywords detected"

        return {
            "category_breakdown": category_breakdown,
            "keyword_evidence": keyword_evidence,
            "reasoning": reasoning,
        }

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def get_category_keywords(self, category: str, top_n: int = 20) -> list[tuple[str, float]]:
        """Return the top-N keywords for a category."""
        if category not in self.category_keywords:
            raise ValueError(
                f"Unknown category '{category}'. "
                f"Available: {list(self.category_keywords.keys())}"
            )
        items = sorted(
            self.category_keywords[category].items(),
            key=lambda x: x[1],
            reverse=True,
        )
        return items[:top_n]

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def _save_cache(self) -> None:
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        with open(self.cache_path, "wb") as f:
            pickle.dump(self.category_keywords, f)
        print(f"[KeywordAnalyzer] Keyword cache saved → {self.cache_path}")

    def _load_cache(self) -> None:
        with open(self.cache_path, "rb") as f:
            self.category_keywords = pickle.load(f)
        print(f"[KeywordAnalyzer] Keyword cache loaded ← {self.cache_path}")


# ======================================================================
# Zero-Shot Embedding Classifier
# ======================================================================

class EmbeddingClassifier:
    """Zero-shot embedding-based classifier using sentence-transformers.
    
    Compares paper embeddings against category prototype descriptions
    using cosine similarity.
    """

    CATEGORY_DESCRIPTIONS = {
        "AI": (
            "artificial intelligence, reasoning, planning, intelligent agents, "
            "decision making, commonsense reasoning, autonomous agents, "
            "cognitive architecture, multi-agent systems, alignment, safety, "
            "chain of thought, tool use, problem solving, strategic reasoning"
        ),
        "ML": (
            "machine learning, deep learning, neural networks, optimization, "
            "training, gradient descent, reinforcement learning, transfer learning, "
            "model distillation, regularization, hyperparameters, loss functions, "
            "supervised learning, unsupervised learning, meta-learning, "
            "representation learning, embeddings, fine-tuning, pretraining"
        ),
        "NLP": (
            "natural language processing, language models, text understanding, "
            "machine translation, summarization, question answering, sentiment analysis, "
            "named entity recognition, text generation, dialogue systems, "
            "tokenization, multilingual, cross-lingual, speech recognition, "
            "large language models, prompting, instruction following"
        ),
        "Computer Vision": (
            "computer vision, image understanding, object detection, visual recognition, "
            "image classification, segmentation, scene understanding, video analysis, "
            "pose estimation, 3D reconstruction, depth estimation, optical flow, "
            "image captioning, visual question answering, multimodal vision"
        ),
        "Robotics": (
            "robotics, robot manipulation, navigation, control systems, "
            "motion planning, grasping, locomotion, trajectory optimization, "
            "sensor fusion, lidar, autonomous vehicles, embodied agents, "
            "actuators, grippers, humanoid robots, SLAM, obstacle avoidance"
        ),
    }

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize with a sentence-transformer model.
        
        Uses all-MiniLM-L6-v2 by default (fast, 80MB, good quality).
        """
        from sentence_transformers import SentenceTransformer
        print(f"[EmbeddingClassifier] Loading {model_name}...")
        self.model = SentenceTransformer(model_name)
        
        # Pre-compute category embeddings
        self.category_embeddings = {}
        for cat, desc in self.CATEGORY_DESCRIPTIONS.items():
            self.category_embeddings[cat] = self.model.encode(desc)
        print(f"[EmbeddingClassifier] Ready with {len(self.category_embeddings)} categories")

    def classify(self, text: str) -> dict[str, float]:
        """Return normalised similarity scores per category."""
        from sklearn.metrics.pairwise import cosine_similarity
        
        paper_emb = self.model.encode(text).reshape(1, -1)
        
        raw_scores = {}
        for cat, cat_emb in self.category_embeddings.items():
            sim = cosine_similarity(paper_emb, cat_emb.reshape(1, -1))[0][0]
            raw_scores[cat] = max(0, float(sim))  # Clamp negatives
        
        # Normalise to sum=1
        total = sum(raw_scores.values())
        if total > 0:
            return {k: v / total for k, v in raw_scores.items()}
        return {k: 0.2 for k in raw_scores}


# ======================================================================
# Quick smoke-test
# ======================================================================

if __name__ == "__main__":
    print("=" * 72)
    print("KeywordAnalyzer – building dictionary & running demo")
    print("=" * 72)

    analyzer = KeywordAnalyzer()

    # Show keyword counts per category
    for cat in KeywordAnalyzer.CATEGORIES:
        total = len(analyzer.category_keywords[cat])
        print(f"  {cat}: {total} keywords")

    # Test 1: NLP paper
    print("\n" + "=" * 72)
    print("Test 1: NLP Paper")
    print("=" * 72)
    nlp_text = (
        "Stance detection, the task of determining whether a text is in favor of, "
        "against, or neutral towards a given target, remains challenging in "
        "low-resource languages. We present a cross-lingual framework that transfers "
        "stance detection capability from English to twelve typologically diverse "
        "languages using multilingual transformer representations."
    )
    result = analyzer.analyze(nlp_text)
    for cat, score in sorted(result["category_breakdown"].items(), key=lambda x: -x[1]):
        kws = [kw for kw, _ in result["keyword_evidence"].get(cat, [])[:4]]
        print(f"  {cat}: {score*100:.1f}%  → {kws}")

    # Test 2: Gemini paper
    print("\n" + "=" * 72)
    print("Test 2: Gemini Multi-Category Paper")
    print("=" * 72)
    gemini_text = (
        "In this report, we introduce the Gemini 2.X model family. "
        "Gemini 2.5 Pro is our most capable model yet, achieving SoTA performance "
        "on frontier coding and reasoning benchmarks. It excels at multimodal "
        "understanding and can process up to 3 hours of video content. "
        "Its unique combination of long context, multimodal and reasoning "
        "capabilities can unlock new agentic workflows."
    )
    result = analyzer.analyze(gemini_text)
    for cat, score in sorted(result["category_breakdown"].items(), key=lambda x: -x[1]):
        kws = [kw for kw, _ in result["keyword_evidence"].get(cat, [])[:4]]
        print(f"  {cat}: {score*100:.1f}%  → {kws}")
    print("\nReasoning:")
    for cat, reason in result["reasoning"].items():
        if result["category_breakdown"].get(cat, 0) > 0.03:
            print(f"  {reason}")

    # Test 3: Embedding classifier
    print("\n" + "=" * 72)
    print("Test 3: Embedding Classifier")
    print("=" * 72)
    emb_clf = EmbeddingClassifier()
    for name, text in [("NLP", nlp_text), ("Gemini", gemini_text)]:
        scores = emb_clf.classify(text)
        print(f"\n{name}:")
        for cat, score in sorted(scores.items(), key=lambda x: -x[1]):
            print(f"  {cat}: {score*100:.1f}%")
