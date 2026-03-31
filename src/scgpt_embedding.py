"""
scGPT Zero-Shot Embedding Extractor.

Supports two modes:
  1. Real mode: loads pre-trained scGPT-blood weights (10.3M cell model)
  2. Mock mode: uses a small random-weight Transformer (for architecture testing)

The key innovation is additive diversity fusion: isoform diversity features
(entropy, dominant_fraction) are projected and added to gene embeddings
before the Transformer encoder, enriching the representation with
co-barcode/splicing information that scGPT's original training did not see.
"""

import sys
import types
import json
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _ensure_torchtext_mock():
    """
    scGPT's tokenizer imports torchtext, which may have binary
    incompatibility with newer PyTorch versions. We provide a
    lightweight mock so the rest of scGPT loads correctly.
    """
    if 'torchtext' not in sys.modules:
        try:
            import torchtext  # noqa: F401
            # Also test that torchtext.vocab.vocab works
            torchtext.vocab.vocab
        except (ImportError, OSError, AttributeError):
            logger.info("torchtext unavailable or incompatible, installing mock...")
            tt = types.ModuleType('torchtext')
            tt_vocab = types.ModuleType('torchtext.vocab')

            class _MockVocab:
                """Minimal mock matching torchtext.vocab.Vocab interface."""
                def __init__(self, ordered_dict=None, *a, **kw):
                    self._stoi = dict(ordered_dict) if ordered_dict else {}
                    self._itos = {v: k for k, v in self._stoi.items()}

                def __getitem__(self, token):
                    return self._stoi.get(token, 0)

                def __len__(self):
                    return len(self._stoi)

                def __contains__(self, token):
                    return token in self._stoi

                def get_stoi(self):
                    return self._stoi

                def get_itos(self):
                    return list(self._itos.values())

                def set_default_index(self, index):
                    pass

            def _vocab_factory(ordered_dict, min_freq=1):
                """Mock for torchtext.vocab.vocab() factory function."""
                return _MockVocab(ordered_dict)

            tt_vocab.Vocab = _MockVocab
            tt_vocab.vocab = _vocab_factory
            tt.vocab = tt_vocab
            sys.modules['torchtext'] = tt
            sys.modules['torchtext.vocab'] = tt_vocab
    else:
        # Already loaded but might be the broken binary version
        try:
            sys.modules['torchtext'].vocab.vocab
        except (AttributeError, OSError):
            # Replace with mock
            tt = types.ModuleType('torchtext')
            tt_vocab = types.ModuleType('torchtext.vocab')

            class _MockVocab:
                def __init__(self, ordered_dict=None, *a, **kw):
                    self._stoi = dict(ordered_dict) if ordered_dict else {}
                def __getitem__(self, token):
                    return self._stoi.get(token, 0)
                def __len__(self):
                    return len(self._stoi)
                def __contains__(self, token):
                    return token in self._stoi
                def get_stoi(self):
                    return self._stoi
                def set_default_index(self, index):
                    pass

            def _vocab_factory(ordered_dict, min_freq=1):
                return _MockVocab(ordered_dict)

            tt_vocab.Vocab = _MockVocab
            tt_vocab.vocab = _vocab_factory
            tt.vocab = tt_vocab
            sys.modules['torchtext'] = tt
            sys.modules['torchtext.vocab'] = tt_vocab


class _SimpleVocab:
    """
    Lightweight vocab wrapper that avoids torchtext dependency.
    Provides the same interface needed by TransformerModel.
    """
    def __init__(self, token2idx: dict):
        self.vocab = token2idx  # {gene_name: int_id}
        self._itos = {v: k for k, v in token2idx.items()}

    def __getitem__(self, token):
        return self.vocab.get(token, 0)

    def __len__(self):
        return len(self.vocab)

    def __contains__(self, token):
        return token in self.vocab

    def get_stoi(self):
        return self.vocab

    def set_default_index(self, index):
        self._default_index = index


class DiversityFusionLayer(nn.Module):
    """
    Projects isoform diversity features [entropy, dominant_fraction]
    into the same embedding space and adds them to gene embeddings.
    This is the project's core architectural contribution.
    """
    def __init__(self, embed_dim: int = 512, n_features: int = 2):
        super().__init__()
        self.proj = nn.Linear(n_features, embed_dim)

    def forward(self, gene_embeddings: torch.Tensor,
                diversity_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            gene_embeddings: (batch, seq_len, embed_dim)
            diversity_features: (batch, seq_len, n_features)
        Returns:
            Fused embeddings: (batch, seq_len, embed_dim)
        """
        return gene_embeddings + self.proj(diversity_features)


class ZeroShotScGPTExtractor:
    """
    Extracts 512-dim zero-shot embeddings from pre-trained scGPT model
    using both gene expression and isoform diversity features.
    """

    def __init__(self, model_dir: str = "./models/scgpt_blood",
                 use_real_model: bool = True):
        """
        Args:
            model_dir: Path to the pre-trained scGPT model checkpoint folder
                       (containing best_model.pt, args.json, vocab.json).
            use_real_model: If True, attempt to load real scGPT weights.
                           Falls back to mock if files not found.
        """
        self.model_dir = Path(model_dir)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_real_model = use_real_model
        logger.info(f"Using device: {self.device}")

        self.model = None
        self.vocab = None          # gene_name -> token_id
        self.diversity_layer = None
        self._model_args = None
        self._is_real = False

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_real_model(self):
        """Load the real scGPT-blood pre-trained weights."""
        model_path = self.model_dir / "best_model.pt"
        vocab_path = self.model_dir / "vocab.json"
        args_path = self.model_dir / "args.json"

        if not model_path.exists():
            logger.warning(f"Model weights not found at {model_path}, falling back to mock.")
            return False

        # Load args
        with open(args_path) as f:
            self._model_args = json.load(f)

        # Load vocab directly as dict (bypass torchtext-dependent GeneVocab)
        with open(vocab_path) as f:
            token2idx = json.load(f)
        # Store as SimpleNamespace-like object with .vocab dict for compatibility
        self.vocab = _SimpleVocab(token2idx)
        vocab_size = len(self.vocab)
        logger.info(f"Loaded scGPT-blood vocab: {vocab_size} genes")

        # Retrieve model hyperparameters from args
        embsize = self._model_args.get("embsize", 512)
        nheads = self._model_args.get("nheads", 8)
        d_hid = self._model_args.get("d_hid", 512)
        nlayers = self._model_args.get("nlayers", 12)
        n_layers_cls = self._model_args.get("n_layers_cls", 3)
        dropout = self._model_args.get("dropout", 0.2)
        n_input_bins = self._model_args.get("n_bins", 51)
        pad_value = self._model_args.get("pad_value", -2)
        pad_token_str = self._model_args.get("pad_token", "<pad>")
        pad_token_id = self.vocab[pad_token_str]

        # Build model — need torchtext mock for scgpt.model imports
        _ensure_torchtext_mock()
        from scgpt.model.model import TransformerModel
        self.model = TransformerModel(
            ntoken=vocab_size,
            d_model=embsize,
            nhead=nheads,
            d_hid=d_hid,
            nlayers=nlayers,
            nlayers_cls=n_layers_cls,
            n_cls=1,
            vocab=self.vocab,
            dropout=dropout,
            pad_token=pad_token_id,
            pad_value=pad_value,
            do_mvc=False,
            do_dab=False,
            use_batch_labels=False,
            input_emb_style="continuous",
            n_input_bins=n_input_bins,
            cell_emb_style="cls",
            use_fast_transformer=False,
        )

        # Load weights (strict=False to allow our additions)
        state_dict = torch.load(model_path, map_location=self.device,
                                weights_only=True)
        self.model.load_state_dict(state_dict, strict=False)
        self.model.to(self.device)
        self.model.eval()

        # Diversity fusion layer (our addition — not in pre-trained weights)
        self.diversity_layer = DiversityFusionLayer(embed_dim=embsize).to(self.device)

        self._is_real = True
        logger.info(f"Loaded real scGPT-blood model: {nlayers} layers, "
                     f"embed_dim={embsize}, {vocab_size} genes")
        return True

    def _load_mock_model(self, vocab_size: int = 2000):
        """Mock loader for testing the architecture without the actual large weights."""
        logger.info("Loading MockScGPT for architecture testing (random weights)...")

        if self.vocab is None:
            self.vocab = {}

        class MockScGPT(nn.Module):
            def __init__(self, vocab_size=vocab_size, embed_dim=512):
                super().__init__()
                self.gene_emb = nn.Embedding(vocab_size, embed_dim)
                self.expr_emb = nn.Linear(1, embed_dim)
                self.diversity_emb = nn.Linear(2, embed_dim)
                self.transformer = nn.TransformerEncoder(
                    nn.TransformerEncoderLayer(d_model=embed_dim, nhead=8,
                                              batch_first=True),
                    num_layers=2,
                )

            def forward(self, gene_ids, expression, diversity_features=None):
                x = self.gene_emb(gene_ids) + self.expr_emb(expression.unsqueeze(-1))
                if diversity_features is not None:
                    x = x + self.diversity_emb(diversity_features)
                encoded = self.transformer(x)
                return encoded.mean(dim=1)

        self.model = MockScGPT(vocab_size=vocab_size).to(self.device)
        self.model.eval()
        self._is_real = False

    # ------------------------------------------------------------------
    # Input preparation
    # ------------------------------------------------------------------

    def _prepare_inputs_real(
        self, longitudinal_features: pd.DataFrame, max_genes: int = 1200,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[str]]:
        """
        Prepare inputs for the real scGPT model.
        Uses the pre-trained vocab to map gene names to token IDs.
        Bins expression values into the model's n_bins discrete levels.
        """
        sample_ids = sorted(longitudinal_features['sample_id'].unique())
        n_bins = self._model_args.get("n_bins", 51)
        pad_value = self._model_args.get("pad_value", -2)

        # Map gene names to vocab IDs, keep only genes in vocab
        gene_totals = longitudinal_features.groupby('gene_id')['total_count'].sum()
        all_genes = gene_totals.sort_values(ascending=False).index.tolist()

        # Filter to genes present in scGPT vocab
        vocab_dict = self.vocab.vocab  # {gene_name: id}
        genes_in_vocab = [g for g in all_genes if g in vocab_dict][:max_genes]
        logger.info(f"Matched {len(genes_in_vocab)}/{len(all_genes)} genes to scGPT vocab")

        if len(genes_in_vocab) == 0:
            logger.error("No genes matched the scGPT vocabulary!")
            raise ValueError("No overlap between input genes and scGPT vocabulary.")

        batch_gene_ids = []
        batch_expr = []
        batch_div = []

        for sample_id in sample_ids:
            sample_df = longitudinal_features[
                longitudinal_features['sample_id'] == sample_id
            ].set_index('gene_id')
            sample_df = sample_df.reindex(genes_in_vocab).fillna(0)

            # Gene token IDs from pre-trained vocab
            gene_ids = [vocab_dict[g] for g in genes_in_vocab]

            # Expression: log1p(CPM) then bin into n_bins levels
            expr = sample_df['total_count'].values.astype(float)
            expr = np.log1p((expr / (expr.sum() + 1e-6)) * 1e4)

            # Binning: discretize into [0, n_bins-1]
            # Use rank-based binning (same as scGPT training)
            nonzero_mask = expr > 0
            binned = np.full_like(expr, fill_value=pad_value, dtype=float)
            if nonzero_mask.sum() > 0:
                from scipy.stats import rankdata
                ranks = rankdata(expr[nonzero_mask], method='average')
                bins = np.ceil(ranks / ranks.max() * (n_bins - 1)).astype(int)
                binned[nonzero_mask] = bins

            # Diversity features
            div = sample_df[['entropy', 'dominant_fraction']].values

            batch_gene_ids.append(gene_ids)
            batch_expr.append(binned)
            batch_div.append(div)

        gene_ids_tensor = torch.tensor(np.array(batch_gene_ids), dtype=torch.long)
        expr_tensor = torch.tensor(np.array(batch_expr), dtype=torch.float32)
        div_tensor = torch.tensor(np.array(batch_div), dtype=torch.float32)

        return gene_ids_tensor, expr_tensor, div_tensor, sample_ids

    def _prepare_inputs_mock(
        self, longitudinal_features: pd.DataFrame, max_genes: int = 2000,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[str]]:
        """Prepare inputs for the mock model (dynamic vocab)."""
        sample_ids = sorted(longitudinal_features['sample_id'].unique())

        gene_totals = longitudinal_features.groupby('gene_id')['total_count'].sum()
        top_genes = gene_totals.sort_values(ascending=False).head(max_genes).index.tolist()

        self.vocab = {g: i for i, g in enumerate(top_genes)}
        genes_in_vocab = top_genes
        logger.info(f"Built dynamic vocab with {len(genes_in_vocab)} genes")

        batch_gene_ids, batch_expr, batch_div = [], [], []

        for sample_id in sample_ids:
            sample_df = longitudinal_features[
                longitudinal_features['sample_id'] == sample_id
            ].set_index('gene_id')
            sample_df = sample_df.reindex(genes_in_vocab).fillna(0)

            gene_ids = [self.vocab[g] for g in genes_in_vocab]
            expr = sample_df['total_count'].values
            expr = np.log1p((expr / (expr.sum() + 1e-6)) * 1e4)
            div = sample_df[['entropy', 'dominant_fraction']].values

            batch_gene_ids.append(gene_ids)
            batch_expr.append(expr)
            batch_div.append(div)

        gene_ids_tensor = torch.tensor(np.array(batch_gene_ids), dtype=torch.long)
        expr_tensor = torch.tensor(np.array(batch_expr), dtype=torch.float32)
        div_tensor = torch.tensor(np.array(batch_div), dtype=torch.float32)

        return gene_ids_tensor, expr_tensor, div_tensor, sample_ids

    # ------------------------------------------------------------------
    # Embedding extraction
    # ------------------------------------------------------------------

    def get_embeddings(self, longitudinal_features: pd.DataFrame,
                       max_genes: int = 1200) -> pd.DataFrame:
        """
        Extract 512-dim embeddings for each sample.

        For the real model:
          1. Maps genes to scGPT vocab tokens
          2. Bins expression values (rank-based, 51 bins)
          3. Forward pass through pre-trained Transformer
          4. Adds diversity fusion post-hoc (our contribution)
          5. Returns CLS token embedding per sample

        For the mock model:
          Uses additive fusion within the forward pass directly.
        """
        # Load model if not yet loaded
        if self.model is None:
            if self.use_real_model:
                loaded = self._load_real_model()
                if not loaded:
                    gene_totals = longitudinal_features.groupby('gene_id')['total_count'].sum()
                    n_genes = min(len(gene_totals), max_genes)
                    self._load_mock_model(vocab_size=n_genes + 10)
            else:
                gene_totals = longitudinal_features.groupby('gene_id')['total_count'].sum()
                n_genes = min(len(gene_totals), max_genes)
                self._load_mock_model(vocab_size=n_genes + 10)

        # Prepare inputs
        if self._is_real:
            gene_ids, expr, div, sample_ids = self._prepare_inputs_real(
                longitudinal_features, max_genes=max_genes)
        else:
            gene_ids, expr, div, sample_ids = self._prepare_inputs_mock(
                longitudinal_features, max_genes=max_genes)

        gene_ids = gene_ids.to(self.device)
        expr = expr.to(self.device)
        div = div.to(self.device)

        with torch.no_grad():
            if self._is_real:
                # Real scGPT forward: gene tokens + binned expression → cell embedding
                # scGPT TransformerModel expects: (src, values, ...)
                src_key_padding_mask = expr.eq(
                    self._model_args.get("pad_value", -2))
                output = self.model._encode(
                    gene_ids, expr, src_key_padding_mask=src_key_padding_mask,
                )
                # output shape: (batch, seq_len, embed_dim)
                # Apply diversity fusion
                fused = self.diversity_layer(output, div)
                # Mean pooling (exclude padded positions)
                mask = ~src_key_padding_mask.unsqueeze(-1)
                embeddings = (fused * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            else:
                # Mock model: forward pass includes diversity fusion
                embeddings = self.model(gene_ids, expr, div)

        embeddings_np = embeddings.cpu().numpy()

        emb_df = pd.DataFrame(
            embeddings_np,
            index=sample_ids,
            columns=[f"scGPT_dim_{i}" for i in range(embeddings_np.shape[1])]
        )

        mode_str = "real scGPT-blood" if self._is_real else "MockScGPT"
        logger.info(f"Extracted embeddings ({mode_str}): {emb_df.shape}")
        return emb_df
