from typing import List, Literal, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
import numpy as np
import logging

logger = logging.getLogger(__name__)

# --- Helper Modules (PositionalEncoding and TimestepEmbedder) ---

class PositionalEncoding(nn.Module):
    """
    Positional Encoding module for Transformer models.
    This module generates positional encodings for input sequences, which are added to the input embeddings.
    The encodings are based on sine and cosine functions of different frequencies.
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        """
        Positional encoding for Transformer models.
        :param d_model: Dimension of the model (embedding size)
        :param dropout: Dropout probability
        :param max_len: Maximum length of the sequence for positional encoding
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe) # Shape: [1, max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for positional encoding.
        :param x: Input tensor of shape [batch_size, seq_len, d_model]
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class TimestepEmbedder(nn.Module):
    """
    Timestep embedding for diffusion models.
    This module generates embeddings for timesteps, which are used to condition the model on the current timestep.
    The embeddings are based on a positional encoding table and an MLP.
    """

    def __init__(self, latent_dim: int, pos_encoder: PositionalEncoding):
        """
        Timestep embedding for diffusion models.
        :param latent_dim: Dimension of the latent space
        :param pos_encoder: Positional encoding instance
        """
        super().__init__()
        self.latent_dim = latent_dim
        # Use the pe table from the provided positional encoder instance
        self.time_pos_encoder_pe = pos_encoder
        
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim),
        )

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor: # timesteps: [bs]
        """
        Forward pass for timestep embedding.
        :param timesteps: Tensor of shape [batch_size] containing timestep indices
        :return: Timestep embedding of shape [batch_size, latent_dim]
        """
        time_pos_encoding_table = self.time_pos_encoder_pe.pe.squeeze(0).to(timesteps.device)
        time_enc = time_pos_encoding_table[timesteps.long()] # [bs, latent_dim]
        return self.mlp(time_enc) # [bs, latent_dim]


# --- Option 2: GRU-based TimestepEmbedder ---

class TimestepEmbedderGRU(nn.Module):
    """
    Timestep embedding for diffusion models using a GRU.
    This module generates embeddings for timesteps, which are used to condition the model on the current timestep.
    The embeddings are based on a positional encoding table and a GRU layer.
    """

    def __init__(self, latent_dim: int, pos_encoder: PositionalEncoding, 
                    gru_num_layers: int = 1, gru_dropout: float = 0.0
                ):
        """
        Timestep embedding for diffusion models using a GRU.

        :param latent_dim: Dimension of the latent space (output dimension).
        :param pos_encoder: Positional encoding instance. Its 'pe' table is used.
        :param gru_num_layers: Number of layers in the GRU.
        :param gru_dropout: Dropout probability for GRU if gru_num_layers > 1.
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.time_pos_encoder_pe = pos_encoder # [max_len, latent_dim]
        
        self.gru = nn.GRU(
            input_size=latent_dim,    # Input features to GRU is latent_dim (from pos_encoder)
            hidden_size=latent_dim,   # GRU hidden size, outputting to latent_dim
            num_layers=gru_num_layers,
            batch_first=True,         # Expects input as [batch, seq_len, features]
            dropout=gru_dropout if gru_num_layers > 1 else 0.0
        )

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor: # timesteps: [bs]
        """
        Forward pass for GRU-based timestep embedding.
        :param timesteps: Tensor of shape [batch_size] containing timestep indices.
        :return: Timestep embedding of shape [batch_size, latent_dim].
        """
        time_pos_encoding_table = self.time_pos_encoder_pe.pe.squeeze(0).to(timesteps.device)
        t_pos_enc = time_pos_encoding_table[timesteps.long()] # [bs, latent_dim]
        t_pos_enc_seq = t_pos_enc.unsqueeze(1) # Convert to [bs, seq_len=1, latent_dim] for GRU
        
        # output_seq: [bs, 1, latent_dim], last_hidden: [num_gru_layers, bs, latent_dim]
        _, last_hidden = self.gru(t_pos_enc_seq)
        
        # Use the hidden state of the last GRU layer
        time_emb = last_hidden[-1] # Takes the hidden state from the top-most layer -> [bs, latent_dim]

        # Optional: if self.final_mlp exists:
        # time_emb = self.final_mlp(time_emb)
            
        return time_emb


# --- MDM with Armature Class Conditioning ---

class ArmatureMDM(nn.Module):
    """
    Armature-conditioned Motion Diffusion Model (ArmatureMDM).
    This model integrates motion features with text and armature class conditions using a Transformer backbone.
    It supports two-stage conditioning policies for text-time and armature integration.
    """
    
    def __init__(self,
                    num_motion_features: int,
                    latent_dim: int = 768,
                    ff_size: int = 1024, # For Transformer backbone
                    num_layers: int = 8,  # For Transformer backbone
                    num_heads: int = 4,   # For Transformer backbone
                    dropout: float = 0.1, # General dropout
                    activation: str = "gelu", # For Transformer backbone
                    sbert_model_name: str = 'all-mpnet-base-v2',
                    max_armature_classes: int = 10,
                    armature_embedding_dim: int = 64, # Raw dimension for armature class ID
                    text_cond_dropout_prob: float = 0.1,
                    armature_cond_dropout_prob: float = 0.1,
                    max_seq_len: int = 5000, # For PositionalEncoding
                    # Two-stage conditioning policies
                    text_time_conditioning_policy: Literal["add", "concat", "mean", "weighted_mean"] = "add",
                    armature_integration_policy: Literal["add", "concat", "mean", "weighted_mean"] = "add",
                    # Optional final MLP refinement for algebraic armature integration
                    use_final_algebraic_refinement_encoder: bool = True,
                    algebraic_refinement_hidden_dim: Optional[int] = None, # Hidden dim for the final refinement MLP
                    timestep_embedder_type: Literal["mlp", "gru"] = "mlp",
                    timestep_gru_num_layers: int = 1, # Used if timestep_embedder_type is "gru"
                    timestep_gru_dropout: float = 0.0  # Used if timestep_embedder_type is "gru"
                ):
        """
        Armature-conditioned Motion Diffusion Model (ArmatureMDM).

        :param num_motion_features: Number of features per frame for input motion x.
        :param latent_dim: Core latent dimension for the model's hidden states and combined conditions.
        :param ff_size: Feedforward size for Transformer layers.
        :param num_layers: Number of Transformer layers in the backbone.
        :param num_heads: Number of attention heads in Transformer layers.
        :param dropout: General dropout for the backbone and positional encoding.
        :param activation: Activation function for Transformer layers (e.g., "gelu").
        :param sbert_model_name: Name of the Sentence-BERT model for text conditioning.
        :param max_armature_classes: Maximum number of armature classes for the nn.Embedding layer.
        :param armature_embedding_dim: Dimension of the raw armature class embedding from nn.Embedding.
        :param text_cond_dropout_prob: Dropout probability for text conditioning (for CFG).
        :param armature_cond_dropout_prob: Dropout probability for armature class conditioning (for CFG).
        :param max_seq_len: Maximum sequence length for positional encoding.
        :param text_time_conditioning_policy: Policy to combine text and time embeddings ("add", "concat", "mean", "weighted_mean").
        :param armature_integration_policy: Policy to integrate armature embedding with combined text-time ("add", "concat", "mean", "weighted_mean").
        :param use_final_algebraic_refinement_encoder: Whether to use an MLP after the final algebraic
                                                       integration of the armature embedding.
        :param algebraic_refinement_hidden_dim: Hidden dim for the final refinement MLP. If None, uses latent_dim.
        :param timestep_embedder_type: Type of timestep embedder ("mlp" or "gru").
        :param timestep_gru_num_layers: Number of GRU layers if using GRU-based timestep embedder.
        :param timestep_gru_dropout: Dropout probability for GRU layers if using GRU-based timestep embedder.
        """
        super().__init__()

        self.num_motion_features = num_motion_features
        self.latent_dim = latent_dim
        self.text_time_conditioning_policy = text_time_conditioning_policy
        self.armature_integration_policy = armature_integration_policy
        self.use_final_algebraic_refinement_encoder = use_final_algebraic_refinement_encoder

        # 1. Input/Output Processing for Motion
        self.motion_input_projection = nn.Linear(num_motion_features, latent_dim)
        self.motion_output_projection = nn.Linear(latent_dim, num_motion_features)

        # 2. Positional Encoding
        self.pos_encoder = PositionalEncoding(latent_dim, dropout, max_len=max_seq_len)
        
        # 3. Timestep Embedding
        if timestep_embedder_type.lower() == "gru":
            logger.info(f"Using GRU-based TimestepEmbedder (GRU layers: {timestep_gru_num_layers}).")
            self.time_embedder = TimestepEmbedderGRU(
                latent_dim, 
                self.pos_encoder, 
                gru_num_layers=timestep_gru_num_layers,
                gru_dropout=timestep_gru_dropout
            )
        elif timestep_embedder_type.lower() == "mlp":
            logger.info("Using MLP-based TimestepEmbedder (classic).")
            self.time_embedder = TimestepEmbedder(latent_dim, self.pos_encoder)
        else:
            raise ValueError(f"Unsupported timestep_embedder_type: {timestep_embedder_type}. Choose 'mlp' or 'gru'.")
        
        # 4. Text Conditioning (Sentence-BERT)
        logger.info(f"Loading Sentence-BERT model: {sbert_model_name}...")
        self.sbert_model = self._load_and_freeze_sbert(sbert_model_name)
        sbert_output_dim = self.sbert_model.get_sentence_embedding_dimension()
        self.text_projection = nn.Linear(sbert_output_dim, latent_dim)
        self.text_cond_dropout_prob = text_cond_dropout_prob
        logger.info(f"Sentence-BERT model '{sbert_model_name}' loaded (Output: {sbert_output_dim}-dim). "
                    f"Text embedding projected to latent_dim: {latent_dim}")

        # 5. Armature Class Conditioning
        logger.info(f"Armature class conditioning: Max classes {max_armature_classes}, Raw Embedding dim {armature_embedding_dim}")
        self.armature_class_embedding = nn.Embedding(max_armature_classes, armature_embedding_dim)
        self.armature_projection_to_latent = nn.Identity() 
        if armature_embedding_dim != latent_dim: # Corrected: Projection to latent_dim if needed for algebraic ops
            self.armature_projection_to_latent = nn.Linear(armature_embedding_dim, latent_dim)
            logger.info(f"Armature embedding will have a projection from {armature_embedding_dim} to {latent_dim} "
                        f"for use in algebraic conditioning policies.")
        self.armature_cond_dropout_prob = armature_cond_dropout_prob

        # 6. Layers for Conditioning Combination Policies
        self.text_time_weights = None
        self.text_time_concat_projection = None
        self.armature_integration_weights = None
        self.final_concat_projection = None
        self.final_algebraic_refinement_encoder = None

        # Layers for Stage 1: Text + Time combination
        if self.text_time_conditioning_policy == "weighted_mean":
            self.text_time_weights = nn.Parameter(torch.ones(2)) 
            logger.info("Text+Time policy: 'weighted_mean' with learnable weights.")
        elif self.text_time_conditioning_policy == "concat":
            self.text_time_concat_projection = nn.Linear(latent_dim * 2, latent_dim)
            logger.info(f"Text+Time policy: 'concat', projecting from {latent_dim*2} to {latent_dim}.")
        elif self.text_time_conditioning_policy not in ["add", "mean"]:
            raise ValueError(f"Unsupported text_time_conditioning_policy: {self.text_time_conditioning_policy}")
        else:
            logger.info(f"Text+Time policy: '{self.text_time_conditioning_policy}'.")

        # Layers for Stage 2: (Text+Time Result) + Armature integration
        if self.armature_integration_policy == "weighted_mean":
            self.armature_integration_weights = nn.Parameter(torch.ones(2))
            logger.info("Armature integration policy: 'weighted_mean' with learnable weights.")
        elif self.armature_integration_policy == "concat":
            # Input: latent_dim (from text+time) + armature_embedding_dim (raw armature dim)
            final_concat_input_dim = latent_dim + armature_embedding_dim 
            self.final_concat_projection = nn.Linear(final_concat_input_dim, latent_dim)
            logger.info(f"Armature integration policy: 'concat', projecting from {final_concat_input_dim} to {latent_dim}.")
        elif self.armature_integration_policy not in ["add", "mean"]:
             raise ValueError(f"Unsupported armature_integration_policy: {self.armature_integration_policy}")
        else:
            logger.info(f"Armature integration policy: '{self.armature_integration_policy}'.")
        
        if self.armature_integration_policy in ["add", "mean", "weighted_mean"] and \
           self.use_final_algebraic_refinement_encoder:
            hidden_dim = algebraic_refinement_hidden_dim if algebraic_refinement_hidden_dim is not None else latent_dim
            self.final_algebraic_refinement_encoder = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, latent_dim)
            )
            logger.info(f"Final algebraic refinement MLP enabled (hidden_dim: {hidden_dim}) "
                        f"for armature integration policy: '{self.armature_integration_policy}'.")
        
        # 7. Backbone Diffusion Model (Transformer Encoder)
        logger.info("Using Transformer Encoder backbone.")
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim, nhead=num_heads, dim_feedforward=ff_size, 
            dropout=dropout, activation=activation, batch_first=True 
        )
        self.backbone = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def _load_and_freeze_sbert(self, model_name: str) -> SentenceTransformer:
        """
        Load and freeze the Sentence-BERT model.
        :param model_name: Name of the Sentence-BERT model to load.
        :return: Loaded and frozen Sentence-BERT model.
        """
        try:
            model = SentenceTransformer(model_name)
            model.eval()
            for param in model.parameters():
                param.requires_grad = False
            return model
        except Exception as e:
            logger.error(f"Failed to load Sentence-BERT model '{model_name}': {e}")
            raise

    def _apply_dropout_mask(self, embedding: torch.Tensor, prob: float, force_mask: bool) -> torch.Tensor:
        """
        Applies dropout mask to the embedding tensor based on the given probability.
        :param embedding: Input tensor to apply dropout to.
        :param prob: Probability of dropout.
        :param force_mask: If True, returns a zero tensor of the same shape as embedding.
        """
        if force_mask:
            return torch.zeros_like(embedding)
        if self.training and prob > 0.0:
            mask = torch.bernoulli(torch.ones_like(embedding[:, :1]) * prob).to(embedding.device)
            return embedding * (1. - mask)
        return embedding

    def _combine_time_and_text_embeddings(self, 
                                          time_emb: torch.Tensor, 
                                          text_emb: torch.Tensor) -> torch.Tensor:
        """
        Combines time and text embeddings based on self.text_time_conditioning_policy.
        :param time_emb: Timestep embedding tensor of shape [batch_size, latent_dim].
        :param text_emb: Text embedding tensor of shape [batch_size, latent_dim].
        :return: Combined embedding tensor of shape [batch_size, latent_dim].
        """
        if self.text_time_conditioning_policy == 'add':
            return time_emb + text_emb
        elif self.text_time_conditioning_policy == 'mean':
            return torch.mean(torch.stack([time_emb, text_emb], dim=0), dim=0)
        elif self.text_time_conditioning_policy == 'weighted_mean':
            if self.text_time_weights is None:
                raise RuntimeError("text_time_weights not initialized for 'weighted_mean' policy.")
            tt_weights = F.softmax(self.text_time_weights, dim=0)
            return tt_weights[0] * time_emb + tt_weights[1] * text_emb
        elif self.text_time_conditioning_policy == 'concat':
            if self.text_time_concat_projection is None:
                raise RuntimeError("text_time_concat_projection not initialized for 'concat' policy.")
            return self.text_time_concat_projection(torch.cat((time_emb, text_emb), dim=-1))
        else: 
            raise ValueError(f"Unsupported text_time_conditioning_policy: {self.text_time_conditioning_policy}")

    def _integrate_armature_embedding(self, 
                                      time_text_combined_emb: torch.Tensor, 
                                      raw_armature_emb: torch.Tensor,         # [bs, armature_embedding_dim]
                                      projected_armature_emb_ld: torch.Tensor,# [bs, latent_dim]
                                      uncond_armature: bool) -> torch.Tensor:
        """
        Integrates armature embedding with combined time-text based on self.armature_integration_policy.
        :param time_text_combined_emb: Combined time-text embedding tensor of shape [batch_size, latent_dim].
        :param raw_armature_emb: Raw armature embedding tensor of shape [batch_size, armature_embedding_dim].
        :param projected_armature_emb_ld: Projected armature embedding tensor of shape [batch_size, latent_dim].
        :param uncond_armature: If True, applies dropout to the armature embedding.
        :return: Final combined embedding tensor of shape [batch_size, latent_dim].
        """
        final_combined_cond_emb: torch.Tensor
        
        if self.armature_integration_policy in ['add', 'mean', 'weighted_mean']:
            armature_emb_for_algebraic = self._apply_dropout_mask(
                projected_armature_emb_ld, # Use version projected to latent_dim
                prob=self.armature_cond_dropout_prob,
                force_mask=uncond_armature
            )
            if armature_emb_for_algebraic.shape[-1] != self.latent_dim:
                 raise ValueError(f"Armature embedding for algebraic policy must be {self.latent_dim}-dim.")

            current_combination: torch.Tensor
            if self.armature_integration_policy == 'add':
                current_combination = time_text_combined_emb + armature_emb_for_algebraic
            elif self.armature_integration_policy == 'mean':
                current_combination = torch.mean(torch.stack([time_text_combined_emb, armature_emb_for_algebraic], dim=0), dim=0)
            elif self.armature_integration_policy == 'weighted_mean':
                if self.armature_integration_weights is None:
                    raise RuntimeError("armature_integration_weights not initialized.")
                ai_weights = F.softmax(self.armature_integration_weights, dim=0)
                current_combination = ai_weights[0] * time_text_combined_emb + ai_weights[1] * armature_emb_for_algebraic
            else: # Should not be reached due to checks
                 current_combination = time_text_combined_emb 

            if self.use_final_algebraic_refinement_encoder and self.final_algebraic_refinement_encoder:
                final_combined_cond_emb = self.final_algebraic_refinement_encoder(current_combination)
            else:
                final_combined_cond_emb = current_combination

        elif self.armature_integration_policy == 'concat':
            if self.final_concat_projection is None:
                raise RuntimeError("final_concat_projection not initialized for 'concat' armature_integration_policy.")
            raw_armature_emb_with_dropout = self._apply_dropout_mask(
                raw_armature_emb, # Use raw (original armature_embedding_dim) for concat
                prob=self.armature_cond_dropout_prob,
                force_mask=uncond_armature
            )
            concatenated_final = torch.cat((time_text_combined_emb, raw_armature_emb_with_dropout), dim=-1)
            final_combined_cond_emb = self.final_concat_projection(concatenated_final)
        else: 
            raise ValueError(f"Unsupported armature_integration_policy: {self.armature_integration_policy}")
            
        return final_combined_cond_emb

    def forward(self, 
                x: torch.Tensor,
                timesteps: torch.Tensor,
                text_conditions: List[str],
                armature_class_ids: torch.Tensor,
                uncond_text: bool = False,
                uncond_armature: bool = False,
                motion_padding_mask: Optional[torch.Tensor] = None # True for PAD
               ) -> torch.Tensor:
        """
        Forward pass for the ArmatureMDM model.
        :param x: Input motion tensor of shape [batch_size, num_frames, num_motion_features].
        :param timesteps: Tensor of shape [batch_size] containing timestep indices.
        :param text_conditions: List of text conditions for each batch item.
        :param armature_class_ids: Tensor of shape [batch_size] containing armature class IDs.
        :param uncond_text: If True, applies dropout to the text embedding.
        :param uncond_armature: If True, applies dropout to the armature embedding.
        :param motion_padding_mask: Optional padding mask for the motion input (True for PAD).
        :return: Output motion tensor of shape [batch_size, num_frames, num_motion_features].
        """
        bs, num_frames, _ = x.shape
        current_device = x.device

        # 1. Timestep Embedding
        time_emb = self.time_embedder(timesteps) # [bs, latent_dim]

        # 2. Text Embedding (processed)
        sbert_embeddings = self.sbert_model.encode(text_conditions, convert_to_tensor=True, device=current_device)
        projected_text_emb = self.text_projection(sbert_embeddings)
        final_text_emb = self._apply_dropout_mask(projected_text_emb, 
                                                  prob=self.text_cond_dropout_prob, 
                                                  force_mask=uncond_text)

        # 3. Armature Class Embedding (get raw and projected_to_latent versions)
        raw_armature_emb = self.armature_class_embedding(armature_class_ids.long()) 
        projected_armature_emb_ld = self.armature_projection_to_latent(raw_armature_emb)
        
        # 4. Combine Conditions (Two Stages using helper methods)
        time_text_combined_emb = self._combine_time_and_text_embeddings(time_emb, final_text_emb)
        final_combined_cond_emb = self._integrate_armature_embedding(
            time_text_combined_emb, raw_armature_emb, projected_armature_emb_ld, uncond_armature
        )
        
        # 5. Process Motion Input
        motion_seq_emb = self.motion_input_projection(x) 
        motion_seq_emb = self.pos_encoder(motion_seq_emb)

        # 6. Condition the motion sequence
        motion_seq_conditioned = motion_seq_emb + final_combined_cond_emb.unsqueeze(1)

        # 7. Pass through Transformer Encoder Backbone
        transformer_output = self.backbone(motion_seq_conditioned, src_key_padding_mask=motion_padding_mask)
        
        # 8. Project back to motion feature space
        output_motion = self.motion_output_projection(transformer_output)

        return output_motion

    def _apply(self, fn: callable) -> 'ArmatureMDM':
        """
        Applies a function to the model and its parameters.
        This is a custom implementation to ensure that the SBERT model is moved to the correct device.
        :param fn: Function to apply to the model and its parameters.
        :return: The model itself after applying the function.
        """
        super()._apply(fn)
        if hasattr(self, 'sbert_model') and self.sbert_model is not None:
            try:
                # Attempt to determine target device from fn
                example_tensor_on_target_device = fn(torch.tensor(0, device=self.sbert_model.device if hasattr(self.sbert_model, 'device') else "cpu")) # Use SBERT's current device or CPU
                sbert_target_device = example_tensor_on_target_device.device
                if hasattr(self.sbert_model, 'to') and callable(getattr(self.sbert_model, 'to')):
                   if str(next(self.sbert_model.parameters()).device) != str(sbert_target_device): # Check if move is needed
                        self.sbert_model.to(sbert_target_device)
            except Exception as e:
                logger.debug(f"Could not move SBERT model in _apply or already on device: {e}")
        return self

    def train(self, mode: bool = True):
        """
        Sets the module in training mode. This is a custom implementation to ensure that the SBERT model is set to eval mode.
        :param mode: If True, sets the module to training mode. If False, sets it to evaluation mode.
        :return: The module itself.
        """
        super().train(mode)
        if hasattr(self, 'sbert_model') and self.sbert_model is not None:
            self.sbert_model.eval() 
        return self