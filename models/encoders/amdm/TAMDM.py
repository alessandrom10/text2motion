import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Literal, Optional

import logging
logger = logging.getLogger(__name__)

class PositionalEncoding(nn.Module):
    """
    Positional Encoding module for Transformer models.
    This module generates positional encodings for input sequences, which are added to the input embeddings.
    The encodings are based on sine and cosine functions of different frequencies.
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        """
        Initializes the PositionalEncoding module.

        :param d_model: The dimension of the embeddings (and the model).
        :param dropout: The dropout probability.
        :param max_len: The maximum length of the input sequences.
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)  # Shape: [max_len, d_model]

    def forward(self, x: torch.Tensor, batch_first_override: Optional[bool] = None) -> torch.Tensor:
        """
        Adds positional encoding to the input tensor.

        :param x: The input tensor. Shape depends on batch_first_override or typical usage.
                  If batch_first_override=True: [batch_size, seq_len, d_model]
                  If batch_first_override=False or None: [seq_len, batch_size, d_model]
        :param batch_first_override: Explicitly set if the input 'x' is batch-first.
                                     If None, assumes x is [seq_len, batch_size, d_model].
        :return: The input tensor with added positional encoding.
        """
        if batch_first_override is True:
            # x is [bs, seq_len, d_model], pe needs to be [1, seq_len, d_model]
            x = x + self.pe[:x.size(1), :].unsqueeze(0)
        else:
            # x is [seq_len, bs, d_model]
            x = x + self.pe[:x.size(0), :].unsqueeze(1)
        return self.dropout(x)


class TimestepEmbedder(nn.Module):
    """
    Embeds timestep indices into a fixed-size vector representation.
    Uses a positional encoding table and a small MLP, similar to MDM's approach.
    """
    def __init__(self, latent_dim: int, sequence_pos_encoder: PositionalEncoding):
        """
        Initializes the TimestepEmbedder.

        :param latent_dim: The dimension of the latent space and the output embedding.
        :param sequence_pos_encoder: An instance of PositionalEncoding which provides the '.pe' table.
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.sequence_pos_encoder = sequence_pos_encoder

        self.time_embed = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.SiLU(),
            nn.Linear(self.latent_dim, self.latent_dim),
        )

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Embeds a batch of timestep indices.

        :param timesteps: A 1D tensor of timestep indices, shape [batch_size].
        :return: A tensor of timestep embeddings, shape [1, batch_size, latent_dim].
        """
        # self.sequence_pos_encoder.pe is [max_len, d_model]
        emb = self.sequence_pos_encoder.pe[timesteps.long(), :].to(timesteps.device)  # [bs, latent_dim]
        emb = self.time_embed(emb)  # [bs, latent_dim]
        return emb.unsqueeze(0)  # Output: [1, bs, latent_dim] for MDM-style concatenation


class InputProcess(nn.Module):
    """
    Processes the raw motion input features into the model's latent dimension.
    Handles permutation for Transformer compatibility if not batch_first.
    """
    def __init__(self, data_rep: str, input_feats: int, latent_dim: int):
        """
        Initializes the InputProcess module.

        :param data_rep: String identifier for the data representation (e.g., 'xyz_flat').
                         Currently used for logging, can be extended for different processing.
        :param input_feats: The total number of features in the input motion vector per frame.
        :param latent_dim: The target latent dimension for the model.
        """
        super().__init__()
        self.data_rep = data_rep
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.poseEmbedding = nn.Linear(self.input_feats, self.latent_dim)
        logger.info(f"InputProcess: Initialized for '{self.data_rep}', input_feats={self.input_feats}, latent_dim={self.latent_dim}.")

    def forward(self, x_motion_data: torch.Tensor, batch_first_input: bool) -> torch.Tensor:
        """
        Processes the input motion data.

        :param x_motion_data: Input motion tensor.
                              Expected shape [batch_size, num_frames, input_feats].
        :param batch_first_input: If False, permutes to [num_frames, batch_size, input_feats]
                                   before projection, for non-batch-first Transformers.
        :return: Processed motion sequence in latent_dim.
                 Shape: [batch_size, num_frames, latent_dim] if batch_first_input is True.
                 Shape: [num_frames, batch_size, latent_dim] if batch_first_input is False.
        """
        if not batch_first_input:
            # Permute [bs, nframes, features] to [nframes, bs, features]
            x = x_motion_data.permute(1, 0, 2)
        else:
            x = x_motion_data
        
        x_embedded = self.poseEmbedding(x) # [nframes, bs, latent_dim] or [bs, nframes, latent_dim]
        return x_embedded


class OutputProcess(nn.Module):
    """
    Projects the Transformer's output from latent dimension back to the motion feature space.
    Handles permutation for compatibility.
    """
    def __init__(self, data_rep: str, output_feats: int, latent_dim: int): # Removed njoints, nfeats_per_joint as they are not used if output is flat
        """
        Initializes the OutputProcess module.

        :param data_rep: String identifier for the data representation.
        :param output_feats: The total number of features in the output motion vector per frame.
        :param latent_dim: The latent dimension of the model's internal representation.
        """
        super().__init__()
        self.data_rep = data_rep
        self.output_feats = output_feats
        self.latent_dim = latent_dim
        self.poseFinal = nn.Linear(self.latent_dim, self.output_feats)
        logger.info(f"OutputProcess: Initialized for '{self.data_rep}', output_feats={self.output_feats}, latent_dim={self.latent_dim}.")

    def forward(self, output_from_transformer: torch.Tensor, batch_first_input_from_transformer: bool) -> torch.Tensor:
        """
        Processes the output from the Transformer.

        :param output_from_transformer: Tensor from the Transformer.
                                        Shape: [batch_size, num_frames, latent_dim] if batch_first_input_from_transformer.
                                        Shape: [num_frames, batch_size, latent_dim] otherwise.
        :param batch_first_input_from_transformer: Indicates if the input 'output_from_transformer' is batch-first.
        :return: Final motion tensor. Shape: [batch_size, num_frames, output_feats].
        """
        projected_output = self.poseFinal(output_from_transformer) # Shape matches input

        if not batch_first_input_from_transformer:
            # Input was [nframes, bs, output_feats], permute to [bs, nframes, output_feats]
            final_output_motion = projected_output.permute(1, 0, 2)
        else:
            # Input was already [bs, nframes, output_feats]
            final_output_motion = projected_output
        
        return final_output_motion


class ArmatureMDM(nn.Module):
    """
    Armature-conditioned Motion Diffusion Model (ArmatureMDM).
    Integrates motion features with text, time, and armature class conditions.
    Supports Transformer Encoder, Decoder, or GRU backbones,
    and MLP or mini-Transformer for conditioning integration.

    This model is designed to handle both flat and structured motion data representations.
    It can be configured to use different conditioning integration methods,
    such as MLP-based policies or a mini-Transformer for more complex conditioning fusion.
    """
    def __init__(self,
                 # Data representation
                 data_rep: str,
                 njoints: int,
                 nfeats_per_joint: int,
                 num_motion_features: Optional[int] = None, # Total motion features if flat

                 # Core model architecture
                 latent_dim: int = 256,
                 arch: Literal['trans_enc', 'trans_dec', 'gru'] = 'trans_enc',
                 
                 # Backbone Transformer/GRU parameters (used if arch is trans_enc/trans_dec/gru)
                 num_layers: int = 8,
                 num_heads: int = 4,       # Only for Transformer
                 ff_size: int = 1024,      # FeedForward size in Transformer, or GRU hidden size if GRU layers=1
                 dropout: float = 0.1,
                 activation: str = "gelu", # For Transformer's FFN
                 batch_first_transformer: bool = False,

                 # Conditioning integration
                 conditioning_integration_mode: Literal["mlp", "transformer"] = "mlp",
                 
                 # Parameters for "mlp" conditioning_integration_mode
                 armature_integration_policy: Literal["add_refined", "concat_refined", "film"] = "add_refined",
                 armature_mlp_hidden_dims: Optional[List[int]] = None,

                 # Parameters for "transformer" conditioning_integration_mode
                 conditioning_transformer_config: Optional[Dict[str, Any]] = None,

                 # Text conditioning
                 sbert_embedding_dim: int = 768,
                 
                 # Armature conditioning
                 max_armature_classes: int = 10,
                 armature_embedding_dim: int = 64, # Raw embedding dim for armature

                 # General
                 max_seq_len_pos_enc: int = 5000,
                 text_cond_mask_prob: float = 0.1,
                 armature_cond_mask_prob: float = 0.1,
                 **kargs): # Catch-all for other params from config like clip_dim, dataset_name
        """
        Initializes the ArmatureMDM model.

        For detailed parameter descriptions, please refer to the configuration file (e.g., diffusion_config.yaml).

        :param data_rep: Data representation type (e.g., 'flat', 'structured').
        :param njoints: Number of joints in the motion data.
        :param nfeats_per_joint: Number of features per joint in the motion data.
        :param num_motion_features: Total number of motion features if using 'flat' data representation.
        :param latent_dim: Dimension of the latent space.
        :param arch: Backbone architecture
        :param num_layers: Number of layers in the Transformer or GRU backbone.
        :param num_heads: Number of attention heads in the Transformer backbone.
        :param ff_size: FeedForward size in Transformer, or GRU hidden size if GRU layers=1.
        :param dropout: Dropout rate for the Transformer or GRU backbone.
        :param activation: Activation function for the Transformer's FeedForward layers.
        :param batch_first_transformer: If True, Transformer backbone expects input in [bs, seq_len, d] format.
        :param conditioning_integration_mode: Method to integrate conditions ('mlp' or 'transformer').
        :param armature_integration_policy: Policy for integrating armature conditions ('add_refined', 'concat_refined', 'film').
        :param armature_mlp_hidden_dims: Hidden dimensions for MLP-based armature conditioning.
        :param conditioning_transformer_config: Configuration for the mini-Transformer used in conditioning integration.
        :param sbert_embedding_dim: Dimension of the SBERT text embeddings.
        :param max_armature_classes: Maximum number of armature classes.
        :param armature_embedding_dim: Dimension of the raw armature class embeddings.
        :param max_seq_len_pos_enc: Maximum sequence length for positional encoding.
        :param text_cond_mask_prob: Probability of masking text conditions during training.
        :param armature_cond_mask_prob: Probability of masking armature conditions during training.
        :param kargs: Additional parameters from the configuration file (e.g., clip_dim, dataset_name).
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.arch = arch
        self.data_rep = data_rep
        self.njoints = njoints
        self.nfeats = nfeats_per_joint
        
        if "flat" in self.data_rep:
            if num_motion_features is None:
                raise ValueError("num_motion_features must be provided for flat data_rep.")
            self.input_feats = num_motion_features
        elif "structured" in self.data_rep: # Not fully implemented in provided Input/OutputProcess
            self.input_feats = self.njoints * self.nfeats
            logger.warning(f"Data representation '{self.data_rep}' implies structured input/output. "
                           "Ensure InputProcess/OutputProcess handle this if it's not flat.")
        else: # Default to flat if not specified, using njoints * nfeats
            self.input_feats = num_motion_features if num_motion_features is not None else self.njoints * self.nfeats
            logger.info(f"Data_rep '{self.data_rep}' not explicitly 'flat' or 'structured'. "
                        f"Using input_feats: {self.input_feats} (from num_motion_features or njoints*nfeats).")


        self.text_cond_mask_prob = text_cond_mask_prob
        self.armature_cond_mask_prob = armature_cond_mask_prob
        self.batch_first_backbone = batch_first_transformer # For the main backbone
        self.conditioning_integration_mode = conditioning_integration_mode

        # --- Standard Modules ---
        self.input_process = InputProcess(self.data_rep, self.input_feats, self.latent_dim)
        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, dropout, max_len=max_seq_len_pos_enc)
        self.embed_timestep = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder)
        self.embed_text = nn.Linear(sbert_embedding_dim, self.latent_dim)
        self.armature_class_embedding = nn.Embedding(max_armature_classes, armature_embedding_dim)
        
        # --- Conditioning Integration Modules ---
        if self.conditioning_integration_mode == "transformer":
            self._setup_conditioning_transformer(armature_embedding_dim, conditioning_transformer_config, dropout, activation)
        elif self.conditioning_integration_mode == "mlp":
            self.armature_integration_policy = armature_integration_policy
            self._setup_armature_mlp_conditioning_layers(armature_embedding_dim, armature_mlp_hidden_dims, kargs)
        else:
            raise ValueError(f"Unsupported conditioning_integration_mode: {self.conditioning_integration_mode}")

        # --- Main Backbone Architecture ---
        if self.arch == 'trans_enc':
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.latent_dim, nhead=num_heads, dim_feedforward=ff_size,
                dropout=dropout, activation=activation, batch_first=self.batch_first_backbone)
            self.backbone = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            logger.info(f"Initialized TransformerEncoder backbone with batch_first={self.batch_first_backbone}")
        elif self.arch == 'trans_dec':
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=self.latent_dim, nhead=num_heads, dim_feedforward=ff_size,
                dropout=dropout, activation=activation, batch_first=self.batch_first_backbone)
            self.backbone = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
            logger.info(f"Initialized TransformerDecoder backbone with batch_first={self.batch_first_backbone}")
        elif self.arch == 'gru':
            # GRU input_size is latent_dim because motion and all conditions are projected to latent_dim first
            # If GRU is used for conditioning fusion, that's separate. This is for main backbone.
            self.backbone = nn.GRU(input_size=self.latent_dim, hidden_size=self.latent_dim,
                                   num_layers=num_layers, dropout=dropout, batch_first=self.batch_first_backbone)
            logger.info(f"Initialized GRU backbone with batch_first={self.batch_first_backbone}")
        else:
            raise ValueError(f"Unsupported arch: {self.arch}. Choose 'trans_enc', 'trans_dec', or 'gru'.")

        self.output_process = OutputProcess(self.data_rep, self.input_feats, self.latent_dim)

    def _setup_armature_mlp_conditioning_layers(self, armature_embedding_dim: int,
                                                armature_mlp_hidden_dims: Optional[List[int]],
                                                kargs: Dict[str, Any]):
        """Helper for MLP-based armature conditioning policies."""
        logger.info(f"Setting up MLP-based armature conditioning with policy: '{self.armature_integration_policy}'")
        if self.armature_integration_policy not in ["concat_refined", "film"]:
            armature_target_dim = self.latent_dim
            if armature_mlp_hidden_dims is None or not armature_mlp_hidden_dims:
                self.armature_projection_mlp = nn.Linear(armature_embedding_dim, armature_target_dim)
            else:
                layers = [nn.Linear(armature_embedding_dim, armature_mlp_hidden_dims[0]), nn.SiLU()]
                for i in range(len(armature_mlp_hidden_dims) - 1):
                    layers.extend([nn.Linear(armature_mlp_hidden_dims[i], armature_mlp_hidden_dims[i+1]), nn.SiLU()])
                layers.append(nn.Linear(armature_mlp_hidden_dims[-1], armature_target_dim))
                self.armature_projection_mlp = nn.Sequential(*layers)

        if self.armature_integration_policy == "add_refined":
            self.final_cond_refinement_mlp = nn.Sequential(
                nn.Linear(self.latent_dim, self.latent_dim), nn.SiLU(),
                nn.Linear(self.latent_dim, self.latent_dim)
            )
        elif self.armature_integration_policy == "concat_refined":
            concat_input_dim = self.latent_dim + armature_embedding_dim
            self.armature_concat_projection_mlp = nn.Sequential(
                nn.Linear(concat_input_dim, self.latent_dim), nn.SiLU(),
                nn.Linear(self.latent_dim, self.latent_dim)
            )
        elif self.armature_integration_policy == "film":
            self.armature_to_film_params = nn.Linear(armature_embedding_dim, 2 * self.latent_dim)
            logger.warning("FiLM policy selected for armature, but application within backbone layers is NOT YET IMPLEMENTED in this snippet.")
        else: # Default or error if not one of the above specific MLP policies
            if self.armature_integration_policy != "add_refined": # add_refined is a common default
                 logger.warning(f"MLP conditioning policy '{self.armature_integration_policy}' not fully defined, ensure layers are set up if custom.")


    def _setup_conditioning_transformer(self, armature_embedding_dim: int,
                                       cond_transformer_config: Optional[Dict[str, Any]],
                                       shared_dropout: float, shared_activation: str):
        """Helper to initialize the mini-Transformer for conditioning fusion."""
        logger.info("Setting up mini-Transformer for conditioning integration.")
        if cond_transformer_config is None:
            cond_transformer_config = {} # Use defaults if no specific config

        self.project_armature_for_cond_transformer = nn.Linear(armature_embedding_dim, self.latent_dim)
        
        cond_num_layers = cond_transformer_config.get('num_layers', max(1, self.model.num_layers // 4 if hasattr(self.model, 'num_layers') else 2))
        cond_num_heads = cond_transformer_config.get('num_heads', max(1, self.model.num_heads // 2 if hasattr(self.model, 'num_heads') else 2))
        cond_ff_size_factor = cond_transformer_config.get('ff_size_factor', 2)
        cond_ff_size = self.latent_dim * cond_ff_size_factor
        cond_dropout = cond_transformer_config.get('dropout', shared_dropout)
        cond_activation = cond_transformer_config.get('activation', shared_activation)
        self.cond_transformer_batch_first = cond_transformer_config.get('batch_first', self.batch_first_backbone) # Match main backbone or be configurable
        self.cond_transformer_aggregation = cond_transformer_config.get('aggregation_method', 'mean')
        self.cond_transformer_use_pe = cond_transformer_config.get('use_positional_encoding', False)

        if self.cond_transformer_use_pe:
            self.conditioning_seq_pos_encoder = PositionalEncoding(self.latent_dim, cond_dropout, max_len=10) # Max 10 condition tokens

        conditioning_encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.latent_dim,
            nhead=cond_num_heads,
            dim_feedforward=cond_ff_size,
            dropout=cond_dropout,
            activation=cond_activation,
            batch_first=self.cond_transformer_batch_first
        )
        self.conditioning_transformer_encoder = nn.TransformerEncoder(
            conditioning_encoder_layer,
            num_layers=cond_num_layers
        )
        logger.info(f"Conditioning Transformer: layers={cond_num_layers}, heads={cond_num_heads}, ff_size={cond_ff_size}, batch_first={self.cond_transformer_batch_first}, aggregation='{self.cond_transformer_aggregation}'")


    def _mask_cond(self, cond_embedding: torch.Tensor, prob: Optional[float], force_mask: bool) -> torch.Tensor:
        """
        Applies CFG mask to a condition embedding.

        :param cond_embedding: The condition embedding tensor. Expected shape [1, bs, d] or [bs, 1, d] if pre-shaped.
        :param prob: Masking probability during training if force_mask is False.
        :param force_mask: If True, always masks the condition.
        :return: Masked condition embedding.
        """
        if force_mask:
            return torch.zeros_like(cond_embedding)
        
        # Check if prob is None or 0, or not in training mode
        if not self.training or prob is None or prob <= 0.0:
            return cond_embedding

        # Determine batch size dimension
        # Assuming cond_embedding can be [1, bs, d] (seq_first for cond token) or [bs, 1, d] (batch_first for cond token)
        bs_dim = 1 if cond_embedding.shape[0] > 1 and cond_embedding.shape[1] == 1 else 0 # Heuristic
        if cond_embedding.shape[0] == 1 and cond_embedding.shape[1] > 1: # [1, bs, d]
            bs = cond_embedding.size(1)
            mask_shape = (1, bs, 1)
        elif cond_embedding.shape[1] == 1 and cond_embedding.shape[0] > 1: # [bs, 1, d]
            bs = cond_embedding.size(0)
            mask_shape = (bs, 1, 1)
        else: # Fallback for ambiguous or single element batch, assume batch is first dim if ambiguous
            bs = cond_embedding.size(0)
            mask_shape = (bs, 1, 1) if len(cond_embedding.shape) > 2 else (bs,1) # adapt if cond_embedding is [bs,d]
            while len(mask_shape) < len(cond_embedding.shape): mask_shape += (1,)


        mask = torch.bernoulli(torch.ones(mask_shape, device=cond_embedding.device) * prob)
        return cond_embedding * (1. - mask)


    def _integrate_conditions_mlp(self, text_emb_masked, time_emb, raw_armature_emb, y_cond_flags):
        """Integrates conditions using the MLP-based policy."""
        text_time_emb = text_emb_masked + time_emb # [1, bs, latent_dim]

        if self.armature_integration_policy == "add_refined":
            # raw_armature_emb is [bs, armature_embedding_dim]
            projected_arm_emb = self.armature_projection_mlp(raw_armature_emb).unsqueeze(0) # -> [1, bs, latent_dim]
            arm_emb_masked = self._mask_cond(projected_arm_emb, self.armature_cond_mask_prob, 
                                              y_cond_flags.get('uncond_armature', y_cond_flags.get('uncond', False)))
            combined_cond = text_time_emb + arm_emb_masked
            emb = self.final_cond_refinement_mlp(combined_cond.squeeze(0)).unsqueeze(0)
        elif self.armature_integration_policy == "concat_refined":
            # raw_armature_emb is [bs, armature_embedding_dim]
            # Make it [1, bs, armature_embedding_dim] for _mask_cond then squeeze back
            raw_arm_emb_unsqueezed = raw_armature_emb.unsqueeze(0)
            raw_arm_emb_masked = self._mask_cond(raw_arm_emb_unsqueezed, self.armature_cond_mask_prob, 
                                                 y_cond_flags.get('uncond_armature', y_cond_flags.get('uncond', False)))
            # text_time_emb: [1,bs,lat], raw_arm_emb_masked: [1,bs,arm_emb_dim]
            emb_to_project = torch.cat((text_time_emb.squeeze(0), raw_arm_emb_masked.squeeze(0)), dim=-1) # [bs, lat+arm_emb_dim]
            emb = self.armature_concat_projection_mlp(emb_to_project).unsqueeze(0)
        elif self.armature_integration_policy == "film":
            film_params_raw = self.armature_to_film_params(raw_armature_emb) # [bs, 2 * latent_dim]
            y_cond_flags['film_params'] = self._mask_cond(film_params_raw.unsqueeze(0), self.armature_cond_mask_prob, 
                                                     y_cond_flags.get('uncond_armature', y_cond_flags.get('uncond', False)))
            emb = text_time_emb # FiLM params are passed via y_cond_flags to backbone
        else:
            raise ValueError(f"Unsupported MLP armature_integration_policy: {self.armature_integration_policy}")
        return emb


    def _integrate_conditions_transformer(self, text_emb_masked, time_emb, raw_armature_emb, y_cond_flags):
        """Integrates conditions using the mini-Transformer."""
        # Project raw armature embedding to latent_dim
        # raw_armature_emb: [bs, armature_embedding_dim]
        projected_arm_emb_for_tf = self.project_armature_for_cond_transformer(raw_armature_emb) # [bs, latent_dim]

        # Prepare for sequence: ensure [1, bs, d] or [bs, 1, d] depending on mini-transformer's batch_first
        # text_emb_masked and time_emb are already [1, bs, latent_dim]
        if self.cond_transformer_batch_first:
            current_text_token = text_emb_masked.permute(1,0,2)      # [bs, 1, latent_dim]
            current_time_token = time_emb.permute(1,0,2)          # [bs, 1, latent_dim]
            current_arm_token_unmasked = projected_arm_emb_for_tf.unsqueeze(1) # [bs, 1, latent_dim]
            dim_to_cat = 1 # Sequence dimension
        else: # Sequence first for conditioning transformer
            current_text_token = text_emb_masked      # [1, bs, latent_dim]
            current_time_token = time_emb          # [1, bs, latent_dim]
            current_arm_token_unmasked = projected_arm_emb_for_tf.unsqueeze(0) # [1, bs, latent_dim]
            dim_to_cat = 0 # Sequence dimension
        
        current_arm_token = self._mask_cond(current_arm_token_unmasked, self.armature_cond_mask_prob,
                                             y_cond_flags.get('uncond_armature', y_cond_flags.get('uncond', False)))

        # Form sequence: [text_emb, time_emb, armature_emb]
        conditioning_sequence = torch.cat(
            (current_text_token, current_time_token, current_arm_token),
            dim=dim_to_cat
        ) # Shape: [bs, 3, d] if batch_first, or [3, bs, d] if seq_first

        if self.cond_transformer_use_pe:
            conditioning_sequence = self.conditioning_seq_pos_encoder(conditioning_sequence, 
                                                                      batch_first_override=self.cond_transformer_batch_first)
        
        # No padding mask needed as it's a fixed short sequence (3 tokens)
        processed_conditioning_tokens = self.conditioning_transformer_encoder(conditioning_sequence, src_key_padding_mask=None)
        # Output shape: same as input conditioning_sequence

        # Aggregate to a single embedding
        if self.cond_transformer_aggregation == "mean":
            if self.cond_transformer_batch_first:
                aggregated_emb = torch.mean(processed_conditioning_tokens, dim=1, keepdim=True) # [bs, 1, d]
            else: # seq_first
                aggregated_emb = torch.mean(processed_conditioning_tokens, dim=0, keepdim=True) # [1, bs, d]
        elif self.cond_transformer_aggregation == "sum":
            if self.cond_transformer_batch_first:
                aggregated_emb = torch.sum(processed_conditioning_tokens, dim=1, keepdim=True) # [bs, 1, d]
            else: # seq_first
                aggregated_emb = torch.sum(processed_conditioning_tokens, dim=0, keepdim=True) # [1, bs, d]
        elif self.cond_transformer_aggregation == "wheighted_mean":
            # This would require a specific weighting mechanism, not implemented here
            raise NotImplementedError("Weighted mean aggregation for conditioning_transformer not yet implemented.")
        elif self.cond_transformer_aggregation == "cls_token":
            # This would require adding a CLS token to conditioning_sequence and taking its output
            raise NotImplementedError("CLS token aggregation for conditioning_transformer not yet implemented.")
        elif isinstance(self.cond_transformer_aggregation, str) and self.cond_transformer_aggregation.isdigit():
            token_idx = int(self.cond_transformer_aggregation)
            if self.cond_transformer_batch_first:
                aggregated_emb = processed_conditioning_tokens[:, token_idx:token_idx+1, :] # [bs, 1, d]
            else: # seq_first
                aggregated_emb = processed_conditioning_tokens[token_idx:token_idx+1, :, :] # [1, bs, d]
        else:
            raise ValueError(f"Unknown conditioning_transformer_aggregation: {self.cond_transformer_aggregation}")
        
        # Ensure final emb is [1, bs, latent_dim] for consistency with MDM style trans_enc input
        if self.batch_first_backbone is False and aggregated_emb.shape[0] != 1 : # if backbone is seq_first
             emb = aggregated_emb.permute(1,0,2) # [bs, 1, d] -> [1, bs, d]
        elif self.batch_first_backbone is True and aggregated_emb.shape[1] != 1: # if backbone is batch_first
             emb = aggregated_emb.permute(1,0,2) # This case needs review, usually emb is [1,bs,d] then adapted
             logger.warning("Conditioning Transformer output aggregation for batch_first_backbone needs review.")
        else:
            emb = aggregated_emb

        return emb


    def forward(self, x: torch.Tensor, timesteps: torch.Tensor, y: Dict[str, Any]) -> torch.Tensor:
        """
        Forward pass of the ArmatureMDM model.

        :param x: Input motion tensor, shape [bs, nframes, input_feats] if batch_first_backbone is True,
                     or [nframes, bs, input_feats] if False.
        :param timesteps: Tensor of timestep indices, shape [bs].
        :param y: Dictionary of conditions, including:
            - 'text_embeddings_batch': SBERT embeddings, shape [bs, sbert_embedding_dim].
            - 'armature_class_ids': Tensor of armature class IDs, shape [bs].
        :return: Output motion tensor, shape [bs, nframes, output_feats] if batch_first_backbone is True,
                 or [nframes, bs, output_feats] if False.
        """
        bs = x.shape[0]
        device = x.device

        # --- Prepare individual condition embeddings ---
        time_emb = self.embed_timestep(timesteps)  # [1, bs, latent_dim]

        sbert_embeds = y['text_embeddings_batch'].to(device)
        projected_text_emb = self.embed_text(sbert_embeds).unsqueeze(0) # [1, bs, latent_dim]
        text_emb_masked = self._mask_cond(projected_text_emb, self.text_cond_mask_prob,
                                          y.get('uncond', y.get('uncond_text', False)))

        armature_ids = y['armature_class_ids'].long().to(device)
        raw_armature_emb = self.armature_class_embedding(armature_ids) # [bs, armature_embedding_dim]
        
        # --- Integrate conditions based on chosen mode ---
        if self.conditioning_integration_mode == "transformer":
            emb = self._integrate_conditions_transformer(text_emb_masked, time_emb, raw_armature_emb, y)
        elif self.conditioning_integration_mode == "mlp":
            emb = self._integrate_conditions_mlp(text_emb_masked, time_emb, raw_armature_emb, y)
        else: # Should have been caught in __init__
            raise ValueError(f"Invalid conditioning_integration_mode: {self.conditioning_integration_mode}")

        # --- Process motion and pass through backbone ---
        processed_motion_seq = self.input_process(x, batch_first_input=self.batch_first_backbone)
        # Shape: [bs, nframes, d] if batch_first_backbone else [nframes, bs, d]

        frames_mask_for_padding = None # True for PADDED positions
        if 'mask' in y and y['mask'] is not None: # y['mask'] is [bs, 1, 1, nframes_x], True if VALID
            valid_motion_frames = y['mask'].squeeze(1).squeeze(1) # [bs, nframes_x]
            frames_mask_for_padding = ~valid_motion_frames        # [bs, nframes_x]

        # --- Backbone Forward Pass ---
        if self.arch == 'trans_enc':
            if self.batch_first_backbone:
                # emb is [1, bs, d] -> needs to be [bs, 1, d] for concat
                xseq = torch.cat((emb.permute(1,0,2), processed_motion_seq), dim=1) # [bs, nframes+1, d]
                if frames_mask_for_padding is not None:
                    cond_pad = torch.zeros(bs, 1, dtype=torch.bool, device=device) # Condition token is never padded
                    transformer_padding_mask = torch.cat((cond_pad, frames_mask_for_padding), dim=1)
                else:
                    transformer_padding_mask = None
            else: # seq_first backbone
                xseq = torch.cat((emb, processed_motion_seq), dim=0) # [nframes+1, bs, d]
                if frames_mask_for_padding is not None:
                    cond_pad = torch.zeros(bs, 1, dtype=torch.bool, device=device)
                    # For src_key_padding_mask [N, S], if batch_first=False, S is batch_size. This is [bs, nframes+1]
                    transformer_padding_mask = torch.cat((cond_pad, frames_mask_for_padding), dim=1)
                else:
                    transformer_padding_mask = None
            
            xseq = self.sequence_pos_encoder(xseq, batch_first_override=self.batch_first_backbone)
            # TransformerEncoder expects src_key_padding_mask as [N, S] if batch_first=False, or [S, N] if batch_first=True
            # Our transformer_padding_mask is currently [bs, nframes+1].
            # If batch_first_backbone is False, nn.TransformerEncoder wants [N,S] meaning [bs, seq_len+1]
            # If batch_first_backbone is True, nn.TransformerEncoder wants [N,S] meaning [bs, seq_len+1] (This seems to be the PyTorch convention for padding mask)

            transformer_output = self.backbone(xseq, src_key_padding_mask=transformer_padding_mask)
            
            if self.batch_first_backbone:
                output_latent_seq = transformer_output[:, 1:]
            else:
                output_latent_seq = transformer_output[1:, :, :]

        elif self.arch == 'trans_dec':
            tgt_seq = self.sequence_pos_encoder(processed_motion_seq, batch_first_override=self.batch_first_backbone)
            
            # memory is 'emb'. If backbone is batch_first=False, memory should be [S_mem, N, E]. emb is [1, N, E]
            # If backbone is batch_first=True, memory should be [N, S_mem, E]. emb.permute(1,0,2) is [N, 1, E]
            memory_for_decoder = emb if not self.batch_first_backbone else emb.permute(1,0,2)
            
            memory_key_padding_mask = None # if text was from BERT and had its own mask (emb has fixed length 1 or 3 if cond_transformer)
            # If emb comes from conditioning_transformer and is e.g. [bs, 3, d], then memory_for_decoder is [bs,3,d]
            # memory_key_padding_mask would be [bs, 3] (all False for no padding)

            output_latent_seq = self.backbone(tgt=tgt_seq, memory=memory_for_decoder,
                                              tgt_key_padding_mask=frames_mask_for_padding,
                                              memory_key_padding_mask=memory_key_padding_mask)
        elif self.arch == 'gru':
            # For GRU, input typically needs to be [seq_len, bs, input_size] or [bs, seq_len, input_size]
            # The 'emb' needs to be fed at each step or as initial hidden state.
            # MDM's GRU implementation concatenates a repeated 'emb' to each frame's features before InputProcess.
            # Let's assume for now our InputProcess gives [seq_len, bs, latent_dim] or [bs, seq_len, latent_dim]
            # And we use 'emb' (e.g., mean over its seq dim if it's a seq) as initial hidden state.
            if emb.shape[0 if not self.batch_first_backbone else 1] > 1: # If emb is a sequence from cond_transformer
                # Aggregate emb to be [num_gru_layers, bs, latent_dim] for h0
                h0_emb = torch.mean(emb, dim=0 if not self.batch_first_backbone else 1) # -> [bs, latent_dim] or [1, bs, latent_dim]
            else:
                h0_emb = emb.squeeze(0 if not self.batch_first_backbone else 1) # -> [bs, latent_dim]

            # Ensure h0_emb is [num_layers, bs, latent_dim]
            h0_emb = h0_emb.expand(self.backbone.num_layers, -1, -1).contiguous() if h0_emb.ndim == 2 else h0_emb.permute(1,0,2).expand(-1, self.backbone.num_layers, -1).permute(1,0,2).contiguous()
            if h0_emb.shape[0] != self.backbone.num_layers or h0_emb.shape[2] != self.latent_dim :
                # Fallback if expansion fails due to unexpected emb dim, needs proper projection
                 h0_emb = torch.zeros(self.backbone.num_layers, bs, self.latent_dim, device=device)


            x_for_gru = self.sequence_pos_encoder(processed_motion_seq, batch_first_override=self.batch_first_backbone)
            output_latent_seq, _ = self.backbone(x_for_gru, h0_emb)
        else:
            raise ValueError(f"Unsupported backbone architecture: {self.arch}")

        output_motion = self.output_process(output_latent_seq, batch_first_output_from_backbone=self.batch_first_backbone)
        return output_motion