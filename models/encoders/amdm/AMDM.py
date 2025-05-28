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
    Armature-conditioned Motion Diffusion Model (ArmatureMDM), aligned with MDM and DiP concepts.
    This model integrates motion features with text, time, and armature class conditions
    using a Transformer backbone (Encoder for MDM-style, Decoder for DiP-style).

    Supports multiple policies for integrating armature conditioning.
    """
    def __init__(self,
                 data_rep: str,
                 njoints: int,
                 nfeats_per_joint: int,
                 num_motion_features: Optional[int] = None,
                 latent_dim: int = 256,
                 ff_size: int = 1024,
                 num_layers: int = 8,
                 num_heads: int = 4,
                 dropout: float = 0.1,
                 activation: str = "gelu",
                 arch: Literal['trans_enc', 'trans_dec'] = 'trans_enc', # For MDM or DiP style
                 sbert_embedding_dim: int = 768,
                 max_armature_classes: int = 10,
                 armature_embedding_dim: int = 64,
                 armature_integration_policy: Literal["add_refined", "concat_refined", "film", "cross_attention"] = "add_refined",
                 armature_mlp_hidden_dims: Optional[List[int]] = None,
                 max_seq_len_pos_enc: int = 5000,
                 text_cond_mask_prob: float = 0.1,
                 armature_cond_mask_prob: float = 0.1,
                 batch_first_transformer: bool = False, # MDM default is False, DiP (decoder) might use True.
                 **kargs):
        """
        Initializes the ArmatureMDM model.

        :param data_rep: Data representation type (e.g., 'xyz_flat').
        :param njoints: Number of joints.
        :param nfeats_per_joint: Number of features per joint (e.g., 3 for XYZ).
        :param num_motion_features: Total number of motion features per frame. If None and data_rep is
                                    'structured', it's njoints * nfeats_per_joint. Must be provided for 'flat'.
        :param latent_dim: Latent dimension of the model.
        :param ff_size: Feed-forward size in Transformer layers.
        :param num_layers: Number of Transformer layers.
        :param num_heads: Number of attention heads.
        :param dropout: Dropout rate.
        :param activation: Activation function for Transformer.
        :param arch: Architecture type: 'trans_enc' (MDM-like) or 'trans_dec' (DiP-like).
        :param sbert_embedding_dim: Dimension of precomputed SBERT text embeddings.
        :param max_armature_classes: Maximum number of armature classes for nn.Embedding.
        :param armature_embedding_dim: Raw dimension of armature class nn.Embedding.
        :param armature_integration_policy: Policy to integrate armature conditioning.
                                            "add_refined": Sums projected armature with (text+time), then refines with MLP.
                                            "concat_refined": Concatenates raw armature with (text+time), then projects and refines.
                                            "film": Armature conditions FiLM layers (TODO).
                                            "cross_attention": Armature embedding used as key/value in a cross-attention layer (TODO).
        :param armature_mlp_hidden_dims: Hidden dimensions for MLP projecting armature embedding if policy involves MLP.
        :param max_seq_len_pos_enc: Max sequence length for PositionalEncoding.
        :param text_cond_mask_prob: Masking probability for text condition (CFG).
        :param armature_cond_mask_prob: Masking probability for armature condition (CFG).
        :param batch_first_transformer: If True, Transformer layers expect batch dimension first.
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
        elif "structured" in self.data_rep:
            self.input_feats = self.njoints * self.nfeats
        else:
            self.input_feats = num_motion_features if num_motion_features is not None else self.njoints * self.nfeats
            logger.warning(f"data_rep '{self.data_rep}' not explicitly 'flat' or 'structured'. "
                           f"Assuming input_feats is {self.input_feats}.")


        self.text_cond_mask_prob = text_cond_mask_prob
        self.armature_cond_mask_prob = armature_cond_mask_prob
        self.batch_first_transformer = batch_first_transformer
        self.armature_integration_policy = armature_integration_policy

        self.input_process = InputProcess(self.data_rep, self.input_feats, self.latent_dim)
        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, dropout, max_len=max_seq_len_pos_enc)
        self.embed_timestep = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder)
        self.embed_text = nn.Linear(sbert_embedding_dim, self.latent_dim)

        self.armature_class_embedding = nn.Embedding(max_armature_classes, armature_embedding_dim)
        
        # Armature conditioning MLP / projection layers based on policy
        self._setup_armature_conditioning_layers(armature_embedding_dim, armature_mlp_hidden_dims, kargs)

        # Transformer Backbone
        if self.arch == 'trans_enc':
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.latent_dim, nhead=num_heads, dim_feedforward=ff_size,
                dropout=dropout, activation=activation, batch_first=self.batch_first_transformer)
            self.seqTransEncoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            logger.info(f"Initialized TransformerEncoder (MDM-style) with batch_first={self.batch_first_transformer}")
        elif self.arch == 'trans_dec':
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=self.latent_dim, nhead=num_heads, dim_feedforward=ff_size,
                dropout=dropout, activation=activation, batch_first=self.batch_first_transformer)
            self.seqTransDecoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
            logger.info(f"Initialized TransformerDecoder (DiP-style) with batch_first={self.batch_first_transformer}")
            if self.armature_integration_policy == "cross_attention": # Specific for decoder
                self.armature_cross_attention = nn.MultiheadAttention(embed_dim=self.latent_dim, num_heads=num_heads, dropout=dropout, batch_first=self.batch_first_transformer)
                logger.info("Added separate cross-attention layer for armature conditioning with TransformerDecoder.")
        else:
            raise ValueError(f"Unsupported arch: {self.arch}. Choose 'trans_enc' or 'trans_dec'.")

        self.output_process = OutputProcess(self.data_rep, self.input_feats, self.latent_dim)


    def _setup_armature_conditioning_layers(self, armature_embedding_dim: int,
                                           armature_mlp_hidden_dims: Optional[List[int]],
                                           kargs: Dict[str, Any]):
        """Helper to initialize layers for different armature integration policies."""
        # Common projection for armature if it's not directly used by concat or as FiLM parameters
        if self.armature_integration_policy not in ["concat_refined", "film"]: # film might use raw or different projection
            armature_target_dim = self.latent_dim
            if armature_mlp_hidden_dims is None or not armature_mlp_hidden_dims:
                self.armature_projection_mlp = nn.Linear(armature_embedding_dim, armature_target_dim)
            else:
                layers = [nn.Linear(armature_embedding_dim, armature_mlp_hidden_dims[0]), nn.SiLU()]
                for i in range(len(armature_mlp_hidden_dims) - 1):
                    layers.extend([nn.Linear(armature_mlp_hidden_dims[i], armature_mlp_hidden_dims[i+1]), nn.SiLU()])
                layers.append(nn.Linear(armature_mlp_hidden_dims[-1], armature_target_dim))
                self.armature_projection_mlp = nn.Sequential(*layers)
            logger.info(f"Armature embedding (raw {armature_embedding_dim}D) -> MLP for policy '{self.armature_integration_policy}' projects to {armature_target_dim}D.")

        if self.armature_integration_policy == "add_refined":
            self.final_cond_refinement_mlp = nn.Sequential(
                nn.Linear(self.latent_dim, self.latent_dim), nn.SiLU(),
                nn.Linear(self.latent_dim, self.latent_dim)
            )
            logger.info("Using 'add_refined' for armature: Summing (text+time) and projected_armature, then refining with MLP.")
        elif self.armature_integration_policy == "concat_refined":
            concat_input_dim = self.latent_dim + armature_embedding_dim # text_time_emb + raw_armature_emb
            self.armature_concat_projection_mlp = nn.Sequential(
                nn.Linear(concat_input_dim, self.latent_dim), nn.SiLU(), # Project to latent
                nn.Linear(self.latent_dim, self.latent_dim) # Optional refinement layer
            )
            logger.info(f"Using 'concat_refined' for armature: Concatenating (text+time) and raw_armature, then MLP from {concat_input_dim}D to {self.latent_dim}D.")
        elif self.armature_integration_policy == "film":
            # FiLM requires generating scale (gamma) and shift (beta) parameters
            # These would typically be applied per layer in the main backbone.
            # This example projects armature_embedding_dim to 2 * latent_dim (for gamma and beta)
            # Actual application of FiLM params needs to be done within the Transformer layers.
            self.armature_to_film_params = nn.Linear(armature_embedding_dim, 2 * self.latent_dim)
            logger.info(f"Using 'film' for armature: Projecting raw armature to generate FiLM parameters (gamma, beta) of size {self.latent_dim}D each. "
                        "NOTE: FiLM layer application in Transformer backbone is NOT implemented in this snippet.")
            # TODO: Modify TransformerEncoder/Decoder layers to accept and apply FiLM params.
        elif self.armature_integration_policy == "cross_attention":
            if self.arch != 'trans_dec':
                logger.warning("'cross_attention' for armature is primarily designed for 'trans_dec' (DiP-style) arch. "
                               "Behavior with 'trans_enc' might require custom handling.")
            # For 'trans_dec', a separate cross-attention layer for armature might be added if not using main memory for it.
            # If armature is to be part of the main memory, its embedding needs to be [seq_len_cond, bs, dim]
            # This is handled by self.armature_cross_attention if arch is trans_dec.
            pass # Handled in main __init__ for trans_dec
        else:
            raise ValueError(f"Unsupported armature_integration_policy: {self.armature_integration_policy}")


    def _mask_cond(self, cond_embedding: torch.Tensor, prob: Optional[float], force_mask: bool) -> torch.Tensor:
        """Applies CFG mask. cond_embedding: [1, bs, d]"""
        if force_mask:
            return torch.zeros_like(cond_embedding)
        if self.training and prob is not None and prob > 0.0: # Check prob is not None
            mask = torch.bernoulli(torch.ones((1, cond_embedding.size(1), 1), device=cond_embedding.device) * prob)
            return cond_embedding * (1. - mask)
        return cond_embedding

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor, y: Dict[str, Any]) -> torch.Tensor:
        """
        Forward pass for the ArmatureMDM model.

        :param x: Input motion tensor (x_t). Expected shape depends on data_rep and model's
                  batch_first_transformer setting. If batch_first_transformer=False,
                  InputProcess expects [bs, nframes, features_flat] and converts to [nframes, bs, ...].
                  If batch_first_transformer=True, InputProcess expects [bs, nframes, features_flat]
                  and keeps it mostly batch-first.
        :param timesteps: Timestep indices, shape [batch_size].
        :param y: Dictionary of conditions, including:
                  'text_embeddings_batch': Precomputed SBERT embeddings [bs, sbert_dim].
                  'armature_class_ids': Armature IDs [bs].
                  'mask': Optional padding mask for x (motion sequence part),
                          shape [bs, 1, 1, nframes_x], True for valid data.
                  'uncond', 'uncond_text', 'uncond_armature': CFG flags.
                  'text' (for DiP 'trans_dec'): Raw text if BERT encoder used in model.
                  'text_mask' (for DiP 'trans_dec'): Mask for raw text if BERT encoder used.
        :return: Predicted motion tensor (x_0), shape [bs, nframes, num_motion_features].
        """
        bs = x.shape[0]
        device = x.device

        time_emb = self.embed_timestep(timesteps)  # Shape: [1, bs, latent_dim]

        sbert_embeds = y['text_embeddings_batch'].to(device)
        projected_text_emb = self.embed_text(sbert_embeds).unsqueeze(0) # [1, bs, latent_dim]
        text_emb_masked = self._mask_cond(projected_text_emb, self.text_cond_mask_prob,
                                          y.get('uncond', y.get('uncond_text', False)))

        armature_ids = y['armature_class_ids'].long().to(device)
        raw_armature_emb = self.armature_class_embedding(armature_ids) # [bs, armature_embedding_dim]
        
        # Armature conditioning based on policy
        # text_time_emb is [1, bs, latent_dim]
        text_time_emb = text_emb_masked + time_emb 

        if self.armature_integration_policy == "add_refined":
            projected_arm_emb_for_sum = self.armature_projection_mlp(raw_armature_emb).unsqueeze(0) # [1,bs,latent_dim]
            arm_emb_masked = self._mask_cond(projected_arm_emb_for_sum, self.armature_cond_mask_prob, y.get('uncond_armature', y.get('uncond', False)))
            combined_cond = text_time_emb + arm_emb_masked
            emb = self.final_cond_refinement_mlp(combined_cond.squeeze(0)).unsqueeze(0) # [1,bs,latent_dim]
        elif self.armature_integration_policy == "concat_refined":
            raw_arm_emb_masked = self._mask_cond(raw_armature_emb.unsqueeze(0), self.armature_cond_mask_prob, y.get('uncond_armature', y.get('uncond', False)))
            # text_time_emb: [1,bs,lat], raw_arm_emb_masked: [1,bs,arm_emb_dim]
            # Squeeze batch for cat, then unsqueeze for model processing style
            emb_to_project = torch.cat((text_time_emb.squeeze(0), raw_arm_emb_masked.squeeze(0)), dim=-1) # [bs, lat+arm_emb_dim]
            emb = self.armature_concat_projection_mlp(emb_to_project).unsqueeze(0) # [1,bs,latent_dim]
        elif self.armature_integration_policy == "film":
            # FiLM params are generated but not applied here; Transformer layers need modification
            film_params = self.armature_to_film_params(raw_armature_emb) # [bs, 2 * latent_dim]
            # TODO: Pass film_params to Transformer layers. For now, 'emb' is just text+time.
            emb = text_time_emb 
            # Store film_params in y if Transformer expects it there
            y['film_params'] = self._mask_cond(film_params.unsqueeze(0), self.armature_cond_mask_prob, y.get('uncond_armature', y.get('uncond',False)))

        elif self.armature_integration_policy == "cross_attention":
            # For cross-attention, text+time is the query. Armature embedding is key/value.
            # If arch is trans_dec, this can be handled by the decoder's cross-attention.
            emb = text_time_emb # This will be the 'memory' for the decoder if text is primary
            # Armature embedding will be prepared as a separate context for its own cross-attention
            # or merged into the main memory depending on DiP's specific multi-condition strategy.
            # Let's assume projected_arm_emb is used if a dedicated armature cross-attn layer exists.
            projected_arm_emb_for_attn = self.armature_projection_mlp(raw_armature_emb) # [bs, latent_dim]
            y['armature_context_for_cross_attn'] = self._mask_cond(projected_arm_emb_for_attn.unsqueeze(0), # [1,bs,latent_dim]
                                                                  self.armature_cond_mask_prob, 
                                                                  y.get('uncond_armature', y.get('uncond', False)))
        else:
            raise ValueError(f"Unsupported armature_integration_policy: {self.armature_integration_policy}")


        # Process motion input x (x_t)
        # input_process expects [bs, nframes, features_flat] and returns based on batch_first_transformer
        processed_motion_seq = self.input_process(x, batch_first_input=self.batch_first_transformer)
        # Shape: [bs, nframes, d] if batch_first_transformer else [nframes, bs, d]

        # Padding mask for Transformer's src_key_padding_mask or memory_key_padding_mask
        # Expected by PyTorch: [bs, key_sequence_length], True for padded positions.
        frames_mask_for_padding = None
        if 'mask' in y and y['mask'] is not None: # y['mask'] is [bs, 1, 1, nframes_x], True if valid
            valid_motion_frames = y['mask'].squeeze(1).squeeze(1) # [bs, nframes_x]
            frames_mask_for_padding = ~valid_motion_frames        # [bs, nframes_x], True if PADDED

        # Transformer Backbone
        if self.arch == 'trans_enc':
            # MDM style: Prepend condition embedding to motion sequence
            if self.batch_first_transformer:
                xseq = torch.cat((emb.permute(1,0,2), processed_motion_seq), dim=1) # [bs, nframes+1, d]
                if frames_mask_for_padding is not None:
                    cond_pad = torch.zeros(bs, 1, dtype=torch.bool, device=device)
                    transformer_padding_mask = torch.cat((cond_pad, frames_mask_for_padding), dim=1) # [bs, nframes+1]
                else:
                    transformer_padding_mask = None
            else: # Not batch_first
                xseq = torch.cat((emb, processed_motion_seq), dim=0) # [nframes+1, bs, d]
                if frames_mask_for_padding is not None: # This mask is [bs, nframes], Transformer expects [N,S] or [S,N] for key_padding
                    cond_pad = torch.zeros(bs, 1, dtype=torch.bool, device=device)
                    transformer_padding_mask = torch.cat((cond_pad, frames_mask_for_padding), dim=1) #[bs, nframes+1]
                else:
                    transformer_padding_mask = None

            xseq = self.sequence_pos_encoder(xseq, batch_first_override=self.batch_first_transformer)
            transformer_output = self.seqTransEncoder(xseq, src_key_padding_mask=transformer_padding_mask)
            
            # Remove the output corresponding to the condition token
            if self.batch_first_transformer:
                output_latent_seq = transformer_output[:, 1:]
            else:
                output_latent_seq = transformer_output[1:, :, :]

        elif self.arch == 'trans_dec': # DiP-style: motion is target, condition is memory
            # processed_motion_seq is the target sequence (tgt)
            # emb is the memory (combined text+time, potentially armature if not cross-attended separately)
            tgt_seq = self.sequence_pos_encoder(processed_motion_seq, batch_first_override=self.batch_first_transformer)
            
            memory = emb # Base memory is text+time
            memory_key_padding_mask = None # if text was from BERT and had its own mask
            if 'text_mask' in y and y['text_mask'] is not None: # y['text_mask'] is for BERT, [bs, text_seq_len], True for pad
                 memory_key_padding_mask = y['text_mask']

            if self.armature_integration_policy == "cross_attention":
                # Option 1: Armature as separate K,V to its own cross-attention layer
                # This would require modifying the TransformerDecoderLayer or having sequential decoders
                # Option 2 (Simpler for now): Concatenate armature context to main memory
                arm_ctx = y.get('armature_context_for_cross_attn') # [1, bs, d] or None
                if arm_ctx is not None:
                    if self.batch_first_transformer: # memory should be [bs, mem_seq_len, d]
                        memory = torch.cat((memory.permute(1,0,2), arm_ctx.permute(1,0,2)), dim=1)
                        if memory_key_padding_mask is not None:
                            arm_pad = torch.zeros(bs, 1, dtype=torch.bool, device=device)
                            memory_key_padding_mask = torch.cat((memory_key_padding_mask, arm_pad), dim=1)
                    else: # memory should be [mem_seq_len, bs, d]
                        memory = torch.cat((memory, arm_ctx), dim=0)
                        if memory_key_padding_mask is not None:
                             logger.warning("Memory key padding for non-batch-first decoder with concatenated armature context needs careful handling.")
                             # This case is tricky for padding mask shapes. Simplest if text_mask also accounts for this.

            output_latent_seq = self.seqTransDecoder(tgt=tgt_seq, memory=memory, 
                                                     tgt_key_padding_mask=frames_mask_for_padding,
                                                     memory_key_padding_mask=memory_key_padding_mask)
        else:
            raise ValueError(f"Architecture {self.arch} not implemented.")

        # Final projection back to motion features
        output_motion = self.output_process(output_latent_seq, batch_first_input_from_transformer=self.batch_first_transformer)
        return output_motion