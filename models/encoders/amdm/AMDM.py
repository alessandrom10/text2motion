import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# import clip # Not used as SBERT is precomputed
# from model.rotation2xyz import Rotation2xyz # Would be needed if you implement full SMPL/rotation logic
# from model.BERT.BERT_encoder import load_bert # Not used
# from utils.misc import WeightedSum # Not used in this direct adaptation

from typing import Dict, Any, List, Optional, Tuple # Added List, Optional, Tuple

import logging
logger = logging.getLogger(__name__)

# --- Helper Modules (PositionalEncoding, TimestepEmbedder, InputProcess, OutputProcess) ---

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model) # MDM style: [max_len, d_model]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe) # name 'pe' as in original MDM's PositionalEncoding

    def forward(self, x): # x: [seq_len, bs, d_model] (MDM Transformer internal style)
        # Original MDM style for sequence_pos_encoder(x) where x is [seqlen+1, bs, d]
        # self.pe is [max_len, d_model]. Sliced to [seq_len, d_model] then unsqueezed for broadcasting.
        x = x + self.pe[:x.size(0), :].unsqueeze(1)
        return self.dropout(x)


class TimestepEmbedder(nn.Module): # Corresponds to MDM's TimestepEmbedder
    def __init__(self, latent_dim, sequence_pos_encoder): # sequence_pos_encoder is the PositionalEncoding instance
        super().__init__()
        self.latent_dim = latent_dim
        self.sequence_pos_encoder = sequence_pos_encoder # Provides pe table

        time_embed_dim = self.latent_dim
        self.time_embed = nn.Sequential(
            nn.Linear(self.latent_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

    def forward(self, timesteps): # timesteps: [bs]
        # self.sequence_pos_encoder.pe is [max_len, d_model]
        emb = self.sequence_pos_encoder.pe[timesteps.long(), :].to(timesteps.device) # [bs, d_model]
        emb = self.time_embed(emb) # [bs, d_model]
        return emb.unsqueeze(0) # Output: [1, bs, d_model] for MDM style


class InputProcess(nn.Module): # Adapted from MDM's InputProcess
    def __init__(self, data_rep, input_feats, latent_dim): # input_feats = njoints * nfeats_per_joint
        super().__init__()
        self.data_rep = data_rep # e.g., 'xyz', 'rot6d'
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.poseEmbedding = nn.Linear(self.input_feats, self.latent_dim)
        # Add self.velEmbedding if 'rot_vel' data_rep is to be supported like in original MDM
        if self.data_rep == 'rot_vel': # Example if you add velocity-based representation
            self.velEmbedding = nn.Linear(self.input_feats, self.latent_dim)
            logger.info("InputProcess: Initialized for 'rot_vel' data representation.")
        else:
            logger.info(f"InputProcess: Initialized for '{self.data_rep}' data representation, using poseEmbedding.")


    def forward(self, x_motion_data):
        # Original MDM expects x: [bs, njoints, nfeats_per_joint, nframes]
        # It permutes to [nframes, bs, njoints, nfeats_per_joint] then reshapes to [nframes, bs, njoints*nfeats_per_joint]
        # Your input x is assumed to be [bs, nframes, total_motion_features (input_feats)]
        # We need to permute it to [nframes, bs, input_feats] for MDM style processing
        
        nframes = x_motion_data.shape[1]
        bs = x_motion_data.shape[0]
        # x should be [nframes, bs, self.input_feats] before poseEmbedding
        x = x_motion_data.permute(1, 0, 2) # [nframes, bs, self.input_feats]

        if self.data_rep in ['rot6d', 'xyz', 'hml_vec', 'your_custom_rep_flat']: # 'your_custom_rep_flat' for your direct features
            x = self.poseEmbedding(x)  # [nframes, bs, latent_dim]
        # elif self.data_rep == 'rot_vel': # Example for velocity based representation
            # first_pose = x[[0]]
            # first_pose = self.poseEmbedding(first_pose)
            # vel = x[1:]
            # vel = self.velEmbedding(vel)
            # x = torch.cat((first_pose, vel), axis=0)
        else:
            raise ValueError(f"Unsupported data_rep '{self.data_rep}' in InputProcess")
        return x


class OutputProcess(nn.Module): # Adapted from MDM's OutputProcess
    def __init__(self, data_rep, input_feats, latent_dim, njoints, nfeats_per_joint):
        super().__init__()
        self.data_rep = data_rep
        self.input_feats = input_feats # total_motion_features
        self.latent_dim = latent_dim
        # These are for reshaping if the output needs to be structured per joint/feature
        self.njoints = njoints
        self.nfeats_per_joint = nfeats_per_joint
        self.poseFinal = nn.Linear(self.latent_dim, self.input_feats)
        # Add self.velFinal if 'rot_vel' data_rep is supported
        if self.data_rep == 'rot_vel':
             self.velFinal = nn.Linear(self.latent_dim, self.input_feats)
             logger.info("OutputProcess: Initialized for 'rot_vel' data representation.")
        else:
            logger.info(f"OutputProcess: Initialized for '{self.data_rep}' data representation, using poseFinal.")

    def forward(self, output_from_transformer):
        # output_from_transformer: [nframes, bs, latent_dim] (MDM style output from Transformer)
        nframes, bs, _ = output_from_transformer.shape

        if self.data_rep in ['rot6d', 'xyz', 'hml_vec', 'your_custom_rep_flat']:
            output = self.poseFinal(output_from_transformer)  # [nframes, bs, input_feats]
        # elif self.data_rep == 'rot_vel':
            # ... (logic for rot_vel if implemented) ...
        else:
            raise ValueError(f"Unsupported data_rep '{self.data_rep}' in OutputProcess")

        # MDM reshapes to [nframes, bs, njoints, nfeats_per_joint] then permutes.
        # Your desired output is [bs, nframes, num_motion_features (input_feats)]
        output = output.permute(1, 0, 2)  # -> [bs, nframes, input_feats]
        
        # If your input_feats was truly njoints * nfeats_per_joint and you need the original MDM output shape:
        # output = output.reshape(nframes, bs, self.njoints, self.nfeats_per_joint)
        # output = output.permute(1, 2, 3, 0)  # -> [bs, njoints, nfeats_per_joint, nframes]
        return output


class ArmatureMDM(nn.Module):
    def __init__(self,
                 # ... (i tuoi parametri __init__ rimangono invariati) ...
                 data_rep: str,
                 njoints: int,
                 nfeats_per_joint: int,
                 num_motion_features: Optional[int] = None, # Può essere derivato o specificato
                 latent_dim: int = 256, # Default da MDM, la tua config usa 768
                 ff_size: int = 1024,
                 num_layers: int = 8,
                 num_heads: int = 4,
                 dropout: float = 0.1,
                 activation: str = "gelu",
                 sbert_embedding_dim: int = 768,
                 max_armature_classes: int = 10,
                 armature_embedding_dim: int = 64,
                 armature_mlp_hidden_dims: Optional[List[int]] = None,
                 max_seq_len_pos_enc: int = 5000,
                 cond_mask_prob: float = 0.1,
                 armature_cond_mask_prob: float = 0.1,
                 arch: str = 'trans_enc',
                 dataset: str = 'custom_armature_dataset',
                 translation: bool = True,
                 glob: bool = True,
                 pose_rep: str = 'xyz',
                 **kargs):
        super().__init__()

        self.latent_dim = latent_dim
        self.njoints = njoints # Aggiunto per coerenza se usato in OutputProcess
        self.nfeats = nfeats_per_joint # Aggiunto per coerenza

        if "flat" in data_rep:
            if num_motion_features is None:
                raise ValueError("num_motion_features must be provided if data_rep indicates flat features.")
            self.input_feats = num_motion_features
        elif "structured" in data_rep: # Non usato dal tuo InputProcess attuale, ma per completezza
            self.input_feats = njoints * nfeats_per_joint
            if num_motion_features is not None and num_motion_features != self.input_feats:
                 logger.warning(f"num_motion_features ({num_motion_features}) provided but overridden by njoints*nfeats_per_joint ({self.input_feats}) for structured data_rep.")
        else: # Default a num_motion_features se data_rep non è specificato come flat/structured
             self.input_feats = num_motion_features
             if self.input_feats is None:
                 self.input_feats = njoints * nfeats_per_joint # Fallback se num_motion_features è None

        self.data_rep = data_rep
        self.dataset = dataset
        self.pose_rep = pose_rep
        self.translation = translation
        self.glob = glob

        self.cond_mask_prob = cond_mask_prob
        self.armature_cond_mask_prob = armature_cond_mask_prob
        self.arch = arch
        
        self.input_process = InputProcess(self.data_rep, self.input_feats, self.latent_dim)
        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, dropout, max_len=max_seq_len_pos_enc)
        self.embed_timestep = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder)

        self.sbert_embedding_dim = sbert_embedding_dim 
        self.embed_text = nn.Linear(self.sbert_embedding_dim, self.latent_dim)
        # logger.info(f"Text embeddings (pre-computed SBERT {self.sbert_embedding_dim}D) projected to {self.latent_dim}D by self.embed_text.") # Già loggato in train.py

        self.armature_class_embedding = nn.Embedding(max_armature_classes, armature_embedding_dim)
        armature_mlp_output_dim = self.latent_dim
        if armature_mlp_hidden_dims is None or not armature_mlp_hidden_dims:
            self.armature_projection_mlp = nn.Linear(armature_embedding_dim, armature_mlp_output_dim)
        else:
            layers = [nn.Linear(armature_embedding_dim, armature_mlp_hidden_dims[0]), nn.SiLU()]
            for i in range(len(armature_mlp_hidden_dims) - 1):
                layers.extend([nn.Linear(armature_mlp_hidden_dims[i], armature_mlp_hidden_dims[i+1]), nn.SiLU()])
            layers.append(nn.Linear(armature_mlp_hidden_dims[-1], armature_mlp_output_dim))
            self.armature_projection_mlp = nn.Sequential(*layers)
        # logger.info(f"Armature embedding (raw {armature_embedding_dim}D) -> MLP to {armature_mlp_output_dim}D.") # Già loggato

        self.batch_first_transformer = kargs.get('batch_first_transformer', False)
        # logger.info(f"Transformer initialized with batch_first={self.batch_first_transformer}") # Già loggato

        if self.arch == 'trans_enc':
            seqTransEncoderLayer = nn.TransformerEncoderLayer(
                d_model=self.latent_dim, nhead=num_heads, dim_feedforward=ff_size,
                dropout=dropout, activation=activation, batch_first=self.batch_first_transformer
            )
            self.seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer, num_layers=num_layers)
        else:
            raise ValueError(f"This aligned model is for arch='trans_enc'. Got {self.arch}")

        self.output_process = OutputProcess(self.data_rep, self.input_feats, self.latent_dim, self.njoints, self.nfeats)

    def _mask_cond(self, cond_embedding: torch.Tensor, prob: float, force_mask: bool) -> torch.Tensor:
        if force_mask:
            return torch.zeros_like(cond_embedding)
        # >>> MODIFICA QUI: Controlla se prob è None prima del confronto <<<
        if self.training and prob is not None and prob > 0.0:
        # <<< FINE MODIFICA >>>
            mask = torch.bernoulli(torch.ones((1, cond_embedding.size(1), 1), device=cond_embedding.device) * prob)
            return cond_embedding * (1. - mask)
        return cond_embedding

    def forward(self, x, timesteps, y: Dict[str, Any]):
        bs, num_frames_input_original_x, _ = x.shape # x è [bs, nframes, features_flat]
        device = x.device

        time_emb = self.embed_timestep(timesteps)  # [1, bs, latent_dim]

        sbert_embeddings = y['text_embeddings_batch'].to(device) 
        projected_text_emb = self.embed_text(sbert_embeddings).unsqueeze(0) 
        text_emb_masked = self._mask_cond(projected_text_emb, self.cond_mask_prob, y.get('uncond', y.get('uncond_text',False)))


        armature_ids = y['armature_class_ids'].long().to(device)
        raw_armature_emb = self.armature_class_embedding(armature_ids)
        projected_armature_emb = self.armature_projection_mlp(raw_armature_emb).unsqueeze(0)
        armature_emb_masked = self._mask_cond(projected_armature_emb, self.armature_cond_mask_prob, y.get('uncond_armature', y.get('uncond', False)))

        emb = text_emb_masked + time_emb + armature_emb_masked  # [1, bs, latent_dim]
        
        x_for_input_process = x 
        processed_motion_seq = self.input_process(x_for_input_process)  # Output: [nframes_input, bs, latent_dim]
        
        xseq_before_pe = torch.cat((emb, processed_motion_seq), dim=0)  # [nframes_input+1, bs, latent_dim]
        
        # >>> MODIFICA QUI: Rimuovi l'argomento batch_first_input dalla chiamata <<<
        # La PositionalEncoding nel tuo AMDM.py si aspetta [seq_len, bs, d]
        xseq_after_pe = self.sequence_pos_encoder(xseq_before_pe) 
        # <<< FINE MODIFICA >>>

        frames_mask = None
        if 'mask' in y and y['mask'] is not None:
            valid_motion_frames = y['mask'].squeeze(1).squeeze(1) 
            motion_padding = ~valid_motion_frames 
            cond_padding = torch.zeros(bs, 1, dtype=torch.bool, device=device)
            frames_mask_batch_first = torch.cat((cond_padding, motion_padding), dim=1)
            frames_mask = frames_mask_batch_first # TransformerEncoder si aspetta [N, S] o [S, N] a seconda di batch_first
                                               # PyTorch nn.TransformerEncoder si aspetta (N,S) se batch_first=True,
                                               # e (S,N) non è un formato di mask standard per esso.
                                               # La src_key_padding_mask è sempre (N,S) (bs, seq_len)

        xseq_for_transformer = xseq_after_pe
        if self.batch_first_transformer:
            xseq_for_transformer = xseq_after_pe.permute(1,0,2) 
        
        transformer_output_full = self.seqTransEncoder(xseq_for_transformer, src_key_padding_mask=frames_mask)
        
        output_latent_seq_transformer = transformer_output_full
        if self.batch_first_transformer:
            output_latent_seq_for_output_process = output_latent_seq_transformer[:, 1:] 
        else: 
            output_latent_seq_for_output_process = output_latent_seq_transformer[1:, :, :] 

        # OutputProcess si aspetta [nframes, bs, d] e restituisce [bs, nframes, d]
        # Quindi, se batch_first_transformer era True, dobbiamo permutare l'input a OutputProcess
        if self.batch_first_transformer:
             # output_latent_seq_for_output_process è [bs, nframes, d]
             # InputProcess e OutputProcess come definite nel tuo AMDM.py (che permutano)
             # si aspettano [nframes, bs, d] come input intermedio se si segue lo stile MDM
             # Tuttavia, l'OutputProcess nel tuo AMDM.py fa:
             # output = output.permute(1, 0, 2)  # -> [bs, nframes, input_feats]
             # Quindi si aspetta [nframes, bs, d] in input.
            output_motion = self.output_process(output_latent_seq_for_output_process.permute(1,0,2))
        else:
             # output_latent_seq_for_output_process è [nframes, bs, d]
            output_motion = self.output_process(output_latent_seq_for_output_process)
        
        return output_motion