"""
This module provides tools to integrate 3D geometric features (from PointNet)
with semantic text features (from CLIP). The goal is to create unified
representations for multi-modal tasks like text-driven 3D animation.

Core Processes:

1.  Alignment (`PointNetCLIPAligner` + `AlignerTrainer`):
    What: A projection MLP (Multi-Layer Perceptron) is trained.
    How: It learns to map PointNet embeddings into the same space as
    CLIP text embeddings, typically using a contrastive loss (e.g., InfoNCE).
    Result: You get 3D embeddings that are "aligned" (semantically
    comparable) with text embeddings.

2.  Fusion (`PointNetCLIPAligner` - `get_conditioned_features` method):
    What: Combines the aligned 3D embedding and the text embedding.
    How: Various methods like sum, weighted sum, or concatenation
    followed by a small `fusion_mlp_module`.
    Result: A single, fused multi-modal embedding representing both
    3D shape and text. The `fusion_mlp_module` will have initial
    (e.g., random) weights unless Step 3 is performed.

3.  Fusion Module Training (Optional - `FusionTrainer`):
    What: This is an  optional  step. If your chosen fusion method
    has trainable parts (like the `fusion_mlp_module` or a learnable weight),
    they can be specifically trained/fine-tuned.
    How: Uses a separate training loop (`FusionTrainer`) and a
    custom loss function designed to optimize the fused embedding for a
    specific characteristic (e.g., to be very similar to the text embedding).
    Result: An potentially improved/specialized fused embedding.

The final aligned or fused embeddings can then be used to condition
downstream models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
from typing import Any, Dict, List, Literal, Optional, Tuple, Union
import logging
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)

# --- Main Aligner Model ---
class PointNetCLIPAligner(nn.Module):
    """
    Aligns PointNet embeddings to CLIP text embeddings using a trainable projection MLP.
    Can also provide fused embeddings by combining the aligned PointNet features
    and CLIP text features for downstream tasks.
    """
    def __init__(
        self,
        pointnet_dim: int = 1024,
        clip_embedding_dim: int = 512,  # Standard for ViT-B/32
        mlp_hidden_dim: int = 768,
        clip_model_name: str = "ViT-B/32",
        temperature_nce: float = 0.07,   # Temperature for InfoNCE loss
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        fusion_type: Optional[Literal["concat_mlp", "sum", "weighted_sum"]] = None,
        fusion_mlp_hidden_dim: int = 512, # Hidden dim for the fusion MLP
        fused_embedding_dim: Optional[int] = None # Output dim of fusion MLP. If None, defaults to clip_embedding_dim
    ):
        """
        Initializes the PointNetCLIPAligner model.

        :param pointnet_dim: Dimension of the PointNet embeddings.
        :param clip_embedding_dim: Dimension of the CLIP text embeddings.
        :param mlp_hidden_dim: Hidden dimension for the projection MLP.
        :param clip_model_name: Name of the CLIP model to load (e.g., "ViT-B/32").
        :param temperature_nce: Temperature for InfoNCE loss.
        :param device: Device to run the model on ("cuda" or "cpu").
        :param fusion_type: Type of fusion mechanism to use ("concat_mlp", "sum", "weighted_sum").
        :param fusion_mlp_hidden_dim: Hidden dimension for the fusion MLP (if using "concat_mlp").
        :param fused_embedding_dim: Output dimension of the fusion MLP. If None, defaults to clip_embedding_dim.
        """
        super().__init__()
        self.pointnet_dim = pointnet_dim
        self.clip_embedding_dim = clip_embedding_dim
        self.mlp_hidden_dim = mlp_hidden_dim
        self.temperature_nce = temperature_nce
        self.device = device
        self.fusion_type = fusion_type
        self.fusion_mlp_hidden_dim = fusion_mlp_hidden_dim
        # If fused_embedding_dim is not provided for concat_mlp, it defaults to clip_embedding_dim
        self.fused_embedding_dim = fused_embedding_dim if fused_embedding_dim is not None else self.clip_embedding_dim

        # MLP to project PointNet embeddings to CLIP space
        self.projection_mlp = nn.Sequential(
            nn.Linear(self.pointnet_dim, self.mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.mlp_hidden_dim, self.clip_embedding_dim)
        ).to(self.device)

        # Load and freeze CLIP model
        self.clip_model = None
        try:
            self.clip_model, _ = clip.load(clip_model_name, device=self.device)
            if self.clip_model:
                for param in self.clip_model.parameters():
                    param.requires_grad = False
                self.clip_model.eval() # Ensure CLIP is in eval mode
        except Exception as e:
            logger.warning(f"CLIP model '{clip_model_name}' could not be loaded. Text encoding will not be available. Error: {e}")


        # Fusion mechanism (parameters are only trained if the fused output is used in an end-to-end setup with an appropriate loss)
        self.fusion_mlp_module = None # Renamed to avoid conflict with self.projection_mlp if a property was named self.mlp
        self.fusion_alpha = None

        if self.fusion_type == "concat_mlp":
            self.fusion_mlp_module = nn.Sequential(
                nn.Linear(self.clip_embedding_dim * 2, self.fusion_mlp_hidden_dim),
                nn.ReLU(),
                nn.Linear(self.fusion_mlp_hidden_dim, self.fused_embedding_dim)
            ).to(self.device)
        elif self.fusion_type == "weighted_sum":
            self.fusion_alpha = nn.Parameter(torch.tensor(0.5, device=self.device)) # Learnable weight

    def encode_pointcloud_to_clip_space(self, pointnet_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Projects PointNet embeddings to CLIP space using the MLP and normalizes them.

        :param pointnet_embeddings: PointNet embeddings to be projected.
        :return: Normalized projected embeddings in CLIP space.
        """
        if pointnet_embeddings.device != self.device:
            pointnet_embeddings = pointnet_embeddings.to(self.device)
        projected_embeddings = self.projection_mlp(pointnet_embeddings)
        return F.normalize(projected_embeddings, dim=-1)

    def encode_text_with_clip(self, texts: List[str]) -> torch.Tensor:
        """
        Encodes a list of text prompts using CLIP's text encoder and normalizes them.

        :param texts: List of text prompts to encode.
        :return: Normalized text embeddings.
        """
        if not self.clip_model:
            raise RuntimeError("CLIP model is not loaded or available. Cannot encode text.")
        tokens = clip.tokenize(texts).to(self.device)
        with torch.no_grad(): # Ensure no gradients are computed for CLIP
            text_features = self.clip_model.encode_text(tokens).float() # Ensure float type
        return F.normalize(text_features, dim=-1)

    def forward(self, pointnet_embeddings: torch.Tensor, texts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Primary forward pass used for training the alignment MLP.
        Returns normalized projected PointNet embeddings and normalized CLIP text embeddings.
        These are the inputs needed for the alignment loss (e.g., InfoNCE).

        :param pointnet_embeddings: PointNet embeddings to be projected.
        :param texts: List of text prompts to encode.
        :return: Tuple of (projected_pointnet_embeddings, text_embeddings).
        """
        projected_pc_embeddings = self.encode_pointcloud_to_clip_space(pointnet_embeddings)
        text_embeddings = self.encode_text_with_clip(texts)
        return projected_pc_embeddings, text_embeddings

    def compute_alignment_loss(
        self,
        projected_pc_embeddings: torch.Tensor,
        text_embeddings: torch.Tensor,
        loss_type: Literal["info_nce", "cosine", "mse"] = "info_nce"
    ) -> torch.Tensor:
        """
        Computes the alignment loss between projected point cloud embeddings and text embeddings.
        Assumes inputs are already normalized for 'info_nce' and 'cosine'.

        :param projected_pc_embeddings: Projected PointNet embeddings in CLIP space.
        :param text_embeddings: CLIP text embeddings.
        :param loss_type: Type of loss to compute ("info_nce", "cosine", "mse").
        :return: Computed loss value.
        """
        if loss_type == "cosine":
            loss = 1 - F.cosine_similarity(projected_pc_embeddings, text_embeddings).mean()
        elif loss_type == "mse":
            loss = F.mse_loss(projected_pc_embeddings, text_embeddings)
        elif loss_type == "info_nce":
            if projected_pc_embeddings.shape[0] != text_embeddings.shape[0]:
                raise ValueError("For InfoNCE loss with in-batch negatives, "
                                 "number of point cloud embeddings and text embeddings must be the same.")
            logits = (projected_pc_embeddings @ text_embeddings.T) / self.temperature_nce
            labels = torch.arange(logits.shape[0], device=logits.device)
            loss_pc_to_text = F.cross_entropy(logits, labels)
            loss_text_to_pc = F.cross_entropy(logits.T, labels) # Symmetric loss
            loss = (loss_pc_to_text + loss_text_to_pc) / 2
        else:
            raise ValueError(f"Unsupported loss_type: {loss_type}")
        return loss

    # This method is used after training to get features for downstream tasks
    # It can return either separate features or a fused feature based on the fusion type specified.
    def get_conditioned_features(
        self,
        pointnet_embeddings: torch.Tensor,
        texts: List[str],
        output_type: Literal["separate", "fused"] = "separate"
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        Generates features for downstream tasks.
        Can return separate or fused features based on the specified output type.

        :param pointnet_embeddings: PointNet embeddings to be projected.
        :param texts: List of text prompts to encode.
        :param output_type: Type of output to return ("separate" or "fused").
        :return: Tuple of (projected_pointnet_embeddings, text_embeddings) if output_type is "separate",
        """
        projected_pc_embeddings = self.encode_pointcloud_to_clip_space(pointnet_embeddings)
        text_embeddings = self.encode_text_with_clip(texts)

        if output_type == "separate":
            return projected_pc_embeddings, text_embeddings
        elif output_type == "fused":
            if self.fusion_type is None:
                raise ValueError("fusion_type was not specified during initialization, cannot create fused embedding.")

            if self.fusion_type == "sum":
                fused_emb = projected_pc_embeddings + text_embeddings
                return F.normalize(fused_emb, dim=-1) # Normalize after sum
            elif self.fusion_type == "weighted_sum":
                if self.fusion_alpha is None:
                    raise RuntimeError("Fusion alpha not initialized for weighted_sum.")
                fused_emb = self.fusion_alpha * projected_pc_embeddings + (1 - self.fusion_alpha) * text_embeddings
                return F.normalize(fused_emb, dim=-1) # Normalize after weighted sum
            elif self.fusion_type == "concat_mlp":
                if self.fusion_mlp_module is None:
                     raise RuntimeError("Fusion MLP module not initialized for concat_mlp.")
                combined = torch.cat((projected_pc_embeddings, text_embeddings), dim=-1)
                fused_emb = self.fusion_mlp_module(combined)
                # The fusion MLP can learn to output normalized embeddings if desired, or normalize here:
                # return F.normalize(fused_emb, dim=-1)
                return fused_emb # Output of MLP might not need explicit normalization here
            else:
                raise ValueError(f"Unsupported fusion_type: {self.fusion_type}")
        else:
            raise ValueError(f"Unsupported output_type: {output_type}")

# --- Trainer Class ---
class Mock3DTextDataset(Dataset): # Replace with actual Dataset
    """A mock dataset for demonstration purposes."""
    def __init__(self, num_samples=1000, pointnet_dim=1024, device="cpu"):
        self.num_samples = num_samples
        self.pointnet_dim = pointnet_dim
        # Simulate pre-computed PointNet embeddings
        self.pc_embeddings = [torch.randn(pointnet_dim) for _ in range(num_samples)]
        self.texts = [f"mockup description for 3d object sample {i}" for i in range(num_samples)]
        self.device = device # Store device if samples need to be on it directly (less common for Dataset)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Typically, datasets return CPU tensors; DataLoader handles batching and moving to GPU
        return self.pc_embeddings[idx], self.texts[idx]

class AlignerTrainer:
    """Handles the training loop for the PointNetCLIPAligner's projection MLP."""
    def __init__(
        self,
        model: PointNetCLIPAligner,
        optimizer: optim.Optimizer,
        alignment_loss_type: Literal["info_nce", "cosine", "mse"] = "info_nce",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        lr_scheduler=None # Optional: e.g., torch.optim.lr_scheduler.StepLR
    ):
        """
        Initializes the trainer with the model, optimizer, and loss type, used for training.

        :param model: The PointNetCLIPAligner model to train.
        :param optimizer: Optimizer for training the model.
        :param alignment_loss_type: Type of loss to use for training ("info_nce", "cosine", "mse").
        :param device: Device to run the training on ("cuda" or "cpu").
        :param lr_scheduler: Optional learning rate scheduler.
        """
        self.model = model.to(device)
        self.optimizer = optimizer
        self.alignment_loss_type = alignment_loss_type
        self.device = device
        self.lr_scheduler = lr_scheduler

        if model.clip_model is None:
            logger.warning("Trainer Warning: AlignerTrainer initialized with a model where CLIP couldn't be loaded. "
                   "Text-dependent operations will fail.")

    def _prepare_batch(self, pointnet_batch_data: Union[List[torch.Tensor], torch.Tensor],
                       text_batch_data: List[str]) -> Tuple[torch.Tensor, List[str]]:
        """
        Moves PointNet embeddings to the specified device and ensures text data is in the correct format.

        :param pointnet_batch_data: PointNet embeddings batch data (list of tensors or single tensor).
        :param text_batch_data: Text data (list of strings).
        :return: Tuple of (pointnet_batch, text_batch).
        """
        # If pointnet_batch_data is a list of tensors (from default collate_fn)
        if isinstance(pointnet_batch_data, list) and isinstance(pointnet_batch_data[0], torch.Tensor):
            pointnet_batch = torch.stack(pointnet_batch_data, dim=0).to(self.device)
        elif isinstance(pointnet_batch_data, torch.Tensor):
            pointnet_batch = pointnet_batch_data.to(self.device)
        else:
            raise TypeError(f"Unsupported pointnet_batch data type: {type(pointnet_batch_data)}")
        # text_batch_data is expected to be a list of strings, handled by model's encode_text_with_clip
        return pointnet_batch, text_batch_data

    def train_epoch(self, data_loader: DataLoader) -> float:
        """
        Trains the model for one epoch.
        
        :param data_loader: DataLoader for the training dataset.
        :return: Average loss for the epoch.
        """
        self.model.train() # Set model to training mode (affects projection_mlp and fusion_mlp if it were trained here)
        total_loss = 0.0
        num_batches = len(data_loader)

        for batch_idx, (pointnet_data, text_data) in enumerate(data_loader):
            pointnet_batch, text_batch = self._prepare_batch(pointnet_data, text_data)

            # The model's forward pass for training the alignment MLP
            # This returns (projected_pc_embeddings, text_embeddings)
            projected_pc_embeddings, text_embeddings = self.model(pointnet_batch, text_batch)

            # Compute the alignment loss
            loss = self.model.compute_alignment_loss(
                projected_pc_embeddings,
                text_embeddings,
                loss_type=self.alignment_loss_type
            )

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            
            if (batch_idx + 1) % max(1, num_batches // 10) == 0: # Log ~10 times per epoch
                logger.info(f"  Batch {batch_idx+1}/{num_batches}, Loss: {loss.item():.4f}")

        avg_loss = total_loss / num_batches
        if self.lr_scheduler:
            self.lr_scheduler.step() # Or scheduler.step(avg_loss) for ReduceLROnPlateau

        return avg_loss

    def evaluate(self, data_loader: DataLoader) -> float:
        """
        Evaluates the model on a validation set.

        :param data_loader: DataLoader for the validation dataset.
        :return: Average loss for the validation set.
        """
        self.model.eval() # Set model to evaluation mode
        total_loss = 0.0
        num_batches = len(data_loader)
        with torch.no_grad():
            for pointnet_data, text_data in data_loader:
                pointnet_batch, text_batch = self._prepare_batch(pointnet_data, text_data)
                
                projected_pc_embeddings, text_embeddings = self.model(pointnet_batch, text_batch)
                loss = self.model.compute_alignment_loss(
                    projected_pc_embeddings,
                    text_embeddings,
                    loss_type=self.alignment_loss_type
                )
                total_loss += loss.item()
        avg_loss = total_loss / num_batches
        return avg_loss

    def train(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None, num_epochs: int = 100):
        """
        Full training loop for the alignment MLP.

        :param train_loader: DataLoader for the training dataset.
        :param val_loader: Optional DataLoader for the validation dataset.
        :param num_epochs: Number of epochs to train for.
        """
        logger.info(f"Starting training for {num_epochs} epochs on device: {self.device} using '{self.alignment_loss_type}' loss.")
        for epoch in range(1, num_epochs + 1):
            logger.info(f"--- Epoch {epoch}/{num_epochs} ---")
            train_loss = self.train_epoch(train_loader)
            logger.info(f"Epoch {epoch} Training Loss: {train_loss:.4f}")

            if val_loader:
                val_loss = self.evaluate(val_loader)
                logger.info(f"Epoch {epoch} Validation Loss: {val_loss:.4f}")
            
            if self.lr_scheduler:
                current_lr = self.optimizer.param_groups[0]['lr'] # Get current LR
                logger.info(f"Current learning rate: {current_lr:.6f}")
            
            # Example of checkpointing (optional)
            # torch.save(self.model.projection_mlp.state_dict(), f"aligner_projection_mlp_epoch_{epoch}.pth")
            # Or save the whole model state: torch.save(self.model.state_dict(), f"aligner_model_epoch_{epoch}.pth")

        logger.info("Training finished.")

class FusionTrainer:
    """
    Handles the training loop for the fusion module (e.g., fusion_mlp_module)
    within the PointNetCLIPAligner model.
    """
    def __init__(
        self,
        model: PointNetCLIPAligner, # The model containing the fusion module to be trained
        optimizer: optim.Optimizer, # Optimizer configured for anly fusion module parameters
        fusion_loss_fn,             # Custom loss function for fusion, e.g., 1 - cosine_similarity
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        lr_scheduler=None
    ):
        """
        Initializes the FusionTrainer.

        :param model: The PointNetCLIPAligner instance.
        :param optimizer: Optimizer for the fusion module's parameters.
        :param fusion_loss_fn: A function (fused_embedding, target_embedding) -> loss_value.
        :param device: Device to run training on.
        :param lr_scheduler: Optional learning rate scheduler.
        """
        self.model = model.to(device)
        self.optimizer = optimizer
        self.fusion_loss_fn = fusion_loss_fn
        self.device = device
        self.lr_scheduler = lr_scheduler

        if not (self.model.fusion_type == "concat_mlp" and self.model.fusion_mlp_module) and \
           not (self.model.fusion_type == "weighted_sum" and self.model.fusion_alpha is not None):
            logger.warning("Warning: FusionTrainer initialized, but the model's fusion_type might not have trainable parameters "
                  "or is not 'concat_mlp' or 'weighted_sum'. Ensure optimizer targets correct parameters.")


    def train_epoch(self, data_loader: DataLoader) -> float:
        """ 
        Trains the fusion module for one epoch. 

        :param data_loader: DataLoader for the training dataset.
        :return: Average loss for the epoch.
        """
        # Ensure parts of the model not being trained are in eval mode and have grads off.
        # This should ideally be set before creating the optimizer for this trainer.
        self.model.projection_mlp.eval()
        if self.model.clip_model:
            self.model.clip_model.eval()

        # Set the specific fusion part to train mode if it's a module
        if self.model.fusion_type == "concat_mlp" and self.model.fusion_mlp_module:
            self.model.fusion_mlp_module.train()
        # For 'weighted_sum', fusion_alpha is a nn.Parameter, its 'train' state is implicit.
        
        total_loss = 0.0
        num_batches = len(data_loader)

        for batch_idx, (pc_emb_list, text_list) in enumerate(data_loader):
            pc_tensor = torch.stack(pc_emb_list, dim=0).to(self.device) if isinstance(pc_emb_list, list) else pc_emb_list.to(self.device)

            self.optimizer.zero_grad()

            # 1. Get the fused embedding from the model
            #    get_conditioned_features uses projection_mlp (eval) and fusion_mlp_module (train)
            fused_embedding = self.model.get_conditioned_features(
                pc_tensor, text_list, output_type="fused"
            )

            # 2. Get the target embedding for the fusion loss (e.g., original text embedding)
            with torch.no_grad(): # Target computation should not affect gradients for fusion module
                target_text_embedding = self.model.encode_text_with_clip(text_list)
            
            # 3. Compute the fusion-specific loss
            loss = self.fusion_loss_fn(fused_embedding, target_text_embedding)

            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

            if (batch_idx + 1) % max(1, num_batches // 10) == 0:
                logger.info(f"  Fusion Batch {batch_idx+1}/{num_batches}, Fusion Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        if self.lr_scheduler:
            self.lr_scheduler.step()
        return avg_loss

    def train(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None, num_epochs: int = 10):
        """ 
        Full training loop for the fusion module. 

        :param train_loader: DataLoader for the training dataset.
        :param val_loader: Optional DataLoader for the validation dataset.
        :param num_epochs: Number of epochs to train for.
        :return: None
        """
        logger.info(f"Starting fusion module training for {num_epochs} epochs on device: {self.device}.")
        
        # Important: Ensure requires_grad is correctly set on model parameters
        # *before* this training loop and optimizer creation.
        # This trainer assumes the optimizer is correctly configured for fusion parameters.

        for epoch in range(1, num_epochs + 1):
            logger.info(f"--- Fusion Training Epoch {epoch}/{num_epochs} ---")
            train_loss = self.train_epoch(train_loader)
            logger.info(f"Epoch {epoch} Fusion Training Loss: {train_loss:.4f}")

            # Optional: Implement an evaluate_epoch method for fusion if needed
            # if val_loader:
            #     val_loss = self.evaluate_epoch(val_loader) # You'd need to create this method
            #     print(f"Epoch {epoch} Fusion Validation Loss: {val_loss:.4f}")
            
            if self.lr_scheduler and hasattr(self.optimizer, 'param_groups'):
                current_lr = self.optimizer.param_groups[0]['lr']
                logger.info(f"Current fusion learning rate: {current_lr:.6f}")
        logger.info("Fusion module training finished.")


def run_demonstration(config: Dict[str, Any]):
    """
    Runs a demonstration of PointNet-CLIP alignment and fusion.
    """
    logger.info(f"Running demonstration with config: {config}")
    DEVICE = config["DEVICE"]

    # --- Model Initialization ---
    # This single model instance will be used for all steps.
    # Its sub-parts (projection_mlp, fusion_mlp_module) will have their
    # requires_grad and train/eval modes managed appropriately per step.
    aligner_model = PointNetCLIPAligner(
        pointnet_dim=config["POINTNET_DIM"],
        clip_embedding_dim=config["CLIP_EMB_DIM"],
        mlp_hidden_dim=config["MLP_HIDDEN_DIM"],
        clip_model_name=config["CLIP_MODEL_NAME"],
        temperature_nce=config["TEMPERATURE_NCE"],
        device=DEVICE,
        fusion_type=config["FUSION_TYPE"],
        fusion_mlp_hidden_dim=config["FUSION_MLP_HIDDEN_DIM"],
        fused_embedding_dim=config["FUSED_EMBEDDING_DIM"]
    ).to(DEVICE)

    if not aligner_model.clip_model:
        logger.error("CLIP model failed to load. Aborting demonstration.")
        return

    # --- Data Preparation (Minimal) ---
    # Using Mock3DTextDataset as defined above
    train_dataset = Mock3DTextDataset(num_samples=config["MOCK_NUM_SAMPLES_TRAIN"], pointnet_dim=config["POINTNET_DIM"])
    train_dataloader = DataLoader(train_dataset, batch_size=config["BATCH_SIZE"], shuffle=True)

    # --- STEP 1: Align PointNet and CLIP Embeddings (Train Projection MLP) ---
    if config["DO_ALIGNMENT_TRAINING"]:
        logger.info("\n--- STEP 1: Training Projection MLP for Alignment ---")
        # Optimizer for the projection_mlp ONLY
        # Ensure only projection_mlp params are trainable for this phase
        for param in aligner_model.parameters(): # Reset all first
            param.requires_grad = False
        for param in aligner_model.projection_mlp.parameters():
            param.requires_grad = True
        
        optimizer_align = optim.Adam(
            filter(lambda p: p.requires_grad, aligner_model.parameters()),
            lr=config["LEARNING_RATE_ALIGNMENT"]
        )
        
        alignment_trainer = AlignerTrainer(
            model=aligner_model,
            optimizer=optimizer_align,
            alignment_loss_type="info_nce",
            device=DEVICE
        )
        alignment_trainer.train(train_dataloader, num_epochs=config["NUM_EPOCHS_ALIGNMENT"])
        logger.info("Alignment MLP training finished.")
    else:
        logger.info("\n--- STEP 1: Alignment Training SKIPPED ---")


    # --- STEP 2: Fuse the Aligned Embeddings (Brief Demo) ---
    logger.info("\n--- STEP 2: Fusing Aligned Embeddings (Demonstration) ---")
    aligner_model.eval() # Set model to evaluation mode for this demo part
    
    try:
        sample_pc_list, sample_text_list = next(iter(train_dataloader))
        sample_pc_tensor = torch.stack(sample_pc_list, dim=0).to(DEVICE) if isinstance(sample_pc_list, list) else sample_pc_list.to(DEVICE)

        with torch.no_grad():
            fused_features_demo = aligner_model.get_conditioned_features(
                sample_pc_tensor, sample_text_list, output_type="fused"
            )
            if fused_features_demo is not None : # Check if fusion was possible (fusion_type might be None)
                 logger.info(f"Shape of fused_features (using FUSION_TYPE: {aligner_model.fusion_type}): {fused_features_demo.shape}")
                 if aligner_model.fusion_type == "concat_mlp":
                    logger.info("Note: If alignment training was done, projection_mlp is trained.")
                    logger.info("      The 'fusion_mlp_module' has initial weights (random or pre-set) at this stage if not yet trained in STEP 4.")
            else:
                logger.info(f"Fusion type is None or invalid ({aligner_model.fusion_type}). Skipping fusion demo.")

    except StopIteration:
        logger.warning("Could not get a sample batch for fusion demonstration.")


    # --- STEP 3: Evaluate Performance (Conceptual Overview) ---
    logger.info("\n--- STEP 3: Evaluating Performance (Conceptual) ---")
    logger.info("1. Alignment Performance: Indicated by validation loss (e.g., InfoNCE) during STEP 1 training.")
    logger.info("2. Fusion Performance: Quality is task-dependent. Requires a downstream task, specific metrics, "
                "and potentially training the fusion module parameters (see STEP 4) with a task-specific loss.")

    # --- STEP 4: Optional - Train Fusion Module ---
    """
    if config["DO_FUSION_TRAINING"] and \
       aligner_model.fusion_type in ["concat_mlp", "weighted_sum"] and \
       (aligner_model.fusion_mlp_module is not None or aligner_model.fusion_alpha is not None) :

        logger.info(f"\n--- STEP 4: Training Fusion Module ('{aligner_model.fusion_type}') ---")

        # Freeze projection_mlp (it's already trained or pre-set)
        for param in aligner_model.projection_mlp.parameters():
            param.requires_grad = False
        
        # Set requires_grad for fusion parameters
        fusion_params_to_train = []
        if aligner_model.fusion_type == "concat_mlp" and aligner_model.fusion_mlp_module:
            for param in aligner_model.fusion_mlp_module.parameters():
                param.requires_grad = True
                fusion_params_to_train.append(param)
        elif aligner_model.fusion_type == "weighted_sum" and aligner_model.fusion_alpha is not None:
            aligner_model.fusion_alpha.requires_grad = True
            fusion_params_to_train.append(aligner_model.fusion_alpha)
        
        if not fusion_params_to_train:
            logger.warning("No trainable parameters found for the specified fusion type. Skipping STEP 4.")
        else:
            optimizer_fusion = optim.Adam(fusion_params_to_train, lr=config["LEARNING_RATE_FUSION"])

            # Define a simple fusion loss: make fused embedding similar to text embedding
            def fusion_cosine_loss(fused_emb, target_text_emb):
                fused_norm = F.normalize(fused_emb, dim=-1)
                # target_text_emb is already normalized by encode_text_with_clip
                return 1 - F.cosine_similarity(fused_norm, target_text_emb).mean()

            fusion_trainer = FusionTrainer(
                model=aligner_model,
                optimizer=optimizer_fusion,
                fusion_loss_fn=fusion_cosine_loss,
                device=DEVICE
            )
            fusion_trainer.train(train_dataloader, num_epochs=config["NUM_EPOCHS_FUSION"])
            logger.info("Fusion module training finished.")

            # Optional: Show fused features again after fusion training
            aligner_model.eval()
            try:
                # Reuse sample_pc_tensor and sample_text_list if they exist
                with torch.no_grad():
                    fused_features_after_train = aligner_model.get_conditioned_features(
                        sample_pc_tensor, sample_text_list, output_type="fused"
                    )
                    logger.info(f"Shape of fused_features after fusion training: {fused_features_after_train.shape}")
            except (StopIteration, NameError): # Handle if sample_pc_tensor was not created
                 logger.warning("Could not get sample batch to show fused features after fusion training.")

    else:
        logger.info("\n--- STEP 4: Fusion Module Training SKIPPED (due to config or unsuitable fusion type) ---")
    """
    logger.info("\nEnd of demonstration function.")


if __name__ == '__main__':
    # --- Configuration for the Demonstration ---
    config = {
        "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
        "POINTNET_DIM": 1024,
        "CLIP_EMB_DIM": 512,
        "MLP_HIDDEN_DIM": 768,
        "FUSION_MLP_HIDDEN_DIM": 512, # Used if FUSION_TYPE is "concat_mlp"
        "FUSED_EMBEDDING_DIM": 512,   # Used if FUSION_TYPE is "concat_mlp"
        "BATCH_SIZE": 8,
        "LEARNING_RATE_ALIGNMENT": 1e-4,
        "LEARNING_RATE_FUSION": 5e-5,
        "NUM_EPOCHS_ALIGNMENT": 2,  # Minimal epochs for quick demo
        "NUM_EPOCHS_FUSION": 2,     # Minimal epochs for quick demo
        "CLIP_MODEL_NAME": "ViT-B/32",
        "TEMPERATURE_NCE": 0.07,
        "FUSION_TYPE": "concat_mlp", # Try "concat_mlp", "sum", "weighted_sum", or None
        "DO_ALIGNMENT_TRAINING": True,
        "DO_FUSION_TRAINING": True,    # Set to False to skip STEP 4
        "MOCK_NUM_SAMPLES_TRAIN": 32,  # e.g., BATCH_SIZE * 4
    }

    run_demonstration(config)
