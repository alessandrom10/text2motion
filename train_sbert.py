import random
import torch
import torch.optim as optim
import logging
import numpy as np
from torch.utils.data import DataLoader
from models.diffusion.models.sbert_enhancer import MLPEnhancer, TransformerEnhancer, TripletLossWrapper, InfoNCELoss
from models.diffusion.sbert_trainer_eval import Trainer, Evaluator, EnhancementService, ActionSimilarityDataset, ActionPairDataset, collate_pairs_for_infonce

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

SEED = 42 
set_seed(SEED)

class EnhancerPipeline:
    def __init__(self, args):
        self.args = args
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info(f"Using device: {self.device}")
        self.model = self._build_model().to(self.device)
        self.loss_fn, self.dataset, self.data_loader = self._init_training_components()

    def _build_model(self):
        if self.args.arch == "mlp":
            logging.info("Using MLP architecture.")
            return MLPEnhancer(embedding_dim=self.args.embedding_dim)
        elif self.args.arch == "transformer":
            logging.info("Using Transformer architecture.")
            return TransformerEnhancer(
                embedding_dim=self.args.embedding_dim,
                nhead=self.args.nhead,
                num_encoder_layers=args.num_encoder_layers,
                dropout_rate=args.dropout_rate
            )
        else:
            raise ValueError(f"Unknown architecture: {self.args.arch}")

    def _init_training_components(self):
        if self.args.loss_type == "triplet":
            loss_fn = TripletLossWrapper(margin=0.1).to(self.device)
            dataset = ActionSimilarityDataset(self.args.csv_path, self.args.embeddings_path)
            loader = DataLoader(dataset, batch_size=self.args.batch_size, shuffle=True, num_workers=4)
        elif self.args.loss_type == "infonce":
            loss_fn = InfoNCELoss().to(self.device)
            dataset = ActionPairDataset(self.args.csv_path, self.args.embeddings_path)
            loader = DataLoader(dataset, batch_size=self.args.batch_size, shuffle=True, num_workers=1, collate_fn=collate_pairs_for_infonce)
        else:
            raise ValueError(f"Unknown loss type: {self.args.loss_type}")
        return loss_fn, dataset, loader

    def train(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        trainer = Trainer(self.model, optimizer, self.loss_fn, self.device, mask_prob=self.args.cfg_mask_prob)
        trainer.run(self.data_loader, epochs=self.args.epochs, save_path=self.args.output_model_path)

    def evaluate(self):
        logging.info(f"\n--- Loading best model from {self.args.output_model_path} for evaluation ---")
        model = self._build_model().to(self.device)
        model.load_state_dict(torch.load(self.args.output_model_path, weights_only=True))
        evaluator = Evaluator(model, self.device)

        anchor = self.dataset.embeddings[0]
        positive = self.dataset.embeddings[1]
        negative = self.dataset.embeddings[3]
        evaluator.evaluate_similarity(anchor, positive, negative)

    def infer(self):
        logging.info("\n--- Running Inference Example ---")
        model = self._build_model().to(self.device)
        model.load_state_dict(torch.load(self.args.output_model_path, weights_only=True))
        service = EnhancementService(model, self.device)

        original = self.dataset.embeddings[0].numpy()
        enhanced = service.enhance(original)

        logging.info(f"Original shape: {original.shape}, Enhanced shape: {enhanced.shape}")
        diff = np.linalg.norm(original - enhanced)
        logging.info(f"L2 Norm difference: {diff:.4f}")

if __name__ == '__main__':
    import argparse

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    parser = argparse.ArgumentParser()

    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--embeddings_path", type=str, required=True)
    parser.add_argument("--output_model_path", type=str, default="enhancer_model2.pt")
    parser.add_argument("--arch", type=str, default="mlp", choices=["mlp", "transformer"])
    parser.add_argument("--embedding_dim", type=int, default=384)
    parser.add_argument("--nhead", type=int, default=2)
    parser.add_argument("--num_encoder_layers", type=int, default=2)
    parser.add_argument("--dropout_rate", type=float, default=0.1)
    parser.add_argument("--loss_type", type=str, default="infonce", choices=["triplet", "infonce"])
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--cfg_mask_prob", type=float, default=0.15)

    args = parser.parse_args()

    pipeline = EnhancerPipeline(args)
    pipeline.train()
    pipeline.evaluate()
    pipeline.infer()