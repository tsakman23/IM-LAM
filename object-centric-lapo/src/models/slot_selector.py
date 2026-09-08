from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import cross_val_score


class SlotSelector:
    """Select task-relevant slots via PCA + action probing.

    Continuous actions use linear regression with negative MSE; discrete
    actions use logistic regression with classification accuracy.
    """

    def __init__(self, pca_components: int = 32, action_space_type: str = "continuous") -> None:
        self.pca_components = pca_components
        self.action_space_type = action_space_type
        self.slot_scores: Dict[int, float] = {}
        self.selected_slots: List[int] = []

    def evaluate_slots(
        self,
        slot_embeddings: np.ndarray,
        actions: np.ndarray,
        n_folds: int = 5,
    ) -> Dict[int, float]:
        """Evaluate each slot's action-prediction quality.

        Args:
            slot_embeddings: (N, K, D) slot embeddings for all frames.
            actions: continuous action vectors or discrete class labels.
            n_folds: number of CV folds.

        Returns:
            Dict mapping slot index to its cross-validation score.
        """
        n_samples, num_slots, slot_dim = slot_embeddings.shape
        self.slot_scores = {}

        for k in range(num_slots):
            slot_k = slot_embeddings[:, k, :]  # (N, D)

            # PCA dimensionality reduction
            n_components = min(self.pca_components, slot_dim, n_samples)
            pca = PCA(n_components=n_components)
            slot_k_pca = pca.fit_transform(slot_k)

            if self.action_space_type == "discrete":
                model = LogisticRegression(max_iter=1000)
                targets = actions.reshape(-1).astype(np.int64)
                scoring = "accuracy"
            else:
                model = LinearRegression()
                targets = actions
                scoring = "neg_mean_squared_error"
            scores = cross_val_score(
                model, slot_k_pca, targets,
                cv=n_folds,
                scoring=scoring,
            )
            self.slot_scores[k] = float(scores.mean())

        return self.slot_scores

    def select(self, num_slots_to_select: int = 1) -> List[int]:
        """Select the best slot(s) based on evaluation scores.

        Args:
            num_slots_to_select: how many slots to select.

        Returns:
            List of selected slot indices.
        """
        assert self.slot_scores, "Must call evaluate_slots first."
        sorted_slots = sorted(self.slot_scores.items(), key=lambda x: x[1], reverse=True)
        self.selected_slots = [idx for idx, _ in sorted_slots[:num_slots_to_select]]
        return self.selected_slots

    @staticmethod
    @torch.no_grad()
    def extract_all_slots(
        videosaur_model,
        dataloader,
        device: str = "cuda",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract slot embeddings and actions from entire dataset.

        Args:
            videosaur_model: trained VideoSAUR model.
            dataloader: DataLoader yielding TensorDicts with 'observation' and 'action'.
            device: compute device.

        Returns:
            slot_embeddings: (N, K, D) numpy array.
            actions: (N, action_dim) numpy array.
        """
        all_slots = []
        all_actions = []
        videosaur_model.eval()

        for batch in dataloader:
            obs = batch["observation"].to(device)
            actions = batch["action"]

            slots, _ = videosaur_model.extract_slots(obs)
            all_slots.append(slots.cpu().numpy())
            all_actions.append(actions.numpy())

        return np.concatenate(all_slots, axis=0), np.concatenate(all_actions, axis=0)
