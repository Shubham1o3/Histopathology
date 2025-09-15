import numpy as np
import torch
from utils.core import PseudoBag
from utils.func import seed_everything


class PseudoMixup:
    def __init__(self, num_cluster=8, num_pseb=40, num_ft=8, 
                param_alpha=1.0, num_iter=10, prob_mixup=0.98, device="cpu", seed=42):
        """
        Args:
            num_cluster (int): number of phenotype clusters
            num_pseb (int): number of pseudo-bags
            num_ft (int): fine-tuning iterations
            param_alpha (float): Beta distribution parameter
            num_iter (int): number of iterations for sampling lam
            prob_mixup (float): probability to apply mixup
            device (str): "cpu" or "cuda"
        """
        self.num_cluster = num_cluster
        self.num_pseb = num_pseb
        self.num_ft = num_ft
        self.param_alpha = param_alpha
        self.num_iter = num_iter
        self.prob_mixup = prob_mixup
        self.device = device

        # Fix randomness
        seed_everything(seed)

        # Initialize PseudoBag
        self.pb = PseudoBag(self.num_pseb, self.num_cluster,
                            proto_method='mean', pheno_cut_method='quantile',
                            iter_fine_tuning=self.num_ft)

        # Sample initial lam
        self.lam_val = self._sample_lam()

    def _sample_lam(self):
        lam = 1.0
        for _ in range(self.num_iter):
            lam = np.random.beta(self.param_alpha, self.param_alpha)
        return lam

    def lam_f(self):
        lam_temp = self.lam_val if self.lam_val != 1.0 else self.lam_val - 1e-5
        lam_int = int(lam_temp * (self.num_pseb + 1))
        return lam_int

    def pseudobag(self, bag_features):
        label_pseudo_bag_A = self.pb.divide(bag_features, ret_pseudo_bag=False).to(self.device)
        return label_pseudo_bag_A

    def fetch_pseudo_bags(self, X, ind_X, n:int, n_parts:int):
        """
        Fetch random pseudo-bags
        Args:
            X: bag features, usually [N, d]
            ind_X: pseudo-bag indicator, shape [N,]
            n: total pseudo-bag number
            n_parts: number of pseudo-bags to fetch
        """
        if len(X.shape) > 2:
            X = X.squeeze(0)

        assert n_parts <= n, 'Invalid number of pseudo-bags to fetch.'

        if n_parts == 0:
            return None

        ind_fetched = torch.randperm(n)[:n_parts]
        X_fetched = torch.cat([X[ind_X == ind] for ind in ind_fetched], dim=0)
        return X_fetched

    def mix_up(self, features1, label1, label2, features2):
        """
        Mixup mechanism with pseudo-bags
        Returns:
            features: mixed bag features
            mixup_ratio: lambda ratio
        """
        label_pseudo_bag_A = self.pseudobag(features1)
        label_pseudo_bag_B = self.pseudobag(features2)

        bag_A = self.fetch_pseudo_bags(features1, label_pseudo_bag_A, self.num_pseb, self.lam_f())
        bag_B = self.fetch_pseudo_bags(features2, label_pseudo_bag_B, self.num_pseb, self.num_pseb - self.lam_f())

        if np.random.rand() <= self.prob_mixup and bag_A is not None and bag_B is not None:
            features = torch.cat([bag_A, bag_B], dim=0)
            mixup_ratio = self.lam_f() / self.num_pseb
        elif np.random.rand() <= self.prob_mixup and bag_A is None:
            features = bag_B
            mixup_ratio = 1.0
        elif np.random.rand() <= self.prob_mixup and bag_B is None:
            features = bag_A
            mixup_ratio = 1.0
        elif bag_A is None and bag_B is None:
            features = features1
            mixup_ratio = 1.0
        else:
            features = bag_A
            mixup_ratio = 1.0

        return features, mixup_ratio
