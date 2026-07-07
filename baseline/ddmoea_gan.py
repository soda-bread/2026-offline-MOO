"""DDMOEA-GAN method implementation copied from the uploaded baseline notebook."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pymoo.core.problem import Problem
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


class RBFN:
    def __init__(self, n_centers=None, gamma=0.5, lambda_reg=1e-6, random_state=0):
        self.n_centers = n_centers
        self.gamma = gamma
        self.lambda_reg = lambda_reg
        self.random_state = random_state
        self.centers = None
        self.weights = None

    def _rbf(self, r):
        return np.exp(-self.gamma * r * r)

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y).reshape(-1, 1)
        n_samples, n_var = X.shape
        n_centers = n_var if self.n_centers is None else self.n_centers
        n_centers = min(n_samples, n_centers)

        kmeans = KMeans(
            n_clusters=n_centers,
            random_state=self.random_state,
            n_init=5,
        )
        kmeans.fit(X)
        self.centers = kmeans.cluster_centers_

        phi = self._rbf(cdist(X, self.centers))
        a = phi.T @ phi + self.lambda_reg * np.eye(n_centers)
        b = phi.T @ y
        self.weights = np.linalg.solve(a, b)

    def predict(self, X_new):
        phi_new = self._rbf(cdist(np.asarray(X_new), self.centers))
        mu = phi_new @ self.weights
        std = np.zeros_like(mu)
        return mu, std


class SurrogateRBFN:
    def __init__(self, gamma=0.5, lambda_reg=1e-6, random_state=0):
        self.model = None
        self.gamma = gamma
        self.lambda_reg = lambda_reg
        self.random_state = random_state

    def fit(self, X, y):
        X = np.asarray(X)
        rbfn = RBFN(
            n_centers=X.shape[1],
            gamma=self.gamma,
            lambda_reg=self.lambda_reg,
            random_state=self.random_state,
        )
        rbfn.fit(X, y)
        self.model = rbfn

    def predict(self, X):
        return self.model.predict(X)


class Generator(nn.Module):
    def __init__(self, z_dim, n_var, n_obj):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, n_var),
            nn.ReLU(),
            nn.Linear(n_var, n_var),
            nn.ReLU(),
            nn.Linear(n_var, n_var + n_obj),
            nn.Tanh(),
        )

    def forward(self, z):
        return self.net(z)


class Discriminator(nn.Module):
    def __init__(self, n_var, n_obj):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_var + n_obj, n_var),
            nn.ReLU(),
            nn.Linear(n_var, n_var),
            nn.ReLU(),
            nn.Linear(n_var, 1),
        )

    def forward(self, x):
        return self.net(x)


def gradient_penalty(discriminator, real_samples, fake_samples, device):
    alpha = torch.rand(real_samples.size(0), 1, device=device)
    alpha = alpha.expand_as(real_samples)
    interpolates = alpha * real_samples + ((1 - alpha) * fake_samples)
    interpolates.requires_grad_(True)

    d_interpolates = discriminator(interpolates)
    fake = torch.ones(real_samples.size(0), 1, device=device)
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    return ((gradients.norm(2, dim=1) - 1) ** 2).mean()


def train_wgan_gp(
    joint_init,
    d_dim,
    n_epochs=2000,
    batch_size=64,
    z_dim=32,
    lambda_gp=10.0,
    n_critic=5,
    lr=1e-4,
    verbose=True,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = Generator(z_dim, d_dim - 2, 2).to(device)
    discriminator = Discriminator(d_dim - 2, 2).to(device)

    optimizer_g = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.9))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.9))

    joint_t = torch.tensor(joint_init, dtype=torch.float32, device=device)
    n_samples = joint_init.shape[0]
    batch_size = min(batch_size, n_samples)

    if verbose:
        print("Training WGAN-GP...")

    for epoch in range(n_epochs):
        idx = np.random.randint(0, n_samples, batch_size)
        real_batch = joint_t[idx]

        for _ in range(n_critic):
            z = torch.randn(batch_size, z_dim, device=device)
            fake_batch = generator(z)

            optimizer_d.zero_grad()
            real_validity = discriminator(real_batch)
            fake_validity = discriminator(fake_batch.detach())
            gp = gradient_penalty(discriminator, real_batch, fake_batch, device)
            d_loss = (
                -torch.mean(real_validity)
                + torch.mean(fake_validity)
                + lambda_gp * gp
            )
            d_loss.backward()
            optimizer_d.step()

        z = torch.randn(batch_size, z_dim, device=device)
        optimizer_g.zero_grad()
        fake_batch = generator(z)
        g_loss = -torch.mean(discriminator(fake_batch))
        g_loss.backward()
        optimizer_g.step()

        if verbose and (epoch + 1) % 500 == 0:
            print(
                f"Epoch {epoch + 1}/{n_epochs} | "
                f"D_loss: {d_loss.item():.4f} | G_loss: {g_loss.item():.4f}"
            )

    if verbose:
        print("WGAN-GP training done.\n")

    return generator, discriminator, device


def build_poly_models(X, F, degree=2):
    polys = []
    regs = []
    for objective_index in range(F.shape[1]):
        poly = PolynomialFeatures(degree=degree, include_bias=True)
        x_poly = poly.fit_transform(X)
        reg = LinearRegression().fit(x_poly, F[:, objective_index])
        polys.append(poly)
        regs.append(reg)
    return polys, regs


def poly_predict(polys, regs, X_new):
    preds = []
    for poly, reg in zip(polys, regs):
        preds.append(reg.predict(poly.transform(X_new)).reshape(-1, 1))
    return np.hstack(preds)


def construct_surrogate_pool_with_gan(
    X_init,
    F_init,
    generator,
    discriminator,
    device,
    n_models,
    select_ratio=0.2,
    poly_degree=2,
    gamma_rbfn=0.5,
    lambda_rbfn=1e-6,
    verbose=True,
):
    n_samples, n_var = X_init.shape
    n_obj = F_init.shape[1]
    if verbose:
        print("Building surrogate pool with GAN augmentation ...")

    polys, regs = build_poly_models(X_init, F_init, degree=poly_degree)
    x_min = X_init.min(axis=0)
    x_max = X_init.max(axis=0)
    f_min = F_init.min(axis=0)
    f_max = F_init.max(axis=0)

    def denorm_x(x_normalized):
        return (x_normalized + 1.0) * 0.5 * (x_max - x_min + 1e-12) + x_min

    surrogate_pools = [[] for _ in range(n_obj)]
    for model_index in range(n_models):
        z = torch.randn(n_samples, generator.net[0].in_features, device=device)
        with torch.no_grad():
            joint_syn = generator(z).cpu().numpy()

        x_syn = denorm_x(joint_syn[:, :n_var])
        with torch.no_grad():
            scores = (
                discriminator(
                    torch.tensor(joint_syn, dtype=torch.float32, device=device)
                )
                .cpu()
                .numpy()
                .flatten()
            )

        top_n = max(1, int(select_ratio * n_samples))
        x_h = x_syn[np.argsort(-scores)[:top_n]]
        f_h = poly_predict(polys, regs, x_h)
        x_train = np.vstack([X_init, x_h])

        for objective_index in range(n_obj):
            y_train = np.vstack(
                [F_init[:, objective_index : objective_index + 1], f_h[:, objective_index : objective_index + 1]]
            )
            model = SurrogateRBFN(
                gamma=gamma_rbfn,
                lambda_reg=lambda_rbfn,
                random_state=model_index,
            )
            model.fit(x_train, y_train)
            surrogate_pools[objective_index].append(model)

        if verbose:
            print(
                f"  Surrogate model {model_index + 1}/{n_models} built. "
                f"Train size = {x_train.shape[0]}"
            )

    if verbose:
        print("Surrogate pool ready.\n")
    return surrogate_pools, (x_min, x_max, f_min, f_max)


def surrogate_predict_with_ensemble(x, surrogate_pools):
    x = np.asarray(x)
    y_mean = np.zeros((x.shape[0], len(surrogate_pools)))
    for objective_index, models in enumerate(surrogate_pools):
        preds = [model.predict(x)[0] for model in models]
        y_mean[:, objective_index : objective_index + 1] = np.hstack(preds).mean(
            axis=1,
            keepdims=True,
        )
    return y_mean


def discriminator_confidence_score(
    x,
    y_pred,
    discriminator,
    device,
    x_min,
    x_max,
    f_min,
    f_max,
):
    x_normalized = 2.0 * (x - x_min) / (x_max - x_min + 1e-12) - 1.0
    y_normalized = 2.0 * (y_pred - f_min) / (f_max - f_min + 1e-12) - 1.0
    joint_t = torch.tensor(
        np.hstack([x_normalized, y_normalized]),
        dtype=torch.float32,
        device=device,
    )
    with torch.no_grad():
        logits = discriminator(joint_t).cpu().numpy().flatten()
    return (1.0 / (1.0 + np.exp(-logits))).reshape(-1, 1)


def critical_fitness(
    x,
    surrogate_pools,
    discriminator,
    device,
    x_min,
    x_max,
    f_min,
    f_max,
    alpha_critic=0.1,
):
    y_mean = surrogate_predict_with_ensemble(x, surrogate_pools)
    confidence = discriminator_confidence_score(
        x,
        y_mean,
        discriminator,
        device,
        x_min,
        x_max,
        f_min,
        f_max,
    )
    return y_mean * (1.0 - alpha_critic * confidence)


class DDMOEAGANProblem(Problem):
    def __init__(
        self,
        n_var,
        n_obj,
        xl,
        xu,
        surrogate_pools,
        discriminator,
        device,
        x_min,
        x_max,
        f_min,
        f_max,
        alpha_critic=0.1,
    ):
        super().__init__(
            n_var=n_var,
            n_obj=n_obj,
            n_constr=0,
            xl=xl,
            xu=xu,
            elementwise_evaluation=False,
        )
        self.surrogate_pools = surrogate_pools
        self.discriminator = discriminator
        self.device = device
        self.x_min = x_min
        self.x_max = x_max
        self.f_min = f_min
        self.f_max = f_max
        self.alpha_critic = alpha_critic

    def _evaluate(self, X, out, *args, **kwargs):
        out["F"] = critical_fitness(
            X,
            self.surrogate_pools,
            self.discriminator,
            self.device,
            self.x_min,
            self.x_max,
            self.f_min,
            self.f_max,
            alpha_critic=self.alpha_critic,
        )
