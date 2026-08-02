---
title: "What are Diffusion Models?"
date: 2024-01-15
draft: false
tags: [diffusion, generative-models]
---

Diffusion models are inspired by non-equilibrium thermodynamics. They define a Markov chain of diffusion steps to slowly add random noise to data and then learn to reverse the diffusion process to construct desired data samples from the noise.

## Forward Diffusion Process

Given a data point sampled from a real data distribution $\mathbf{x}_0 \sim q(\mathbf{x})$, we define a forward diffusion process in which we add small amount of Gaussian noise in $T$ steps, producing a sequence of noisy samples $\mathbf{x}_1, \dots, \mathbf{x}_T$.

$$
q(\mathbf{x}_t \mid \mathbf{x}_{t-1}) = \mathcal{N}(\mathbf{x}_t;\ \sqrt{1-\beta_t}\,\mathbf{x}_{t-1},\ \beta_t \mathbf{I})
$$

The step sizes are controlled by a variance schedule $\{\beta_t \in (0,1)\}_{t=1}^T$. A nice property is that we can sample $\mathbf{x}_t$ at any arbitrary timestep $t$ in closed form. Let $\alpha_t = 1 - \beta_t$ and $\bar{\alpha}_t = \prod_{i=1}^{t} \alpha_i$:

$$
\begin{aligned}
\mathbf{x}_t &= \sqrt{\alpha_t}\,\mathbf{x}_{t-1} + \sqrt{1-\alpha_t}\,\boldsymbol{\epsilon}_{t-1} \\
&= \sqrt{\bar{\alpha}_t}\,\mathbf{x}_0 + \sqrt{1-\bar{\alpha}_t}\,\boldsymbol{\epsilon}
\end{aligned}
$$

where $\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$. As $T \to \infty$, $\mathbf{x}_T$ converges to a pure Gaussian distribution.

## Reverse Diffusion Process

If we can reverse the forward process and sample from $q(\mathbf{x}_{t-1} \mid \mathbf{x}_t)$, we can reconstruct the original signal from Gaussian noise. Unfortunately this reverse conditional probability is intractable since it requires knowing the entire dataset. So we learn a model $p_\theta$ to approximate it:

$$
p_\theta(\mathbf{x}_{t-1} \mid \mathbf{x}_t) = \mathcal{N}(\mathbf{x}_{t-1};\ \boldsymbol{\mu}_\theta(\mathbf{x}_t, t),\ \boldsymbol{\Sigma}_\theta(\mathbf{x}_t, t))
$$

The full reverse process is:

$$
p_\theta(\mathbf{x}_{0:T}) = p(\mathbf{x}_T) \prod_{t=1}^{T} p_\theta(\mathbf{x}_{t-1} \mid \mathbf{x}_t)
$$

## Training Objective

Training is done by optimizing the variational lower bound on the negative log likelihood:

$$
\mathbb{E}[-\log p_\theta(\mathbf{x}_0)] \leq \mathbb{E}_q \left[ -\log \frac{p_\theta(\mathbf{x}_{0:T})}{q(\mathbf{x}_{1:T} \mid \mathbf{x}_0)} \right] = \mathcal{L}
$$

We can rewrite this as a sum of KL divergences:

$$
\mathcal{L} = \mathbb{E}_q \left[ \underbrace{D_\text{KL}(q(\mathbf{x}_T \mid \mathbf{x}_0) \| p(\mathbf{x}_T))}_{\mathcal{L}_T} + \sum_{t=2}^{T} \underbrace{D_\text{KL}(q(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0) \| p_\theta(\mathbf{x}_{t-1} \mid \mathbf{x}_t))}_{\mathcal{L}_{t-1}} \underbrace{- \log p_\theta(\mathbf{x}_0 \mid \mathbf{x}_1)}_{\mathcal{L}_0} \right]
$$

## Simplified Loss

Ho et al. (2020) showed that training with a simplified objective works better in practice. Rather than predicting $\boldsymbol{\mu}_\theta$, we train the network to predict the noise $\boldsymbol{\epsilon}$:

$$
\mathcal{L}_\text{simple} = \mathbb{E}_{t, \mathbf{x}_0, \boldsymbol{\epsilon}} \left[ \| \boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\sqrt{\bar{\alpha}_t}\,\mathbf{x}_0 + \sqrt{1-\bar{\alpha}_t}\,\boldsymbol{\epsilon},\ t) \|^2 \right]
$$

This is essentially a denoising score matching objective. The network $\boldsymbol{\epsilon}_\theta$ takes the noisy image and timestep as input and predicts the noise that was added.

## Parameterization of $\beta_t$

The variance schedule $\beta_t$ can be set in different ways. Ho et al. used a linear schedule:

$$
\beta_1 = 10^{-4}, \quad \beta_T = 0.02, \quad T = 1000
$$

Nichol & Dhariwal (2021) proposed a cosine schedule which gives better results:

$$
\bar{\alpha}_t = \frac{f(t)}{f(0)}, \quad f(t) = \cos\left(\frac{t/T + s}{1 + s} \cdot \frac{\pi}{2}\right)^2
$$

where $s = 0.008$ is a small offset to prevent $\beta_t$ from being too small near $t=0$.

## Connection to Score Matching

Diffusion models have a deep connection to score-based generative models. The score function is defined as:

$$
\nabla_{\mathbf{x}} \log p(\mathbf{x})
$$

The denoising objective is equivalent to learning the score of the data distribution at each noise level. Specifically:

$$
\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t) \approx -\sqrt{1-\bar{\alpha}_t}\, \nabla_{\mathbf{x}_t} \log q(\mathbf{x}_t)
$$

This connection unifies two previously separate lines of work — denoising diffusion models and score-based models.

## Sampling

Once trained, we sample by starting from $\mathbf{x}_T \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ and iteratively denoising:

$$
\mathbf{x}_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left( \mathbf{x}_t - \frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}} \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t) \right) + \sigma_t \mathbf{z}
$$

where $\mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ and $\sigma_t^2 = \beta_t$. This requires $T$ forward passes through the network, which is slow. DDIM (Song et al. 2020) accelerates this to as few as 50 steps with no retraining.

## Summary

| Property | Diffusion Models | GANs | VAEs |
|---|---|---|---|
| Training stability | High | Low | High |
| Sample quality | Very high | High | Medium |
| Sample diversity | High | Low | High |
| Sampling speed | Slow | Fast | Fast |
| Likelihood | Tractable | No | Lower bound |

Diffusion models achieve state-of-the-art results on image generation benchmarks while being significantly more stable to train than GANs.