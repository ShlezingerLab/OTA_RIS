# Gradient Approximation for Model-Free Training

[cite_start]**Source:** "Model-free Training of End-to-end Communication Systems" by F. Ait Aoudia and J. Hoydis [cite: 1-6].

## 1. The Problem: Missing Channel Gradient
In an end-to-end autoencoder, the transmitter ($f_{\theta_T}^{(T)}$) is typically trained by backpropagating gradients through the channel. [cite_start]The gradient of the loss $\mathcal{L}$ with respect to the transmitter parameters $\theta_T$ is given by [cite: 91-95]:

$$
\nabla_{\theta_{T}}\mathcal{L} = \mathbb{E}_{m} \left\{ \int l(f_{\theta_{R}}^{(R)}(y),m) \nabla_{\theta_{T}}f_{\theta_{T}}^{(T)}(m) \nabla_{x}p(y|x)|_{x=f_{\theta_{T}}^{(T)}(m)}dy \right\}
$$

* **Issue:** The term $\nabla_{x}p(y|x)$ represents the gradient of the channel distribution with respect to its input. [cite_start]In real-world channels (or "black box" models), this gradient is **unknown** or **undefined** (e.g., due to non-differentiable components like quantization)[cite: 95].

## 2. The Solution: Stochastic Relaxation
To circumvent the missing gradient, the authors propose relaxing the transmitter's behavior. [cite_start]Instead of producing a deterministic output $x$, the transmitter output is treated as a **random variable** following a distribution $\hat{\pi}_{\overline{x},\sigma}$ parametrized by the deterministic output $\overline{x}$ and a standard deviation $\sigma$ [cite: 96-99].

* [cite_start]**Interpretation:** This effectively converts the transmitter into a Reinforcement Learning (RL) agent using a stochastic policy, where the "action" is the transmitted symbol $x$ and the "state" is the message $m$[cite: 103, 106].

## 3. The Surrogate Gradient
By applying this relaxation, the loss function is redefined as an expectation over the distribution of $x$. [cite_start]The gradient of this surrogate loss $\hat{\mathcal{L}}$ can be computed using the **log-derivative trick** (REINFORCE algorithm), which removes the dependence on $\nabla_{x}p(y|x)$ [cite: 111-113]:

$$
\nabla_{\theta_{T}}\hat{\mathcal{L}} = \mathbb{E}_{m,x,y} \left\{ l(f_{\theta_{R}}^{(R)}(y),m) \nabla_{\theta_{T}}f_{\theta_{T}}^{(T)}(m) \cdot \nabla_{\overline{x}}\log(\hat{\pi}_{\overline{x},\sigma}(x))|_{\overline{x}=f_{\theta_{T}}^{(T)}(m)} \right\}
$$

### The Estimator (Equation 8)
In practice, this expectation is estimated by sampling. [cite_start]The final gradient estimator used for training the transmitter is[cite: 128]:

$$
\nabla_{\theta_{T}}\hat{\mathcal{L}} \approx \frac{1}{S}\sum_{i=1}^{S} l(f_{\theta_{R}}^{(R)}(y^{(i)}),m^{(i)}) \nabla_{\overline{x}}\log(\hat{\pi}_{\overline{x},\sigma}(x^{(i)}))|_{\overline{x}=f_{\theta_{T}}^{(T)}(m^{(i)})} \nabla_{\theta_{T}}f_{\theta_{T}}^{(T)}(m^{(i)})
$$

Where:
* $S$ is the batch size.
* $l(\cdot)$ is the per-example loss (feedback from receiver).
* $\nabla_{\overline{x}}\log(\hat{\pi}_{\dots})$ captures the direction to move the mean $\overline{x}$ to increase the probability of the sampled symbol $x$.
* $\nabla_{\theta_{T}}f_{\theta_{T}}^{(T)}$ is the standard backpropagation gradient of the transmitter NN.

## 4. Implementation Details
* [cite_start]**Gaussian Perturbation:** The relaxation is typically implemented by adding Gaussian noise $w \sim \mathcal{N}(0, \sigma^2 I)$ to the transmitter output: $x = \overline{x} + w$ [cite: 214-215].
* **Trade-off:** The parameter $\sigma$ controls the exploration-exploitation trade-off.
    * [cite_start]**Small $\sigma$:** Better approximation of the true gradient (low bias) but high variance in the estimator [cite: 218-219].
    * **Large $\sigma$:** Lower variance but higher bias (poor approximation of the deterministic system).
* [cite_start]**Convergence:** Theorem 1 in the paper proves that as $\sigma \to 0$, the surrogate gradient $\nabla_{\theta_{T}}\hat{\mathcal{L}}$ converges to the true gradient $\nabla_{\theta_{T}}\mathcal{L}$ [cite: 117-123].

## 5. Summary of Benefits
1.  **No Channel Model Needed:** The estimator relies only on the observed loss $l$ and the sampled input $x$.
2.  [cite_start]**Differentiability:** It works even if the channel $p(y|x)$ is non-differentiable[cite: 144].
3.  [cite_start]**Simplicity:** It requires only a reliable feedback link to send losses back to the transmitter[cite: 209].
