Here is a structured Markdown document summarizing the root cause, mathematical breakdown, and solution to the rank deficiency issue. You can copy and paste this directly into a `.md` file for your documentation.

---

# Geometric Rank Deficiency in Cascaded RIS Channels

**Symptom:** The network suffers from representational collapse during evaluation, resulting in a uniformly classified decision boundary (a "pure blue" checkerboard).

**Context:** The architecture tests the necessity of depth by isolating a strictly linear, bias-free intermediate layer, $W_{lin} \in \mathbb{R}^{24 \times 24}$. In the wireless evaluation panel, this learned layer is replaced by a physical surrogate: a cascaded Reconfigurable Intelligent Surface (RIS) channel parameterized by phase shifts $\boldsymbol{\phi}$.

## 1. The Dimensionality Mapping

The neural network operates in the real domain, while the physical channel operates in the complex domain. The $24$-dimensional real vectors are mapped to $12$-dimensional complex vectors ($N_t = 12, N_r = 12$).

To faithfully surrogate the intermediate network topology without information loss, the physical environment must successfully map a $12$-dimensional complex input vector $\mathbf{s}$ to a $12$-dimensional complex target $\mathbf{y}_{learned}$.

The cascaded channel equation is:


$$\mathbf{y}_{learned} = H_2 \text{diag}(\boldsymbol{\phi}) H_1 \mathbf{s}$$

Where:

* $H_1 \in \mathbb{C}^{100 \times 12}$ (Tx to RIS channel)
* $\boldsymbol{\phi} \in \mathbb{C}^{100}$ (RIS phase shifts, optimized via gradient descent)
* $H_2 \in \mathbb{C}^{12 \times 100}$ (RIS to Rx channel)

For the optimization to successfully steer any arbitrary input $\mathbf{s}$ to its target $\mathbf{y}_{learned}$, the effective end-to-end matrix $H_{eq} = H_2 \text{diag}(\boldsymbol{\phi}) H_1$ must have a rank of **$12$**.

## 2. The Bottleneck: Rank-1 Collapse

The initial environment generation utilized high Ricean K-factors ($13$ dB for $H_1$, $7$ dB for $H_2$), simulating an environment heavily dominated by a single Line-of-Sight (LoS) path.

Mathematically, a pure LoS MIMO channel matrix is formed by the outer product of the geometric steering vectors at the transmitter and receiver. For high K-factors, the channel matrices degrade to:


$$H_1 \approx \mathbf{a}_{RIS,1} \mathbf{a}_{Tx}^H$$

$$H_2 \approx \mathbf{a}_{Rx} \mathbf{a}_{RIS,2}^H$$

Substituting these into the cascaded equation yields:


$$\mathbf{y}_{ris} \approx \mathbf{a}_{Rx} \Big( \mathbf{a}_{RIS,2}^H \text{diag}(\boldsymbol{\phi}) \mathbf{a}_{RIS,1} \Big) \mathbf{a}_{Tx}^H \mathbf{s}$$

Because both $\big( \mathbf{a}_{RIS,2}^H \text{diag}(\boldsymbol{\phi}) \mathbf{a}_{RIS,1} \big)$ and $\big(\mathbf{a}_{Tx}^H \mathbf{s}\big)$ evaluate to complex scalars, their product is simply a scaling factor, $c$. The output collapses to:


$$\mathbf{y}_{ris} \approx c \cdot \mathbf{a}_{Rx}$$

**The Consequence:** Regardless of the transmitted data $\mathbf{s}$ or the precision of the optimized RIS phases $\boldsymbol{\phi}$, the received signal is permanently locked into a 1-dimensional complex subspace defined entirely by the receiver's physical steering vector $\mathbf{a}_{Rx}$. Because the decoder receives vectors that always point in the same direction, it triggers the same ReLUs universally, resulting in a single fixed logit and a uniform decision boundary.

## 3. The Solution: Restoring Subspace Capacity

To prevent representational collapse, the physical channels must be full-rank, possessing enough linearly independent vectors to span the required $12$-dimensional target space.

This is achieved by simulating a rich multipath scattering environment, thereby breaking the rank deficiency.

### Implementation Fixes

Modify the channel generation parameters to eliminate LoS dominance:

**Option A: Rayleigh Fading**
Switch the channel topology entirely to eliminate the stationary LoS component.

* `channel_type="rayleigh"`

**Option B: Suppressed LoS**
Maintain the geometric model but heavily attenuate the LoS component to simulate a Non-Line-of-Sight (NLoS) dominated environment.

* `k_factor_h1_db=-100.0`
* `k_factor_h2_db=-100.0`

By enforcing rich scattering, $\text{Rank}(H_1) = 12$ and $\text{Rank}(H_2) = 12$. The $100$ passive elements at the RIS now have the necessary spatial degrees of freedom to fully mimic the $W_{lin}$ weight matrix through the complex manifold.
