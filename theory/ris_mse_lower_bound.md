# A Lower Bound on the MSE of RIS-Based Analog Linear Synthesis

*Rigorous version of the "rank matters" result of `README.md` §7.*

This note states and proves, in full, the claim that a unit-modulus RIS cannot
synthesize an arbitrary target vector $\mathbf{y}\in\mathbb{C}^{N_r}$ when the
propagation links are line-of-sight dominated, and quantifies the residual error
at finite Rician factor $\kappa$.

The headline results are:

* **Theorem 2** — in the pure-LoS limit the minimum achievable synthesis error is
  *exactly* $\|\mathbf{P}^{\perp}_{\mathbf{a}_{\mathrm{rx}}}\mathbf{y}\|_2^2$, with
  no relaxation and no asymptotics.
* **Theorem 3** — at finite $\kappa$ a non-asymptotic bound that converges to that
  floor at rate $\mathcal{O}\!\big(M|\mathbf{a}_{\mathrm{tx}}^H\mathbf{s}|/\sqrt{\kappa}\big)$.
* **Theorem 1 / Corollaries 1–2** — the exact value of the norm-relaxed problem, which is
  the machinery the modal-SNR formula was reaching for, together with the precise
  condition under which that formula is a valid bound (it fails in the high-LoS
  regime; see the [Errata](#8-errata-relative-to-the-draft-derivation)).
* **Corollary 4** — digital pre- and post-processing
  ($\mathbf{U}^H\mathbf{H}_2\mathbf{\Phi}\mathbf{H}_1\mathbf{P}_{\mathrm{tx}}$) does not
  escape the pure-LoS floor: the physical map stays rank one.

The ICASSP draft this note corresponds to is
`theory/DID_Over_the_Air_Edge_Inference_ICASSP_2027.pdf`. Numbering there is
shifted, and the PDF's last two sections are the *draft* derivation that
[§8](#8-errata-relative-to-the-draft-derivation) rejects. Use this table when
quoting the PDF:

| PDF | This note |
|---|---|
| Theorem 1 (range floor; pure-LoS exactness) | Theorem 2 |
| Theorem 2 (finite $\kappa$) | Theorem 3 |
| Corollary 1 (cosine / AGC) | Corollary 3 |
| Corollary 2 (pre- and post-processing) | Corollary 4 |
| §6 spectral route, (19)–(21); Remark 3 | Theorem 1, Corollary 1, Corollary 2, (6.2) |
| §7 assembled result, (23)–(25) | §6, (6.1)–(6.3) |
| §8 limitations | §7 |
| §§9–10, headed "Gemini" | [Appendix](#11-appendix-draft-derivation-as-printed-in-the-pdf-sections-9-10); not a result |

Sections 1–8 of the PDF are a shortened rendering of §§1–7 here (Lemmas 3–7 and
the full KKT argument are only in this note). Sections 9–10 restate the model
and then repeat the invalid derivation; they are reproduced in the Appendix so
they can be quoted without opening the PDF.

Numerical verification: `theory/verify_mse_lower_bound.py` (synthetic model).
Experimental study on the trained CIFAR RIS pipeline is in
[§12](#12-experimental-validation-against-the-cifar-ris-pipeline)
(`framework/cifar_minimal_dnn.py --mse_sweep`). Its finding: at finite `kappa` the
RIS *can* synthesize `W_lin s` exactly (the noiseless optimum is ~0), so the floor
binds not unconditionally but through a **K-vs-SNR tradeoff** — the physical
(noise-aware) error rises to `||P_perp y||^2` only once `kappa >~ SNR - 10 log10 N_r`.

---

## 1. Notation, model, standing assumptions

Throughout, $M\equiv N_m$ is the number of RIS elements. Vectors are columns,
$(\cdot)^H$ is conjugate transpose, $\lambda_1(\mathbf{X})\ge\cdots\ge\lambda_n(\mathbf{X})$
are the ordered eigenvalues of a Hermitian $\mathbf{X}\in\mathbb{C}^{n\times n}$,
$s_{\max}(\cdot)$ and $s_{\min}(\cdot)$ are extreme singular values, $\|\cdot\|_2$ is the
Euclidean norm on vectors and the spectral norm on matrices, and
$(x)_+ \triangleq \max(x,0)$.

**Forward model.** For a fixed input latent $\mathbf{s}\in\mathbb{C}^{N_t}$,

$$
\mathbf{y}_{\mathrm{synth}}
= \mathbf{H}_2\,\mathbf{\Phi}\,\mathbf{H}_1\,\mathbf{s} + \mathbf{w},
\qquad
\mathbf{\Phi}=\operatorname{diag}(\boldsymbol{\phi}),
\qquad
\mathbf{w}\sim\mathcal{CN}(\mathbf{0},\sigma^2\mathbf{I}_{N_r}),
$$

with $\mathbf{H}_1\in\mathbb{C}^{M\times N_t}$, $\mathbf{H}_2\in\mathbb{C}^{N_r\times M}$ and
the control variable constrained to the complex torus

$$
\boldsymbol{\phi}\in\mathbb{T}^M\triangleq\{\boldsymbol{\phi}\in\mathbb{C}^M:\ |\phi_m|=1,\ m=1,\dots,M\}.
$$

**Channel statistics.** Write $\varepsilon\triangleq\frac{1}{\kappa+1}\in(0,1]$, so that
$\frac{\kappa}{\kappa+1}=1-\varepsilon$ and $\kappa\to\infty \iff \varepsilon\to 0$. Then

$$
\mathbf{H}_1=\sqrt{1-\varepsilon}\,\mathbf{a}_{\mathrm{ris},1}\mathbf{a}_{\mathrm{tx}}^H+\sqrt{\varepsilon}\,\tilde{\mathbf{H}}_1,
\qquad
\mathbf{H}_2=\sqrt{1-\varepsilon}\,\mathbf{a}_{\mathrm{rx}}\mathbf{a}_{\mathrm{ris},2}^H+\sqrt{\varepsilon}\,\tilde{\mathbf{H}}_2 .
$$

**Standing assumptions.**

* **(A1)** $\|\mathbf{a}_{\mathrm{tx}}\|_2^2=N_t$, $\|\mathbf{a}_{\mathrm{rx}}\|_2^2=N_r$, and
  $|[\mathbf{a}_{\mathrm{ris},i}]_m|=1$ for all $m$ and $i\in\{1,2\}$ (hence
  $\|\mathbf{a}_{\mathrm{ris},i}\|_2^2=M$).
* **(A2)** $\tilde{\mathbf{H}}_1,\tilde{\mathbf{H}}_2$ have i.i.d.
  $\mathcal{CN}(0,1)$ entries and are mutually independent and independent of $\mathbf{w}$.
* **(A3)** $\alpha\triangleq|\mathbf{a}_{\mathrm{tx}}^H\mathbf{s}|>0$ and $\beta\triangleq\|\mathbf{s}\|_2$.
* **(A4)** $M\ge N_r\ge 2$.

All statements about $\boldsymbol{\phi}$ and $\mathbf{w}$ are made **conditionally on a
channel realization** $(\mathbf{H}_1,\mathbf{H}_2)$; the only randomness left in the
optimization is $\mathbf{w}$. Statements containing "with probability at least
$1-\delta$" refer to the draw of $(\tilde{\mathbf{H}}_1,\tilde{\mathbf{H}}_2)$.

**Objective.**

$$
\mathrm{MSE}_{\min}(\mathbf{s})
\triangleq
\min_{\boldsymbol{\phi}\in\mathbb{T}^M}\
\mathbb{E}_{\mathbf{w}}\Big[\big\|\mathbf{y}-(\mathbf{H}_2\mathbf{\Phi}\mathbf{H}_1\mathbf{s}+\mathbf{w})\big\|_2^2\Big].
$$

---

## 2. Reduction to a linear synthesis problem

### Lemma 1 (noise separation)

For every $\boldsymbol{\phi}$,
$\ \mathbb{E}_{\mathbf{w}}\|\mathbf{y}-\mathbf{A}\boldsymbol{\phi}-\mathbf{w}\|_2^2
=\|\mathbf{y}-\mathbf{A}\boldsymbol{\phi}\|_2^2+N_r\sigma^2 .$

*Proof.* Let $\mathbf{r}=\mathbf{y}-\mathbf{A}\boldsymbol{\phi}$, a deterministic vector.
Then $\|\mathbf{r}-\mathbf{w}\|_2^2=\|\mathbf{r}\|_2^2-2\operatorname{Re}(\mathbf{r}^H\mathbf{w})+\|\mathbf{w}\|_2^2$.
Since $\mathbb{E}[\mathbf{w}]=\mathbf{0}$ the cross term vanishes, and
$\mathbb{E}\|\mathbf{w}\|_2^2=\operatorname{tr}(\sigma^2\mathbf{I}_{N_r})=N_r\sigma^2$. $\blacksquare$

### Lemma 2 (linearization in $\boldsymbol{\phi}$)

$\mathbf{\Phi}\mathbf{H}_1\mathbf{s}=\operatorname{diag}(\mathbf{H}_1\mathbf{s})\boldsymbol{\phi}$, hence

$$
\mathbf{H}_2\mathbf{\Phi}\mathbf{H}_1\mathbf{s}=\mathbf{A}(\mathbf{s})\boldsymbol{\phi},
\qquad
\mathbf{A}(\mathbf{s})\triangleq\mathbf{H}_2\operatorname{diag}(\mathbf{H}_1\mathbf{s})\in\mathbb{C}^{N_r\times M},
\qquad
[\mathbf{A}(\mathbf{s})]_{:,m}=(\mathbf{g}_{1,m}^T\mathbf{s})\,\mathbf{h}_{2,m},
$$

where $\mathbf{g}_{1,m}^T$ is the $m$-th row of $\mathbf{H}_1$ and $\mathbf{h}_{2,m}$ the $m$-th
column of $\mathbf{H}_2$.

*Proof.* Componentwise, $[\mathbf{\Phi}\mathbf{H}_1\mathbf{s}]_m=\phi_m[\mathbf{H}_1\mathbf{s}]_m
=[\mathbf{H}_1\mathbf{s}]_m\phi_m=[\operatorname{diag}(\mathbf{H}_1\mathbf{s})\boldsymbol{\phi}]_m$;
both $\mathbf{\Phi}$ and $\operatorname{diag}(\mathbf{H}_1\mathbf{s})$ are diagonal, so the product
is symmetric in which factor carries the variable. The column formula follows from
$\mathbf{A}\mathbf{e}_m=\mathbf{H}_2\mathbf{e}_m[\mathbf{H}_1\mathbf{s}]_m$. $\blacksquare$

Combining Lemmas 1 and 2 and writing $\mathbf{A}\equiv\mathbf{A}(\mathbf{s})$,

$$
\boxed{\ \mathrm{MSE}_{\min}(\mathbf{s})=N_r\sigma^2+\mathcal{E}(\mathbf{s}),
\qquad
\mathcal{E}(\mathbf{s})\triangleq\min_{\boldsymbol{\phi}\in\mathbb{T}^M}\|\mathbf{y}-\mathbf{A}\boldsymbol{\phi}\|_2^2 .\ }
\tag{2.1}
$$

The minimum in (2.1) is attained: $\mathbb{T}^M$ is compact and the objective continuous.
Everything below bounds the **synthesis residual** $\mathcal{E}(\mathbf{s})$; the additive
$N_r\sigma^2$ is carried along and must not be dropped.

Define the Hermitian PSD **equivalent Gram operator** and its spectral data:

$$
\mathbf{R}_{\mathrm{eq}}\triangleq\mathbf{A}\mathbf{A}^H=\mathbf{H}_2\operatorname{diag}\!\big(|\mathbf{H}_1\mathbf{s}|^2\big)\mathbf{H}_2^H
=\sum_{k=1}^{N_r}\lambda_k\,\mathbf{u}_k\mathbf{u}_k^H,
\qquad
y_k\triangleq\mathbf{u}_k^H\mathbf{y},
\qquad
\mathrm{SNR}_k\triangleq\frac{\lambda_k}{\sigma^2}.
\tag{2.2}
$$

$\{\mathbf{u}_k\}$ is an orthonormal basis of $\mathbb{C}^{N_r}$, so
$\sum_k|y_k|^2=\|\mathbf{y}\|_2^2$ (Parseval).

---

## 3. Part I — exact analysis of the norm-relaxed problem

This part replaces Steps 1–3 of the draft derivation. It is stated in full because the
draft's Step 3 substitution $\mu\propto\sigma^2$ is not licensed by Step 2; see
Corollary 2 and the Errata.

### Lemma 3 (relaxation)

$\mathbb{T}^M\subset\mathcal{S}_M\triangleq\{\boldsymbol{\phi}:\|\boldsymbol{\phi}\|_2^2\le M\}$, hence

$$
\mathcal{E}(\mathbf{s})\ \ge\ \mathcal{E}_{\mathrm{rel}}\triangleq\min_{\|\boldsymbol{\phi}\|_2^2\le M}\|\mathbf{y}-\mathbf{A}\boldsymbol{\phi}\|_2^2 .
$$

*Proof.* $\boldsymbol{\phi}\in\mathbb{T}^M\Rightarrow\|\boldsymbol{\phi}\|_2^2=\sum_m|\phi_m|^2=M$.
Minimizing over a superset cannot increase the minimum. $\blacksquare$

### Theorem 1 (exact value of the relaxed problem)

Let $h(\mu)\triangleq\sum_{k=1}^{N_r}\dfrac{\lambda_k|y_k|^2}{(\lambda_k+\mu)^2}$ for $\mu>0$, and
$h(0^+)\triangleq\sum_{k:\lambda_k>0}\dfrac{|y_k|^2}{\lambda_k}\in[0,+\infty]$. Let
$\mathbf{P}_{\mathbf{A}}$ be the orthogonal projector onto $\operatorname{range}(\mathbf{A})
=\operatorname{span}\{\mathbf{u}_k:\lambda_k>0\}$ and $\mathbf{P}^\perp_{\mathbf{A}}=\mathbf{I}-\mathbf{P}_{\mathbf{A}}$. Then:

1. If $h(0^+)\le M$, then $\ \mathcal{E}_{\mathrm{rel}}=\|\mathbf{P}^\perp_{\mathbf{A}}\mathbf{y}\|_2^2
   =\sum_{k:\lambda_k=0}|y_k|^2$, attained at $\boldsymbol{\phi}^\star=\mathbf{A}^{+}\mathbf{y}$ (KKT multiplier $\mu^\star=0$).
2. Otherwise there is a unique $\mu^\star>0$ solving $h(\mu^\star)=M$, and

$$
\mathcal{E}_{\mathrm{rel}}
=\sum_{k=1}^{N_r}\left(\frac{\mu^\star}{\lambda_k+\mu^\star}\right)^{2}|y_k|^2 ,
\qquad
\boldsymbol{\phi}^\star=\big(\mathbf{A}^H\mathbf{A}+\mu^\star\mathbf{I}_M\big)^{-1}\mathbf{A}^H\mathbf{y}.
$$

*Proof.* The problem minimizes a convex quadratic over the convex set $\mathcal{S}_M$;
$\boldsymbol{\phi}=\mathbf{0}$ is strictly feasible, so Slater's condition holds and the KKT
conditions are necessary and sufficient. With
$\mathcal{L}(\boldsymbol{\phi},\mu)=\|\mathbf{y}-\mathbf{A}\boldsymbol{\phi}\|_2^2+\mu(\|\boldsymbol{\phi}\|_2^2-M)$,
Wirtinger stationarity $\nabla_{\boldsymbol{\phi}^*}\mathcal{L}
=\mathbf{A}^H(\mathbf{A}\boldsymbol{\phi}-\mathbf{y})+\mu\boldsymbol{\phi}=\mathbf{0}$ gives, for $\mu>0$,

$$
\boldsymbol{\phi}^\star(\mu)=(\mathbf{A}^H\mathbf{A}+\mu\mathbf{I}_M)^{-1}\mathbf{A}^H\mathbf{y}.
$$

*Push-through identity.* From $\mathbf{A}^H(\mathbf{A}\mathbf{A}^H+\mu\mathbf{I}_{N_r})
=(\mathbf{A}^H\mathbf{A}+\mu\mathbf{I}_M)\mathbf{A}^H$, left-multiplying by
$(\mathbf{A}^H\mathbf{A}+\mu\mathbf{I}_M)^{-1}$ and right-multiplying by
$(\mathbf{A}\mathbf{A}^H+\mu\mathbf{I}_{N_r})^{-1}$ yields
$(\mathbf{A}^H\mathbf{A}+\mu\mathbf{I})^{-1}\mathbf{A}^H=\mathbf{A}^H(\mathbf{A}\mathbf{A}^H+\mu\mathbf{I})^{-1}$
(both inverses exist for $\mu>0$). Hence

$$
\mathbf{A}\boldsymbol{\phi}^\star(\mu)=\mathbf{R}_{\mathrm{eq}}(\mathbf{R}_{\mathrm{eq}}+\mu\mathbf{I})^{-1}\mathbf{y},
\qquad
\mathbf{y}-\mathbf{A}\boldsymbol{\phi}^\star(\mu)=\mu(\mathbf{R}_{\mathrm{eq}}+\mu\mathbf{I})^{-1}\mathbf{y}
=\sum_k\frac{\mu}{\lambda_k+\mu}y_k\mathbf{u}_k,
\tag{3.1}
$$

and by orthonormality
$\|\mathbf{y}-\mathbf{A}\boldsymbol{\phi}^\star(\mu)\|_2^2=\sum_k\big(\tfrac{\mu}{\lambda_k+\mu}\big)^2|y_k|^2$.
Likewise

$$
\|\boldsymbol{\phi}^\star(\mu)\|_2^2
=\mathbf{y}^H(\mathbf{R}_{\mathrm{eq}}+\mu\mathbf{I})^{-1}\mathbf{R}_{\mathrm{eq}}(\mathbf{R}_{\mathrm{eq}}+\mu\mathbf{I})^{-1}\mathbf{y}
=h(\mu).
\tag{3.2}
$$

$h$ is continuous and strictly decreasing on $(0,\infty)$ whenever
$\sum_k\lambda_k|y_k|^2>0$, with $h(\mu)\to0$ as $\mu\to\infty$ and $h(\mu)\uparrow h(0^+)$ as
$\mu\downarrow0$. Complementary slackness requires either $\mu^\star=0$ with
$\|\boldsymbol{\phi}^\star\|_2^2\le M$ — possible exactly when $h(0^+)\le M$, in which case the
minimum-norm least-squares solution $\mathbf{A}^+\mathbf{y}$ is feasible and the residual is the
projection residual $\mathbf{P}^\perp_{\mathbf{A}}\mathbf{y}$ — or $\mu^\star>0$ with
$h(\mu^\star)=M$, which by strict monotonicity has a unique root when $h(0^+)>M$.
(If $\mathbf{A}^H\mathbf{y}=\mathbf{0}$ then $h\equiv0\le M$, case 1 applies and
$\mathcal{E}_{\mathrm{rel}}=\|\mathbf{y}\|_2^2$.) $\blacksquare$

### Corollary 1 (a monotone family of valid bounds; weak duality)

Define $G(\mu)\triangleq\sum_k\big(\tfrac{\mu}{\lambda_k+\mu}\big)^2|y_k|^2$ and
$D(\mu)\triangleq\sum_k\tfrac{\mu}{\lambda_k+\mu}|y_k|^2-\mu M$. Then:

1. $G$ is nondecreasing on $[0,\infty)$, and $\ \mathcal{E}(\mathbf{s})\ge\mathcal{E}_{\mathrm{rel}}=G(\mu^\star)\ge G(\mu)$
   **for every $0\le\mu\le\mu^\star$, and only for those $\mu$.**
2. **(Unconditional.)** For every $\mu\ge0$, $\ \mathcal{E}(\mathbf{s})\ge\mathcal{E}_{\mathrm{rel}}\ge D(\mu)$, and
   $\sup_{\mu\ge0}D(\mu)=\mathcal{E}_{\mathrm{rel}}$.

*Proof.* (1) Each map $\mu\mapsto\frac{\mu}{\lambda_k+\mu}=\frac{1}{1+\lambda_k/\mu}$ is
nondecreasing in $\mu\ge0$ and nonnegative, so $G$ is nondecreasing; combine with
Theorem 1 and Lemma 3.
(2) For feasible $\boldsymbol{\phi}$ and $\mu\ge0$ we have $\mu(\|\boldsymbol{\phi}\|_2^2-M)\le0$, so
$\|\mathbf{y}-\mathbf{A}\boldsymbol{\phi}\|_2^2\ge\mathcal{L}(\boldsymbol{\phi},\mu)\ge\min_{\boldsymbol{\phi}'}\mathcal{L}(\boldsymbol{\phi}',\mu)=:d(\mu)$.
Evaluating the unconstrained minimum with (3.1)–(3.2),

$$
d(\mu)=\|\mathbf{y}\|_2^2-\mathbf{y}^H\mathbf{R}_{\mathrm{eq}}(\mathbf{R}_{\mathrm{eq}}+\mu\mathbf{I})^{-1}\mathbf{y}-\mu M
=\sum_k\Big(1-\tfrac{\lambda_k}{\lambda_k+\mu}\Big)|y_k|^2-\mu M
=D(\mu).
$$

Strong duality holds by Slater, giving $\sup_\mu D(\mu)=\mathcal{E}_{\mathrm{rel}}$. $\blacksquare$

Part 2 is the honest replacement for the draft's Step 3: it is valid for *every* $\mu\ge0$,
requires no assumption, and at the natural choice $\mu=\sigma^2$ reads

$$
\mathcal{E}(\mathbf{s})\ \ge\ \sum_{k=1}^{N_r}\frac{|\mathbf{u}_k^H\mathbf{y}|^2}{1+\mathrm{SNR}_k}\ -\ M\sigma^2 .
\tag{3.3}
$$

Note the exponent $1$ (not $2$) and the $-M\sigma^2$ penalty; both are forced by duality.

### Corollary 2 (when the modal-SNR form $\sum_k|y_k|^2/(1+\mathrm{SNR}_k)^2$ is a valid bound)

$G(\sigma^2)=\sum_k\frac{|y_k|^2}{(1+\mathrm{SNR}_k)^2}$ lower-bounds $\mathcal{E}(\mathbf{s})$
**if and only if** $\sigma^2\le\mu^\star$, i.e. iff

$$
h(\sigma^2)=\sum_{k=1}^{N_r}\frac{\lambda_k|y_k|^2}{(\lambda_k+\sigma^2)^2}\ \ge\ M.
\tag{C}
$$

If (C) fails then $\mu^\star<\sigma^2$ and $G(\sigma^2)\ge\mathcal{E}_{\mathrm{rel}}$: the
expression is then an *upper* bound on the relaxed optimum and says nothing about
$\mathcal{E}(\mathbf{s})$.

*Proof.* Immediate from Corollary 1(1), Theorem 1 and strict monotonicity of $h$
(so $\sigma^2\le\mu^\star\iff h(\sigma^2)\ge h(\mu^\star)=M$). $\blacksquare$

> **Remark (why this matters).** Using $\lambda_{k\ge2}=\Theta(M\alpha^2/\kappa)$ from
> Proposition 8 and $Y=\sum_{k\ge2}|y_k|^2$, condition (C) holds only in the **intermediate
> band**
> $$\frac{M\alpha^2}{Y}\ \lesssim\ \kappa\ \lesssim\ \frac{M\alpha^2}{\sigma^2}.$$
> Below it the problem is well conditioned and $h(\sigma^2)\approx\sum_k|y_k|^2/\lambda_k\ll M$;
> above it $\lambda_{k\ge2}\ll\sigma^2$ and $h(\sigma^2)\approx\sigma^{-4}\sum_k\lambda_k|y_k|^2\to0$.
> So (C) fails at **both** ends, and in particular in the $\kappa\to\infty$ limit the draft uses
> it for. Section 9 exhibits explicit numerical violations at both ends. The conclusion
> survives, but it must be reached by a different route — Parts II and III.

> **Remark (two different $\mu$'s).** The $\mu$ in Theorem 1 is a KKT multiplier for the
> *power* constraint, pinned by $h(\mu^\star)=M$. Setting $\mu=\sigma^2$ is instead a
> Tikhonov/LMMSE *design* choice for a regularized estimator. They coincide only
> accidentally; identifying them is not a proof step.

---

## 4. Part II — the unconditional geometric floor

The relaxation of Part I is not needed for the headline result. The following holds for
*arbitrary* $\boldsymbol{\phi}\in\mathbb{C}^M$, unconstrained.

### Theorem 2 (range floor, and exactness in the pure-LoS limit)

Let $\mathbf{P}^\perp_{\mathbf{A}}$ project onto $\operatorname{range}(\mathbf{A}(\mathbf{s}))^\perp$. Then

$$
\mathcal{E}(\mathbf{s})\ \ge\ \inf_{\boldsymbol{\phi}\in\mathbb{C}^M}\|\mathbf{y}-\mathbf{A}\boldsymbol{\phi}\|_2^2
=\big\|\mathbf{P}^\perp_{\mathbf{A}}\mathbf{y}\big\|_2^2 ,
\qquad
\operatorname{range}(\mathbf{A}(\mathbf{s}))=\operatorname{span}\{\mathbf{h}_{2,m}:[\mathbf{H}_1\mathbf{s}]_m\ne0\}\subseteq\operatorname{range}(\mathbf{H}_2).
\tag{4.1}
$$

Moreover, in the **pure-LoS case** $\varepsilon=0$ (i.e. $\kappa=\infty$), with
$\mathbf{P}^\perp\triangleq\mathbf{I}_{N_r}-\frac{\mathbf{a}_{\mathrm{rx}}\mathbf{a}_{\mathrm{rx}}^H}{N_r}$:

$$
\mathcal{E}(\mathbf{s})\ \ge\ \|\mathbf{P}^\perp\mathbf{y}\|_2^2
=\|\mathbf{y}\|_2^2-\frac{|\mathbf{a}_{\mathrm{rx}}^H\mathbf{y}|^2}{N_r},
\tag{4.2}
$$

with **equality** whenever $\dfrac{|\mathbf{a}_{\mathrm{rx}}^H\mathbf{y}|}{N_r}\le M\alpha$.

*Proof.* The first display is the definition of the distance from $\mathbf{y}$ to the subspace
$\operatorname{range}(\mathbf{A})$, attained by orthogonal projection; the inequality holds because
$\mathbb{T}^M\subset\mathbb{C}^M$. The range identity follows from Lemma 2: the $m$-th column of
$\mathbf{A}$ is $[\mathbf{H}_1\mathbf{s}]_m\mathbf{h}_{2,m}$, which spans the same line as
$\mathbf{h}_{2,m}$ when $[\mathbf{H}_1\mathbf{s}]_m\ne0$ and vanishes otherwise.

For $\varepsilon=0$ we have $\mathbf{H}_1=\mathbf{a}_{\mathrm{ris},1}\mathbf{a}_{\mathrm{tx}}^H$ and
$\mathbf{H}_2=\mathbf{a}_{\mathrm{rx}}\mathbf{a}_{\mathrm{ris},2}^H$, so
$\mathbf{H}_1\mathbf{s}=(\mathbf{a}_{\mathrm{tx}}^H\mathbf{s})\mathbf{a}_{\mathrm{ris},1}$, whose
entries all have modulus $\alpha>0$ by (A1) and (A3). Hence

$$
\mathbf{A}(\mathbf{s})
=(\mathbf{a}_{\mathrm{tx}}^H\mathbf{s})\,\mathbf{a}_{\mathrm{rx}}\,\mathbf{a}_{\mathrm{ris},2}^H\operatorname{diag}(\mathbf{a}_{\mathrm{ris},1})
\tag{4.3}
$$

has rank exactly $1$ with $\operatorname{range}(\mathbf{A})=\operatorname{span}\{\mathbf{a}_{\mathrm{rx}}\}$,
giving (4.2).

For exactness, (4.3) gives
$\mathbf{A}\boldsymbol{\phi}=(\mathbf{a}_{\mathrm{tx}}^H\mathbf{s})\,c(\boldsymbol{\phi})\,\mathbf{a}_{\mathrm{rx}}$ with
$c(\boldsymbol{\phi})=\sum_{m=1}^{M}\overline{[\mathbf{a}_{\mathrm{ris},2}]_m}\,[\mathbf{a}_{\mathrm{ris},1}]_m\,\phi_m$.
Each summand has unit modulus and, as $\phi_m$ ranges over the unit circle, arbitrary
phase; therefore $\{c(\boldsymbol{\phi}):\boldsymbol{\phi}\in\mathbb{T}^M\}$ is exactly the closed disk of
radius $M$. (Containment in the disk is the triangle inequality. Conversely, fix a target
phase $\theta$; pairing the elements as $e^{i(\theta\pm\omega)}$ realizes
$2\cos\omega\,e^{i\theta}$ per pair, so for even $M$ every modulus in $[0,M]$ is attainable,
and for odd $M$ the same holds using $\lfloor M/2\rfloor$ pairs plus one free element,
whose reachable set is $\{((M-1)\cos\omega+1)e^{i\theta}\}\supseteq[0,M]e^{i\theta}$.)
The residual for the achievable set $\{z\,\mathbf{a}_{\mathrm{rx}}:|z|\le M\alpha\}$ is minimized
by the orthogonal projection $z^\star=\mathbf{a}_{\mathrm{rx}}^H\mathbf{y}/N_r$, which is feasible
exactly when $|\mathbf{a}_{\mathrm{rx}}^H\mathbf{y}|/N_r\le M\alpha$, the stated condition.
Then $\mathcal{E}(\mathbf{s})=\|\mathbf{P}^\perp\mathbf{y}\|_2^2$. $\blacksquare$

> **Interpretation.** Under pure LoS the RIS controls exactly **one complex scalar**: the
> received vector is locked to the direction $\mathbf{a}_{\mathrm{rx}}$ no matter what
> $\mathbf{s}$ or $\boldsymbol{\phi}$ are. This is the formal content of `README.md` §7: a
> decoder downstream sees a one-dimensional signal and collapses to a constant decision.

> **Caveat.** At any finite $\kappa$ with $M\ge N_r$, $\mathbf{A}(\mathbf{s})$ has full row rank
> almost surely, so (4.1) degenerates to $0$. The finite-$\kappa$ statement therefore cannot
> be a rank statement — it must be a **conditioning** statement. That is Part III.

### Corollary 3 (cosine-similarity / AGC form)

The active code path matches $\boldsymbol{\phi}$ by cosine similarity and rescales
$\mathbf{y}_{\mathrm{ris}}$ to $\|\mathbf{y}\|_2$ (AGC). For that objective,

$$
\max_{\boldsymbol{\phi}\in\mathbb{T}^M}
\frac{|\langle\mathbf{y},\mathbf{A}\boldsymbol{\phi}\rangle|}{\|\mathbf{y}\|_2\,\|\mathbf{A}\boldsymbol{\phi}\|_2}
\ \le\
\frac{\|\mathbf{P}_{\mathbf{A}}\mathbf{y}\|_2}{\|\mathbf{y}\|_2}
\ \xrightarrow[\ \varepsilon\to0\ ]{}\
\frac{|\mathbf{a}_{\mathrm{rx}}^H\mathbf{y}|}{\sqrt{N_r}\,\|\mathbf{y}\|_2},
$$

and the AGC-matched squared error $\min_{c>0}\|\mathbf{y}-c\,\mathbf{A}\boldsymbol{\phi}\|_2^2
=\|\mathbf{y}\|_2^2(1-\cos^2\theta)$ obeys the same floor $\|\mathbf{P}^\perp_{\mathbf{A}}\mathbf{y}\|_2^2$.

*Proof.* Cauchy–Schwarz applied to $\langle\mathbf{y},\mathbf{v}\rangle=\langle\mathbf{P}_{\mathbf{A}}\mathbf{y},\mathbf{v}\rangle$
for any $\mathbf{v}\in\operatorname{range}(\mathbf{A})$; the scale-optimized residual identity is
$\min_c\|\mathbf{y}-c\hat{\mathbf{v}}\|_2^2=\|\mathbf{y}\|_2^2-|\hat{\mathbf{v}}^H\mathbf{y}|^2$ for unit $\hat{\mathbf{v}}$. $\blacksquare$

### Corollary 4 (digital pre- and post-processing does not escape the floor)

Let
$\mathbf{W}_{\mathrm{phys}}=\mathbf{U}^H\mathbf{H}_2\mathbf{\Phi}\mathbf{H}_1\mathbf{P}_{\mathrm{tx}}$
for arbitrary $\mathbf{P}_{\mathrm{tx}}\in\mathbb{C}^{N_t\times N_t}$,
$\mathbf{U}\in\mathbb{C}^{N_r\times N_r}$ and $\boldsymbol{\phi}\in\mathbb{T}^M$.
At $\varepsilon=0$, $\operatorname{rank}(\mathbf{W}_{\mathrm{phys}})\le 1$, and therefore for any target
$\mathbf{W}\in\mathbb{C}^{N_r\times N_t}$,

$$
\frac{\|\mathbf{W}_{\mathrm{phys}}-\mathbf{W}\|_F}{\|\mathbf{W}\|_F}
\ \ge\
\sqrt{1-\frac{\sigma_1(\mathbf{W})^2}{\|\mathbf{W}\|_F^2}},
\tag{4.4}
$$

where $\sigma_1(\mathbf{W})$ is the largest singular value of $\mathbf{W}$.

*Proof.* From (4.3),
$\mathbf{H}_2\mathbf{\Phi}\mathbf{H}_1=c(\boldsymbol{\phi})\,\mathbf{a}_{\mathrm{rx}}\mathbf{a}_{\mathrm{tx}}^H$
is rank one for every $\boldsymbol{\phi}\in\mathbb{T}^M$. Rank does not increase under left
multiplication by $\mathbf{U}^H$ or right multiplication by $\mathbf{P}_{\mathrm{tx}}$, so
$\operatorname{rank}(\mathbf{W}_{\mathrm{phys}})\le 1$. The displayed bound is the Eckart–Young–Mirsky
theorem in the Frobenius norm: the distance from $\mathbf{W}$ to the set of rank-at-most-one
matrices is $\sqrt{\|\mathbf{W}\|_F^2-\sigma_1(\mathbf{W})^2}$. $\blacksquare$

> This is the AirFC fitting problem $\mathbf{U}^H\mathbf{H}_2\operatorname{diag}(\boldsymbol{\phi})\mathbf{H}_1\mathbf{P}\approx\mathbf{W}$
> in the pure-LoS limit. Extra digital matrices $\mathbf{P}_{\mathrm{tx}}$ and $\mathbf{U}$ do not
> restore the lost receive dimensions. (PDF Corollary 2.)

---

## 5. Part III — finite $\kappa$: a non-asymptotic quantitative floor

The key structural fact is that the LoS component of $\mathbf{H}_2$ is annihilated *exactly*
by $\mathbf{P}^\perp$, so everything outside the beam direction is $\mathcal{O}(\sqrt{\varepsilon})$.

### Lemma 4 (exact annihilation)

$\mathbf{P}^\perp\mathbf{H}_2=\sqrt{\varepsilon}\,\mathbf{P}^\perp\tilde{\mathbf{H}}_2$, hence
$\|\mathbf{P}^\perp\mathbf{H}_2\|_2\le\sqrt{\varepsilon}\,s_{\max}(\tilde{\mathbf{H}}_2)$.

*Proof.* $\mathbf{P}^\perp\mathbf{a}_{\mathrm{rx}}=\mathbf{a}_{\mathrm{rx}}-\mathbf{a}_{\mathrm{rx}}\frac{\mathbf{a}_{\mathrm{rx}}^H\mathbf{a}_{\mathrm{rx}}}{N_r}=\mathbf{0}$
by (A1), so the rank-one LoS term of $\mathbf{H}_2$ is killed. Projectors are contractions. $\blacksquare$

### Lemma 5 (incident-power concentration)

Let $z_m\triangleq\tilde{\mathbf{g}}_{1,m}^T\mathbf{s}$. Then $z_m\stackrel{\text{iid}}{\sim}\mathcal{CN}(0,\beta^2)$ and

$$
\big|[\mathbf{H}_1\mathbf{s}]_m\big|\ \le\ \sqrt{1-\varepsilon}\,\alpha+\sqrt{\varepsilon}\,|z_m|,
\qquad
\big|[\mathbf{H}_1\mathbf{s}]_m\big|\ \ge\ \sqrt{1-\varepsilon}\,\alpha-\sqrt{\varepsilon}\,|z_m| .
$$

For any $\delta\in(0,1)$, with probability at least $1-\delta$,
$\ Z\triangleq\max_m|z_m|\le\beta\sqrt{\ln(M/\delta)}$.

*Proof.* $[\mathbf{H}_1\mathbf{s}]_m=\sqrt{1-\varepsilon}(\mathbf{a}_{\mathrm{tx}}^H\mathbf{s})[\mathbf{a}_{\mathrm{ris},1}]_m+\sqrt{\varepsilon}z_m$
and $|[\mathbf{a}_{\mathrm{ris},1}]_m|=1$; apply the triangle inequality both ways. Rows of
$\tilde{\mathbf{H}}_1$ are i.i.d. $\mathcal{CN}(\mathbf{0},\mathbf{I}_{N_t})$, so
$z_m\sim\mathcal{CN}(0,\|\mathbf{s}\|_2^2)$ and $|z_m|/\beta$ is Rayleigh with
$\Pr(|z_m|/\beta>t)=e^{-t^2}$; a union bound over $m$ gives $\Pr(Z>\beta t)\le Me^{-t^2}$, and
$t=\sqrt{\ln(M/\delta)}$ closes it. $\blacksquare$

### Theorem 3 (non-asymptotic floor at finite $\kappa$)

Let $\mathbf{P}^\perp=\mathbf{I}_{N_r}-\mathbf{a}_{\mathrm{rx}}\mathbf{a}_{\mathrm{rx}}^H/N_r$. Then deterministically,

$$
\mathcal{E}(\mathbf{s})\ \ge\
\Big(\big\|\mathbf{P}^\perp\mathbf{y}\big\|_2-\sqrt{M}\,\big\|\mathbf{P}^\perp\mathbf{A}(\mathbf{s})\big\|_2\Big)_+^2,
\qquad
\big\|\mathbf{P}^\perp\mathbf{A}(\mathbf{s})\big\|_2\le\sqrt{\varepsilon}\,s_{\max}(\tilde{\mathbf{H}}_2)\,\max_m\big|[\mathbf{H}_1\mathbf{s}]_m\big| .
\tag{5.1}
$$

Consequently, for any $\delta\in(0,1)$, with probability at least $1-\delta$ over the NLoS draw,

$$
\boxed{\;
\mathrm{MSE}_{\min}(\mathbf{s})\ \ge\ N_r\sigma^2+
\Bigg(\big\|\mathbf{P}^\perp\mathbf{y}\big\|_2-
\underbrace{\sqrt{\frac{M}{\kappa+1}}\Big(\sqrt{M}+\sqrt{N_r}+\tau\Big)\Big(\alpha+\tfrac{\beta}{\sqrt{\kappa+1}}\sqrt{\ln\tfrac{2M}{\delta}}\Big)}_{\textstyle \Delta(\kappa,M,\delta)}
\Bigg)_+^2
\;}
\tag{5.2}
$$

with $\tau=\sqrt{2\ln(2/\delta)}$. In particular, for fixed $M,N_r,\mathbf{s},\mathbf{y}$,

$$
\Delta(\kappa,M,\delta)=\mathcal{O}\!\left(\frac{M\alpha}{\sqrt{\kappa}}\right)
\quad\text{and}\quad
\liminf_{\kappa\to\infty}\ \mathcal{E}(\mathbf{s})\ \ge\ \big\|\mathbf{P}^\perp\mathbf{y}\big\|_2^2 ,
$$

recovering Theorem 2 continuously. The floor is non-vacuous as soon as
$\kappa+1\gtrsim M^2\alpha^2/\|\mathbf{P}^\perp\mathbf{y}\|_2^2$.

*Proof.* Fix any $\boldsymbol{\phi}\in\mathbb{T}^M$. Since $\mathbf{P}^\perp$ is an orthogonal projector,
$\|\mathbf{r}\|_2\ge\|\mathbf{P}^\perp\mathbf{r}\|_2$, so with $\mathbf{r}=\mathbf{y}-\mathbf{A}\boldsymbol{\phi}$,

$$
\|\mathbf{y}-\mathbf{A}\boldsymbol{\phi}\|_2
\ \ge\ \|\mathbf{P}^\perp\mathbf{y}-\mathbf{P}^\perp\mathbf{A}\boldsymbol{\phi}\|_2
\ \ge\ \|\mathbf{P}^\perp\mathbf{y}\|_2-\|\mathbf{P}^\perp\mathbf{A}\boldsymbol{\phi}\|_2
\ \ge\ \|\mathbf{P}^\perp\mathbf{y}\|_2-\|\mathbf{P}^\perp\mathbf{A}\|_2\,\|\boldsymbol{\phi}\|_2,
$$

using the reverse triangle inequality and the definition of the spectral norm. On the
torus $\|\boldsymbol{\phi}\|_2=\sqrt{M}$ exactly. Taking positive parts and squaring (both sides
nonnegative) and minimizing over $\boldsymbol{\phi}$ gives the first part of (5.1). For the second
part, $\mathbf{P}^\perp\mathbf{A}=\mathbf{P}^\perp\mathbf{H}_2\operatorname{diag}(\mathbf{H}_1\mathbf{s})$, and
$\|\mathbf{X}\mathbf{D}\|_2\le\|\mathbf{X}\|_2\|\mathbf{D}\|_2$ with
$\|\operatorname{diag}(\mathbf{H}_1\mathbf{s})\|_2=\max_m|[\mathbf{H}_1\mathbf{s}]_m|$; apply Lemma 4.

For (5.2): by the Gaussian concentration bound for the largest singular value of a matrix
with i.i.d. standard complex Gaussian entries (Davidson–Szarek; see also Vershynin,
Cor. 5.35), $\Pr\big(s_{\max}(\tilde{\mathbf{H}}_2)>\sqrt{M}+\sqrt{N_r}+\tau\big)\le e^{-\tau^2/2}$;
choose $\tau=\sqrt{2\ln(2/\delta)}$ so this has probability $\le\delta/2$. By Lemma 5 with
$\delta/2$, $\max_m|[\mathbf{H}_1\mathbf{s}]_m|\le\sqrt{1-\varepsilon}\,\alpha+\sqrt{\varepsilon}\beta\sqrt{\ln(2M/\delta)}
\le\alpha+\sqrt{\varepsilon}\beta\sqrt{\ln(2M/\delta)}$ with probability $\ge1-\delta/2$. A union
bound and $\sqrt{\varepsilon}=1/\sqrt{\kappa+1}$ give (5.2); add $N_r\sigma^2$ via (2.1).
The rate follows since $\sqrt{M}(\sqrt{M}+\sqrt{N_r}+\tau)=\Theta(M)$ for $M\ge N_r$. $\blacksquare$

> Theorem 3 needs no eigendecomposition, no relaxation, and no random-matrix *lower*
> bound — only the standard operator-norm upper bound. It is strictly stronger and simpler
> than the modal-SNR route, and it is the statement to cite.

### 5.1 Spectrum of $\mathbf{R}_{\mathrm{eq}}$ (the corrected Step 4)

For completeness — and because the modal picture is useful for intuition and for
evaluating (3.3) — here is the rigorous version of the eigenvalue scaling.

**Lemma 6 (congruence sandwich).** Let $\mathbf{D}=\operatorname{diag}(|\mathbf{H}_1\mathbf{s}|^2)$ and
$\mathbf{G}=\mathbf{H}_2\mathbf{H}_2^H$. If $d_{\min}\mathbf{I}\preceq\mathbf{D}\preceq d_{\max}\mathbf{I}$ then
$d_{\min}\lambda_k(\mathbf{G})\le\lambda_k(\mathbf{R}_{\mathrm{eq}})\le d_{\max}\lambda_k(\mathbf{G})$ for all $k$.

*Proof.* $\mathbf{D}-d_{\min}\mathbf{I}\succeq0\Rightarrow
\mathbf{H}_2(\mathbf{D}-d_{\min}\mathbf{I})\mathbf{H}_2^H\succeq0\Rightarrow\mathbf{R}_{\mathrm{eq}}\succeq d_{\min}\mathbf{G}$,
and Weyl monotonicity ($\mathbf{X}\succeq\mathbf{Y}\Rightarrow\lambda_k(\mathbf{X})\ge\lambda_k(\mathbf{Y})$)
applies; symmetrically for $d_{\max}$. By Lemma 5,
$d_{\min},d_{\max}=(1-\varepsilon)\alpha^2\big(1\pm\mathcal{O}(\sqrt{\varepsilon\ln(M/\delta)}\,\beta/\alpha)\big)^2$.
$\blacksquare$

**Lemma 7 (spectrum of $\mathbf{G}$).** Write $\mathbf{G}=\varepsilon\mathbf{W}+\mathbf{N}$ with
$\mathbf{W}=\tilde{\mathbf{H}}_2\tilde{\mathbf{H}}_2^H$ and
$\mathbf{N}=(1-\varepsilon)M\,\mathbf{a}_{\mathrm{rx}}\mathbf{a}_{\mathrm{rx}}^H
+\sqrt{\varepsilon(1-\varepsilon)}\big(\mathbf{a}_{\mathrm{rx}}\mathbf{b}^H+\mathbf{b}\mathbf{a}_{\mathrm{rx}}^H\big)$,
$\mathbf{b}=\tilde{\mathbf{H}}_2\mathbf{a}_{\mathrm{ris},2}$, using
$\mathbf{a}_{\mathrm{ris},2}^H\mathbf{a}_{\mathrm{ris},2}=M$. Then $\operatorname{rank}(\mathbf{N})\le2$ and,
for $N_r\ge3$,

$$
\varepsilon\,\lambda_{k+1}(\mathbf{W})\ \le\ \lambda_k(\mathbf{G})\ \le\ \varepsilon\,\lambda_{k-1}(\mathbf{W}),
\qquad 2\le k\le N_r-1 .
$$

*Proof.* $\mathbf{N}$ is supported on $\operatorname{span}\{\mathbf{a}_{\mathrm{rx}},\mathbf{b}\}$, so
$\operatorname{rank}(\mathbf{N})\le2$; in the orthonormal basis $\{\mathbf{u},\mathbf{v}\}$ of that span
with $\mathbf{u}=\mathbf{a}_{\mathrm{rx}}/\sqrt{N_r}$, its nonzero block is
$\left[\begin{smallmatrix}a&c\\ \bar c&0\end{smallmatrix}\right]$ with determinant $-|c|^2\le0$,
so $\mathbf{N}$ has at most one positive and at most one negative eigenvalue; with $N_r\ge3$
this forces $\lambda_2(\mathbf{N})=\lambda_{N_r-1}(\mathbf{N})=0$. Weyl's inequalities
$\lambda_{i+j-1}(\mathbf{X}+\mathbf{Y})\le\lambda_i(\mathbf{X})+\lambda_j(\mathbf{Y})$ with $(i,j)=(k-1,2)$ and
$\lambda_{i+j-N_r}(\mathbf{X}+\mathbf{Y})\ge\lambda_i(\mathbf{X})+\lambda_j(\mathbf{Y})$ with $(i,j)=(k+1,N_r-1)$
give the claim. $\blacksquare$

**Proposition 8 (eigenvalue scaling).** With probability $\ge1-\delta$, for $M\ge N_r\ge3$ and
$\tau=\sqrt{2\ln(4/\delta)}$:

$$
\lambda_1(\mathbf{R}_{\mathrm{eq}})=\Big(\tfrac{\kappa}{\kappa+1}\Big)^{2}M N_r\,\alpha^2\,(1+o(1)),
\qquad
\lambda_k(\mathbf{R}_{\mathrm{eq}})=c_k\,\frac{\kappa\,M\,\alpha^2}{(\kappa+1)^2}\ \ (2\le k\le N_r-1),
$$

where $c_k\in\big[(1-\sqrt{N_r/M}-\tau/\sqrt M)^2,\ (1+\sqrt{N_r/M}+\tau/\sqrt M)^2\big]\cdot(1+o(1))$ and the
$o(1)$ terms are $\mathcal{O}(\sqrt{\varepsilon\ln(M/\delta)}\,\beta/\alpha)+\mathcal{O}(\sqrt{N_r/M})$.

*Proof.* Combine Lemmas 6 and 7 with the two-sided Wishart edge
$\big(\sqrt M-\sqrt{N_r}-\tau\big)^2\le\lambda_j(\mathbf{W})\le\big(\sqrt M+\sqrt{N_r}+\tau\big)^2$
for all $j$ (Davidson–Szarek, probability $\ge1-2e^{-\tau^2/2}$), and
$d_{\min},d_{\max}\to(1-\varepsilon)\alpha^2$ from Lemma 5. For $\lambda_1$, the Rayleigh quotient
at $\mathbf{u}=\mathbf{a}_{\mathrm{rx}}/\sqrt{N_r}$ gives
$\lambda_1(\mathbf{G})\ge(1-\varepsilon)MN_r-2\sqrt{\varepsilon N_r}|\mathbf{u}^H\mathbf{b}|$ and Weyl gives
$\lambda_1(\mathbf{G})\le\lambda_1(\mathbf{N})+\varepsilon\lambda_1(\mathbf{W})$; both match
$(1-\varepsilon)MN_r$ to relative order $\mathcal{O}(\sqrt{\varepsilon/N_r}+\varepsilon)$. Multiply by
$d_{\min/\max}\to(1-\varepsilon)\alpha^2$. $\blacksquare$

> **Correction.** The draft asserts $\lambda_{k\ge2}\approx c_kM/(\kappa+1)^2$. The correct
> scaling carries one factor $\kappa$ from the *transmit-side LoS power* that illuminates the
> surface and the factor $\alpha^2$ from the input:
> $\lambda_{k\ge2}=\Theta\!\big(\kappa M\alpha^2/(\kappa+1)^2\big)=\Theta\!\big(M\alpha^2/\kappa\big)$,
> i.e. decay $1/\kappa$, **not** $1/\kappa^2$. Only the NLoS component of $\mathbf{H}_2$ leaks
> into $\mathbf{a}_{\mathrm{rx}}^\perp$; the $\mathbf{H}_1$ side contributes its *LoS* power
> $\tfrac{\kappa}{\kappa+1}\alpha^2$ there, not its NLoS power.

---

## 6. Assembled result

Collecting (2.1), Corollary 1(2), Theorem 3 and Proposition 8, with probability $\ge1-\delta$:

$$
\mathrm{MSE}_{\min}(\mathbf{s})\ \ge\ N_r\sigma^2+\max\Bigg\{
\underbrace{\Big(\|\mathbf{P}^\perp\mathbf{y}\|_2-\Delta(\kappa,M,\delta)\Big)_+^2}_{\text{Theorem 3 — geometric}},\ \
\underbrace{\sup_{\mu\ge0}\Big[\sum_{k}\frac{\mu\,|\mathbf{u}_k^H\mathbf{y}|^2}{\lambda_k+\mu}-\mu M\Big]}_{\text{Corollary 1 — spectral}}
\Bigg\}
\tag{6.1}
$$

with $\lambda_1\approx\big(\tfrac{\kappa}{\kappa+1}\big)^2MN_r\alpha^2$ and
$\lambda_{k\ge2}\approx c_k\kappa M\alpha^2/(\kappa+1)^2$. A convenient closed form for the
spectral branch: dropping the $k=1$ term and bounding $\lambda_{k\ge2}\le\bar\lambda$,

$$
\sup_{\mu\ge0}\Big[\frac{\mu\,Y}{\bar\lambda+\mu}-\mu M\Big]
=\Big(\sqrt{Y}-\sqrt{\bar\lambda M}\Big)_+^2,
\qquad
Y\triangleq\sum_{k\ge2}|\mathbf{u}_k^H\mathbf{y}|^2 ,
\tag{6.2}
$$

attained at $\mu=\sqrt{Y\bar\lambda/M}-\bar\lambda$. With
$\bar\lambda=\Theta(M\alpha^2/\kappa)$ this reads
$\big(\sqrt Y-\Theta(M\alpha/\sqrt\kappa)\big)_+^2$ — the same $\mathcal{O}(M\alpha/\sqrt\kappa)$
rate as Theorem 3, confirming the two routes agree.

**High-LoS limit.** Taking $\kappa\to\infty$ at fixed $M,N_r,\sigma^2$, $\Delta\to0$ and
$\mathbf{u}_1\to\mathbf{a}_{\mathrm{rx}}/\sqrt{N_r}$ (Davis–Kahan, since the spectral gap
$\lambda_1-\lambda_2=\Theta(MN_r\alpha^2)$ dominates the off-diagonal coupling
$\mathcal{O}(\sqrt{\varepsilon M}N_r)$), so

$$
\boxed{\;
\lim_{\kappa\to\infty}\mathrm{MSE}_{\min}(\mathbf{s})\ \ge\ N_r\sigma^2+
\Big\|\Big(\mathbf{I}_{N_r}-\tfrac{\mathbf{a}_{\mathrm{rx}}\mathbf{a}_{\mathrm{rx}}^H}{N_r}\Big)\mathbf{y}\Big\|_2^2
=N_r\sigma^2+\|\mathbf{y}\|_2^2-|\mathbf{u}_1^H\mathbf{y}|^2 .
\;}
\tag{6.3}
$$

No secondary limit $\mathrm{SNR}_1\to\infty$ is needed: the geometric floor is a
statement about the *range* of $\mathbf{A}(\mathbf{s})$ and is independent of $\sigma^2$, which
enters only through the separate additive $N_r\sigma^2$.

**Consequence for the article.** If the target is $\mathbf{y}=\mathbf{W}_{\mathrm{lin}}\mathbf{s}$ for a
full-rank $\mathbf{W}_{\mathrm{lin}}\in\mathbb{C}^{N_r\times N_t}$ and $\mathbf{s}$ ranges over a
distribution that is not concentrated on
$\mathbf{W}_{\mathrm{lin}}^{-1}\operatorname{span}\{\mathbf{a}_{\mathrm{rx}}\}$, then
$\mathbb{E}_{\mathbf{s}}\|\mathbf{P}^\perp\mathbf{W}_{\mathrm{lin}}\mathbf{s}\|_2^2
=\Theta\big(\mathbb{E}\|\mathbf{W}_{\mathrm{lin}}\mathbf{s}\|_2^2\big)$ for $N_r\ge2$: a
LoS-dominated link cannot carry a full-rank linear map, at any SNR, with any number of RIS
elements. Rich scattering (low or negative $K$-factor, `geometric_rayleigh`) is what makes
$\|\mathbf{P}^\perp_{\mathbf{A}}\mathbf{y}\|_2=0$ possible.

---

## 7. Sharpness and limitations

* **The relaxation gap.** Lemma 3 is a genuine relaxation for $2\le\operatorname{rank}(\mathbf{A})$;
  Theorem 2 shows it is *not* lossy in the rank-1 regime, where the reachable set
  $\{\mathbf{A}\boldsymbol{\phi}:\boldsymbol{\phi}\in\mathbb{T}^M\}$ is the full disk
  $\{z\mathbf{a}_{\mathrm{rx}}:|z|\le M\alpha\}$. At finite $\kappa$ the gap is unquantified here;
  standard SDR arguments would give a constant-factor guarantee.
* **Theorem 3 is one-sided.** It bounds what the RIS *cannot* do. It does not certify that a
  particular optimizer (e.g. the repo's cosine-similarity gradient descent
  `_optimize_phi_gd`) approaches the bound.
* **Per-input statement.** All bounds are for a fixed $\mathbf{s}$ and a fixed channel
  realization. Averaging over a data distribution requires $\mathbf{a}_{\mathrm{tx}}^H\mathbf{s}\ne0$
  a.s. and an integrable $\beta/\alpha$.
* **Joint scaling.** The $o(1)$ terms in Proposition 8 need $\varepsilon\ln M\ll(\alpha/\beta)^2$,
  i.e. $\kappa\gtrsim\ln M\cdot(\beta/\alpha)^2$; a regime with $M\to\infty$ and $\kappa$ fixed is
  *not* covered.
* **Path loss** is omitted (it rescales $\mathbf{H}_1,\mathbf{H}_2$ and hence all $\lambda_k$ and
  $\Delta$ by a common factor; the geometric floor (4.2) is unaffected).

---

## 8. Errata relative to the draft derivation

| # | Draft step | Issue | Fix |
|---|---|---|---|
| E1 | Step 3, "matching the regularization scale to the noise variance, $\mu\propto\sigma^2$" | Conflates the KKT multiplier for the **power** constraint (pinned by $h(\mu^\star)=M$) with a Tikhonov **design** parameter. $G(\sigma^2)$ lower-bounds the optimum only if $\mu^\star\ge\sigma^2$, i.e. condition (C) — which **fails** in the high-LoS regime the result is applied to, making the chain invalid exactly where it is used. | Theorem 1 (exact $\mu^\star$), Corollary 1 (valid family, incl. the unconditional dual bound (3.3)), Corollary 2 (precise validity condition). |
| E2 | Step 4, $\lambda_{k\ge2}\approx c_kM/(\kappa+1)^2$ | Off by a factor $\kappa$ and missing $\alpha^2$. Only $\mathbf{H}_2$'s NLoS part leaks into $\mathbf{a}_{\mathrm{rx}}^\perp$; the $\mathbf{H}_1$ side contributes its **LoS** power $\tfrac{\kappa}{\kappa+1}\alpha^2$. Correct: $\Theta(\kappa M\alpha^2/(\kappa+1)^2)=\Theta(M\alpha^2/\kappa)$. | Proposition 8. Changes the approach rate to the floor from $\mathcal{O}(M/\kappa^2)$ to $\mathcal{O}(M\alpha/\sqrt\kappa)$. |
| E3 | Step 4, $\mathcal{O}(1/\sqrt{\kappa+1})$ remainders | Remainders are random matrices/vectors; "$\approx$" hides a $\max$ over $M$ meta-atoms (a $\sqrt{\ln M}$ factor) and an operator-norm bound that grows with $M$. Never controlled. | Lemmas 4–7 give explicit high-probability, non-asymptotic control. |
| E4 | Final displays | Drop the additive $N_r\sigma^2$ from Step 1, and state the floor only under a double limit $\kappa\to\infty,\mathrm{SNR}_1\to\infty$. | (2.1) keeps $N_r\sigma^2$ throughout; (6.3) needs no SNR limit. |
| E5 | Steps 1–3 as the route to the floor | The spectral route is unnecessary for the headline claim. | Theorem 2 gets the floor from a one-line range argument, **with equality** in the pure-LoS case; Theorem 3 makes it quantitative at finite $\kappa$ with elementary tools. |
| E6 | $\mathbf{R}_{\mathrm{eq}}$ spectral split | The displayed split writes the scattered term as $\tfrac{1}{(\kappa+1)^2}\tilde{\mathbf{R}}_{\mathrm{scat}}$ and silently drops the rank-2 LoS$\times$NLoS cross term, whose spectral norm $\Theta(\sqrt{\varepsilon M}N_r)$ dominates $\varepsilon M$ once $\kappa\gtrsim M/N_r^2$. | Lemma 7 handles it exactly via rank-2 Weyl interlacing rather than discarding it. |

---

## 9. Numerical verification

`python theory/verify_mse_lower_bound.py` (seed 0, $N_t=N_r=4$). The torus optimum is
computed by multi-restart Adam on the phases, which yields an *achievable* value and hence
an **upper** bound on $\mathcal{E}(\mathbf{s})$ — so any claimed lower bound exceeding it is
conclusively invalid.

**Theorem 1** ($M=16$): the closed form matches projected gradient descent on the relaxed
problem, with GD above it in every case as required (gap $+9.5\!\cdot\!10^{-5}$,
$+2.1\!\cdot\!10^{-4}$, $+9.0\!\cdot\!10^{-3}$ at $\kappa=0.1,10,10^3$; the first two are the
$\mu^\star=0$ case where the exact optimum is $0$).

**Corollary 2** ($M=32$, $\sigma^2=10^{-3}$): the draft's $G(\sigma^2)$ against the achievable
torus value, showing violations exactly where (C) fails at both ends of the band:

| $\kappa$ | $h(\sigma^2)$ vs $M=32$ | (C) | draft $G(\sigma^2)$ | achievable $\mathcal{E}(\mathbf{s})\le$ | verdict |
|---|---|---|---|---|---|
| $10^{-1}$ | $0.078$ | fails | $1.36\cdot10^{-9}$ | $8.2\cdot10^{-30}$ | **violated** |
| $10^{1}$ | $1.10$ | fails | $3.21\cdot10^{-7}$ | $8.6\cdot10^{-31}$ | **violated** |
| $10^{2}$ | $1.75$ | fails | $1.03\cdot10^{-6}$ | $9.7\cdot10^{-25}$ | **violated** |
| $10^{3}$ | $620$ | holds | $1.123$ | $2.373$ | valid |
| $10^{5}$ | $1146$ | holds | $0.845$ | $4.118$ | valid |
| $10^{9}$ | $1.88$ | fails | $3.696950$ | $3.696759$ | **violated** |

**Proposition 8** ($M=128$, 24 trials/point): $\lambda_1$ divided by the predicted
$(\tfrac{\kappa}{\kappa+1})^2MN_r\alpha^2$ converges to $1.000$. The ratio
$\lambda_2(\kappa+1)^2/(\kappa M\alpha^2)$ stabilizes at $\approx1.15$ — inside the predicted
$c_k$ interval $[(1-\sqrt{N_r/M})^2,(1+\sqrt{N_r/M})^2]=[0.68,1.39]$ — while the draft's
$\lambda_2(\kappa+1)^2/M$ grows linearly in $\kappa$ ($11\to4.2\!\cdot\!10^{5}$ over
$\kappa\in[1,10^5]$). The fitted log-log slope of $\lambda_2$ versus $\kappa$ is
$\mathbf{-1.017}$, against $-1$ predicted here and $-2$ in the draft.

**Theorem 3** ($M=32$): the bound (5.1) never exceeds the achievable value, is vacuous
($=0$) for $\kappa\lesssim10^3$ as the condition
$\kappa+1\gtrsim M^2\alpha^2/\|\mathbf{P}^\perp\mathbf{y}\|_2^2\approx10^{3.5}$ predicts, and
tightens to within $3\%$ of the floor by $\kappa=10^{7}$ (bound $4.769$, achievable $4.895$,
floor $\|\mathbf{P}^\perp\mathbf{y}\|_2^2=4.931$).

---

## 10. Symbol table

| Symbol | Meaning |
|---|---|
| $M\equiv N_m$, $N_t$, $N_r$ | RIS elements, Tx antennas, Rx antennas |
| $\mathbf{s},\mathbf{y}$ | input latent, target latent ($=\mathbf{W}_{\mathrm{lin}}\mathbf{s}$ in the article) |
| $\boldsymbol{\phi}\in\mathbb{T}^M$ | unit-modulus RIS profile |
| $\mathbf{A}(\mathbf{s})=\mathbf{H}_2\operatorname{diag}(\mathbf{H}_1\mathbf{s})$ | input-conditioned equivalent channel |
| $\mathbf{R}_{\mathrm{eq}}=\mathbf{A}\mathbf{A}^H$, $\lambda_k$, $\mathbf{u}_k$, $y_k$ | Gram operator, its spectrum, target coordinates |
| $\varepsilon=1/(\kappa+1)$ | NLoS power fraction |
| $\alpha=|\mathbf{a}_{\mathrm{tx}}^H\mathbf{s}|$, $\beta=\|\mathbf{s}\|_2$ | beam-aligned and total input amplitude |
| $\mathbf{P}^\perp=\mathbf{I}-\mathbf{a}_{\mathrm{rx}}\mathbf{a}_{\mathrm{rx}}^H/N_r$ | projector off the Rx beam direction |
| $\mu^\star$, $h(\mu)$, $G(\mu)$, $D(\mu)$ | KKT multiplier, $\|\boldsymbol{\phi}^\star(\mu)\|^2$, primal-at-$\mu$, dual function |
| $\mathcal{E}(\mathbf{s})$ | synthesis residual, $\mathrm{MSE}_{\min}=N_r\sigma^2+\mathcal{E}$ |
| $\mathbf{W}_{\mathrm{phys}}=\mathbf{U}^H\mathbf{H}_2\mathbf{\Phi}\mathbf{H}_1\mathbf{P}_{\mathrm{tx}}$ | digitally pre-/post-processed physical map (Corollary 4; AirFC) |
| $\sigma_1(\mathbf{W})$ | largest singular value of the target matrix $\mathbf{W}$ |

---

## 11. Appendix: draft derivation as printed in the PDF (sections 9-10)

> **Not a result.** PDF §§9–10 are headed "Gemini: System Model and Problem
> Formulation". They restate §1 and then give the derivation that
> [§8](#8-errata-relative-to-the-draft-derivation) rejects. Kept here so the
> draft can be quoted against the PDF equation numbers. Do not cite (A.6)–(A.8)
> as bounds.

**§9 restates the model.** Forward model (PDF (26)), linearization
$\mathbf{y}_{\mathrm{synth}}=\mathbf{A}(\mathbf{s})\boldsymbol{\phi}+\mathbf{w}$ with
$\mathbf{A}(\mathbf{s})=\mathbf{H}_2\operatorname{diag}(\mathbf{H}_1\mathbf{s})$ (PDF (27)–(29)),
Rician factorization (PDF (30)–(32)), and $\mathrm{MSE}_{\min}$ (PDF (33)).
These match §1 upon identifying $\sqrt{\kappa/(\kappa+1)}=\sqrt{1-\varepsilon}$ and
$\sqrt{1/(\kappa+1)}=\sqrt{\varepsilon}$. Column formula (PDF (29)) is Lemma 2.

**§10.1–10.2 are the relaxation, done correctly.** Noise separation (PDF (34))
is Lemma 1. The torus sits inside the ball $\|\boldsymbol{\phi}\|_2^2\le M$ (PDF (35)–(36)),
which is Lemma 3. The Lagrangian (PDF (38)), stationarity, and

$$
\boldsymbol{\phi}^\star(\mu)=\big(\mathbf{A}^H\mathbf{A}+\mu\mathbf{I}_M\big)^{-1}\mathbf{A}^H\mathbf{y},
\qquad
\mathbf{y}-\mathbf{A}\boldsymbol{\phi}^\star(\mu)=\mu\,(\mathbf{R}_{\mathrm{eq}}+\mu\mathbf{I})^{-1}\mathbf{y}
\tag{A.1}
$$

are Theorem 1's stationary point (PDF (40), (43)). The Gram is
$\mathbf{R}_{\mathrm{eq}}=\mathbf{H}_2\operatorname{diag}(|\mathbf{H}_1\mathbf{s}|^2)\mathbf{H}_2^H$
(PDF (42)). The modal expansion of the *relaxed* residual at a chosen $\mu$ is

$$
\|\mathbf{y}-\mathbf{A}\boldsymbol{\phi}^\star(\mu)\|_2^2
=\sum_{k=1}^{N_r}\left(\frac{\mu}{\lambda_k+\mu}\right)^2|\mathbf{u}_k^H\mathbf{y}|^2 .
\tag{A.2}
$$

This is $G(\mu)$ from Corollary 1. It equals $\mathcal{E}_{\mathrm{rel}}$ only at
$\mu=\mu^\star$ with $h(\mu^\star)=M$ (or $\mu^\star=0$); the draft never imposes that.

**§10.3 is erratum E1.** The draft sets $\mu\propto\sigma^2$, defines
$\mathrm{SNR}_k=\lambda_k/\sigma^2$, and claims (PDF (47)–(48))

$$
\|\mathbf{y}-\mathbf{A}(\mathbf{s})\boldsymbol{\phi}\|_2^2
\ \ge\
\sum_{k=1}^{N_r}\frac{|\mathbf{u}_k^H\mathbf{y}|^2}{(1+\mathrm{SNR}_k)^2}.
\tag{A.3}
$$

(A.3) is $G(\sigma^2)$. By Corollary 2 it lower-bounds $\mathcal{E}(\mathbf{s})$ if and
only if condition (C), $h(\sigma^2)\ge M$, which fails at both ends of the
$\kappa$ range, including $\kappa\to\infty$. The legal unconditional substitute
is $D(\mu)$ (Corollary 1(2), PDF (19) in the corrected half): exponent $1$ and a
$-\mu M$ penalty, not (A.3).

**§10.4 is erratum E2 (and E6).** The draft's dominant eigenvalue (PDF (56))

$$
\lambda_1\approx\Big(\frac{\kappa}{\kappa+1}\Big)^2 M N_r\,|\mathbf{a}_{\mathrm{tx}}^H\mathbf{s}|^2
\tag{A.4}
$$

matches Proposition 8. The scattered eigenvalues do not. The draft prints
(PDF (58))

$$
\lambda_k\approx\frac{c_k M}{(\kappa+1)^2},\qquad k\ge 2,
\qquad c_k=\mathcal{O}(1),
\tag{A.5}
$$

dropping both the transmit-side LoS factor $\kappa$ and $\alpha^2$. Proposition 8
replaces (A.5) by $\lambda_k=c_k\,\kappa M\alpha^2/(\kappa+1)^2=\Theta(M\alpha^2/\kappa)$.
The displayed split (PDF (55)) also writes the scattered term as
$\tilde{\mathbf{R}}_{\mathrm{scat}}/(\kappa+1)^2$ and drops the rank-2
LoS$\times$NLoS cross term (erratum E6; Lemma 7).

**§10.5–10.6 are the draft's final claim (errata E2, E4).** Substituting (A.4)
and (A.5) into (A.3) gives the draft bound (PDF (60))

$$
\mathrm{MSE}_{\min}(\mathbf{s})
\ \ge\
\frac{|\mathbf{u}_1^H\mathbf{y}|^2}
{\Big(1+\big(\tfrac{\kappa}{\kappa+1}\big)^2 M N_r|\mathbf{a}_{\mathrm{tx}}^H\mathbf{s}|^2\cdot\mathrm{SNR}_0\Big)^2}
+\sum_{k=2}^{N_r}
\frac{|\mathbf{u}_k^H\mathbf{y}|^2}
{\Big(1+\dfrac{c_k M}{(\kappa+1)^2}\cdot\mathrm{SNR}_0\Big)^2},
\tag{A.6}
$$

with $\mathrm{SNR}_0=1/\sigma^2$. Sending $\kappa\to\infty$ inside (A.6) produces
(PDF (61))

$$
\lim_{\kappa\to\infty}\mathrm{MSE}_{\min}(\mathbf{s})
\ \ge\
\sum_{k=2}^{N_r}|\mathbf{u}_k^H\mathbf{y}|^2
+\frac{|\mathbf{u}_1^H\mathbf{y}|^2}{\big(1+M N_r|\mathbf{a}_{\mathrm{tx}}^H\mathbf{s}|^2\cdot\mathrm{SNR}_0\big)^2},
\tag{A.7}
$$

and a further limit $\mathrm{SNR}_1\to\infty$ drops the second term, leaving
(PDF (62))

$$
\lim_{\kappa\to\infty,\;\mathrm{SNR}_1\to\infty}\mathrm{MSE}_{\min}(\mathbf{s})
\ \ge\
\|\mathbf{y}\|_2^2-|\mathbf{u}_1^H\mathbf{y}|^2
=\Big\|\Big(\mathbf{I}_{N_r}-\frac{\mathbf{a}_{\mathrm{rx}}\mathbf{a}_{\mathrm{rx}}^H}{\|\mathbf{a}_{\mathrm{rx}}\|_2^2}\Big)\mathbf{y}\Big\|_2^2 .
\tag{A.8}
$$

(A.8) is the right geometric floor, reached by an invalid route and under a
stronger hypothesis than necessary: it drops the additive $N_r\sigma^2$ from
Lemma 1, and it invokes $\mathrm{SNR}_1\to\infty$. The corrected statement is
(6.3), which needs only $\kappa\to\infty$ at fixed $M,N_r,\sigma^2$ and keeps
$N_r\sigma^2$. The finite-$\kappa$ quantitative bound to cite is Theorem 3
(PDF Theorem 2), rate $\mathcal{O}(M\alpha/\sqrt{\kappa})$, not (A.6).

---
## 12. Experimental validation against the CIFAR RIS pipeline

`§9` verifies the theorems on the paper's own synthetic model. This section
records what the trained **image pipeline** shows when the learned bias-free layer
`W_lin` is synthesized over the geometric 28 GHz channel of `channels.py`, and — as
important — corrects two subtle errors in an earlier version of this experiment
that made a finite-K "floor" look fundamental when it is not.

### 12.0 The correction (read this first)

An earlier run plotted a "torus optimum" that saturated at the Theorem 2 floor
`||P_perp y||^2` at every large K and concluded "no phi, no N_m, no SNR can beat
the floor at finite K." **That is wrong.** Two independent solvers (a per-sample
Levenberg–Marquardt feasibility solver, `_lm_feasible_phi`, and an external check)
show the *free-scale* synthesis residual is `~1e-12` at every finite K:

| K (dB) | Adam-from-random (old curve) | LM feasibility (true free optimum) |
|---:|---:|---:|
| 30 | 0.83–1.0 | 2.7e-12 |
| 40 | 0.83 | 1.0e-12 |
| 50–70 | 0.8–1.0 | ~1e-14 |

The reason is structural: at any finite K, `A(s) = H_2 diag(H_1 s)` has full row
rank `N_r`, and the unit-modulus system `A(s) phi = c y` is wildly
underdetermined (`M = 64` phases for `2 N_r = 16` real equations), so it is
generically **feasible** — a torus `phi` reproduces `y` exactly, and the free scale
`c` absorbs the amplitude. The old curve was **Adam-from-random settling into the
strong-LoS basin**, not the optimum. Theorem 2 (exact rank-1) holds only at
`epsilon = 0` (`K = infinity`); Theorem 3 bounds the *fixed-scale* residual, not
the free-scale one the pipeline (with AGC) actually realizes.

### 12.1 What the finite-K floor really is: a K–SNR tradeoff

The exact-synthesis solution is not free — it must **null the strong LoS beam** and
route `y` through the weak scattered path, and that costs received power. Measured,
the cost is clean:

```
received power of the beat-the-floor solution  ≈  −(K + 10 log10 N_r) dB   relative to the LoS beam
```

(measured −38.8, −49.8 dB at K = 30, 40 dB vs the predicted 39.0, 49.0 dB). So with
an **absolute** noise floor and nominal SNR `S` dB, the effective SNR of the
exact-synthesis solution is `S − K − 10 log10 N_r` dB. The physical problem the
receiver actually solves is

```
min_{phi in T^M, g}  ||y - g A(s) phi||^2 + g^2 N_r sigma^2 ,
```

— Theorem 3 bounds the first term at each fixed `g`; the second term (receiver-gain
noise) is what makes nulling worthless. The floor therefore **binds only once
`K ≳ S − 10 log10 N_r`**: below that, exact synthesis is both feasible and
noise-affordable (error → 0); above it, the only affordable `phi` is the LoS beam,
whose residual is exactly `||P_perp y||^2` (Theorem 2). This K–SNR tradeoff, not an
unconditional floor, is the finite-K result.

### 12.2 What the sweep plots

`framework/cifar_minimal_dnn.py --mse_sweep` (K in **dB**, see 12.4). All curves are
NMSE `= sum||y_t - hat y||^2 / sum||y_t||^2`, on a shared image subset:

| Curve | Definition | Meaning |
|---|---|---|
| achieved (AGC) | post-noise, norm-matched student (decoder input) | the cosine+AGC heuristic |
| achieved (LS) | best complex scale for the pipeline's `phi` | heuristic, best-scaled |
| free-scale optimum | `min_{|phi|=1,c} ||y_t - c A phi||^2` via `_lm_feasible_phi` + cosine candidate | best achievable, **noiseless**: ~0 at finite K |
| physical optimum | `min_phi [ ||y_t||^2 - |<A phi,y_t>|^2 / (||A phi||^2 + N_r sigma^2) ]`, opt-in `--mse_abs_snr` | the K–SNR tradeoff floor |
| Thm 2 floor | `||P_perp y_t||^2`, `P_perp = I - a_rx a_rx^H / N_r` | the `K → infinity` (rank-1) limit |

`a_rx` is reconstructed from the fixed geometry by
`channels.los_rx_steering_vector` (checked: `||P_perp H_2||/||H_2|| → 0` as K grows).
The **free-scale optimum ≈ 0 at finite K** is the honest headline: the RIS *can*
synthesize `W_lin s`; the earlier "it can't" was an optimizer artifact. What
actually limits it is the K–SNR tradeoff, visible only with `--mse_abs_snr`, where
the **physical optimum** rises to the Theorem 2 floor as `K` passes
`S − 10 log10 N_r`.

> The g-scaled "Theorem 3 curve" from the earlier version was **removed**: with `g`
> taken from the optimizer's own `phi` it was circular (a LoS-nulling `phi` has huge
> `g`, giving 0). Theorem 3 remains correct as a *fixed-scale* statement in §5; it is
> just not what the free-scale pipeline realizes, so it is not plotted.

### 12.3 What the classification-accuracy collapse actually is

The kappa accuracy sweep shows accuracy falling (e.g. 43% → 20%) as K rises. Given
12.0–12.1, this is **not** a fundamental channel floor at those K. Two causes:

1. **`_optimize_phi_gd` is 100 Adam steps from `randn`**, which lands in the
   LoS-aligned basin — the same failure the "torus optimum" had. A stronger phi
   solver would recover much of the mid-K regime.
2. **`noise()` is relative** (`sigma^2 = mean|y_ris|^2 / SNR`), so nulling the beam
   costs nothing in-simulation — the sim cannot represent the K–SNR tradeoff at all.
   The genuine degradation only appears under an absolute noise floor
   (`--mse_abs_snr`), and even then only for `K ≳ SNR − 10 log10 N_r`.

So the honest statement is: **the RIS is fundamentally floored only in the strong-LoS
/ low-effective-SNR corner; elsewhere the observed degradation is the phi-optimizer
and the relative-noise model, not the channel.**

### 12.4 Modeling notes / gotchas

1. **K is in dB.** `make_ris_channel_pools` passes the sweep value as
   `k_factor_h{1,2}_db`, so `50` = 50 dB = linear `K = 1e5`; the wide default is
   `0..70 dB`. Passing a large *linear* value overflows `10**(k_db/10)`.
2. **Scale-resolution is mandatory.** `_optimize_phi_gd` uses a scale-invariant
   cosine loss, so only the post-AGC (or LS-scaled) student can be compared to the
   floor.
3. **Absolute-noise reference.** `--mse_abs_snr` fixes `sigma^2` from the pipeline's
   mean received power at the *lowest* K in the sweep (the rich-scattering operating
   point) and holds it constant, so the K–SNR tradeoff is a fair comparison.
4. **Path loss is commented out** in `channels.py` (`#TODO(pl)`); it also changes the
   accuracy/AirFC/SimNet numbers, and makes `apply_pathloss` a dead flag repo-wide
   until restored.
5. `||P_perp y||^2 ≈ (N_r-1)/N_r` holds only in expectation over isotropic `y`.
6. **The plotted optima are the best of two candidate `phi`** — the LM feasibility
   solution and the pipeline's cosine `phi` (`_torus_optimum_nmse`) — not a global
   search, so they are *achievable* values (upper bounds on the true optima). This is
   tight enough for the argument: the feasibility candidate drives the free-scale
   optimum to ~0, and the two candidates bracket the physical optimum's two regimes
   (exact-synthesis at low K, LoS-beam at high K). At finite K the physical optimum can
   sit slightly *below* `||P_perp y||^2` because that floor is the `K -> infinity` limit.
