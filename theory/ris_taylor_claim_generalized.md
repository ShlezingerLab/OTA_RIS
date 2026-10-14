# RIS Approximation of a General DNN via First-Order Taylor Expansion

**Claim and proof.** Over-the-air realization of a trained feedforward network
$f=f_L\circ\cdots\circ f_1$ by a reconfigurable intelligent surface (RIS) cascade
$H_\phi(s)=H_2\operatorname{diag}(\phi)H_1\,s+\beta$.

This generalizes `ris_taylor_claim.md` (single layer $f(s)=\sigma(Ws+b)$) to any
depth. The key observation is that **only the Taylor term uses the one-layer
structure**: the RIS realizes a *single linear map* in $s$, so Steps 0, 2, 3, 4 of
the original proof see only the end-to-end Jacobian $F$ and the fixed channels
$H_1,H_2$ — they never look inside $f$ and transfer verbatim. The entire
generalization lives in replacing the scalar second-derivative bound by a
**composition Hessian bound**.

---

## Setup

* Input $s\in\mathcal S\subseteq\mathbb C^{d}$ random with law $\mu$, mean $\mu_s$,
  covariance $\Sigma_s=\mathbb E[(s-\mu_s)(s-\mu_s)^{\mathsf H}]$.
* For functions $g:\mathcal S\to\mathbb C^{m}$, the Hilbert norm
  $\lVert g\rVert^2:=\mathbb E_{s\sim\mu}\lVert g(s)\rVert_2^2$.
* Network $f=f_L\circ\cdots\circ f_1$ with layers
  $f_\ell(x)=\sigma_\ell(W_\ell x+b_\ell)$,
  $W_\ell\in\mathbb C^{d_\ell\times d_{\ell-1}}$, $d_0=d$, $d_L=m$,
  $\sigma_\ell$ applied entrywise and holomorphic on the region of interest.
* Reference point $s_0$; end-to-end Jacobian $F=Df(s_0)\in\mathbb C^{m\times d}$;
  first-order model $T(s)=f(s_0)+F(s-s_0)$.
  *(The simulation code calls this matrix $G$.)*
* RIS cascade $H_\phi(s)=H(\phi)\,s+\beta$ with
  $H(\phi)=H_2\operatorname{diag}(\phi)H_1\in\mathbb C^{m\times d}$,
  $H_1\in\mathbb C^{N\times d}$, $H_2\in\mathbb C^{m\times N}$, bias / direct path $\beta$.
* $\|D^2f(s')\|_{\mathrm{op}}:=\sup_{\lVert u\rVert=1}\lVert D^2f(s')[u,u]\rVert_2$,
  the operator norm of the symmetric bilinear second derivative.

### Assumptions

* **(A1) Smooth activations.** Each $\sigma_\ell$ is holomorphic (so $C^2$) on an
  open set containing the pre-activations reached on the region of interest. This
  **excludes ReLU** (its second derivative is distributional / unbounded) and admits
  tanh, (logistic) sigmoid, softplus, GELU, SiLU.
* **(A2) Bounded derivatives.** Over that region,
  $B_1:=\max_\ell\sup|\sigma_\ell'|<\infty$ and
  $B_2:=\max_\ell\sup|\sigma_\ell''|<\infty$.
  (Holomorphic $\sigma$ is bounded on compact regions away from its poles; for the
  logistic sigmoid the poles sit at $z=i\pi(2k{+}1)$, so the region of interest must
  avoid them — the same caveat as the single-layer claim.)
* **(A3) Locality.** The input law concentrates near $s_0$, quantified by
  $m_4:=(\mathbb E_\mu\lVert s-s_0\rVert^4)^{1/2}$; equivalently restrict attention to a
  ball $\mathcal B(s_0,r)$ carrying most of the mass.

---

## Theorem

$$
\lVert f-H_\phi\rVert\ \le\ \varepsilon_T+\varepsilon_H,\qquad
\varepsilon_T\le \tfrac12\,C_2^{\mathrm{tot}}\,m_4,\quad
\varepsilon_H=\big\lVert (H(\phi)-F)\,\Sigma_s^{1/2}\big\rVert_F,
$$

where $\varepsilon_H$ is attained at the optimal bias $\beta$, and the end-to-end
curvature constant obeys the **composition bound**

$$
C_2^{\mathrm{tot}}\ :=\ \sup_{s'\in\mathcal B(s_0,r)}\lVert D^2f(s')\rVert_{\mathrm{op}}
\ \le\ \sum_{\ell=1}^{L}\Big(\prod_{j=\ell+1}^{L}a_j\Big)\,b_\ell\,
\Big(\prod_{j=1}^{\ell-1}a_j\Big)^{2},
\qquad a_\ell=B_1\lVert W_\ell\rVert_2,\ \ b_\ell=B_2\lVert W_\ell\rVert_2^2 .
$$

Moreover, with
$\mathcal U:=\{H:\operatorname{col}(H)\subseteq\operatorname{col}(H_2),\
\operatorname{row}(H)\subseteq\operatorname{row}(H_1)\}$,

$$
\min_\phi\varepsilon_H\ \ge\ \operatorname{dist}_{\Sigma_s}(F,\mathcal U)
:=\min_{H\in\mathcal U}\lVert(H-F)\Sigma_s^{1/2}\rVert_F ,
$$

with equality to $\lVert(F-P_2FP_1)\Sigma_s^{1/2}\rVert_F$ when $\Sigma_s\propto I$
($P_2,P_1$ the orthogonal projectors onto $\operatorname{col}(H_2),\operatorname{row}(H_1)$);
and $\varepsilon_H=0$ is achievable when $N\ge md$ with
$\operatorname{rank}H_1=d,\ \operatorname{rank}H_2=m$.

For $L=1$ this reduces to the single-layer claim, since
$C_2^{\mathrm{tot}}\le B_2\lVert W\rVert_2^2=C_2\lVert W\rVert_2^2$.

---

## Proof

### Step 0 — split

$L^2(\mu)$ is a Hilbert space, so the triangle inequality gives

$$
\lVert f-H_\phi\rVert\le\lVert f-T\rVert+\lVert T-H_\phi\rVert=:\varepsilon_T+\varepsilon_H .
$$

*(Identical to the single-layer proof.)*

### Step 1 — Taylor term $\varepsilon_T$

**(a) Pointwise remainder.** Fix $s$, let $u=s-s_0$ and
$\gamma(t)=s_0+tu$, $t\in[0,1]$ (the segment lies in the convex ball
$\mathcal B(s_0,r)$). Put $g(t)=f(\gamma(t))$; then
$g'(t)=Df(\gamma(t))[u]$ and $g''(t)=D^2f(\gamma(t))[u,u]$, both well defined by (A1).
Taylor's theorem with integral remainder (valid along the holomorphic segment):

$$
f(s)=g(1)=g(0)+g'(0)+\int_0^1(1-t)\,g''(t)\,dt
=f(s_0)+F u+\int_0^1(1-t)\,D^2f(\gamma(t))[u,u]\,dt .
$$

Hence $f(s)-T(s)=\int_0^1(1-t)\,D^2f(\gamma(t))[u,u]\,dt$, and since
$\int_0^1(1-t)\,dt=\tfrac12$,

$$
\lVert f(s)-T(s)\rVert
\le\tfrac12\Big(\sup_{t}\lVert D^2f(\gamma(t))\rVert_{\mathrm{op}}\Big)\lVert u\rVert^2
\le\tfrac12\,C_2^{\mathrm{tot}}\,\lVert s-s_0\rVert^2 .
$$

Taking $\mathbb E_\mu$ and $\sqrt{\cdot}$:

$$
\varepsilon_T=\big(\mathbb E_\mu\lVert f-T\rVert^2\big)^{1/2}
\le\tfrac12\,C_2^{\mathrm{tot}}\big(\mathbb E_\mu\lVert s-s_0\rVert^4\big)^{1/2}
=\tfrac12\,C_2^{\mathrm{tot}}\,m_4 .
$$

**(b) Composition bound on $C_2^{\mathrm{tot}}$.** Write $F_\ell=f_\ell\circ\cdots\circ f_1$
($F_0=\mathrm{id}$) and set $A_\ell:=\sup\lVert DF_\ell\rVert_{\mathrm{op}}$,
$S_\ell:=\sup\lVert D^2F_\ell\rVert_{\mathrm{op}}$ over the region.

*Per layer.* With $z=W_\ell x+b_\ell$:
$Df_\ell(x)=\operatorname{diag}(\sigma_\ell'(z))\,W_\ell$, so
$\lVert Df_\ell\rVert_{\mathrm{op}}\le B_1\lVert W_\ell\rVert_2=a_\ell$.
The second derivative is entrywise, $\big(D^2f_\ell(x)[u,u]\big)_k=\sigma_\ell''(z_k)\,(W_\ell u)_k^2$, so

$$
\lVert D^2f_\ell(x)[u,u]\rVert_2
\le B_2\Big(\sum_k|(W_\ell u)_k|^4\Big)^{1/2}
\le B_2\sum_k|(W_\ell u)_k|^2
=B_2\lVert W_\ell u\rVert_2^2
\le B_2\lVert W_\ell\rVert_2^2\lVert u\rVert^2 ,
$$

using $\big(\sum_k|v_k|^4\big)^{1/2}\le\sum_k|v_k|^2$. Thus
$\lVert D^2f_\ell\rVert_{\mathrm{op}}\le B_2\lVert W_\ell\rVert_2^2=b_\ell$.

*Chain rule.* For $F_\ell=f_\ell\circ F_{\ell-1}$,
$DF_\ell=Df_\ell(F_{\ell-1})\,DF_{\ell-1}$ gives $A_\ell\le a_\ell A_{\ell-1}$, i.e.
$A_\ell\le\prod_{j\le\ell}a_j$. For the second derivative,

$$
D^2F_\ell[u,u]=Df_\ell(F_{\ell-1})\,D^2F_{\ell-1}[u,u]
+D^2f_\ell(F_{\ell-1})\big[DF_{\ell-1}u,\,DF_{\ell-1}u\big],
$$

so $S_\ell\le a_\ell S_{\ell-1}+b_\ell A_{\ell-1}^2$. Unrolling from $S_0=0$:

$$
C_2^{\mathrm{tot}}=S_L\le\sum_{\ell=1}^{L}\Big(\prod_{j=\ell+1}^{L}a_j\Big)b_\ell\,A_{\ell-1}^2
\le\sum_{\ell=1}^{L}\Big(\prod_{j=\ell+1}^{L}a_j\Big)b_\ell\Big(\prod_{j=1}^{\ell-1}a_j\Big)^2 .
\qquad\square
$$

### Step 2 — RIS term $\varepsilon_H$ (exact equality)

Write $T(s)=Fs+t_0$ with $t_0=f(s_0)-Fs_0$, and $\Delta:=H(\phi)-F$. The error is
affine in $s$: $T(s)-H_\phi(s)=\Delta s+(t_0-\beta)$. The bias–variance identity
(cross term vanishes since $\mathbb E[\Delta(s-\mu_s)]=0$),

$$
\mathbb E_\mu\lVert\Delta s+c\rVert^2
=\mathbb E_\mu\lVert\Delta(s-\mu_s)\rVert^2+\lVert\Delta\mu_s+c\rVert^2,
$$

is minimized over the bias at $c=t_0-\beta=-\Delta\mu_s$, leaving
$\mathbb E_\mu\lVert\Delta(s-\mu_s)\rVert^2$. Then

$$
\varepsilon_H^2=\mathbb E_\mu\lVert\Delta(s-\mu_s)\rVert^2
=\operatorname{tr}\!\big(\Delta\,\Sigma_s\,\Delta^{\mathsf H}\big)
=\lVert\Delta\,\Sigma_s^{1/2}\rVert_F^2 .
$$

Hence $\varepsilon_H=\lVert(H(\phi)-F)\Sigma_s^{1/2}\rVert_F$ — an **equality**, not a
bound. The argument uses only that $T$ is affine with linear part $F$; it is
indifferent to how $F$ arose, so it is unchanged from the single-layer proof. $\square$

### Step 3 — channel-rank floor

Every $H(\phi)=H_2\operatorname{diag}(\phi)H_1$ has
$\operatorname{col}(H(\phi))\subseteq\operatorname{col}(H_2)$ and, since
$H(\phi)^{\mathsf T}=H_1^{\mathsf T}\operatorname{diag}(\phi)H_2^{\mathsf T}$, also
$\operatorname{row}(H(\phi))\subseteq\operatorname{row}(H_1)$. Thus
$\{H(\phi):\phi\in\mathbb C^N\}\subseteq\mathcal U$, and minimizing over a subset is
$\ge$ the minimum over $\mathcal U$:

$$
\min_\phi\varepsilon_H\ \ge\ \min_{H\in\mathcal U}\lVert(H-F)\Sigma_s^{1/2}\rVert_F
=\operatorname{dist}_{\Sigma_s}(F,\mathcal U).
$$

When $\Sigma_s=\sigma_s^2 I$, the objective $\lVert H-F\rVert_F$ over
$\mathcal U=\{H:P_2H=H,\ HP_1=H\}$ is minimized by $H^\star=P_2FP_1$, giving
$\operatorname{dist}=\sigma_s\lVert F-P_2FP_1\rVert_F$. Target-agnostic, hence
unchanged. $\square$

### Step 4 — exact realization

$\mathcal U$ is a subspace of dimension $\operatorname{rank}(H_1)\cdot\operatorname{rank}(H_2)$.
If $\operatorname{rank}H_1=d$ and $\operatorname{rank}H_2=m$ then
$\mathcal U=\mathbb C^{m\times d}$. The map $\phi\mapsto H(\phi)$ satisfies
$\operatorname{vec}(H(\phi))=K\phi$ with $K=H_1^{\mathsf T}\bullet H_2$ (Khatri–Rao
product); for $N\ge md$ and full-rank channels $K$ has rank $md$, so
$\{H(\phi)\}=\mathbb C^{m\times d}\ni F$. Choosing $\phi$ with $H(\phi)=F$ and
$\beta=t_0$ gives $\varepsilon_H=0$. Target-agnostic, hence unchanged. $\square$

### Combine

Steps 0–2 give $\lVert f-H_\phi\rVert\le\varepsilon_T+\varepsilon_H$ with the stated
$\varepsilon_T$ bound (new composition constant) and exact $\varepsilon_H$; Steps 3–4
give the floor and the exact-realization condition. $\blacksquare$

---

## Corollary ($\varepsilon$-guarantee)

Given $\varepsilon>0$, choose the working radius
$r\le\sqrt{\varepsilon/C_2^{\mathrm{tot}}}$ so that on $\mathcal B(s_0,r)$ the
fourth-moment term gives $\varepsilon_T\le\tfrac12 C_2^{\mathrm{tot}}m_4\le\varepsilon/2$,
and $N\ge md$ with full-rank channels so that $\varepsilon_H\le\varepsilon/2$; then
$\lVert f-H_\phi\rVert_{L^2(\mu)}\le\varepsilon$. The usable radius shrinks as the
end-to-end curvature grows — see Remark 1.

---

## Remarks

1. **Depth penalty (why this is a *local* result).** With equal layers
   $a_\ell\equiv a$, $b_\ell\equiv b$ the bound collapses to
   $C_2^{\mathrm{tot}}\le b\sum_{\ell=1}^{L}a^{\,L+\ell-2}$. For expansive layers
   ($a=B_1\lVert W\rVert_2>1$) this is $\Theta(a^{2L})$ — **exponential in depth** —
   so the linearization radius $r\lesssim\sqrt{\varepsilon/C_2^{\mathrm{tot}}}$
   collapses with depth; for contractive layers ($a<1$) it is
   $\lesssim b\,a^{\,L-1}/(1-a)$ and stays small. A single RIS realizes one linear
   map, so approximating a deep nonlinear net by it is inherently local.
2. **Dual to depth separation.** This is the flip side of the article's
   depth-separation thesis: depth separation says *no global* linear map reproduces
   the network, and this theorem says a RIS reproduces it only on a neighborhood
   whose size collapses with depth. The two statements are consistent, not in
   tension.
3. **ReLU is excluded.** (A1)–(A2) require $B_2<\infty$. A ReLU network has no
   bounded Hessian; for it one needs a smoothed surrogate (softplus/GELU) or a
   first-order (Lipschitz) argument instead of this second-order one. Useful
   constants: logistic sigmoid $B_1=\tfrac14$, $B_2=\tfrac{1}{6\sqrt3}\approx0.0962$;
   $\tanh$ $B_1=1$, $B_2\approx0.7698$; softplus $B_1=1$, $B_2=\tfrac14$; GELU/SiLU
   finite.
4. **No norm mixing.** Step 2 shows the RIS term is *exactly* the $L^2(\mu)$ error;
   only the Taylor term is an inequality — as in the single-layer claim.
5. **Degrees of freedom / unit-modulus.** Unchanged:
   $\dim\{H(\phi)\}\le\min(N,\operatorname{rank}H_1\cdot\operatorname{rank}H_2,md)$, and
   $|\phi_n|=1$ enters only in Step 4 (reached up to a receiver scalar $\eta$ as
   $N\to\infty$), while Steps 2–3 hold for every $\phi$.
6. **Tighter target.** Replacing the Jacobian $F$ by the $L^2$-regression optimum
   $F^\star=\mathbb E[(f-\bar f)(s-\mu_s)^{\mathsf H}]\Sigma_s^{-1}$ only lowers
   $\varepsilon_T$ (it is the best linear map in $L^2(\mu)$), so the bound still holds —
   and the RIS side is unaffected since it only ever sees a target matrix.
7. **Cascaded surfaces (a genuinely deeper generalization).** If instead one wants
   the RIS to realize the *depth* rather than a local linearization, that is a
   different theorem: a cascade
   $f\approx\sigma_L\!\big(H^{(L)}_\phi\cdots\sigma_1(H^{(1)}_\phi s)\big)$ with a
   physical nonlinearity between surfaces. Each surface approximates one $W_\ell$
   (Steps 2–4 per layer) and the per-layer errors telescope with Lipschitz
   amplification $\prod_{j>\ell}a_j$; no linearization, but it requires an
   between-surface hardware nonlinearity. Out of scope here, noted as the natural
   next step.

---

## Summary

1. **Bound on $\varepsilon_T$.** $\varepsilon_T\le\tfrac12 C_2^{\mathrm{tot}}m_4$ with
   $C_2^{\mathrm{tot}}\le\sum_\ell(\prod_{j>\ell}a_j)\,b_\ell(\prod_{j<\ell}a_j)^2$,
   $a_\ell=B_1\lVert W_\ell\rVert_2$, $b_\ell=B_2\lVert W_\ell\rVert_2^2$.
2. **Equality for $\varepsilon_H$.** $\varepsilon_H=\lVert(H(\phi)-F)\Sigma_s^{1/2}\rVert_F$, with $F=Df(s_0)$ the end-to-end Jacobian.
3. **Lower bound on $\varepsilon_H$.** If $\Sigma_s\propto I$, then $\min_\phi\varepsilon_H\ge\lVert(F-P_2FP_1)\Sigma_s^{1/2}\rVert_F$.
4. **Full rank, large RIS, free phases.** If $\operatorname{rank}H_1=d$, $\operatorname{rank}H_2=m$, $N\ge md$, and $\phi\in\mathbb C^N$ is unconstrained, then $\varepsilon_H=0$.
5. **What changed vs. the single layer.** Only Step 1 (the curvature constant). The RIS realizability result (Steps 2–4) is architecture-agnostic.

---

## Simulating the generalized theory (different architectures)

The current numerical study verifies the single-layer claim with:

```bash
python framework/ris_bounds_sim.py --experiment taylor_sweep --teacher sigmoid
python framework/ris_bounds_sim.py --experiment nm_sweep --teacher sigmoid --channel_type synthetic_rayleigh
```

To verify the generalized claim we vary the **teacher architecture** (depth,
width, activation) and check the same two predictions. Per the repo conventions
(`CLAUDE.md`: *add new approaches behind a flag alongside the existing one, and put
exploratory ideas in separate scripts rather than deeply integrating them*), the
existing CLI and the single-layer `Teacher` path must stay byte-for-byte
runnable. Nothing below edits or removes an existing flag, function, or default.

### What already transfers unchanged (no new code)

In `framework/ris_bounds_sim.py` the whole RIS side is **already
architecture-agnostic** — it consumes only the Jacobian $F$ and the channels:

* `input_stats`, `ris_unconstrained_nmse`, `ris_rank_floor_nmse`,
  `ris_unitmod_nmse`, and the $N_m$ loop in `run_nm_sweep` take `G = teacher.jacobian(mu_s)` and the channel pools — they do not care how many layers produced `G`.
* The *empirical* half of `taylor_gap` calls only `teacher(s)`, `teacher(s0)` and
  `G`, so it already measures $\varepsilon_T=\lVert f-T\rVert_{L^2}$ for any teacher.

So Step 2–4 (the $\varepsilon_H$ curves and the $N_m=n_rn_t$ threshold) need **zero**
new code — they only need a deeper teacher object to hand them an $F$.

### The one piece that is single-layer-specific

Only `taylor_bound(...)` encodes the one-layer constant
$\tfrac12 C_2\lVert W\rVert_2^2 m_4$ (it reads `teacher.W`, `spectral_norm_W()`,
`C2_SIGMOID_REAL`). The generalization supplies a parallel
`taylor_bound_general(...)` computing $\tfrac12 C_2^{\mathrm{tot}} m_4$ from the
composition formula — added, not substituted.

### Recommended non-breaking approach: a sibling script

Create `framework/ris_bounds_sim_general.py` that **imports** the reusable,
architecture-agnostic machinery from `ris_bounds_sim` and only adds the deep
teacher and the new constant. This leaves `ris_bounds_sim.py` untouched:

```python
# framework/ris_bounds_sim_general.py  (new file; imports, does not modify)
from ris_bounds_sim import (
    generate_s, input_stats, taylor_gap, make_ris_channel_pools,
    ris_unconstrained_nmse, ris_rank_floor_nmse, ris_unitmod_nmse,
    csigmoid, csigmoid_prime, _show_plots,
)

# --- smooth activations with analytic 1st/2nd derivative sups (B1, B2) ---
ACTS = {
    # name:        (fn,        fn',             B1,     B2)
    "sigmoid": (csigmoid, csigmoid_prime, 0.25,   0.09622504486),
    "tanh":    (torch.tanh, lambda z: 1 - torch.tanh(z)**2, 1.0, 0.7698),
    "softplus":(lambda z: torch.log1p(torch.exp(z)), csigmoid, 1.0, 0.25),
}

class DeepTeacher:
    """f = f_L o ... o f_1, f_l(x) = act(W_l x + b_l). Holomorphic act => the
    complex-linear Jacobian is an exact layer product (no complex autograd)."""
    def __init__(self, dims, act, device, seed=0):
        g = torch.Generator("cpu").manual_seed(seed)
        self.Ws, self.bs = [], []
        for din, dout in zip(dims[:-1], dims[1:]):
            W = torch.view_as_complex(torch.randn(dout, din, 2, generator=g)) / din**0.5
            b = torch.view_as_complex(torch.randn(dout, 2, generator=g)) * 0.5
            self.Ws.append(W.to(device)); self.bs.append(b.to(device))
        self.fn, self.dfn, self.B1, self.B2 = ACTS[act]

    def __call__(self, s):
        x = s
        for W, b in zip(self.Ws, self.bs):
            x = self.fn(x @ W.T + b)
        return x

    def _preacts(self, s0):               # cache z_l^(0) along the forward pass
        zs, x = [], s0
        for W, b in zip(self.Ws, self.bs):
            z = x @ W.T + b; zs.append(z); x = self.fn(z)
        return zs

    def jacobian(self, s0):               # F = prod_l diag(act'(z_l)) W_l
        zs = self._preacts(s0.unsqueeze(0) if s0.dim() == 1 else s0)
        F = None
        for W, z in zip(self.Ws, zs):
            J = self.dfn(z).reshape(-1, 1) * W      # diag(act'(z)) W
            F = J if F is None else J @ F
        return F

    def c2_total(self):                   # sum_l (prod_{j>l} a_j) b_l (prod_{j<l} a_j)^2
        a = [self.B1 * torch.linalg.matrix_norm(W, 2).item() for W in self.Ws]
        b = [self.B2 * torch.linalg.matrix_norm(W, 2).item()**2 for W in self.Ws]
        import math
        pre = [math.prod(a[:l]) for l in range(len(a))]        # prod_{j<l}
        suf = [math.prod(a[l+1:]) for l in range(len(a))]      # prod_{j>l}
        return sum(suf[l] * b[l] * pre[l]**2 for l in range(len(a)))

def taylor_bound_general(teacher, m4):
    return 0.5 * teacher.c2_total() * m4
```

Then reuse the existing `taylor_sweep` / `nm_sweep` logic (either copy the two
`run_*` bodies and swap in `DeepTeacher` + `taylor_bound_general`, or import and
parametrize them). Suggested new flags: `--depth`, `--width`, `--activation`
(`sigmoid|tanh|softplus`). The single-layer script keeps `--teacher sigmoid|linear`
exactly as is.

*Minimal-additive alternative:* if you prefer one file, add `"mlp"` to
`TEACHER_KINDS`, add a `DeepTeacher`, and have `taylor_bound` dispatch on
`teacher.kind` (sigmoid/linear fall through to the current code path). This is
purely additive — existing commands are unaffected — but editing the live script
is more intrusive than a sibling file, so the sibling script is preferred.

### How to run it

Parallel to your current commands:

```bash
# Taylor gap vs input radius for a depth-3 tanh MLP:
# empirical eps_T must sit UNDER (1/2) C2_tot m4 and scale ~ r^2 (slope 2).
python framework/ris_bounds_sim_general.py --experiment taylor_sweep \
    --activation tanh --depth 3 --width 16

# eps_H vs N_m for the same teacher over iid Rayleigh channels:
# curves and the N_m = n_r*n_t threshold should look like the single-layer case.
python framework/ris_bounds_sim_general.py --experiment nm_sweep \
    --activation tanh --depth 3 --width 16 --channel_type synthetic_rayleigh

# depth sweep: C2_tot (hence the eps_T intercept) grows with depth for ||W||_2>1.
for Ld in 1 2 3 4; do
  python framework/ris_bounds_sim_general.py --experiment taylor_sweep \
      --activation sigmoid --depth $Ld --width 16
done
```

### Predictions to confirm

* **`taylor_sweep`** — the empirical $\varepsilon_T$ stays below
  $\tfrac12 C_2^{\mathrm{tot}}m_4$ and keeps slope $2$ in $r$; the bound's intercept
  rises with depth/width (larger $C_2^{\mathrm{tot}}$), so the usable radius shrinks
  — the quantitative content of Remark 1.
* **`nm_sweep`** — **unchanged in shape**: unit-modulus $\ge$ unconstrained-LS
  $\ge$ rank-floor, and $\varepsilon_H\to 0$ once $N_m\ge n_r n_t$ with full-rank
  (Rayleigh / low-$K$) channels. A deeper teacher only changes the target matrix
  $F$, not the realizability geometry — this *is* the architecture-agnostic claim
  (Steps 2–4) under test.
* **Curvature-constant check (optional, worth adding)** — confirm the analytic
  $C_2^{\mathrm{tot}}$ really upper-bounds the measured
  $\sup\lVert D^2f\rVert_{\mathrm{op}}$ via finite-difference Hessian–vector products
  on random directions.

### Channel caveat (unchanged, README §7)

Strong LoS (high Ricean $K$ on $H_1$/$H_2$) collapses the cascade toward rank 1
and hides the $N_m=n_rn_t$ threshold. Keep using `--channel_type synthetic_rayleigh`
(or a low/negative `--kappa`) for the $\varepsilon_H$ experiments, exactly as in the
single-layer run.
