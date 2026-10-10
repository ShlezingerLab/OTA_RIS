# RIS Approximation of a One-Layer DNN via First-Order Taylor Expansion

**Claim and proof.** Over-the-air realization of a trained network $f(s)=\\sigma(Ws+b)$ by
a reconfigurable intelligent surface (RIS) cascade $H\_\\phi(s)=H\_2\\operatorname{diag}(\\phi)H\_1,s+\\beta$.

\---

## Setup

* Input $s\\in\\mathcal S\\subseteq\\mathbb C^{d}$ random with law $\\mu$, mean $\\mu\_s$, covariance
$\\Sigma\_s=\\mathbb E\[(s-\\mu\_s)(s-\\mu\_s)^{\\mathsf H}]$.
* For functions $g:\\mathcal S\\to\\mathbb C^{m}$, the Hilbert norm
$\\lVert g\\rVert^2:=\\mathbb E\_{s\\sim\\mu}\\lVert g(s)\\rVert\_2^2$.
* Network $f(s)=\\sigma(Ws+b)$, $W\\in\\mathbb C^{m\\times d}$, $\\sigma$ applied entrywise,
holomorphic on the region of interest with $C\_2:=\\sup\_z|\\sigma''(z)|<\\infty$.
* Reference point $s\_0$; $z\_0=Ws\_0+b$; Jacobian $F=\\operatorname{diag}(\\sigma'(z\_0)),W$;
first-order model $T(s)=f(s\_0)+F(s-s\_0)$.
* RIS cascade $H\_\\phi(s)=H(\\phi),s+\\beta$ with
$H(\\phi)=H\_2\\operatorname{diag}(\\phi)H\_1\\in\\mathbb C^{m\\times d}$,
$H\_1\\in\\mathbb C^{N\\times d}$, $H\_2\\in\\mathbb C^{m\\times N}$, bias / direct path $\\beta$.

\---

## Theorem

$$
\\lVert f-H\_\\phi\\rVert\\ \\le\\ \\varepsilon\_T+\\varepsilon\_H,\\qquad
\\varepsilon\_T\\le \\tfrac12 C\_2\\lVert W\\rVert\_2^2,m\_4,\\quad
\\varepsilon\_H=\\big\\lVert (H(\\phi)-F),\\Sigma\_s^{1/2}\\big\\rVert\_F,
$$

where $m\_4=(\\mathbb E\_\\mu\\lVert s-s\_0\\rVert^4)^{1/2}$ and $\\varepsilon\_H$ is attained at the optimal
bias $\\beta$. Moreover, with
$\\mathcal U:={H:\\operatorname{col}(H)\\subseteq\\operatorname{col}(H\_2),\\ \\operatorname{row}(H)\\subseteq\\operatorname{row}(H\_1)}$,

$$
\\min\_\\phi\\varepsilon\_H\\ \\ge\\ \\operatorname{dist}*{\\Sigma\_s}(F,\\mathcal U)
:=\\min*{H\\in\\mathcal U}\\lVert(H-F)\\Sigma\_s^{1/2}\\rVert\_F,
$$

with equality to $\\lVert(F-P\_2FP\_1)\\Sigma\_s^{1/2}\\rVert\_F$ when $\\Sigma\_s\\propto I$
($P\_2,P\_1$ the orthogonal projectors onto $\\operatorname{col}(H\_2),\\operatorname{row}(H\_1)$);
and $\\varepsilon\_H=0$ is achievable when $N\\ge md$ with
$\\operatorname{rank}H\_1=d,\\ \\operatorname{rank}H\_2=m$.

\---

## Proof

### Step 0 — split

$L^2(\\mu)$ is a Hilbert space, so the triangle inequality gives

$$
\\lVert f-H\_\\phi\\rVert\\le\\lVert f-T\\rVert+\\lVert T-H\_\\phi\\rVert=:\\varepsilon\_T+\\varepsilon\_H.
$$

### Step 1 — Taylor term $\\varepsilon\_T$

Fix $s$ and let $\\xi=W(s-s\_0)\\in\\mathbb C^m$, so coordinate $k$ of the pre-activation moves
from $z\_{0,k}$ to $z\_{0,k}+\\xi\_k$. Taylor's theorem with integral remainder for the scalar
holomorphic $\\sigma$ along the segment:

$$
f\_k(s)-T\_k(s)=\\sigma(z\_{0,k}+\\xi\_k)-\\big\[\\sigma(z\_{0,k})+\\sigma'(z\_{0,k})\\xi\_k\\big]
=\\xi\_k^2!\\int\_0^1(1-t),\\sigma''(z\_{0,k}+t\\xi\_k),dt.
$$

Since $\\int\_0^1(1-t),dt=\\tfrac12$ and $|\\sigma''|\\le C\_2$,

$$
|f\_k(s)-T\_k(s)|\\le\\tfrac12 C\_2,|\\xi\_k|^2.
$$

Summing over $k$ and using $\\sum\_k|\\xi\_k|^4\\le\\big(\\sum\_k|\\xi\_k|^2\\big)^2=\\lVert\\xi\\rVert^4$:

$$
\\lVert f(s)-T(s)\\rVert^2=\\sum\_k|f\_k-T\_k|^2\\le\\tfrac14 C\_2^2\\lVert\\xi\\rVert^4
\\le\\tfrac14 C\_2^2\\lVert W\\rVert\_2^4\\lVert s-s\_0\\rVert^4,
$$

i.e. $\\lVert f(s)-T(s)\\rVert\\le\\tfrac12 C\_2\\lVert W\\rVert\_2^2\\lVert s-s\_0\\rVert^2$ pointwise.
Taking $\\mathbb E\_\\mu$ and $\\sqrt{\\cdot}$:

$$
\\varepsilon\_T=\\big(\\mathbb E\_\\mu\\lVert f-T\\rVert^2\\big)^{1/2}
\\le\\tfrac12 C\_2\\lVert W\\rVert\_2^2\\big(\\mathbb E\_\\mu\\lVert s-s\_0\\rVert^4\\big)^{1/2}
=\\tfrac12 C\_2\\lVert W\\rVert\_2^2,m\_4. \\qquad\\square
$$

### Step 2 — RIS term $\\varepsilon\_H$ (exact equality)

Write $T(s)=Fs+t\_0$ with $t\_0=f(s\_0)-Fs\_0$, and $\\Delta:=H(\\phi)-F$. The error is affine in $s$:

$$
T(s)-H\_\\phi(s)=\\Delta,s+(t\_0-\\beta).
$$

For any constant $c$, the bias–variance identity (cross term vanishes since
$\\mathbb E\[\\Delta(s-\\mu\_s)]=0$):

$$
\\mathbb E\_\\mu\\lVert \\Delta s+c\\rVert^2
=\\mathbb E\_\\mu\\lVert\\Delta(s-\\mu\_s)\\rVert^2+\\lVert\\Delta\\mu\_s+c\\rVert^2,
$$

minimized over the bias at $c=t\_0-\\beta=-\\Delta\\mu\_s$, leaving
$\\mathbb E\_\\mu\\lVert\\Delta(s-\\mu\_s)\\rVert^2$. Using
$\\lVert\\Delta x\\rVert^2=\\operatorname{tr}(\\Delta,xx^{\\mathsf H}\\Delta^{\\mathsf H})$ and linearity of
$\\operatorname{tr},\\mathbb E$:

$$
\\varepsilon\_H^2=\\mathbb E\_\\mu\\lVert\\Delta(s-\\mu\_s)\\rVert^2
=\\operatorname{tr}!\\big(\\Delta,\\Sigma\_s,\\Delta^{\\mathsf H}\\big)
=\\operatorname{tr}!\\big((\\Delta\\Sigma\_s^{1/2})(\\Delta\\Sigma\_s^{1/2})^{\\mathsf H}\\big)
=\\lVert\\Delta,\\Sigma\_s^{1/2}\\rVert\_F^2.
$$

Hence $\\varepsilon\_H=\\lVert(H(\\phi)-F)\\Sigma\_s^{1/2}\\rVert\_F$ — an **equality**, not a bound. This is
where the $L^2(\\mu)$ error equals a (covariance-weighted) Frobenius norm. $\\square$

### Step 3 — channel-rank floor

Every $H(\\phi)=H\_2\\operatorname{diag}(\\phi)H\_1$ has $\\operatorname{col}(H(\\phi))\\subseteq\\operatorname{col}(H\_2)$
and, since $H(\\phi)^{\\mathsf T}=H\_1^{\\mathsf T}\\operatorname{diag}(\\phi)H\_2^{\\mathsf T}$, also
$\\operatorname{row}(H(\\phi))\\subseteq\\operatorname{row}(H\_1)$. Thus
${H(\\phi):\\phi\\in\\mathbb C^N}\\subseteq\\mathcal U$, and minimizing over a subset is $\\ge$ the
minimum over $\\mathcal U$:

$$
\\min\_\\phi\\varepsilon\_H
\\ \\ge\\ \\min\_{H\\in\\mathcal U}\\lVert(H-F)\\Sigma\_s^{1/2}\\rVert\_F
=\\operatorname{dist}\_{\\Sigma\_s}(F,\\mathcal U).
$$

When $\\Sigma\_s=\\sigma\_s^2 I$, the objective $\\lVert H-F\\rVert\_F$ over
$\\mathcal U={H:P\_2H=H,\\ HP\_1=H}$ is minimized by the orthogonal projection
$H^\\star=P\_2FP\_1$ (the two constraints act on the independent left/right factors), giving
$\\operatorname{dist}=\\sigma\_s\\lVert F-P\_2FP\_1\\rVert\_F$. $\\square$

### Step 4 — exact realization

$\\mathcal U$ is a subspace of dimension $\\operatorname{rank}(H\_1)\\cdot\\operatorname{rank}(H\_2)$. If
$\\operatorname{rank}H\_1=d$ and $\\operatorname{rank}H\_2=m$ then $\\mathcal U=\\mathbb C^{m\\times d}$. The map
$\\phi\\mapsto H(\\phi)$ satisfies $\\operatorname{vec}(H(\\phi))=K\\phi$ with $K=H\_1^{\\mathsf T}!\\bullet H\_2$
(Khatri–Rao product); for $N\\ge md$ and full-rank channels $K$ has rank $md$, so
${H(\\phi)}=\\mathbb C^{m\\times d}\\ni F$. Choosing $\\phi$ with $H(\\phi)=F$ and $\\beta=t\_0$ gives
$\\varepsilon\_H=0$. $\\square$

### Combine

Steps 0–2 give $\\lVert f-H\_\\phi\\rVert\\le\\varepsilon\_T+\\varepsilon\_H$ with the stated $\\varepsilon\_T$
bound and exact $\\varepsilon\_H$; Steps 3–4 give the floor and the exact-realization condition.
$\\blacksquare$

\---

## Corollary ($\\varepsilon$-guarantee)

Given $\\varepsilon>0$, choose $r\\le\\sqrt{\\varepsilon/(C\_2\\lVert W\\rVert\_2^2)}$ so that on
$\\mathcal B(s\_0,r)$ the fourth-moment term gives $\\varepsilon\_T\\le\\varepsilon/2$, and $N$ large enough
(full-rank channels) that $\\varepsilon\_H\\le\\varepsilon/2$; then
$\\lVert f-H\_\\phi\\rVert\_{L^2(\\mu)}\\le\\varepsilon$.

\---

## Remarks

1. **Unit-modulus constraint.** $|\\phi\_n|=1$ enters only in Step 4: with fixed magnitudes,
$\\varepsilon\_H=0$ is reached up to a receiver scalar $\\eta$ and as $N\\to\\infty$, while Steps 2–3
(equality and floor) hold for every $\\phi$.
2. **No norm mixing.** Step 2 shows the RIS term is *exactly* the $L^2(\\mu)$ error; only the Taylor
term is an inequality.
3. **Degrees of freedom.** $\\dim{H(\\phi)}\\le\\min(N,\\ \\operatorname{rank}H\_1\\cdot\\operatorname{rank}H\_2,\\ md)$:
beyond $N^\\star=\\operatorname{rank}H\_1\\cdot\\operatorname{rank}H\_2$ extra RIS elements add SNR/array gain
but no new degrees of freedom.
4. **Tighter target.** Replacing the Taylor $F$ by the $L^2$-regression optimum
$F^\\star=\\mathbb E\[(f-\\bar f)(s-\\mu\_s)^{\\mathsf H}]\\Sigma\_s^{-1}$ only lowers $\\varepsilon\_T$, so the
bound still holds.

---

## Summary

1. **Bound on $\\varepsilon_T$.** $\\varepsilon_T\\le\\tfrac12 C_2\\|W\\|_2^2\\,m_4$.
2. **Equality for $\\varepsilon_H$.** $\\varepsilon_H=\\|(H(\\phi)-F)\\Sigma_s^{1/2}\\|_F$.
3. **Lower bound on $\\varepsilon_H$.** If $\\Sigma_s\\propto I$, then $\\min_\\phi\\varepsilon_H\\ge\\|(F-P_2FP_1)\\Sigma_s^{1/2}\\|_F$.
4. **Full rank, large RIS, free phases.** If $\\operatorname{rank}H_1=d$, $\\operatorname{rank}H_2=m$, $N\\ge md$, and $\\phi\\in\\mathbb C^N$ is unconstrained, then $\\varepsilon_H=0$.
