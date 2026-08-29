# Mean Field Schrödinger Bridge

- Microscopic Population Steering Problem
  $$
  \begin{aligned}
  &\min_{\alpha} \sum_{i=1}^N \left[\int_0^T \ell(t, x_i(t), \alpha_i(t), m_{i}(\cdot,t)) \rmd t \right]\\
  &\text{s.t.} \quad \rmd x_i(t) = f(t, x_i(t), \alpha_i(t), m_{i}(\cdot,t))\rmd t + \sigma\rmd B_i(t) \\
  &\quad\qquad x_i(0) \sim \rho_0, \quad x_i(T) \sim \rho_1
  \end{aligned}
  $$
  - $m_{i}(x,t) = \frac{1}{N-1}\sum_{i\ne j }\delta(x-x_j(t))$: Empirical distribution

- Example: Quadratic Cost with McKean-Vlassov Dynamics
  $$
  \begin{aligned}
  &\ell = \frac{1}{2}\|\alpha_i(t)\|^2\\
  &f = -\frac{1}{N-1}\nabla W(x_i(t)-x_j(t)) + \alpha_i(t) = - [\nabla W * m_i](x_i(t)) + \alpha_i(t)
  \end{aligned}
  $$

---

- Mean Field Limit
  $$
  \begin{aligned}
  &\min_{\alpha} \int_0^T \int \ell(t,x,\alpha(x,t),\rho(\cdot,t)) \rho(x,t) \rmd x \rmd t \\
  &\text{s.t.} \quad \partial_t \rho(x,t) = \nabla\cdot[\rho(x,t)f(t,x,\alpha(x,t),\rho(\cdot,t))] + \frac{\sigma^2}{2}\Delta \rho(x,t) \\
  &\qquad\quad \rho(\cdot, 0) = \rho_0, \quad \rho(\cdot, T) = \rho_1
  \end{aligned}
  $$


- Example: Quadratic Cost with McKean-Vlassov Dynamics
  $$
  \begin{aligned}
  &\min_{\alpha} \int_0^T \int \frac{1}{2}\|\alpha(x,t)\|^2 \rho(x,t) \rmd x \rmd t \\
  &\text{s.t.} \quad \partial_t \rho(x,t) = \nabla\cdot(\rho(x,t)\left\{[\nabla W * \rho](x,t) - \alpha(x,t)\right\}) + \frac{\sigma^2}{2}\Delta \rho(x,t) \\
  &\qquad\quad \rho(\cdot, 0) = \rho_0, \quad \rho(\cdot, T) = \rho_1
  \end{aligned}
  $$


---

# Existing Studies

- Backhoff, Julio, et al. (2020) "The mean field Schrödinger problem: ergodic behavior, entropy estimates and functional inequalities." Probability Theory and Related Fields
  - Derived equivalent FBSDE for Quadratic Cost with McKean-Vlassov Dynamics
  - No numerical computations were performed
- Hernández, Camilo, and Ludovic Tangpi (2024) “Propagation of Chaos for Mean Field Schrödinger Problems.” arXiv
  - They extended the cost function of Backhoff's work to a general form
  - No numerical computations were performed
- Liu, Guan-Horng et al. (2022) "Deep Generalized Schrödinger Bridge." NeurIPS
  - Derived equivalent FBSDE for $f = f(t,x,m(x,t)) + \alpha$, $\ell = \frac{1}{2\sigma^2} \|\alpha(x,t)\|^2 + F(t,x,m(x,t))$ (local interaction with respect to density)
  - Proposed a learning method using neural networks

---

- Rapakoulias, George, et al. (2025) "Steering Large Agent Populations Using Mean-Field Schrodinger Bridges with Gaussian Mixture Models." arXiv
  - $f$: McKean-Vlasov with $W(x)=x^\top Q x$
  - $\ell$: Quadratic
  - They showed that the problem is formulated as an LQR and proposed an algorithm to solve it using SDP

---


## Summary

|          | Backhoff (2020) | Hernández (2024)       | Liu (2022)                                        | Rapakoulias (2025)        |
| -------- | --------------- | ---------------------- | ------------------------------------------------- | ------------------------- |
| Dynamics | McKean-Vlasov   | McKean-Vlasov | General (local interaction)                       | McKean-Vlasov (Quadratic) |
| Cost     | Quadratic       | General               | Quadratic for input + general (local interaction) | Quadratic                 |
| Numerics | No              | No                     | Yes (NN)                                          | Yes (SDP)             |


- What has not been addressed yet: Schrödinger Bridge for 
  1. non-quadratic McKean-Vlasov Dynamics
  2. General non-local Mean Field Dynamics (which includes 1)
- In either direction, density estimation is required


---

# Problem I Want to Consider
- Non-local Mean Field Schrödinger problem
  $$
  \small
  \begin{aligned}
  &\min_{\alpha} \int_0^T \int \frac{1}{2\sigma^2}\|\alpha(x,t)\|^2 + F(x,t,\rho(\cdot,t))\rho(x,t) \rmd x \rmd t \\
  &\text{s.t.} \quad \partial_t \rho(x,t) = - \nabla\cdot(\rho(x,t)\left\{f(x,t,\rho(\cdot,t)) + \alpha(x,t)\right\}) + \frac{\sigma^2}{2}\Delta \rho(x,t) \\
  &\qquad\quad \rho(\cdot, 0) = \rho_0, \quad \rho(\cdot, T) = \rho_1
  \end{aligned}
  $$

- Similar to Liu (2022), but differs in that the density terms are non-local
  - Liu (2022) addresses non-local problems in the formulation but only treats the following two cases in numerical computations:
    - $F=g(\rho(x,t))$: Local
    - $F=\bbE_{y\sim \rho}[g(x,y)]$: Monte Carlo approximation ($O(N^2)$ for $N$ particles)
- I want to develop a method with lower computational complexity

---

# Review of Liu (2022) Method


## Forward-Backward SDEs
- Forward SDE (starting from $X_t \sim \rho_0$):
  $$
  \begin{aligned}
  dX_t &= (f_t + \sigma Z_t)dt + \sigma dW_t \\
  dY_t &= \left(\frac{1}{2}\|Z_t\|^2 + F_t\right)dt + Z_t^\top dW_t \\
  d\hat{Y}_t &= \left(\frac{1}{2}\|\hat{Z}_t\|^2 + \nabla\cdot(\sigma\hat{Z}_t - f_t) + \hat{Z}_t^\top Z_t - F_t\right)dt + \hat{Z}_t^\top dW_t
  \end{aligned}
  $$
  - $Y_t = \log\Psi(X_t,t)$, $Z_t = \sigma\nabla\log\Psi(X_t,t)$
  - $\hat Y_t = \log\hat\Psi(X_t,t)$, $\hat Z_t = \sigma\nabla\log\hat\Psi(X_t,t)$
  - $\alpha(X_t,t) = \sigma Z_t$

---

- Backward SDE (starting from $\bar{X}_s \sim \rho_{1}$)
  $$
  \begin{aligned}
  d\bar{X}_s &= (-f_s + \sigma\hat{Z}_s)ds + \sigma dW_s \\[6pt]
  dY_s &= \left(\frac{1}{2}\|Z_s\|^2 + \nabla\cdot(\sigma Z_s + f_s) + Z_s^\top\hat{Z}_s - F_s\right)ds + Z_s^\top dW_s \\[6pt]
  d\hat{Y}_s &= \left(\frac{1}{2}\|\hat{Z}_s\|^2 + F_s\right)ds + \hat{Z}_s^\top dW_s
  \end{aligned}
  $$
  - $Y_s = \log\Psi(\bar X_s,t)$, $Z_s = \sigma\nabla\log\Psi(\bar X_s,t)$
  - $\hat Y_s = \log\hat\Psi(\bar X_s,t)$, $\hat Z_s = \sigma\nabla\log\hat\Psi(\bar X_s,t)$
  - $\bar \alpha(\bar X_s,t) = \sigma \hat Z_s$

---

## Loss Function

- Parametrize $Y_\theta$, $\hat Y_\phi$, $Z_\theta$, $\hat Z_\phi$ by NNs and try to minimize the loss functions
$$\mathcal{L} = \mathcal{L}_{\text{IPF}} + \lambda_{\text{TD}}\mathcal{L}_{\text{TD}} + \lambda_{\text{FK}}\mathcal{L}_{\text{FK}}$$

  - $\mathcal{L}_{\text{IPF}}(\theta) = \int \mathbb{E}\left[\frac{1}{2}\|\widehat{Z}_t^\phi + Z_\theta^t\|^2 + \nabla \cdot (\sigma \widehat{Z}_t^\phi - f)\right] dt$
    - Ensures transport from $\rho_0 \to \rho_1$

  - $\mathcal{L}_{\text{TD}}(\theta) = \sum_{s=0}^{T} \mathbb{E}\left[\|Y_\theta(\overline{X}_s, s) - \text{TD}_s\|\right] \delta s$
    - $\text{TD}_{s+\delta s} := Y_s^\theta + \left(\frac{1}{2}\|Z_s^\theta\|^2 + \nabla \cdot (\sigma Z_s^\theta + f_s) + Z_s^\theta{}^T \widehat{Z}_s^\phi - F_s\right) \delta s + Z_s^\theta{}^T \delta W_s$
    - Ensures consistency when $F \neq 0$

  - $\mathcal{L}_{\text{FK}}(\theta) = \sum_{s=0}^{T} \mathbb{E}\left[\|\sigma \nabla Y_\theta(\overline{X}_s, s) - Z_\theta(\overline{X}_s, s)\|\right] \delta s$
    - Enforces gradient relationship for added robustness

---


## Alternating Optimization Loop


1. Backward drift learning
   1. Simulate forward SDE using current forward drift $Y_\theta(x,t)$ and $Z_\theta(x,t)$
   2. Compute loss $\mathcal{L}(\phi)$ by using the forward trajectory
   3. Update backward parameters $\phi$ with the gradient of $\nabla_\phi\, \mathcal{L}(\phi)$

2. Forward drift learning
   1. Simulate backward SDE using current backward drift $Y_\theta(x,t)$ and $Z_\theta(x,t)$
   2. Compute loss $\mathcal{L}(\theta)$ by using the backward trajectory
   3. Update forward parameters $\theta$ with the gradient of $\nabla_\theta\, \mathcal{L}(\theta)$

- In both steps, to simulate SDE and compute loss, we need to compute  
  $f_t = f(X_t^{(i)},t,m^{(i)}(\cdot,t))$ and $F_t = F(X_t^{(i)},t,m^{(i)}(\cdot,t))$,   
  which requires $O(N^2)$ complexity


---

# Proposed Method

- Parameterize $f(x,t,m(x,t))$ as $f_{\alpha}(x,t)$ and $F(x,t,m(x,t))$ as $F_{\beta}(x,t)$
- Loss function for learning interaction:
  $$\mathcal{L}_{f}(\alpha) = \mathbb{E}\left[\|f(x, t, m(\cdot,t)) - f_{\alpha}(x,t)\|^2\right]$$
  $$\mathcal{L}_{F}(\beta) = \mathbb{E}\left[\|F(x, t, m(\cdot,t)) - F_{\beta}(x,t)\|^2\right]$$
- Following Han et al (2024), we reduce computational effort by not including the density function as an argument to the learning NNs

### Learning Cycle
1. Backward drift learning: computed independently using learned interaction terms
2. Interaction learning
3. Forward drift learning: computed independently using learned interaction terms
4. Interaction learning:
       

---

## 既存手法との比較

| Aspect | DeepGSB (Liu et al.) | Proposed Method |
|--------|---------------------|-----------------|
| Steps | 2 (alternating optimization) | 4 (+ interaction learning) |
| MF Computation | Direct MC every iteration | MC during training, NN for inference |
| Complexity (inference) | O(N²) per timestep | O(N) per timestep |
