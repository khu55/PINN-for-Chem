# PINN-for-Chem

Dataset: https://doi.org/10.1007/s00214-024-03120-1
Mechanistic investigation on the gas‑phase thermal decomposition of triazene‑bridged nitro‑1,2,4‑triazole

# PINN for Chemistry: First-Order Thermal Decomposition Kinetics of TBBT

A Physics-Informed Neural Network (PINN) applied to chemical reaction kinetics, with a focus on validating its extrapolation advantage over a standard neural network (Plain NN) under **sparse / partial-time-window** training data.

## Background

TBBT (from my own published DFT mechanistic study) decomposes thermally. DSC experiments observed decomposition at 154.6°C, with a half-life of approximately 300 seconds (5 minutes). This is a first-order reaction:

$$\frac{dy}{dt} = -k_1 y, \qquad y(t) = y_0\, e^{-k_1 t}$$

The rate constant is derived directly from the experimental half-life:

$$k_1 = \frac{\ln 2}{t_{1/2}} = \frac{\ln 2}{300} \approx 2.31\times10^{-3}\ \text{s}^{-1}$$

This $k_1$ comes straight from an experimental observation, with no additional assumed activation energy or pre-exponential factor layered on top — making it the most defensible part of the whole pipeline. The project deliberately models only this single step, rather than stacking together a multi-step mechanism with loosely justified rate constants just to look more elaborate.

## Core Question

**When training data only covers part of the time window, can a PINN extrapolate more accurately into the uncovered region than a standard data-driven NN?**

Intuitively, a purely data-driven NN is essentially guessing once it leaves the range of its training data. A PINN additionally constrains the network's derivative behavior via the ODE residual across the *entire* domain — so even without data in some region, the governing equation itself should still anchor the network's behavior there. This project tests that intuition quantitatively, and asks how strong the effect actually is.

## Method

1. **Data generation**: The "ground truth" curve is generated from the closed-form analytic solution $y=e^{-k_1t}$ (a first-order reaction has an exact solution, so there's no need for a numerical ODE solver, which would otherwise introduce its own numerical error).
2. **Time normalization**: Real time $t\in[0, 1500\text{s}]$ (about 5 half-lives) is mapped to dimensionless $\tau\in[0,1]$, giving $k_{1,\text{scaled}} = k_1 \times T_{MAX} \approx 3.47$ — this keeps gradients in a numerically stable range during training.
3. **Models**: PINN and Plain NN share the identical network architecture (3 fully-connected layers with Tanh activations, Sigmoid output). The only difference is the loss function:
   - Plain NN: data MSE loss only
   - PINN: data MSE loss + 0.1 × ODE residual loss (residual of $\frac{dy}{d\tau}+k_{1,\text{scaled}}\,y=0$, evaluated across the full time domain)
4. **Comparison experiment**: Training data is restricted to only the first 10% / 30% / 50% / 70% of the time window. Both models are then evaluated across the *entire* time domain, with particular attention to the extrapolation region — i.e., the part the training data never covered.
5. **Metrics**: MSE and MAE within the extrapolation region, plus the ratio between the two models' errors (improvement factor).

## Results

| Training Coverage | Plain NN Extrap. MSE | PINN Extrap. MSE | Improvement Factor |
|---|---|---|---|
| 10% | 0.016254 | 0.000042 | **388.5×** |
| 30% | 0.000171 | 0.000010 | 16.5× |
| 50% | 0.000027 | 0.000004 | 7.1× |
| 70% | 0.000014 | 0.000001 | 16.6× |

**Conclusion**: The sparser the training data, the larger the PINN's extrapolation advantage over the Plain NN. In the extreme case of only 10% coverage, the PINN's extrapolation error is nearly 400× smaller than the Plain NN's. This confirms that the physics constraint effectively substitutes for missing data in data-scarce regimes — which is precisely the core value proposition of PINNs over purely data-driven models.

## Files

- `PINN_for_chem_TBBT_decomposition.ipynb` — Full code: data generation, model training, visualization, quantitative comparison

## Dependencies

```
numpy
torch
matplotlib
pandas
```

Runs directly in Google Colab (all dependencies pre-installed). For local execution, `pip install` the packages above first.

## Possible Extensions

- Add noise to the training data to test PINN robustness under noisy + sparse conditions
- Repeat with different random seeds to confirm the result isn't a one-off artifact of a single run
- Extend to multi-step consecutive reactions — but only if every step has a reliable activation energy / pre-exponential factor; otherwise the physics constraint itself is wrong, and a more "accurate-looking" PINN result would be meaningless.
