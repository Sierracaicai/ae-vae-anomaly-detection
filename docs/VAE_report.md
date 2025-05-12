## 🧪 Variational Autoencoder Series (VAE / β-VAE)

Building upon the Autoencoder, we explored the **Variational Autoencoder (VAE)** and its variant **β-VAE**, both offering structured latent space modeling for anomaly detection.

### 📌 Method Overview

#### 1. **Basic VAE**
- Introduces latent space distribution constraint via KL Divergence;
- Output reconstructs input; loss = reconstruction loss + KL divergence;
- Anomaly score is based on reconstruction error.

#### 2. **β-VAE**
- Adds a coefficient β before the KL term to balance compression vs information preservation;
- Higher β leads to tighter latent space but may harm reconstruction quality;
- We tested values like β=4 and β=10 and analyzed their effects.

#### 3. **KL Annealing Strategy**
- To stabilize training and reduce KL dominance early on, we adopted **KL Warm-up**:
  - Initially set β=0 to focus on reconstruction;
  - Gradually increase β to the target (e.g., β=4);
- This helped prevent unstable training caused by overwhelming KL loss.

### 📈 Summary of Experimental Results

| Model            | Precision | Recall | F1-Score | AUC   | Notes                         |
|------------------|-----------|--------|----------|--------|-------------------------------|
| AE               | ✅ High    | ✅ High | ✅ Good   | ✅ High | Fast training, stable results |
| VAE              | ❌ Low     | ❌ Low  | ❌ Poor   | ❌ Mid  | KL-dominated, slow convergence|
| β-VAE            | ✅ Higher  | ❌ Lower| ⚠️ Moderate| ⚠️ Mid  | Requires tuning β and KL weight|
| β-VAE + KL Warmup| ✅ Best    | ⚠️ Drop | ⚠️ Slight ↑| ⚠️ Mid+ | More stable but gains are small|

### 🛑 Current Conclusion

Despite the theoretical advantages, VAE variants underperformed compared to AE on this dataset:
- Longer training time, sensitive to hyperparameters;
- High β improves separation but significantly harms **recall**;
- KL term dominates loss and must be tuned carefully.

🔍 **Conclusion:** We have **paused further optimization on the VAE series** and will continue refining the AE-based models as the primary path forward.
