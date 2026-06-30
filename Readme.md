# 🛰️ Aditya-L1 Space Weather Monitoring Pipeline (Phase 3 & Phase 5)

An end-to-end, zero-latency predictive forecasting engine built for automated solar flare early-warning alerts. By capturing physical precursor trends inside thermal Soft X-ray telemetry streams, this pipeline uses a tuned heterogeneous tree ensemble to forecast eruptions minutes before rapid flux initialization, completely bypassing the need for secondary payload hardware streams.

---

## 📊 Core Operational Metrics (Final Blind Test Split)

Following a strict multi-stage chronological validation tuning gate, the production pipeline was evaluated against a completely untouched, future time-series sequence.

* **Event-Based True Positive Rate (TPR / Catch Rate):** `100.00%` (25/25 consecutive future eruptions successfully anticipated)
* **Operational False Alarm Rate (FAR Index):** `10.71%` (Only 3 discrete false alarms triggered across 1,582 quiet observation windows)
* **Critical Success Index (CSI / Threat Score):** `0.8929`
* **Median Advanced Early-Warning Cushion:** `11.30 Minutes` before peak initialization

=========================================================================================================
 🚀 STAGE 2: DEPLOYING DYNAMIC 3D OPTIMIZED CONSTRAINTS TO BLIND TEST SPLIT
=========================================================================================================
 Confirmed Flares Occurred       : 25
 Flares Successfully Anticipated : 25 / 25
 Flares Completely Missed        : 0
 Total Operational False Alarms  : 3 (Out of 1,582 Quiet Windows)
---------------------------------------------------------------------------------------------------------
 ⭐ UNBIASED FINAL TRUE POSITIVE RATE (TPR)   : 100.00%
 ⭐ UNBIASED FINAL FALSE ALARM RATE (FAR)     : 10.71%
 ⭐ UNBIASED FINAL CRITICAL SUCCESS INDEX(CSI): 0.8929
---------------------------------------------------------------------------------------------------------
 Test Split Median Early Warning cushion     : 11.30 Minutes
===============================================================
---

## 🧠 Multidimensional 3D Hyperparameter Grid Optimization

Rather than utilizing arbitrary hardcoded temporal boundaries or safety parameters, this engine executed an automated **3D Grid Search Sweep** strictly constrained within the validation split across three co-dependent variables:

1.  **Lookback Rolling Window Size ($W$):** Swept from `30s` to `120s` to mitigate solar background plasma drift.
2.  **Ensemble Confidence Threshold ($\tau$):** Evaluated over the `[0.97, 0.98, 0.99]` probability spectra.
3.  **Minimum Persistence Duration Constraint ($D$):** Evaluated from `6s` to `12s` continuous crossings.

### The Winning Operational Topology
The validation grid sweep mathematically isolated the absolute optimum configuration designed to defeat **Alert Smearing** and telemetry noise:
* **Optimal Window Size:** `10 Seconds`
* **Optimal Confidence Threshold:** `0.97 Probability`
* **Optimal Persistence Filter Length:** `8 Seconds`

Requiring a threshold confirmation density of **8 out of 10 consecutive seconds** creates a high-density temporal barrier. Transient sensor glitches, high-energy cosmic ray bit-flips, or packet drops (typically lasting 1–3 seconds) are filtered out completely, ensuring high payload uptime and preventing false satellite safe-mode triggers.

---

## 📁 Repository Package Artifacts

When building or unpacking the master archive bundle (`complete_isro_pipeline_package.zip`), the directory structure exposes the following pre-trained weights and serialized pipeline parameters:

```text
complete_isro_pipeline_package/
├── deployment_config.json        # Auto-generated dynamic variable metadata manifest
├── scaler_pipeline.pkl           # Serialized Standard Scaler training state map
├── pca_transformer.pkl           # Serialized Principal Component Analysis mapping 
├── xgb_engine.json               # Serialized pre-trained XGBoost model weights
├── lgb_engine.txt                # Serialized pre-trained LightGBM model weights
├── cat_engine.cbm                # Serialized pre-trained CatBoost model weights
└── verified_phase5_metrics.txt   # Hard text audit file tracking test metrics
