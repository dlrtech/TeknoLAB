#Dengesiz Veri Seti (Imbalanced Dataset)

**Veri Seti:** German Credit Risk (`german_credit.csv`)
**Amaç:** Lojistik regresyon kullanarak Risk (1) sınıfı için Recall ve F1 skorunu artırmak.

**Uygulanan Yöntemler:**
- Baseline Logistic Regression
- Class Weights (Sınıf Ağırlıkları)
- Threshold Moving (Eşik Kaydırma)
- Random Oversampling
- SMOTE

**Sonuç:** Class Weights yöntemi en yüksek F1 skorunu (0.61) ve en yüksek Recall değerini (0.76) verdi.
