# Ödev 1: Dengesiz Veri Seti (Imbalanced Dataset)

Kredi başvurularında **riskli müşterileri** (azınlık sınıf) daha iyi yakalamak için dengesiz veri tekniklerini lojistik regresyon üzerinde karşılaştırdım.

## Veri seti

- **Kaynak:** German Credit Risk ([`german_credit.csv`](german_credit.csv))
- **Boyut:** 1000 satır, 20 özellik ve 1 hedef sütun
- **Hedef:** `class`. `bad` → **1 (Risk)**, `good` → **0 (Safe)**
- **Dağılım:** %70 Safe, %30 Risk

## Yöntem

1. **Veri bölme:** Stratified olarak %60 train, %20 validation ve %20 test.
2. **Ön işleme** (`ColumnTransformer` ile):
   - Sayısal sütunlar: median imputation ve `StandardScaler`
   - Kategorik sütunlar: most-frequent imputation ve `OneHotEncoder`
3. **Model:** `LogisticRegression(max_iter=1000)`
4. **Denenen teknikler:**
   - Baseline (tekniksiz)
   - Class Weights (`class_weight='balanced'`)
   - Threshold Moving (eşik 0.2)
   - Random Oversampling (`imblearn`)
   - SMOTE-NC (kategorik sütunları destekleyen SMOTE)

Veri sızıntısını önlemek için oversampling adımları `imblearn.Pipeline` içinde tutuldu. Böylece yalnızca eğitim verisine uygulandılar.

## Sonuçlar (test seti, Risk sınıfı)

| Teknik | Accuracy | Precision | Recall | F1 |
|--------|---------:|----------:|-------:|---:|
| Baseline | 0.75 | 0.59 | 0.57 | 0.58 |
| **Class Weights** | 0.71 | 0.51 | 0.77 | **0.61** |
| Threshold Moving (0.2) | 0.66 | 0.46 | **0.85** | 0.60 |
| Random Oversampling | 0.71 | 0.51 | 0.70 | 0.59 |
| SMOTE-NC | 0.69 | 0.48 | 0.53 | 0.51 |

## Çıkarımlar

- **En yüksek F1:** Class Weights (0.61). Recall ile Precision arasında en dengeli seçenek bu.
- **En yüksek Recall:** Threshold Moving (0.85). Riskli müşterilerin çoğunu yakalıyor ama Precision 0.46'ya düşüyor.
- Recall yükseldikçe Precision ve Accuracy düşüyor. Hangi tekniğin seçileceği, riskli bir müşteriyi kaçırmanın (FN) maliyetine göre belirlenmeli.
- SMOTE-NC bu veri setinde baseline'ın da gerisinde kaldı.

## Çalıştırma

```bash
pip install pandas numpy scikit-learn imbalanced-learn matplotlib jupyter
jupyter notebook german-credit-homework.ipynb
```
