import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, fbeta_score, matthews_corrcoef, classification_report
import matplotlib.pyplot as plt
import os
import joblib
import warnings  # Inyección para limpiar la consola de advertencias secundarias
from pathlib import Path

# Silenciar UserWarnings de nombres de características para un reporte técnico impecable
warnings.filterwarnings('ignore', category=UserWarning)

# ===================================================================
# 1. RUTA RELATIVA Y CARGA DE DATOS DE ATAQUE
# ===================================================================
BASE_DIR = Path(__file__).resolve().parents[2]

print("[CARRIL ATAQUES] Cargando matrices de la Fase 3...")
X_train_raw = np.load(os.path.join(BASE_DIR,'data','final', 'X_train_attack.npy'))
X_test_raw = np.load(os.path.join(BASE_DIR,'data','final', 'X_test_attack.npy'))
y_train = np.load(os.path.join(BASE_DIR,'data','final', 'y_train_attack.npy'))
y_test = np.load(os.path.join(BASE_DIR,'data','final', 'y_test_attack.npy'))

le_attack = joblib.load(os.path.join(BASE_DIR,'models', 'label_encoder_attack.pkl'))

# ===================================================================
# 2. FASE 5: EXTRACCIÓN DE EMBEDDINGS DESDE LA CNN_ATTACK
# ===================================================================
print("\n[Fase 5] Cargando CNN_Attack y extrayendo representaciones latentes (16D)...")
modelo_cnn = load_model(os.path.join(BASE_DIR,'src','Embedding', 'best_cnn_attack_model.keras'))

extractor_embeddings = Model(inputs=modelo_cnn.input, outputs=modelo_cnn.get_layer('Embedding_Attack').output)

X_train_reshaped = np.expand_dims(X_train_raw, axis=-1)
X_test_reshaped = np.expand_dims(X_test_raw, axis=-1)

X_train_embeddings = extractor_embeddings.predict(X_train_reshaped)
X_test_embeddings = extractor_embeddings.predict(X_test_reshaped)

# ===================================================================
# 3. FASE 6: ENTRENAMIENTO DE LIGHTGBM MULTICLASE
# ===================================================================
print("\n[Fase 6] Entrenando LightGBM Multiclase sobre el espacio latente...")
lgb_attack = LGBMClassifier(
    n_estimators=100,
    learning_rate=0.05,
    num_leaves=31,          # Parámetro clave de LightGBM para controlar el crecimiento leaf-wise
    objective='multiclass',
    random_state=42,
    n_jobs=-1,
    verbose=-1              # Desactiva advertencias innecesarias en consola
)

lgb_attack.fit(X_train_embeddings, y_train)
joblib.dump(lgb_attack, os.path.join(BASE_DIR, 'src', 'Results', 'attack_classifier_lgb.pkl'))
print("-> Clasificador LightGBM de ataques guardado con éxito.")

# ===================================================================
# 3.5. FASE 6.5: INFERENCIA CRUZADA PARA DIAGNÓSTICO DE OVERFITTING
# ===================================================================
print("\n[Fase 6.5] Ejecutando inferencia cruzada para análisis de generalización...")
# Predecimos sobre ambas particiones para evaluar la brecha operativa real
y_train_pred = lgb_attack.predict(X_train_embeddings)
y_test_pred = lgb_attack.predict(X_test_embeddings)

# --- Cálculo de Métricas para el Set de Entrenamiento (Train Set) ---
acc_train = accuracy_score(y_train, y_train_pred)
prec_train = precision_score(y_train, y_train_pred, average='macro', zero_division=0)
rec_train = recall_score(y_train, y_train_pred, average='macro', zero_division=0)
f1_train = f1_score(y_train, y_train_pred, average='macro', zero_division=0)
f2_train = fbeta_score(y_train, y_train_pred, beta=2.0, average='macro', zero_division=0)
mcc_train = matthews_corrcoef(y_train, y_train_pred)

# --- CORRECCIÓN: Cálculo de Métricas usando 'y_test_pred' uniformemente ---
acc_test = accuracy_score(y_test, y_test_pred)
prec_test = precision_score(y_test, y_test_pred, average='macro', zero_division=0)
rec_test = recall_score(y_test, y_test_pred, average='macro', zero_division=0)
f1_test = f1_score(y_test, y_test_pred, average='macro', zero_division=0)
f2_test = fbeta_score(y_test, y_test_pred, beta=2.0, average='macro', zero_division=0)
mcc_test = matthews_corrcoef(y_test, y_test_pred)

# ===================================================================
# 4. EXPORTACIÓN VISUAL DEL ANÁLISIS DE SOBREAJUSTE (GRÁFICA)
# ===================================================================
print("\nGenerando gráfica científica de diagnóstico de Overfitting...")
metrics_names = ['Exactitud\n(Accuracy)', 'Precisión\n(Precision)', 'Sensibilidad\n(Recall)', 'F1-Score\n(Macro)', 'F2-Score\n(Defensivo)', 'Coef. Matthews\n(MCC)']
train_metrics = [acc_train, prec_train, rec_train, f1_train, f2_train, mcc_train]
test_metrics = [acc_test, prec_test, rec_test, f1_test, f2_test, mcc_test]

x_indices = np.arange(len(metrics_names))
bar_width = 0.35

fig, ax = plt.subplots(figsize=(12, 6.5), dpi=300)

bars_train = ax.bar(x_indices - bar_width/2, train_metrics, bar_width, label='Entrenamiento (Train Set)', color='#1f77b4', alpha=0.9)
bars_test = ax.bar(x_indices + bar_width/2, test_metrics, bar_width, label='Prueba Independiente (Test Set)', color='#ff7f0e', alpha=0.9)

ax.set_ylabel('Valor Numérico de la Métrica (0.00 - 1.00)', fontsize=11, fontweight='bold')
ax.set_title('Evaluación de Generalización y Diagnóstico de Overfitting - LightGBM (Carril A)', fontsize=13, fontweight='bold', pad=15)
ax.set_xticks(x_indices)
ax.set_xticklabels(metrics_names, fontsize=10)
ax.set_ylim(0, 1.15)
ax.legend(loc='lower left', fontsize=11)
ax.grid(True, linestyle=':', alpha=0.5)

def label_bars(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.4f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 4),  
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

label_bars(bars_train)
label_bars(bars_test)

plt.tight_layout()
ruta_grafico_ml = os.path.join(BASE_DIR, 'src', 'Results', 'diagnostic_overfitting_lgbm_attack.png')
plt.savefig(ruta_grafico_ml, dpi=300, bbox_inches='tight')
plt.show()
print(f"¡Gráfica de control de LightGBM guardada exitosamente en: '{ruta_grafico_ml}'!")

# ===================================================================
# 5. REPORTE TÉCNICO FORMAL COMPARATIVO PARA TU TESIS
# ===================================================================
print("\n" + "="*70)
print("   CONTRALORÍA DE CONVERGENCIA EN MACHINE LEARNING SUPERFICIAL")
print("="*70)
print(f" Métrica Analizada       |  Train Set  |  Test Set   |  Brecha (Gap)")
print("-"*70)
print(f" Exactitud (Accuracy)    |   {acc_train:.4f}    |   {acc_test:.4f}    |   {abs(acc_train - acc_test):.4f}")
print(f" Precisión Macro         |   {prec_train:.4f}    |   {prec_test:.4f}    |   {abs(prec_train - prec_test):.4f}")
print(f" Sensibilidad Macro      |   {rec_train:.4f}    |   {rec_test:.4f}    |   {abs(rec_train - rec_test):.4f}")
print(f" Puntuación F1 Macro     |   {f1_train:.4f}    |   {f1_test:.4f}    |   {abs(f1_train - f1_test):.4f}")
print(f" Puntuación F2 Defensiva |   {f2_train:.4f}    |   {f2_test:.4f}    |   {abs(f2_train - f2_test):.4f}")
print(f" Coef. de Matthews (MCC) |   {mcc_train:.4f}    |   {mcc_test:.4f}    |   {abs(mcc_train - mcc_test):.4f}")
print("="*70)

print("\n=== DESGLOSE DETALLADO DE DESEMPEÑO POR MACRO-CLASE (TEST SET) ===")
# CORRECCIÓN: Mapeo de reporte detallado utilizando 'y_test_pred'
print(classification_report(y_test, y_test_pred, target_names=le_attack.classes_, zero_division=0))
print("="*70)