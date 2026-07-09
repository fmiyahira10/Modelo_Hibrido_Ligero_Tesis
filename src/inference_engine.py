import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
import joblib
from pathlib import Path

# Desactivar logs innecesarios de TensorFlow para una consola limpia
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class InferenceAndCorrelationEngine:
    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.tesis_dir = Path(base_dir)
        
        print("[-] Inicializando Motor de Correlación Híbrido Ligero...")
        self._load_artifacts()
        self._build_extractors()
        print("[+] Todos los módulos y modelos cargados con éxito en memoria.")

    def _load_artifacts(self):
        """Carga escaladores, codificadores y clasificadores de ensamble."""
        try:
            # Carril A: Ataques
            self.scaler_attack = joblib.load(self.tesis_dir / 'models' / 'robust_scaler_attack.pkl')
            self.le_attack = joblib.load(self.tesis_dir / 'models' / 'label_encoder_attack.pkl')
            self.clf_rf_attack = joblib.load(self.tesis_dir / 'src' / 'Results' / 'attack_classifier_rf.pkl')
            
            # Carril B: Cifrado
            self.scaler_encryption = joblib.load(self.tesis_dir / 'models' / 'robust_scaler_encryption.pkl')
            self.le_encryption = joblib.load(self.tesis_dir / 'models' / 'label_encoder_encryption.pkl')
            self.clf_lgb_encryption = joblib.load(self.tesis_dir / 'src' / 'Results' / 'encryption_classifier_lgb.pkl')
            
            # Modelos Keras Completos
            self.cnn_full_attack = load_model(self.tesis_dir / 'src' / 'Embedding' /'best_cnn_attack_model.keras')
            self.cnn_full_encryption = load_model(self.tesis_dir / 'src' / 'Embedding' /'best_cnn_encryption_model.keras')
        except Exception as e:
            raise RuntimeError(f"Error crítico al cargar artefactos desde {self.tesis_dir}: {str(e)}")

    def _build_extractors(self):
        """Trunca las capas de clasificación de las CNNs para usarlas como extractores latentes."""
        self.extractor_attack = Model(
            inputs=self.cnn_full_attack.input, 
            outputs=self.cnn_full_attack.get_layer('Embedding_Attack').output
        )
        self.extractor_encryption = Model(
            inputs=self.cnn_full_encryption.input, 
            outputs=self.cnn_full_encryption.get_layer('Embedding_Encryption').output
        )

    def process_flow(self, features_attack_raw, features_encryption_raw):
        """
        Procesa de manera híbrida un flujo de red simulado o real.
        features_attack_raw: Arreglo de 13 características crudas para el carril A.
        features_encryption_raw: Arreglo de 62 características crudas para el carril B.
        """
        # Asegurar formato matricial de 2D para los escaladores de scikit-learn
        feat_at_2d = np.array(features_attack_raw).reshape(1, -1)
        feat_enc_2d = np.array(features_encryption_raw).reshape(1, -1)
        
        # 1. ESCALADO ROBUSTO INDEPENDIENTE (Prevención de distorsión por outliers)
        at_scaled = self.scaler_attack.transform(feat_at_2d)
        enc_scaled = self.scaler_encryption.transform(feat_enc_2d)
        
        # 2. REFORMATO 3D PARA REDES CONVOLUCIONALES (muestras, variables, 1)
        at_3d = np.expand_dims(at_scaled, axis=-1)
        enc_3d = np.expand_dims(enc_scaled, axis=-1)
        
        # 3. EXTRACCIÓN DE CARACTERÍSTICAS LATENTES (Fase 5)
        emb_attack = self.extractor_attack.predict(at_3d, verbose=0)
        emb_encryption = self.extractor_encryption.predict(enc_3d, verbose=0)
        
        # 4. INFERENCIA MEDIANTE CLASIFICADORES DE ENSAMBLE (Fase 6)
        pred_at_code = self.clf_rf_attack.predict(emb_attack)[0]
        pred_enc_code = self.clf_lgb_encryption.predict(emb_encryption)[0]
        
        # Decodificación de los nombres reales de las etiquetas
        label_attack = self.le_attack.inverse_transform([pred_at_code])[0]
        label_encryption = self.le_encryption.inverse_transform([pred_enc_code])[0]
        
        # 5. MATRIZ DE DECISIÓN DEL MOTOR DE CORRELACIÓN LOGICA (Fase 11)
        veredicto_final = self._correlate_verdicts(label_attack, label_encryption)
        
        return {
            'Carril_Aataques': label_attack,
            'Carril_Bcifrado': label_encryption,
            'Veredicto_IDS': veredicto_final
        }

    def _correlate_verdicts(self, attack, encryption):
        """Aplica las reglas lógicas institucionales para la mitigación de falsos positivos."""
        if attack == "Normal" and encryption == "No Cifrado":
            return "Tráfico Benigno Estándar - Permitido"
            
        elif attack == "Normal" and encryption == "Cifrado":
            return "Alerta Lenta: Uso de Túnel Criptográfico (VPN/Tor) sin anomalía aparente - Monitorear"
            
        elif attack != "Normal" and encryption == "No Cifrado":
            return f"CRÍTICO: Intrusión Activa Detectada en Texto Plano [{attack}] - Bloqueo Inmediato"
            
        elif attack != "Normal" and encryption == "Cifrado":
            # Regla de oro de tu matriz: disipar el falso positivo si es malware/exploit enmascarado
            if attack in ["Malware_Exploits", "Access_Attacks"]:
                return f"ALERTA SEVERA: Amenaza Altamente Ofuscada Encapsulada [{attack} sobre canal {encryption}] - Activando Inspección Especial"
            else:
                return f"CRÍTICO: Intrusión Detectada en Canal Cifrado [{attack}] - Bloquear IP de Origen"
        
        return "Estado Indeterminado - Revisión de Sonda"

# ===================================================================
# INSTANCIACIÓN DE PRUEBA DE INFERENCIA
# ===================================================================
if __name__ == "__main__":
    # Define la raíz de tu proyecto local
    RAIZ_PROYECTO = Path(__file__).resolve().parents[1]
    
    # Instanciamos el motor
    engine = InferenceAndCorrelationEngine(base_dir=RAIZ_PROYECTO)
    
    # Simulamos la llegada de un flujo sospechoso de DoS (13 features crudas) que viaja en texto plano (62 features crudas)
    flujo_ataque_simulado_13d = [1.5, 4.0, 2.0, 500.0, 120.0, 125.0, 60.0, 0.5, 0.2, 1024, 1024, 4.0, 413.3]
    flujo_cifrado_simulado_62d = [0.0] * 62 # Llenamos de ceros simulados para la prueba
    
    # Procesamos el flujo
    resultado = engine.process_flow(flujo_ataque_simulado_13d, flujo_cifrado_simulado_62d)
    
    print("\n" + "="*50)
    print("        RESULTADO DEL MOTOR DE CORRELACIÓN CORE")
    print("="*50)
    print(f" Diagnóstico Carril A (Ataques) : {resultado['Carril_Aataques']}")
    print(f" Diagnóstico Carril B (Cifrado) : {resultado['Carril_Bcifrado']}")
    print(f" -> VEREDICTO INTEGRADO NIDS    : {resultado['Veredicto_IDS']}")
    print("="*50)