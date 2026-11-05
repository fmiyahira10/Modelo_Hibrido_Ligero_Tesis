# Contrato de captura e inferencia — revisión 2026-10-18

## Estado real

El código adaptado carga CNN+RF, valida entradas y atiende inferencia local. No implementa todavía PCAP -> flujos -> 62 características. No activa respuestas en pfSense, Wazuh o Shuffle. El servicio puede estar activo esperando entradas sin que se esté observando tráfico.

La descarga de `Tesis_gitignore.zip` (2.245.094.098 bytes) no pudo realizarse por el límite de 256 MiB de la integración. Se inspeccionaron los cuatro escaladores/codificadores del repositorio. No se cargaron las CNN ni los RF. Las pruebas con clasificadores sustitutos verifican lógica y transporte, no calidad predictiva.

## Hallazgos que afectan la validez

1. Ataques: 13 entradas; cifrado: 62. El orden verdadero está en `feature_names_in_` de cada escalador; se incluye en `feature-contract.json`.
2. La sonda original toma registros de dos datasets independientes y los empareja artificialmente. Además, las descripciones de sus tres casos no coinciden con sus filtros. No usar esa demostración para métricas conjuntas o evaluación en GNS3.
3. El ejemplo original del motor rellena las 62 características con ceros. No constituye una medición de cifrado.
4. `unificador_vectorataque.py` renombra duraciones/IAT de CICIDS y UNSW sin una conversión explícita visible. Hay que revisar si los Parquet de entrada ya estaban normalizados. No cambiar solo las unidades de inferencia si el entrenamiento conserva otras: podría requerir regenerar los datos y reentrenar.
5. El entrenamiento del carril B asigna VPN/Tor a Cifrado y NonVPN/Non-Tor a No Cifrado. HTTPS no es automáticamente el positivo correcto para esa etiqueta. Registrar por separado transporte TLS y uso de túnel; evaluar cada objetivo según el alcance real del modelo.
6. El LabelEncoder real devuelve 0=Cifrado y 1=No Cifrado. En scripts que usan métricas binarias con `pos_label=1`, el positivo será No Cifrado. No interpretar esas cifras como métricas del cifrado sin corregir la clase positiva.
7. Se conserva el baseline operacional CNN+RF que carga el motor. No se ha elegido automáticamente ResNet, XGBoost o LightGBM; esa selección requiere resultados y artefactos consistentes.

## Entrada por TCP

Dirección: 127.0.0.1:9999, accesible solo dentro de VM-MODELO. No se necesita exponerla en 0.0.0.0 cuando captura e inferencia viven en esta VM. Una conexión por registro; JSON UTF-8 terminado en salto de línea; máximo 65536 bytes. Las entradas y respuestas son completas mediante framing, no por un único recv(4096).

Campos obligatorios:

| Campo | Contenido |
|---|---|
| schema_version | 1 |
| run_id | Identificador de corrida |
| flow_id | Identificador del flujo, único dentro de la corrida |
| source_kind | captured_flow o test_fixture |
| extractor_id | Versión, configuración y perfil de unidades del extractor |
| flow | src_ip, dst_ip, src_port, dst_port, protocol, started_at, ended_at |
| features_attack | 13 números crudos, preferentemente diccionario con nombres |
| features_encryption | 62 números crudos, preferentemente diccionario con nombres |

`protocol` es 6 para TCP o 17 para UDP. Tiempos ISO 8601 con zona horaria. Ambos vectores deben proceder del MISMO flujo bidireccional, con la misma orientación y ventana temporal. El servidor valida estructura pero no puede demostrar la procedencia: la sonda debe garantizarla y conservar PCAP para auditoría.

No enviar etiquetas verdaderas al modelo, ni pasarle arrays ya normalizados. No completar características faltantes con cero. `test_fixture` está rechazado por defecto; solo admitirlo con --allow-test-fixtures para pruebas técnicas separadas de los resultados científicos.

## Para completar la sonda

Elegir y fijar una implementación de extracción compatible con las características CIC usadas al entrenar. Comparar un PCAP conocido con la herramienta de referencia y comprobar cada descriptor: unidades, definición de longitud (payload/cabeceras), orientación, ventanas TCP, IAT y desviación estándar, bulk/subflows e idle. Renombrar columnas de otra herramienta no prueba equivalencia numérica.

Resolver la duda de unidades del entrenamiento antes de fijar el perfil de extracción. Mantener estado de cada flujo a través de la rotación de PCAP: cortar archivos cada N segundos no equivale a terminar todos los flujos en ese instante. Registrar timeouts y criterio de cierre.

En GNS3: NIC de administración en VLAN40; NIC dedicada sin IPv4/IPv6 conectada al espejo, con promiscuo permitido. El espejo debe cubrir los trayectos que realmente se quieren medir. Tráfico entre equipos de un mismo switch de acceso puede no atravesar el core. Verificar cobertura y duplicados con una captura antes de las corridas.

Mantener PCAP con espacio acotado, métricas de pérdida de captura, cola de flujos, conteos de entradas procesadas/rechazadas y latencia desde cierre del flujo. El `inference_ms` del motor mide solamente escalado + DL + ML + correlación: no mide tiempo total desde el primer paquete.

## Salidas y evaluación

`runtime/logs/predictions.jsonl` conserva ambas predicciones, scores, timestamps, metadatos del flujo y latencia del núcleo. Los scores de RF no se declaran probabilidades calibradas. Inicio del servicio registra hashes de los artefactos. Los logs rotan a 20 MB con 10 respaldos; archivar al finalizar cada corrida antes de superar esa retención.

El servicio devuelve `action_executed=none`: detectar no equivale a bloquear. Normal+Cifrado no genera alerta de ataque por sí solo. Conservar la salida de cada carril para evaluar los efectos de la correlación de forma separada.

Los generadores deben registrar run_id, escenario, origen/destino, protocolo/transporte, tiempos y resultado real de ejecución. Un intento fallido o una petición inocua llamada malware no demuestra una infección o exploit exitoso. Asociar verdad de referencia a los flujos con criterios explícitos; no etiquetar toda una ventana como ataque si también contiene tráfico benigno.

El transporte actual procesa una solicitud a la vez; es una base para bajo volumen y hay que medir su capacidad antes de estrés. No hay cola persistente ni deduplicación de reintentos: la futura sonda debe conservar pendientes y el análisis debe deduplicar por (run_id, flow_id) si reenvía solicitudes.
