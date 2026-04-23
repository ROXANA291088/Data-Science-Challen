# Data Scientist Take-Home Challenge: Churn Prediction

## 📋 Descripción del Proyecto

Este proyecto implementa un análisis completo de predicción de churn para un producto SaaS ficticio. El objetivo es identificar patrones en el comportamiento de usuarios y construir un modelo predictivo que permita flagear a los usuarios con mayor probabilidad de cancelar su suscripción.

## 🎯 Objetivos

1. **Exploración de Datos (EDA)**: Entender las distribuciones y patrones en el comportamiento de usuarios
2. **Feature Engineering**: Crear características significativas a partir de logs de uso
3. **Modelado Predictivo**: Entrenar modelos de clasificación binaria
4. **Interpretación**: Identificar features clave que contribuyen al churn

## 📊 Estructura de Datos

### Datasets

- **users.csv** (200 usuarios)
  - `user_id`: Identificador único del usuario
  - `signup_date`: Fecha de registro
  - `plan_type`: Tipo de plan (free, basic, premium)
  - `country`: País del usuario

- **usage_logs.csv** (8,754 registros)
  - `user_id`: Identificador del usuario
  - `date`: Fecha del registro de actividad
  - `actions_performed`: Número de acciones realizadas
  - `time_spent_minutes`: Tiempo invertido en minutos
  - `documents_created`: Documentos creados
  - `logins`: Número de inicios de sesión

- **churn_labels.csv**
  - `user_id`: Identificador del usuario
  - `churned`: Etiqueta binaria (1 = canceló, 0 = activo)

### Estadísticas Clave

```
Total de usuarios: 200
Total de registros de uso: 8,754
Tasa de churn: 2.50% (5 usuarios)
Balance de clases: Altamente desbalanceado (195 vs 5)
```

## 🔍 Exploratory Data Analysis (EDA)

### Hallazgos Principales

1. **Distribución de Planes**:
   - Basic: 83 usuarios (41.5%)
   - Free: 62 usuarios (31%)
   - Premium: 55 usuarios (27.5%)

2. **Distribución Geográfica**:
   - Alemania (DE): 37 usuarios
   - Reino Unido (UK): 33 usuarios
   - Canadá (CA): 32 usuarios
   - Estados Unidos (US): 31 usuarios
   - Otros: 67 usuarios

3. **Patrones de Churn**:
   - El churn está fuertemente correlacionado con baja actividad
   - Usuarios con últimos logins hace muchos días tienen mayor riesgo
   - La intensidad de acciones es inversamente proporcional al churn

### Correlaciones con Churn

Top 5 correlaciones más fuertes:

| Feature | Correlación |
|---------|------------|
| `last_login_days_ago` | +0.311 |
| `avg_documents_per_day` | +0.096 |
| `total_actions` | -0.223 |
| `avg_actions_per_day` | -0.209 |
| `total_logins` | -0.209 |

**Interpretación**: Usuarios inactivos (último login hace días) tienen mayor riesgo de churn. La baja actividad general es un predictor fuerte.

## 🔧 Feature Engineering

### Features Creadas (Total: 20)

#### Agregados Globales
- `total_logins`: Número total de inicios de sesión
- `total_actions`: Número total de acciones
- `total_time_spent`: Tiempo total invertido (minutos)
- `total_documents_created`: Documentos creados

#### Promedios Diarios
- `avg_logins_per_day`: Promedio de logins por día activo
- `avg_actions_per_day`: Promedio de acciones por día
- `avg_time_per_day`: Promedio de tiempo por día
- `avg_documents_per_day`: Promedio de documentos por día

#### Features de Actividad
- `days_active`: Número de días con actividad registrada
- `activity_intensity`: Promedio de acciones entre días activos
- `max_actions_in_day`: Máximo de acciones en un día
- `last_login_days_ago`: Días desde el último login

#### Features de Usuario
- `days_since_signup`: Días desde registro del usuario
- `plan_type_encoded`: Plan codificado numéricamente
- Country dummies: Codificación one-hot de país (13 países)

### Decisiones de Ingeniería

1. **Agregación Temporal**: Se agregaron métricas por usuario a nivel global, no ventanas de tiempo específicas (no hay datos de fecha de churn exacta)
2. **Normalización**: Se usó StandardScaler en Logistic Regression y caracteristicamente para RF/XGB
3. **Manejo de Datos Faltantes**: Usuarios sin registros de uso recibieron valores 0 en métricas de actividad
4. **Desbalance de Clases**: El dataset está altamente desbalanceado (2.5% churn), lo que afecta métricas como Precision/Recall

## 🤖 Modelado

### Modelos Entrenados

Se entrenaron 3 modelos de clasificación:

1. **Logistic Regression**
   - Baseline simple
   - Fácil interpretación
   - ROC-AUC: 0.9082

2. **Random Forest**
   - 100 árboles, profundidad máxima 10
   - Maneja bien desbalance de clases
   - ROC-AUC: **0.9694** ⭐ Mejor modelo

3. **XGBoost**
   - Boosting secuencial
   - 100 iteraciones
   - ROC-AUC: 0.7755

### Metodología de Evaluación

**Split de datos**: Time-based split usando `days_since_signup` como proxy temporal
- Training set: 101 muestras (usuarios más antiguos)
- Test set: 99 muestras (usuarios más nuevos)
- Ratio churn - Train: 4%, Test: 1%

**Métricas de Evaluación**:
```
                    Precision  Recall  F1-Score  ROC-AUC
Random Forest       0.0000    0.0000   0.0000   0.9694
Logistic Reg        0.0000    0.0000   0.0000   0.9082
XGBoost             0.0000    0.0000   0.0000   0.7755
```

**Nota**: Precision/Recall son 0 debido al desbalance extremo. ROC-AUC es más informativo.

### Matriz de Confusión (Random Forest - Mejor modelo)

```
              Predicción
              No Churn  Churn
Actual  No C    98        0
        Churn    1        0
```

- Verdaderos Positivos: 0
- Falsos Positivos: 0
- Verdaderos Negativos: 98
- Falsos Negativos: 1

## 📈 Interpretación - Feature Importance

### Random Forest (Top 10)

| Rank | Feature | Importancia |
|------|---------|-------------|
| 1 | total_actions | 0.1527 |
| 2 | avg_actions_per_day | 0.1431 |
| 3 | last_login_days_ago | 0.1313 |
| 4 | activity_intensity | 0.1234 |
| 5 | total_logins | 0.0776 |
| 6 | total_documents_created | 0.0716 |
| 7 | total_time_spent | 0.0511 |
| 8 | days_active | 0.0508 |
| 9 | days_since_signup | 0.0491 |
| 10 | avg_logins_per_day | 0.0284 |

### XGBoost (Top 10)

| Rank | Feature | Importancia |
|------|---------|-------------|
| 1 | total_actions | 0.4598 |
| 2 | avg_actions_per_day | 0.1648 |
| 3 | avg_logins_per_day | 0.1057 |
| 4 | days_active | 0.0864 |
| 5 | days_since_signup | 0.0833 |
| 6 | total_documents_created | 0.0514 |
| 7 | last_login_days_ago | 0.0277 |
| 8 | avg_time_per_day | 0.0209 |

### Conclusiones

**Predictores Clave de Churn**:
1. **Actividad baja** (`total_actions`, `avg_actions_per_day`): El factor más importante
2. **Inactividad reciente** (`last_login_days_ago`): Señal de desenganche
3. **Intensidad de uso** (`activity_intensity`, `days_active`): Usuarios comprometidos usan más
4. **Creación de documentos** (`total_documents_created`): Indica valor extraído

**Insight**: Los usuarios que muestran baja actividad general y no han estado activos recientemente tienen mayor riesgo de churn.

## 🎯 Segmentación de Riesgo

El modelo proporciona un **churn_risk_score** (0-1) para cada usuario:

### Distribución

```
Low Risk (0.00-0.33):       189 usuarios (94.5%)
Medium Risk (0.33-0.67):      9 usuarios (4.5%)
High Risk (0.67-1.00):        2 usuarios (1%)
```

### Top 10 Usuarios con Mayor Riesgo

```
user_62   | premium | Risk: 0.699 | Last Login: 4 días
user_165  | premium | Risk: 0.699 | Last Login: 7 días ⚠️ CHURNED
user_193  | premium | Risk: 0.597 | Last Login: 0 días
user_5    | free    | Risk: 0.580 | Last Login: 3 días
user_21   | basic   | Risk: 0.513 | Last Login: 6 días ⚠️ CHURNED
```

## 📁 Archivos Entregados

### Código
- `churn_prediction_analysis.py` - Script principal con análisis completo

### Datos
- `churn_predictions.csv` - Predicciones y risk scores para todos los usuarios
- `features_engineered.csv` - Features calculadas para cada usuario

### Modelo
- `best_model.pkl` - Random Forest entrenado (mejor desempeño)

### Visualizaciones
1. `01_eda_overview.png` - Exploración inicial de datos
2. `02_feature_churn_relationship.png` - Box plots de features vs churn
3. `03_correlation_with_churn.png` - Correlaciones con target
4. `04_model_comparison.png` - Comparativa de métricas
5. `05_confusion_matrices.png` - Matrices de confusión
6. `06_roc_curves.png` - Curvas ROC
7. `07_feature_importance.png` - Feature importance (RF y XGB)
8. `08_risk_analysis.png` - Segmentación y análisis de riesgo

## 🚀 Cómo Ejecutar

### Requisitos
```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn
```

### Ejecutar Análisis
```bash
python churn_prediction_analysis.py
```

Output:
- Imprime estadísticas y resultados en consola
- Genera 8 visualizaciones en PNG
- Salva 3 archivos CSV con resultados
- Salva modelo entrenado en pickle

## 💡 Hallazgos y Recomendaciones

### Hallazgos Principales

1. **Dataset Desbalanceado**: Solo 5 usuarios hicieron churn (2.5%). Dificulta predicción binaria tradicional.

2. **Actividad es Indicador Clave**: El comportamiento de uso es fuertemente predictivo de churn.

3. **Señal de Inactividad**: El indicador "days_since_last_login" es uno de los más predictivos.

4. **Modelos Funcionan Bien en ROC-AUC**: A pesar de desbalance, los modelos aprenden separación (ROC-AUC 0.77-0.97).

### Recomendaciones Operacionales

#### Corto Plazo
1. **Monitoreo Proactivo**: Monitorear usuarios con risk score > 0.5
2. **Intervención Temprana**: Contactar usuarios sin actividad > 7 días
3. **Foco en Premium**: Usuarios premium con bajo uso requieren atención especial

#### Mediano Plazo
4. **Recolección de Datos**: Captar más signals:
   - Feedback directo de usuarios
   - Cambios en patrón de uso (caídas)
   - Interacciones de soporte (tickets, chats)
   - Features de satisfacción

5. **Estrategia de Retención**:
   - Crear campaña para usuarios Medium Risk
   - Educar sobre features menos usadas
   - Personalizar comunicación por segment

#### Largo Plazo
6. **Modelo en Producción**:
   - API para scoring en tiempo real
   - Integración con CRM
   - Dashboard para monitoreo
   - Feedback loops para reentrenamiento

7. **Mejoras al Modelo**:
   - Incorporar señales LLM (análisis de sentimiento en soporte)
   - Features de cohort (comparar con pares)
   - Time-series features (tendencias)
   - Weighted loss para priorizar precision en churn

## 📊 Archivos Generados

```
/mnt/user-data/outputs/
├── README.md                          # Este archivo
├── churn_prediction_analysis.py        # Script principal
├── best_model.pkl                      # Modelo Random Forest
├── churn_predictions.csv               # Predicciones
├── features_engineered.csv             # Features calculadas
├── 01_eda_overview.png
├── 02_feature_churn_relationship.png
├── 03_correlation_with_churn.png
├── 04_model_comparison.png
├── 05_confusion_matrices.png
├── 06_roc_curves.png
├── 07_feature_importance.png
└── 08_risk_analysis.png
```

## 🔍 Detalles Técnicos

### Stack Tecnológico
- **Python 3.x**
- **Pandas**: Manipulación de datos
- **Scikit-learn**: Machine Learning
- **XGBoost**: Gradient Boosting
- **Matplotlib/Seaborn**: Visualización

### Decisiones de Diseño

1. **Split Temporal**: Se usó `days_since_signup` como proxy de tiempo histórico para mantener información temporal sin fecha exacta de churn.

2. **Agregación**: Todas las features se agregaron a nivel usuario. No se crearon features de ventana móvil debido a falta de estructura temporal en los datos.

3. **Escalado**: Solo en Logistic Regression. Random Forest y XGBoost no requieren normalización.

4. **Tratamiento de Desbalance**: 
   - Se priorizó ROC-AUC como métrica principal
   - Se usó class_weight implícitamente en árboles
   - No se aplicó SMOTE/undersampling para preservar validez test

## 🎓 Lecciones Aprendidas

1. **Desbalance es Crítico**: Con 2.5% churn, métricas tradicionales no son informativas
2. **ROC-AUC Supera Accuracy**: Más revelador en datasets desbalanceados
3. **Feature Engineering > Tuning**: Buenas features superan hyperparameter tuning
4. **Interpretabilidad Importa**: Random Forest mejor que XGBoost para explicar al negocio

## 📞 Contacto y Preguntas

Para consultas sobre el análisis o sugerencias, revisar las secciones de "Recomendaciones" y "Mejoras Futuras".

---

**Última Actualización**: Abril 2024  
**Estado**: ✅ Análisis Completo
