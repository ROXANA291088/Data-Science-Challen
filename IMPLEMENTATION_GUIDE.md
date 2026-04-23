# 🚀 GUÍA DE IMPLEMENTACIÓN
## Cómo Usar el Modelo en Producción

---

## 📦 Requisitos Previos

```bash
# Dependencias necesarias
pip install pandas numpy scikit-learn pickle

# Versiones recomendadas
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
```

---

## 1️⃣ CARGAR Y USAR EL MODELO

### Opción A: Predicción Simple (Python)

```python
import pickle
import pandas as pd
import numpy as np

# Cargar modelo entrenado
with open('best_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Cargar features calculadas anteriormente
features_df = pd.read_csv('features_engineered.csv')

# Para un usuario específico
user_id = 'user_165'
user_features = features_df[features_df['user_id'] == user_id].iloc[0]

# Hacer predicción
X_user = user_features[feature_cols].values.reshape(1, -1)
churn_probability = model.predict_proba(X_user)[0, 1]
risk_segment = 'High Risk' if churn_probability > 0.67 else \
               'Medium Risk' if churn_probability > 0.33 else 'Low Risk'

print(f"Usuario: {user_id}")
print(f"Probabilidad de Churn: {churn_probability:.2%}")
print(f"Segmento: {risk_segment}")
```

### Opción B: Predicción en Batch

```python
# Procesar todos los usuarios
predictions = model.predict_proba(features_df[feature_cols])[:, 1]
features_df['churn_risk_score'] = predictions
features_df['risk_segment'] = pd.cut(
    predictions,
    bins=[0, 0.33, 0.67, 1.0],
    labels=['Low Risk', 'Medium Risk', 'High Risk']
)

# Guardar resultados
features_df.to_csv('predictions_latest.csv', index=False)

# Usuarios de alto riesgo
high_risk = features_df[features_df['risk_segment'] == 'High Risk']
print(f"Usuarios de alto riesgo: {len(high_risk)}")
print(high_risk[['user_id', 'churn_risk_score', 'last_login_days_ago']])
```

---

## 2️⃣ CREAR CARACTERÍSTICAS PARA NUEVOS USUARIOS

Si tienes nuevos usuarios y logs de uso:

```python
def engineer_features(user_id, usage_logs_df, users_df):
    """
    Calcular features para un usuario nuevo
    """
    import pandas as pd
    
    # Filtrar datos del usuario
    user_data = usage_logs_df[usage_logs_df['user_id'] == user_id]
    user_info = users_df[users_df['user_id'] == user_id].iloc[0]
    
    if len(user_data) == 0:
        # Usuario sin logs
        features = {
            'total_logins': 0,
            'total_actions': 0,
            'total_time_spent': 0,
            'total_documents_created': 0,
            'avg_logins_per_day': 0,
            'avg_actions_per_day': 0,
            'avg_time_per_day': 0,
            'avg_documents_per_day': 0,
            'max_actions_in_day': 0,
            'days_active': 0,
            'activity_intensity': 0,
            'last_login_days_ago': 999999,
        }
    else:
        max_date = usage_logs_df['date'].max()
        user_max_date = user_data['date'].max()
        
        features = {
            'total_logins': user_data['logins'].sum(),
            'total_actions': user_data['actions_performed'].sum(),
            'total_time_spent': user_data['time_spent_minutes'].sum(),
            'total_documents_created': user_data['documents_created'].sum(),
            'avg_logins_per_day': user_data['logins'].mean(),
            'avg_actions_per_day': user_data['actions_performed'].mean(),
            'avg_time_per_day': user_data['time_spent_minutes'].mean(),
            'avg_documents_per_day': user_data['documents_created'].mean(),
            'max_actions_in_day': user_data['actions_performed'].max(),
            'days_active': len(user_data),
            'activity_intensity': user_data['actions_performed'].sum() / len(user_data),
            'last_login_days_ago': (max_date - user_max_date).days,
        }
    
    # Agregar features de usuario
    features['plan_type_encoded'] = pd.Categorical([user_info['plan_type']]).codes[0]
    features['days_since_signup'] = (user_info['signup_date'] - users_df['signup_date'].min()).days
    features['country'] = user_info['country']
    
    return features

# Ejemplo de uso
new_user_features = engineer_features('user_201', usage_logs, users)
print(new_user_features)
```

---

## 3️⃣ CREAR API PARA SCORING EN TIEMPO REAL

### Opción A: Flask (Simple)

```python
from flask import Flask, jsonify, request
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

# Cargar modelo al iniciar
with open('best_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Cargar feature names
feature_cols = [
    'total_logins', 'total_actions', 'total_time_spent',
    'total_documents_created', 'avg_logins_per_day',
    'avg_actions_per_day', 'avg_time_per_day',
    'avg_documents_per_day', 'max_actions_in_day',
    'days_active', 'activity_intensity', 'last_login_days_ago',
    'plan_type_encoded', 'days_since_signup',
    'country_AU', 'country_BR', 'country_CA', 'country_DE',
    'country_IN', 'country_UK', 'country_US'
]

@app.route('/predict_churn', methods=['POST'])
def predict_churn():
    """
    POST /predict_churn
    
    Body:
    {
        "user_id": "user_165",
        "total_logins": 30,
        "total_actions": 200,
        ...
    }
    """
    try:
        data = request.get_json()
        
        # Preparar features en orden correcto
        X = np.array([data.get(col, 0) for col in feature_cols]).reshape(1, -1)
        
        # Predicción
        churn_prob = model.predict_proba(X)[0, 1]
        risk_segment = 'High Risk' if churn_prob > 0.67 else \
                      'Medium Risk' if churn_prob > 0.33 else 'Low Risk'
        
        return jsonify({
            'user_id': data.get('user_id'),
            'churn_risk_score': float(churn_prob),
            'risk_segment': risk_segment,
            'status': 'success'
        })
    
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 400

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'model': 'churn_prediction_v1'})

if __name__ == '__main__':
    app.run(port=5000, debug=False)
```

**Usar la API:**

```bash
curl -X POST http://localhost:5000/predict_churn \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user_165",
    "total_logins": 20,
    "total_actions": 150,
    "total_time_spent": 500,
    "total_documents_created": 5,
    "avg_logins_per_day": 0.5,
    "avg_actions_per_day": 3.5,
    "avg_time_per_day": 12.5,
    "avg_documents_per_day": 0.1,
    "max_actions_in_day": 15,
    "days_active": 45,
    "activity_intensity": 3.5,
    "last_login_days_ago": 7,
    "plan_type_encoded": 2,
    "days_since_signup": 134,
    "country_AU": 0,
    "country_BR": 0,
    "country_CA": 0,
    "country_DE": 0,
    "country_IN": 0,
    "country_UK": 0,
    "country_US": 0
  }'
```

### Opción B: FastAPI (Production-ready)

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import numpy as np

app = FastAPI(title="Churn Prediction API", version="1.0")

# Cargar modelo
with open('best_model.pkl', 'rb') as f:
    model = pickle.load(f)

class PredictionRequest(BaseModel):
    user_id: str
    total_logins: float
    total_actions: float
    total_time_spent: float
    total_documents_created: float
    avg_logins_per_day: float
    avg_actions_per_day: float
    avg_time_per_day: float
    avg_documents_per_day: float
    max_actions_in_day: float
    days_active: int
    activity_intensity: float
    last_login_days_ago: int
    plan_type_encoded: int
    days_since_signup: int
    country_AU: int = 0
    country_BR: int = 0
    country_CA: int = 0
    country_DE: int = 0
    country_IN: int = 0
    country_UK: int = 0
    country_US: int = 0

class PredictionResponse(BaseModel):
    user_id: str
    churn_risk_score: float
    risk_segment: str
    status: str

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """Predict churn risk for a user"""
    try:
        # Convertir a array ordenado
        features = [
            request.total_logins, request.total_actions, request.total_time_spent,
            request.total_documents_created, request.avg_logins_per_day,
            request.avg_actions_per_day, request.avg_time_per_day,
            request.avg_documents_per_day, request.max_actions_in_day,
            request.days_active, request.activity_intensity, request.last_login_days_ago,
            request.plan_type_encoded, request.days_since_signup,
            request.country_AU, request.country_BR, request.country_CA, request.country_DE,
            request.country_IN, request.country_UK, request.country_US
        ]
        
        X = np.array(features).reshape(1, -1)
        churn_prob = float(model.predict_proba(X)[0, 1])
        
        if churn_prob > 0.67:
            risk_segment = "High Risk"
        elif churn_prob > 0.33:
            risk_segment = "Medium Risk"
        else:
            risk_segment = "Low Risk"
        
        return PredictionResponse(
            user_id=request.user_id,
            churn_risk_score=churn_prob,
            risk_segment=risk_segment,
            status="success"
        )
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/health")
def health():
    """Health check"""
    return {"status": "healthy", "model": "churn_prediction_v1"}

# Usar: uvicorn app:app --reload
```

---

## 4️⃣ INTEGRACIÓN CON SISTEMAS EXISTENTES

### Con Zapier (Sin Código)

1. Trigger: Nuevo usuario en CRM
2. Action: Llamar API de scoring
3. Conditional: Si risk_score > 0.5
4. Action: Enviar email a CS team

### Con Python (Periodicamente)

```python
import schedule
import time
from datetime import datetime

def scoring_job():
    """Job ejecutado cada día a las 9 AM"""
    print(f"[{datetime.now()}] Ejecutando scoring...")
    
    # Cargar datos nuevos
    usage_logs = pd.read_csv('usage_logs_updated.csv')
    users = pd.read_csv('users.csv')
    
    # Calcular features para todos
    predictions = []
    for user_id in users['user_id']:
        features = engineer_features(user_id, usage_logs, users)
        pred = predict_churn(features)
        predictions.append({
            'user_id': user_id,
            'risk_score': pred,
            'timestamp': datetime.now()
        })
    
    # Guardar
    pd.DataFrame(predictions).to_csv('churn_scores_latest.csv')
    
    # Alertar si hay nuevos high-risk
    high_risk = [p for p in predictions if p['risk_score'] > 0.7]
    if high_risk:
        send_slack_alert(f"⚠️ {len(high_risk)} usuarios nuevos en alto riesgo")
    
    print(f"✅ Scoring completado a las {datetime.now()}")

# Schedule para ejecutar
schedule.every().day.at("09:00").do(scoring_job)

while True:
    schedule.run_pending()
    time.sleep(60)
```

---

## 5️⃣ MONITOREO Y REENTRENAMIENTO

### Dashboard (en Streamlit)

```python
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Churn Risk Dashboard", layout="wide")

st.title("🎯 Churn Risk Monitoring Dashboard")

# Cargar datos
predictions = pd.read_csv('churn_predictions.csv')
features = pd.read_csv('features_engineered.csv')

# KPIs
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Users", len(predictions))
with col2:
    high_risk = len(predictions[predictions['churn_risk_score'] > 0.7])
    st.metric("High Risk Users", high_risk)
with col3:
    avg_risk = predictions['churn_risk_score'].mean()
    st.metric("Avg Risk Score", f"{avg_risk:.2%}")
with col4:
    actual_churn = predictions['churned'].sum()
    st.metric("Actual Churn", actual_churn)

# Distribución de riesgo
st.subheader("Risk Score Distribution")
fig, ax = plt.subplots()
ax.hist(predictions['churn_risk_score'], bins=30)
st.pyplot(fig)

# Usuarios de alto riesgo
st.subheader("⚠️ High Risk Users - Immediate Action")
high_risk_df = predictions[
    predictions['churn_risk_score'] > 0.7
][['user_id', 'plan_type', 'churn_risk_score', 'last_login_days_ago']].sort_values(
    'churn_risk_score', ascending=False
)
st.dataframe(high_risk_df)

# Exportar para acción
if st.button("📧 Export High Risk Users for Outreach"):
    high_risk_df.to_csv('high_risk_users.csv', index=False)
    st.success("Exported to high_risk_users.csv")
```

**Ejecutar:**
```bash
streamlit run churn_dashboard.py
```

### Reentrenamiento Automático

```python
def should_retrain():
    """
    Chequear si modelo necesita reentrenamiento
    """
    # Cargar modelo viejo
    with open('best_model.pkl', 'rb') as f:
        old_model = pickle.load(f)
    
    # Cargar datos nuevos
    X_new = pd.read_csv('new_data.csv')[feature_cols]
    y_new = pd.read_csv('new_data.csv')['churned']
    
    # Evaluar
    old_auc = roc_auc_score(y_new, old_model.predict_proba(X_new)[:, 1])
    
    if old_auc < 0.92:  # Threshold
        print(f"⚠️ AUC dropped to {old_auc:.4f}, retraining...")
        retrain_model()
        return True
    else:
        print(f"✅ AUC ok: {old_auc:.4f}")
        return False

def retrain_model():
    """Entrenar modelo nuevamente con nuevos datos"""
    # Cargar datos
    X = pd.read_csv('all_data.csv')[feature_cols]
    y = pd.read_csv('all_data.csv')['churned']
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Entrenar
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluar
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    print(f"✅ New model AUC: {auc:.4f}")
    
    # Guardar
    with open('best_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    # Log
    with open('retraining_log.txt', 'a') as f:
        f.write(f"Retrained at {datetime.now()}, AUC: {auc:.4f}\n")

# Ejecutar cada mes
schedule.every().month.do(should_retrain)
```

---

## 6️⃣ CHECKLIST DE DEPLOYMENT

- [ ] Modelo cargado y testeado localmente
- [ ] API creada y documentada
- [ ] Tests unitarios escritos
- [ ] Feature engineering pipeline validado
- [ ] Datos de entrada en formato correcto
- [ ] Monitoreo setup (alertas, logs)
- [ ] Reentrenamiento schedule configurado
- [ ] Dashboard deployado
- [ ] Documentación actualizada
- [ ] Team training completado

---

## 7️⃣ TROUBLESHOOTING

### Problema: API lenta

**Solución:**
```python
# Agregar caching
from functools import lru_cache

@lru_cache(maxsize=1000)
def cached_predict(features_tuple):
    return model.predict_proba([features_tuple])[0, 1]
```

### Problema: Features desconocidas

**Solución:**
```python
# Validar entrada
required_features = set(feature_cols)
input_features = set(request.dict().keys())

if not required_features.issubset(input_features):
    missing = required_features - input_features
    raise ValueError(f"Missing features: {missing}")
```

### Problema: Model drift

**Solución:**
```python
# Comparar distribuciones
from scipy import stats

old_dist = pickle.load(open('feature_dist_baseline.pkl'))
new_dist = X_new.describe()

for col in X_new.columns:
    ks_stat = stats.ks_2samp(
        old_dist[col],
        new_dist[col]
    )
    if ks_stat.pvalue < 0.05:
        print(f"⚠️ Drift detected in {col}")
```

---

## 📞 SOPORTE

Para preguntas o problemas:
1. Revisar README.md (documentación técnica)
2. Revisar PRODUCTION_IMPROVEMENTS.md (roadmap)
3. Ejecutar tests: `python -m pytest tests/`
4. Contactar al equipo de data science

---

**Last Updated**: April 21, 2024  
**Status**: Production Ready ✅
