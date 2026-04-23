"""
Data Scientist Take-Home Challenge: Churn Prediction
======================================================

Objetivo: Analizar datos de usuarios de un SaaS y predecir qué usuarios 
van a abandonar el servicio (churn).

Estructura del análisis:
1. EDA - Exploración de datos
2. Feature Engineering - Creación de características
3. Modelado - Entrenamiento de modelo predictivo
4. Interpretación - Análisis de importancia de features
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score, 
    confusion_matrix, classification_report, roc_curve, auc
)
import warnings
warnings.filterwarnings('ignore')

# Configuración de visualización
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ============================================================================
# PARTE 1: CARGA Y EXPLORACIÓN INICIAL
# ============================================================================

print("=" * 80)
print("PARTE 1: EXPLORACIÓN DE DATOS (EDA)")
print("=" * 80)

# Cargar datos
users = pd.read_csv('/mnt/user-data/uploads/users.csv')
usage_logs = pd.read_csv('/mnt/user-data/uploads/usage_logs.csv')
churn_labels = pd.read_csv('/mnt/user-data/uploads/churn_labels.csv')

print("\n📊 ESTRUCTURA DE DATOS")
print("\nUsers shape:", users.shape)
print(users.head())
print("\nUsage Logs shape:", usage_logs.shape)
print(usage_logs.head())
print("\nChurn Labels shape:", churn_labels.shape)
print(churn_labels.head())

# Información general
print("\n" + "=" * 80)
print("INFORMACIÓN GENERAL")
print("=" * 80)
print(f"\nTotal de usuarios: {users.shape[0]}")
print(f"Total de registros de uso: {usage_logs.shape[0]}")
print(f"Tasa de churn: {churn_labels['churned'].mean():.2%}")
print(f"Usuarios que hicieron churn: {churn_labels['churned'].sum()}")

# Verificar valores faltantes
print("\n" + "=" * 80)
print("VALORES FALTANTES")
print("=" * 80)
print("\nUsers:\n", users.isnull().sum())
print("\nUsage Logs:\n", usage_logs.isnull().sum())
print("\nChurn Labels:\n", churn_labels.isnull().sum())

# ============================================================================
# PARTE 1.1: ANÁLISIS EXPLORATORIO DETALLADO
# ============================================================================

print("\n" + "=" * 80)
print("ANÁLISIS EXPLORATORIO DETALLADO")
print("=" * 80)

# Distribución de plan types
print("\n📈 Distribución de Plan Types:")
print(users['plan_type'].value_counts())

# Distribución de países
print("\n📍 Distribución geográfica (Top 10):")
print(users['country'].value_counts().head(10))

# Fecha de signup
users['signup_date'] = pd.to_datetime(users['signup_date'])
print("\n📅 Rango de fechas de signup:")
print(f"Desde: {users['signup_date'].min()} hasta {users['signup_date'].max()}")

# Días desde signup
users['days_since_signup'] = (users['signup_date'].max() - users['signup_date']).dt.days
print("\n⏱️  Estadísticas de días desde signup:")
print(users['days_since_signup'].describe())

# ============================================================================
# CREAR VISUALIZACIONES EDA
# ============================================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Exploración de Datos de Usuarios', fontsize=16, fontweight='bold')

# 1. Distribución de plan types
ax1 = axes[0, 0]
users['plan_type'].value_counts().plot(kind='bar', ax=ax1, color='steelblue')
ax1.set_title('Distribución de Plan Types', fontweight='bold')
ax1.set_xlabel('Plan Type')
ax1.set_ylabel('Cantidad de Usuarios')
ax1.tick_params(axis='x', rotation=45)

# 2. Tasa de churn por plan type
ax2 = axes[0, 1]
churn_by_plan = users.merge(churn_labels, on='user_id').groupby('plan_type')['churned'].agg(['sum', 'count'])
churn_by_plan['rate'] = churn_by_plan['sum'] / churn_by_plan['count']
churn_by_plan['rate'].plot(kind='bar', ax=ax2, color='coral')
ax2.set_title('Tasa de Churn por Plan Type', fontweight='bold')
ax2.set_xlabel('Plan Type')
ax2.set_ylabel('Tasa de Churn')
ax2.tick_params(axis='x', rotation=45)
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1%}'.format(y)))

# 3. Días desde signup
ax3 = axes[1, 0]
users['days_since_signup'].hist(bins=30, ax=ax3, color='green', alpha=0.7)
ax3.set_title('Distribución: Días desde Signup', fontweight='bold')
ax3.set_xlabel('Días')
ax3.set_ylabel('Frecuencia')

# 4. Usuarios vs Churn
ax4 = axes[1, 1]
churn_dist = churn_labels['churned'].value_counts()
colors = ['#2ecc71', '#e74c3c']
ax4.pie(churn_dist.values, labels=['No Churned', 'Churned'], autopct='%1.1f%%', 
        colors=colors, startangle=90)
ax4.set_title('Distribución General de Churn', fontweight='bold')

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/01_eda_overview.png', dpi=300, bbox_inches='tight')
print("\n✅ Visualización 1 guardada: 01_eda_overview.png")
plt.close()

# ============================================================================
# PARTE 2: FEATURE ENGINEERING
# ============================================================================

print("\n" + "=" * 80)
print("PARTE 2: FEATURE ENGINEERING")
print("=" * 80)

# Convertir fecha de usage_logs
usage_logs['date'] = pd.to_datetime(usage_logs['date'])

# Crear features agregadas por usuario
print("\n🔧 Creando features agregadas...")

features_dict = {}

for user_id in users['user_id']:
    user_data = usage_logs[usage_logs['user_id'] == user_id]
    
    if len(user_data) == 0:
        # Usuario sin registros de uso
        features_dict[user_id] = {
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
            'last_login_days_ago': 999999,  # Muy antiguo
        }
    else:
        max_date = usage_logs['date'].max()
        user_max_date = user_data['date'].max()
        
        features_dict[user_id] = {
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

features_df = pd.DataFrame.from_dict(features_dict, orient='index').reset_index()
features_df.rename(columns={'index': 'user_id'}, inplace=True)

# Agregar features de usuarios
features_df = features_df.merge(users[['user_id', 'plan_type', 'days_since_signup', 'country']], 
                                 on='user_id', how='left')

# Agregar target (churn)
features_df = features_df.merge(churn_labels, on='user_id', how='left')

print(f"\n✅ Features creadas. Shape: {features_df.shape}")
print("\nPrimeras filas:")
print(features_df.head())

print("\nEstadísticas de features numéricos:")
print(features_df.describe())

# ============================================================================
# VISUALIZAR RELACIONES CON CHURN
# ============================================================================

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Relación entre Features y Churn', fontsize=16, fontweight='bold')

features_to_plot = ['avg_logins_per_day', 'avg_actions_per_day', 'avg_time_per_day',
                    'days_active', 'last_login_days_ago', 'days_since_signup']

for idx, feature in enumerate(features_to_plot):
    ax = axes[idx // 3, idx % 3]
    
    # Box plot
    churned_data = features_df[features_df['churned'] == 1][feature]
    retained_data = features_df[features_df['churned'] == 0][feature]
    
    bp = ax.boxplot([retained_data, churned_data], labels=['Retained', 'Churned'],
                     patch_artist=True)
    
    for patch, color in zip(bp['boxes'], ['#2ecc71', '#e74c3c']):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_title(f'{feature}', fontweight='bold')
    ax.set_ylabel('Valor')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/02_feature_churn_relationship.png', dpi=300, bbox_inches='tight')
print("\n✅ Visualización 2 guardada: 02_feature_churn_relationship.png")
plt.close()

# ============================================================================
# CORRELACIÓN CON CHURN
# ============================================================================

numeric_features = features_df.select_dtypes(include=[np.number]).columns.tolist()
numeric_features.remove('churned')  # Remover target

correlation_with_churn = features_df[numeric_features + ['churned']].corr()['churned'].drop('churned')
correlation_with_churn = correlation_with_churn.sort_values(ascending=False)

print("\n" + "=" * 80)
print("CORRELACIÓN CON CHURN")
print("=" * 80)
print(correlation_with_churn)

fig, ax = plt.subplots(figsize=(10, 8))
correlation_with_churn.plot(kind='barh', ax=ax, color=['#e74c3c' if x < 0 else '#2ecc71' for x in correlation_with_churn.values])
ax.set_title('Correlación de Features con Churn', fontweight='bold', fontsize=14)
ax.set_xlabel('Correlación de Pearson')
ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/03_correlation_with_churn.png', dpi=300, bbox_inches='tight')
print("\n✅ Visualización 3 guardada: 03_correlation_with_churn.png")
plt.close()

# ============================================================================
# PARTE 3: PREPARACIÓN DE DATOS Y MODELADO
# ============================================================================

print("\n" + "=" * 80)
print("PARTE 3: MODELADO")
print("=" * 80)

# Preparar datos para modelado
X = features_df.copy()

# Codificar variables categóricas
X['plan_type_encoded'] = pd.Categorical(X['plan_type']).codes
X = pd.get_dummies(X, columns=['country'], drop_first=True)

# Features a usar
feature_cols = [col for col in X.columns if col not in ['user_id', 'plan_type', 'country', 'churned']]
X = X[feature_cols]
y = features_df['churned'].values

print(f"\n📊 Features finales: {X.shape[1]}")
print(f"Muestras: {X.shape[0]}")
print(f"Balance de clases: {np.sum(y)} churned, {np.sum(1-y)} retained")

# Train-test split (time-based preferred - usar días_since_signup como proxy)
median_days = features_df['days_since_signup'].median()
train_idx = features_df['days_since_signup'] >= median_days
test_idx = features_df['days_since_signup'] < median_days

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

print(f"\n🔄 Split de datos (basado en días desde signup):")
print(f"Train: {X_train.shape[0]} muestras ({y_train.mean():.1%} churn)")
print(f"Test: {X_test.shape[0]} muestras ({y_test.mean():.1%} churn)")

# Escalar features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================================================
# ENTRENAR MODELOS
# ============================================================================

print("\n" + "=" * 80)
print("ENTRENAMIENTO DE MODELOS")
print("=" * 80)

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
    'XGBoost': XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, verbosity=0)
}

results = {}

for model_name, model in models.items():
    print(f"\n🚀 Entrenando {model_name}...")
    
    if model_name == 'Logistic Regression':
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Evaluar
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    results[model_name] = {
        'model': model,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'cm': confusion_matrix(y_test, y_pred)
    }
    
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    print(f"  ROC-AUC: {roc_auc:.4f}")

# ============================================================================
# COMPARACIÓN DE MODELOS
# ============================================================================

print("\n" + "=" * 80)
print("RESUMEN DE RESULTADOS")
print("=" * 80)

comparison_df = pd.DataFrame({
    'Precision': [results[m]['precision'] for m in results],
    'Recall': [results[m]['recall'] for m in results],
    'F1-Score': [results[m]['f1'] for m in results],
    'ROC-AUC': [results[m]['roc_auc'] for m in results]
}, index=results.keys())

print("\n", comparison_df.round(4))

# Visualizar comparación
fig, axes = plt.subplots(1, 4, figsize=(16, 5))
fig.suptitle('Comparación de Modelos', fontsize=16, fontweight='bold')

metrics = ['Precision', 'Recall', 'F1-Score', 'ROC-AUC']
for idx, metric in enumerate(metrics):
    ax = axes[idx]
    comparison_df[metric].plot(kind='bar', ax=ax, color=['#3498db', '#2ecc71', '#f39c12'])
    ax.set_title(metric, fontweight='bold')
    ax.set_ylabel('Score')
    ax.tick_params(axis='x', rotation=45)
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/04_model_comparison.png', dpi=300, bbox_inches='tight')
print("\n✅ Visualización 4 guardada: 04_model_comparison.png")
plt.close()

# ============================================================================
# CONFUSION MATRIX Y ROC CURVE
# ============================================================================

fig, axes = plt.subplots(1, 3, figsize=(16, 4))
fig.suptitle('Análisis Detallado de Modelos', fontsize=16, fontweight='bold')

for idx, (model_name, model_results) in enumerate(results.items()):
    ax = axes[idx]
    cm = model_results['cm']
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False,
                xticklabels=['No Churn', 'Churn'], yticklabels=['No Churn', 'Churn'])
    ax.set_title(f'{model_name}\nConfusion Matrix', fontweight='bold')
    ax.set_ylabel('Actual')
    ax.set_xlabel('Predicción')

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/05_confusion_matrices.png', dpi=300, bbox_inches='tight')
print("✅ Visualización 5 guardada: 05_confusion_matrices.png")
plt.close()

# ROC Curves
fig, ax = plt.subplots(figsize=(10, 8))

for model_name, model_results in results.items():
    fpr, tpr, _ = roc_curve(y_test, model_results['y_pred_proba'])
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.3f})', linewidth=2)

ax.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=2)
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('ROC Curves - Comparación de Modelos', fontweight='bold', fontsize=14)
ax.legend(loc='lower right', fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/06_roc_curves.png', dpi=300, bbox_inches='tight')
print("✅ Visualización 6 guardada: 06_roc_curves.png")
plt.close()

# ============================================================================
# PARTE 4: INTERPRETACIÓN - FEATURE IMPORTANCE
# ============================================================================

print("\n" + "=" * 80)
print("PARTE 4: INTERPRETACIÓN - FEATURE IMPORTANCE")
print("=" * 80)

# Random Forest Feature Importance
rf_model = results['Random Forest']['model']
rf_importances = rf_model.feature_importances_
rf_indices = np.argsort(rf_importances)[::-1][:15]

print("\n🔍 Top 15 Features (Random Forest):")
for i, idx in enumerate(rf_indices, 1):
    print(f"{i:2d}. {feature_cols[idx]:30s} - {rf_importances[idx]:.4f}")

# XGBoost Feature Importance
xgb_model = results['XGBoost']['model']
xgb_importances = xgb_model.feature_importances_
xgb_indices = np.argsort(xgb_importances)[::-1][:15]

print("\n🔍 Top 15 Features (XGBoost):")
for i, idx in enumerate(xgb_indices, 1):
    print(f"{i:2d}. {feature_cols[idx]:30s} - {xgb_importances[idx]:.4f}")

# Visualizar feature importance
fig, axes = plt.subplots(1, 2, figsize=(16, 8))
fig.suptitle('Feature Importance Analysis', fontsize=16, fontweight='bold')

# Random Forest
top_n = 15
rf_top_indices = np.argsort(rf_importances)[::-1][:top_n]
rf_top_names = [feature_cols[i] for i in rf_top_indices]
rf_top_importances = rf_importances[rf_top_indices]

axes[0].barh(rf_top_names, rf_top_importances, color='steelblue')
axes[0].set_xlabel('Importance Score')
axes[0].set_title('Random Forest - Top 15 Features', fontweight='bold')
axes[0].invert_yaxis()

# XGBoost
xgb_top_indices = np.argsort(xgb_importances)[::-1][:top_n]
xgb_top_names = [feature_cols[i] for i in xgb_top_indices]
xgb_top_importances = xgb_importances[xgb_top_indices]

axes[1].barh(xgb_top_names, xgb_top_importances, color='coral')
axes[1].set_xlabel('Importance Score')
axes[1].set_title('XGBoost - Top 15 Features', fontweight='bold')
axes[1].invert_yaxis()

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/07_feature_importance.png', dpi=300, bbox_inches='tight')
print("\n✅ Visualización 7 guardada: 07_feature_importance.png")
plt.close()

# ============================================================================
# PREDICCIONES Y ANÁLISIS DE RIESGOS
# ============================================================================

print("\n" + "=" * 80)
print("PREDICCIONES Y SEGMENTACIÓN DE RIESGO")
print("=" * 80)

# Usar el mejor modelo (XGBoost)
best_model = results['XGBoost']['model']
best_model_name = 'XGBoost'

# Predecir en todos los datos
X_full = features_df.copy()
X_full['plan_type_encoded'] = pd.Categorical(X_full['plan_type']).codes
X_full_encoded = pd.get_dummies(X_full, columns=['country'], drop_first=True)
X_full_encoded = X_full_encoded[feature_cols]

predictions = best_model.predict_proba(X_full_encoded)[:, 1]
features_df['churn_risk_score'] = predictions

# Clasificar riesgo
features_df['risk_segment'] = pd.cut(features_df['churn_risk_score'], 
                                      bins=[0, 0.33, 0.67, 1.0],
                                      labels=['Low Risk', 'Medium Risk', 'High Risk'])

print("\n📊 Distribución de Segmentos de Riesgo:")
print(features_df['risk_segment'].value_counts().sort_index())

# Usuarios de alto riesgo
print("\n⚠️  TOP 10 USUARIOS CON MAYOR RIESGO DE CHURN:")
high_risk = features_df.nlargest(10, 'churn_risk_score')[['user_id', 'plan_type', 'days_since_signup', 
                                                            'avg_logins_per_day', 'last_login_days_ago', 
                                                            'churn_risk_score', 'churned']]
print(high_risk.to_string())

# Visualizar segmentación de riesgo
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Análisis de Riesgo de Churn', fontsize=16, fontweight='bold')

# 1. Distribución de risk scores
ax1 = axes[0, 0]
ax1.hist(features_df['churn_risk_score'], bins=30, color='steelblue', alpha=0.7, edgecolor='black')
ax1.set_xlabel('Churn Risk Score')
ax1.set_ylabel('Frecuencia')
ax1.set_title('Distribución de Churn Risk Scores', fontweight='bold')
ax1.grid(True, alpha=0.3)

# 2. Risk segments
ax2 = axes[0, 1]
risk_counts = features_df['risk_segment'].value_counts().sort_index()
colors_risk = ['#2ecc71', '#f39c12', '#e74c3c']
ax2.bar(risk_counts.index, risk_counts.values, color=colors_risk)
ax2.set_ylabel('Cantidad de Usuarios')
ax2.set_title('Usuarios por Segmento de Riesgo', fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y')

# 3. Tasa de churn real por segmento
ax3 = axes[1, 0]
actual_churn_by_risk = features_df.groupby('risk_segment')['churned'].mean()
ax3.bar(actual_churn_by_risk.index, actual_churn_by_risk.values, color=colors_risk)
ax3.set_ylabel('Tasa de Churn Real')
ax3.set_title('Tasa de Churn Real por Segmento', fontweight='bold')
ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1%}'.format(y)))
ax3.grid(True, alpha=0.3, axis='y')

# 4. Risk score vs logins
ax4 = axes[1, 1]
scatter = ax4.scatter(features_df['avg_logins_per_day'], features_df['churn_risk_score'],
                     c=features_df['churned'], cmap='RdYlGn_r', s=100, alpha=0.6, edgecolor='black')
ax4.set_xlabel('Promedio de Logins por Día')
ax4.set_ylabel('Churn Risk Score')
ax4.set_title('Risk Score vs Actividad de Login', fontweight='bold')
ax4.grid(True, alpha=0.3)
cbar = plt.colorbar(scatter, ax=ax4)
cbar.set_label('Churned (1=Yes, 0=No)')

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/08_risk_analysis.png', dpi=300, bbox_inches='tight')
print("\n✅ Visualización 8 guardada: 08_risk_analysis.png")
plt.close()

# ============================================================================
# REPORTE DE CLASIFICACIÓN DETALLADO
# ============================================================================

print("\n" + "=" * 80)
print(f"REPORTE DE CLASIFICACIÓN DETALLADO ({best_model_name})")
print("=" * 80)

y_pred_best = results[best_model_name]['y_pred']
print("\n", classification_report(y_test, y_pred_best, target_names=['No Churn', 'Churn']))

# ============================================================================
# GUARDAR RESULTADOS Y MODELO
# ============================================================================

import pickle

print("\n" + "=" * 80)
print("GUARDANDO RESULTADOS")
print("=" * 80)

# Guardar modelo
with open('/mnt/user-data/outputs/best_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)
print("✅ Modelo guardado: best_model.pkl")

# Guardar predictions
predictions_df = features_df[['user_id', 'plan_type', 'churn_risk_score', 'risk_segment', 'churned']]
predictions_df.to_csv('/mnt/user-data/outputs/churn_predictions.csv', index=False)
print("✅ Predicciones guardadas: churn_predictions.csv")

# Guardar features
features_df.to_csv('/mnt/user-data/outputs/features_engineered.csv', index=False)
print("✅ Features guardadas: features_engineered.csv")

print("\n" + "=" * 80)
print("✅ ANÁLISIS COMPLETADO")
print("=" * 80)
print(f"\n📁 Todas las visualizaciones y resultados se encuentran en /mnt/user-data/outputs/")
