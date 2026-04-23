"""
Customer Churn Prediction
Author: Gunjan
Description: ML classification model to predict customer churn
             using multiple algorithms with feature analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve, accuracy_score)
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
COLORS = ['#6c63ff','#ff6584','#43aa8b','#f59f00']

# ── Data Generation ────────────────────────────────────────
def generate_churn_data(n=2000):
    np.random.seed(42)
    tenure      = np.random.randint(1, 72, n)
    monthly_fee = np.random.uniform(500, 5000, n)
    support_calls = np.random.randint(0, 15, n)
    contract    = np.random.choice(['Monthly','Annual','Two-Year'], n, p=[0.5,0.3,0.2])
    internet    = np.random.choice(['Fiber','DSL','None'], n, p=[0.5,0.35,0.15])
    payment     = np.random.choice(['Auto','Manual'], n, p=[0.45,0.55])
    num_services = np.random.randint(1, 8, n)
    satisfaction = np.random.randint(1, 6, n)

    cont_mult = {'Monthly':0.4,'Annual':0.2,'Two-Year':0.05}
    churn_prob = (
        0.3 - tenure * 0.004 +
        support_calls * 0.04 +
        [cont_mult[c] for c in contract] +
        (monthly_fee > 3000).astype(float) * 0.15 +
        (satisfaction < 3).astype(float) * 0.25 -
        num_services * 0.02 +
        np.random.normal(0, 0.1, n)
    )
    churn = (np.clip(churn_prob, 0, 1) > 0.4).astype(int)

    df = pd.DataFrame({
        'tenure': tenure, 'monthly_fee': monthly_fee.round(2),
        'support_calls': support_calls, 'contract': contract,
        'internet_service': internet, 'payment_method': payment,
        'num_services': num_services, 'satisfaction_score': satisfaction,
        'total_charges': (tenure * monthly_fee).round(2), 'churn': churn
    })
    return df


def prepare_features(df):
    df = df.copy()
    df['contract_enc'] = df['contract'].map({'Monthly':0,'Annual':1,'Two-Year':2})
    df['internet_enc'] = df['internet_service'].map({'None':0,'DSL':1,'Fiber':2})
    df['payment_enc']  = df['payment_method'].map({'Manual':0,'Auto':1})
    df['high_support'] = (df['support_calls'] > 5).astype(int)
    df['long_tenure']  = (df['tenure'] > 24).astype(int)
    return df, ['tenure','monthly_fee','support_calls','num_services',
                'satisfaction_score','total_charges','contract_enc',
                'internet_enc','payment_enc','high_support','long_tenure']


def train_and_evaluate(df):
    df, features = prepare_features(df)
    X, y = df[features], df['churn']
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s  = scaler.transform(X_te)

    models = {
        'Logistic Regression': LogisticRegression(max_iter=500, random_state=42),
        'Random Forest':       RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting':   GradientBoostingClassifier(n_estimators=100, random_state=42),
        'SVM':                 SVC(probability=True, random_state=42),
    }

    results = {}
    print(f"\n{'Model':<25} {'Accuracy':>10} {'ROC-AUC':>10} {'Precision':>10} {'Recall':>10}")
    print("-" * 68)
    for name, model in models.items():
        X_t = X_tr_s; X_v = X_te_s
        model.fit(X_t, y_tr)
        preds = model.predict(X_v)
        proba = model.predict_proba(X_v)[:,1]
        acc = accuracy_score(y_te, preds)
        auc = roc_auc_score(y_te, proba)
        rep = classification_report(y_te, preds, output_dict=True)
        results[name] = {'model':model,'preds':preds,'proba':proba,
                         'acc':acc,'auc':auc,'report':rep,'X_te':X_te_s}
        print(f"  {name:<23} {acc:>10.4f} {auc:>10.4f} "
              f"{rep['1']['precision']:>10.4f} {rep['1']['recall']:>10.4f}")

    best = max(results, key=lambda k: results[k]['auc'])
    print(f"\n  Best: {best} (AUC={results[best]['auc']:.4f})")
    return results, y_te, features, df


def plot_results(results, y_te, features, df):
    import os; os.makedirs('outputs', exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle('Customer Churn Prediction Analysis', fontsize=16, fontweight='bold')

    # 1. ROC Curves
    for i, (name, r) in enumerate(results.items()):
        fpr, tpr, _ = roc_curve(y_te, r['proba'])
        axes[0,0].plot(fpr, tpr, label=f"{name.split()[0]} (AUC={r['auc']:.3f})",
                       color=COLORS[i], linewidth=2)
    axes[0,0].plot([0,1],[0,1],'k--',linewidth=1)
    axes[0,0].set_title('ROC Curves — All Models', fontweight='bold')
    axes[0,0].set_xlabel('False Positive Rate'); axes[0,0].set_ylabel('True Positive Rate')
    axes[0,0].legend(fontsize=9)

    # 2. Confusion Matrix (Best)
    best = max(results, key=lambda k: results[k]['auc'])
    cm = confusion_matrix(y_te, results[best]['preds'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0,1],
                xticklabels=['No Churn','Churn'], yticklabels=['No Churn','Churn'])
    axes[0,1].set_title(f'Confusion Matrix — {best}', fontweight='bold')
    axes[0,1].set_ylabel('Actual'); axes[0,1].set_xlabel('Predicted')

    # 3. Feature Importance
    rf = results['Random Forest']['model']
    imp = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=True).tail(8)
    axes[1,0].barh(imp.index, imp.values, color='#43aa8b')
    axes[1,0].set_title('Top Feature Importances', fontweight='bold')

    # 4. Churn by Contract Type
    churn_rate = df.groupby('contract')['churn'].mean() * 100
    axes[1,1].bar(churn_rate.index, churn_rate.values, color=COLORS[:3])
    axes[1,1].set_title('Churn Rate by Contract Type', fontweight='bold')
    axes[1,1].set_ylabel('Churn Rate (%)')
    for i, v in enumerate(churn_rate.values):
        axes[1,1].text(i, v + 0.5, f'{v:.1f}%', ha='center', fontweight='bold')

    plt.tight_layout()
    plt.savefig('outputs/churn_prediction.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: outputs/churn_prediction.png")


if __name__ == '__main__':
    print("Generating churn dataset...")
    df = generate_churn_data()
    print(f"Dataset: {df.shape} | Churn Rate: {df['churn'].mean()*100:.1f}%")
    results, y_te, features, df_fe = train_and_evaluate(df)
    plot_results(results, y_te, features, df_fe)
    print("Done!")
