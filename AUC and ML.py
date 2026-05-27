# -*- coding: utf-8 -*-
"""
Created on Wed May 27 09:02:24 2026

@author: USER
"""

import os
import random
import pandas as pd
import numpy as np
import scipy.stats as st
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
import matplotlib.pyplot as plt
import warnings
import sys
warnings.filterwarnings('ignore')

# ==========================================
# 🌟 완벽한 재현성을 위한 Random Seed 전역 고정
# ==========================================
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    
seed_everything(42)

# 🌟 [저널 제출용 궁극의 폰트/출력 세팅] 
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150 
plt.rcParams['pdf.fonttype'] = 42 
plt.rcParams['ps.fonttype'] = 42

# ==========================================
# 1. DeLong's Test 함수
# ==========================================
def compute_delong_pvalue(y_true, preds_A, preds_B):
    def compute_midrank(x):
        J = np.argsort(x)
        Z = x[J]
        N = len(x)
        T = np.zeros(N, dtype=float)
        i = 0
        while i < N:
            j = i
            while j < N and Z[j] == Z[i]:
                j += 1
            T[i:j] = 0.5 * (i + j - 1)
            i = j
        T2 = np.empty(N, dtype=float)
        T2[J] = T + 1
        return T2

    def fast_delong(y_true, preds):
        pos = preds[y_true == 1]
        neg = preds[y_true == 0]
        m = len(pos)
        n = len(neg)
        theta = roc_auc_score(y_true, preds)
        
        r_pos = compute_midrank(pos)
        r_neg = compute_midrank(neg)
        r_all = compute_midrank(preds)
        
        V10 = np.empty(m)
        for i in range(m):
            V10[i] = (r_all[y_true == 1][i] - r_pos[i]) / n
        V01 = np.empty(n)
        for i in range(n):
            V01[i] = 1 - (r_all[y_true == 0][i] - r_neg[i]) / m
            
        return theta, V10, V01, m, n

    theta_A, V10_A, V01_A, m, n = fast_delong(y_true, preds_A)
    theta_B, V10_B, V01_B, _, _ = fast_delong(y_true, preds_B)
    
    S10 = np.cov(V10_A, V10_B)[0, 1] if m > 1 else 0
    S01 = np.cov(V01_A, V01_B)[0, 1] if n > 1 else 0
    S10_A = np.var(V10_A)
    S10_B = np.var(V10_B)
    S01_A = np.var(V01_A)
    S01_B = np.var(V01_B)
    
    var_A = S10_A/m + S01_A/n
    var_B = S10_B/m + S01_B/n
    cov = S10/m + S01/n
    
    z = (theta_A - theta_B) / np.sqrt(np.maximum(var_A + var_B - 2*cov, 1e-8))
    p_value = 2 * (1 - st.norm.cdf(abs(z)))
    return p_value

# ==========================================
# 2. 평가지표 계산 함수 & 부트스트래핑(실시간 중계)
# ==========================================
def calculate_metrics(y_true, y_pred_prob):
    auroc = roc_auc_score(y_true, y_pred_prob)
    auprc = average_precision_score(y_true, y_pred_prob)
    brier = brier_score_loss(y_true, y_pred_prob)
    return auroc, auprc, brier

def get_roc_curve_with_ci(y_true, y_score, n_bootstrap=500, desc="Bootstrapping"):
    from sklearn.metrics import roc_curve
    fpr_grid = np.linspace(0, 1, 100)
    tpr_list = []
    
    fpr, tpr, _ = roc_curve(y_true, y_score)
    base_tpr = np.interp(fpr_grid, fpr, tpr)
    base_tpr[0] = 0.0
    
    rng = np.random.default_rng(42)
    indices = np.arange(len(y_true))
    
    print(f"      [{desc}] 부트스트래핑 신뢰구간 연산: ", end="")
    sys.stdout.flush()
    for i in range(n_bootstrap):
        boot_idx = rng.choice(indices, size=len(indices), replace=True)
        if len(np.unique(y_true[boot_idx])) < 2:
            continue
        f, t, _ = roc_curve(y_true[boot_idx], y_score[boot_idx])
        t_interp = np.interp(fpr_grid, f, t)
        t_interp[0] = 0.0
        tpr_list.append(t_interp)
        
        if (i + 1) % 100 == 0:
            print(f"{i + 1}.. ", end="")
            sys.stdout.flush()
            
    print("완료!")
    
    tpr_arr = np.array(tpr_list)
    tpr_lower = np.percentile(tpr_arr, 2.5, axis=0)
    tpr_upper = np.percentile(tpr_arr, 97.5, axis=0)
    
    return fpr_grid, base_tpr, tpr_lower, tpr_upper

# ==========================================
# 3. 데이터 로드 및 전처리 가드레일
# ==========================================
print("🔄 데이터 로드 및 전처리 시작...")
df = pd.read_excel('./Dataset/Final_data_analysis_with_Age_Group_4_Final.xlsx')

def clean_and_map_binary(val):
    if pd.isna(val): return np.nan
    s = str(val).strip().lower()
    if s in ['yes', '1', '1.0', 'female']: return 1.0
    if s in ['no', '0', '0.0', 'male']: return 0.0
    try: return float(val)
    except ValueError: return np.nan

if 'Sex' in df.columns:
    df['Sex_encoded'] = df['Sex'].apply(clean_and_map_binary).fillna(0.0)
else:
    df['Sex_encoded'] = 0.0

if 'Salivary pH' in df.columns:
    df.loc[(df['Salivary pH'] < 5.0) | (df['Salivary pH'] > 9.0), 'Salivary pH'] = np.nan

xerogenic_keywords = ['Antidepressive', 'Hypnotics', 'Sedatives', 'Anti-Allergic', 'Urological', 'Anticonvulsants', 'Antihypertensive']
def count_high_confidence_xerogenic(text):
    if pd.isna(text): return 0
    return sum(1 for kw in xerogenic_keywords if kw.lower() in str(text).lower())

df['High_confidence_xerogenic_count'] = df['Current Medication Classes >=10 (PubMed Terms)'].apply(count_high_confidence_xerogenic)

target_features_to_clean = ['Visual analog scale', 'Sticky saliva', 'Dental calculus', 'Tongue coating', 'Halitosis']
for col in target_features_to_clean:
    if col in df.columns:
        df[col] = df[col].apply(clean_and_map_binary)

# ==========================================
# 4. Feature Blocks 구조 선언
# ==========================================
features_m1 = ['Age', 'Sex_encoded']
features_m2 = features_m1 + ['Number of Systemic Disease Categories', 'Number of Current Medication Classes', 'High_confidence_xerogenic_count']
oral_clinical = [c for c in target_features_to_clean if c in df.columns]
features_m3 = features_m2 + oral_clinical
features_m4 = features_m3 + [c for c in ['Salivary pH', 'Salivary Buffer Capacity'] if c in df.columns]

blocks = {'Model 1': features_m1, 'Model 2': features_m2, 'Model 3': features_m3, 'Model 4': features_m4}
outcomes = {'UFR-defined hyposalivation': 'Xerostomia_UFR', 'SFR-defined hyposalivation': 'Xerostomia_SFR'}
saved_oof_predictions = {}
results_table = []

# ==========================================
# 5. 메인 임상 실험 루프 
# ==========================================
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')), 
    ('model', XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42))
])

cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)

print("\n🚀 본격적인 인공지능 모델 학습을 시작합니다... (5-Fold x 5번 반복)")

for outcome_name, target_col in outcomes.items():
    if target_col not in df.columns: continue
    
    print(f"\n📊 타겟 분석 중: {outcome_name}")
    df[target_col] = df[target_col].apply(clean_and_map_binary)
    valid_idx = df[target_col].notna()
    df_valid = df[valid_idx]
    y = df_valid[target_col].values.astype(int)
    
    prev_oof_prob = None
    
    for block_name, feature_list in blocks.items():
        valid_features = [f for f in feature_list if f in df_valid.columns]
        X = df_valid[valid_features]
        
        y_pred_prob_sum = np.zeros(len(y))
        
        print(f"    ▶ {block_name} 25번 반복 학습 중: ", end="")
        sys.stdout.flush()
        
        for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y)):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train = y[train_idx]
            
            pipeline.fit(X_train, y_train)
            preds = pipeline.predict_proba(X_test)[:, 1]
            y_pred_prob_sum[test_idx] += preds
            
            if (fold_idx + 1) % 5 == 0:
                print(f"{(fold_idx + 1)//5}세트.. ", end="")
                sys.stdout.flush()
                
        print("완료!")
        
        y_pred_prob = y_pred_prob_sum / cv.n_repeats
        saved_oof_predictions[f"{outcome_name}_{block_name}"] = (y, y_pred_prob)
        
        auroc, auprc, brier = calculate_metrics(y, y_pred_prob)
        
        if prev_oof_prob is not None:
            p_val_delong = compute_delong_pvalue(y, y_pred_prob, prev_oof_prob)
            p_val_str = f"<0.001***" if p_val_delong < 0.001 else f"{p_val_delong:.3f}"
        else: p_val_str = "Reference"
            
        results_table.append({
            'Outcome': outcome_name, 'Model': block_name, 'AUROC': round(auroc, 3), 
            'p-value (vs Prev Model)': p_val_str, 'AUPRC': round(auprc, 3), 'Brier Score': round(brier, 3)
        })
        prev_oof_prob = y_pred_prob

print("\n" + "="*60)
print("✅ [Table 4] 각 모델 단일 성능 지표")
print("="*60)
display(pd.DataFrame(results_table).set_index(['Outcome', 'Model']))

# ==========================================
# 6. Supplementary Table S2 자동 생성 
# ==========================================
supp_s2_results = []
comparisons = [
    ('Model 1', 'Model 2'),
    ('Model 2', 'Model 3'),
    ('Model 3', 'Model 4'),
    ('Model 1', 'Model 4')
]

for outcome_name in outcomes.keys():
    for model_A_name, model_B_name in comparisons:
        key_A = f"{outcome_name}_{model_A_name}"
        key_B = f"{outcome_name}_{model_B_name}"
        
        if key_A in saved_oof_predictions and key_B in saved_oof_predictions:
            y_true, preds_A = saved_oof_predictions[key_A]
            _, preds_B = saved_oof_predictions[key_B]
            
            auc_A = roc_auc_score(y_true, preds_A)
            auc_B = roc_auc_score(y_true, preds_B)
            delta_auc = auc_B - auc_A
            
            pval = compute_delong_pvalue(y_true, preds_B, preds_A)
            pval_str = "<0.001" if pval < 0.001 else f"{pval:.3f}"
            
            supp_s2_results.append({
                'Outcome': outcome_name,
                'Comparison': f"{model_B_name} vs {model_A_name}",
                'Model A AUROC': f"{auc_A:.3f}",
                'Model B AUROC': f"{auc_B:.3f}",
                'ΔAUROC': f"{delta_auc:.3f}",
                'DeLong p-value': pval_str
            })

print("\n" + "="*60)
print("✅ [Supplementary Table S2] 모델 간 AUROC 비교 (DeLong test)")
print("="*60)
display(pd.DataFrame(supp_s2_results).set_index(['Outcome', 'Comparison']))

# ==========================================
# 7. 🌟 시각화 - 타이틀 O / 타이틀 X 버전 모두 저장
# ==========================================
print("\n🎨 시각화 및 부트스트랩(신뢰구간) 계산 시작...")

file_suffixes = {'UFR-defined hyposalivation': 'Figure_3_UFR', 'SFR-defined hyposalivation': 'Figure_3_SFR'}
title_names = {
    'UFR-defined hyposalivation': '(A) UFR-defined hyposalivation (UFR < 0.1 mL/min)',
    'SFR-defined hyposalivation': '(B) SFR-defined hyposalivation (SFR < 0.7 mL/min)'
}

for out_key in outcomes.keys():
    model1_key = f"{out_key}_Model 1"
    model4_key = f"{out_key}_Model 4"
    
    if model1_key not in saved_oof_predictions or model4_key not in saved_oof_predictions:
        continue

    print(f"\n  ▶ {out_key} 그래프 생성 작업...")
    
    # 공통 데이터 추출
    y_true, y_score_m1 = saved_oof_predictions[model1_key]
    fpr_m1, tpr_m1, low_m1, up_m1 = get_roc_curve_with_ci(y_true, y_score_m1, desc="Model 1")
    auc_m1 = roc_auc_score(y_true, y_score_m1)
    
    _, y_score_m4 = saved_oof_predictions[model4_key]
    fpr_m4, tpr_m4, low_m4, up_m4 = get_roc_curve_with_ci(y_true, y_score_m4, desc="Model 4")
    auc_m4 = roc_auc_score(y_true, y_score_m4)
    
    p_val_final = compute_delong_pvalue(y_true, y_score_m4, y_score_m1)
    delta_auc = auc_m4 - auc_m1
    
    color_m1 = '#2E7D32' if 'UFR' in file_suffixes[out_key] else '#006064'
    color_m4 = '#C62828' if 'UFR' in file_suffixes[out_key] else '#FF7043'
    
    # ---------------------------------------------------------
    # [버전 1] 타이틀 없는 버전 (No Title)
    # ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.5, 7.0))
    ax.grid(True, linestyle='--', alpha=0.5, color='#D3D3D3')
    ax.set_facecolor('#FAFAFA')
    
    ax.plot(fpr_m1, tpr_m1, color=color_m1, lw=3.0, label=f'Model 1: AUC = {auc_m1:.3f}')
    ax.fill_between(fpr_m1, low_m1, up_m1, color=color_m1, alpha=0.08)
    ax.plot(fpr_m4, tpr_m4, color=color_m4, lw=3.0, label=f'Model 4: AUC = {auc_m4:.3f}')
    ax.fill_between(fpr_m4, low_m4, up_m4, color=color_m4, alpha=0.1)
    ax.plot([0, 1], [0, 1], color='#888888', linestyle='--', lw=1.5)
    
    ax.set_xlabel('1 - Specificity', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_ylabel('Sensitivity', fontsize=12, fontweight='bold', labelpad=10)
    
    ax.text(0.05, 0.95, f'DeLong P = {p_val_final:.3f}\nΔAUC = {delta_auc:.3f}', 
            fontsize=12, fontweight='bold', va='top', 
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='#D3D3D3', alpha=0.9))
    
    ax.legend(loc='lower right', fontsize=11, frameon=True, facecolor='white', edgecolor='none')
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])
    
    plt.tight_layout()
    
    base_filename_no_title = f'./{file_suffixes[out_key]}_Final_NoTitle'
    plt.savefig(base_filename_no_title + '.png', dpi=600, bbox_inches='tight')
    plt.savefig(base_filename_no_title + '.tiff', dpi=600, format='tiff', bbox_inches='tight')
    plt.savefig(base_filename_no_title + '.pdf', format='pdf', bbox_inches='tight') 
    plt.close(fig)
    print(f"    - 저장 완료 (No Title): {base_filename_no_title}.png/.pdf/.tiff")

    # ---------------------------------------------------------
    # [버전 2] 타이틀 있는 버전 (With Title - A/B 라벨 포함)
    # ---------------------------------------------------------
    fig2, ax2 = plt.subplots(figsize=(7.5, 7.0))
    ax2.grid(True, linestyle='--', alpha=0.5, color='#D3D3D3')
    ax2.set_facecolor('#FAFAFA')
    
    ax2.plot(fpr_m1, tpr_m1, color=color_m1, lw=3.0, label=f'Model 1: AUC = {auc_m1:.3f}')
    ax2.fill_between(fpr_m1, low_m1, up_m1, color=color_m1, alpha=0.08)
    ax2.plot(fpr_m4, tpr_m4, color=color_m4, lw=3.0, label=f'Model 4: AUC = {auc_m4:.3f}')
    ax2.fill_between(fpr_m4, low_m4, up_m4, color=color_m4, alpha=0.1)
    ax2.plot([0, 1], [0, 1], color='#888888', linestyle='--', lw=1.5)
    
    # 🌟 타이틀 추가
    ax2.set_title(title_names[out_key], fontsize=14, fontweight='bold', pad=15)
    ax2.set_xlabel('1 - Specificity', fontsize=12, fontweight='bold', labelpad=10)
    ax2.set_ylabel('Sensitivity', fontsize=12, fontweight='bold', labelpad=10)
    
    ax2.text(0.05, 0.95, f'DeLong P = {p_val_final:.3f}\nΔAUC = {delta_auc:.3f}', 
            fontsize=12, fontweight='bold', va='top', 
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='#D3D3D3', alpha=0.9))
    
    ax2.legend(loc='lower right', fontsize=11, frameon=True, facecolor='white', edgecolor='none')
    ax2.set_xlim([-0.01, 1.01])
    ax2.set_ylim([-0.01, 1.01])
    
    plt.tight_layout()
    
    base_filename_with_title = f'./{file_suffixes[out_key]}_Final_WithTitle'
    plt.savefig(base_filename_with_title + '.png', dpi=600, bbox_inches='tight')
    plt.savefig(base_filename_with_title + '.tiff', dpi=600, format='tiff', bbox_inches='tight')
    plt.savefig(base_filename_with_title + '.pdf', format='pdf', bbox_inches='tight') 
    plt.show() # 타이틀 있는 버전 화면 출력
    plt.close(fig2)
    print(f"    - 저장 완료 (With Title): {base_filename_with_title}.png/.pdf/.tiff")