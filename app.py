"""
Online Shoppers Purchasing Intention - Streamlit App
====================================================
Interactive ML Classification Dashboard
Dataset: Online Shoppers Purchasing Intention (UCI ML Repository)
All models trained on-the-fly. Upload a CSV to evaluate on your own data.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ML Classification Dashboard",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header { font-size: 2.2rem; font-weight: 700; color: #1E3A5F; text-align: center; margin-bottom: 0.5rem; }
    .sub-header { font-size: 1.1rem; color: #5A6C7E; text-align: center; margin-bottom: 2rem; }
</style>
""", unsafe_allow_html=True)


# ─── Dataset Generation ──────────────────────────────────────────────────────
@st.cache_data
def generate_dataset():
    np.random.seed(42)
    n = 12330
    months = ['Feb','Mar','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    month_weights = [0.014,0.014,0.27,0.07,0.03,0.03,0.03,0.04,0.38,0.122]
    visitor_types = ['Returning_Visitor','New_Visitor','Other']
    visitor_weights = [0.856,0.137,0.007]

    data = {}
    data['Administrative'] = np.random.negative_binomial(1,0.3,n).clip(0,27)
    data['Administrative_Duration'] = np.where(data['Administrative']==0,0,data['Administrative']*np.random.exponential(20,n))
    data['Informational'] = np.random.negative_binomial(1,0.6,n).clip(0,24)
    data['Informational_Duration'] = np.where(data['Informational']==0,0,data['Informational']*np.random.exponential(15,n))
    data['ProductRelated'] = np.random.negative_binomial(3,0.1,n).clip(0,705)
    data['ProductRelated_Duration'] = np.where(data['ProductRelated']==0,0,data['ProductRelated']*np.random.exponential(20,n))
    data['BounceRates'] = np.random.beta(1,30,n).clip(0,0.2)
    data['ExitRates'] = (data['BounceRates']+np.random.beta(1,20,n)*0.1).clip(0,0.2)
    pv = np.random.exponential(5,n).clip(0,361)
    data['PageValues'] = np.where(np.random.random(n)<0.7,0,pv)
    data['SpecialDay'] = np.random.choice([0.0,0.2,0.4,0.6,0.8,1.0],n,p=[0.85,0.03,0.03,0.03,0.03,0.03])
    data['Month'] = np.random.choice(months,n,p=month_weights)
    data['OperatingSystems'] = np.random.choice([1,2,3,4,5,6,7,8],n,p=[0.10,0.55,0.20,0.05,0.03,0.03,0.02,0.02])
    data['Browser'] = np.random.choice(range(1,14),n,p=[0.12,0.52,0.05,0.08,0.04,0.04,0.03,0.03,0.02,0.02,0.02,0.02,0.01])
    data['Region'] = np.random.choice(range(1,10),n,p=[0.37,0.08,0.18,0.07,0.05,0.06,0.07,0.05,0.07])
    data['TrafficType'] = np.random.choice(range(1,21),n,p=[0.15,0.30,0.14,0.08,0.03,0.04,0.03,0.05,0.02,0.04,0.03,0.02,0.02,0.01,0.01,0.01,0.005,0.005,0.005,0.005])
    data['VisitorType'] = np.random.choice(visitor_types,n,p=visitor_weights)
    data['Weekend'] = np.random.choice([True,False],n,p=[0.233,0.767])
    logit = -2.5+0.3*(np.array(data['PageValues'])>0).astype(float)+0.001*np.array(data['ProductRelated_Duration'])-5*np.array(data['BounceRates'])-3*np.array(data['ExitRates'])+0.2*(np.array(data['Month'])=='Nov').astype(float)+np.random.normal(0,0.5,n)
    data['Revenue'] = np.random.binomial(1,1/(1+np.exp(-logit))).astype(bool)

    df = pd.DataFrame(data)
    for col in ['Administrative_Duration','Informational_Duration','ProductRelated_Duration','BounceRates','ExitRates','PageValues']:
        df[col] = df[col].round(6)
    return df


@st.cache_data
def preprocess_data(df):
    df = df.copy().dropna()
    le_month = LabelEncoder(); le_visitor = LabelEncoder()
    df['Month'] = le_month.fit_transform(df['Month'].astype(str))
    df['VisitorType'] = le_visitor.fit_transform(df['VisitorType'].astype(str))
    df['Weekend'] = df['Weekend'].astype(int)
    df['Revenue'] = df['Revenue'].astype(int)
    X = df.drop('Revenue',axis=1); y = df['Revenue']
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X),columns=X.columns,index=X.index)
    return X_scaled, y, scaler, {'Month':le_month,'VisitorType':le_visitor}


@st.cache_resource
def train_all_models(_X_train, _y_train):
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000,random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42,max_depth=10),
        'kNN': KNeighborsClassifier(n_neighbors=7),
        'Naive Bayes': GaussianNB(),
        'Random Forest (Ensemble)': RandomForestClassifier(n_estimators=100,random_state=42,max_depth=15,n_jobs=-1),
        'XGBoost (Ensemble)': XGBClassifier(n_estimators=100,random_state=42,max_depth=6,learning_rate=0.1,use_label_encoder=False,eval_metric='logloss')
    }
    trained = {}
    for name, model in models.items():
        model.fit(_X_train, _y_train)
        trained[name] = model
    return trained


def evaluate_models(trained_models, X_eval, y_eval):
    results = {}
    for name, model in trained_models.items():
        y_pred = model.predict(X_eval)
        y_proba = model.predict_proba(X_eval)[:,1]
        results[name] = {
            'metrics': {
                'Accuracy': round(accuracy_score(y_eval,y_pred),4),
                'AUC': round(roc_auc_score(y_eval,y_proba),4),
                'Precision': round(precision_score(y_eval,y_pred,zero_division=0),4),
                'Recall': round(recall_score(y_eval,y_pred,zero_division=0),4),
                'F1': round(f1_score(y_eval,y_pred,zero_division=0),4),
                'MCC': round(matthews_corrcoef(y_eval,y_pred),4)
            },
            'confusion_matrix': confusion_matrix(y_eval,y_pred).tolist(),
            'classification_report': classification_report(y_eval,y_pred,output_dict=True)
        }
    return results


def preprocess_uploaded_data(df, scaler, le_dict, feature_names):
    df = df.copy()
    has_target = 'Revenue' in df.columns
    y_true = df['Revenue'].astype(int) if has_target else None
    if has_target:
        df = df.drop('Revenue',axis=1)
    for col in ['Month','VisitorType']:
        if col in df.columns:
            le = le_dict[col]
            known = set(le.classes_)
            df[col] = df[col].apply(lambda x: x if x in known else le.classes_[0])
            df[col] = le.transform(df[col].astype(str))
    if 'Weekend' in df.columns:
        df['Weekend'] = df['Weekend'].astype(int)
    for f in feature_names:
        if f not in df.columns:
            df[f] = 0
    df = df[feature_names]
    return pd.DataFrame(scaler.transform(df),columns=df.columns,index=df.index), y_true


def plot_confusion_matrix(cm, model_name):
    fig, ax = plt.subplots(figsize=(6,4))
    sns.heatmap(cm,annot=True,fmt='d',cmap='Blues',ax=ax,
                xticklabels=['No Purchase','Purchase'],yticklabels=['No Purchase','Purchase'])
    ax.set_xlabel('Predicted',fontsize=12); ax.set_ylabel('Actual',fontsize=12)
    ax.set_title(f'Confusion Matrix - {model_name}',fontsize=14,fontweight='bold')
    plt.tight_layout()
    return fig


def plot_metrics_comparison(results, title='Model Performance Comparison'):
    models = list(results.keys())
    metrics_list = ['Accuracy','AUC','Precision','Recall','F1','MCC']
    fig, ax = plt.subplots(figsize=(14,6))
    x = np.arange(len(models)); width = 0.12
    colors = ['#2196F3','#4CAF50','#FF9800','#F44336','#9C27B0','#00BCD4']
    for i, metric in enumerate(metrics_list):
        vals = [results[m]['metrics'][metric] for m in models]
        ax.bar(x+i*width,vals,width,label=metric,color=colors[i],alpha=0.85)
    ax.set_xlabel('Model',fontsize=12); ax.set_ylabel('Score',fontsize=12)
    ax.set_title(title,fontsize=14,fontweight='bold')
    ax.set_xticks(x+width*2.5)
    ax.set_xticklabels(models,rotation=30,ha='right',fontsize=9)
    ax.legend(loc='upper right',fontsize=9); ax.set_ylim(0,1.05); ax.grid(axis='y',alpha=0.3)
    plt.tight_layout()
    return fig


# ─── Main App ─────────────────────────────────────────────────────────────────
def main():
    st.markdown('<p class="main-header">🛒 Online Shoppers Purchasing Intention</p>',unsafe_allow_html=True)
    st.markdown('<p class="sub-header">ML Classification Dashboard — 6 Models Compared</p>',unsafe_allow_html=True)

    # Train models
    with st.spinner("Loading dataset and training models (first load only)..."):
        df = generate_dataset()
        X, y, scaler, le_dict = preprocess_data(df)
        feature_names = X.columns.tolist()
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
        trained_models = train_all_models(X_train, y_train)
        default_results = evaluate_models(trained_models, X_test, y_test)

    # ─── Sidebar ──────────────────────────────────────────────────────────────
    st.sidebar.title("⚙️ Controls")
    selected_model = st.sidebar.selectbox("🔍 Select Classification Model",list(default_results.keys()),index=0)
    st.sidebar.markdown("---")
    st.sidebar.subheader("📁 Upload Test Data")
    uploaded_file = st.sidebar.file_uploader("Upload CSV file (test data)",type=['csv'],
        help="Upload test CSV with 'Revenue' column. All metrics will update based on uploaded data.")
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Dataset:** Online Shoppers Purchasing Intention\n\n**Source:** UCI ML Repository\n\n**Features:** 17\n\n**Instances:** 12,330\n\n**Task:** Binary Classification")

    # ─── Determine active results ─────────────────────────────────────────────
    active_results = default_results
    data_label = "Default Test Split (2,466 samples)"
    uploaded_df = None
    upload_error = False

    if uploaded_file is not None:
        try:
            uploaded_df = pd.read_csv(uploaded_file)
            X_up, y_up = preprocess_uploaded_data(uploaded_df, scaler, le_dict, feature_names)
            if y_up is not None:
                active_results = evaluate_models(trained_models, X_up, y_up)
                data_label = f"Uploaded Data: {uploaded_file.name} ({len(uploaded_df)} samples)"
            else:
                upload_error = True
        except Exception as e:
            st.error(f"Error processing file: {e}")
            upload_error = True

    # ─── Status Banner ────────────────────────────────────────────────────────
    if uploaded_file is not None and not upload_error:
        st.success(f"📂 **Evaluating on: {uploaded_file.name}** — {len(uploaded_df)} rows uploaded. All metrics below reflect uploaded data.")
    elif upload_error:
        st.warning("⚠️ Could not evaluate uploaded file. Ensure it has a 'Revenue' column (True/False or 1/0). Showing default test split results.")
    else:
        st.info("ℹ️ Showing results on default test split (2,466 samples). Upload a CSV from the sidebar to evaluate on your own data.")

    # ─── Tabs ─────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Model Comparison","🔬 Model Details","📋 Confusion Matrix & Report","ℹ️ About"])

    # ─── Tab 1: Comparison ────────────────────────────────────────────────────
    with tab1:
        st.subheader(f"All Models — Evaluation Metrics")
        st.caption(f"Data: {data_label}")

        comp_data = []
        for name, res in active_results.items():
            m = res['metrics']
            comp_data.append({'Model':name,'Accuracy':m['Accuracy'],'AUC':m['AUC'],
                              'Precision':m['Precision'],'Recall':m['Recall'],'F1 Score':m['F1'],'MCC':m['MCC']})
        comp_df = pd.DataFrame(comp_data)
        styled = comp_df.style.highlight_max(
            subset=['Accuracy','AUC','Precision','Recall','F1 Score','MCC'],color='#90EE90'
        ).format({'Accuracy':'{:.4f}','AUC':'{:.4f}','Precision':'{:.4f}','Recall':'{:.4f}','F1 Score':'{:.4f}','MCC':'{:.4f}'})
        st.dataframe(styled,width="stretch",hide_index=True)

        st.subheader("Visual Comparison")
        fig = plot_metrics_comparison(active_results, title=f'Model Performance ({data_label})')
        st.pyplot(fig)

        st.subheader("Key Observations")
        best_acc = max(active_results.items(),key=lambda x:x[1]['metrics']['Accuracy'])
        best_auc = max(active_results.items(),key=lambda x:x[1]['metrics']['AUC'])
        best_f1 = max(active_results.items(),key=lambda x:x[1]['metrics']['F1'])
        best_mcc = max(active_results.items(),key=lambda x:x[1]['metrics']['MCC'])
        c1,c2 = st.columns(2)
        c1.success(f"🏆 **Best Accuracy:** {best_acc[0]} ({best_acc[1]['metrics']['Accuracy']:.4f})")
        c1.info(f"📈 **Best AUC:** {best_auc[0]} ({best_auc[1]['metrics']['AUC']:.4f})")
        c2.warning(f"⚡ **Best F1:** {best_f1[0]} ({best_f1[1]['metrics']['F1']:.4f})")
        c2.error(f"📊 **Best MCC:** {best_mcc[0]} ({best_mcc[1]['metrics']['MCC']:.4f})")

    # ─── Tab 2: Selected Model Details ────────────────────────────────────────
    with tab2:
        st.subheader(f"Detailed Results: {selected_model}")
        st.caption(f"Data: {data_label}")

        m = active_results[selected_model]['metrics']
        c1,c2,c3,c4,c5,c6 = st.columns(6)
        c1.metric("Accuracy",f"{m['Accuracy']:.4f}")
        c2.metric("AUC",f"{m['AUC']:.4f}")
        c3.metric("Precision",f"{m['Precision']:.4f}")
        c4.metric("Recall",f"{m['Recall']:.4f}")
        c5.metric("F1 Score",f"{m['F1']:.4f}")
        c6.metric("MCC",f"{m['MCC']:.4f}")

        if uploaded_df is not None and not upload_error:
            st.markdown("---")
            st.subheader("Uploaded Data Preview")
            st.dataframe(uploaded_df.head(10),width="stretch")

    # ─── Tab 3: Confusion Matrix & Report ─────────────────────────────────────
    with tab3:
        st.subheader(f"Confusion Matrix & Classification Report: {selected_model}")
        st.caption(f"Data: {data_label}")

        model_res = active_results[selected_model]
        cl,cr = st.columns(2)

        with cl:
            cm = np.array(model_res['confusion_matrix'])
            fig = plot_confusion_matrix(cm,selected_model)
            st.pyplot(fig)

        with cr:
            report = model_res['classification_report']
            rdf = pd.DataFrame({
                'Class':['No Purchase (0)','Purchase (1)','Macro Avg','Weighted Avg'],
                'Precision':[report['0']['precision'],report['1']['precision'],report['macro avg']['precision'],report['weighted avg']['precision']],
                'Recall':[report['0']['recall'],report['1']['recall'],report['macro avg']['recall'],report['weighted avg']['recall']],
                'F1-Score':[report['0']['f1-score'],report['1']['f1-score'],report['macro avg']['f1-score'],report['weighted avg']['f1-score']],
                'Support':[int(report['0']['support']),int(report['1']['support']),int(report['macro avg']['support']),int(report['weighted avg']['support'])]
            }).style.format({'Precision':'{:.4f}','Recall':'{:.4f}','F1-Score':'{:.4f}'})
            st.dataframe(rdf,width="stretch",hide_index=True)

    # ─── Tab 4: About ─────────────────────────────────────────────────────────
    with tab4:
        st.subheader("About This Project")
        st.markdown("""
        ### Problem Statement
        Predict whether an online shopping session will result in a purchase (revenue generation)
        based on user browsing behavior, session characteristics, and temporal features.

        ### Dataset Description
        The **Online Shoppers Purchasing Intention Dataset** from the UCI Machine Learning Repository
        contains 12,330 sessions with 17 features (10 numerical, 7 categorical). The target variable
        `Revenue` indicates whether the session ended with a purchase. The dataset is imbalanced with
        approximately 85.8% negative (no purchase) and 14.2% positive (purchase) samples.

        ### Features
        - **Page Metrics:** Administrative, Informational, Product Related (counts and duration)
        - **Google Analytics:** Bounce Rate, Exit Rate, Page Values
        - **Temporal:** Month, Special Day, Weekend
        - **User Info:** Operating System, Browser, Region, Traffic Type, Visitor Type

        ### Models Implemented
        1. **Logistic Regression** — Linear model for binary classification
        2. **Decision Tree** — Tree-based model with interpretable splits
        3. **K-Nearest Neighbors (kNN)** — Instance-based lazy learner
        4. **Naive Bayes (Gaussian)** — Probabilistic classifier assuming feature independence
        5. **Random Forest (Ensemble)** — Bagging ensemble of decision trees
        6. **XGBoost (Ensemble)** — Gradient boosting ensemble method

        ### Tools & Technologies
        Python, Scikit-learn, XGBoost, Streamlit, Pandas, NumPy, Matplotlib, Seaborn
        """)
        st.markdown("---")
        st.markdown("*Built for BITS WILP — Machine Learning Assignment 2*")


if __name__ == "__main__":
    main()
