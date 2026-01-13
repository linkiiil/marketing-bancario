import re
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import streamlit as st
import streamlit.components.v1 as components

# ---------------------------
# App: Ferramenta de Triagem - Score de Propensão (versão simples com explainer cacheado)
# ---------------------------

st.set_page_config(page_title="Ferramenta de Triagem — Score de Propensão", layout="wide")

st.markdown(
    """
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button { width: 100%; border-radius: 8px; height: 3.5em; background-color: #004a99; color: white; font-weight: bold; }
    .result-card { background-color: white; padding: 20px; border-radius: 10px; border-left: 6px solid #004a99; box-shadow: 2px 2px 8px rgba(0,0,0,0.08); }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------
# 1) Carregar modelo (cache)
# ---------------------------
@st.cache_resource
def load_model(path="modelo_bank_marketing_xgboost.pkl"):
    return joblib.load(path)

model_pipeline = load_model()

# ---------------------------
# 1b) Cache do SHAP explainer (sem argumentos para evitar hashing de objetos não-hashable)
# ---------------------------
@st.cache_resource
def load_explainer():
    model = model_pipeline.named_steps['model']
    return shap.TreeExplainer(model)

# ---------------------------
# 2) Mapeamentos e utilitários
# ---------------------------
job_map = {
    "Gerência": "management", "Técnico": "technician", "Empreendedor": "entrepreneur",
    "Operário": "blue-collar", "Aposentado": "retired", "Administrativo": "admin.",
    "Serviços": "services", "Autônomo": "self-employed", "Desempregado": "unemployed",
    "Empregado Doméstico": "housemaid", "Estudante": "student", "Desconhecido": "unknown"
}
marital_map = {"Casado": "married", "Solteiro": "single", "Divorciado": "divorced"}
education_map = {"Superior": "tertiary", "Secundário": "secondary", "Primário": "primary", "Desconhecido": "unknown"}
month_map = {
    "Janeiro": "jan", "Fevereiro": "feb", "Março": "mar", "Abril": "apr", "Maio": "may", "Junho": "jun",
    "Julho": "jul", "Agosto": "aug", "Setembro": "sep", "Outubro": "oct", "Novembro": "nov", "Dezembro": "dec"
}
sim_nao_map = {"Sim": "yes", "Não": "no"}

# Features que representam leakage operacional e devem ser omitidas da explicação
LEAKAGE_FEATURES = {"duration", "contact", "poutcome"}

# ---------------------------
# 3) Função de limpeza de nomes (cols seguras + labels legíveis)
# ---------------------------
def clean_feature_names(raw_names):
    clean_cols = []
    clean_display = []

    for orig in raw_names:
        name = orig
        name = re.sub(r'^(?:cat|bin|num|semiord)[_\-\.]*__+', '', name, flags=re.IGNORECASE)
        name = re.sub(r'^(?:cat|bin|num|semiord)[_\-\.]+', '', name, flags=re.IGNORECASE)
        if '__' in name:
            parts = name.split('__', 1)
            if parts[1].strip() != "":
                name = parts[1]
        name = re.sub(r'(?i)[_\-\.\s]*\(?\b(?:bin|num|semiord|cat)\b\)?$', '', name)
        col_safe = re.sub(r'[ \.\-]+', '_', name).strip('_')
        display = col_safe
        display = re.sub(r'(\d+)_(\d+)$', r'\1-\2', display)
        display = display.replace('_', ' ')
        display = re.sub(r'^(quarter)\s+([A-Za-z0-9]+)$', r'\1: \2', display, flags=re.IGNORECASE)
        if col_safe == "":
            col_safe = orig
        if display == "":
            display = orig
        clean_cols.append(col_safe)
        clean_display.append(display)

    counts = {}
    unique_cols = []
    unique_display = []
    for col, disp in zip(clean_cols, clean_display):
        base = col
        if base in counts:
            counts[base] += 1
            new_col = f"{base}_{counts[base]}"
            new_disp = f"{disp} ({counts[base]})"
        else:
            counts[base] = 0
            new_col = base
            new_disp = disp
        unique_cols.append(new_col)
        unique_display.append(new_disp)

    return unique_cols, unique_display

# ---------------------------
# 4) Pré-processamento de entrada (renomeado day_of_month)
# ---------------------------
def preprocess_input(df):
    df_transformed = df.copy()
    bin_map = {"yes": 1, "no": 0}
    for col in ["default", "housing", "loan"]:
        if col in df_transformed.columns:
            df_transformed[col] = df_transformed[col].map(bin_map)
    if "pdays" in df_transformed.columns:
        df_transformed["pdays"] = df_transformed["pdays"].apply(lambda x: max(-1, int(x)))
    if "previous" in df_transformed.columns:
        df_transformed["previous"] = df_transformed["previous"].apply(lambda x: max(0, int(x)))

    def group_day(d):
        if 1 <= d <= 5: return "1_5"
        elif 6 <= d <= 10: return "6_10"
        elif 11 <= d <= 15: return "11_15"
        elif 16 <= d <= 20: return "16_20"
        elif 21 <= d <= 25: return "21_25"
        else: return "26_31"

    # usa day_of_month (correção semântica)
    df_transformed["days"] = df_transformed["day_of_month"].apply(group_day)

    q_map = {
        "jan": "Q1", "feb": "Q1", "mar": "Q1", "apr": "Q2", "may": "Q2", "jun": "Q2",
        "jul": "Q3", "aug": "Q3", "sep": "Q3", "oct": "Q4", "nov": "Q4", "dec": "Q4"
    }
    df_transformed["quarter"] = df_transformed["month"].map(q_map)
    return df_transformed.drop(columns=["day_of_month", "month"])

# ---------------------------
# 5) Session state inicial
# ---------------------------
if "prediction_done" not in st.session_state:
    st.session_state.prediction_done = False
if "proba" not in st.session_state:
    st.session_state.proba = None
if "input_final" not in st.session_state:
    st.session_state.input_final = None
if "X_trans" not in st.session_state:
    st.session_state.X_trans = None
if "feat_names_raw" not in st.session_state:
    st.session_state.feat_names_raw = None
if "feat_names_cols" not in st.session_state:
    st.session_state.feat_names_cols = None
if "feat_names_display" not in st.session_state:
    st.session_state.feat_names_display = None
if "sv" not in st.session_state:
    st.session_state.sv = None
if "ev" not in st.session_state:
    st.session_state.ev = None
if "ref_probas" not in st.session_state:
    st.session_state.ref_probas = None
if "percentile" not in st.session_state:
    st.session_state.percentile = None

# ---------------------------
# 6) Inputs na sidebar (usa day_of_month)
# ---------------------------
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2830/2830284.png", width=80)
st.sidebar.header("Perfil do Lead")

def get_inputs():
    age = st.sidebar.slider("Idade", 18, 95, 40)
    job = st.sidebar.selectbox("Ocupação", list(job_map.keys()))
    marital = st.sidebar.selectbox("Estado Civil", list(marital_map.keys()))
    edu = st.sidebar.selectbox("Escolaridade", list(education_map.keys()))
    balance = st.sidebar.number_input("Saldo Médio Anual (€)", value=1300)
    st.sidebar.divider()
    default = st.sidebar.selectbox("Inadimplente?", list(sim_nao_map.keys()))
    housing = st.sidebar.selectbox("Tem Empréstimo Imobiliário?", list(sim_nao_map.keys()))
    loan = st.sidebar.selectbox("Tem Empréstimo Pessoal?", list(sim_nao_map.keys()))
    st.sidebar.divider()
    day = st.sidebar.slider("Dia do Mês", 1, 31, 15)
    month = st.sidebar.selectbox("Mês", list(month_map.keys()))
    campaign = st.sidebar.slider("Contatos nesta Campanha", 1, 60, 1)
    pdays = st.sidebar.number_input(
        "Dias desde último contato",
        value=-1,
        min_value=-1,
        help="-1 significa que o cliente nunca foi contactado anteriormente"
    )
    st.sidebar.caption("Nota: Dias desde último contato = -1 → nunca contactado anteriormente")
    prev = st.sidebar.number_input("Contatos Anteriores", value=0, min_value=0)
    return pd.DataFrame({
        'age': age, 'job': job_map[job], 'marital': marital_map[marital], 'education': education_map[edu],
        'default': sim_nao_map[default], 'balance': balance, 'housing': sim_nao_map[housing],
        'loan': sim_nao_map[loan], 'day_of_month': day, 'month': month_map[month],
        'campaign': campaign, 'pdays': pdays, 'previous': prev
    }, index=[0])

input_raw = get_inputs()

# ---------------------------
# 7) Nota de origem dos dados (adicionada)
# ---------------------------
st.markdown(
    """
    **Origem dos dados:** Os dados referem-se a campanhas de marketing direto de uma instituição bancária portuguesa.
    """
)

# ---------------------------
# 8) Carregar referência de probabilidades (opcional)
# ---------------------------
st.sidebar.divider()
st.sidebar.subheader("Distribuição de referência (opcional)")
st.sidebar.write("Carregue um arquivo .npy (1D) ou .csv com uma coluna de probabilidades para calcular percentis/ranking.")
uploaded_ref = st.sidebar.file_uploader("Arquivo de referência (.npy ou .csv)", type=["npy", "csv"])
if uploaded_ref is not None:
    try:
        if uploaded_ref.name.lower().endswith(".npy"):
            ref = np.load(uploaded_ref)
        else:
            df_ref = pd.read_csv(uploaded_ref)
            numeric_cols = df_ref.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) == 0:
                raise ValueError("CSV sem colunas numéricas")
            ref = df_ref[numeric_cols[0]].to_numpy()
        ref = np.array(ref).astype(float)
        st.session_state.ref_probas = ref
        st.sidebar.success(f"Referência carregada ({len(ref)} registros)")
    except Exception as e:
        st.sidebar.error(f"Erro ao carregar referência: {e}")

# ---------------------------
# 9) Função para calcular e salvar predição + SHAP no session_state
# ---------------------------
def calcular_propensao():
    input_final = preprocess_input(input_raw)
    proba = model_pipeline.predict_proba(input_final)[0, 1]

    preprocessor = model_pipeline.named_steps['preprocess']
    xgb_model = model_pipeline.named_steps['model']

    X_trans = preprocessor.transform(input_final)
    feat_names_raw = preprocessor.get_feature_names_out()

    # usa explainer cacheado (sem passar o modelo como argumento)
    explainer = load_explainer()
    shap_vals = explainer.shap_values(X_trans)

    if isinstance(shap_vals, list):
        if len(shap_vals) == 2:
            sv = shap_vals[1]
            ev = explainer.expected_value[1] if hasattr(explainer.expected_value, "__len__") else explainer.expected_value
        else:
            sv = shap_vals[0]
            ev = explainer.expected_value[0] if hasattr(explainer.expected_value, "__len__") else explainer.expected_value
    else:
        sv = shap_vals
        ev = explainer.expected_value

    cols_safe, labels_display = clean_feature_names(list(feat_names_raw))

    st.session_state.prediction_done = True
    st.session_state.proba = float(proba)
    st.session_state.input_final = input_final.to_dict(orient="records")[0]
    try:
        st.session_state.X_trans = np.array(X_trans).tolist()
    except Exception:
        st.session_state.X_trans = X_trans.tolist() if hasattr(X_trans, "tolist") else X_trans
    st.session_state.feat_names_raw = list(feat_names_raw)
    st.session_state.feat_names_cols = cols_safe
    st.session_state.feat_names_display = labels_display
    st.session_state.sv = np.array(sv).tolist()
    st.session_state.ev = ev.tolist() if hasattr(ev, "tolist") else ev

    if st.session_state.ref_probas is not None:
        ref = np.array(st.session_state.ref_probas).astype(float)
        percentile = (ref < proba).mean() * 100
        st.session_state.percentile = float(percentile)
    else:
        st.session_state.percentile = None

# ---------------------------
# 10) Função: montar tabela SHAP (sem feature_value)
# ---------------------------
def build_shap_table(sv_sample, feat_names_display, feat_names_cols, filter_leakage=True):
    sv = np.array(sv_sample).flatten()
    names_display = list(feat_names_display)
    names_cols = list(feat_names_cols)

    if filter_leakage:
        df_tmp = pd.DataFrame({"col_name": names_cols, "display": names_display, "shap_value": sv})
        df_tmp["base"] = df_tmp["col_name"].apply(lambda x: x.split('_')[0].lower())
        df_tmp = df_tmp[~df_tmp["base"].isin(LEAKAGE_FEATURES)].reset_index(drop=True)
        sv = df_tmp["shap_value"].to_numpy()
        names_display = df_tmp["display"].to_list()
        names_cols = df_tmp["col_name"].to_list()

    df = pd.DataFrame({"feature": names_display, "shap_value": sv, "col_name": names_cols})
    df["abs_shap"] = df["shap_value"].abs()
    df = df.sort_values("abs_shap", ascending=False).reset_index(drop=True)
    total = df["abs_shap"].sum() if df["abs_shap"].sum() != 0 else 1.0
    df["impact_pct"] = (df["abs_shap"] / total * 100).round(1)
    return df

# ---------------------------
# 11) Layout principal (guias)
# ---------------------------
tab_prop, tab_expl = st.tabs(["📈 Triagem (Score)", "💡 Explicação (SHAP)"])

with tab_prop:
    col_res, _ = st.columns([1, 0.6])
    with col_res:
        st.subheader("Calcular score de propensão (triagem)")
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        st.write("Ajuste os parâmetros na barra lateral e clique em calcular. Esta ferramenta apoia decisões de triagem e ranking.")
        if st.button("🚀 Calcular score (triagem)"):
            calcular_propensao()
            st.success("Cálculo realizado. Vá para a guia 'Explicação (SHAP)' para ver contexto e detalhes.")
        st.markdown('</div>', unsafe_allow_html=True)

    if st.session_state.prediction_done:
        st.markdown("---")
        st.subheader("Último resultado (resumo)")
        proba = st.session_state.proba
        st.metric(label="Score de propensão (probabilidade)", value=f"{proba:.2%}")

        if st.session_state.ref_probas is not None and st.session_state.percentile is not None:
            pct = st.session_state.percentile
            st.metric(label="Percentil (referência)", value=f"{pct:.1f}º percentil")
            if pct >= 90:
                st.info("Sugestão de triagem: Top 10% — priorizar contato")
            elif pct >= 50:
                st.info("Sugestão de triagem: Prioridade média — considerar critérios adicionais")
            else:
                st.info("Sugestão de triagem: Baixa prioridade")
        else:
            st.info("Sem distribuição de referência carregada. Carregue um arquivo na barra lateral para ver percentil/ranking relativo.")

        if input_raw['campaign'].iloc[0] > 20:
            st.error(f"🚨 Fadiga detectada: {input_raw['campaign'].iloc[0]} contatos. A conversão tende a cair.")

with tab_expl:
    st.subheader("Explicação com SHAP (contextualizada)")
    st.markdown(
        "**Aviso:** SHAP mostra a contribuição do modelo para a predição; não é explicação causal nem contrafactual. Os valores SHAP estão na escala de log-odds (link='logit'). Use como apoio interpretativo e preferencialmente em conjunto com análises agregadas."
    )
    st.write("A tabela abaixo mostra as features mais relevantes (sufixos/prefixos técnicos removidos). O force plot é exibido por padrão apenas para leads em Top-K (configurável).")

    if not st.session_state.prediction_done:
        st.info("Nenhum cálculo encontrado. Vá para a guia 'Triagem (Score)' e clique em 'Calcular score'.")
    else:
        feat_raw = st.session_state.feat_names_raw
        feat_cols = st.session_state.feat_names_cols
        feat_display = st.session_state.feat_names_display
        sv = np.array(st.session_state.sv)
        ev = st.session_state.ev
        X_trans = np.array(st.session_state.X_trans)

        sv_sample = sv[0] if sv.ndim == 2 else sv
        X_row = X_trans[0] if X_trans.ndim == 2 else X_trans

        df_shap = build_shap_table(
            sv_sample=sv_sample,
            feat_names_display=feat_display,
            feat_names_cols=feat_cols,
            filter_leakage=True
        )

        choice = st.radio("Mostrar tabela:", ("Top 3", "Todas"), horizontal=True)
        if choice == "Top 3":
            df_show = df_shap.head(3).copy()
        else:
            df_show = df_shap.copy()

        df_display = df_show[["feature", "shap_value", "impact_pct"]].copy()
        df_display = df_display.rename(columns={
            "feature": "Feature",
            "shap_value": "SHAP (logit)",
            "impact_pct": "Impacto (%)"
        })
        df_display["SHAP (logit)"] = df_display["SHAP (logit)"].round(4)
        if "Impacto (%)" in df_display.columns:
            df_display["Impacto (%)"] = df_display["Impacto (%)"].round(1)

        st.dataframe(df_display.reset_index(drop=True), use_container_width=True)

        st.markdown("---")
        st.subheader("Force plot interativo (condicionado ao contexto)")

        TOP_K_PERCENTILE = st.sidebar.slider("Mostrar force plot automaticamente para percentil ≥ ", 70, 99, 90)
        show_shap = False
        if st.session_state.ref_probas is not None and st.session_state.percentile is not None:
            if st.session_state.percentile >= TOP_K_PERCENTILE:
                show_shap = True

        if st.button("🔍 Forçar exibição do force plot (ignorar regra)"):
            show_shap = True

        if not show_shap:
            st.info(f"Por padrão, o force plot é mostrado apenas para leads em percentil ≥ {TOP_K_PERCENTILE}. Use 'Forçar exibição' para visualizar.")
        else:
            shap_js = ""
            try:
                shap_js = shap.getjs()
            except Exception:
                try:
                    shap.initjs()
                except Exception:
                    pass
                shap_js = ""

            try:
                df_orig = pd.DataFrame([X_row], columns=feat_raw)
                rename_map = {orig: col for orig, col in zip(feat_raw, feat_cols)}
                X_for_plot_full = df_orig.rename(columns=rename_map)
                cols_to_keep = [c for c in X_for_plot_full.columns if c.split('_')[0].lower() not in LEAKAGE_FEATURES]
                X_for_plot = X_for_plot_full[cols_to_keep]
            except Exception:
                try:
                    X_for_plot = pd.DataFrame([X_row], columns=feat_cols)
                    cols_to_keep = [c for c in X_for_plot.columns if c.split('_')[0].lower() not in LEAKAGE_FEATURES]
                    X_for_plot = X_for_plot[cols_to_keep]
                except Exception:
                    st.error("Não foi possível construir o DataFrame para o force plot. Verifique a consistência das features.")
                    X_for_plot = None

            if X_for_plot is not None:
                try:
                    orig_cols = list(st.session_state.feat_names_cols)
                    df_map = pd.DataFrame({"col": orig_cols, "shap": np.array(sv_sample).flatten()})
                    df_map["base"] = df_map["col"].apply(lambda x: x.split('_')[0].lower())
                    df_map = df_map[~df_map["base"].isin(LEAKAGE_FEATURES)].reset_index(drop=True)
                    sv_filtered = df_map["shap"].to_numpy()
                except Exception:
                    sv_filtered = sv_sample

                try:
                    force_plot = shap.force_plot(st.session_state.ev, sv_filtered, X_for_plot, link="logit")
                    html_to_render = (shap_js or "") + f'<div style="width:100%">{force_plot._repr_html_()}</div>'
                    n_features = X_for_plot.shape[1]
                    height = int(min(2400, max(700, 450 + 40 * n_features)))
                    components.html(html_to_render, height=height, scrolling=True)
                    st.caption("Vermelho aumenta a probabilidade; Azul diminui. Lembrete: SHAP não é causal.")
                except Exception as e:
                    st.error(f"Erro ao renderizar force plot: {e}")

# ---------------------------
# Rodapé / instruções rápidas
# ---------------------------
st.divider()
st.caption("Nota operacional: este app é uma ferramenta de apoio à decisão (triagem/ranking). Evite usar SHAP individual como prova de causalidade.")
