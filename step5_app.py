
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Real Estate Investment Advisor", page_icon="🏠", layout="wide")

st.markdown('''
<style>
.good-box  { background:#e8f5e9; border:2px solid #27ae60; border-radius:12px; padding:1.2rem; text-align:center; }
.bad-box   { background:#fce4ec; border:2px solid #e74c3c; border-radius:12px; padding:1.2rem; text-align:center; }
.price-box { background:#e3f2fd; border:2px solid #1565C0; border-radius:12px; padding:1.2rem; text-align:center; }
</style>
''', unsafe_allow_html=True)

@st.cache_resource
def load_models():
    try:
        return {
            'clf'     : joblib.load("models/classifier.pkl"),
            'reg'     : joblib.load("models/regressor.pkl"),
            'features': joblib.load("models/feature_names.pkl"),
            'imputer' : joblib.load("models/imputer.pkl"),
            'ok'      : True,
        }
    except:
        return {'ok': False}

@st.cache_data
def load_data():
    try:    return pd.read_csv("cleaned_data_raw.csv")
    except: return None

m      = load_models()
df_raw = load_data()

st.markdown('<h1 style="text-align:center">🏠 Real Estate Investment Advisor</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align:center;color:#555">AI-powered · Classification + 5-Year Price Forecast</p>', unsafe_allow_html=True)
st.divider()

with st.sidebar:
    st.header("📋 Enter Property Details")
    st.subheader("📍 Location")
    state = st.selectbox("State", ["Maharashtra","Karnataka","Delhi","Tamil Nadu","West Bengal","Gujarat","Other"])
    city  = st.text_input("City", "Mumbai")

    st.subheader("🏗️ Property")
    prop_type  = st.selectbox("Property Type", ["Apartment","Villa","House","Studio"])
    bhk        = st.slider("BHK", 1, 5, 2)
    size       = st.number_input("Size (sq ft)", 200, 10000, 1000, 50)
    price      = st.number_input("Current Price (Lakhs)", 5.0, 1000.0, 80.0, 5.0)
    year_built = st.number_input("Year Built", 1970, 2024, 2010)

    st.subheader("🏢 Features")
    furnished  = st.selectbox("Furnished", ["Unfurnished","Semi-Furnished","Fully Furnished"])
    floor_no   = st.slider("Floor Number",   0, 40, 3)
    tot_floors = st.slider("Total Floors",   1, 50, 10)
    parking    = st.slider("Parking Spaces", 0,  4,  1)
    schools    = st.slider("Nearby Schools",   0, 15, 5)
    hospitals  = st.slider("Nearby Hospitals", 0, 10, 3)
    transport  = st.selectbox("Transport Access", ["Low","Medium","High"])
    security   = st.selectbox("Security", ["Gated","CCTV","Guard","None"])
    amenities  = st.multiselect("Amenities", ["Gym","Pool","Clubhouse","Garden","Lift"], default=["Gym"])
    facing     = st.selectbox("Facing", ["North","South","East","West"])
    owner_type = st.selectbox("Owner Type", ["Individual","Builder","Agent"])
    avail      = st.selectbox("Availability", ["Available","Under Construction","Sold"])
    predict_btn= st.button("🔮 Predict Now", use_container_width=True, type="primary")

def build_input(feature_names):
    tier = "High" if price > 150 else ("Mid" if price > 60 else "Low")
    row  = {
        "BHK"                    : bhk,
        "Size_in_SqFt"           : size,
        "Price_in_Lakhs"         : price,
        "Floor_No"               : floor_no,
        "Total_Floors"           : tot_floors,
        "Nearby_Schools"         : schools,
        "Nearby_Hospitals"       : hospitals,
        "Parking_Space"          : parking,
        "Price_per_SqFt"         : (price*100000)/size,
        "Age_of_Property"        : 2024 - year_built,
        "School_Density_Score"   : schools/15.0,
        "Amenity_Score"          : len(amenities),
        "Property_Type_enc"      : {"Apartment":0,"Villa":3,"House":1,"Studio":2}.get(prop_type,0),
        "Furnished_Status_enc"   : {"Unfurnished":2,"Semi-Furnished":1,"Fully Furnished":0}.get(furnished,1),
        "Facing_enc"             : {"East":0,"North":1,"South":2,"West":3}.get(facing,1),
        "Owner_Type_enc"         : {"Agent":0,"Builder":1,"Individual":2}.get(owner_type,1),
        "Availability_Status_enc": {"Available":0,"Sold":2,"Under Construction":1}.get(avail,0),
        "Security_enc"           : {"CCTV":0,"Gated":1,"Guard":2,"None":3}.get(security,1),
        "City_Price_Tier_enc"    : {"Low":1,"Mid":2,"High":0}.get(tier,1),
        "State_price_enc"        : price*1.05,
        "City_price_enc"         : price,
        "Locality_price_enc"     : price*0.98,
        "Transport_enc"          : {"Low":0,"Medium":1,"High":2}.get(transport,1),
    }
    inp = pd.DataFrame([row])
    for col in feature_names:
        if col not in inp.columns:
            inp[col] = 0
    return inp[feature_names]

tab1, tab2, tab3 = st.tabs(["🔮 Prediction", "📊 Market Insights", "📈 Model Info"])

with tab1:
    if predict_btn:
        if not m['ok']:
            st.error("Models not found! Run step3_model.py first.")
        else:
            with st.spinner("Analysing..."):
                inp    = build_input(m['features'])
                inp_sc = m['imputer'].transform(inp)
                clf_pred  = m['clf'].predict(inp_sc)[0]
                clf_proba = m['clf'].predict_proba(inp_sc)[0]
                reg_pred  = m['reg'].predict(inp_sc)[0]
                conf      = clf_proba[clf_pred]*100

            col1, col2 = st.columns(2)
            with col1:
                if clf_pred == 1:
                    st.markdown(f'<div class="good-box"><h2>✅ Good Investment!</h2><h3>Confidence: {conf:.1f}%</h3></div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="bad-box"><h2>❌ Not Recommended</h2><h3>Confidence: {conf:.1f}%</h3></div>', unsafe_allow_html=True)
            with col2:
                growth = reg_pred - price
                pct    = ((reg_pred/price)-1)*100
                st.markdown(f'<div class="price-box"><h2>📈 5-Year Forecast</h2><h1 style="color:#1565C0">₹{reg_pred:.1f}L</h1><p>Growth: +₹{growth:.1f}L (+{pct:.1f}%)</p></div>', unsafe_allow_html=True)

            st.subheader("Confidence Chart")
            fig, ax = plt.subplots(figsize=(7,2.5))
            ax.barh(["Not Good","Good Investment"], clf_proba*100, color=["#e74c3c","#27ae60"], height=0.4)
            for i,v in enumerate(clf_proba*100):
                ax.text(v+1, i, f"{v:.1f}%", va='center')
            ax.set_xlim(0,115); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
            st.pyplot(fig); plt.close()

            c1,c2,c3,c4 = st.columns(4)
            c1.metric("Price/SqFt",  f"₹{(price*100000/size):.0f}")
            c2.metric("Property Age",f"{2024-year_built} yrs")
            c3.metric("5Y Growth",   f"+₹{growth:.1f}L")
            c4.metric("Return %",    f"+{pct:.1f}%")
    else:
        st.info("👈 Fill sidebar and click Predict Now")

with tab2:
    if df_raw is None:
        st.warning("Run step1_preprocessing.py first.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Avg Price by City**")
            city_avg = df_raw.groupby('City')['Price_in_Lakhs'].mean().sort_values(ascending=False).head(10)
            fig, ax  = plt.subplots(figsize=(7,4))
            ax.barh(city_avg.index, city_avg.values, color=sns.color_palette("Blues_r",len(city_avg)))
            ax.set_xlabel("Avg Price (Lakhs)")
            ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
            st.pyplot(fig); plt.close()
        with col2:
            if 'Good_Investment' in df_raw.columns:
                st.markdown("**Investment Distribution**")
                cnts = df_raw['Good_Investment'].map({1:'Good',0:'Not Good'}).value_counts()
                fig, ax = plt.subplots(figsize=(5,4))
                ax.pie(cnts.values, labels=cnts.index, autopct='%1.1f%%',
                       colors=['#27ae60','#e74c3c'], wedgeprops=dict(edgecolor='white'))
                st.pyplot(fig); plt.close()

with tab3:
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**🎯 Classification**")
        st.markdown("|Metric|Score|\n|---|---|\n|Algorithm|XGBoost|\n|Accuracy|~98%|\n|F1-Score|~0.99|\n|ROC-AUC|~1.00|")
    with col2:
        st.markdown("**📉 Regression**")
        st.markdown("|Metric|Value|\n|---|---|\n|Target|Future Price 5Y|\n|CV R²|~0.99|")
    st.code("mlflow ui\n# http://localhost:5000")

st.divider()
st.caption("Real Estate Investment Advisor · XGBoost · MLflow · Streamlit")
