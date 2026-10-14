"""Rice yield forecast website (ICT336 sample project).

Run locally:   streamlit run app.py
Deploy:        push this folder to GitHub, then create an app on Streamlit Community Cloud.
"""
import json

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Rice Yield Forecast", page_icon="🌾", layout="wide")

# ---------------------------------------------------------------- text (EN / TH)
TEXT = {
    "EN": {
        "title": "🌾 Rice Yield Forecast",
        "subtitle": "Predict a farm's rice yield (kg per rai) about six weeks before harvest.",
        "note": "Sample project for ICT336. The model was trained on simulated teaching data, so forecasts are for demonstration only.",
        "farm": "Farm", "water": "Water and weather", "soil": "Soil and management", "sat": "Satellite",
        "region": "Region", "variety": "Rice variety", "season": "Season", "irrigated": "Irrigated farm",
        "size": "Farm size (rai)", "exp": "Farmer experience (years)", "year": "Crop year",
        "rain": "Season rainfall (mm)", "temp": "Average temperature (°C)",
        "ph": "Soil pH", "om": "Organic matter (%)", "n": "Nitrogen fertiliser (kg/rai)",
        "seed": "Seed rate (kg/rai)", "pest": "Pest level",
        "has_ndvi": "NDVI is available (about 6 weeks before harvest)", "ndvi": "Peak NDVI",
        "tab_forecast": "Forecast", "tab_model": "About the model", "tab_data": "Data",
        "pred": "Forecast yield", "range": "Likely range", "total": "Total paddy for this farm",
        "vs_avg": "vs average {v}", "tonnes": "tonnes",
        "no_ndvi": "Without NDVI the forecast is less accurate (pre-season mode).",
        "whatif": "What if the farm used more or less nitrogen fertiliser?",
        "whatif_x": "Nitrogen fertiliser (kg/rai)", "whatif_y": "Forecast yield (kg/rai)",
        "compare": "This farm compared with the average of each variety",
        "this_farm": "This farm",
        "scores": "Test-set scores (300 farms the model never saw)", "model": "Model",
        "baseline": "Baseline (mean)", "linear": "Linear Regression", "gb": "Tuned Gradient Boosting",
        "imp": "Which inputs matter most (permutation importance, kg/rai)",
        "how": "How it works: the website loads the trained scikit-learn pipeline saved by the notebook "
               "(`rice_yield_model.joblib`) and calls `predict()` on the values you enter.",
        "data_info": "{n} farm-seasons, 2019–2024. Average yield by variety:",
        "sample": "First rows of the dataset",
        "regions": {"Central": "Central", "Northeast": "Northeast", "North": "North"},
        "seasons": {"main": "Main season", "off": "Off season"},
        "pests": {"none": "None", "low": "Low", "high": "High"},
    },
    "TH": {
        "title": "🌾 พยากรณ์ผลผลิตข้าว",
        "subtitle": "ทำนายผลผลิตข้าวของฟาร์ม (kg ต่อไร่) ล่วงหน้าประมาณหกสัปดาห์ก่อนเก็บเกี่ยว",
        "note": "โปรเจกต์ตัวอย่างวิชา ICT336 โมเดลเทรนจากข้อมูลจำลองเพื่อการสอน ผลพยากรณ์ใช้เพื่อสาธิตเท่านั้น",
        "farm": "ข้อมูลฟาร์ม", "water": "น้ำและสภาพอากาศ", "soil": "ดินและการจัดการ", "sat": "ข้อมูลดาวเทียม",
        "region": "ภาค", "variety": "พันธุ์ข้าว", "season": "ฤดูปลูก", "irrigated": "อยู่ในเขตชลประทาน",
        "size": "ขนาดฟาร์ม (ไร่)", "exp": "ประสบการณ์เกษตรกร (ปี)", "year": "ปีเพาะปลูก",
        "rain": "ปริมาณฝนทั้งฤดู (mm)", "temp": "อุณหภูมิเฉลี่ย (°C)",
        "ph": "pH ของดิน", "om": "อินทรียวัตถุ (%)", "n": "ปุ๋ยไนโตรเจน (kg/ไร่)",
        "seed": "อัตราเมล็ดพันธุ์ (kg/ไร่)", "pest": "ระดับศัตรูพืช",
        "has_ndvi": "มีค่า NDVI แล้ว (ประมาณ 6 สัปดาห์ก่อนเก็บเกี่ยว)", "ndvi": "ค่า NDVI สูงสุด",
        "tab_forecast": "ผลพยากรณ์", "tab_model": "เกี่ยวกับโมเดล", "tab_data": "ข้อมูล",
        "pred": "ผลผลิตที่พยากรณ์", "range": "ช่วงที่น่าจะเป็น", "total": "ข้าวเปลือกทั้งฟาร์ม",
        "vs_avg": "เทียบค่าเฉลี่ย {v}", "tonnes": "ตัน",
        "no_ndvi": "เมื่อไม่มี NDVI ผลพยากรณ์จะแม่นยำน้อยลง (โหมดก่อนฤดูปลูก)",
        "whatif": "ถ้าใช้ปุ๋ยไนโตรเจนมากขึ้นหรือน้อยลง ผลผลิตจะเป็นอย่างไร",
        "whatif_x": "ปุ๋ยไนโตรเจน (kg/ไร่)", "whatif_y": "ผลผลิตที่พยากรณ์ (kg/ไร่)",
        "compare": "ฟาร์มนี้เทียบกับผลผลิตเฉลี่ยของแต่ละพันธุ์",
        "this_farm": "ฟาร์มนี้",
        "scores": "คะแนนบน test set (300 ฟาร์มที่โมเดลไม่เคยเห็น)", "model": "โมเดล",
        "baseline": "Baseline (ค่าเฉลี่ย)", "linear": "Linear Regression", "gb": "Gradient Boosting ที่จูนแล้ว",
        "imp": "ข้อมูลไหนมีผลมากที่สุด (permutation importance, kg/ไร่)",
        "how": "การทำงาน: เว็บไซต์โหลด scikit-learn pipeline ที่ notebook บันทึกไว้ "
               "(`rice_yield_model.joblib`) แล้วเรียก `predict()` กับค่าที่กรอก",
        "data_info": "ข้อมูล {n} ฟาร์ม-ฤดู ปี 2019–2024 ผลผลิตเฉลี่ยแต่ละพันธุ์:",
        "sample": "ตัวอย่างข้อมูลแถวแรก ๆ",
        "regions": {"Central": "ภาคกลาง", "Northeast": "ภาคตะวันออกเฉียงเหนือ", "North": "ภาคเหนือ"},
        "seasons": {"main": "นาปี", "off": "นาปรัง"},
        "pests": {"none": "ไม่มี", "low": "น้อย", "high": "มาก"},
    },
}
VARIETIES = ["KDML105", "RD6", "RD41", "PTT1"]
FEATURES = ["crop_year", "region", "rice_variety", "season", "irrigated", "rainfall_mm", "avg_temp_c",
            "soil_ph", "organic_matter_pct", "n_fertilizer_kg_rai", "seed_rate_kg_rai", "farm_size_rai",
            "pest_level", "farmer_experience_yrs", "ndvi_peak"]


# ---------------------------------------------------------------- load model and data
@st.cache_data
def load_data():
    return pd.read_csv("rice_yield_th.csv")


@st.cache_data
def load_info():
    with open("model_info.json") as f:
        return json.load(f)


@st.cache_resource
def load_model():
    try:
        return joblib.load("rice_yield_model.joblib")
    except Exception:
        # Fallback: if the saved file was made with a different scikit-learn version,
        # retrain the same pipeline from the CSV with the tuned settings.
        from sklearn.compose import ColumnTransformer
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler
        df = load_data()
        X, y = df[FEATURES], df["yield_kg_per_rai"]
        cat = ["region", "rice_variety", "season", "pest_level"]
        num = [c for c in FEATURES if c not in cat]
        pre = ColumnTransformer([
            ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), num),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat)])
        model = make_pipeline(pre, GradientBoostingRegressor(random_state=42, **load_info()["best_params"]))
        return model.fit(X, y)


df, info, model = load_data(), load_info(), load_model()

# ---------------------------------------------------------------- sidebar inputs
lang = st.sidebar.radio("ภาษา / Language", ["TH", "EN"], horizontal=True)
t = TEXT[lang]

sb = st.sidebar
sb.subheader(t["farm"])
region = sb.selectbox(t["region"], list(t["regions"]), format_func=t["regions"].get, index=1)
variety = sb.selectbox(t["variety"], VARIETIES)
season = sb.selectbox(t["season"], list(t["seasons"]), format_func=t["seasons"].get)
irrigated = sb.toggle(t["irrigated"], value=False)
farm_size = sb.number_input(t["size"], 1.0, 200.0, 15.0, 1.0)
experience = sb.slider(t["exp"], 1, 40, 20)
year = sb.selectbox(t["year"], list(range(2024, 2018, -1)))

sb.subheader(t["water"])
rain = sb.slider(t["rain"], 100, 1800, 1050, 10)
temp = sb.slider(t["temp"], 25.0, 31.5, 28.0, 0.1)

sb.subheader(t["soil"])
ph = sb.slider(t["ph"], 4.0, 7.8, 5.6, 0.1)
om = sb.slider(t["om"], 0.2, 5.0, 1.2, 0.1)
nfert = sb.slider(t["n"], 0.0, 35.0, 12.0, 0.5)
seed = sb.slider(t["seed"], 6, 35, 17)
pest = sb.selectbox(t["pest"], list(t["pests"]), format_func=t["pests"].get, index=1)

sb.subheader(t["sat"])
has_ndvi = sb.checkbox(t["has_ndvi"], value=True)
ndvi = sb.slider(t["ndvi"], 0.25, 0.95, 0.60, 0.01, disabled=not has_ndvi)


def farm_row(**changes):
    row = {"crop_year": year, "region": region, "rice_variety": variety, "season": season,
           "irrigated": int(irrigated), "rainfall_mm": rain, "avg_temp_c": temp, "soil_ph": ph,
           "organic_matter_pct": om, "n_fertilizer_kg_rai": nfert, "seed_rate_kg_rai": seed,
           "farm_size_rai": farm_size, "pest_level": pest, "farmer_experience_yrs": experience,
           "ndvi_peak": ndvi if has_ndvi else np.nan}
    row.update(changes)
    return row


# ---------------------------------------------------------------- page
st.title(t["title"])
st.write(t["subtitle"])
st.caption(t["note"])

tab1, tab2, tab3 = st.tabs([t["tab_forecast"], t["tab_model"], t["tab_data"]])

with tab1:
    pred = float(model.predict(pd.DataFrame([farm_row()]))[0])
    rmse = info["test_rmse"]
    avg = info["variety_mean"][variety]
    c1, c2, c3 = st.columns(3)
    c1.metric(t["pred"], f"{pred:,.0f} kg/rai" if lang == "EN" else f"{pred:,.0f} kg/ไร่",
              f"{pred - avg:+,.0f} {t['vs_avg'].format(v=variety)}")
    c2.metric(t["range"], f"{pred - rmse:,.0f} – {pred + rmse:,.0f}", f"± {rmse:.0f} (test RMSE)", delta_color="off")
    c3.metric(t["total"], f"{pred * farm_size / 1000:,.1f} {t['tonnes']}")
    if not has_ndvi:
        st.info(t["no_ndvi"])

    left, right = st.columns(2)
    with left:
        st.markdown(f"##### {t['whatif']}")
        grid = np.arange(0, 35.5, 1.0)
        curve = model.predict(pd.DataFrame([farm_row(n_fertilizer_kg_rai=v) for v in grid]))
        st.line_chart(pd.DataFrame({t["whatif_x"]: grid, t["whatif_y"]: curve}),
                      x=t["whatif_x"], y=t["whatif_y"], color="#00897E")
    with right:
        st.markdown(f"##### {t['compare']}")
        comp = pd.DataFrame({"name": [t["this_farm"]] + VARIETIES,
                             "kg": [pred] + [info["variety_mean"][v] for v in VARIETIES]})
        comp["is_farm"] = comp["name"] == t["this_farm"]
        bars = alt.Chart(comp).mark_bar().encode(
            x=alt.X("name:N", sort=None, title=None, axis=alt.Axis(labelAngle=0)),
            y=alt.Y("kg:Q", title=t["whatif_y"]),
            color=alt.condition("datum.is_farm", alt.value("#00897E"), alt.value("#B9C6DA")),
            tooltip=["name", alt.Tooltip("kg:Q", format=",.0f")])
        st.altair_chart(bars, width="stretch")

with tab2:
    st.markdown(f"##### {t['scores']}")
    scores = pd.DataFrame({
        t["model"]: [t["baseline"], t["linear"], t["gb"]],
        "RMSE": [info["baseline_rmse"], info["linear_rmse"], info["test_rmse"]]})
    st.dataframe(scores, hide_index=True, width="stretch")
    st.write(f"MAE = {info['test_mae']} · R² = {info['test_r2']}")
    st.markdown(f"##### {t['imp']}")
    imp = (pd.Series(info["importance"]).clip(lower=0).sort_values(ascending=False)
           .rename_axis("feature").reset_index(name="importance"))
    st.altair_chart(alt.Chart(imp).mark_bar(color="#6B85B8").encode(
        x=alt.X("importance:Q", title="kg/rai"), y=alt.Y("feature:N", sort=None, title=None),
        tooltip=["feature", "importance"]), width="stretch")
    st.caption(t["how"])

with tab3:
    st.write(t["data_info"].format(n=info["n_rows"]))
    st.bar_chart(pd.Series(info["variety_mean"]), color="#00897E")
    st.markdown(f"##### {t['sample']}")
    st.dataframe(df.head(20), width="stretch")
