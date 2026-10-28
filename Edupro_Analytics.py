import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ==========================================
# PAGE CONFIGURATION & SETUP
# ==========================================
st.set_page_config(
    page_title="EduPro Analytics",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 15px;
        border-left: 5px solid #0d6efd;
    }
    .stApp {
        font-family: 'Arial', sans-serif;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# DATA SYNTHESIS / RETRIEVAL ENGINE
# ==========================================
@st.cache_data
def load_and_preprocess_data():
    np.random.seed(42)
    n_courses = 60
    
    categories = ['Programming', 'Design', 'Business', 'Data Science', 'Cybersecurity', 'AI & ML']
    levels = ['Beginner', 'Intermediate', 'Advanced']
    course_types = ['Paid', 'Free']
    expertise_list = ['Data Analytics', 'Web Dev', 'Digital Marketing', 'Machine Learning', 'Finance', 'UI/UX']
    
    course_ids = [f"CR{i+1:05d}" for i in range(n_courses)]
    categories_col = np.random.choice(categories, n_courses)
    levels_col = np.random.choice(levels, n_courses)
    types_col = np.random.choice(course_types, n_courses, p=[0.8, 0.2])
    
    prices = []
    for t in types_col:
        prices.append(0.0 if t == 'Free' else float(np.random.choice([499, 999, 1499, 1999, 2999, 4999])))
        
    durations = np.random.randint(5, 60, size=n_courses)
    course_ratings = np.round(np.random.uniform(3.5, 5.0, size=n_courses), 1)
    teacher_exp = np.random.randint(1, 20, size=n_courses)
    teacher_ratings = np.round(np.random.uniform(3.8, 5.0, size=n_courses), 1)
    expertise_col = np.random.choice(expertise_list, n_courses)
    
    # Target calculations with realistic domain constraints
    base_demand = (
        (course_ratings * 150) + 
        (teacher_exp * 25) + 
        (teacher_ratings * 100) - 
        (durations * 2) - 
        (np.array(prices) * 0.05) + 
        np.random.normal(0, 50, n_courses)
    )
    enrollments = np.maximum(20, np.round(base_demand)).astype(int)
    # Free courses have zero revenue
    revenue = np.where(types_col == 'Free', 0.0, enrollments * np.array(prices))

    df = pd.DataFrame({
        'CourseID': course_ids,
        'CourseCategory': categories_col,
        'CourseType': types_col,
        'CourseLevel': levels_col,
        'CoursePrice': prices,
        'CourseDuration': durations,
        'CourseRating': course_ratings,
        'YearsOfExperience': teacher_exp,
        'TeacherRating': teacher_ratings,
        'Expertise': expertise_col,
        'EnrollmentCount': enrollments,
        'CourseRevenue': revenue
    })
    
    # Feature Engineering
    df['PriceBand'] = pd.cut(df['CoursePrice'], bins=[-1, 0, 1000, 2500, 10000], labels=['Free', 'Low', 'Medium', 'High'])
    df['DurationBucket'] = pd.cut(df['CourseDuration'], bins=[0, 15, 35, 100], labels=['Short', 'Medium', 'Long'])
    df['RatingTier'] = pd.cut(df['CourseRating'], bins=[0, 4.0, 4.5, 5.0], labels=['Average', 'Good', 'Excellent'])
    
    return df

df_data = load_and_preprocess_data()

# ==========================================
# MACHINE LEARNING ENGINE
# ==========================================
@st.cache_resource
def train_predictive_models(df):
    features = [
        'CourseCategory', 'CourseType', 'CourseLevel', 'CoursePrice', 
        'CourseDuration', 'CourseRating', 'YearsOfExperience', 
        'TeacherRating', 'Expertise'
    ]
    
    X = df[features]
    y_demand = df['EnrollmentCount']
    y_revenue = df['CourseRevenue']
    
    categorical_cols = ['CourseCategory', 'CourseType', 'CourseLevel', 'Expertise']
    numerical_cols = ['CoursePrice', 'CourseDuration', 'CourseRating', 'YearsOfExperience', 'TeacherRating']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ],
        remainder='passthrough'
    )
    
    rf_demand = Pipeline([('prep', preprocessor), ('model', RandomForestRegressor(n_estimators=100, random_state=42))])
    gb_revenue = Pipeline([('prep', preprocessor), ('model', GradientBoostingRegressor(n_estimators=100, random_state=42))])
    
    rf_demand.fit(X, y_demand)
    gb_revenue.fit(X, y_revenue)
    
    return rf_demand, gb_revenue, features

model_demand, model_revenue, model_features = train_predictive_models(df_data)

# ==========================================
# NAVIGATION & SIDEBAR
# ==========================================
st.sidebar.title("🎓 EduPro Analytics")
st.sidebar.markdown("**Predictive Intelligence Suite**")
page = st.sidebar.radio(
    "Select Module:",
    ["Overview Dashboard", "Demand & Revenue Predictor", "Feature Importance Explorer", "Category Analysis"]
)

st.sidebar.markdown("---")
st.sidebar.info("System Status: **Active**\nModel Artifacts: **Loaded**")

# ==========================================
# MODULE 1: OVERVIEW DASHBOARD
# ==========================================
if page == "Overview Dashboard":
    st.title("📈 Executive Overview & Core KPIs")
    st.caption("Strategic platform performance and forecasting baseline")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Active Courses", len(df_data))
    with col2:
        st.metric("Total Historical Enrollments", f"{df_data['EnrollmentCount'].sum():,}")
    with col3:
        st.metric("Total Revenue Generated", f"${df_data['CourseRevenue'].sum():,.2f}")
    with col4:
        st.metric("Catalog Avg Rating", f"{df_data['CourseRating'].mean():.2f} / 5.0")
        
    st.markdown("---")
    
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Revenue Contribution by Category")
        cat_rev = df_data.groupby('CourseCategory')['CourseRevenue'].sum().reset_index()
        fig_rev = px.bar(cat_rev, x='CourseCategory', y='CourseRevenue', color='CourseCategory',
                         labels={'CourseRevenue': 'Revenue ($)', 'CourseCategory': 'Category'},
                         template="plotly_white")
        st.plotly_chart(fig_rev, use_container_width=True)
        
    with c2:
        st.subheader("Enrollment Distribution by Course Level")
        level_enr = df_data.groupby('CourseLevel')['EnrollmentCount'].sum().reset_index()
        fig_enr = px.pie(level_enr, values='EnrollmentCount', names='CourseLevel', hole=0.4,
                         template="plotly_white")
        st.plotly_chart(fig_enr, use_container_width=True)

# ==========================================
# MODULE 2: DEMAND & REVENUE PREDICTOR
# ==========================================
elif page == "Demand & Revenue Predictor":
    st.title("🎯 Interactive What-If Scenario Predictor")
    st.caption("Adjust course and instructor configurations to forecast future performance")
    
    col_input, col_output = st.columns([1, 1])
    
    with col_input:
        st.subheader("Input Course Specifications")
        val_category = st.selectbox("Course Category", df_data['CourseCategory'].unique())
        val_type = st.selectbox("Course Type", ['Paid', 'Free'])
        val_level = st.selectbox("Course Level", ['Beginner', 'Intermediate', 'Advanced'])
        val_price = st.slider("Price ($)", 0.0, 5000.0, 1499.0 if val_type == 'Paid' else 0.0, step=100.0)
        val_duration = st.slider("Duration (Hours)", 1, 100, 20)
        val_cr_rating = st.slider("Expected Course Rating", 1.0, 5.0, 4.5, step=0.1)
        
        st.markdown("---")
        st.subheader("Instructor Profile")
        val_exp = st.slider("Instructor Years of Experience", 0, 30, 8)
        val_tr_rating = st.slider("Instructor Rating", 1.0, 5.0, 4.8, step=0.1)
        val_expertise = st.selectbox("Instructor Expertise Domain", df_data['Expertise'].unique())
        
    input_payload = pd.DataFrame([{
        'CourseCategory': val_category,
        'CourseType': val_type,
        'CourseLevel': val_level,
        'CoursePrice': 0.0 if val_type == 'Free' else val_price,
        'CourseDuration': val_duration,
        'CourseRating': val_cr_rating,
        'YearsOfExperience': val_exp,
        'TeacherRating': val_tr_rating,
        'Expertise': val_expertise
    }])
    
    pred_demand = int(model_demand.predict(input_payload)[0])
    pred_rev = float(model_revenue.predict(input_payload)[0])
    if val_type == 'Free':
        pred_rev = 0.0

    with col_output:
        st.subheader("Forecasted Results")
        
        o_col1, o_col2 = st.columns(2)
        with o_col1:
            st.metric("Predicted Enrollment Count", f"{pred_demand:,} Students")
        with o_col2:
            st.metric("Predicted Revenue", f"${pred_rev:,.2f}")
            
        st.markdown("---")
        st.subheader("Price Sensitivity Sweep")
        
        # Sensitivity Analysis over Price range
        sweep_prices = np.linspace(0, 5000, 20)
        sweep_demands = []
        sweep_revenues = []
        
        for p in sweep_prices:
            temp_payload = input_payload.copy()
            temp_payload['CoursePrice'] = p
            temp_payload['CourseType'] = 'Free' if p == 0 else 'Paid'
            
            d_pred = model_demand.predict(temp_payload)[0]
            r_pred = model_revenue.predict(temp_payload)[0] if p > 0 else 0.0
            
            sweep_demands.append(d_pred)
            sweep_revenues.append(r_pred)
            
        fig_sweep = go.Figure()
        fig_sweep.add_trace(go.Scatter(x=sweep_prices, y=sweep_revenues, name="Forecasted Revenue ($)", mode='lines+markers', yaxis="y1"))
        fig_sweep.add_trace(go.Scatter(x=sweep_prices, y=sweep_demands, name="Forecasted Demand (Enrollments)", mode='lines+markers', yaxis="y2"))
        
        fig_sweep.update_layout(
            title="Price vs. Revenue & Demand Sensitivity",
            xaxis=dict(title="Price ($)"),
            yaxis=dict(title="Revenue ($)", side="left"),
            yaxis2=dict(title="Demand (Enrollments)", side="right", overlaying="y"),
            template="plotly_white",
            legend=dict(x=0.1, y=1.1, orientation="h")
        )
        st.plotly_chart(fig_sweep, use_container_width=True)

# ==========================================
# MODULE 3: FEATURE IMPORTANCE EXPLORER
# ==========================================
elif page == "Feature Importance Explorer":
    st.title("💡 Model Interpretability & Feature Importance")
    st.caption("Quantitative identification of primary course success drivers")
    
    target_choice = st.radio("Select Prediction Target to Explore:", ["Enrollment Count (Demand)", "Course Revenue"])
    
    pipeline = model_demand if target_choice == "Enrollment Count (Demand)" else model_revenue
    prep = pipeline.named_steps['prep']
    model = pipeline.named_steps['model']
    
    cat_feature_names = prep.named_transformers_['cat'].get_feature_names_out(['CourseCategory', 'CourseType', 'CourseLevel', 'Expertise'])
    num_feature_names = ['CoursePrice', 'CourseDuration', 'CourseRating', 'YearsOfExperience', 'TeacherRating']
    all_feature_names = list(cat_feature_names) + num_feature_names
    
    importances = model.feature_importances_
    df_fi = pd.DataFrame({'Feature': all_feature_names, 'Importance': importances}).sort_values('Importance', ascending=True)
    
    fig_fi = px.bar(df_fi.tail(10), x='Importance', y='Feature', orientation='h',
                    title=f"Top 10 Feature Importances for {target_choice}",
                    template="plotly_white")
    st.plotly_chart(fig_fi, use_container_width=True)
    
    st.subheader("Business Insights")
    st.markdown("""
    - **Course & Instructor Ratings**: High ratings correlate strongly with increased enrollment conversions.
    - **Pricing Strategy**: Price elasticity significantly impacts revenue yield; moderate price adjustments optimize total return without drastically diminishing volume.
    - **Instructor Experience**: Faculty expertise drives trust, positively serving as a secondary predictor of course demand.
    """)

# ==========================================
# MODULE 4: CATEGORY ANALYSIS
# ==========================================
elif page == "Category Analysis":
    st.title("📊 Category-Level Demand & Revenue Benchmark")
    st.caption("Aggregated market metrics across course categories")
    
    cat_summary = df_data.groupby('CourseCategory').agg(
        TotalCourses=('CourseID', 'count'),
        TotalEnrollments=('EnrollmentCount', 'sum'),
        TotalRevenue=('CourseRevenue', 'sum'),
        AvgPrice=('CoursePrice', 'mean'),
        AvgRating=('CourseRating', 'mean')
    ).reset_index()
    
    st.dataframe(cat_summary.style.format({
        'TotalRevenue': "${:,.2f}",
        'AvgPrice': "${:,.2f}",
        'AvgRating': "{:.2f}"
    }), use_container_width=True)
    
    fig_radar = go.Figure()
    for _, row in cat_summary.iterrows():
        fig_radar.add_trace(go.Scatterpolar(
            r=[row['TotalCourses'], row['AvgRating']*10, row['AvgPrice']/100],
            theta=['Course Offerings', 'Scaled Rating (x10)', 'Scaled Price (/100)'],
            fill='toself',
            name=row['CourseCategory']
        ))
    fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True)), showlegend=True, template="plotly_white")
    st.subheader("Multi-Dimensional Category Comparison")
    st.plotly_chart(fig_radar, use_container_width=True)
