import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import os

# Set page configuration for dark theme and wide layout
st.set_page_config(
    page_title="Student Performance Analysis",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="📊"
)

# Custom CSS for Poppins font, minimalist design
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
    }

    .stApp {
        background-color: #1A1A1A;
        color: #E0E0E0;
    }

    .sidebar .sidebar-content {
        background-color: #2A2A2A;
    }

    h1, h2, h3, h4, h5, h6 {
        color: #FFFFFF;
    }

    .stButton>button {
        background-color: #4A4A4A;
        color: #FFFFFF;
        border-radius: 8px;
        border: none;
        padding: 10px 20px;
        font-weight: 500;
        transition: all 0.3s ease;
    }

    .stButton>button:hover {
        background-color: #6A6A6A;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }

    .stSelectbox, .stMultiSelect, .stTextInput, .stNumberInput {
        background-color: #3A3A3A;
        border-radius: 8px;
        border: 1px solid #4A4A4A;
    }

    /* Navigation tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        height: 40px;
        padding: 0 20px;
        background-color: #3A3A3A;
        border-radius: 8px 8px 0 0;
        transition: all 0.3s ease;
    }

    .stTabs [data-baseweb="tab"]:hover {
        background-color: #4A4A4A;
    }

    .stTabs [aria-selected="true"] {
        background-color: #4CAF50 !important;
        color: white !important;
    }

    /* Card styling */
    .card {
        background-color: #2A2A2A;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }

    /* Make graphs smaller */
    .stPlot {
        max-width: 400px !important;
    }

    /* Hide Streamlit menu and footer */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# Initialize session state for page navigation
if 'page' not in st.session_state:
    st.session_state.page = "Home"


@st.cache_data
def load_and_preprocess_data():
    """Load and preprocess the dataset from hardcoded path"""
    file_path = r"C:\Users\bijay\Documents\dsp_project_sem4\StudentsPerformance.csv"
    try:
        df = pd.read_csv(file_path)

        # Rename columns
        df = df.rename(columns={
            'race/ethnicity': 'ethnicity',
            'parental level of education': 'parental_education',
            'test preparation course': 'test_prep',
            'math score': 'math_score',
            'reading score': 'reading_score',
            'writing score': 'writing_score'
        })

        # Handle missing values
        df.fillna(df.select_dtypes(include=np.number).mean(), inplace=True)
        df.fillna(df.select_dtypes(include=object).mode().iloc[0], inplace=True)

        # Drop duplicates
        df.drop_duplicates(inplace=True)

        # Calculate average score
        df['average_score'] = (df['math_score'] + df['reading_score'] + df['writing_score']) / 3

        # Features and target
        X = df[['gender', 'ethnicity', 'parental_education', 'lunch', 'test_prep']]
        y = df['average_score']

        # Split dataset
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        return X_train, X_test, y_train, y_test, df

    except FileNotFoundError:
        st.error(f"Error: File not found at {file_path}")
        return None, None, None, None, None
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None, None, None


@st.cache_resource
def build_and_evaluate_models(X_train, X_test, y_train, y_test):
    """Build and evaluate regression models"""
    if X_train is None:
        return None, None, None, None

    categorical_features = ['gender', 'ethnicity', 'parental_education', 'lunch', 'test_prep']
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])

    models = {
        'Random Forest': Pipeline([
            ('preprocessor', preprocessor),
            ('scaler', StandardScaler(with_mean=False)),
            ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
        ]),
        'Linear Regression': Pipeline([
            ('preprocessor', preprocessor),
            ('scaler', StandardScaler(with_mean=False)),
            ('regressor', LinearRegression())
        ]),
        'SVR': Pipeline([
            ('preprocessor', preprocessor),
            ('scaler', StandardScaler(with_mean=False)),
            ('regressor', SVR(kernel='rbf'))
        ])
    }

    results = {}
    predictions = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        results[name] = {'RMSE': rmse, 'R2': r2, 'MAE': mae}
        predictions[name] = y_pred

    return models, results, predictions, y_test


def plot_results(results, predictions, y_test):
    """Generate regression plots with reduced size"""
    st.subheader("RMSE Comparison")
    rmses = [results[model]['RMSE'] for model in results]
    models = list(results.keys())
    fig, ax = plt.subplots(figsize=(5, 3))  # Reduced size
    sns.barplot(x=models, y=rmses, ax=ax, palette='Blues')
    ax.set_title('RMSE Comparison Across Models', fontsize=10)
    ax.set_ylabel('RMSE', fontsize=8)
    ax.set_xlabel('', fontsize=8)
    ax.tick_params(axis='both', labelsize=8)
    ax.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(rotation=45)
    st.pyplot(fig, use_container_width=False)
    plt.close()

    for model_name, y_pred in predictions.items():
        st.subheader(f'Predicted vs Actual Scores ({model_name})')
        fig, ax = plt.subplots(figsize=(5, 3))  # Reduced size
        ax.scatter(y_test, y_pred, alpha=0.5, s=10)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=1)
        ax.set_xlabel('Actual Scores', fontsize=8)
        ax.set_ylabel('Predicted Scores', fontsize=8)
        ax.set_title(f'Predicted vs Actual Scores ({model_name})', fontsize=10)
        ax.tick_params(axis='both', labelsize=8)
        ax.grid(True, linestyle='--', alpha=0.7)
        st.pyplot(fig, use_container_width=False)
        plt.close()

        st.subheader(f'Residual Plot ({model_name})')
        residuals = y_test - y_pred
        fig, ax = plt.subplots(figsize=(5, 3))  # Reduced size
        ax.scatter(y_pred, residuals, alpha=0.5, s=10)
        ax.axhline(y=0, color='r', linestyle='--', linewidth=1)
        ax.set_xlabel('Predicted Scores', fontsize=8)
        ax.set_ylabel('Residuals', fontsize=8)
        ax.set_title(f'Residual Plot ({model_name})', fontsize=10)
        ax.tick_params(axis='both', labelsize=8)
        ax.grid(True, linestyle='--', alpha=0.7)
        st.pyplot(fig, use_container_width=False)
        plt.close()


def predict_performance(df, model):
    """Predict student performance based on user input"""
    with st.container():
        st.subheader("Student Performance Prediction")

        # Create a card-like container for the form
        with st.form("prediction_form"):
            col1, col2 = st.columns(2)

            with col1:
                gender = st.selectbox("Gender", options=df['gender'].unique(), key='gender_select')
                ethnicity = st.selectbox("Ethnicity", options=df['ethnicity'].unique(), key='ethnicity_select')
                parental_education = st.selectbox("Parental Education",
                                                  options=df['parental_education'].unique(),
                                                  key='parental_education_select')

            with col2:
                lunch = st.selectbox("Lunch Type", options=df['lunch'].unique(), key='lunch_select')
                test_prep = st.selectbox("Test Preparation", options=df['test_prep'].unique(), key='test_prep_select')

            submitted = st.form_submit_button("Predict Performance", type="primary")

            if submitted:
                input_data = pd.DataFrame({
                    'gender': [gender],
                    'ethnicity': [ethnicity],
                    'parental_education': [parental_education],
                    'lunch': [lunch],
                    'test_prep': [test_prep]
                })

                with st.spinner("Predicting..."):
                    avg_pred = model.predict(input_data)[0]
                    performance = ("Excellent" if avg_pred >= 90 else "Very Good" if avg_pred >= 80 else
                    "Good" if avg_pred >= 70 else "Satisfactory" if avg_pred >= 60 else "Needs Improvement")

                    # Display results in a card
                    st.markdown(f"""
                    <div class="card">
                        <h4 style="color: #4CAF50; margin-bottom: 10px;">Prediction Results</h4>
                        <p><strong>Predicted Average Score:</strong> {avg_pred:.2f}</p>
                        <p><strong>Overall Performance:</strong> {performance}</p>
                    </div>
                    """, unsafe_allow_html=True)


def analyze_factors(df):
    """Analyze and visualize factors affecting student performance"""
    st.subheader("Analyze Factors Affecting Performance")

    visualizations = {
        "Bar Graph: Average Scores by Gender": "gender",
        "Bar Graph: Average Scores by Ethnicity": "ethnicity",
        "Bar Graph: Average Scores by Parental Education": "parental_education",
        "Bar Graph: Average Scores by Lunch Type": "lunch",
        "Bar Graph: Average Scores by Test Preparation": "test_prep",
        "Line Graph: Subject Performance Over Average Scores": "subject_performance",
        "Pie Chart: Pass Rate by Test Preparation": "pass_rate",
        "Histogram: Distribution of Average Scores": "distribution",
        "Show All Visualizations": "all",
        "Performance Insights": "insights"
    }

    selected = st.multiselect(
        "Select Visualizations",
        options=list(visualizations.keys()),
        default=["Bar Graph: Average Scores by Gender"]
    )

    # Handle "Show All Visualizations"
    if "Show All Visualizations" in selected:
        selected = [key for key in visualizations.keys() if
                    key not in ["Show All Visualizations", "Performance Insights"]]

    for choice in selected:
        with st.container():
            if choice == "Bar Graph: Average Scores by Gender":
                st.subheader(choice)
                fig, ax = plt.subplots(figsize=(5, 3))  # Reduced size
                sns.barplot(x='gender', y='average_score', data=df, errorbar=None, ax=ax, palette='Blues')
                ax.set_title('Average Scores by Gender', fontsize=10)
                ax.set_xlabel('Gender', fontsize=8)
                ax.set_ylabel('Average Score', fontsize=8)
                ax.tick_params(axis='both', labelsize=8)
                ax.grid(True, linestyle='--', alpha=0.7)
                st.pyplot(fig, use_container_width=False)
                plt.close()

            elif choice == "Bar Graph: Average Scores by Ethnicity":
                st.subheader(choice)
                fig, ax = plt.subplots(figsize=(5, 3))  # Reduced size
                sns.barplot(x='ethnicity', y='average_score', data=df, errorbar=None, ax=ax, palette='Blues')
                ax.set_title('Average Scores by Ethnicity', fontsize=10)
                ax.set_xlabel('Ethnicity', fontsize=8)
                ax.set_ylabel('Average Score', fontsize=8)
                ax.tick_params(axis='both', labelsize=8)
                ax.grid(True, linestyle='--', alpha=0.7)
                st.pyplot(fig, use_container_width=False)
                plt.close()

            elif choice == "Bar Graph: Average Scores by Parental Education":
                st.subheader(choice)
                fig, ax = plt.subplots(figsize=(5, 3))  # Reduced size
                sns.barplot(x='parental_education', y='average_score', data=df, errorbar=None, ax=ax, palette='Blues')
                ax.set_title('Average Scores by Parental Education', fontsize=10)
                ax.set_xlabel('Parental Education', fontsize=8)
                ax.set_ylabel('Average Score', fontsize=8)
                ax.tick_params(axis='both', labelsize=8)
                plt.xticks(rotation=45)
                ax.grid(True, linestyle='--', alpha=0.7)
                st.pyplot(fig, use_container_width=False)
                plt.close()

            elif choice == "Bar Graph: Average Scores by Lunch Type":
                st.subheader(choice)
                fig, ax = plt.subplots(figsize=(5, 3))  # Reduced size
                sns.barplot(x='lunch', y='average_score', data=df, errorbar=None, ax=ax, palette='Blues')
                ax.set_title('Average Scores by Lunch Type', fontsize=10)
                ax.set_xlabel('Lunch Type', fontsize=8)
                ax.set_ylabel('Average Score', fontsize=8)
                ax.tick_params(axis='both', labelsize=8)
                ax.grid(True, linestyle='--', alpha=0.7)
                st.pyplot(fig, use_container_width=False)
                plt.close()

            elif choice == "Bar Graph: Average Scores by Test Preparation":
                st.subheader(choice)
                fig, ax = plt.subplots(figsize=(5, 3))  # Reduced size
                sns.barplot(x='test_prep', y='average_score', data=df, errorbar=None, ax=ax, palette='Blues')
                ax.set_title('Average Scores by Test Preparation', fontsize=10)
                ax.set_xlabel('Test Preparation', fontsize=8)
                ax.set_ylabel('Average Score', fontsize=8)
                ax.tick_params(axis='both', labelsize=8)
                ax.grid(True, linestyle='--', alpha=0.7)
                st.pyplot(fig, use_container_width=False)
                plt.close()

            elif choice == "Line Graph: Subject Performance Over Average Scores":
                st.subheader(choice)
                fig, ax = plt.subplots(figsize=(5, 3))  # Reduced size
                subjects = ['math_score', 'reading_score', 'writing_score']
                subject_names = ['Math', 'Reading', 'Writing']
                for subject, name in zip(subjects, subject_names):
                    sns.lineplot(x=df['average_score'], y=df[subject], label=name, ax=ax, linewidth=1)
                ax.set_title('Subject Performance Over Average Scores', fontsize=10)
                ax.set_xlabel('Average Score', fontsize=8)
                ax.set_ylabel('Subject Score', fontsize=8)
                ax.tick_params(axis='both', labelsize=8)
                ax.legend(fontsize=8)
                ax.grid(True, linestyle='--', alpha=0.7)
                st.pyplot(fig, use_container_width=False)
                plt.close()

            elif choice == "Pie Chart: Pass Rate by Test Preparation":
                st.subheader(choice)
                pass_rate = df.groupby('test_prep')['average_score'].apply(lambda x: (x >= 60).mean()) * 100
                fig, ax = plt.subplots(figsize=(5, 3))  # Reduced size
                ax.pie(pass_rate, labels=pass_rate.index, autopct='%1.1f%%', startangle=90,
                       colors=sns.color_palette('Blues'), textprops={'fontsize': 8})
                ax.set_title('Pass Rate by Test Preparation', fontsize=10)
                st.pyplot(fig, use_container_width=False)
                plt.close()

            elif choice == "Histogram: Distribution of Average Scores":
                st.subheader(choice)
                fig, ax = plt.subplots(figsize=(5, 3))  # Reduced size
                sns.histplot(df['average_score'], bins=12, kde=True, ax=ax, color='skyblue', linewidth=0.5)
                ax.set_title('Distribution of Average Scores', fontsize=10)
                ax.set_xlabel('Average Score', fontsize=8)
                ax.set_ylabel('Number of Students', fontsize=8)
                ax.tick_params(axis='both', labelsize=8)
                ax.grid(True, linestyle='--', alpha=0.7)
                ax.axvline(df['average_score'].mean(), color='red', linestyle='--',
                           label=f'Mean: {df["average_score"].mean():.2f}', linewidth=1)
                ax.legend(fontsize=8)
                st.pyplot(fig, use_container_width=False)
                plt.close()

            elif choice == "Performance Insights":
                st.subheader(choice)
                gender_avg = df.groupby('gender')['average_score'].mean().to_dict()
                ethnicity_avg = df.groupby('ethnicity')['average_score'].mean().to_dict()
                education_avg = df.groupby('parental_education')['average_score'].mean().to_dict()
                lunch_avg = df.groupby('lunch')['average_score'].mean().to_dict()
                test_prep_avg = df.groupby('test_prep')['average_score'].mean().to_dict()

                highest_gender = max(gender_avg.items(), key=lambda x: x[1])
                lowest_gender = min(gender_avg.items(), key=lambda x: x[1])
                highest_ethnicity = max(ethnicity_avg.items(), key=lambda x: x[1])
                lowest_ethnicity = min(ethnicity_avg.items(), key=lambda x: x[1])
                highest_education = max(education_avg.items(), key=lambda x: x[1])
                lowest_education = min(education_avg.items(), key=lambda x: x[1])
                highest_lunch = max(lunch_avg.items(), key=lambda x: x[1])
                lowest_lunch = min(lunch_avg.items(), key=lambda x: x[1])
                highest_test_prep = max(test_prep_avg.items(), key=lambda x: x[1])
                lowest_test_prep = min(test_prep_avg.items(), key=lambda x: x[1])

                df['passed'] = df['average_score'] >= 60
                pass_rate_test_prep = df.groupby('test_prep')['passed'].mean() * 100
                percentiles = np.percentile(df['average_score'], [25, 50, 75])
                subject_avg = {
                    'Math': df['math_score'].mean(),
                    'Reading': df['reading_score'].mean(),
                    'Writing': df['writing_score'].mean()
                }
                strongest_subject = max(subject_avg.items(), key=lambda x: x[1])
                weakest_subject = min(subject_avg.items(), key=lambda x: x[1])
                test_prep_improvement = test_prep_avg['completed'] - test_prep_avg['none']

                st.markdown(f"""
                <div class="card">
                    <h4 style="color: #4CAF50; margin-bottom: 15px;">Key Performance Insights</h4>

                    <div style="margin-bottom: 15px;">
                        <h5 style="color: #4CAF50; margin-bottom: 5px;">1. Performance by Demographic Groups</h5>
                        <ul style="margin-top: 0;">
                            <li>Highest performing gender: <strong>{highest_gender[0]}</strong> (Average: {highest_gender[1]:.2f})</li>
                            <li>Highest performing ethnicity: <strong>{highest_ethnicity[0]}</strong> (Average: {highest_ethnicity[1]:.2f})</li>
                            <li>Highest performing parental education: <strong>{highest_education[0]}</strong> (Average: {highest_education[1]:.2f})</li>
                        </ul>
                    </div>

                    <div style="margin-bottom: 15px;">
                        <h5 style="color: #4CAF50; margin-bottom: 5px;">2. Critical Factors</h5>
                        <ul style="margin-top: 0;">
                            <li>Lunch type impact: Students with <strong>{highest_lunch[0]}</strong> lunch score {abs(highest_lunch[1] - lowest_lunch[1]):.2f} points higher</li>
                            <li>Test prep impact: Students who <strong>{highest_test_prep[0]}</strong> score {abs(highest_test_prep[1] - lowest_test_prep[1]):.2f} points higher</li>
                        </ul>
                    </div>

                    <div style="margin-bottom: 15px;">
                        <h5 style="color: #4CAF50; margin-bottom: 5px;">3. Achievement Distribution</h5>
                        <ul style="margin-top: 0;">
                            <li>25% of students score below {percentiles[0]:.2f}</li>
                            <li>50% of students score below {percentiles[1]:.2f} (median)</li>
                            <li>75% of students score below {percentiles[2]:.2f}</li>
                            <li>Overall average score: {df['average_score'].mean():.2f}</li>
                        </ul>
                    </div>

                    <div style="margin-bottom: 15px;">
                        <h5 style="color: #4CAF50; margin-bottom: 5px;">4. Subject Performance</h5>
                        <ul style="margin-top: 0;">
                            <li>Best subject: <strong>{strongest_subject[0]}</strong> (Average: {strongest_subject[1]:.2f})</li>
                            <li>Weakest subject: <strong>{weakest_subject[0]}</strong> (Average: {weakest_subject[1]:.2f})</li>
                        </ul>
                    </div>

                    <div>
                        <h5 style="color: #4CAF50; margin-bottom: 5px;">5. Intervention Opportunities</h5>
                        <ul style="margin-top: 0;">
                            <li>Test prep improves scores by {test_prep_improvement:.2f} points</li>
                            <li>Pass rate (completed test prep): {pass_rate_test_prep['completed']:.2f}%</li>
                            <li>Pass rate (no test prep): {pass_rate_test_prep['none']:.2f}%</li>
                        </ul>
                    </div>
                </div>
                """, unsafe_allow_html=True)


def home_page():
    """Home page content"""
    st.header("Welcome to Student Performance Analysis")

    # Hero section
    st.markdown("""
    <div class="card">
        <h3 style="color: #4CAF50;">Analyze, Predict, and Improve Student Performance</h3>
        <p>This application helps educators and administrators understand factors affecting student performance 
        and predict outcomes based on various demographic and academic factors.</p>
    </div>
    """, unsafe_allow_html=True)

    # Features overview
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="card">
            <h4 style="color: #4CAF50;">📊 Data Analysis</h4>
            <p>Explore visualizations of student performance across different demographics and factors.</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="card">
            <h4 style="color: #4CAF50;">🔮 Performance Prediction</h4>
            <p>Predict student performance based on input features using machine learning models.</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="card">
            <h4 style="color: #4CAF50;">📈 Model Evaluation</h4>
            <p>Compare different machine learning models and their performance metrics.</p>
        </div>
        """, unsafe_allow_html=True)

    # Quick start guide
    st.markdown("""
    <div class="card">
        <h4 style="color: #4CAF50;">🚀 Quick Start</h4>
        <ol>
            <li>Use the tabs above to navigate between different sections</li>
            <li>For analysis, select visualizations from the dropdown</li>
            <li>For predictions, fill out the form and click "Predict"</li>
            <li>View model performance metrics in the evaluation section</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)


def main():
    """Main function to run the Streamlit app"""
    st.title("📊 Student Performance Analysis")

    # Use tabs for main navigation
    tab1, tab2, tab3, tab4 = st.tabs(["🏠 Home", "📈 Model Evaluation", "🔮 Prediction", "📊 Analysis"])

    X_train, X_test, y_train, y_test, df = load_and_preprocess_data()

    if df is None:
        return

    models, results, predictions, y_test = build_and_evaluate_models(X_train, X_test, y_train, y_test)

    if models is None:
        return

    with tab1:
        home_page()

    with tab2:
        st.header("Model Evaluation")

        # Model metrics in cards
        st.subheader("Performance Metrics")
        cols = st.columns(3)
        for i, (name, metrics) in enumerate(results.items()):
            with cols[i % 3]:
                st.markdown(f"""
                <div class="card">
                    <h4 style="color: #4CAF50;">{name}</h4>
                    <p><strong>RMSE:</strong> {metrics['RMSE']:.2f}</p>
                    <p><strong>R² Score:</strong> {metrics['R2']:.2f}</p>
                    <p><strong>MAE:</strong> {metrics['MAE']:.2f}</p>
                </div>
                """, unsafe_allow_html=True)

        with st.expander("View Detailed Plots"):
            plot_results(results, predictions, y_test)

    with tab3:
        predict_performance(df, models['Random Forest'])

    with tab4:
        analyze_factors(df)


if __name__ == "__main__":
    main()