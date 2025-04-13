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

# File path
file_path = r"C:\Users\bijay\Documents\dsp_project_sem4\code\StudentsPerformance.csv"


def load_and_preprocess_data():
    """Load and preprocess the dataset with enhanced checks"""
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

        # Check for missing values
        print("\nMissing Values:")
        print(df.isnull().sum())

        # Handle missing values
        df.fillna(df.select_dtypes(include=np.number).mean(), inplace=True)
        df.fillna(df.select_dtypes(include=object).mode().iloc[0], inplace=True)

        # Check for duplicates
        print("\nDuplicate Rows:", df.duplicated().sum())
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
        print(f"Error: File not found at {file_path}")
        return None, None, None, None, None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None, None, None, None


def build_and_evaluate_models(X_train, X_test, y_train, y_test):
    """Build and evaluate three regression models"""
    if X_train is None:
        return None, None, None, None

    # Define categorical features
    categorical_features = ['gender', 'ethnicity', 'parental_education', 'lunch', 'test_prep']

    # Create preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])

    # Define models
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

    # Train and evaluate models
    results = {}
    predictions = {}
    for name, model in models.items():
        # Train
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)

        # Evaluate
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)  # Added MAE

        results[name] = {'RMSE': rmse, 'R2': r2, 'MAE': mae}
        predictions[name] = y_pred

    return models, results, predictions, y_test


def plot_results(results, predictions, y_test):
    """Generate required regression plots"""
    # RMSE Bar Chart
    plt.figure(figsize=(10, 6))
    rmses = [results[model]['RMSE'] for model in results]
    models = list(results.keys())
    sns.barplot(x=models, y=rmses)
    plt.title('RMSE Comparison Across Models')
    plt.ylabel('RMSE')
    plt.xticks(rotation=45)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

    # Predicted vs Actual Plot and Residual Plot for each model
    for model_name, y_pred in predictions.items():
        # Predicted vs Actual
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.xlabel('Actual Scores')
        plt.ylabel('Predicted Scores')
        plt.title(f'Predicted vs Actual Scores ({model_name})')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.show()

        # Residual Plot
        residuals = y_test - y_pred
        plt.figure(figsize=(10, 6))
        plt.scatter(y_pred, residuals, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted Scores')
        plt.ylabel('Residuals')
        plt.title(f'Residual Plot ({model_name})')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.show()


def validate_input(value, options):
    """Validate user input against a list of valid options (case-insensitive)"""
    options_lower = [option.lower() for option in options]
    while value.lower() not in options_lower:
        print(f"Invalid input. Please choose from: {', '.join(options)}")
        value = input("Try again: ")
    return options[options_lower.index(value.lower())]


def predict_performance(df, model):
    """Predict student performance based on user input"""
    print("\n--- Student Performance Prediction ---\n")

    # Get unique values
    gender_options = df['gender'].unique().tolist()
    ethnicity_options = df['ethnicity'].unique().tolist()
    education_options = df['parental_education'].unique().tolist()
    lunch_options = df['lunch'].unique().tolist()
    test_prep_options = df['test_prep'].unique().tolist()

    # Collect user inputs with validation
    gender = validate_input(input(f"Gender ({', '.join(gender_options)}): "), gender_options)
    ethnicity = validate_input(input(f"Ethnicity ({', '.join(ethnicity_options)}): "), ethnicity_options)
    parental_education = validate_input(input(f"Parental education ({', '.join(education_options)}): "),
                                        education_options)
    lunch = validate_input(input(f"Lunch type ({', '.join(lunch_options)}): "), lunch_options)
    test_prep = validate_input(input(f"Test prep ({', '.join(test_prep_options)}): "), test_prep_options)

    # Create input dataframe
    input_data = pd.DataFrame({
        'gender': [gender],
        'ethnicity': [ethnicity],
        'parental_education': [parental_education],
        'lunch': [lunch],
        'test_prep': [test_prep]
    })

    # Predict
    avg_pred = model.predict(input_data)[0]

    # Display results
    print("\n----- Predicted Score -----")
    print(f"Average Score: {avg_pred:.2f}")

    # Interpret score
    performance = "Excellent" if avg_pred >= 90 else "Very Good" if avg_pred >= 80 else \
        "Good" if avg_pred >= 70 else "Satisfactory" if avg_pred >= 60 else "Needs Improvement"
    print(f"Overall Performance: {performance}")


def analyze_factors(df):
    """Analyze and visualize factors affecting student performance"""
    print("\n--- Analyzing Factors Affecting Student Performance ---\n")

    # Define available visualizations
    visualizations = {
        "1": "Bar Graph: Average Scores by Gender",
        "2": "Bar Graph: Average Scores by Ethnicity",
        "3": "Bar Graph: Average Scores by Parental Education",
        "4": "Bar Graph: Average Scores by Lunch Type",
        "5": "Bar Graph: Average Scores by Test Preparation",
        "6": "Line Graph: Subject Performance Over Average Scores",
        "7": "Pie Chart: Pass Rate by Test Preparation",
        "8": "Histogram: Distribution of Average Scores",
        "9": "Show All Visualizations",
        "10": "Performance Insights",
        "11": "Exit"
    }

    while True:
        print("\nSelect visualizations to display (separate choices by commas):")
        for key, value in visualizations.items():
            print(f"{key}. {value}")

        choices = input("\nEnter your choices (e.g., 1,2,3): ").strip().split(',')

        if "11" in choices:
            print("Exiting visualization menu.")
            break

        if "10" in choices:
            calculate_performance_insights(df)
            continue

        if "9" in choices:
            choices = [str(i) for i in range(1, 9)]

        for choice in choices:
            choice = choice.strip()
            if choice == "1":
                plt.figure(figsize=(8, 6))
                sns.barplot(x='gender', y='average_score', data=df, errorbar=None)
                plt.title('Average Scores by Gender')
                plt.xlabel('Gender')
                plt.ylabel('Average Score')
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.show()

            elif choice == "2":
                plt.figure(figsize=(10, 6))
                sns.barplot(x='ethnicity', y='average_score', data=df, errorbar=None)
                plt.title('Average Scores by Ethnicity')
                plt.xlabel('Ethnicity')
                plt.ylabel('Average Score')
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.show()

            elif choice == "3":
                plt.figure(figsize=(12, 6))
                sns.barplot(x='parental_education', y='average_score', data=df, errorbar=None)
                plt.title('Average Scores by Parental Education')
                plt.xlabel('Parental Education')
                plt.ylabel('Average Score')
                plt.xticks(rotation=45)
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.show()

            elif choice == "4":
                plt.figure(figsize=(8, 6))
                sns.barplot(x='lunch', y='average_score', data=df, errorbar=None)
                plt.title('Average Scores by Lunch Type')
                plt.xlabel('Lunch Type')
                plt.ylabel('Average Score')
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.show()

            elif choice == "5":
                plt.figure(figsize=(8, 6))
                sns.barplot(x='test_prep', y='average_score', data=df, errorbar=None)
                plt.title('Average Scores by Test Preparation')
                plt.xlabel('Test Preparation')
                plt.ylabel('Average Score')
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.show()

            elif choice == "6":
                plt.figure(figsize=(10, 6))
                subjects = ['math_score', 'reading_score', 'writing_score']
                subject_names = ['Math', 'Reading', 'Writing']
                for subject, name in zip(subjects, subject_names):
                    sns.lineplot(x=df['average_score'], y=df[subject], label=name)
                plt.title('Subject Performance Over Average Scores')
                plt.xlabel('Average Score')
                plt.ylabel('Subject Score')
                plt.legend()
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.show()

            elif choice == "7":
                pass_rate = df.groupby('test_prep')['average_score'].apply(lambda x: (x >= 60).mean()) * 100
                plt.figure(figsize=(8, 6))
                plt.pie(pass_rate, labels=pass_rate.index, autopct='%1.1f%%', startangle=90)
                plt.title('Pass Rate by Test Preparation')
                plt.show()

            elif choice == "8":
                plt.figure(figsize=(10, 6))
                sns.histplot(df['average_score'], bins=20, kde=True)
                plt.title('Distribution of Average Scores')
                plt.xlabel('Average Score')
                plt.ylabel('Number of Students')
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.axvline(df['average_score'].mean(), color='red', linestyle='--',
                            label=f'Mean: {df["average_score"].mean():.2f}')
                plt.legend()
                plt.show()

            else:
                print(f"Invalid choice: {choice}. Skipping.")


def calculate_performance_insights(df):
    """Calculate and display key performance insights"""
    print("\n----- Key Performance Insights -----")

    # Calculate average scores by various factors
    gender_avg = df.groupby('gender')['average_score'].mean().to_dict()
    ethnicity_avg = df.groupby('ethnicity')['average_score'].mean().to_dict()
    education_avg = df.groupby('parental_education')['average_score'].mean().to_dict()
    lunch_avg = df.groupby('lunch')['average_score'].mean().to_dict()
    test_prep_avg = df.groupby('test_prep')['average_score'].mean().to_dict()

    # Find highest and lowest performing groups
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

    # Calculate pass rate
    df['passed'] = df['average_score'] >= 60
    pass_rate_gender = df.groupby('gender')['passed'].mean() * 100
    pass_rate_test_prep = df.groupby('test_prep')['passed'].mean() * 100

    # Calculate percentiles
    percentiles = np.percentile(df['average_score'], [25, 50, 75])

    # Calculate strongest and weakest subjects
    subject_avg = {
        'Math': df['math_score'].mean(),
        'Reading': df['reading_score'].mean(),
        'Writing': df['writing_score'].mean()
    }
    strongest_subject = max(subject_avg.items(), key=lambda x: x[1])
    weakest_subject = min(subject_avg.items(), key=lambda x: x[1])

    # Calculate improvement potential
    test_prep_improvement = test_prep_avg['completed'] - test_prep_avg['none']

    # Display insights
    print("\n1. Performance by Demographic Groups:")
    print(f"   - Highest performing gender: {highest_gender[0]} (Average: {highest_gender[1]:.2f})")
    print(f"   - Highest performing ethnicity: {highest_ethnicity[0]} (Average: {highest_ethnicity[1]:.2f})")
    print(f"   - Highest performing parental education: {highest_education[0]} (Average: {highest_education[1]:.2f})")

    print("\n2. Critical Factors:")
    print(
        f"   - Lunch type impact: Students with {highest_lunch[0]} lunch score {abs(highest_lunch[1] - lowest_lunch[1]):.2f} points higher")
    print(
        f"   - Test prep impact: Students who {highest_test_prep[0]} score {abs(highest_test_prep[1] - lowest_test_prep[1]):.2f} points higher")

    print("\n3. Achievement Distribution:")
    print(f"   - 25% of students score below {percentiles[0]:.2f}")
    print(f"   - 50% of students score below {percentiles[1]:.2f} (median)")
    print(f"   - 75% of students score below {percentiles[2]:.2f}")
    print(f"   - Overall average score: {df['average_score'].mean():.2f}")

    print("\n4. Subject Performance:")
    print(f"   - Best subject: {strongest_subject[0]} (Average: {strongest_subject[1]:.2f})")
    print(f"   - Weakest subject: {weakest_subject[0]} (Average: {weakest_subject[1]:.2f})")

    print("\n5. Intervention Opportunities:")
    print(f"   - Test prep improves scores by {test_prep_improvement:.2f} points")
    print(f"   - Pass rate (completed test prep): {pass_rate_test_prep['completed']:.2f}%")
    print(f"   - Pass rate (no test prep): {pass_rate_test_prep['none']:.2f}%")

    # Calculate influential factors
    impact_values = {
        'Gender': abs(highest_gender[1] - lowest_gender[1]),
        'Ethnicity': abs(highest_ethnicity[1] - lowest_ethnicity[1]),
        'Parental Education': abs(highest_education[1] - lowest_education[1]),
        'Lunch Type': abs(highest_lunch[1] - lowest_lunch[1]),
        'Test Preparation': abs(highest_test_prep[1] - lowest_test_prep[1])
    }

    sorted_factors = sorted(impact_values.items(), key=lambda x: x[1], reverse=True)
    print("\n6. Most Influential Factors (Ranked):")
    for i, (factor, impact) in enumerate(sorted_factors, 1):
        print(f"   {i}. {factor}: {impact:.2f} point difference")


def main():
    """Main function to run the program"""
    # Load and preprocess data
    X_train, X_test, y_train, y_test, df = load_and_preprocess_data()

    if df is None:
        return

    # Build and evaluate models
    models, results, predictions, y_test = build_and_evaluate_models(X_train, X_test, y_train, y_test)

    if models is None:
        return

    # Print evaluation results
    print("\n--- Model Evaluation Results ---")
    for model, metrics in results.items():
        print(f"\n{model}:")
        print(f"RMSE: {metrics['RMSE']:.2f}")
        print(f"R2 Score: {metrics['R2']:.2f}")
        print(f"MAE: {metrics['MAE']:.2f}")

    # Plot results
    plot_results(results, predictions, y_test)

    # Interactive menu
    while True:
        print("\n=== Student Performance Analysis System ===")
        print("1. Student Performance Prediction")
        print("2. Analyze Factors Affecting Students")
        print("3. Exit")

        choice = input("\nEnter your choice (1-3): ")

        if choice == '1':
            predict_performance(df, models['Random Forest'])
        elif choice == '2':
            analyze_factors(df)
        elif choice == '3':
            print("Exiting program. Goodbye!")
            break
        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    main()