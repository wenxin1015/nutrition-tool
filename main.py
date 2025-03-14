import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from groq import Groq

# Set page title and layout
st.set_page_config(page_title="Nutritional Insights Tool", layout="wide")

# Title with emojis
st.title("Nutritional Insights Tool")
st.write("Upload CSV files for drinks and food to analyze nutritional data.")

# Sidebar for file uploads
st.sidebar.header("Upload CSV Files")
drinks_file = st.sidebar.file_uploader("Drinks CSV", type=["csv"], help="Upload a CSV file containing drinks data.")
food_file = st.sidebar.file_uploader("Food CSV", type=["csv"], help="Upload a CSV file containing food data.")

# Load CSV file into pandas DataFrame
def load_csv(file):
    try:
        df = pd.read_csv(file)
        
        # Strip whitespace from column names
        df.columns = df.columns.str.strip()

        # Define numeric columns
        numeric_columns = ["Calories", "Fat (g)", "Carb. (g)", "Fiber (g)", "Protein (g)", "Sodium"]
        
        # Convert numeric columns to float64, coercing errors to NaN
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').astype('float64')  # Convert to float64

        # Replace '-' with NaN
        df.replace('-', np.nan, inplace=True)
        
        st.success(f"Successfully loaded: {file.name}")
        return df
    except Exception as e:
        st.error(f"Error loading {file.name}: {e}")
        return None

# Filters dataset based on user-defined criteria
def filter_data(df, criteria):
    try:
        if "max_calories" in criteria:
            df = df[df["Calories"] <= criteria["max_calories"]]  # Filter by max calories
        if "low_fat" in criteria:
            df = df[df["Fat (g)"] <= criteria["low_fat"]]  # Filter by low fat
        return df
    except Exception as e:
        st.error(f"Error filtering data: {e}")
        return None

# Generates bar chart comparing nutrient across menu items
def visualize_nutrition(df, title, nutrient, top_n=10):
    try:
        # Sort the DataFrame by the nutrient column
        sorted_df = df.sort_values(by=nutrient, ascending=False).head(top_n)

        # Create an interactive bar chart using Plotly
        fig = px.bar(
            sorted_df,
            x="Unnamed: 0",
            y=nutrient,
            title=f"{title}: Top {top_n} Items by {nutrient}",
            labels={"Unnamed: 0": "Menu Item", nutrient: nutrient},
            color=nutrient,
            color_continuous_scale="Viridis"
        )
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Error generating visualization: {e}")

# Calculate descriptive statistics for drinks and food datasets
def calculate_statistics(drinks_df, food_df):
    try:
        stats = {
            "drinks": {
                "total_calories": round(float(drinks_df["Calories"].sum()), 2),
                "avg_fat": round(float(drinks_df["Fat (g)"].mean()), 2),
                "avg_carbs": round(float(drinks_df["Carb. (g)"].mean()), 2),
                "avg_fiber": round(float(drinks_df["Fiber (g)"].mean()), 2),
                "avg_protein": round(float(drinks_df["Protein (g)"].mean()), 2),
                "avg_sodium": round(float(drinks_df["Sodium"].mean()), 2),
            },
            "food": {
                "total_calories": round(float(food_df["Calories"].sum()), 2),
                "avg_fat": round(float(food_df["Fat (g)"].mean()), 2),
                "avg_carbs": round(float(food_df["Carb. (g)"].mean()), 2),
                "avg_fiber": round(float(food_df["Fiber (g)"].mean()), 2),
                "avg_protein": round(float(food_df["Protein (g)"].mean()), 2)
            }
        }
        return stats
    except Exception as e:
        st.error(f"Error calculating statistics: {e}")
        return None

# Compare key metrics between drinks and food datasets
def compare_datasets(drinks_df, food_df):
    try:
        comparison = {
            "avg_calories": {
                "drinks": round(float(drinks_df["Calories"].mean()), 2),
                "food": round(float(food_df["Calories"].mean()), 2)
            },
            "avg_fat": {
                "drinks": round(float(drinks_df["Fat (g)"].mean()), 2),
                "food": round(float(food_df["Fat (g)"].mean()), 2)
            },
            "avg_carbs": {
                "drinks": round(float(drinks_df["Carb. (g)"].mean()), 2),
                "food": round(float(food_df["Carb. (g)"].mean()), 2)
            },
            "avg_fiber": {
                "drinks": round(float(drinks_df["Fiber (g)"].mean()), 2),
                "food": round(float(food_df["Fiber (g)"].mean()), 2)
            },
            "avg_protein": {
                "drinks": round(float(drinks_df["Protein (g)"].mean()), 2),
                "food": round(float(food_df["Protein (g)"].mean()), 2)
            },
            "avg_sodium": {
                "drinks": round(float(drinks_df["Sodium"].mean()), 2), 
                "food": None 
            },
        }
        return comparison
    except Exception as e:
        st.error(f"Error comparing datasets: {e}")
        return None

# Summarizes nutritional insights using the Groq LLM API
def summarize_nutrition(drinks_df, food_df):
    try:
        # Extract top 5 drinks by calories
        top_drinks_calories = drinks_df.sort_values(by="Calories", ascending=False).head(5)
        top_drinks_calories = top_drinks_calories[["Unnamed: 0", "Calories"]].to_dict(orient="records")

        # Extract top 5 food items by calories
        top_food_calories = food_df.sort_values(by="Calories", ascending=False).head(5)
        top_food_calories = top_food_calories[["Unnamed: 0", "Calories"]].to_dict(orient="records")

        # Prepare the prompt for Groq LLM
        prompt = f"""
        Here are the top 5 drinks by calories:
        {top_drinks_calories}

        Here are the top 5 food items by calories:
        {top_food_calories}

        Summarize the nutritional insights from this data. Highlight which drinks and food items are the most calorie-dense.
        Also, provide a suggestion for a healthy menu.
        """

        # Initialize Groq client
        api_key = st.secrets["GROQ_API_KEY"]  # Access API key from secrets
        client = Groq(api_key=api_key)

        # Call the Groq LLM API
        response = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model="mixtral-8x7b-32768",  # Use the appropriate Groq model
        )

        # Extract and return the summary
        summary = response.choices[0].message.content
        return summary
    except Exception as e:
        st.error(f"Error summarizing nutrition: {e}")
        return None

# Answers user questions about the dataset using the Groq LLM API
def ask_question(drinks_df, food_df, question):
    try:
        # Prepare the prompt for Groq LLM
        prompt = f"""
        Here is the drinks dataset:
        {drinks_df.head().to_dict(orient="records")}

        Here is the food dataset:
        {food_df.head().to_dict(orient="records")}

        Question: {question}

        Answer the question based on the dataset.
        """

        # Initialize Groq client
        api_key = st.secrets["GROQ_API_KEY"]  # Access API key from secrets
        client = Groq(api_key=api_key)

        # Call the Groq LLM API
        response = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model="mixtral-8x7b-32768",  # Use the appropriate Groq model
        )

        # Extract and return the answer
        answer = response.choices[0].message.content
        return answer
    except Exception as e:
        st.error(f"Error answering question: {e}")
        return None

# Main app logic
if drinks_file is not None and food_file is not None:
    # Load data
    drinks_df = load_csv(drinks_file)
    food_df = load_csv(food_file)

    if drinks_df is not None and food_df is not None:
        # Display data previews
        st.subheader("Data Previews")
        st.write("### Drinks Data")
        st.write(drinks_df.head())

        st.write("### Food Data")
        st.write(food_df.head())

        # Filtering options
        st.header("Filter Data")

        # Slider to filter drinks data
        max_calories_drinks = st.slider("Maximum calories for drinks", 0, 1000, 500)
        low_fat_drinks = st.slider("Maximum fat (g) for drinks", 0, 50, 10)

        # Filter data
        filtered_drinks = filter_data(drinks_df, {"max_calories": max_calories_drinks, "low_fat": low_fat_drinks})

        if filtered_drinks is not None:
            st.subheader("Filtered Drinks Data")
            st.write(filtered_drinks)

        # Slider to filter food data
        max_calories_food = st.slider("Maximum calories for food", 0, 1000, 500)
        low_fat = st.slider("Maximum fat (g) for food", 0, 50, 10)

        # Filter data
        filtered_food = filter_data(food_df, {"max_calories": max_calories_food, "low_fat": low_fat})
        
        if filtered_food is not None:
            st.subheader("Filtered Food Data")
            st.write(filtered_food)
            

        # Visualizations
        st.subheader("Visualizations")
        nutrient_options = ["Calories", "Fat (g)", "Carb. (g)", "Fiber (g)", "Protein (g)"]
        selected_nutrient = st.selectbox("Select a nutrient to visualize", nutrient_options)
        if st.subheader(f"Show Top 10 Items by {selected_nutrient}"):
            visualize_nutrition(drinks_df, "Drinks", selected_nutrient, top_n=10)
            visualize_nutrition(food_df, "Food", selected_nutrient, top_n=10)

        # Calculate and display statistics
        if st.subheader("Nutritional Statistics"):
            stats = calculate_statistics(drinks_df, food_df)
            if stats:
                st.write("### Drinks Statistics")
                st.write(stats["drinks"])
                st.write("### Food Statistics")
                st.write(stats["food"])

        # Compare datasets
        if st.subheader("Dataset Comparison"):
            comparison = compare_datasets(drinks_df, food_df)
            if comparison:
                st.write(comparison)

        # Summarize nutritional insights
        if st.subheader("Summarize Nutritional Insights"):
            with st.spinner("Generating insights..."):
                summary = summarize_nutrition(drinks_df, food_df)
                if summary:
                    st.write(summary)

        # Ask a question
        st.header("❓ Ask a Question")
        question = st.text_input("Ask a question about the dataset:")
        if st.button("Submit Question"):
            if question:
                with st.spinner("Thinking..."):
                    answer = ask_question(drinks_df, food_df, question)
                    if answer:
                        st.subheader("💡 Answer")
                        st.write(answer)
            else:
                st.warning("Please enter a question.")