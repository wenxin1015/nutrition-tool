# Nutritional Insights Tool

A Streamlit app for analyzing nutritional data from drinks and food datasets. The app allows users to upload CSV files, filter data, visualize nutrients, calculate statistics, compare datasets, and generate insights using the Groq LLM API.

---

## Live App

You can access the live app here: [Nutritional Insights Tool](https://nutrition-tool-46vpq9ug8akd3szufcwuvy.streamlit.app/)

---

## Features

- **Upload CSV Files**: Upload drinks and food datasets in CSV format.
- **Filter Data**: Filter datasets by maximum calories and fat content.
- **Visualize Nutrients**: Generate interactive bar charts for top items by selected nutrients.
- **Calculate Statistics**: Compute total calories, average fat, carbs, fiber, protein, and sodium for drinks and food.
- **Compare Datasets**: Compare key metrics (e.g., average calories, fat, carbs) between drinks and food.
- **Generate Insights**: Use the Groq LLM API to summarize nutritional insights and provide healthy menu suggestions.
- **Ask Questions**: Ask questions about the dataset and get answers from the Groq LLM API.

---

## Prerequisites

- Python 3.8 or higher
- Streamlit
- Pandas
- NumPy
- Plotly
- Groq Python SDK

---
## Sample Data
You can download sample CSV files for drinks and food menus to test the app:

starbucks-menu-nutrition-drinks.csv

starbucks-menu-nutrition-food.csv

## How to Use the Live App

1. **Access the Live App**:
   - Go to [Nutritional Insights Tool](https://nutrition-tool-46vpq9ug8akd3szufcwuvy.streamlit.app/).

2. **Upload CSV Files**:
   - Download the sample CSV files from the links above.
   - Use the sidebar to upload the `starbucks-menu-nutrition-drinks.csv` and `starbucks-menu-nutrition-food.csv` files.

3. **Filter Data**:
   - Use the sliders to filter datasets by maximum calories and fat content.

4. **Visualize Nutrients**:
   - Select a nutrient from the dropdown menu to generate a bar chart of the top 10 items.

5. **Ask Questions**:
   - Enter a question in the text box and click "Submit Question" to get an answer from the Groq LLM API.

---
