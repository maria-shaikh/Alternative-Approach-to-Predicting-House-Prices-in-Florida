# Predicting House Prices in Florida: An Alternative Approach

This repository focuses on developing innovative models to predict house prices in Florida, especially during unforeseen events like the COVID-19 pandemic.

## Project Overview
- **Objective**: Improve the accuracy of house price predictions during black swan events like the COVID-19 pandemic, using AI and machine learning techniques.
- **Methodology**: Utilized Large Language Models (LLMs) such as OpenAI's GPT 3.5 and GPT 4.0, alongside traditional economic indicators, for forecasting. Employed sentiment analysis, extracting data from forums, social media, and real estate marketplaces.
- **Results**: Compared the LLM predictions with baseline models, demonstrating the potential and limitations of AI in real estate forecasting.

## Modeling Techniques
- **ARIMA (AutoRegressive Integrated Moving Average)**: Implemented for time series forecasting to understand and predict future house price trends.
- **XGBoost (eXtreme Gradient Boosting)**: Utilized for its efficiency and effectiveness in handling large datasets with numerous features, aiding in predictive accuracy.
- **Random Forest**: Employed to assess the importance of various features in predicting house prices and to provide a robust model against overfitting.

## Model Performance and Comparison
- **ARIMA**: Exhibited strong performance in stable market conditions but less effective during unpredictable events like the pandemic.
- **XGBoost**: Demonstrated high accuracy in feature-rich environments, outperforming other models in handling complex interactions.
- **Random Forest**: Provided consistent results across various scenarios, with an emphasis on generalizability over precision.

**Comparative Analysis**:
- XGBoost generally outperformed others in terms of precision.
- Random Forest showed the best generalizability, making it more reliable across different market conditions.
- ARIMA was most effective in stable, trend-following scenarios.

## Key Findings
- LLMs showed promising results but are not yet fully reliable for predictive analysis.
- Public sentiment significantly influences house price trends.
- Integrating AI and machine learning in real estate prediction offers new insights but requires further development.

## Limitations and Future Work
- Data post-2020 was not included, limiting the scope.
- High cost and restrictions in data acquisition.
- AI models are still evolving and are somewhat of a 'black box'.

Your feedback and contributions are welcome!
