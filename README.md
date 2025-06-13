# MLOps Project – Traffic Forecasting and Incident Analysis

## Objective

The goal of this project is to build a complete MLOps pipeline for a real-world use case: **forecasting road traffic** and **analyzing traffic incidents** using real-time data from the TomTom API and a weather API.

## Key Features

- Daily data updates through automated ingestion
- Fully automated processing pipeline (ETL, training, deployment)
- Continuous monitoring of model performance and drift
- Secure and scalable prediction API
- (Optional) Automatic incident summaries generated via LLMs

## Project Structure

_To be completed..._

## Data Sources

- **Incidents and traffic**: [TomTom Traffic API](https://developer.tomtom.com/traffic-api/api-explorer)
- **Weather (upcoming)**: External API (e.g., OpenWeatherMap)

## Running the Script

Before running, define your TomTom API key as an environment variable:

```bash
export API_TOMTOM=your_key_here
