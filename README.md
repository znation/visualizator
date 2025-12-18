---
title: Visualizator
emoji: 📊
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.0.0
app_file: app.py
pinned: false
---

# 📊 Visualizator

A Hugging Face Space application that generates data visualizations from URLs using natural language queries and AI-powered Vega-Lite specification generation.

## Features

- 🔗 Load data from remote URLs (CSV and TSV formats)
- 💬 Natural language queries for visualization requests
- 🤖 AI-powered Vega-Lite specification generation using Hugging Face Inference API
- 🔄 Auto-retry mechanism (up to 5 attempts) for robust spec generation
- 🎨 Interactive visualizations rendered with Vega-Lite
- 🔐 Optional HuggingFace OAuth authentication for API access

## Installation

### Local Development

1. Clone the repository:
```bash
git clone <repository-url>
cd visualizator
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python app.py
```

The application will launch and be accessible at `http://localhost:7860`

### Deploy to Hugging Face Spaces

1. Create a new Space on Hugging Face
2. Upload these files to your Space:
   - `app.py`
   - `requirements.txt`
   - `README.md`

3. (Optional) Enable OAuth for automatic authentication:
   - Go to your Space settings
   - Enable OAuth under the "Security" section
   - Follow the guide: https://huggingface.co/docs/hub/en/spaces-oauth

## Usage

1. **Enter Data URL**: Provide a URL to your CSV or TSV data file
   - Example: `https://raw.githubusercontent.com/vega/vega-datasets/master/data/cars.json`

2. **Write Your Query**: Describe the visualization you want in natural language
   - Example: "Show me a scatter plot of horsepower vs miles per gallon"
   - Example: "Create a bar chart of average acceleration by origin"
   - Example: "Display a line chart showing the trend of displacement over time"

3. **Add Token (Optional)**: Enter your HuggingFace API token for authenticated access
   - Get your token at: https://huggingface.co/settings/tokens
   - Required if running without OAuth or for higher rate limits

4. **Generate**: Click "Generate Visualization" and the app will:
   - Load your data from the URL
   - Analyze the data schema
   - Generate a Vega-Lite specification using AI
   - Render the visualization
   - Auto-retry up to 5 times if generation fails

## How It Works

1. **Data Loading**: Downloads data from the provided URL and detects format (CSV/TSV)
2. **Schema Extraction**: Analyzes the data structure and column types
3. **LLM Generation**: Uses Llama-3.3-70B-Instruct via Hugging Face Inference API to generate Vega-Lite specs
4. **Validation**: Checks the generated specification for required fields
5. **Auto-Retry**: If generation fails, automatically retries up to 5 times
6. **Rendering**: Displays the final visualization using Gradio's Plot component

## Supported Data Formats

- ✅ CSV (Comma-Separated Values)
- ✅ TSV (Tab-Separated Values)

## Architecture

- **Frontend**: Gradio for interactive UI
- **Data Processing**: Pandas for data manipulation
- **LLM Integration**: Hugging Face Inference API (Llama-3.3-70B-Instruct)
- **Visualization**: Vega-Lite specifications rendered via Gradio Plot
- **Authentication**: Optional HuggingFace OAuth integration

## Example Queries

- "Show me the relationship between X, Y, and Z"
- "Create a heat map of time of day vs. number of hits"
- "Display a bar chart comparing sales across regions"
- "Generate a scatter plot with price on x-axis and quantity on y-axis, colored by category"
- "Show a line chart of temperature trends over time"

## Configuration

### Environment Variables

- `HF_TOKEN`: Optional default Hugging Face API token

## Troubleshooting

**Issue**: "Error loading data"
- Ensure the URL is accessible and returns valid CSV/TSV data
- Check that the URL uses HTTPS (HTTP is auto-upgraded)

**Issue**: "Failed to generate valid specification after 5 attempts"
- Try rephrasing your query to be more specific
- Ensure your data has appropriate columns for the requested visualization
- Verify your HuggingFace token is valid (if using one)

**Issue**: Rate limiting
- Provide a HuggingFace API token for higher rate limits
- Consider deploying to a Space with OAuth enabled

## License

MIT

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
