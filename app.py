import altair as alt
import gradio as gr
import pandas as pd
import requests
import json
import os
import sys
from huggingface_hub import InferenceClient
from typing import Optional, Tuple
import traceback

# Initialize Hugging Face Inference Client
def get_inference_client(token: Optional[str] = None) -> InferenceClient:
    """Initialize the Hugging Face Inference Client with optional token."""
    if token:
        return InferenceClient(token=token)
    return InferenceClient()

def load_data_from_url(url: str) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Load data from a URL supporting CSV, TSV, and JSON formats.

    Args:
        url: URL to the data file

    Returns:
        Tuple of (DataFrame, error_message)
    """
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        # Detect file type from URL or content-type
        if url.endswith('.json') or 'application/json' in response.headers.get('content-type', ''):
            df = pd.read_json(pd.io.common.StringIO(response.text))
        elif url.endswith('.tsv') or 'text/tab-separated-values' in response.headers.get('content-type', ''):
            df = pd.read_csv(pd.io.common.StringIO(response.text), sep='\t')
        elif url.endswith('.csv') or 'text/csv' in response.headers.get('content-type', ''):
            df = pd.read_csv(pd.io.common.StringIO(response.text))
        else:
            # Default to CSV
            df = pd.read_csv(pd.io.common.StringIO(response.text))

        return df, None
    except Exception as e:
        return None, f"Error loading data: {str(e)}"

def get_data_schema(df: pd.DataFrame) -> str:
    """
    Generate a schema description of the DataFrame for the LLM.

    Args:
        df: pandas DataFrame

    Returns:
        String description of the schema
    """
    schema_parts = []
    schema_parts.append(f"Dataset has {len(df)} rows and {len(df.columns)} columns.\n")
    schema_parts.append("Columns:")

    for col in df.columns:
        dtype = str(df[col].dtype)
        sample_values = df[col].dropna().head(3).tolist()
        schema_parts.append(f"  - {col} ({dtype}): sample values {sample_values}")

    return "\n".join(schema_parts)

def generate_vega_lite_spec(
    query: str,
    schema: str,
    data_url: str,
    token: Optional[str] = None
) -> Tuple[Optional[dict], Optional[str]]:
    """
    Generate a Vega-Lite specification using an LLM.

    Args:
        query: User's visualization query
        schema: Data schema description
        data_url: URL to the data file
        token: Optional HuggingFace token

    Returns:
        Tuple of (vega_lite_spec_dict, error_message)
    """
    try:
        client = get_inference_client(token)

        prompt = f"""You are a data visualization expert. Generate a valid Vega-Lite specification (JSON) based on the user's query and data schema.

User Query: {query}

Data Schema:
{schema}

Data URL: {data_url}

Requirements:
1. Generate ONLY valid Vega-Lite JSON specification
2. Use the data URL provided in the "data" field with "url" property
3. Choose appropriate mark types and encodings based on the query
4. Include appropriate titles and labels
5. Make sure the field names match exactly with the column names from the schema
6. Return ONLY the JSON object, no markdown formatting or explanations
7. Use the Vega-Lite schema version 5, as described at https://vega.github.io/schema/vega-lite/v5.json

Generate the Vega-Lite specification now:"""

        # Use a capable model for code generation
        response = client.chat.completions.create(
            model="meta-llama/Llama-3.3-70B-Instruct",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2000
        )

        spec_text = response.choices[0].message.content.strip()

        # Clean up markdown code blocks if present
        if spec_text.startswith('```'):
            lines = spec_text.split('\n')
            spec_text = '\n'.join(lines[1:-1]) if len(lines) > 2 else spec_text
            spec_text = spec_text.replace('```json', '').replace('```', '').strip()

        # Parse JSON
        spec = json.loads(spec_text)

        # Ensure the data URL is set correctly
        if 'data' not in spec:
            spec['data'] = {}
        spec['data']['url'] = data_url

        return spec, None

    except Exception as e:
        return None, f"Error generating specification: {str(e)}\n{traceback.format_exc()}"

def create_visualization(
    data_url: str,
    query: str,
    token: Optional[str] = None,
    max_retries: int = 5
) -> Tuple[Optional[dict], Optional[str], str]:
    """
    Create a visualization by loading data and generating Vega-Lite spec with auto-retry.

    Args:
        data_url: URL to the data file
        query: User's visualization query
        token: Optional HuggingFace token
        max_retries: Maximum number of retry attempts

    Returns:
        Tuple of (vega_lite_spec, error_message, log_message)
    """
    log_messages = []

    # Load data
    log_messages.append("Loading data from URL...")
    df, error = load_data_from_url(data_url)
    if error:
        return None, error, "\n".join(log_messages)

    log_messages.append(f"✓ Data loaded successfully: {len(df)} rows, {len(df.columns)} columns")

    # Get schema
    schema = get_data_schema(df)
    log_messages.append(f"✓ Schema extracted")

    # Try to generate valid spec with retries
    for attempt in range(max_retries):
        log_messages.append(f"\nAttempt {attempt + 1}/{max_retries}: Generating Vega-Lite specification...")

        spec, error = generate_vega_lite_spec(query, schema, data_url, token)

        if error:
            log_messages.append(f"✗ Generation failed: {error}")
            if attempt < max_retries - 1:
                log_messages.append("  Retrying...")
            continue

        # Basic validation
        try:
            if not isinstance(spec, dict):
                raise ValueError("Specification is not a dictionary")
            if 'mark' not in spec and 'layer' not in spec and 'hconcat' not in spec and 'vconcat' not in spec:
                raise ValueError("Specification missing mark or composition")
            if 'encoding' not in spec and 'layer' not in spec and 'hconcat' not in spec and 'vconcat' not in spec:
                raise ValueError("Specification missing encoding or composition")

            log_messages.append("✓ Specification generated and validated successfully!")
            return spec, None, "\n".join(log_messages)

        except Exception as e:
            log_messages.append(f"✗ Validation failed: {str(e)}")
            if attempt < max_retries - 1:
                log_messages.append("  Retrying...")
            continue

    error_msg = f"Failed to generate valid specification after {max_retries} attempts"
    log_messages.append(f"\n✗ {error_msg}")
    return None, error_msg, "\n".join(log_messages)

def visualize(data_url: str, query: str, oauth_token: gr.OAuthToken | None):
    """
    Main function to create visualization for Gradio interface.

    Args:
        data_url: URL to the data file
        query: User's visualization query
        oauth_token: OAuth token from Gradio (None if not logged in)

    Returns:
        Tuple of (vega_lite_spec_dict, log_message, error_message)
    """
    if not data_url or not data_url.strip():
        return None, "", "Please provide a data URL"

    if not query or not query.strip():
        return None, "", "Please provide a visualization query"

    # Extract token from OAuth if user is logged in
    token = oauth_token.token if oauth_token is not None else None

    spec, error, log = create_visualization(data_url.strip(), query.strip(), token)

    if error:
        return None, log, error
    
    # fix up schema in case the LLM hallucinated (sometimes it generates wrong URLs resulting in a silent error in Altair)
    spec['$schema'] = 'https://vega.github.io/schema/vega-lite/v5.json'

    print(f"Spec is {spec}", file=sys.stderr)
    chart = alt.Chart.from_dict(spec, validate=True)
    print(f"Chart is {chart}", file=sys.stderr)
    return chart, log, ""

# Create Gradio interface
def create_app():
    with gr.Blocks(title="Visualizator") as app:
        gr.Markdown("# 📊 Visualizator")
        gr.Markdown("Generate data visualizations from URLs using natural language queries")

        # Add login button
        gr.LoginButton()

        # Dataset suggestions
        dataset_suggestions = {
            "Cars Dataset (Vega)": "https://raw.githubusercontent.com/vega/vega-datasets/master/data/cars.json",
            "Movies Dataset (Vega)": "https://raw.githubusercontent.com/vega/vega-datasets/master/data/movies.csv",
            "Iris Flowers": "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv",
            "Titanic Passengers": "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv",
            "World Population": "https://raw.githubusercontent.com/datasets/population/master/data/population.csv",
            "COVID-19 Data": "https://raw.githubusercontent.com/datasets/covid-19/master/data/countries-aggregated.csv",
            "Global Temperature": "https://raw.githubusercontent.com/datasets/global-temp/master/data/annual.csv",
            "Netflix Titles": "https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2021/2021-04-20/netflix_titles.csv",
            "Pokemon Stats": "https://raw.githubusercontent.com/lgreski/pokemonData/master/Pokemon.csv",
            "NYC Airbnb Data": "https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2022/2022-04-05/airbnb.csv"
        }

        with gr.Row():
            with gr.Column():
                dataset_dropdown = gr.Dropdown(
                    label="Select a Sample Dataset (Optional)",
                    choices=list(dataset_suggestions.keys()),
                    value=None
                )
                data_url_input = gr.Textbox(
                    label="Data URL",
                    placeholder="https://example.com/data.csv",
                    lines=1
                )
                query_input = gr.Textbox(
                    label="Visualization Query",
                    placeholder="Show me the relationship between X and Y",
                    lines=3
                )
                submit_btn = gr.Button("Generate Visualization", variant="primary")

            with gr.Column():
                output_plot = gr.Plot(label="Visualization")

        with gr.Row():
            log_output = gr.Textbox(label="Process Log", lines=10, interactive=False)
            error_output = gr.Textbox(label="Error Message", lines=5, interactive=False)

        gr.Markdown("""
        ### Examples
        - **Query**: "Show me a bar chart of sales by category"
        - **Query**: "Create a scatter plot of price vs quantity with color by region"
        - **Query**: "Display a line chart showing the trend over time"

        ### Supported Formats
        - CSV files (.csv)
        - TSV files (.tsv)
        - JSON files (.json)

        ### Note
        Sign in with your Hugging Face account to use the Inference API for generating visualizations.
        """)

        # Update URL field when dataset is selected
        def update_url_from_dropdown(dataset_name):
            if dataset_name:
                return dataset_suggestions[dataset_name]
            return ""

        dataset_dropdown.change(
            fn=update_url_from_dropdown,
            inputs=[dataset_dropdown],
            outputs=[data_url_input]
        )

        submit_btn.click(
            fn=visualize,
            inputs=[data_url_input, query_input],
            outputs=[output_plot, log_output, error_output]
        )

    return app

if __name__ == "__main__":
    app = create_app()
    app.launch()
