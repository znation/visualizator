import altair as alt
import gradio as gr
import pandas as pd
import requests
import json
import os
import sys
import io
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
    Load data from a URL supporting CSV, TSV, JSON, and Parquet formats.

    Args:
        url: URL to the data file

    Returns:
        Tuple of (DataFrame, error_message)
    """
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        # Detect file type from URL or content-type
        if url.endswith('.parquet') or url.endswith('.parq') or 'application/vnd.apache.parquet' in response.headers.get('content-type', ''):
            # For parquet, we need to use BytesIO since it's a binary format
            df = pd.read_parquet(io.BytesIO(response.content))
        elif url.endswith('.json') or 'application/json' in response.headers.get('content-type', ''):
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
    token: Optional[str] = None,
    previous_error: Optional[str] = None,
    is_parquet: bool = False
) -> Tuple[Optional[dict], Optional[str]]:
    """
    Generate a Vega-Lite specification using an LLM.

    Args:
        query: User's visualization query
        schema: Data schema description
        data_url: URL to the data file
        token: Optional HuggingFace token
        previous_error: Optional error message from a previous attempt
        is_parquet: Whether the data source is a parquet file

    Returns:
        Tuple of (vega_lite_spec_dict, error_message)
    """
    try:
        client = get_inference_client(token)

        error_feedback = ""
        if previous_error:
            error_feedback = f"""

IMPORTANT: A previous specification attempt resulted in an error. Please fix the issue:
Error: {previous_error}

Make sure to address this error in your new specification.
"""

        # For parquet files, don't include data field in spec since we'll inject it
        data_instruction = ""
        if is_parquet:
            data_instruction = """
2. DO NOT include a "data" field in the specification - the data will be injected automatically"""
        else:
            data_instruction = f"""
2. Use the data URL provided in the "data" field with "url" property: {data_url}"""

        prompt = f"""You are a data visualization expert. Generate a valid Vega-Lite specification (JSON) based on the user's query and data schema.

User Query: {query}

Data Schema:
{schema}
{error_feedback}
Requirements:
1. Generate ONLY valid Vega-Lite JSON specification{data_instruction}
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

        # Ensure the data URL is set correctly (only for non-parquet files)
        if not is_parquet:
            if 'data' not in spec:
                spec['data'] = {}
            spec['data']['url'] = data_url

        return spec, None

    except Exception as e:
        return None, f"Error generating specification: {str(e)}\n{traceback.format_exc()}"

def extract_fields_from_spec(spec: dict) -> set:
    """
    Extract all field names referenced in a Vega-Lite specification.

    Args:
        spec: Vega-Lite specification dictionary

    Returns:
        Set of field names used in the spec
    """
    fields = set()

    def extract_from_encoding(encoding: dict):
        """Recursively extract fields from encoding object."""
        if isinstance(encoding, dict):
            for key, value in encoding.items():
                if isinstance(value, dict):
                    if 'field' in value:
                        fields.add(value['field'])
                    # Recursively check nested structures
                    extract_from_encoding(value)
                elif isinstance(value, list):
                    for item in value:
                        extract_from_encoding(item)

    # Check encoding
    if 'encoding' in spec:
        extract_from_encoding(spec['encoding'])

    # Check layers (for layered charts)
    if 'layer' in spec:
        for layer in spec['layer']:
            fields.update(extract_fields_from_spec(layer))

    # Check concatenated views
    for concat_type in ['hconcat', 'vconcat', 'concat']:
        if concat_type in spec:
            for view in spec[concat_type]:
                fields.update(extract_fields_from_spec(view))

    # Check facets
    if 'facet' in spec:
        if isinstance(spec['facet'], dict) and 'field' in spec['facet']:
            fields.add(spec['facet']['field'])
        if 'spec' in spec:
            fields.update(extract_fields_from_spec(spec['spec']))

    # Check transforms (filter, calculate, etc. may reference fields)
    if 'transform' in spec:
        for transform in spec['transform']:
            if isinstance(transform, dict):
                if 'field' in transform:
                    fields.add(transform['field'])
                if 'from' in transform and isinstance(transform['from'], dict):
                    if 'fields' in transform['from']:
                        fields.update(transform['from']['fields'])

    return fields

def map_invalid_fields_to_valid(
    invalid_fields: set,
    valid_columns: list,
    token: Optional[str] = None
) -> dict:
    """
    Use LLM to map invalid field names to the most similar valid column names.

    Args:
        invalid_fields: Set of invalid field names from the spec
        valid_columns: List of actual column names in the data
        token: Optional HuggingFace token

    Returns:
        Dictionary mapping invalid field names to suggested valid column names
    """
    try:
        client = get_inference_client(token)

        prompt = f"""You are helping fix a data visualization specification. The spec references field names that don't exist in the dataset.

Invalid field names: {', '.join(sorted(invalid_fields))}
Actual column names in dataset: {', '.join(valid_columns)}

For each invalid field name, determine the most similar matching column name from the actual columns. Consider:
- Exact matches with different casing
- Similar names (e.g., "movie_title" vs "Title")
- Abbreviated vs full names
- Underscores vs camelCase vs spaces

Return ONLY a JSON object mapping each invalid field to its best match, like this:
{{"invalid_field1": "ActualColumn1", "invalid_field2": "ActualColumn2"}}

Generate the mapping now:"""

        response = client.chat.completions.create(
            model="meta-llama/Llama-3.3-70B-Instruct",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500
        )

        mapping_text = response.choices[0].message.content.strip()

        # Clean up markdown code blocks if present
        if mapping_text.startswith('```'):
            lines = mapping_text.split('\n')
            mapping_text = '\n'.join(lines[1:-1]) if len(lines) > 2 else mapping_text
            mapping_text = mapping_text.replace('```json', '').replace('```', '').strip()

        mapping = json.loads(mapping_text)
        return mapping

    except Exception as e:
        print(f"Error mapping fields: {e}", file=sys.stderr)
        return {}

def replace_fields_in_spec(spec: dict, field_mapping: dict) -> dict:
    """
    Replace field names in a Vega-Lite spec according to the mapping.
    Also updates titles to reflect the new field names.

    Args:
        spec: Vega-Lite specification dictionary
        field_mapping: Dictionary mapping old field names to new field names

    Returns:
        Updated specification with corrected field names and titles
    """
    import copy
    spec = copy.deepcopy(spec)

    def replace_text_with_mapping(text):
        """Replace any occurrences of old field names in text with new field names.
        Handles case-insensitive matching to catch capitalized versions in titles."""
        if not isinstance(text, str):
            return text

        import re
        result = text
        for old_field, new_field in field_mapping.items():
            # Case-sensitive exact match first
            result = result.replace(old_field, new_field)

            # Also try case-insensitive replacement for titles
            # This catches cases like "horsepower" field with "Horsepower" in title
            pattern = re.compile(re.escape(old_field), re.IGNORECASE)

            # Find all matches to preserve the original capitalization pattern
            matches = list(pattern.finditer(result))
            for match in reversed(matches):  # Process from end to avoid index shifting
                matched_text = match.group()
                # If the matched text is different from old_field (different case)
                if matched_text != old_field:
                    # Try to preserve the capitalization style
                    if matched_text[0].isupper() and matched_text[1:].islower():
                        # Title case: "Horsepower" -> capitalize new_field
                        replacement = new_field.capitalize() if new_field else new_field
                    elif matched_text.isupper():
                        # All caps: "HORSEPOWER" -> uppercase new_field
                        replacement = new_field.upper() if new_field else new_field
                    else:
                        # Mixed or other case -> use new_field as-is
                        replacement = new_field

                    result = result[:match.start()] + replacement + result[match.end():]

        return result

    def replace_in_encoding(obj):
        """Recursively replace field names in encoding structures and update titles."""
        if isinstance(obj, dict):
            # Check if this is an encoding channel with a field
            if 'field' in obj and isinstance(obj['field'], str) and obj['field'] in field_mapping:
                old_field = obj['field']
                new_field = field_mapping[old_field]
                obj['field'] = new_field

                # Update title at encoding level if it exists
                if 'title' in obj and isinstance(obj['title'], str):
                    obj['title'] = replace_text_with_mapping(obj['title'])

                # Update axis titles if they exist
                if 'axis' in obj and isinstance(obj['axis'], dict):
                    if 'title' in obj['axis'] and isinstance(obj['axis']['title'], str):
                        obj['axis']['title'] = replace_text_with_mapping(obj['axis']['title'])

                # Update legend titles if they exist
                if 'legend' in obj and isinstance(obj['legend'], dict):
                    if 'title' in obj['legend'] and isinstance(obj['legend']['title'], str):
                        obj['legend']['title'] = replace_text_with_mapping(obj['legend']['title'])

                # Update header titles if they exist
                if 'header' in obj and isinstance(obj['header'], dict):
                    if 'title' in obj['header'] and isinstance(obj['header']['title'], str):
                        obj['header']['title'] = replace_text_with_mapping(obj['header']['title'])

            # Recursively process nested structures
            for key, value in obj.items():
                if key not in ['field', 'title', 'axis', 'legend', 'header']:  # Already handled above
                    replace_in_encoding(value)
        elif isinstance(obj, list):
            for item in obj:
                replace_in_encoding(item)

    # Replace in encoding
    if 'encoding' in spec:
        replace_in_encoding(spec['encoding'])

    # Update chart-level title
    if 'title' in spec:
        if isinstance(spec['title'], str):
            spec['title'] = replace_text_with_mapping(spec['title'])
        elif isinstance(spec['title'], dict) and 'text' in spec['title']:
            spec['title']['text'] = replace_text_with_mapping(spec['title']['text'])

    # Replace in layers
    if 'layer' in spec:
        for i, layer in enumerate(spec['layer']):
            spec['layer'][i] = replace_fields_in_spec(layer, field_mapping)

    # Replace in concatenated views
    for concat_type in ['hconcat', 'vconcat', 'concat']:
        if concat_type in spec:
            for i, view in enumerate(spec[concat_type]):
                spec[concat_type][i] = replace_fields_in_spec(view, field_mapping)

    # Replace in facets
    if 'facet' in spec:
        if isinstance(spec['facet'], dict) and 'field' in spec['facet']:
            if spec['facet']['field'] in field_mapping:
                old_field = spec['facet']['field']
                new_field = field_mapping[old_field]
                spec['facet']['field'] = new_field
                # Update facet title if it matches the old field
                if 'title' in spec['facet'] and isinstance(spec['facet']['title'], str):
                    spec['facet']['title'] = replace_text_with_mapping(spec['facet']['title'])
        if 'spec' in spec:
            spec['spec'] = replace_fields_in_spec(spec['spec'], field_mapping)

    # Replace in transforms
    if 'transform' in spec:
        for transform in spec['transform']:
            if isinstance(transform, dict) and 'field' in transform:
                if transform['field'] in field_mapping:
                    transform['field'] = field_mapping[transform['field']]

    return spec

def validate_spec_fields(spec: dict, df: pd.DataFrame) -> Tuple[bool, Optional[str]]:
    """
    Validate that all fields in the spec exist in the DataFrame.

    Args:
        spec: Vega-Lite specification dictionary
        df: pandas DataFrame with the actual data

    Returns:
        Tuple of (is_valid, error_message)
    """
    spec_fields = extract_fields_from_spec(spec)
    data_columns = set(df.columns)

    # Find fields that don't exist in data
    invalid_fields = spec_fields - data_columns

    if invalid_fields:
        return False, f"Spec references non-existent fields: {', '.join(sorted(invalid_fields))}. Available columns: {', '.join(sorted(data_columns))}"

    return True, None

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

    # Check if this is a parquet file (Vega-Lite doesn't support parquet URLs)
    is_parquet = data_url.endswith('.parquet') or data_url.endswith('.parq')

    # Get schema
    schema = get_data_schema(df)
    log_messages.append(f"✓ Schema extracted")

    if is_parquet:
        log_messages.append("  Note: Parquet file - data will be injected inline (max 5000 rows)")

    # Try to generate valid spec with retries
    previous_error = None
    for attempt in range(max_retries):
        log_messages.append(f"\nAttempt {attempt + 1}/{max_retries}: Generating Vega-Lite specification...")

        spec, error = generate_vega_lite_spec(query, schema, data_url, token, previous_error, is_parquet)

        if error:
            log_messages.append(f"✗ Generation failed: {error}")
            previous_error = error
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

            # Validate field names
            spec_fields = extract_fields_from_spec(spec)
            data_columns = set(df.columns)
            invalid_fields = spec_fields - data_columns

            if invalid_fields:
                log_messages.append(f"⚠ Found invalid field names: {', '.join(sorted(invalid_fields))}")
                log_messages.append("  Attempting to map to correct column names...")

                # Use LLM to map invalid fields to valid columns
                field_mapping = map_invalid_fields_to_valid(invalid_fields, list(df.columns), token)

                if field_mapping:
                    log_messages.append(f"  Suggested mapping: {field_mapping}")
                    # Apply the mapping to fix the spec
                    spec = replace_fields_in_spec(spec, field_mapping)

                    # Re-validate after fixing
                    is_valid, field_error = validate_spec_fields(spec, df)
                    if not is_valid:
                        error_msg = f"Field mapping failed: {field_error}"
                        log_messages.append(f"✗ {error_msg}")
                        previous_error = error_msg
                        if attempt < max_retries - 1:
                            log_messages.append("  Retrying from scratch...")
                        continue
                    else:
                        log_messages.append("✓ Field names corrected successfully!")
                else:
                    error_msg = "Could not generate field mapping"
                    log_messages.append(f"✗ {error_msg}")
                    previous_error = error_msg
                    if attempt < max_retries - 1:
                        log_messages.append("  Retrying from scratch...")
                    continue

            # Fix up schema in case the LLM hallucinated
            spec['$schema'] = 'https://vega.github.io/schema/vega-lite/v5.json'

            # For parquet files, inject the data as inline values
            if is_parquet:
                log_messages.append("  Injecting inline data for parquet file...")
                # Sample data if it's too large (Vega-Lite can struggle with large datasets)
                MAX_ROWS = 5000
                data_to_inject = df
                if len(df) > MAX_ROWS:
                    log_messages.append(f"  Sampling {MAX_ROWS} rows from {len(df)} total rows")
                    data_to_inject = df.sample(n=MAX_ROWS, random_state=42)

                # Prepare data for JSON serialization
                data_to_inject = data_to_inject.copy()

                # Convert datetime columns to ISO format strings
                for col in data_to_inject.columns:
                    if pd.api.types.is_datetime64_any_dtype(data_to_inject[col]):
                        # Convert to string, handling NaT values
                        data_to_inject[col] = data_to_inject[col].astype(str).replace('NaT', None)

                # Replace NaN and infinity values with None for JSON compatibility
                data_to_inject = data_to_inject.replace([float('inf'), float('-inf')], None)
                data_to_inject = data_to_inject.where(pd.notna(data_to_inject), None)

                # Convert to records format (list of dicts)
                spec['data'] = {'values': data_to_inject.to_dict('records')}
                log_messages.append(f"✓ Injected {len(data_to_inject)} rows of data")

            # Validate with Altair to catch rendering errors
            log_messages.append("  Validating specification with Altair...")
            try:
                chart = alt.Chart.from_dict(spec, validate=True)
                log_messages.append("✓ Specification generated and validated successfully!")
                return spec, None, "\n".join(log_messages)
            except Exception as altair_error:
                error_msg = f"Altair validation failed: {str(altair_error)}"
                log_messages.append(f"✗ {error_msg}")
                previous_error = error_msg
                if attempt < max_retries - 1:
                    log_messages.append("  Retrying with error feedback...")
                continue

        except Exception as e:
            error_msg = f"Validation failed: {str(e)}"
            log_messages.append(f"✗ {error_msg}")
            previous_error = error_msg
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
    # Check if user is logged in
    if oauth_token is None:
        return None, "", "Please sign in with your Hugging Face account to generate visualizations. Click the 'Sign in with Hugging Face' button above."

    if not data_url or not data_url.strip():
        return None, "", "Please provide a data URL"

    if not query or not query.strip():
        return None, "", "Please provide a visualization query"

    # Extract token from OAuth if user is logged in
    token = oauth_token.token

    spec, error, log = create_visualization(data_url.strip(), query.strip(), token)

    if error:
        return None, log, error

    # Create chart from the validated spec
    # Note: spec has already been validated in create_visualization, including schema fix
    try:
        print(f"Spec is {spec}", file=sys.stderr)
        chart = alt.Chart.from_dict(spec, validate=False)  # Already validated
        print(f"Chart is {chart}", file=sys.stderr)
        return chart, log, ""
    except Exception as e:
        # This should rarely happen since we validated in create_visualization
        error_msg = f"Failed to create chart: {str(e)}"
        print(error_msg, file=sys.stderr)
        return None, log, error_msg

# Create Gradio interface
def create_app():
    with gr.Blocks(title="Visualizator") as app:
        gr.Markdown("# 📊 Visualizator")
        gr.Markdown("Generate data visualizations from URLs using natural language queries")

        # Add login button
        gr.LoginButton()

        # Login required message (shown when not logged in)
        login_required_group = gr.Markdown("""
        ## Login Required

        Please sign in with your Hugging Face account to use Visualizator.

        This app uses the Hugging Face Inference API to generate visualizations,
        which requires authentication.

        Click the "Sign in with Hugging Face" button above to get started.
        """, visible=True)

        # Dataset suggestions
        dataset_suggestions = {
            "Cars Dataset (Vega)": "https://raw.githubusercontent.com/vega/vega-datasets/master/data/cars.json",
            "Movies Dataset (Vega)": "https://raw.githubusercontent.com/vega/vega-datasets/master/data/movies.json",
            "Iris Flowers": "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv",
            "Titanic Passengers": "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv",
            "World Population": "https://raw.githubusercontent.com/datasets/population/master/data/population.csv",
            "COVID-19 Data": "https://raw.githubusercontent.com/datasets/covid-19/master/data/countries-aggregated.csv",
            "Global Temperature": "https://raw.githubusercontent.com/datasets/global-temp/master/data/annual.csv",
            "Netflix Titles": "https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2021/2021-04-20/netflix_titles.csv",
            "Pokemon Stats": "https://raw.githubusercontent.com/lgreski/pokemonData/master/Pokemon.csv",
            "Seattle Weather": "https://raw.githubusercontent.com/vega/vega-datasets/master/data/seattle-weather.csv",
            "NYC Taxi Trips (Parquet)": "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-01.parquet",
            "Spotify Top Songs (Parquet)": "https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet",
            "Wine Quality (Parquet)": "https://huggingface.co/datasets/scikit-learn/wine-quality/resolve/main/wine-quality.parquet"
        }

        # Main UI (hidden by default, shown when logged in)
        with gr.Column(visible=False) as main_ui_group:
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

            gr.Markdown("""
            ### Examples
            - **Query**: "Show me a bar chart of sales by category"
            - **Query**: "Create a scatter plot of price vs quantity with color by region"
            - **Query**: "Display a line chart showing the trend over time"

            ### Supported Formats
            - CSV files (.csv)
            - TSV files (.tsv)
            - JSON files (.json)
            - Parquet files (.parquet)
            """)

            with gr.Row():
                log_output = gr.Textbox(label="Process Log", lines=10, interactive=False)
                error_output = gr.Textbox(label="Error Message", lines=5, interactive=False)

        # Function to check login state and toggle UI visibility
        def check_login_state(profile: gr.OAuthProfile | None):
            if profile is not None:
                # User is logged in - show main UI, hide login message
                return gr.update(visible=False), gr.update(visible=True)
            else:
                # User is not logged in - show login message, hide main UI
                return gr.update(visible=True), gr.update(visible=False)

        # Check login state on page load
        app.load(
            fn=check_login_state,
            outputs=[login_required_group, main_ui_group]
        )

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
