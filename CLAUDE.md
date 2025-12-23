# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**visualizator** is a Python project currently in its initial setup phase. The project structure and dependencies have not yet been defined.

## Project Status

This is an ongoing project. As the project develops, this file should be updated with:
- Build and test commands
- Project architecture and structure
- Key dependencies and their purposes
- Development workflow specifics

As of now, a barebones initial implementation is runnable, roughly complete, and manually testable.

## Goals

We are creating an app called Visualizator, to be run as a Hugging Face Space. The goal is to:

* Take in a URL to a data file, could be parquet, csv, tsv, jsonl, etc. For now just csv and tsv.
* Take in a query from the user about what to visualize, like "I want to see the relationship between X, Y, and Z", or "Create a heat map of time of day vs. number of hits"
* Use an LLM to generate a vega-lite spec to match the appropriate type of chart to accommodate the user's query, and the data bindings to accommodate their data file URL's schema.
    * For this LLM, use Hugging Face Inference Providers. Since we are running inside a Space, it probably makes sense to implement authentication like this: https://huggingface.co/docs/hub/en/spaces-oauth
* Apply the vega-lite spec to the data and show the result to the user.
    * Auto-retry up to 5 times if the generated vega-lite spec doesn't work (results in an exception thrown or an error of some kind when generated spec is used with vega-lite).
