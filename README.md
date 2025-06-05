# Turnip

> Experiment definition, management, and execution for LLM research

Turnip provides a small framework for orchestrating LLM based
experiments.  It offers:

- Async wrappers for multiple providers (OpenAI, Together.ai and vLLM)
- A flexible `TurnProcessor` base class for running multi turn
  interactions
- Optional caching of results in a PostgreSQL database

This repository contains a minimal implementation intended to be easily
extended for research use.
