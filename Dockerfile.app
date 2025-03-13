FROM huggingface/neural-forge-ai:latest
CMD uvicorn neural-forge-ai.app:app --host 0.0.0.0 --port 7860 --reload --workers 4
