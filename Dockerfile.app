FROM huggingface/neural_forge_ai:latest
CMD uvicorn neural_forge_ai.app:app --host 0.0.0.0 --port 7860 --reload --workers 4
