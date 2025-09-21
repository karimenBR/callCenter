# Call Center AI Classifier
### it's an intelligent chatbot that doesn’t only reply, but dynamically routes each request to the right NLP engine, integrated inside a full MLOps architecture.
## Steps
### * Receives customer tickets via a REST API (or simple interface).
### * Cleans sensitive data (PII scrub).
### * Analyzes the request (language, length, complexity).
### * Decides which model to use:
### * TF–IDF + SVM for simple and fast cases.
### * Transformer for complex or multilingual cases.
### * Returns the prediction (category, confidence) with an explanation of the choice.
### * Exposes Prometheus metrics for monitoring.
