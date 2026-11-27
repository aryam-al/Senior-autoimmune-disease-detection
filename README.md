# Senior-autoimmune-disease-detection
ImmunoAI

A web app that classifies autoimmune diseases from lab results using two AI models: XGBoost (machine learning) and a fine-tuned Qwen2.5-7B (deep learning).

Research prototype, not a medical device. Always ask a doctor.

Features:
Upload one or more lab PDFs. Values are read and combined automatically.
Missing values are allowed and never guessed.
Choose XGBoost, Qwen, or both to compare them on the same report.
Explainable AI: SHAP shows the lab values behind each XGBoost result.
Chatbot powered by Qwen, in English or Arabic.
Google sign-in, strong passwords, and history of past analyses.
