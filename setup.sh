#!/bin/bash

# Install Python dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('vader_lexicon', quiet=True); nltk.download('stopwords', quiet=True); nltk.download('averaged_perceptron_tagger', quiet=True)"

# Create necessary directories
mkdir -p .streamlit

# Create config.toml for Streamlit
cat > .streamlit/config.toml << EOF
[server]
headless = true
enableCORS = false
enableXsrfProtection = false
port = \$PORT

[browser]
gatherUsageStats = false
EOF

echo "✅ Setup completed successfully!"
