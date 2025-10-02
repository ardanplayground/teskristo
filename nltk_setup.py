import nltk
import os

def setup_nltk():
    print("Setting up NLTK data...")
    
    # Create NLTK data directory in a writable location
    nltk_data_path = os.path.join(os.getcwd(), 'nltk_data')
    os.makedirs(nltk_data_path, exist_ok=True)
    
    # Add to NLTK path
    nltk.data.path.append(nltk_data_path)
    
    # Download required packages
    packages = [
        'punkt',
        'vader_lexicon', 
        'stopwords',
        'averaged_perceptron_tagger'
    ]
    
    for package in packages:
        try:
            nltk.download(package, download_dir=nltk_data_path, quiet=True)
            print(f"✅ {package} downloaded")
        except Exception as e:
            print(f"❌ Error downloading {package}: {e}")
    
    print("NLTK setup completed!")

if __name__ == "__main__":
    setup_nltk()
