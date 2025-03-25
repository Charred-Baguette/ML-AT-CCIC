print('importing')

try:
    import pandas as pd
    from datasets import load_dataset
    from PIL import Image
    import numpy as np
    import time
    import openai
    import os
    import re
    from tqdm import tqdm
    from collections import OrderedDict
    from sklearn.model_selection import train_test_split
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import accuracy_score, confusion_matrix
    import matplotlib.pyplot as plt
    from sklearn.naive_bayes import GaussianNB
    from imblearn.over_sampling import SMOTE
    smote = SMOTE()
    print('Import complete')
except ImportError as e:
    print(f"Error importing module: {e}")

# Set OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")
import getpass
if openai.api_key is None:
    print(' ValueError("API key is missing. Please set the OPENAI_API_KEY environment variable.")')
    openai.api_key = getpass.getpass("Enter your API key: ")

# Initialize the OpenAI client
client = openai
try:
    print('Loading Dataset...')
    prompts = pd.read_csv('df_subset.csv')
    df=pd.read_csv('WordList.csv')
except FileNotFoundError as e:
    print(f"Error loading dataset: {e}")



def classify_word(word):
    # Define a system message and user prompt to classify each word
    prompt = f"Classify the word '{word}' into one of the following categories: 'illegal', 'explicit', 'violent', 'innocent'. Provide just the category as the output. Ensure one of these categories is no matter what stated."

    # Call the OpenAI API with the prompt
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that classifies words."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=10
        )
        
        # Parse the response and extract the classification
        classification = response.choices[0].message.content.strip()
        if 'illegal' in classification:
            classification = 'illegal'
        elif 'explicit' in classification:
            classification = 'explicit'
        elif 'violent' in classification:
            classification = 'violent'
        elif 'innocent' in classification:
            classification = 'innocent'
        else:
            response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that classifies words."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=30
        )
        # Map the response to a unique value
        category_mapping = {
            "illegal": 1,
            "explicit": 2,
            "violent": 3,
            "innocent": 0
        }
        
        # Return the mapped value
        return category_mapping.get(classification, -1)  # Return -1 for unknown categories
    
    except Exception as e:
        print(f"Error classifying word '{word}': {e}")
        return -1
    
def numberify(word):
    # Define the user prompt
    prompt = f"HERE IS THE WORD: {word}"

    try:
        # First classification: 4-digit number
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": """Assign the word a number based on the following criteria.
    The number will be 4 digits. The first 4 digits determine how 'bad' the word may be.
    - Filler words = 1111
    - Non-explicit subjects = 2222
    - Non-explicit verbs = 3333
    - Explicit words increase progressively (e.g., 5555, 6666, 7777, 8888)
    - Use 9999 only for extreme cases.
    Respond with just the number!."""},
                {"role": "user", "content": prompt}
            ],
            max_tokens=10
        )
        classification = response.choices[0].message.content.strip()

        # Second classification: 2-digit number based on implication
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": """Assign the word a number from 00 to 99 based on severity.
    - Use a balanced scale with 99 reserved for extreme words.
    Respond with just the number!."""},
                {"role": "user", "content": prompt}
            ],
            max_tokens=10
        )
        secondary = response.choices[0].message.content.strip()

        # Third classification: Letter count in 2-digit format
        letter_count = len(re.findall(r'[a-zA-Z]', word))
        formatted_count = str(letter_count).zfill(2)  # Ensures 2-digit format

        # Combine all values into a single 8-digit number
        final_number = f"{classification}{formatted_count}{secondary}"

        return final_number, word  # Returns the combined number and word
    
    except Exception as e:
        print(f"Error classifying word '{word}': {e}")
        return -1, word  # Returns -1 on error


print("Training models...")
X = prompts['Numbers'].apply(lambda x: sum(x) if isinstance(x, list) else 0).values.reshape(-1, 1)  # If 'Numbers' is a list, sum them as a feature
y = prompts['allowed']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
nb = GaussianNB()
nb.fit(X_train_resampled, y_train_resampled)
knn = KNeighborsClassifier(n_neighbors=2)
knn.fit(X_train, y_train)
print("Model training complete.")

while True:
    custom_prompt = input("Enter your prompt (or 'exit' to quit): ")
    if custom_prompt.lower() == 'exit':
        break
    elif custom_prompt.strip() == '':
        print("Prompt cannot be empty.")
        continue
    word_classifications = df.set_index('Word')['Classification'].to_dict()
    rows = []
    Classprompt = custom_prompt
    Numprompt = custom_prompt

            # Clean up non-alphabetical characters (remove special characters)
    Classprompt = re.sub(r'[^a-zA-Z\s]', '', Classprompt)  
    Numprompt = re.sub(r'[^a-zA-Z\s]', '', Numprompt)

    # Process words and classify them
    for word in re.findall(r'\b\w+\b', Classprompt):  # Loop through each word
        if word not in word_classifications:
            classification_value = classify_word(word)  # Classify word (you'll have to define this)
            numeric_value, interpreted_word = numberify(word)  # Interpret numeric value (define this too)
            word_classifications[word] = {'Classification': classification_value, 
                                        'Number': numeric_value, 'NumericInterpretedWord': interpreted_word}
        else:
            classification_value = word_classifications[word]['Classification']
            numeric_value = word_classifications[word]['Number']
        
        # Replace the word in the prompts with the classification and numeric value
        Classprompt = re.sub(r'\b' + re.escape(word) + r'\b', str(classification_value), Classprompt)
        Numprompt = re.sub(r'\b' + re.escape(word) + r'\b', str(numeric_value), Numprompt)

        # Output the processed prompt with classifications and numeric values
    print("Processed Classprompt:", Classprompt)
    print("Processed Numprompt:", Numprompt)
    print("Pre Checks")
    Class_3_count = Classprompt.count('3')
    Class_2_count = Classprompt.count('2')
    Class_1_count = Classprompt.count('1')

    Class_Sum_count = Class_3_count + Class_2_count + Class_1_count

    # Apply filtering conditions
    Guess_allowed = 1  # Default assumption (if conditions are not violated)
    if Class_3_count > 4 or Class_2_count > 4 or Class_1_count > 4 or Class_Sum_count > 4:
        Guess_allowed = 0

    # Extract numbers from Numprompt
    Numbers = list(map(int, re.findall(r'\d+', Numprompt)))
    Average_Number = sum(Numbers)/len(Numbers) if Numbers else 0

    if Average_Number > 50000000:
        Guess_allowed = 0
    elif any(num > 60000000 for num in Numbers):
        Guess_allowed = 0

    # Print the result for the custom prompt
    print(f"Filter check allowed: {Guess_allowed}")

    print("running model...")
    def preprocess_custom_prompt(prompt):
        # Extract numbers from the prompt
        numbers = re.findall(r'\d+', prompt)
        
        # Convert the numbers to integers
        numbers = list(map(int, numbers))
        
        # Sum the numbers (or any other feature engineering you want to use)
        sum_of_numbers = sum(numbers) if numbers else 0
        
        # Return the feature as an array (reshape to match the model input)
        return np.array([[sum_of_numbers]])
    X_custom = preprocess_custom_prompt(custom_prompt)
    custom_prediction = knn.predict(X_custom)
    
    print(f"KNN Prediction for the custom prompt: {custom_prediction[0]}")
    NBPredict = nb.predict(X_custom)
    print(f"Naive Bayes Prediction for the custom prompt: {NBPredict[0]}")

    

    