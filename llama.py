#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
import os
import ast
from groq import Groq

# Set the environment variable for the API key
os.environ['GROQ_API_KEY'] = "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"

# Initialize the Groq client globally
client = Groq(api_key=os.environ['GROQ_API_KEY'])

def process_words(words, strikes, isOneAway, correctGroups, previousGuesses, error):
    # Prepare a set of words that have already been used
    used_words = set(word for group in correctGroups for word in group)

    # Filter out used words to get available words
    available_words = [word for word in words if word not in used_words]

    # Define examples to include in the prompt
    examples = """
    Example Group 1: ["cat", "dog", "elephant", "lion"] -> Animals
    Example Group 2: ["banana", "apple", "orange", "strawberry"] -> Fruits
    Example Group 3: ["car", "bus", "train", "bicycle"] -> Vehicles
    Example Group 4: ["laptop", "phone", "tablet", "computer"] -> Electronics
    """

    # Generate the prompt
    prompt = f"""
    TASK: Group the following words into related groups of exactly 4 words each.

    INSTRUCTIONS:
    1. The groups should be based on strong semantic similarity (e.g., categories like animals, fruits, vehicles).
    2. Each group must contain exactly 4 words.
    3. Do not reuse words across groups.
    4. Avoid using previous incorrect guesses.

    Examples of Correct Groupings:
    {examples}

    Current Words: {available_words}
    Previous Incorrect Guesses: {previousGuesses}
    Correct Groups Identified So Far: {correctGroups}

    YOUR TASK:
    Based on the words provided, suggest a new group of 4 words that fits a clear category.
    Ensure that:
    - The group is not a repeat of previous guesses.
    - The group uses words that haven't been correctly grouped yet.
    - The group is semantically coherent.

    Provide your answer in the following format:
    ["word1", "word2", "word3", "word4"]
    """

    # Fetch response from the Groq model
    try:
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama3-70b-8192",
            temperature=0.7,
            max_tokens=512
        )
        response = chat_completion.choices[0].message.content

        # Extract the list from the response
        start_index = response.find("[")
        end_index = response.find("]", start_index) + 1
        if start_index == -1 or end_index == -1:
            print("Invalid response format from the model.")
            return []

        response_list = response[start_index:end_index]
        guessed_group = ast.literal_eval(response_list)

        # Ensure the guessed group is a list of strings
        if not isinstance(guessed_group, list) or not all(isinstance(word, str) for word in guessed_group):
            print("Invalid guess format.")
            return []

        return guessed_group

    except Exception as e:
        print(f"API Error or invalid response: {e}")
        # Return an empty list if there's an error
        return []