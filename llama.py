import os
import ast
import json
from groq import Groq

# Set the environment variable for the API key
os.environ['GROQ_API_KEY'] = "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"

# Initialize the Groq client globally
client = Groq(api_key=os.environ['GROQ_API_KEY'])

def load_json_data(filename):
    """Load data from a JSON file."""
    with open(filename, 'r') as file:
        data = json.load(file)
    return data


def process_words(game_data, strikes, isOneAway, correctGroups, previousGuesses, error):
    """
    Function to group 16 words into 4 groups of 4 words each using the Groq model with weighted instructions.
    """
    examples = """
    EXAMPLES:

    Example 1:
    Input: 
    [
        {"words": ["cat", "dog", "elephant", "lion"], "category": "Animals"},
        {"words": ["banana", "apple", "orange", "strawberry"], "category": "Fruits"},
        {"words": ["car", "bus", "train", "bicycle"], "category": "Vehicles"},
        {"words": ["laptop", "phone", "tablet", "computer"], "category": "Electronics"}
    ]
    Output: [["cat", "dog", "elephant", "lion"], ["banana", "apple", "orange", "strawberry"], ["car", "bus", "train", "bicycle"], ["laptop", "phone", "tablet", "computer"]]

    Example 2 (Overlapping Categories):
    Input:
    [
        {"words": ["bank", "river", "stream", "flow"], "category": "Water Bodies"},
        {"words": ["bank", "loan", "interest", "credit"], "category": "Finance"},
        {"words": ["apple", "banana", "pear", "grape"], "category": "Fruits"},
        {"words": ["dog", "cat", "parrot", "fish"], "category": "Pets"}
    ]
    Output: [["bank", "river", "stream", "flow"], ["bank", "loan", "interest", "credit"], ["apple", "banana", "pear", "grape"], ["dog", "cat", "parrot", "fish"]]
    """

    # Prepare the prompt with weighted instructions
    prompt = f"""
    TASK: Group the provided words into exactly 4 distinct categories with 4 words each.

    INSTRUCTIONS:
    - Prioritize grouping words that are **synonyms** or **very closely related**. Strong semantic connections should be preferred over loose associations.
    - Use advanced semantic analysis, word embeddings, and contextual similarity to form coherent groups.
    - Leverage hierarchical clustering to identify the strongest associations between words.
    - **Focus on grouping words that are most semantically similar** and avoid grouping words that only have weak or broad similarities.
    - Do not reuse words across groups; each word should only belong to one group.
    - Words within each group should be **highly related** based on linguistic patterns, latent connections, or usage in similar contexts.
    - Ignore the provided category labels; focus purely on grouping based on semantics.

    {examples}

    Here is the input data:
    {json.dumps(game_data, indent=4)}

    YOUR TASK:
    Based on the words provided, group them into exactly 4 groups of 4 words each using the techniques described above.
    Ensure that:
    - Each group is distinct, semantically coherent, and does not overlap with other groups.
    - Use your internal knowledge to determine categories based on context, meaning, and latent correlations.
    - Prioritize grouping words that are most closely related in meaning.

    Provide your answer in the following format:
    [["word1", "word2", "word3", "word4"], ["word5", "word6", "word7", "word8"], ["word9", "word10", "word11", "word12"], ["word13", "word14", "word15", "word16"]]
    """

    # Fetch response from the Groq model
    try:
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama3-70b-8192",
            temperature=0.5,  # Lower temperature for more deterministic responses
            max_tokens=512
        )
        response = chat_completion.choices[0].message.content.strip()

        # Extract and process the response
        start_index = response.find("[")
        end_index = response.rfind("]") + 1
        if start_index == -1 or end_index == -1:
            print("Invalid response format from the model.")
            return []

        response_list = response[start_index:end_index]

        # Safely evaluate the response
        guessed_groups = ast.literal_eval(response_list)

        # Validate format of the guessed groups
        if not isinstance(guessed_groups, list) or len(guessed_groups) != 4:
            print("Invalid group format.")
            return []

        for group in guessed_groups:
            if not isinstance(group, list) or len(group) != 4 or not all(isinstance(word, str) for word in group):
                print("Each group must contain exactly 4 words.")
                return []

        return guessed_groups

    except Exception as e:
        print(f"API Error or invalid response: {e}")
        return []


# Load the data from the JSON file
json_data = load_json_data('sample_data.json')

# Iterate over each game board in the JSON data
for game_data in json_data:
    strikes = 0
    isOneAway = False
    correctGroups = []
    previousGuesses = []
    error = False

    # Call the process_words function with the structured input
    guessed_groups = process_words(game_data, strikes, isOneAway, correctGroups, previousGuesses, error)

    # Print the output
    print("\nGuessed Groups:")
    for group in guessed_groups:
        print(group)
