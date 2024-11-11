import json
from llama import process_words
from semantic import calculate_average_similarity


def model(words, strikes, isOneAway, correctGroups, previousGuesses, error):
    """
    The main model function that generates guesses to solve the puzzle.
    Continues until it finds the correct group or exhausts maximum attempts.
    """
    endTurn = False
    max_attempts = 10  # Maximum number of attempts to find a valid solution
    attempt = 0

    while attempt < max_attempts:
        attempt += 1
        print(f"Attempt {attempt}:")

        # Generate a new guess using process_words function
        guess = process_words(words, strikes, isOneAway, correctGroups, previousGuesses, error)

        # Check if the function failed to generate a valid guess
        if not guess:
            print("Llama failed to generate a valid guess.")
            endTurn = True
            return [], endTurn

        # Ensure the generated guess contains exactly 4 words
        if len(guess) != 4:
            print("Generated guess does not contain exactly 4 words. Trying again...")
            continue

        # Check if the guess is a repeat of previous guesses or already correct groups
        if any(set(guess) == set(prev) for prev in previousGuesses) or \
                any(set(guess) == set(correct) for correct in correctGroups):
            print("Generated guess is a repeat. Trying again...")
            continue

        print(f"New guess: {guess}")

        # Calculate the average similarity of the guessed group
        avg_similarity = calculate_average_similarity(guess)
        print(f"Average similarity of the guess: {avg_similarity:.2f}")

        similarity_threshold = 0.5

        # If the similarity score is too low, end the turn
        if avg_similarity < similarity_threshold:
            print(f"Low similarity score for group {guess}. Avg Similarity: {avg_similarity:.2f}")
            endTurn = True
            return guess, endTurn

        # Check if the guess forms a new correct group
        correctGroups.append(guess)
        print("Correct group found!")
        endTurn = False
        return guess, endTurn

    # If max attempts reached without solving
    print("Exceeded maximum attempts to generate a valid guess.")
    endTurn = True
    return [], endTurn


if __name__ == "__main__":
    # Load data from sample_data.json
    try:
        with open('../../datathon2/sample_data.json', 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print("Error: 'sample_data.json' not found. Please check the file path.")
        exit(1)

    # Iterate over each puzzle
    for puzzle_index, puzzle in enumerate(data):
        # Ensure there are at least 4 groups in the puzzle
        if len(puzzle) < 4:
            print(f"Puzzle {puzzle_index} has less than 4 groups, skipping...")
            continue

        # Extract the first 4 groups for each puzzle
        selected_groups = puzzle[:4]

        # Flatten the list of words from the selected groups into a single list of 16 words
        words = [word for group in selected_groups for word in group['words']]

        print(f"Words for puzzle {puzzle_index}: {words}")

        # Initialize game state variables
        strikes = 0
        isOneAway = False
        correctGroups = []
        previousGuesses = []
        error = "0"

        # Solve the puzzle until it gets the correct groups
        while not isOneAway:
            guess, endTurn = model(words, strikes, isOneAway, correctGroups, previousGuesses, error)
            if endTurn:
                break
            if guess:
                correctGroups.append(guess)
                previousGuesses.append(guess)

        print(f"Final Groups for puzzle {puzzle_index}: {correctGroups}")
        print("=" * 40)
