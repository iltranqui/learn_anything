import csv
import genanki

def create_anki_deck_from_csv(csv_file, deck_name, output_file):
    # Create a deck with a unique ID
    deck_id = 1234567890  # Choose a random but unique ID
    my_deck = genanki.Deck(
        deck_id,
        deck_name
    )

    # Define a simple model (question-answer template)
    my_model_spanish = genanki.Model(
        1091735104,  # Model ID, must be unique
        'Simple Model',
        fields=[
            {'name': 'Question'},
            {'name': 'Answer'},
        ],
        templates=[
            {
                'name': 'Card 1',
                'qfmt': '{{Question}}',
                'afmt': '{{FrontSide}}<hr id="answer">{{Answer}}',
            },
        ]
    )

    # Define a simple model (question-answer template)
    my_model_english = genanki.Model(
        1091735104,  # Model ID, must be unique
        'Simple Model',
        fields=[
            {'name': 'Question'},
            {'name': 'Answer'},
        ],
        templates=[
            {
                'name': 'Card 1',
                'qfmt': '{{Question}}',
                'afmt': '{{FrontSide}}<hr id="answer">{{Answer}}',
            },
        ]
    )

    # Read the CSV file
    with open(csv_file, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            # Ensure the row has two columns (Front, Back)
            if len(row) == 2:
                front = row[0]
                back = row[1]
                
                # Create a note (card) for each row
                my_note = genanki.Note(
                    model=my_model,
                    fields=[front, back]
                )
                
                # Add the note (card) to the deck
                my_deck.add_note(my_note)

    # Generate the .apkg file (Anki deck package)
    genanki.Package(my_deck).write_to_file(output_file)
    print(f"Anki deck '{deck_name}' created as '{output_file}'")

# Example usage:
# create_anki_deck_from_csv('flashcards.csv', 'My New Deck', 'my_deck.apkg')
