import csv
import genanki

# Function to generate anki cards from a CSV file with English, Italian on the front and Spanish on the back
def create_anki_deck(csv_file, deck_name="My Language Deck"):
    # Load CSV
    with open(csv_file, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=';')
        data = list(reader)
    
    # Define the model (field layout)
    model = genanki.Model(
        1607392319,
        'Language Model',
        fields=[
            {'name': 'English'},
            {'name': 'Italian'},
            {'name': 'Spanish'},
        ],
        templates=[
            {
                'name': 'Card 1',
                'qfmt': 'English: {{English}}<br>Italian: {{Italian}}',
                'afmt': '{{FrontSide}}<hr id="answer">Spanish: {{Spanish}}',
            },
        ])

    # Create a deck
    deck = genanki.Deck(
        2059400110,
        deck_name)

    # Add cards to the deck
    for row in data:
        if len(row) >= 3:  # Ensure there are three columns (English, Italian, Spanish)
            english = row[0]  # "Hello"
            italian = row[1]  # "Ciao"
            spanish = row[2]  # "Hola"
            note = genanki.Note(
                model=model,
                fields=[english, italian, spanish])
            deck.add_note(note)

    # Save the deck to a .apkg file
    genanki.Package(deck).write_to_file(f"{deck_name}.apkg")
    print(f"Anki deck '{deck_name}.apkg' created successfully!")

# Example of usage
create_anki_deck('Spanish_words.csv', 'Spanish_Enrico')
