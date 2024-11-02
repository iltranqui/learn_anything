import csv
import genanki

# Function to generate anki cards from a CSV file with English, Italian on the front and Spanish on the back
def create_anki_deck(csv_file, deck_name="My Language Deck"):
    # Load CSV
    with open(csv_file, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=';')
        data = list(reader)
    
    model_english = genanki.Model(
        1607392320,
        'Language Model',
        fields=[
            {'name': 'Italian'},
            {'name': 'English'},
        ],
        templates=[
            {
                'name': 'Card 1',
                'qfmt': 'Italiano: {{Italian}}',  # Display Italian on the front
                'afmt': '{{FrontSide}}<hr id="answer">English: {{English}}',  # Display English on the back
            },
        ])

    # Create a deck
    deck = genanki.Deck(
        2059400110,
        deck_name)

    # Add cards to the deck
    for row in data:
        if len(row) >= 2:  # Ensure there are three columns (English, Italian, Spanish)
            italian = row[0]  # "Ciao"
            english = row[1]  # "Hello"
            note = genanki.Note(
                model=model_english, # or model_spanish
                fields=[italian, english]) # or [english, italian, spanish]
            deck.add_note(note)

    # Save the deck to a .apkg file
    genanki.Package(deck).write_to_file(f"{deck_name}.apkg")
    print(f"Anki deck '{deck_name}.apkg' created successfully!")

# Example of usage
create_anki_deck('english_words.csv', 'Inglese_Semplice') # or Spanish_words.csv
