import csv, genanki

model = genanki.Model(
  1607392319,
  'Basic JP-CN',
  fields=[
    {'name': 'Grammar'},
    {'name': 'Meaning'},
    {'name': 'ExampleJP'},
    {'name': 'ExampleCN'},
  ],
  templates=[
    {
      'name': 'Card 1',
      'qfmt': '{{Grammar}}<br><br>{{ExampleJP}}',
      'afmt': '{{FrontSide}}<hr id="answer">{{Meaning}}<br>{{ExampleCN}}',
    },
  ])

deck = genanki.Deck(
  2059400110,
  'JLPT N1 Grammar 60')

with open(r"C:\Users\YAO\Downloads\jlpt_n1_grammar60.csv", newline='', encoding='utf-8-sig') as f:
    for g,m,j,c in csv.reader(f):
        note = genanki.Note(model=model, fields=[g,m,j,c])
        deck.add_note(note)

genanki.Package(deck).write_to_file('jlpt_n1_grammar60.apkg')
