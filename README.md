# Chess Knowledge Extractor

This project aims to extract and analyze chess-related wisdom from multiple sources. It focuses on answering questions like "What is the best way to learn chess?" based on advice from the legendary José Raúl Capablanca and popular chess resources like the Say Chess Substack.

## Features

- Downloads a chess book from the [Gutenberg Project](https://www.gutenberg.org/ebooks/33870).
- Analyzes the text using OpenAI's GPT-3.5 Turbo.
- Extracts advice and answers to common chess questions.

## Technologies Used

- Python
- OpenAI's GPT-3.5 Turbo
- Langchain

## Code Overview

### Part 1: Chess Fundamentals by Capablanca

The first segment of the script focuses on José Raúl Capablanca's book, "Chess Fundamentals." It performs the following tasks:

- Downloads the text of the book from the Gutenberg Project.
- Preprocesses the text to prepare it for analysis.
- Utilizes OpenAI's GPT-3.5 Turbo to answer the query, "What is the best way to learn chess?" based on Capablanca's insights.
- Leverages the Langchain library for text loading, preprocessing, and question-answering functionalities.

### Part 2: Say Chess Substack

The second part zeroes in on an article from the Say Chess Substack titled "10 Simple Pieces of Chess Advice." It carries out the following:

- Retrieves the article via web scraping.
- Splits the article into digestible chunks.
- Embeds each chunk using OpenAI's language model for further analysis.
- Searches for relevant chunks to answer specific questions and generates responses using Langchain's retriever and rag-prompt models.

## Output

After running the script, the program will:

- Download the book from the Gutenberg Project.
- Analyze the text to answer the question: "What is the best way to learn chess?"
- Fetch articles from [Say Chess Substack](https://saychess.substack.com/) to provide additional insights.

## Future Work

- Extend the project to analyze multiple books and articles.
- Implement a user interface for easier interaction.
- Add support for more NLP features.

## License

This project is open-source and available under the [MIT License](LICENSE).