# Detoxigram

Detoxigram is a tool designed to analyze and reduce toxicity in different contexts, combining the strengths of BERT classifiers and generative Language Models (LLMs) to promote healthier online interactions.

## Table of Contents
- [Overview](#overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Overview
Inspired by literature (Zhixue et al., 2021; Ousidhoum et al., 2021; Fortuna et al., 2021), Detoxigram identifies and classifies toxic content using a five-level toxicity scale. It leverages BERT models for initial classification and generative LLMs for detailed analysis and detoxification suggestions.

## Project Structure
- `dataset`: Contains datasets for training and evaluation.
- `detoxigram_bot`: The implementation of the bot for toxicity analysis.
- `classifiers`: Contains the classes and files used for text classification
- `script-download-channels`: Scripts to download data from various channels.
- `requirements.txt`: Python dependencies for the project.

## Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/LIA-DiTella/Detoxigram.git
    cd Detoxigram
    ```
2. Create a virtual environment and activate it:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```
3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Contributing
1. Fork the repository.
2. Create a new branch (`git checkout -b feature/YourFeature`).
3. Commit your changes (`git commit -am 'Add a new feature'`).
4. Push to the branch (`git push origin feature/YourFeature`).
5. Create a new Pull Request.
