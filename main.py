#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 02:13:13 2024

@author: alexbozyck
"""

import pandas as pd
import PyPDF2
import docx
from transformers import BartTokenizer, BartForConditionalGeneration

# Function to read CSV file
def read_csv(file_path):
    df = pd.read_csv(file_path)
    return df

# Function to read PDF file
def read_pdf(file_path):
    text = ""
    with open(file_path, "rb") as f:
        reader = PyPDF2.PdfFileReader(f)
        for page_num in range(reader.numPages):
            page = reader.getPage(page_num)
            text += page.extractText()
    return text

# Function to read Word document
def read_docx(file_path):
    doc = docx.Document(file_path)
    text = ""
    for paragraph in doc.paragraphs:
        text += paragraph.text + "\n"
    return text

# Function to generate text chunks and summaries
def generate_text_chunks(text, tokenizer, model, max_chunk_length=512, max_output_length=50, num_beams=4):
    # Tokenize the input text
    tokenized_text = tokenizer(text, return_tensors="pt", max_length=max_chunk_length, truncation=True)

    # Split the input text into chunks of max_chunk_length
    input_ids = tokenized_text['input_ids']
    num_chunks = (input_ids.size(1) - 1) // max_chunk_length + 1

    generated_text = ""
    for i in range(num_chunks):
        start_idx = i * max_chunk_length
        end_idx = min((i + 1) * max_chunk_length, input_ids.size(1))
        input_chunk = input_ids[:, start_idx:end_idx]

        # Generate text for the current chunk
        summary_chunk = model.generate(input_chunk, max_length=max_output_length, num_beams=num_beams, early_stopping=True)

        # Decode the generated text and append it to the result
        generated_text += tokenizer.decode(summary_chunk[0], skip_special_tokens=True)

    return generated_text

# Function to convert survival label to text
def convert_survival_label(label):
    if label == 0:
        return "did not survive"
    elif label == 1:
        return "survived"
    else:
        return "unknown"

def main():
    model_name = "facebook/bart-large-cnn"
    tokenizer = BartTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name)

    #Testing
    # Read from CSV
    csv_data = read_csv("/Users/alexbozyck/Downloads/gender_submission.csv")
    for _, row in csv_data.iterrows():
        summary = generate_text_chunks(row["survived"], tokenizer, model)
        passenger_status = convert_survival_label(row['survived'])
        summary_text = f"The passenger {passenger_status}.\nSummary: {summary}"
        print(summary_text)

    # Read from PDF
    pdf_text = read_pdf("example.pdf")
    summary_pdf = generate_text_chunks(pdf_text, tokenizer, model)
    print("Document Summary (from PDF):", summary_pdf)

    # Read from Word document
    docx_text = read_docx("example.docx")
    summary_docx = generate_text_chunks(docx_text, tokenizer, model)
    print("Document Summary (from Word document):", summary_docx)

if __name__ == "__main__":
    main()

