import pandas as pd
import PyPDF2
import docx

def read_csv(file_path):
    df = pd.read_csv(file_path)
    return df.to_string(index=False)

def read_pdf(file_path):
    text = ""
    with open(file_path, "rb") as f:
        reader = PyPDF2.PdfFileReader(f)
        for page_num in range(reader.numPages):
            page = reader.getPage(page_num)
            text += page.extractText()
    return text

def read_docx(file_path):
    doc = docx.Document(file_path)
    text = ""
    for paragraph in doc.paragraphs:
        text += paragraph.text + "\n"
    return text

def main():
    model_name = "bert-base-uncased"
    tokenizer = transformers.BertTokenizer.from_pretrained(model_name)
    model = transformers.BertModel.from_pretrained(model_name)

    # Read from CSV
    csv_text = read_csv("example.csv")
    inputs_csv = tokenizer(csv_text, return_tensors="pt")
    summary_csv = model.generate(inputs_csv['input_ids'], max_length=50, num_beams=4, early_stopping=True)
    print("Document Summary (from CSV):", tokenizer.decode(summary_csv[0], skip_special_tokens=True))

    # Read from PDF
    pdf_text = read_pdf("example.pdf")
    inputs_pdf = tokenizer(pdf_text, return_tensors="pt")
    summary_pdf = model.generate(inputs_pdf['input_ids'], max_length=50, num_beams=4, early_stopping=True)
    print("Document Summary (from PDF):", tokenizer.decode(summary_pdf[0], skip_special_tokens=True))

    # Read from Word document
    docx_text = read_docx("example.docx")
    inputs_docx = tokenizer(docx_text, return_tensors="pt")
    summary_docx = model.generate(inputs_docx['input_ids'], max_length=50, num_beams=4, early_stopping=True)
    print("Document Summary (from Word document):", tokenizer.decode(summary_docx[0], skip_special_tokens=True))

if __name__ == "__main__":
    main()
