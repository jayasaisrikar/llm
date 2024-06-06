import google.generativeai as genai
import os
from dotenv import load_dotenv, find_dotenv
from pdf2image import convert_from_path
import sys
import json

def ask_and_get_answer(prompt, img):
    model = genai.GenerativeModel("gemini-pro-vision")
    response = model.generate_content([prompt, img])
    return response.text

def extract_first_image_from_pdf(pdf_path):
    try:
        # Convert PDF to a list of images
        images = convert_from_path(pdf_path)
        if images:
            return images[0]
        else:
            return None
    except Exception as e:
        return None

def main():
    load_dotenv(find_dotenv(), override=True)
    genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))

    # Get PDF file path from command-line arguments
    if len(sys.argv) < 2:
        return
    pdf_path = sys.argv[1]
    if not os.path.isfile(pdf_path):
        return

    # Extract first image from PDF
    pil_image = extract_first_image_from_pdf(pdf_path)
    if not pil_image:
        return

    # Get user prompt
    prompt = '''
Retrieve the following details from the invoice, regardless of how they are labeled in the document:

- Invoice Number (may be labeled as "Invoice Number", "Inv Number", "Invoice No", etc.)
- Vendor (sender) (may be labeled as "Vendor", "Sender", "Seller", "From", "Supplier", etc.)
- Client (recipient) (may be labeled as "Client", "Recipient", "To", "Customer", etc.)
- Due Date (may be labeled as "Due Date", "Payment Due", "Pay By", etc.)
- Invoice Issue Date (may be labeled as "Invoice Date", "Date of Issue", "Billing Date", etc.)
- Total Amount (may be labeled as "Total Amount", "Amount Due", "Total", "Bill Amount", etc.)
It's crucial to accurately interpret the scraped text, especially in calculating the due date. If the invoice lacks a specified due date, consider any terms or clauses regarding payment deadlines, such as "payment due within n number of days," and calculate the due date accordingly. 
Ensuring precise calculations is essential for meeting the client's needs effectively.

'''

    # Generate answer
    answer = ask_and_get_answer(prompt, pil_image)

    # Extract required fields from response
    output_lines = answer.split('\n')
    
    invoice_number = None
    vendor = None
    client = None
    due_date = None
    invoice_issue_date = None
    total_amount = None

    for line in output_lines:
        if "Invoice Number" in line:
            invoice_number = line.split(":")[1].strip()
        elif "Vendor" in line:
            vendor = line.split(":")[1].strip()
        elif "Client" in line:
            client = line.split(":")[1].strip()
        elif "Due Date" in line:
            due_date = line.split(":")[1].strip()
        elif "Invoice Issue Date" in line:
            invoice_issue_date = line.split(":")[1].strip()
        elif "Total Amount" in line:
            total_amount = line.split(":")[1].strip()
    
    # Print the extracted data in JSON format for main.py to capture
    output_data = {
        "Invoice Number": invoice_number,
        "Vendor (from)": vendor,
        "Client (to)": client,
        "Due Date": due_date,
        "Invoice Issue Date": invoice_issue_date,
        "Total Amount": total_amount,
    }
    print(json.dumps(output_data))

if __name__ == '__main__':
    main()
