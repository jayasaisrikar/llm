import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from subprocess import run
import json

import warnings
warnings.filterwarnings("ignore")

# Load environment variables
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=google_api_key)

def get_pdf_text(pdf_docs):
    """Extract text from PDF files."""
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
            else:
                # If no text is found, assume it's an image-based PDF
                return None
    return text

def get_text_chunks(text):
    """Split text into manageable chunks."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=15000, chunk_overlap=500)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    """Generate embeddings and create vector store."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain(prompt_template):
    """Initialize conversational chain with a prompt template."""
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "history", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question, chat_history):
    """Handle user input and generate response."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    
    # Prompt template for conversational chain
    prompt_template = """
    Properly read the entire pdf and recognise the following: 
    Invoice Number, Vendor (from), Client (to), Due Date, Invoice Issue Date, and Total Amount of bill. 
    Here are the conditions you need to check: if and only if in the provided document there are more than one invoice, then each input must be given separately, e.g., if there are n invoices, there should be n different required fields.
    If there is no Due Date or any Keyword that means Due Date then return Due Date as Empty.
    \n\n
    Context:\n {context}?\n
    History:\n {history}\n
    Question:\n {question}\n

    Answer:
    """
    chain = get_conversational_chain(prompt_template)

    response = chain({"input_documents": docs, "context": "", "history": chat_history, "question": user_question},
                     return_only_outputs=True)

    # Extract required fields from response
    output = response['output_text']
    output_lines = output.split('\n')
    
    invoice_details = []

    for line in output_lines:
        if "Invoice Number" in line:
            invoice_details.append({"Invoice Number": line.split(":")[1].strip()})
        elif "Vendor" in line:
            invoice_details[-1]["Vendor (from)"] = line.split(":")[1].strip()
        elif "Client" in line:
            invoice_details[-1]["Client (to)"] = line.split(":")[1].strip()
        elif "Due Date" in line:
            invoice_details[-1]["Due Date"] = line.split(":")[1].strip()
        elif "Invoice Issue Date" in line:
            invoice_details[-1]["Invoice Issue Date"] = line.split(":")[1].strip()
        elif "Total Amount" in line:
            invoice_details[-1]["Total Amount"] = line.split(":")[1].strip()
    
    return invoice_details

def main():
    """Main function to run the app."""
    # Replace with your own PDF files
    pdf_docs = ["sample1.pdf"]
    raw_text = get_pdf_text(pdf_docs)
    
    if raw_text is None:
        # If no text could be extracted, assume the PDF is image-based
        for pdf in pdf_docs:
            result = run(["python", "run.py", pdf], capture_output=True, text=True)
            # Print stdout to inspect the output
            print("Subprocess stdout:", result.stdout)
            if result.stdout:
                try:
                    output_data = json.loads(result.stdout)
                    print("Invoice Number:", output_data.get("Invoice Number"))
                    print("Vendor (from):", output_data.get("Vendor (from)"))
                    print("Client (to):", output_data.get("Client (to)"))
                    print("Due Date:", output_data.get("Due Date"))
                    print("Invoice Issue Date:", output_data.get("Invoice Issue Date"))
                    print("Total Amount:", output_data.get("Total Amount"))
                except json.JSONDecodeError as e:
                    print("JSONDecodeError:", e)
            else:
                print("No output from run.py")
        return

    text_chunks = get_text_chunks(raw_text)
    get_vector_store(text_chunks)

    # Replace with your own question and chat history
    user_question = "Follow the prompt template"
    chat_history = []
    invoices = user_input(user_question, chat_history)
    for i, invoice in enumerate(invoices, start=1):
        print(f"Details for Invoice {i}:")
        print("Invoice Number:", invoice.get("Invoice Number"))
        print("Vendor (from):", invoice.get("Vendor (from)"))
        print("Client (to):", invoice.get("Client (to)"))
        print("Due Date:", invoice.get("Due Date"))
        print("Invoice Issue Date:", invoice.get("Invoice Issue Date"))
        print("Total Amount:", invoice.get("Total Amount"))

    # Update chat history
    #chat_history.append({"user_question": user_question, "answer": response})

if __name__ == "__main__":
    main()

'''
Improvements that can be made: 

- The code should run multiple times and the results must be compared for each variable. 
Where the value that is repeated max times is considered as correct value.

- More variables can be added **ADD THE LIST HERE**

- Modify the prompt to scrape tabled details and save to variables


'''
