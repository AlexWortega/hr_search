#some stuff gonna happen here
import os
import PyPDF2
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util
from telegram import Update
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext

# Load the model
model = SentenceTransformer('AlexWortega/hh_search')

# Read the job descriptions from a CSV file
df = pd.read_csv('job_descriptions.csv', delimiter=';')
job_descriptions = list(df['description'])

# Encode the job descriptions
job_embeddings = model.encode(job_descriptions, convert_to_tensor=True)

def pdf_to_string(file_path):
    """
    Extract text from a PDF file.
    """
    pdf_file_obj = open(file_path, 'rb')
    pdf_reader = PyPDF2.PdfReader(pdf_file_obj)
    num_pages = len(pdf_reader.pages)
    full_text = ""

    for page in range(num_pages):
        page_obj = pdf_reader.pages[page]
        full_text += page_obj.extract_text()

    pdf_file_obj.close()
    return full_text

def find_matching_job_names(query_text, top_k=5):
    """
    Find the top matching job names based on the query text.
    """
    query_embedding = model.encode(query_text, convert_to_tensor=True).to(model.device)
    cos_scores = util.cos_sim(query_embedding, job_embeddings)[0]
    top_results = torch.topk(cos_scores, k=top_k)
    result = [str(name)[:100] for name in df['names'][top_results.indices].tolist()]
    return result

def start(update: Update, context: CallbackContext) -> None:
    """
    Handle the /start command in the Telegram bot.
    """
    update.message.reply_text('Hello! Please send me a PDF file.')

def handle_document(update: Update, context: CallbackContext) -> None:
    """
    Handle PDF documents sent to the Telegram bot.
    """
    file = context.bot.getFile(update.message.document.file_id)
    fname = os.path.join('downloads', '{}.{}'.format(file.file_id, 'pdf'))
    file.download(custom_path=fname)  # save file
    result = find_matching_job_names(pdf_to_string(fname))
    os.remove(fname)  # remove file after processing
    update.message.reply_text('\n'.join(result))

def main() -> None:
    """
    Start the Telegram bot.
    """
    updater = Updater(Config.API_KEY, use_context=True)

    dispatcher = updater.dispatcher

    dispatcher.add_handler(CommandHandler("start", start))
    dispatcher.add_handler(MessageHandler(Filters.document.pdf, handle_document))

    updater.start_polling()

    updater.idle()

if __name__ == '__main__':
    main()
