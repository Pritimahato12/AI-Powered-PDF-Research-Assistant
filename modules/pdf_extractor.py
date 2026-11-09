import pymupdf

def extract_text_from_pdf(pdf_file):

  doc = pymupdf.open(stream=pdf_file.getvalue(), filetype="pdf")

  extracted_pages = []

  for page_number, page in enumerate(doc, start=1):

    text = page.get_text("text", sort=True)

    extracted_pages.append({
      "page_number": page_number,
      "text": text
    })

  doc.close()

  return extracted_pages