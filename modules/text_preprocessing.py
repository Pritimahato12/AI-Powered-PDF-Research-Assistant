import re


def clean_text(text):

    # Remove special tokens
    text = re.sub(
        r"<(?:EOS|PAD|pad|UNK|unk)>",
        "",
        text,
        flags=re.IGNORECASE
    )

    # Remove extra spaces
    text = re.sub(r"[ \t]+", " ", text)

    # Remove repeated consecutive lines
    lines = text.splitlines()

    cleaned_lines = []
    previous_line = None

    for line in lines:

        line = line.strip()

        if not line:
            continue

        if line != previous_line:
            cleaned_lines.append(line)

        previous_line = line

    text = "\n".join(cleaned_lines)

    return text.strip()


def preprocess_pages(extracted_pages):

  processed_pages = []

  for page in extracted_pages:

    cleaned_text = clean_text(page["text"])

    if cleaned_text:

      processed_pages.append({
          "page_number": page["page_number"],
          "text": cleaned_text
      })

  return processed_pages


from langchain_text_splitters import RecursiveCharacterTextSplitter


def create_chunks(processed_pages):

  splitter = RecursiveCharacterTextSplitter(
      chunk_size=1000,
      chunk_overlap=200,
      separators=["\n\n", "\n", " ", ""]
  )

  all_chunks = []

  for page in processed_pages:

    page_chunks = splitter.split_text(page["text"])

    for chunk_number, chunk_text in enumerate(page_chunks, start=1):

      all_chunks.append({

          "page_number": page["page_number"],

          "chunk_number": chunk_number,

          "text": chunk_text

      })

  return all_chunks