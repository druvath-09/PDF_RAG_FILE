import pdfplumber

def extract_pdf_tables(pdf_path, page_number):
    """
    Extract tables from a specific page in a PDF file.

    Args:
        pdf_path (str): Path to the PDF file.
        page_number (int): Page number (0-indexed) to extract tables from.

    Returns:
        tables (list): List of extracted tables.
    """
    try:
        with pdfplumber.open(pdf_path) as pdf:
            # Check if the page number is valid
            if page_number < len(pdf.pages):
                print(f"Extracting tables from Page {page_number + 1}...")
                tables = pdf.pages[page_number].extract_tables()
                return tables
            else:
                print("Invalid page number. Page does not exist.")
                return []
    except Exception as e:
        print(f"Error occurred: {e}")
        return []

if __name__ == "__main__":
    # Input PDF file path
    pdf_path = "example.pdf"  # Update with the path to your PDF
    page_number = 5 # Page 6 (0-indexed, so use 5)

    # Extract tables
    tables = extract_pdf_tables(pdf_path, page_number)

    # Display extracted tables
    if tables:
        print("\nExtracted Tables:")
        for i, table in enumerate(tables):
            print(f"\nTable {i + 1}:")
            for row in table:
                print(row)
    else:
        print("No tables found on the specified page.")

