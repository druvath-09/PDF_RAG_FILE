import pdfplumber
import pandas as pd

def extract_tables_all_pages(pdf_path):
    """
    Extract tables from all pages of a PDF.

    Args:
        pdf_path (str): Path to the PDF file.

    Returns:
        all_tables (list): A list containing tables from all pages.
    """
    all_tables = []

    with pdfplumber.open(pdf_path) as pdf:
        # Loop through all pages
        for page_number, page in enumerate(pdf.pages):
            print(f"\nExtracting tables from page {page_number + 1}...")
            tables = page.extract_tables()

            # Save tables with page number info
            for table in tables:
                all_tables.append({"page": page_number + 1, "table": table})

                # Print for preview
                print(f"Table on Page {page_number + 1}:")
                for row in table:
                    print(row)

    return all_tables

# Extract tables from all pages
pdf_path = "example.pdf"
all_tables = extract_tables_all_pages(pdf_path)

# Save all tables as DataFrames (optional)
for i, table_data in enumerate(all_tables):
    page_number = table_data["page"]
    table = table_data["table"]
    df = pd.DataFrame(table[1:], columns=table[0])  # First row as headers
    print(f"\nTable {i + 1} on Page {page_number} as DataFrame:")
    print(df)
    # Save extracted tables to CSV
    output_file = f"table_page_{page_number + 1}.csv"
    df.to_csv(output_file, index=False)
    print(f"Table saved to {output_file}")


