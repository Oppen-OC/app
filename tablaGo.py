from openpyxl import Workbook, load_workbook

'''
# Function to create a new Excel file with specified text in a cell
def create_excel_with_text(output_excel_file, sheet_name, cell, text):
    # Create a new Workbook
    wb = Workbook()
    ws = wb.active
    ws.title = sheet_name
    
    # Insert text into the specified cell
    ws[cell] = text 
    for i in range(1,30):
        a = 'C' + str(i)
        ws[a] = "A"
    ws[cell] = text
    
    # Save the workbook
    wb.save(output_excel_file[:-5] + "TesteoFeo.xlsx")
    print(f'Excel file created and text inserted: {output_excel_file}')
'''

# Rellena las casillas de la Ficha GO
def update_excel_with_text(input_excel_file, sheet_name):
    # Load an existing Workbook
    wb = load_workbook(input_excel_file)
    if sheet_name in wb.sheetnames:
        ws = wb[sheet_name]
    else:
        ws = wb.create_sheet(sheet_name)
    
    # Inserta texto en las casillas importantes
    for i in [3, 4, 5, 6, 9, 10, 12, 13, 15, 17, 19, 21, 23, 24, 26, 27, 29, 31, 33, 35]:
        cellB = 'B' + str(i)
        cellA = 'A' + str(i)
        ws[cellB] = "TESTING"
    
    # Save the workbook
    wb.save(input_excel_file[:-5] +"Res.xlsx")
    print(f'Text inserted into {input_excel_file}')

# Example usage:
output_excel_file = 'fichaTest.xlsx'
sheet_name = 'A1 Ficha GO'


# Create a new Excel file and insert text into a cell
update_excel_with_text(output_excel_file, sheet_name)
#create_excel_with_text(output_excel_file, sheet_name, cell, text)

# Update an existing Excel file and insert text
