from openpyxl import load_workbook
import json
from io import BytesIO
import re

class tablaGo:
    def __init__(self, input_excel_file, prompts, sheets):
        #Path al archivo .xlsx
        self.input_excel_file = load_workbook(input_excel_file)

        #Dirección al archivo json con la info
        if isinstance(prompts, str):  # If prompts is a file path (string)
            with open(prompts, 'r', encoding='utf-8') as f:
                self.prompts = json.load(f)
        elif hasattr(prompts, 'read'):  # If prompts is an UploadedFile (e.g., from Django)
            file_content = prompts.read().decode('utf-8')  # Read the file and decode bytes to string
            self.prompts = json.loads(file_content)  # Load the string content as JSON
        else:
            self.prompts = prompts  # Direct JSON object

        self.sheets = sheets
        self.questions = [self.prompts[sheets[0]]["preguntas"], self.prompts[sheets[1]]["preguntas"]]
        self.casillas = [self.prompts[sheets[0]]["casillas"], self.prompts[sheets[1]]["casillas"]]

        self.file = load_workbook(input_excel_file)

        self.err = self.prompts["err_404"]

    def merge(self, sheet, cell1, cell2):
        if sheet not in self.sheets:
            self.file.create_sheet(title=sheet)
        self.file[sheet].merge_cells(f'{cell1}:{cell2}')
        
        
    # Rellena las casillas de la Ficha GO
    def update_excel(self, sheet, cell, answer):
        self.input_excel_file[sheet][cell] = answer


    def modify(self, sheet, cell, txt):
        self.file[sheet][cell] = txt

    def save_file(self):
        output = BytesIO()
        self.file.save(output)
        output.seek(0)  # Rewind the buffer
        return output

    def contains_any_phrases(self, input_string):
        for phrase in self.err:
            #print(f"{input_string} | {phrase}")
            if phrase in input_string:
                return True
        return False
    
    def contains_any_phrases_aux(self, input_string):
        for phrase in self.err:
            #print(f"{input_string} | {phrase}")
            if phrase in input_string:
                return True
        return False

    def contains_any_phrases_l(self, input_strings, check):
        res = [1]*len(input_strings)
        for i, response in enumerate(input_strings):
            if (check[i] == 1 and self.contains_any_phrases(response)) or response == "N SUPERADA":
                res[i] = 0

        print(res)
        return res

    def next_q(self, sheet, n, lista):
        res = [""]*lista
        dic = self.questions[sheet]

        for i, val in enumerate(dic.values()):
            print(val)
            if lista[i] == 1:
                if n <= len(val):
                    res.append(val[n])
                else:
                    res.append(False)
                    print(str(n) + " es mayor que: " + str(val[n]))
            else: 
                res.append("")

        print(res)
        return res
    
    def normalizar_monto(self, monto):

        match = re.search(r'(\d{1,3}(?:\.\d{3})*,\d{2})', monto)
        
        if match:
            # Reemplaza los puntos (.) que separan los miles y cambia la coma (,) por un punto (.)
            monto = match.group(1).replace('.', '').replace(',', '.')
            return float(monto)
        else:
            return monto
        
    #Ambas páginas comparten apartados y es lógico que estos contengan la misma info.
    #Es útil tener dos respuestas distintas que puedan aportar mas información, pero en caso
    #de que una no tenga respuesta, esta se duplica de la página que si lo hizo.
    def consistencia(self):
        print("TEST CONSISTENCIA")
        keys = {"Titulo":["B4","B4"], "Organismo":["B5","B5"], "Presu":["B6","B7"], "Sol_tec":["B12","B16"], "Equipo":["B15","B18"]}
        sheet1 = self.sheets[0]  # Objeto Worksheet de openpyxl
        sheet2 = self.sheets[1]

        for key, value in keys.items():
            cell1 = self.file[sheet1][value[0]]  # Por ejemplo, 'B4'
            cell2 = self.file[sheet2][value[0]]  # Por ejemplo, 'B4'

            if cell1.value == "NO SE PUDO ENCONTRAR RESPUESTA":
                if cell2.value == "NO SE PUDO ENCONTRAR RESPUESTA":
                    print("A MAL B MAL")
                    self.modify(sheet1, value[0], "Error de concistencia")
                    self.modify(sheet2, value[1], "Error de concistencia")
                
                else:
                    self.modify(sheet1, value[0], cell2.value)
            
            elif cell2.value == "NO SE PUDO ENCONTRAR RESPUESTA":
                self.modify(sheet2, value[1], cell1.value)



def main():    

    input_xlsx = "ficha.xlsx"
    prompts = "prompts.json"
    tabla = tablaGo(input_xlsx, prompts)
    tabla.update_excel("PRUEBA", 'A1', "Hola")
    tabla.save("res.xlsx")

if __name__ == "__main__":
    main()