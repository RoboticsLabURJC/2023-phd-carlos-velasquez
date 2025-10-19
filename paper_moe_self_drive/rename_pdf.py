import os
from PyPDF2 import PdfReader

carpeta = "./"

for archivo in os.listdir(carpeta):
    if archivo.lower().endswith(".pdf"):
        ruta = os.path.join(carpeta, archivo)
        reader = PdfReader(ruta)
        titulo = reader.metadata.title
        
        if titulo:
            # Limpia caracteres no válidos para nombres de archivo
            nuevo_nombre = "".join(c for c in titulo if c.isalnum() or c in " _-").strip() + ".pdf"
            nueva_ruta = os.path.join(carpeta, nuevo_nombre)
            os.rename(ruta, nueva_ruta)
            print(f"Renombrado: {archivo} → {nuevo_nombre}")
        else:
            print(f"⚠️ No se encontró título en {archivo}")

