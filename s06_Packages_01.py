import nltk

# Descargar los recursos necesarios
print("\nDescargado Paquetes ....")
nltk.download('punkt', download_dir='D:\\DonBrandon\\Programing\\IA\\Bot\\nltk_data')  # Ajusta la ruta
nltk.download('punkt_tab', download_dir='D:\\DonBrandon\\Programing\\IA\\Bot\\nltk_data') 
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')

print("Paquetes descargados correctamente.")
print(nltk.data.path)