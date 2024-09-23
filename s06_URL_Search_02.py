from googlesearch import search

# Definir la consulta de búsqueda
def busqueda(query,num_results):
    # Buscar en Google y obtener las primeras URLs
    print('\n Páginas con información relacionada: ')
    results = search(query)
   #  for url in search(query): # , num_results=3
   #      print(url)
    for i, result in enumerate(results): # , num_results=3
      if i>=num_results:
          break 
      print(result)