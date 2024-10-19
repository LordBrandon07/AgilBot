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

# Definir la consulta de búsqueda
def busquedaTelegram(query, num_results, message, bot):
    # Buscar en Google y obtener las primeras URLs
    results = search(query)
    
    # Si hay resultados, envía los enlaces al chat de Telegram
    if results:
        bot.reply_to(message, 'Páginas con información relacionada:')
        for i, result in enumerate(results):
            if i >= num_results:
                break
            bot.reply_to(message, result)
    else:
        bot.reply_to(message, 'No se encontraron resultados para tu búsqueda.')
