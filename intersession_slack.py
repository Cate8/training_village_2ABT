# 1. un bucle for que lee todos los archivos subject.csv
# para cada df generar un plot y guardarlo en un directorio /home/pi/village_projects/cate_task/data/sessions/ itear aqui para cada RATON 
# guardarlos en pdf con fecha
# enviar todos los pdfs creados por slack 


from village.settings import settings

data_directory = settings.get("DATA_DIRECTORY")


print(data_directory)