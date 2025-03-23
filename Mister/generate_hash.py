import hashlib

def hash_dato(dato):
    # Convertir el dato a una cadena JSON y luego calcular el hash SHA-256
    dato_str = str(dato)
    hash_object = hashlib.sha256(dato_str.encode())
    return hash_object.hexdigest()