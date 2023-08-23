import logging


class Logger:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.init_logger(*args, **kwargs)
        return cls._instance

    def init_logger(self, logger_name, log_file=None):
        # Crear un nuevo logger con el nombre proporcionado
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logging.DEBUG)

        # Configurar un manejador para mostrar los mensajes en la consola
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)

        # Definir el formato de los mensajes de registro
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)

        # Asociar el manejador al logger (asegurarse de agregarlo solo una vez)
        if not self.logger.handlers:
            self.logger.addHandler(console_handler)

        # Configurar un manejador para guardar los mensajes en un archivo
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

    def get_logger(self):
        return self.logger