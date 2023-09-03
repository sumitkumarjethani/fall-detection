# Librerias para correos
import smtplib
import ssl
from email.message import EmailMessage


class NotificationManager:
    def __init__(self):
        # iniciar el serivicio de mail
        self._users = []
        self._notified_users = []
        self.email_sender = "dennistrosman@gmail.com"  # Tu mail. Debe ser el mail en el que estés corriendo este colab.
        self.email_password = (
            "obmdrdniaroysdem"  # La contraseña que salio de la autenticación
        )

    def add_user(self, user_email: str):
        self._users.append(user_email)

    def send_notification(self, user_email, result=None, image=None):
        if user_email not in self._users:
            return
        if user_email in self._notified_users:
            return

        em = EmailMessage()
        subject = "Caida detectada!!!"
        body = """Caida detectada"""
        em["From"] = self.email_sender
        em["To"] = user_email
        em["Subject"] = subject
        em.set_content(body)

        context = ssl.create_default_context()

        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as smtp:
            smtp.login(self.email_sender, self.email_password)
            smtp.sendmail(self.email_sender, user_email, em.as_string())

        self._notified_users.append(user_email)


if __name__ == "__main__":
    notificator = NotificationManager()
    # notificator.add_user("vitostamatti@gmail.com")
    notificator.send_notification("vitostamatti@gmail.com")
