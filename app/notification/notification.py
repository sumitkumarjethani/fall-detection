# Librerias para correos
import smtplib
import ssl
import os
import datetime
import cv2

from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage


EMAIL_SENDER = os.environ.get("EMAIL_SENDER", None)
EMAIL_PASSWORD = os.environ.get("EMAIL_PASSWORD", None)


class NotificationManager:
    def __init__(self):
        # verify sender and password are set
        if EMAIL_SENDER is None or EMAIL_PASSWORD is None:
            print("Email sender or password variables are not configured")
        
        self._notified_users = []
        self.email_sender = EMAIL_SENDER
        self.email_password = EMAIL_PASSWORD
    
    def _get_timestamp(self):
        return datetime.datetime.now()
    
    def _check_send_notification(self, user_email):
        if self.email_sender is None or self.email_password is None:
            return False
         
        for user_data in self._notified_users:
            if user_data["email"] == user_email:
                last_notification = user_data["last_notification"]
                current_time = self._get_timestamp()
                time_difference = current_time - last_notification
                
                if time_difference.total_seconds() > 180:
                    return True
                else:
                    return False
        return True
    
    def _update_notified_users(self, user_email):
        for user_data in self._notified_users:
            if user_data["email"] == user_email:
                user_data["last_notification"] = self._get_timestamp()
                return
        self._notified_users.append({
            "email": user_email,
            "last_notification": self._get_timestamp() 
        })

    def send_notification(self, user_email, result=None, image=None):
        if self._check_send_notification(user_email):
            msg = MIMEMultipart()
            msg["From"] = self.email_sender
            msg["To"] = user_email
            msg["Subject"] = "ALERTA: CAIDA DETECTADA"

            msg.attach(MIMEText("Caida detectada. Llame a 112 de manera urgente"))
            image_encoded = cv2.imencode(".jpg", image)[1].tobytes()
            msg.attach(MIMEImage(image_encoded))
            try:
                context = ssl.create_default_context()
                with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as smtp:
                    smtp.login(self.email_sender, self.email_password)
                    smtp.sendmail(self.email_sender, user_email, msg.as_string())
                self._update_notified_users(user_email)
            except Exception as e:
                print(e)
            