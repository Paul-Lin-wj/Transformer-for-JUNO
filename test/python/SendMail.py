import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from email.header import Header
import os


class SendMail:
    def __init__(self, sender_email):
        # send email
        self.smtp_server = "smtp.qq.com"
        self.smtp_port = 587
        self.sender_email = sender_email
        self.password = "ddsunwtcbgxvibij"  # the password of the sender email

    def send(self, subject, content, attachment_paths=None):
        msg = MIMEMultipart()
        msg["Subject"] = Header(subject, "utf-8")
        msg["From"] = self.sender_email
        msg["To"] = self.sender_email

        # Content
        msg.attach(MIMEText(content, "plain", "utf-8"))

        # Attachment
        if attachment_paths:
            for attachment_path in attachment_paths:
                with open(attachment_path, "rb") as attachment:
                    part = MIMEBase("application", "octet-stream")
                    part.set_payload(attachment.read())
                    encoders.encode_base64(part)
                    part.add_header(
                        "Content-Disposition",
                        f"attachment; filename= {os.path.basename(attachment_path)}",
                    )
                    msg.attach(part)

        try:
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.sender_email, self.password)
                server.sendmail(self.sender_email, [msg["To"]], msg.as_string())
                server.quit()
            print("mail sent successfully, to:", msg["To"])
        except smtplib.SMTPException as e:
            print("mail sent failed:", e)
