import smtplib
import logging
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os
import numpy as np
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('vehicle_alerting')

# Load environment variables
load_dotenv()

class AlertManager:
    def __init__(self):
        # Email settings
        self.smtp_server = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
        self.smtp_port = int(os.getenv('SMTP_PORT', 587))
        self.email_sender = os.getenv('EMAIL_SENDER')
        self.email_password = os.getenv('EMAIL_PASSWORD')
        self.email_recipient = os.getenv('EMAIL_RECIPIENT', self.email_sender)
        
        # SMS settings
        self.recipient_phone = os.getenv('RECIPIENT_PHONE')
        self.sms_gateway = os.getenv('SMS_GATEWAY', '')
        
        # Log configuration
        logger.info("AlertManager initialized with the following settings:")
        logger.info(f"SMTP Server: {self.smtp_server}:{self.smtp_port}")
        logger.info(f"Email Sender: {self.email_sender}")
        logger.info(f"Email Recipient: {self.email_recipient}")
        logger.info(f"SMS Gateway: {self.sms_gateway if self.sms_gateway else 'Not configured'}")
    
    def send_email_alert(self, subject, message, recipient=None):
        """Send an email alert with enhanced error handling"""
        # Check configuration
        if not all([self.email_sender, self.email_password]):
            error_msg = "Email credentials not fully configured. "
            error_msg += f"Sender: {'Set' if self.email_sender else 'Missing'}, "
            error_msg += f"Password: {'Set' if self.email_password else 'Missing'}"
            logger.error(error_msg)
            return False
            
        recipient = recipient or self.email_recipient
        if not recipient:
            logger.error("No recipient specified for email alert.")
            return False
            
        logger.info(f"Attempting to send email to {recipient} via {self.smtp_server}:{self.smtp_port}")
        
        try:
            # Create message
            msg = MIMEMultipart()
            msg['From'] = self.email_sender
            msg['To'] = recipient
            msg['Subject'] = f"🚨 {subject}"
            msg.attach(MIMEText(message, 'plain'))
            
            # Connect to SMTP server
            logger.debug(f"Connecting to {self.smtp_server}:{self.smtp_port}")
            server = smtplib.SMTP(self.smtp_server, self.smtp_port, timeout=10)
            
            try:
                # Start TLS encryption
                logger.debug("Starting TLS...")
                server.starttls()
                
                # Login
                logger.debug(f"Logging in as {self.email_sender}")
                server.login(self.email_sender, self.email_password)
                
                # Send email
                logger.debug("Sending email...")
                server.send_message(msg)
                
                logger.info(f"✅ Email alert sent successfully to {recipient}")
                return True
                
            except smtplib.SMTPAuthenticationError as e:
                logger.error(f"SMTP Authentication Error: {e}")
                logger.error("Please check your email and password. If using Gmail, you may need to use an App Password.")
                return False
                
            except smtplib.SMTPException as e:
                logger.error(f"SMTP Error: {e}")
                return False
                
            except Exception as e:
                logger.error(f"Unexpected error while sending email: {e}")
                return False
                
            finally:
                try:
                    server.quit()
                except:
                    pass
                    
        except Exception as e:
            logger.error(f"Failed to connect to SMTP server: {e}")
            logger.error(f"Server: {self.smtp_server}, Port: {self.smtp_port}")
            return False
    
    def send_sms_alert(self, message):
        """Send an SMS alert using available methods"""
        if not self.recipient_phone:
            logger.warning("Recipient phone number not configured. Skipping SMS alert.")
            return False
            
        message = f"🚨 Vehicle Alert: {message}"
        
        # Method 1: Email to SMS Gateway
        if self.sms_gateway:
            try:
                sms_gateway_email = f"{self.recipient_phone}@{self.sms_gateway}"
                logger.info(f"Sending SMS via email gateway to {sms_gateway_email}")
                return self.send_email_alert("Vehicle Alert", message, sms_gateway_email)
            except Exception as e:
                logger.error(f"Failed to send SMS via email gateway: {e}")
        
        logger.warning("No SMS method configured. Please set up SMS_GATEWAY in .env")
        return False

def get_contribution_scores(window, model, scaler):
    """
    Calculate the contribution of each sensor to the anomaly score.
    
    Args:
        window: Input window of sensor data (1D array)
        model: Trained autoencoder model
        scaler: Fitted scaler used for normalization
        
    Returns:
        dict: Dictionary mapping sensor names to their contribution scores
    """
    if window is None or model is None or scaler is None:
        return {}
        
    # Ensure window is 2D (samples, features)
    window = np.array(window).reshape(1, -1) if len(window.shape) == 1 else window
    
    # Get the model's prediction
    window_scaled = scaler.transform(window)
    window_scaled = window_scaled.reshape(1, window.shape[0], window.shape[1])
    reconstruction = model.predict(window_scaled, verbose=0)
    
    # Calculate error for each feature
    mse_per_feature = np.mean(np.square(window_scaled - reconstruction), axis=1)[0]
    
    # Map to feature names
    feature_names = ['rpm', 'coolant_temp', 'intake_pressure', 'maf', 'throttle_pos', 
                    'engine_load', 'vehicle_speed', 'intake_air_temp', 'voltage']
    
    return {feature: float(score) for feature, score in zip(feature_names, mse_per_feature)}
