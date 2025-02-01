from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from config_manager import initialize_config, save_or_update_config, load_config, add_camera_ip
import uvicorn
from TcpConnector import TcpConnector
from configParams import Parameters
# Initialize the configuration file
initialize_config()
#uvicorn configServer:app --reload --host 0.0.0.0 --port 8000
# FastAPI app
app = FastAPI()
connection = TcpConnector()
params=Parameters()

# Pydantic model for updating configurations
class ConfigUpdateRequest(BaseModel):
    section: str
    key: str
    value: str


class CameraIPRequest(BaseModel):
    ip: str
    username:str
    password:str
    isnotrstp:bool


class Relay(BaseModel):
    isconnect:bool
    
    
class EmailClass(BaseModel):
    plateNumber:str
    eDate:str
    eTime:str
    
    

    

@app.get("/config")
def get_config():
    """Retrieve the entire configuration."""
    config = load_config()
    return {"status": "success", "config": config}


@app.post("/config")
def update_config(request: ConfigUpdateRequest):
    """Update or add a configuration key-value pair."""
    try:
        # Save or update the configuration
        save_or_update_config(request.section, request.key, request.value)
        return {"status": "success", "message": f"{request.key} updated to {request.value} in section {request.section}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))










@app.post("/cameras")
def add_camera(request: CameraIPRequest):
    """Add a new camera IP."""
    print(request.isnotrstp)
    try:
        if not request.isnotrstp:
            print("is not rt")
            rtspIp=f"{request.ip}"
        else:
            rtspIp=f"rtsp://{request.username}:{request.password}@{request.ip}:554/mainstream"
        new_camera = add_camera_ip(rtspIp)
        return {"status": "success", "new_camera": new_camera}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@app.post("/iprelay")
def connectrelay(request:Relay,ip,port):
    
    
    connection.setConnectionProperties(f"{ip}", int(port))
    if(request.isconnect):
         #on
        if(connection.connectToServer()):
              
                return{"massage":"connect"}
        else:
                
                return{"massage":"problem connect"}
    else:
        if(connection.closeConnection()):
            return{"massage":"disconnect"}
        else:
            return{"massage":"problem dissconnect"}

        
@app.get("/iprelay")
def onOff(onOff,relay):
    #on
    if(onOff=="true"):
        if(relay==int(relay)):
            data = b'\x00\x01\x02\x03\x04\x05\x06\x07\x08\x01\x00\x01\x01\x01\x00\x00\x00\x00\x00\x03\x01\x01\x00' #relay 1
            if (connection.sendPacket(bData=data)):
                return {'massage': connection.receivePacket(23, 2)}
            else:
                return {"massage":f"problem : {connection.receivePacket(23, 2)}"}
        else:
            data = b'\x00\x01\x02\x03\x04\x05\x06\x07\x08\x01\x00\x01\x01\x01\x00\x00\x00\x00\x00\x03\x02\x01\x00' #relay 2
            if (connection.sendPacket(bData=data)):
                return {'massage': connection.receivePacket(23, 2)}
            else:
                return {"massage":f"problem : {connection.receivePacket(23, 2)}"}
            ########################
            #of
    else:
        if(relay==int(relay)):
            data = b'\x00\x01\x02\x03\x04\x05\x06\x07\x08\x01\x00\x01\x01\x01\x00\x00\x00\x00\x00\x03\x01\x00\x00' #relay 1
            if (connection.sendPacket(bData=data)):
                return {'massage': connection.receivePacket(23, 2)}
            else:
                return {"massage":f"problem : {connection.receivePacket(23, 2)}"}
        else:
            data = b'\x00\x01\x02\x03\x04\x05\x06\x07\x08\x01\x00\x01\x01\x01\x00\x00\x00\x00\x00\x03\x02\x00\x00' #relay 2
            if (connection.sendPacket(bData=data)):
                return {'massage': connection.receivePacket(23, 2)}
            else:
                return {"massage":f"problem : {connection.receivePacket(23, 2)}"}
            
        
                
                

def emailHandler(email,plateNumber,edate,etime):
    


    import smtplib
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart

    # Gmail SMTP server settings
    SMTP_SERVER = "smtp.gmail.com"
    SMTP_PORT = 587

    # Sender's email credentials
    SENDER_EMAIL = "amnafarin4@gmail.com"
    SENDER_PASSWORD = "vioz mxiw nedg rybh"

    # Recipient email
    RECIPIENT_EMAIL = email

    # Email content
    subject = f"{edate} شناسایی پلاک در تاریخ "
    body = f""" 
    پلاک:{plateNumber}
    تاریخ:{edate}
    زمان:{etime}
     """

    # Create the email message
    msg = MIMEMultipart()
    msg["From"] = SENDER_EMAIL
    msg["To"] = RECIPIENT_EMAIL
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    try:
        # Connect to the SMTP server
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()  # Upgrade the connection to secure
        server.login(SENDER_EMAIL, SENDER_PASSWORD)  # Log in to the server

        # Send the email
        server.sendmail(SENDER_EMAIL, RECIPIENT_EMAIL, msg.as_string())
        print("Email sent successfully!")

    except Exception as e:
        print(f"Failed to send email: {e}")

    finally:
        server.quit()  # Close the connection

    #vioz mxiw nedg rybh

            
    
@app.post('/email')
def sendEmail(request:EmailClass,email):
    try:
        
        emailHandler(email,request.plateNumber,request.eDate,request.eTime)
        return {"massage":"email send Apporved"}
    except:
        return{"massage":"Failed"}
    

    
    


        
    








if __name__ == "__main__":
    uvicorn.run("configServer:app", host="127.0.0.1", port=int(params.serverport), log_level="info")