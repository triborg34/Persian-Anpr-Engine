import socket
import select


class TcpConnector:
    def __init__(self , ip= None, port = None):
        self.__ip = ip
        self.__port = port
    __isConnected = False
    __socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    def setConnectionProperties(self , ip , port):
        self.__ip = ip
        self.__port = port

    def connectToServer(self):
        try:
            self.__socket.connect((self.__ip , self.__port))
            self.__isConnected = True
            return True
        except Exception as e:
            print ("Error in connecting to server: " , e)
            return False
    def closeConnection(self):
        try:
            self.__socket.close()
            self.__socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.__isConnected = False
            return True
        except Exception as e:
            print ("Error in closing connection")
            return False

    def sendPacket(self, bData):
        if (not self.__isConnected): return False
        try:
            self.__socket.send(bData)
            return True
        except Exception as e:
            print ("Error in sending data" ,e )
            return False


    def receivePacket(self , length , timeout): # timeout in seconds
        result = select.select([self.__socket], [], [], timeout)
        if result[0]:
            data = self.__socket.recv(length)
            return data
        else:
            return None