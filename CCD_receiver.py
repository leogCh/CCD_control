import socket
import serial
from time import sleep

CCD_command_list = ['0xFF0x010x000x070x000x01',
                    '0xFF0x010x000x070x000x02',
                    '0xFF0x010x000x070x000x03',
                    '0xFF0x010x000x070x000x04',
                    '0xFF0x010x000x070x000x05',
                    '0xFF0x010x000x070x000x06',
                    '0xFF0x010x000x070x000x07',
                    '0xFF0x010x000x070x000x08',
                    '0xFF0x010x000x070x000x09',
                    '0xFF0x010x000x200x000x00',
                    '0xFF0x010x000x400x000x00']

try:
    ser = serial.Serial('COM5', 9600)
    # ser = serial.Serial('COM5', 9600)
    ser.rs485_mode = serial.rs485.RS485Settings(rts_level_for_tx=True, rts_level_for_rx=False, loopback=False,
                                                delay_before_tx=None, delay_before_rx=None,)
    print('ser.isOpen():', ser.isOpen())
except Exception as e:
    print(repr(e))

def checksum(s_list):
    check_s = 0
    s_out = ""
    # 計算Byte2至Byte6 相加
    for i in range(1,len(s_list)):
        check_s += int(s_list[i],16)
        s_out = s_out+str(s_list[i])+" "
    #print(check_s)

    # 找餘數
    check_s = check_s % 255
    check_s = format(check_s,'x')
    try:
        if int(check_s,16)<16:
            check_s = '0'+check_s
    except:
        print(check_s)
    s_out = s_out + check_s
    print(s_out)
    return s_out

def CCD_control(s_in):
    print('s_in:', s_in)
    commad = bytearray.fromhex(checksum(s_in.split('0x')))
    print('CCD_control:', commad)
    ser.write(commad)
    

def client_program():
    host = socket.gethostname()  # as both code is running on same pc
    port = 7171  # socket server port number

    while(True):
        while(True):
            client_socket = socket.socket()  # instantiate
            try:
                print('connecting to server ... ')
                client_socket.connect((host, port))  # connect to the server
                break
            except Exception as e:
                print(repr(e))
            client_socket.close()
            del client_socket
            sleep(3)
            
        print('connected')

        try:
            while(True):
                recv_cmd = client_socket.recv(1024).decode()  # receive response
                print('Received from server: ' + recv_cmd)  # show in terminal

                if len(recv_cmd)==0 or not recv_cmd:
                    break

                if recv_cmd in CCD_command_list:
                    CCD_control(recv_cmd)
                
                client_socket.send('done'.encode())  # send message
        except Exception as e:
            print(repr(e))
        client_socket.close()


if __name__ == '__main__':
    try:
        client_program()
    except Exception as e:
        print(repr(e))
    try:
        ser.close()
    except:
        pass