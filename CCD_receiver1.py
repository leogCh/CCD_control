import socket
import serial
from time import sleep


# c_host = socket.gethostname()
c_host = "10.137.125.46"
c_port = 7171  # sender server port number

s_host = socket.gethostname()
s_port = 7172  # receiver2 port

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
# define preset      0xFF0x010x000x030x000x??

try:
    ser = serial.Serial('/dev/ttyUSB1', 9600)
    # ser = serial.Serial('COM5', 9600)
    # ser.rs485_mode = serial.rs485.RS485Settings(rts_level_for_tx=True, rts_level_for_rx=False, loopback=False,
    #                                             delay_before_tx=None, delay_before_rx=None,)
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
    print(check_s)

    # 找餘數
    check_s = check_s % 255
    check_s = format(check_s,'x')
    
    if int(check_s,16)<16:
        check_s = '0'+check_s

    s_out = s_out + check_s
    print(s_out)
    return s_out

def CCD_control(s_in):
    print('s_in:', s_in)
    commad = bytearray.fromhex(checksum(s_in.split('0x')))
    print('CCD_control:', commad)
    try:
        ser.write(commad)
    except:
        print('warning: ser is not open')

def client_program():
    while(True):

        server_socket = socket.socket()  # get instance
        # look closely. The bind() function takes tuple as argument
        server_socket.bind((s_host, s_port))  # bind host address and port together

        # configure how many client the server can listen simultaneously
        print('start listening ...')
        server_socket.listen(1)
        recv2_conn, address = server_socket.accept()  # accept new connection
        print("Connection from: " + str(address))
        print('receiver2 connected')

        while(True):
            client_socket = socket.socket()  # instantiate
            try:
                print('connecting to server ... ')
                client_socket.connect((c_host, c_port))  # connect to the server
                break
            except Exception as e:
                print(repr(e))
            client_socket.close()
            del client_socket
            sleep(3)
        print('server connected')

        try:
            while(True):
                recv_cmd = client_socket.recv(1024).decode()  # receive response

                recv_cmd = recv_cmd.replace('\r', '')

                if '\n' in recv_cmd:
                    recv_cmd = recv_cmd.split('\n')[1]

                recv_cmd = recv_cmd.replace('X', 'x')

                if '0xFF' in recv_cmd:
                    while(recv_cmd[:4]!='0xFF'):
                        recv_cmd = recv_cmd[1:]
                    # recv_cmd = recv_cmd[:24]

                print('Received from server: ' + recv_cmd)  # show in terminal

                if len(recv_cmd)==0 or not recv_cmd:
                    break

                # if recv_cmd in CCD_command_list or recv_cmd[:20]=='0xFF0x010x000x030x00':
                if recv_cmd[:4]=='0xFF':
                    CCD_control(recv_cmd[:24])
                    sleep(5)
                    recv2_conn.sendall(f'dd2 {recv_cmd}'.encode())
                    recv_cmd = recv2_conn.recv(1024).decode()
                    print("from receiver2: " + str(recv_cmd), type(recv_cmd))
                    CCD_control('0xFF0x010x000x070x000x05')

                if recv_cmd in ['dd1', 'dd2']:
                    recv2_conn.sendall(recv_cmd.encode())
                    recv_cmd = recv2_conn.recv(1024).decode()
                    print("from receiver2: " + str(recv_cmd), type(recv_cmd))

                if recv_cmd == 'exit':
                    recv2_conn.sendall(recv_cmd.encode())
                    break

                # client_socket.sendall('done'.encode())  # send message
        except Exception as e:
            print(repr(e))
        client_socket.close()
        recv2_conn.shutdown(2)
        recv2_conn.close()
        if recv_cmd == 'exit':
            break


if __name__ == '__main__':
    
    client_program()
    #try:
    #    client_program()
    #except Exception as e:
    #    print(repr(e))
    try:
        ser.close()
    except:
        pass
