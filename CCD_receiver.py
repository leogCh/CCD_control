import socket
import serial
import serial.rs485

try:
    ser = serial.Serial('COM5', 9600)
    ser.rs485_mode = serial.rs485.RS485Settings(rts_level_for_tx=True, rts_level_for_rx=False, loopback=False,
                                                delay_before_tx=None, delay_before_rx=None,)
    print('ser.isOpen():', ser.isOpen())
except Exception as e:
    print(repr(e))

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

def server_program():
    # get the hostname
    host = socket.gethostname()
    port = 7171  # initiate port no above 1024

    server_socket = socket.socket()  # get instance
    # look closely. The bind() function takes tuple as argument
    server_socket.bind((host, port))  # bind host address and port together

    while(True):
        # configure how many client the server can listen simultaneously
        print('start listen ...')
        server_socket.listen(1)
        conn, address = server_socket.accept()  # accept new connection
        print("Connection from: " + str(address))
        while True:
            # receive data stream. it won't accept data packet greater than 1024 bytes
            recv_cmd = conn.recv(1024).decode()

            if len(recv_cmd)==0 or not recv_cmd:
                # if recv_cmd is not received break
                break

            print("from connected user: " + str(recv_cmd), type(recv_cmd))
            # 0xFF0x010x000x070x000x01

            if recv_cmd in CCD_command_list:
                CCD_control(recv_cmd)

            data = input(' -> ')
            conn.send(data.encode())  # send data to the client

        conn.close()  # close the connection
        print('end connection ...')

def checksum(s_list):
    check_s = 0
    s_out = "FF "
    # 計算Byte2至Byte6 相加
    for i in range(1,len(s_list)):
        check_s += int(s_list[i],16)
        s_out = s_out+str(s_list[i])+" "
    #print(check_s)

    # 找餘數
    check_s = check_s % 255
    check_s = format(check_s,'x')
    #print(check_s)

    try:
        if int(check_s,16)<16:
            check_s = '0'+check_s
    except:
        print(check_s)
    s_out = s_out + check_s
    print(s_out)
    return s_out

def CCD_control(s_in):
    # s_in = ['FF','01','00','07','00','02']
    commad = bytearray.fromhex(checksum(s_in.split('0x')))
    print('CCD_control:', commad)
    # ser.write(commad)

if __name__ == '__main__':
    server_program()