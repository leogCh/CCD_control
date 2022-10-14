import socket
from time import sleep

def server_program():
    # get the hostname
    host = socket.gethostname()
    port = 7171  # initiate port no above 1024

    server_socket = socket.socket()  # get instance
    # look closely. The bind() function takes tuple as argument
    server_socket.bind((host, port))  # bind host address and port together

    while(True):
        # configure how many client the server can listen simultaneously
        print('start listening ...')
        server_socket.listen(1)
        conn, address = server_socket.accept()  # accept new connection
        print("Connection from: " + str(address))
        try:
            # raise Exception('Test')
            while True:
                data = input(' -> ')
                
                if len(data)==2:
                    data = '0xFF0x010x000x070x000x'+data

                conn.sendall(data.encode())

                if data == 'exit':
                    break

                recv_cmd = conn.recv(1024).decode()
                if len(recv_cmd)==0 or not recv_cmd:
                    break


                print("from connected user: " + str(recv_cmd), type(recv_cmd))
        except Exception as e:
            print(repr(e))
        conn.shutdown(2)
        conn.close()  # close the connection
        print('end connection ...')
        
        if data == 'exit':
            break
        
        sleep(5)

if __name__ == '__main__':
    server_program()