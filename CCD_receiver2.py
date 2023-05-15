import socket
from time import sleep
from detect_video import detect
from return_result import return_result

model_path = '/tflite/test_area/CCD_control/tflite/mobilenetV2_model.tflite'
label_path = '/tflite/test_area/CCD_control/tflite/labelmap.txt'
conf_th = 0.5
camera_no = 1

def client_program():
    host = socket.gethostname()  # as both code is running on same pc
    port = 7172  # socket server port number

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
                
                frame = None
                if recv_cmd[:3] == 'dd1':
                    r1,r2,r3,frame = detect(model_path, label_path, conf_th, camera_no=camera_no,
                           save_result_img=False, keyboard_input=True)
                
                if recv_cmd[:3] == 'dd2':
                    r1,r2,r3,frame = detect(model_path, label_path, conf_th, camera_no=camera_no,
                           save_result_img=True, keyboard_input=False)
                    return_result(recv_cmd[4:], r1,r2,r3,frame)
                
                if recv_cmd[:4] == 'exit':
                    break

                client_socket.sendall('done'.encode())  # send message
        except Exception as e:
            print(repr(e))
        client_socket.close()
        if recv_cmd == 'exit':
            break


if __name__ == '__main__':
    try:
        client_program()
    except Exception as e:
        print(repr(e))
