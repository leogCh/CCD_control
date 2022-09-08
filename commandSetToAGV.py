import cv2 
import numpy
from imutils.video import VideoStream
import serial
import serial.rs485
import cv2
import time

## 計算 checksum 的 function
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
    # print(s_out)
    return s_out

# 連接 RS 485 端
ser = serial.Serial('COM5', 9600)
ser.rs485_mode = serial.rs485.RS485Settings(rts_level_for_tx=True, rts_level_for_rx=False, loopback=False,
                                            delay_before_tx=None, delay_before_rx=None,)

print(ser.isOpen())

# #PELCO D protocol commands for left right and stop action
# thestring = bytearray.fromhex('FF 01 00 04 3F FF 44')
# thestring2 = bytearray.fromhex('FF 01 00 02 20 3F 62')
# stop = bytearray.fromhex('FF 01 00 00 00 00 01')
# print(thestring)
# ser.write(thestring2)
# time.sleep(0.5)
# ser.write(stop)
# time.sleep(1)
# ser.write(thestring)
# time.sleep(3)
# ser.write(stop)
# ser.close()



# stop = bytearray.fromhex('FF 01 00 00 00 00 01')
# set_preset_99 = ['FF','01','00','03','00','40']
# commad = checksum(set_preset_99)
# print(commad)
# preset_99= bytearray.fromhex(commad)
# ser.write(preset_99)
# time.sleep(10)

# s_in = ['FF','01','00','07','00','02']
# commad = checksum(s_in)
# print(commad)
# gotopreset_2 = bytearray.fromhex(commad)
# ser.write(gotopreset_2)
# time.sleep(5)

# s_in = ['FF','01','00','07','00','03']
# commad = checksum(s_in)
# print(commad)
# gotopreset_2 = bytearray.fromhex(commad)
# ser.write(gotopreset_2)
# time.sleep(5)

# go_preset_99 = ['FF','01','00','07','00','40']
# commad = checksum(go_preset_99)
# print(commad)
# gopreset_99= bytearray.fromhex(commad)
# ser.write(gopreset_99)
# time.sleep(5)

# ser.close()


## 連接camera 1 
cap = VideoStream(src=1).start()
isOpened = True
control_index = 0

## 按下enter , 進行設定
while(isOpened):
    
    frame = cap.read()
    if frame is None :
        break
    cv2.imshow('frame', frame)
    # 若按下 q 鍵則離開迴圈
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == 13: #enter
        print('Please Input command :')
        command_s = input()
        command_s = command_s.split("0x")[1:]
        print("Your input = ",command_s)
        check_sum = checksum(command_s)
        print(check_sum)
        command= bytearray.fromhex(check_sum)
        ser.write(command)
    # else:
    #     print("Input Key = ",key)
    #     time.sleep(10)
    
# 釋放攝影機
cap.stream.release()
ser.close()

# 關閉所有 OpenCV 視窗
cv2.destroyAllWindows()
