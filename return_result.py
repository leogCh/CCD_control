import requests
def return_result(command,r1,r2,r3,r4):
    c = command.split(',')
    url = f"""http://twkhh01wtappw1v.tw-khh01.nxp.com:8080/BatchWatcherService.asmx/AGV_RECORDING_CONTENT?
            CARRIERID={c[0]}&
            MID={c[1]}&
            PRIORITY={c[2]}&
            VEHICLEID={c[3]}&
            DESTPORT={c[4]}&
            INSPECTRESULT={r1}&
            JUDGERATE={r2}&
            ROTATION={r3}&
            IMAGES={r4}
            """
    print(url)
    url = url.replace(' ','')
    url = url.replace('\n','')
    try:
        print(requests.get(url).text)
    except:
        pass

if __name__=='__main__':
    import numpy as np
    import matplotlib.pyplot as plt
    import base64
    import sys
    import io

    temp_io = io.BytesIO()
    temp = np.random.rand(16).reshape((4,4))
    plt.imsave(temp_io, temp, format='png')
    frame = temp_io.getvalue()
    frame = base64.urlsafe_b64encode(frame).decode('utf-8')

    return_result('0xFF0x010x000x070x000x36,CTDM-01,50,CT-02,CT_TSK3-151', True, '85', '66', frame)