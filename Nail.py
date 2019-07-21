import argparse
import logging
import sys
import os
import pandas as pd
from urllib import parse
"""
“华为云杯”2019深圳开放数据应用创新大赛由深圳市政务服务数据管理局联合坪山区人民政府、前海管理局共同主办，在内地和港澳地区范围内征集基于开放数据的创新应用解决方案。
大赛旨在全面推动深圳市新型智慧城市和“数字政府”建设，推动政府部门进一步开放数据，释放数据能量和激发社会创新活力，并形成智慧产业链和高效的城市治理方案，促进深圳经济高速发展的同时，推动大湾区数据跨区域流动和高水平互通。

· 其中地图经度和地图纬度都是经过地图的纠偏规则进行纠偏后，符合地图底层经纬度设置的值。

· GPS时间为车载终端产生这条轨迹数据时的北京时间。

· 事件参考808协议中关于事件的解释，常见的是点火、熄火。

· 报警编码参考808协议中关于报警的解释，当条数据无报警时此值为空。

· 系统接收时间是指联网联控平台接收到此条数据时的北京时间。
"""


class GetData(object):
    """
    (0)地图经度，即纠偏后的经度，除以600,000后，得到WGS 84坐标系下的经度:
    (1)地图纬度，即纠偏后的纬度，除以600,000后，得到WGS 84坐标系下的纬度:
    (2)GPS时间:
    (3)GPS速度:
    (4)方向:
    (5)事件:
    (6)报警编码:
    (7)GPS经度:
    (8)GPS纬度:
    (9)海拔:
    (10)行驶记录仪速度:
    (11)里程:
    (12)错误类型(0:正常;1:经度错误;2:纬度错误;3:时间错误;4:速度错误;5:方向错误):
    (13)上传的运营商的接入码:
    (14)系统接收时间
    """

    def __init__(self, path, output):
        self.path = path
        self.output = output
        self.vehicle_color = {1: "蓝色", 2: "黄色", 3: "黑色", 4: "白色", 9: "其他"}
        self.header = ['mapLongitude', 'mapLatitude', 'GPSTime', 'GPSSpeed', 'direction', 'events',
                       'alarmCode', 'GPSLongitude', 'GPSLatitude', 'altitude',
                       'speedOfDriverRecorder', 'mileage', 'errorType',
                       'uploadedOperatorAccessCode', 'systemReceptionTime']
        self.data = None
        if os.path.exists(output):
            os.remove(output)
        self.extractfile()

    def readfile(self, file):
        self.data = pd.read_csv(file, sep=':', names=self.header)

    def extractfile(self):
        file = os.listdir(self.path)
        length = file.__len__()
        num = 0
        logger.info('Found ' + str(length) + ' files')

        for i, f in enumerate(file):
            if not i % int(length / 100):
                progress = '\t||\t' + '=' * num + '-' * (100 - num) + '> ' + i.__str__() + ' / ' + length.__str__()
                logger.info(progress)
                num += 1

            vehicle_info = f.strip('.txt').split('_')
            number = parse.unquote(vehicle_info[-1], encoding='GB2312')
            self.readfile(os.path.join(self.path, f))
            self.data['vehicleNumber'] = number
            self.data.to_csv(self.output, mode='a', index=None, encoding='utf-8', header=None)

        logger.info(f'Merge Finished! Merge completed shared time ' + self.output)


if __name__ == '__main__':
    logger = logging.getLogger(__name__)

    logging.basicConfig(
        format='%(asctime)s : %(threadName)s : %(levelname)s : %(message)s',
        level=logging.INFO
    )
    logger.info("running %s", " ".join(sys.argv))

    # check and process cmdline input
    program = os.path.basename(sys.argv[0])
    if len(sys.argv) < 2:
        print(globals()['__doc__'] % locals())
        sys.exit(1)

    parser = argparse.ArgumentParser()
    parser.add_argument("-input", help="Please enter merged data", required=True)
    parser.add_argument("-output", help="Please enter merge completion data", required=True)

    args = parser.parse_args()
    GetData(args.input, args.output)