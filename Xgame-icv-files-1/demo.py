# zstub相关通信SDK
from zstub.zstub_impl import ZStubImpl
from zstub.constants.constants import *
from zstub.io.interfaces.subscription import Subscription
from zstub.constants.constants import *
from zstub.io.interfaces.zmsg import ZMsg
from zstub.io.interfaces.conf.conf import *

# 其他业务相关依赖
import time
import threading
from datetime import datetime
import logging
import argparse
import json
import numpy as np
import pulp

# 初始化SDK，建议相关参数非必要不修改
logging.basicConfig(level=logging.INFO,
                    format='[%(levelname)s][%(asctime)s][%(name)s][%(filename)s,%(funcName)s,%(lineno)s][%(message)s]')
logger = logging.getLogger(__name__)
data_type_list = [MEC_T_TF, MEC_T_RSM, MEC_T_SPAT]      # 接入数据种类配置：RSM、TF和SPAT3种数据接入
output_dst = "app1"
config = '{"application_id":"app0","mec_service":"127.0.0.1","default_mqtt_broker":"127.0.0.1","default_mqtt_port":1883}'       # 运行环境数据来源配置
out = None
zstub = None
sub = None

# 自行确定的参数
lanes = {}
map = {}
nodes = {}
with open("X-Y_map.json",'r') as file:
    map = json.load(file)
for node in map['nodes']:
    id = node['id']['id']
    nodes[id] = [[[],[],[]],[[],[],[]],[[],[],[]],[[],[],[]]]
    for linkex in node['inLinks_ex']:
        lon = linkex['refLine'][0]['posOffset']['offsetLL']['lon']
        lat = linkex['refLine'][0]['posOffset']['offsetLL']['lat']
        pos = -1
        if abs(lon) > abs(lat):
            if lon > 0:
                pos = 1
            elif lon < 0:
                pos = 3
        else:
            if lat > 0:
                pos = 2
            elif lat < 0:
                pos = 0
        for section in linkex['sections']:
            for lane in section['lanes']:
                lane_id = lane['ext_id']
                lanes[lane_id] = {}
                if 'maneuvers' in lane:
                    lane_direction = lane['maneuvers']['behavior']
                    if lane_direction & 1 > 0:
                        nodes[id][pos][1].append(lane_id)
                    if lane_direction & 2 > 0 or lane_direction & 8 > 0:
                        nodes[id][pos][0].append(lane_id)
                    if lane_direction & 4 > 0:
                        nodes[id][pos][2].append(lane_id)
                else:
                    nodes[id][pos][1].append(lane_id)


# 接入数据后回调，用户请根据接入数据类型构建相应的逻辑代码，以实现算法实时接入数据
# 接入后即为json格式的数据

def min_probability(s1,s2,s3,ele):
    if ele <= s1:
        return [1,0,0,0]
    elif ele <= (s1+s2)/2:
        return [(s1+s2-2*ele)/(s2-s1),2*(ele-s1)/(s2-s1),0,0]
    elif ele <= (s2+s3)/2:
        return [0,(s2+s3-2*ele)/(s3-s1),(2*ele-s1-s2)/(s3-s1),0]
    elif ele <= s3:
        return [0,0,2*(s3-ele)/(s3-s2),(2*ele-s2-s3)/(s3-s2)]
    else:
        return [0,0,0,1]
def max_probability(s1,s2,s3,ele):
    if ele >= s1:
        return [1,0,0,0]
    elif ele >= (s1+s2)/2:
        return [(s1+s2-2*ele)/(s2-s1),2*(ele-s1)/(s2-s1),0,0]
    elif ele >= (s2+s3)/2:
        return [0,(s2+s3-2*ele)/(s3-s1),(2*ele-s1-s2)/(s3-s1),0]
    elif ele >= s3:
        return [0,0,2*(s3-ele)/(s3-s2),(2*ele-s2-s3)/(s3-s2)]
    else:
        return [0,0,0,1]
def likelydelay(green_start_queue,green_time,red_time,arrive,leave):
    delay_t = 0
    n = green_start_queue
    tg = green_time
    tr = red_time
    a = arrive
    l = leave
    for i in range(n):
        delay_t += i / l
    if tg*(l - a) > n:
        for i in range(1,int(a*n/(l-a))+1):
            delay_t += (n-(l-a)*i/a)/l
        for i in range(int(a*tr)):
            delay_t += i/l+tr
    else:
        t0 = int((l*tg-n)/a)
        for i in range(1,int(a*t0)+1):
            delay_t += (n-(l-a)*i/a)/l
        for i in range(int(a*(tr+tg-t0))):
            delay_t += i/l + tr
    delay = delay_t/(n + (tg + tr)*a)
    return delay
def best_SPAT(arrive,B,c):
    a = arrive
    n = len(arrive)  # 向量和矩阵的维度

    # 创建问题实例
    problem = pulp.LpProblem("Integer Linear Programming", pulp.LpMinimize)

    # 定义变量
    t = []
    for i in range(n):
        t.append(pulp.LpVariable(f"t{i}", lowBound=0, cat='Integer'))

    # 定义目标函数
    problem += pulp.lpDot(a, t)

    # 添加约束条件
    for i in range(n):
        problem += pulp.lpDot(B[i], t) >= c[i]

    # 求解问题
    status = problem.solve()
    # 获取最优解的 t 向量
    t_values = []
    if status == pulp.LpStatusOptimal:
        for i in range(n):
            t_values.append(pulp.value(t[i]))
    return t_values

def sdk_msg_handler(msg_type, msg_body, origin):
    dt = datetime.fromtimestamp(datetime.now().timestamp())
    if msg_type == MEC_T_RSM:
        logger.info("Received participant data: {}".format(msg_body))
        rsm = json.loads(msg_body)
        for participant in rsm['participants']:
            if 'lane_ext_id' in participant:
                id = participant['lane_ext_id']
                if not id in lanes:
                    lanes[id] = {}
                    lanes[id]['dt'] = dt
                    lanes[id]['car_num'] = 0
                    lanes[id]['avg_speed'] = 0
                    #lanes[id]['car_width'] = 0
                if dt != lanes[id]['dt']:
                    lanes[id]['car_num'] = 0
                    lanes[id]['avg_speed'] = 0
                    #lanes[id]['car_width'] = 0
                if 'ptcType' in participant and participant['ptcType'] == 1:
                    if not 'car_num' in lanes[id]:
                        lanes[id]['car_num'] = 0
                    lanes[id]['car_num'] += 1
                    if not 'avg_speed' in lanes[id]:
                        lanes[id]['avg_speed'] = 0
                    if 'speed' in participant:
                        lanes[id]['avg_speed'] = (lanes[id]['avg_speed'] * (lanes[id]['car_num'] - 1) + participant['speed'] / 50) / lanes[id]['car_num'] #LBS: 0.02m/s -> 1m/s
            
            #计算拥堵程度
                p = [[0]*4]*5
                u = [2,1,1,3,2]
                if 'saturation' in lanes[id]:
                    p[0] = min_probability(60,80,90,lanes[id]['saturation'])
                if 'time_occupation' in lanes[id]:
                    p[1] = min_probability(70,85,95,lanes[id]['time_occupation'])
                if 'avg_speed' in lanes[id]:
                    p[2] = max_probability(25 * 1000 / 3600,19 * 1000 / 3600,16 * 1000 / 3600,lanes[id]['avg_speed'])
                if 'vor' in lanes[id]:
                    p[3] = max_probability(6 * 1000 * 1000 / 3600,2.8 * 1000 * 1000 / 3600,1.1 * 1000 * 1000 / 3600,lanes[id]['vor'])
                if 'delay' in lanes[id]:
                    p[4] = min_probability(5,10,20,lanes[id]['delay'])
                h = [0]*4
                for i in range(4):
                    for j in range(5):
                        h[i] += u[j] * p[j][i]
                confidence = 0.6
                s = 0
                i = -1
                while s < confidence:
                    i += 1
                    s += h[i]
                lanes[id]['congestion'] = 64 * i + 63
    elif msg_type == MEC_T_SPAT:
        logger.info("Received Signal Phase And Timing data: {}".format(msg_body))
        spat = json.loads(msg_body)
        for intersection in spat['intersections']:
            for lane in lanes.values():
                lane['red_time'] = lane['green_time'] = 0
            if not 'phases' in intersection:
                continue
            for phase in intersection['phases']:
                id = phase['id']
                for node in map['nodes']:
                    ifbreak = False
                    for linkex in node['inLinks_ex']:
                        for movement in linkex['movements_ex']:
                            if id == movement['phaseId']:
                                ifbreak = True
                                break
                        if ifbreak:
                            break
                    if ifbreak:
                        break
                phase_direction = movement['turnDirection']
                for section in linkex['sections']:
                    for lane in section['lanes']:
                        if 'maneuvers' in lane and 'behavior' in lane['maneuvers']:
                            lane_direction = lane['maneuvers']['behavior']
                        else:
                            lane_direction = 1
                        if (lane_direction & 1 > 0 and phase_direction == 0) or (lane_direction & 2 > 0 and phase_direction == 1) or (lane_direction & 4 > 0 and phase_direction == 2) or (lane_direction & 8 > 0 and phase_direction == 3):
                            id = lane['ext_id']
                            if not id in lanes:
                                lanes[id] = {}
                            if not 'red_time' in lanes[id]:
                                lanes[id]['red_time'] = 0
                            if not 'green_time' in lanes[id]:
                                lanes[id]['green_time'] = 0
                            for phaseState in phase['phaseStates']:
                                if phaseState['light'] == 'dark' or phaseState['light'] == 'stopAndRemain' or phaseState['light'] == 'preMovement':
                                    lanes[id]['red_time'] += (phaseState['timing']['likelyEndTime'] - phaseState['timing']['startTime']) / 10 #LSB 0.1s -> 1s
                                elif phaseState['light'] == 'stopThenProceed' or phaseState['light'] == 'permissiveMovementAllowed' or phaseState['light'] == 'protectedMovementAllowed':
                                    lanes[id]['green_time'] += (phaseState['timing']['likelyEndTime'] - phaseState['timing']['startTime']) / 10 #LSB: 0.1s -> 1s

    elif msg_type == MEC_T_TF:
        logger.info("Received Traffic Flow data: {}".format(msg_body))
        tf = json.loads(msg_body)
        for stat in tf['stats']:
            if 'map_element_type' in stat and stat['map_element_type'] == 'DE_LaneStatInfo' and 'map_element' in stat and 'ext_id' in stat['map_element']:
                id = stat['map_element']['ext_id']
                if not id in lanes:
                    lanes[id] = {}

                #计算平均速度
                if not 'avg_speed' in lanes[id]:
                    if 'time_headway' in stat and 'space_headway' in stat:
                        if stat['time_headway'] != 0:
                            lanes[id]['avg_speed'] = stat['space_headway'] / stat['time_headway'] 
                        else: 
                            lanes[id]['avg_speed'] =25 
                            #LSB: 1m/s

                #计算饱和度
                if 'ext' in stat and 'map_element' in stat['ext'] and len(stat['ext']['map_element']) > 0 and 'avg_saturation' in stat['ext']['map_element'][0]:
                    lanes[id]['saturation'] = stat['ext']['map_element'][0]['avg_saturation'] / 100 #LSB:0.01% -> 1%
                elif 'saturation' in stat:
                    lanes[id]['saturation'] = stat['saturation'] / 100 #LSB: 1%
                elif 'ext' in stat and 'map_element' in stat['ext'] and len(stat['ext']['map_element']) > 0 and 'capacity' in stat['ext']['map_element'][0]:
                    capacity = stat['ext']['map_element'][0]['capacity'] / 100 / 3600 #LSB: 0.01pcu/h -> 1pcu/s
                    if 'volume' in stat:
                        volume = stat['volume'] / 100 / 3600 #LSB: 0.01pcu/h -> 1pcu/s
                        if capacity != 0:
                            lanes[id]['saturation'] = volume / capacity / 100
                        else: 
                            lanes[id]['saturation'] = 0
                            #LSB : 1%
                    elif 'avg_speed' in lanes[id] and 'density' in stat:
                        avg_speed = lanes[id]['avg_speed']
                        density = stat['density'] / 100 / 1000 #LSB: 0.01pcu/km -> 1pcu/m
                        lanes[id]['saturation'] = avg_speed * density
                
                #计算速度占有率比
                if 'avg_speed' in lanes[id]:
                    speed = lanes[id]['avg_speed']
                    if 'ext' in stat and 'map_element' in stat['ext'] and len(stat['ext']['map_element']) > 0 and 'avg_occupation' in stat['ext']['map_element'][0]:
                        occ = stat['ext']['map_element'][0]['avg_occupation'] / 100 #LSB: 1%
                        if occ != 0:
                            lanes[id]['vor'] = speed / occ
                        else: 
                            lanes[id]['vor'] = 10
                    elif 'occupation' in stat:
                        occ = stat['occupation'] / 100 #LSB: 1%
                        if occ != 0:
                            lanes[id]['vor'] = speed / occ 
                        else:
                            lanes[id]['vor'] = 10
                    elif 'density' in stat and 'queue_length' in stat and 'queued_vehicles' in stat:
                        density = stat['density'] / 100 / 1000 #LSB: 0.01pcu/km -> 1pcu/m
                        queue_len = stat['queue_length'] / 10 #LSB: 0.1m -> 1m
                        queue_veh = stat['queued_vehicles'] #LSB: 1pcu
                        if queue_veh != 0:
                            occ = density * queue_len / queue_veh * 2 / 3
                        else:
                            occ = 0
                        #if 'width' in lanes[id] and 'car_width' in lanes[id]:
                            #occ = occ * lanes[id]['car_width'] / lanes[id]['width']
                        #else:
                            #occ = occ * 2 / 3
                        if occ != 0:
                            lanes[id]['vor'] = speed / occ
                        else:
                            lanes[id]['vor'] = 10

                #计算延迟
                if 'delay' in stat:
                    lanes[id]['delay'] = stat['delay'] / 10 #LSB: 0.1s/pcu -> 1s/pcu
                else:
                    n = tg = tr = a = l = -1
                    m = -1
                    if 'ext' in stat and 'signal' in stat['ext'] and len(stat['ext']['signal']) > 0 and 'green_start_queue' in stat['ext']['signal'][0]:
                        n = stat['ext']['signal'][0]['green_start_queue']
                        lanes[id]['queue'] = n
                    elif 'ext' in stat and 'map_element' in stat['ext'] and len(stat['ext']['map_element']) > 0 and 'avg_green_start_queue' in stat['ext']['map_element'][0]:
                        n = stat['ext']['map_element'][0]['avg_green_start_queue']
                        lanes[id]['queue'] = n
                    if 'ext' in stat and 'signal' in stat['ext'] and len(stat['ext']['signal']) > 0 and 'red_start_queue' in stat['ext']['signal'][0]:
                        m = stat['ext']['signal'][0]['red_start_queue']
                    elif 'ext' in stat and 'map_element' in stat['ext'] and len(stat['ext']['map_element']) > 0 and 'avg_red_start_queue' in stat['ext']['map_element'][0]:
                        m = stat['ext']['map_element'][0]['avg_red_start_queue']
                    if 'green_time' in lanes[id]:
                        tg = lanes[id]['green_time']
                    if 'red_time' in lanes[id]:
                        tr = lanes[id]['red_time']
                    if n > 0 and tg > 0 and tr > 0 and m > 0:
                        a = m / tr
                        l = m / tg
                        lanes[id]['arrive'] = a
                        lanes[id]['leave'] = l
                        lanes[id]['delay'] = likelydelay(n,tg,tr,a,l)

                #计算时间占有率
                if 'ext' in stat and 'map_element' in stat['ext'] and len(stat['ext']['map_element']) > 0 and 'avg_time_occupation' in stat['ext']['map_element'][0]:
                    lanes[id]['time_occupation'] = stat['ext']['map_element'][0]['avg_time_occupation'] / 100 #LSB: 0.01% -> 1%
                elif 'time_occupation' in stat:
                    lanes[id]['time_occupation'] = stat['time_occupation'] / 100 #LSB: 0.01% -> 1%
                
                #计算拥堵程度
                p = [[0]*4]*5
                u = [2,1,1,3,2]
                if 'saturation' in lanes[id]:
                    p[0] = min_probability(60,80,90,lanes[id]['saturation'])
                if 'time_occupation' in lanes[id]:
                    p[1] = min_probability(70,85,95,lanes[id]['time_occupation'])
                if 'avg_speed' in lanes[id]:
                    p[2] = max_probability(25 * 1000 / 3600,19 * 1000 / 3600,16 * 1000 / 3600,lanes[id]['avg_speed'])
                if 'vor' in lanes[id]:
                    p[3] = max_probability(6 * 1000 * 1000 / 3600,2.8 * 1000 * 1000 / 3600,1.1 * 1000 * 1000 / 3600,lanes[id]['vor'])
                if 'delay' in lanes[id]:
                    p[4] = min_probability(5,10,20,lanes[id]['delay'])
                h = [0]*4
                for i in range(4):
                    for j in range(5):
                        h[i] += u[j] * p[j][i]
                confidence = 0.6
                s = 0
                i = -1
                while s < confidence:
                    i += 1
                    s += h[i]
                lanes[id]['congestion'] = 64 * i + 63

# SDK错误回调，建议相关错误写入日志方便排查
def sdk_err_handler(err_code, err_msg):
    logger.error("handle err, err_code=%d, err_msg=%s", err_code, err_msg)

# 模拟算法：定时每1s发出一次MSG_SignalScheme信号优化方案
# 实际发出时机，由用户自行修改
def timed_output():
    while True:
        #time.sleep(1)
        dt = datetime.fromtimestamp(datetime.now().timestamp())
        print(
            f'-----------------{dt.hour}:{dt.minute}:{dt.second}-----------------')
        # 构建符合消息结构定义的json字符串
        scheme_id = 1
        for id in nodes.keys():
            ifneed = True
            node = nodes[id]
            '''for lane in node:
                for i in range(3):
                    for lane_id in lane[i]:
                        if 'congestion' in lanes[lane_id] and lanes[lane_id]['congestion'] >= 64 * 2 + 63:
                            ifneed = True'''
            if ifneed:
                link_num = 0
                for link in node:
                    if len(link) != 0:
                        link_num += 1
                arrive = [0]*link_num*3
                leave = [0]*link_num*3
                Q = [0]*link_num*3
                B = [[1] * link_num * 3 for i in range(link_num * 3)]
                cong = [0] * link_num * 3
                for i in range(link_num):
                    B[i][i] = -2
                position = 0
                for link in node:
                    if len(link) != 0:
                        for lane in link:
                            if len(lane) == 0:
                                arrive[position] = leave[position] = Q[position] = 0
                                cong[position] = 63
                                position += 1
                            else:
                                anum = 0
                                lnum = 0
                                qnum = 0
                                cnum = 0
                                for lane_id in lane:
                                    if 'arrive' in lanes[lane_id]:
                                        arrive[position] += lanes[lane_id]['arrive']
                                        anum += 1
                                    if 'leave' in lanes[lane_id]:
                                        leave[position] += lanes[lane_id]['leave']
                                        lnum += 1
                                    if 'queue' in lanes[lane_id]:
                                        Q[position] += lanes[lane_id]['queue']
                                        qnum += 1
                                    elif 'car_num' in lanes[lane_id]:
                                        Q[position] += lanes[lane_id]['car_num']
                                        qnum += 1
                                    if 'congestion' in lanes[lane_id]:
                                        cong[position] += lanes[lane_id]['congestion']
                                if anum != 0:
                                    arrive[position] /= anum
                                else:
                                    arrive[position] = 0
                                if lnum != 0:
                                    leave[position] /= lnum
                                else:
                                    leave[position] = 0
                                if qnum != 0:
                                    Q[position] /= qnum
                                else:
                                    Q[position] = 0
                                if cnum != 0:
                                    cong[position] /= cnum
                                else:
                                    cong[position] = 63
                                position += 1
                c = [0] * link_num * 3
                for i in range(position):
                    if leave[i] != arrive[i]:
                        c[i] = 3 * Q[i] / (leave[i] - arrive[i])
                    elif leave[i] != 0:
                        c[i] = 3 * Q[i] / leave[i]
                    else:
                        c[i] = 0
                nozero = True
                for con in cong:
                    if con == 0:
                        nozero = False
                        break
                if nozero:
                    for i in range(len(arrive)):
                        arrive[i] = arrive[i] * cong[i]
                arrive = np.array(arrive)
                B = np.array(B)
                c = np.array(c)
                red_time = best_SPAT(arrive,B,c)
                green_time = []
                if len(red_time) != 0:
                    green_time = B @ red_time
                    green_time = [int (tg / (link_num - 1)) for tg in green_time]
                s = {}
                s['scheme_id'] = scheme_id
                for nodeToFind in map['nodes']:
                    if nodeToFind['id']['id'] == id:
                        break
                s['node_id'] = {'region':nodeToFind['id']['region'], 'id':nodeToFind['id']['id']}
                now = datetime.now()
                month = now.month
                month_abbr = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN","JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
                day = now.day
                weekday = now.weekday()
                weekday_abbr = ["MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN"]
                current_hour = now.hour
                current_minute = now.minute
                current_second = now.second
                s['time_span'] = {
                    "month_filter": [
                        month_abbr[month - 1]
                    ],
                    "day_filter": [
                        day
                    ],
                    "weekday_filter": [
                        weekday_abbr[weekday - 1]
                    ],
                    "from_time_point": {
                        "hh": current_hour,
                        "mm": current_minute,
                        "ss": current_second
                    },
                    "to_time_point": {
                        "hh": (current_hour + 8) % 24,
                        "mm": current_minute,
                        "ss": current_second
                    }
                }
                s['cycle'] = sum(green_time) + 33
                s['control_mode'] = "CYCLIC_FIXED"
                tg_max = [max(green_time[i], green_time[i + 6]) for i in range(6)]
                tg_min = [min(green_time[i], green_time[i + 6]) for i in range(6)]
                tg = [(tg_max[i] + tg_min[i]) / 2 for i in range(6)]
                s["min_cycle"] = 2 * sum(tg_min)
                s["max_cycle"] = 2 * sum(tg_max)
                s["base_signal_scheme_id"] = 1
                s["offset"] = 0
                s['phases'] = [
                    {
                        "id": 1,
                        "order": 1,
                        "movements": [
                            "10",
                            "2",
                        ],
                        "green": tg[1],
                        "yellow": 3,
                        "allred": 5,
                        "min_green": tg_min[1],
                        "max_green": tg_max[1]
                    },
                    {
                        "id": 2,
                        "order": 2,
                        "movements": [
                            "9",
                            "1",
                        ],
                        "green": tg[0],
                        "yellow": 3,
                        "allred": 0,
                        "min_green": tg_min[0],
                        "max_green": tg_max[0]
                    },
                    {
                        "id": 3,
                        "order": 3,
                        "movements": [
                            "11",
                            "3"
                        ],
                        "green": tg[2],
                        "yellow": 3,
                        "allred": 5,
                        "min_green": tg_min[2],
                        "max_green": tg_max[2]
                    },
                    {
                        "id": 4,
                        "order": 4,
                        "movements": [
                            "14",
                            "6",
                        ],
                        "green": tg[4],
                        "yellow": 3,
                        "allred": 0,
                        "min_green": tg_min[4],
                        "max_green": tg_max[4]
                    },
                    {
                        "id": 5,
                        "order": 5,
                        "movements": [
                            "13",
                            "5"
                        ],
                        "green": tg[3],
                        "yellow": 3,
                        "allred": 5,
                        "min_green": tg_min[3],
                        "max_green": tg_max[3]
                    },
                    {
                        "id": 6,
                        "order": 6,
                        "movements": [
                            "15",
                            "7"
                        ],
                        "green": tg[5],
                        "yellow": 3,
                        "allred": 0,
                        "min_green": tg_min[5],
                        "max_green": tg_max[5]
                    }
                ]
                '''s = {
                "scheme_id": scheme_id,
                "node_id": {
                    "region": nodeToFind['id']['region'],
                    "id": nodeToFind['id']['id']
                },
                "time_span": {
                    "month_filter": [
                        month_abbr[month - 1]
                    ],
                    "day_filter": [
                        day
                    ],
                    "weekday_filter": [
                        weekday_abbr[weekday - 1]
                    ],
                    "from_time_point": {
                        "hh": current_hour,
                        "mm": current_minute,
                        "ss": current_second
                    },
                    "to_time_point": {
                        "hh": (current_hour + 8) % 24,
                        "mm": current_minute,
                        "ss": current_second
                    }
                },
                "cycle": sum(green_time) + 33,
                "control_mode": "CYCLIC_FIXED",
                "min_cycle": 2 * sum(tg_min),
                "max_cycle": 2 * sum(tg_max),
                "base_signal_scheme_id": 1,
                "offset": 0,
                "phases": [
                    {
                        "id": 1,
                        "order": 1,
                        "movements": [
                            "SouthGoStraight",
                            "NorthGoStraight",
                            "NorthPedestrainPass",
                            "SouthPedestrainPass"
                        ],
                        "green": tg[1],
                        "yellow": 3,
                        "allred": 5,
                        "min_green": tg_min[1],
                        "max_green": tg_max[1]
                    },
                    {
                        "id": 2,
                        "order": 2,
                        "movements": [
                            "SouthTurnLeft",
                            "NorthTurnLeft",
                        ],
                        "green": tg[0],
                        "yellow": 3,
                        "allred": 0,
                        "min_green": tg_min[0],
                        "max_green": tg_max[0]
                    },
                    {
                        "id": 3,
                        "order": 3,
                        "movements": [
                            "SouthTurnRight",
                            "NorthTurnRight"
                        ],
                        "green": tg[2],
                        "yellow": 3,
                        "allred": 5,
                        "min_green": tg_min[2],
                        "max_green": tg_max[2]
                    },
                    {
                        "id": 4,
                        "order": 4,
                        "movements": [
                            "WestGoStraight",
                            "EastGoStraight",
                            "EastPedestrainPass",
                            "WestPedestrainPass"
                        ],
                        "green": tg[4],
                        "yellow": 3,
                        "allred": 0,
                        "min_green": tg_min[4],
                        "max_green": tg_max[4]
                    },
                    {
                        "id": 5,
                        "order": 5,
                        "movements": [
                            "WestTurnLeft",
                            "EastTurnLeft"
                        ],
                        "green": tg[3],
                        "yellow": 3,
                        "allred": 5,
                        "min_green": tg_min[3],
                        "max_green": tg_max[3]
                    },
                    {
                        "id": 6,
                        "order": 6,
                        "movements": [
                            "WestTurnRight",
                            "EastTurnRight"
                        ],
                        "green": tg[5],
                        "yellow": 3,
                        "allred": 0,
                        "min_green": tg_min[5],
                        "max_green": tg_max[5]
                    }
                ],
                "msg_id": 1
                }'''
                scheme_id += 1
                test_ss = json.dumps(s)
                # 构造SDK消息体对象zmsg，填入消息种类代号MEC_T_SS
                # 注意字符串消息需要encode成为bytes对象
                zmsg = ZMsg(MEC_T_SS, test_ss.encode(), K_CONTENT_TYPE_JSON)
                # 使用SDK输出
                out.send(zmsg)

# 初始化SDK相关变量和算法线程
def run(args):
    global out, zstub, sub
    debug = False
    zstub = ZStubImpl(config, debug)
    sub = zstub.subscribe(data_type_list)
    sub.set_msg_handler(sdk_msg_handler)
    sub.set_err_handler(sdk_err_handler)
    out = zstub.out_connect(output_dst)
    output_thread = threading.Thread(target=timed_output)
    output_thread.daemon = True
    output_thread.start()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t", "--test", help='Run test data', type=int, default=0)
    args = parser.parse_args()
    run(args)

