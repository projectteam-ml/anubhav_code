import torch
import numpy as np
from gym.spaces import Box

class Environment:
    def __init__(self):
        super(Environment, self).__init__()
        # 定义迷宫的大小和起始位置、目标位置
        self.size = (300, 300, 300)
        self.num_IoTD = 5   #IoTD数量
        self.start_position_A = np.array([0, 299,0])     # 1号无人机起点
        self.start_position_B = np.array([299,299, 200])    # 2号无人机起点
        self.current_position_A = self.start_position_A
        self.current_position_B = self.start_position_B
        self.AoI = np.zeros(self.num_IoTD)    # AoI计数器
        self.R_min = 0.1
        self.time = 0  # 时间片
        self.T = 150   # 一次游戏的总时间
        self.E = 0  # 总能耗
        self.A = 0  # 总AoI
        self.avg_E = 0  # 平均能耗
        self.avg_A = 0  # 平均AoI
        self.done = False

        # 定义动作空间和观察空间
        self.action_space = Box(low=np.array([-1] * 9), high=np.array([1] * 9))
        self.observation_space = Box(low=np.array([0] * 13), high=np.array([1] * 13))

        # 定义迷宫布局
        self.iotd_position = np.array([
            [50, 50, 0], [75, 150, 0], [100, 100, 0], [100, 250, 0], [150, 150, 0]
        ])
        self.e_position = np.array([
            [250, 100, 0], [280, 200, 0], [175, 225, 0], [200, 200, 0], [100, 150, 0]
        ])

    def reset(self):
        # 重置环境，将智能体放到起始位置
        self.current_position_A = self.start_position_A
        self.current_position_B = self.start_position_B
        self.AoI = np.zeros(self.num_IoTD)
        self.time = 0  # 时间片
        self.E = 0  # 总能耗
        self.A = 0  # 总AoI
        self.avg_E = 0  # 平均能耗
        self.avg_A = 0  # 平均AoI
        UAV_A_position = np.array([self.start_position_A[0], self.start_position_A[1], self.start_position_A[2]])
        UAV_B_position = np.array([self.start_position_B[0], self.start_position_B[1], self.start_position_B[2]])
        energy_A = np.array([0])
        energy_B = np.array([0])
        self.done = False
        state = np.concatenate((UAV_A_position, UAV_B_position ,self.AoI , energy_A, energy_B))   # 状态
        return state

    def energy(self, pos_A, pos_B):
        energy = (79.85 * (1 + 3 * (np.linalg.norm(pos_A - pos_B) / 120) ** 2)
                    + 88.63 * np.sqrt(1 + 0.25 *
                                      (np.linalg.norm(pos_A - pos_B) / 4.03) ** 4)
                    - 0.5 * np.sqrt((np.linalg.norm(pos_A - pos_B) / 4.03) ** 2)
                    + 0.5 * 0.6 * 1.225 * 0.05 * 0.503 * np.linalg.norm(pos_A - pos_B) ** 3
                    )
        energy = np.array([energy])
        return  energy

    #action；水平移动、竖直移动、选定IOTD
    def step(self, action):
        # action形状：2+2+5   2控制移动，5控制选取
        self.done = (self.time >= self.T)
        reward = 0
        # 将动作解析为新的位置
        action_A = np.array([action[0]*20, action[1]*20])
        action_B = np.array([action[2]*20, action[3]*20])
        choose_A = np.zeros((5, 1))
        for i in range(2, 7):
            choose_A[i - 2, 0] = action[i + 2]
        select_A = torch.argmax(torch.tensor(choose_A), dim=0)

        for i in range(0, 5):
            choose_A[i] = 0
            # choose_B[i] = 0
        choose_A[select_A] = 1

        new_A_x = int(self.current_position_A[0] + action_A[0])
        new_A_y = int(self.current_position_A[1] + action_A[1])
        new_A_z = int(self.current_position_A[2])

        new_B_x = int(self.current_position_B[0] + action_B[0])
        new_B_y = int(self.current_position_B[1] + action_B[1])
        new_B_z = int(self.current_position_B[2])

        if new_A_x < 0 or new_A_y > 299 or new_A_x > 299 or new_A_y < 0 or new_B_x < 0 or new_B_y > 299 or new_B_x > 299 or new_B_y < 0:
            # past_position_A = self.current_position_A  # 留档之前的位置，用于计算距离
            # past_position_B = self.current_position_B
            if new_A_x < 0:
                new_A_x = 0
            if new_A_x >299:
                new_A_x = 299
            if new_A_y < 0:
                new_A_y = 0
            if new_A_y > 299:
                new_A_y = 299
            if new_B_x < 0:
                new_B_x = 0
            if new_B_x >299:
                new_B_x = 299
            if new_B_y < 0:
                new_B_y = 0
            if new_B_y > 299:
                new_B_y = 299
            self.current_position_A = np.array([new_A_x, new_A_y, new_A_z])
            self.current_position_B = np.array([new_B_x, new_B_y, new_B_z])
            for index, v in enumerate(self.AoI):
                self.AoI[index]+=1
            reward = -99
            self.time += 1
            energy_A = np.array([500])
            energy_B = np.array([500])
            state = np.concatenate((self.current_position_A, self.current_position_B ,self.AoI , energy_A, energy_B))
            return state, reward, self.done, {}

        # 更新当前位置
        past_position_A = self.current_position_A   # 留档之前的位置，用于计算距离
        past_position_B = self.current_position_B
        self.current_position_A = np.array([new_A_x, new_A_y, new_A_z])
        self.current_position_B = np.array([new_B_x, new_B_y, new_B_z])

        self.time += 1
        #走一步的能耗
        energy_A = self.energy(self.current_position_A, past_position_A)
        energy_B = self.energy(self.current_position_B, past_position_B)

        # 计算 UAV 到所有 IoTD 的距离
        Distance_UAV_IoTD = np.zeros(5)
        for i in range(0, 5):
            dist = np.linalg.norm(self.current_position_A - self.iotd_position[i])
            Distance_UAV_IoTD[i] = dist
       #计算所有 IoTD 到所有 窃听者 的距离
        Distance_IoTD_e = np.zeros((5, 5))
        for i in range(0, 5):
            for j in range(0, 5):
                dist = np.linalg.norm(self.iotd_position[i] - self.e_position[j])
                Distance_IoTD_e[i][j] = dist
        # 计算 UAV 到 Jammer 的距离
        Distance_UAV_Jammer = np.linalg.norm(self.current_position_A - self.current_position_B)
        # 计算 Jammer 到所有 窃听者 的距离
        Distance_Jammer_e = np.zeros(5)
        for i in range(0, 5):
            dist = np.linalg.norm(self.current_position_B - self.e_position[i])
            Distance_Jammer_e[i] = dist


        # 安全性评估
        H_UAV_IoTD = np.zeros(5)
        H_IoTD_e = np.zeros((5, 5))
        H_UAV_Jammer = 0
        H_Jammer_e = np.zeros(5)

        for i in range(0, 5):
            H_UAV_IoTD[i] = (np.sqrt(0.001 / (Distance_UAV_IoTD[i] ** 2))
                             * np.sqrt(1 / 2)
                             * (np.e ** (-1j * 2 * np.pi * Distance_UAV_IoTD[i] / 0.12))
                             + np.sqrt(1 / 2) * np.random.normal(0, 1)
                             )
        for i in range(0, 5):
            for j in range(0, 5):
                H_IoTD_e[i][j] = (np.sqrt(0.001 / (Distance_IoTD_e[i][j] ** 2))
                                  *(np.sqrt(1 / 2)
                                  * (np.e ** (-1j * 2 * np.pi * Distance_IoTD_e[i][j] / 0.12)))
                                  + np.sqrt(1 / 2) * np.random.normal(0, 1)
                                  )
        H_UAV_Jammer = (np.sqrt(0.001 / (Distance_UAV_Jammer ** 2))
                                  *(np.sqrt(1 / 2)
                                  * (np.e ** (-1j * 2 * np.pi * Distance_UAV_Jammer / 0.12)))
                                  + np.sqrt(1 / 2) * np.random.normal(0, 1)
                        )
        for i in range(0, 5):
            H_Jammer_e[i] = (np.sqrt(0.001 / (Distance_Jammer_e[i] ** 2))
                             * np.sqrt(1 / 2)
                             * (np.e ** (-1j * 2 * np.pi * Distance_Jammer_e[i] / 0.12))
                             + np.sqrt(1 / 2) * np.random.normal(0, 1)
                             )

        R_D = np.zeros(5)
        R_E = np.zeros((5, 5))

        for i in range(0, 5):
            R_D[i] = np.log2(1 + 0.01 * (H_UAV_IoTD[i] ** 2) / (1e-13 + 0.001 * (H_UAV_Jammer ** 2)))

        for x in range(0, 5):
            for y in range(0, 5):
                R_E[x][y] = np.log2(1 + 0.01 * (H_IoTD_e[x][y] ** 2) / (1e-13 + 0.001 * H_Jammer_e[y]**2))

        def max_value(lst):
            # print(lst)
            max_value = -9999
            for item in lst:
                if item > max_value:
                    max_value = item
            return max_value

        R_sec = np.zeros(5)
        ttt = []
        for i in range(0, 5):      #第i个IoTD
            for j in range(0,5):   #第j个窃听者
                ttt.append(
                    R_D[i] - R_E[i][j]
            )
            R_sec[i] = (
                max(0, max_value(ttt))
            )

        self.A = 0
        for index, v in enumerate(self.AoI):
            if R_sec[index] > self.R_min and choose_A[index] == 1:
                self.A += self.AoI[index]
                self.AoI[index] = 1
                reward += 100
            else:
                self.AoI[index] += 1

        energy = energy_A + energy_B
        self.E += energy
        self.avg_E = self.E / self.time

        reward = reward - 0.1 * energy - 0.1*self.A
        reward = reward.squeeze(0)
        reward = reward.item()

        done = (self.time >= self.T)

        state = np.concatenate((self.current_position_A, self.current_position_B ,self.AoI , energy_A, energy_B))
        self.avg_A = self.A / self.time
        # 返回新的观察值、奖励、是否结束、额外信息（可选）
        return state, reward, done, {}
