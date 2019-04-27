import os
import sys

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

import traci
import traci.constants as tc

sumoBinary = "sumo"  # or "sumo-gui"
sumoCmd = [sumoBinary, "-c", "data/circles.sumocfg"]

traci.start(sumoCmd)

# 先运行几个仿真步，保证车辆进入路网
for step in range(5):
    traci.simulationStep()

# 将其中一辆车的vehID所在的路段和位置进行订阅
vehIDs = traci.vehicle.getIDList()
vehID = vehIDs[0]
traci.vehicle.subscribe(vehID, (tc.VAR_ROAD_ID, tc.VAR_LANEPOSITION))

for step in range(30):
    print("step", step)
    traci.simulationStep()
    print(traci.vehicle.getSubscriptionResults(vehID))
traci.close()
