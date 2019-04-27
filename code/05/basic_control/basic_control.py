import os
import sys

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

import traci

sumoBinary = "sumo"  # or "sumo-gui"
sumoCmd = [sumoBinary, "-c", "data/circles.sumocfg"]

traci.start(sumoCmd)
print(traci.getVersion())
step = 0
while step < 1000:
    traci.simulationStep()
    step += 1

traci.close()
