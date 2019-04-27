import os
from subprocess import call
import sys
try:
    sys.path.append(os.path.join(os.path.dirname(
        __file__), '..', '..', '..', '..', "tools"))
    sys.path.append(os.path.join(os.environ.get("SUMO_HOME", os.path.join(
        os.path.dirname(__file__), "..", "..", "..")), "tools"))
    from sumolib import checkBinary
except ImportError:
    sys.exit("please declare environment variable 'SUMO_HOME'")

netconvBinary = checkBinary('netconvert')
jtrrouterBinary = checkBinary('jtrrouter')
sumoBinary = checkBinary('sumo-gui')
import randomTrips

call([netconvBinary, '-c', 'data/netconvert.netccfg'])
randomTrips.main(randomTrips.get_options([
    '--flows', '1000',
    '-b', '0',
    '-e', '1',
    '-n', 'data/map.net.xml',
    '-o', 'data/flows.xml',
    '--jtrrouter',
    '--trip-attributes', 'departPos="random" departSpeed="max"']))
call([jtrrouterBinary, '-c', 'data/netconvert.jtrrcfg'])
call([sumoBinary, '-c', 'data/netconvert.sumocfg', '--duration-log.statistics', '-e' '1000'])