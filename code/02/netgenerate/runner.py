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

netgenBinary = checkBinary('netgenerate')
netconvBinary = checkBinary('netconvert')
jtrrouterBinary = checkBinary('jtrrouter')
sumoBinary = checkBinary('sumo-gui')
import randomTrips

call([netgenBinary, '-c', 'data/netgenerate.netgcfg'])
call([netconvBinary, '-c', 'data/netgenerate.netccfg'])
randomTrips.main(randomTrips.get_options([
    '--flows', '1000',
    '-b', '0',
    '-e', '1',
    '-n', 'data/grid.tls.net.xml',
    '-o', 'data/flows.xml',
    '--jtrrouter',
    '--trip-attributes', 'departPos="random" departSpeed="max"']))
call([jtrrouterBinary, '-c', 'data/netgenerate.jtrrcfg'])
call([sumoBinary, '-c', 'data/netgenerate.sumocfg', '--duration-log.statistics', '-e' '1000'])