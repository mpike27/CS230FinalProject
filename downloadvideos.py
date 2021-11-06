import SoccerNet
from SoccerNet.Downloader import SoccerNetDownloader

mySoccerNetDownloader = SoccerNetDownloader(
    LocalDirectory="~/Data/SoccerNet")
mySoccerNetDownloader.password = "s0cc3rn3t"

# # download LQ Videos
mySoccerNetDownloader.downloadGame(files=["1.mkv", "Labels.json"], game = "england_epl/2014-2015/2015-02-21 - 18-00 Chelsea 1 - 1 Burnley")
