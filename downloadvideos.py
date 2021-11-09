import SoccerNet
from SoccerNet.Downloader import SoccerNetDownloader

mySoccerNetDownloader.password = "s0cc3rn3t"

# # download LQ Videos

#train
mySoccerNetDownloader = SoccerNetDownloader(
    LocalDirectory="train/Data/SoccerNet")
mySoccerNetDownloader.downloadGame(files=["1.mkv", "Labels.json"], game = "england_epl/2015-2016/2015-09-26 - 17-00 Liverpool 3 - 2 Aston Villa")
mySoccerNetDownloader.downloadGame(files=["1.mkv", "Labels.json"], game = "england_epl/2015-2016/2015-10-31 - 15-45 Chelsea 1 - 3 Liverpool")
mySoccerNetDownloader.downloadGame(files=["1.mkv", "Labels.json"], game = "england_epl/2015-2016/2015-11-21 - 20-30 Manchester City 1 - 4 Liverpool")
mySoccerNetDownloader.downloadGame(files=["1.mkv", "Labels.json"], game = "england_epl/2015-2016/2015-12-19 - 18-00 Chelsea 3 - 1 Sunderland")
mySoccerNetDownloader.downloadGame(files=["1.mkv", "Labels.json"], game = "england_epl/2016-2017/2016-08-14 - 18-00 Arsenal 3 - 4 Liverpool")
mySoccerNetDownloader.downloadGame(files=["1.mkv", "Labels.json"], game = "england_epl/2016-2017/2016-08-14 - 18-00 Arsenal 3 - 4 Liverpool")
mySoccerNetDownloader.downloadGame(files=["1.mkv", "Labels.json"], game = "england_epl/2016-2017/2016-09-17 - 17-00 Hull City 1 - 4 Arsenal")
mySoccerNetDownloader.downloadGame(files=["1.mkv", "Labels.json"], game = "england_epl/2016-2017/2016-10-29 - 14-30 Sunderland 1 - 4 Arsenal")
mySoccerNetDownloader.downloadGame(files=["1.mkv", "Labels.json"], game = "england_epl/2016-2017/2016-11-06 - 17-15 Liverpool 6 - 1 Watford")
mySoccerNetDownloader.downloadGame(files=["1.mkv", "Labels.json"], game = "england_epl/2016-2017/2016-12-10 - 20-30 Leicester 4 - 2 Manchester City")

#test
mySoccerNetDownloader = SoccerNetDownloader(
    LocalDirectory="test/Data/SoccerNet")
mySoccerNetDownloader.downloadGame(files=["1.mkv", "Labels.json"], game = "england_epl/2016-2017/2016-12-27 - 20-15 Liverpool 4 - 1 Stoke City")
mySoccerNetDownloader.downloadGame(files=["1.mkv", "Labels.json"], game = "england_epl/2016-2017/2017-02-27 - 23-00 Leicester 3 - 1 Liverpool")

#val
mySoccerNetDownloader = SoccerNetDownloader(
    LocalDirectory="val/Data/SoccerNet")
mySoccerNetDownloader.downloadGame(files=["1.mkv", "Labels.json"], game = "england_epl/2016-2017/2017-01-14 - 20-30 Leicester 0 - 3 Chelsea")
mySoccerNetDownloader.downloadGame(files=["1.mkv", "Labels.json"], game = "england_epl/2015-2016/2015-08-08 - 19-30 Chelsea 2 - 2 Swansea")
