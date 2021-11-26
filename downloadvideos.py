import SoccerNet
from SoccerNet.Downloader import SoccerNetDownloader

password = "s0cc3rn3t"
dataset_path = '/scratch/users/mpike27/CS230/data/'

# # download LQ Videos

# Lables-v2.json

#train
mySoccerNetDownloader = SoccerNetDownloader(
    LocalDirectory=dataset_path + "train/SoccerNet")
mySoccerNetDownloader.password = password
mySoccerNetDownloader.downloadGame(files=["1.mkv", "Labels-v2.json"], game = "england_epl/2015-2016/2015-09-26 - 17-00 Liverpool 3 - 2 Aston Villa")
mySoccerNetDownloader.downloadGame(files=["1.mkv", "Labels-v2.json"], game = "england_epl/2015-2016/2015-10-31 - 15-45 Chelsea 1 - 3 Liverpool")
mySoccerNetDownloader.downloadGame(files=["1.mkv", "Labels-v2.json"], game = "england_epl/2015-2016/2015-11-21 - 20-30 Manchester City 1 - 4 Liverpool")
mySoccerNetDownloader.downloadGame(files=["1.mkv", "Labels-v2.json"], game = "england_epl/2015-2016/2015-12-19 - 18-00 Chelsea 3 - 1 Sunderland")
mySoccerNetDownloader.downloadGame(files=["1.mkv", "Labels-v2.json"], game = "england_epl/2016-2017/2016-08-14 - 18-00 Arsenal 3 - 4 Liverpool")
mySoccerNetDownloader.downloadGame(files=["1.mkv", "Labels-v2.json"], game = "england_epl/2016-2017/2016-08-14 - 18-00 Arsenal 3 - 4 Liverpool")
mySoccerNetDownloader.downloadGame(files=["1.mkv", "Labels-v2.json"], game = "england_epl/2016-2017/2016-09-17 - 17-00 Hull City 1 - 4 Arsenal")
mySoccerNetDownloader.downloadGame(files=["1.mkv", "Labels-v2.json"], game = "england_epl/2016-2017/2016-10-29 - 14-30 Sunderland 1 - 4 Arsenal")
mySoccerNetDownloader.downloadGame(files=["1.mkv", "Labels-v2.json"], game = "england_epl/2016-2017/2016-11-06 - 17-15 Liverpool 6 - 1 Watford")
mySoccerNetDownloader.downloadGame(files=["1.mkv", "Labels-v2.json"], game = "england_epl/2016-2017/2016-12-10 - 20-30 Leicester 4 - 2 Manchester City")
mySoccerNetDownloader.downloadGame(files=["1.mkv", "Labels-v2.json"], game = "england_epl/2014-2015/2015-02-21 - 18-00 Chelsea 1 - 1 Burnley") 
mySoccerNetDownloader.downloadGame(files=["1.mkv", "Labels-v2.json"], game = "england_epl/2014-2015/2015-02-21 - 18-00 Crystal Palace 1 - 2 Arsenal") 
mySoccerNetDownloader.downloadGame(files=["1.mkv", "Labels-v2.json"], game = "england_epl/2014-2015/2015-02-21 - 18-00 Swansea 2 - 1 Manchester United") 
mySoccerNetDownloader.downloadGame(files=["1.mkv", "Labels-v2.json"], game = "england_epl/2014-2015/2015-02-22 - 19-15 Southampton 0 - 2 Liverpool") 
mySoccerNetDownloader.downloadGame(files=["1.mkv", "Labels-v2.json"], game = "england_epl/2014-2015/2015-04-11 - 19-30 Burnley 0 - 1 Arsenal") 
mySoccerNetDownloader.downloadGame(files=["1.mkv", "Labels-v2.json"], game = "england_epl/2014-2015/2015-05-17 - 18-00 Manchester United 1 - 1 Arsenal")
mySoccerNetDownloader.downloadGame(files=["1.mkv", "Labels-v2.json"], game = "england_epl/2015-2016/2015-08-08 - 19-30 Chelsea 2 - 2 Swansea")
mySoccerNetDownloader.downloadGame(files=["1.mkv", "Labels-v2.json"], game = "england_epl/2015-2016/2015-08-16 - 18-00 Manchester City 3 - 0 Chelsea")
mySoccerNetDownloader.downloadGame(files=["1.mkv", "Labels-v2.json"], game = "england_epl/2015-2016/2015-08-23 - 15-30 West Brom 2 - 3 Chelsea")
mySoccerNetDownloader.downloadGame(files=["1.mkv", "Labels-v2.json"], game = "england_epl/2015-2016/2015-08-29 - 17-00 Chelsea 1 - 2 Crystal Palace")
mySoccerNetDownloader.downloadGame(files=["1.mkv", "Labels-v2.json"], game = "england_epl/2015-2016/2015-08-29 - 17-00 Liverpool 0 - 3 West Ham")
mySoccerNetDownloader.downloadGame(files=["1.mkv", "Labels-v2.json"], game = "england_epl/2015-2016/2015-08-29 - 17-00 Manchester City 2 - 0 Watford")
mySoccerNetDownloader.downloadGame(files=["1.mkv", "Labels-v2.json"], game = "england_epl/2015-2016/2015-08-30 - 18-00 Swansea 2 - 1 Manchester United")
mySoccerNetDownloader.downloadGame(files=["1.mkv", "Labels-v2.json"], game = "england_epl/2015-2016/2015-09-12 - 14-45 Everton 3 - 1 Chelsea")
mySoccerNetDownloader.downloadGame(files=["1.mkv", "Labels-v2.json"], game = "england_epl/2015-2016/2015-09-12 - 17-00 Crystal Palace 0 - 1 Manchester City")
mySoccerNetDownloader.downloadGame(files=["1.mkv", "Labels-v2.json"], game = "england_epl/2015-2016/2015-09-19 - 19-30 Manchester City 1 - 2 West Ham")
# mySoccerNetDownloader.downloadGame(files=["1.mkv", "Labels-v2.json"], game = "england_epl/2015-2016/2015-09-20 - 18-00 Southampton 2 - 3 Manchester United")
# mySoccerNetDownloader.downloadGame(files=["1.mkv", "Labels-v2.json"], game = "england_epl/2015-2016/2015-09-26 - 17-00 Leicester 2 - 5 Arsenal")
# mySoccerNetDownloader.downloadGame(files=["1.mkv", "Labels-v2.json"], game = "england_epl/2015-2016/2015-09-26 - 17-00 Manchester United 3 - 0 Sunderland")
# mySoccerNetDownloader.downloadGame(files=["1.mkv", "Labels-v2.json"], game = "england_epl/2015-2016/2015-09-26 - 19-30 Newcastle Utd 2 - 2 Chelsea")
# mySoccerNetDownloader.downloadGame(files=["1.mkv", "Labels-v2.json"], game = "england_epl/2015-2016/2015-10-03 - 17-00 Manchester City 6 - 1 Newcastle Utd")
# mySoccerNetDownloader.downloadGame(files=["1.mkv", "Labels-v2.json"], game = "england_epl/2015-2016/2015-10-03 - 19-30 Chelsea 1 - 3 Southampton")
# mySoccerNetDownloader.downloadGame(files=["1.mkv", "Labels-v2.json"], game = "england_epl/2015-2016/2015-10-17 - 17-00 Chelsea 2 - 0 Aston Villa")
# mySoccerNetDownloader.downloadGame(files=["1.mkv", "Labels-v2.json"], game = "england_epl/2015-2016/2015-10-24 - 17-00 West Ham 2 - 1 Chelsea")
# mySoccerNetDownloader.downloadGame(files=["1.mkv", "Labels-v2.json"], game = "england_epl/2015-2016/2015-11-07 - 18-00 Manchester United 2 - 0 West Brom")
# mySoccerNetDownloader.downloadGame(files=["1.mkv", "Labels-v2.json"], game = "england_epl/2015-2016/2015-11-07 - 20-30 Stoke City 1 - 0 Chelsea")
# mySoccerNetDownloader.downloadGame(files=["1.mkv", "Labels-v2.json"], game = "england_epl/2015-2016/2015-11-08 - 19-00 Arsenal 1 - 1 Tottenham")
# mySoccerNetDownloader.downloadGame(files=["1.mkv", "Labels-v2.json"], game = "england_epl/2015-2016/2015-11-29 - 15-00 Tottenham 0 - 0 Chelsea")
# mySoccerNetDownloader.downloadGame(files=["1.mkv", "Labels-v2.json"], game = "england_epl/2015-2016/2015-12-05 - 20-30 Chelsea 0 - 1 Bournemouth")
# mySoccerNetDownloader.downloadGame(files=["1.mkv", "Labels-v2.json"], game = "england_epl/2015-2016/2015-12-26 - 18-00 Chelsea 2 - 2 Watford")
# mySoccerNetDownloader.downloadGame(files=["1.mkv", "Labels-v2.json"], game = "england_epl/2015-2016/2015-12-26 - 18-00 Manchester City 4 - 1 Sunderland")
# mySoccerNetDownloader.downloadGame(files=["1.mkv", "Labels-v2.json"], game = "england_epl/2015-2016/2015-12-28 - 20-30 Manchester United 0 - 0 Chelsea")
# mySoccerNetDownloader.downloadGame(files=["1.mkv", "Labels-v2.json"], game = "england_epl/2015-2016/2016-01-03 - 16-30 Crystal Palace 0 - 3 Chelsea")
# mySoccerNetDownloader.downloadGame(files=["1.mkv", "Labels-v2.json"], game = "england_epl/2015-2016/2016-01-13 - 22-45 Chelsea 2 - 2 West Brom")
# mySoccerNetDownloader.downloadGame(files=["1.mkv", "Labels-v2.json"], game = "england_epl/2015-2016/2016-01-23 - 20-30 West Ham 2 - 2 Manchester City")
# mySoccerNetDownloader.downloadGame(files=["1.mkv", "Labels-v2.json"], game = "england_epl/2015-2016/2016-01-24 - 19-00 Arsenal 0 - 1 Chelsea")
# mySoccerNetDownloader.downloadGame(files=["1.mkv", "Labels-v2.json"], game = "england_epl/2015-2016/2016-02-03 - 22-45 Watford 0 - 0 Chelsea")
# mySoccerNetDownloader.downloadGame(files=["1.mkv", "Labels-v2.json"], game = "england_epl/2015-2016/2016-02-07 - 19-00 Chelsea 1 - 1 Manchester United")
# mySoccerNetDownloader.downloadGame(files=["1.mkv", "Labels-v2.json"], game = "england_epl/2015-2016/2016-02-13 - 20-30 Chelsea 5 - 1 Newcastle Utd")


#test
mySoccerNetDownloader = SoccerNetDownloader(
    LocalDirectory=dataset_path + "test/SoccerNet")
mySoccerNetDownloader.password = password
mySoccerNetDownloader.downloadGame(files=["1.mkv", "Labels-v2.json"], game = "england_epl/2016-2017/2016-12-27 - 20-15 Liverpool 4 - 1 Stoke City")
mySoccerNetDownloader.downloadGame(files=["1.mkv", "Labels-v2.json"], game = "england_epl/2016-2017/2017-02-27 - 23-00 Leicester 3 - 1 Liverpool")

#val
mySoccerNetDownloader = SoccerNetDownloader(
    LocalDirectory=dataset_path + "val/SoccerNet")
mySoccerNetDownloader.password = password
mySoccerNetDownloader.downloadGame(files=["1.mkv", "Labels-v2.json"], game = "england_epl/2016-2017/2017-01-14 - 20-30 Leicester 0 - 3 Chelsea")
mySoccerNetDownloader.downloadGame(files=["1.mkv", "Labels-v2.json"], game = "england_epl/2015-2016/2015-08-08 - 19-30 Chelsea 2 - 2 Swansea")
