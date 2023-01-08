# NFLBasicPredict
Hello! Thank you for checking out my Python NFL models. 

This code first converts NFL play data to per-game NFL stats with Pandas. It then uses Pytorch feed-forward neural networks and other ML methods from scikit-learn to predict game outcomes. Currently, the NN can predict over 68% of games correctly, excluding the first two games of each season. The random forest predicts 65% of games correctly, including every game of each season.

Specifically, each game is represented as a vector of approximately 40 stats for a given team from all games prior to the one being predicted for the current season and the same 40 stats averaged per-game for the prior season. This is concatenated with similar stats for the opposing team and labeled with a point spread that can be positive or negative. The sign of the point spread can then be converted to a 1 (for win) or 0 (for loss), or kept as is in order to do regression. The order of the two teams can be switched to effectively double the number of games available for training and testing. This is critical, as a lower number of games (and less data) is the biggest difficulty in NFL modelling. 

The vectors can be computed in a variety of ways. The most successful neural network actually excludes the first two games of each season from prediction and excludes the prior season altogether. There is also a weighted representation that gives more weight to more recent games, but this did not produce superior results. The random forest model and other SK-learn models tended to have best results when using information from the current and prior season.
 
Note that, because the dataset used only goes from 2009-2018, this does not predict current NFL games. However, with a dataset in the right format, it should achieve similar performance on more recent games.

To run, the code requires downloading several CSVs of data from other sources (they are not my CSVs) as well as one CSV tabulated manually from other sources. Please contact me for details.
