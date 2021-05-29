import axelrod as axl
from axelrod import Game
import pprint
import numpy as np
import math
import scipy.stats
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

matplotlib.use('TkAgg')


"""
This file is used for novelty injection purposes. Novelty is injected by changing the values of R,S,T and P and studying the
effects on agent scores on the Axelrod tournaments.
Prenovelty tournament uses the default values of R,S,T,P such that T>R>P>S where R=6, S=0, T=10, P=1.
Two metrics are computed using the scores in the different post-novelty situations vs pre-novelty case:
    - Per-agent Robustness
    - Global Impact
"""


def flatten_list(mean_std_dict_per_player_comb, players, nov_number):
    matrix = [[[0 for k in range(2)] for i in range(len(players))] for j in range(len(players))]
    count_i = 0
    count_j = 0

    std = [[[0 for k in range(2)] for i in range(len(players))] for j in range(len(players))]

    for k, v in mean_std_dict_per_player_comb.items():
        matrix[count_i][count_j][0] = v[0][0]
        matrix[count_i][count_j][1] = v[0][2]
        std[count_i][count_j][0] = v[0][1]
        std[count_i][count_j][1] = v[0][3]

        matrix[count_j][count_i][0] = v[0][2]
        matrix[count_j][count_i][1] = v[0][0]
        std[count_j][count_i][0] = v[0][3]
        std[count_j][count_i][1] = v[0][1]

        count_j += 1
        if count_j < len(players):
            continue
        else:
            count_i += 1
            count_j = count_i

    final_list = list()
    std_list = list()

    for item in matrix:
        for i in item:
            final_list.append(i[0])
    for item in std:
        for i in item:
            std_list.append(i[0])


    plot_score_matrix = [[0 for i in range(len(players))] for j in range(len(players))]

    for i in range(len(players)):
        for j in range(len(players)):
            plot_score_matrix[i][j] = matrix[i][j][0]

    print("novelty number: " + str(nov_number))
    print(plot_score_matrix)

    agents = ['Cooperator', 'Defector', 'TitForTat', 'Alternator', 'Adaptor', 'Grudger', 'AvgCopier', 'Appeaser', 'FirmButFair', 'FirstByAnonymous',
        'TitFor2Tats', '2TitsForTat', 'DefectorHunter', 'Punisher', 'InvPunisher', 'AdaptorBrief',
        'AdaptorLong', 'AdaptiveTitForTat', 'AntiTitForTat', 'Bully', 'Gradual', 'GradKiller', 'EasyGo',
        'Handshake', 'HardProber', 'ArrogantQLearner', 'CautiousQArrogantQLearner', 'HesitantQArrogantQLearner', 'Ressurection', 'LimRetaliate']

    # plt.imshow(plot_score_matrix)
    # plt.switch_backend('QT4Agg') #default on my system

    fig, ax = plt.subplots()
    ax = sns.heatmap(plot_score_matrix)
    ax.set_xticks(np.arange(len(agents)))
    ax.set_yticks(np.arange(len(agents)))
    ax.set_xticklabels(agents)
    ax.set_yticklabels(agents)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    plt.setp(ax.get_yticklabels(), rotation=360)

    # for i in range(len(agents)):
    #     for j in range(len(agents)):
    #         text = ax.text(j, i, plot_score_matrix[i][j],
    #                        ha="center", va="center", color="w")

    if nov_number == 0:
        ax.set_title("Scores of agents for default case")
    else:
        ax.set_title("Scores of agents for Novelty " + str(nov_number))
    fig.tight_layout()

    plt.show()

    return final_list, std_list


def print_results(lst, players, flag=False):
    for i in range(int(len(lst)/len(players))):
        if flag:
            print(lst[i*len(players):(i+1)*len(players)], np.sum(lst[i*len(players):(i+1)*len(players)]))
        else:
            print(lst[i*len(players):(i+1)*len(players)])


players = [axl.Cooperator(), axl.Defector(), axl.TitForTat(), axl.Alternator(), axl.Adaptive(),
               axl.Grudger(), axl.AverageCopier(), axl.Appeaser(), axl.FirmButFair(), axl.FirstByAnonymous(),
               axl.TitFor2Tats(), axl.TwoTitsForTat(), axl.DefectorHunter(), axl.Punisher(), axl.InversePunisher(),
               axl.AdaptorBrief(), axl.AdaptorLong(), axl.AdaptiveTitForTat(), axl.AntiTitForTat(), axl.Bully(),
               axl.Gradual(), axl.GradualKiller(), axl.EasyGo(), axl.Handshake(), axl.HardProber(),
               axl.ArrogantQLearner(), axl.CautiousQLearner(), axl.HesitantQLearner(), axl.Resurrection(), axl.LimitedRetaliate()]  # Create players

# default pre-novelty game
tournament = axl.Tournament(players, turns=200, repetitions=1, seed=1, game=Game(r=6, s=0, t=10, p=1))  # Create a tournament
results1, mean_std_dict_per_player_comb = tournament.play()  # Play the tournament
# print(mean_std_dict_per_player_comb)
# print(results1.ranked_names)
# print(results1.scores)
mean_list_def, std_list_def = flatten_list(mean_std_dict_per_player_comb, players, 0)


def play_game(R=6000, S=1100, T=10000, P=1, nov_num=0):
    players = [axl.Cooperator(), axl.Defector(), axl.TitForTat(), axl.Alternator(), axl.Adaptive(),
               axl.Grudger(), axl.AverageCopier(), axl.Appeaser(), axl.FirmButFair(), axl.FirstByAnonymous(),
               axl.TitFor2Tats(), axl.TwoTitsForTat(), axl.DefectorHunter(), axl.Punisher(), axl.InversePunisher(),
               axl.AdaptorBrief(), axl.AdaptorLong(), axl.AdaptiveTitForTat(), axl.AntiTitForTat(), axl.Bully(),
               axl.Gradual(), axl.GradualKiller(), axl.EasyGo(), axl.Handshake(), axl.HardProber(),
               axl.ArrogantQLearner(), axl.CautiousQLearner(), axl.HesitantQLearner(), axl.Resurrection(), axl.LimitedRetaliate()]  # Create players
    tournament = axl.Tournament(players, turns=200, repetitions=1, seed=1, game=Game(r=R, s=S, t=T, p=P))  # Create a tournament
    results2, mean_std_dict_per_player_comb = tournament.play()  # Play the tournament
    mean_list_2, std_list_2 = flatten_list(mean_std_dict_per_player_comb, players, nov_num+1)

    a_s = mean_list_def
    b_s = mean_list_2

    a_z = std_list_def
    b_z = std_list_2

    abs_diff_mean_list = list()
    abs_diff_std_list = list()
    # abs_diff_mean_list_2 = list()
    # abs_diff_std_list_2 = list()

    mult = int(math.sqrt(len(a_s)))
    print("Number of agents = ", mult)

    for i in range(mult):
        for j in range(0, i):
            abs_diff_mean_list.append(abs(a_s[i*mult+j] - b_s[i*mult+j]))
            abs_diff_std_list.append(abs(a_z[i*mult+j] - b_z[i*mult+j]))

    # for i in range(mult):
    #     for j in range(i+1, mult):
    #         abs_diff_mean_list_2.append(abs(a_s[i*mult+j] - b_s[i*mult+j]))
    #         abs_diff_std_list_2.append(abs(a_z[i*mult+j] - b_z[i*mult+j]))

    print(len(abs_diff_mean_list))
    print("Mean mean abs diff = ", np.mean(abs_diff_mean_list))
    print("Mean Std abs diff = ", np.std(abs_diff_mean_list))

    print(len(abs_diff_std_list))
    print("Std mean abs diff = ", np.mean(abs_diff_std_list))
    print("Std Std abs diff = ", np.std(abs_diff_std_list))

    # print(len(abs_diff_mean_list))
    # print("Mean mean abs diff 2 = ", np.mean(abs_diff_mean_list_2))
    # print("Mean Std abs diff 2 = ", np.std(abs_diff_mean_list_2))
    #
    # print(len(abs_diff_std_list))
    # print("Std mean abs diff 2 = ", np.mean(abs_diff_std_list_2))
    # print("Std Std abs diff 2 = ", np.std(abs_diff_std_list_2))
    return mean_list_2, std_list_2, np.mean(abs_diff_mean_list), np.std(abs_diff_mean_list)


agent1 = list()
agent2 = list()
agent3 = list()
agent4 = list()
agent5 = list()
agent6 = list()
agent7 = list()
agent8 = list()
agent9 = list()
agent10 = list()
agent11 = list()
agent12 = list()
agent13 = list()
agent14 = list()
agent15 = list()
agent16 = list()
agent17 = list()
agent18 = list()
agent19 = list()
agent20 = list()
agent21 = list()
agent22 = list()
agent23 = list()
agent24 = list()
agent25 = list()
agent26 = list()
agent27 = list()
agent28 = list()
agent29 = list()
agent30 = list()

diff_mean_list = list()
diff_std_list = list()

novelties = [{'R':600, 'S':0, 'T':1000, 'P':100}, {'R':60, 'S':0, 'T':100, 'P':10}, {'R':60, 'S':0, 'T':1000, 'P':6}, {'R':1, 'S':0, 'T':10, 'P':6},
             {'R':1, 'S':10, 'T':0, 'P':6}, {'R':6, 'S':10, 'T':0, 'P':1}, {'R':6000, 'S':20, 'T':1, 'P':10000}, {'R':60, 'S':20000, 'T':1, 'P':100},
             {'R':60, 'S':20000, 'T':20000, 'P':60}, {'R':60000, 'S':20, 'T':20, 'P':60000}, {'R':10, 'S':1, 'T':1, 'P':10}, {'R':100, 'S':100, 'T':100, 'P':100},
             {'R':10, 'S':10, 'T':1000, 'P':1000}, {'R':10, 'S':1000, 'T':10, 'P':1000}, {'R':1000, 'S':1000, 'T':10, 'P':1000}, {'R':1000, 'S':10, 'T':1000, 'P':1000},
             {'R':10, 'S':1000, 'T':1000, 'P':1000}, {'R':1000, 'S':1000, 'T':1000, 'P':10}, {'R':8520, 'S':0, 'T':1011, 'P':102}, {'R':6, 'S':15200, 'T':1110, 'P':1}]

for i in range(len(novelties)):
    n = novelties[i]
    print("nov", n)
    mean_all, std_all, mean_diff, std_diff = play_game(R=n['R'], S=n['S'], T=n['T'], P=n['P'], nov_num=i)
    agent1.append(np.mean(mean_all[1:30]))      # agent1 list of 20 per-agent robustness values
    agent2.append(np.mean(mean_all[30:31]+mean_all[32:60]))
    agent3.append(np.mean(mean_all[60:62]+mean_all[63:90]))
    agent4.append(np.mean(mean_all[90:93]+mean_all[94:120]))
    agent5.append(np.mean(mean_all[120:124]+mean_all[125:150]))
    agent6.append(np.mean(mean_all[150:155]+mean_all[156:180]))
    agent7.append(np.mean(mean_all[180:186]+mean_all[187:210]))
    agent8.append(np.mean(mean_all[210:217]+mean_all[218:240]))
    agent9.append(np.mean(mean_all[240:248]+mean_all[249:270]))
    agent10.append(np.mean(mean_all[270:279]+mean_all[280:300]))
    agent11.append(np.mean(mean_all[300:310]+mean_all[311:330]))
    agent12.append(np.mean(mean_all[330:341]+mean_all[342:360]))
    agent13.append(np.mean(mean_all[360:372]+mean_all[373:390]))
    agent14.append(np.mean(mean_all[390:403]+mean_all[404:420]))
    agent15.append(np.mean(mean_all[420:434]+mean_all[435:450]))
    agent16.append(np.mean(mean_all[450:465]+mean_all[466:480]))
    agent17.append(np.mean(mean_all[480:496]+mean_all[497:510]))
    agent18.append(np.mean(mean_all[510:527]+mean_all[528:540]))
    agent19.append(np.mean(mean_all[540:558]+mean_all[559:570]))
    agent20.append(np.mean(mean_all[570:589]+mean_all[590:600]))
    agent21.append(np.mean(mean_all[600:620]+mean_all[621:630]))
    agent22.append(np.mean(mean_all[630:651]+mean_all[652:660]))
    agent23.append(np.mean(mean_all[660:672]+mean_all[673:690]))
    agent24.append(np.mean(mean_all[690:713]+mean_all[714:720]))
    agent25.append(np.mean(mean_all[720:744]+mean_all[745:750]))
    agent26.append(np.mean(mean_all[750:775]+mean_all[776:780]))
    agent27.append(np.mean(mean_all[780:806]+mean_all[807:810]))
    agent28.append(np.mean(mean_all[810:837]+mean_all[838:840]))
    agent29.append(np.mean(mean_all[840:868]+mean_all[869:870]))
    agent30.append(np.mean(mean_all[870:899]))

    diff_mean_list.append(mean_diff)
    diff_std_list.append(std_diff/math.sqrt(29))

mean_default, std_default, mean_diff_default, std_diff_default = play_game(R=6, S=0, T=10, P=1)
agent1_def = (np.mean(mean_default[1:30]))
agent2_def = (np.mean(mean_default[30:31]+mean_default[32:60]))
agent3_def = (np.mean(mean_default[60:62]+mean_default[63:90]))
agent4_def = (np.mean(mean_default[90:93]+mean_default[94:120]))
agent5_def = (np.mean(mean_default[120:124]+mean_default[125:150]))
agent6_def = (np.mean(mean_default[150:155]+mean_default[156:180]))
agent7_def = (np.mean(mean_default[180:186]+mean_default[187:210]))
agent8_def = (np.mean(mean_default[210:217]+mean_default[218:240]))
agent9_def = (np.mean(mean_default[240:248]+mean_default[249:270]))
agent10_def = (np.mean(mean_default[270:279]+mean_default[280:300]))
agent11_def = (np.mean(mean_default[300:310]+mean_default[311:330]))
agent12_def = (np.mean(mean_default[330:341]+mean_default[342:360]))
agent13_def = (np.mean(mean_default[360:372]+mean_default[373:390]))
agent14_def = (np.mean(mean_default[390:403]+mean_default[404:420]))
agent15_def = (np.mean(mean_default[420:434]+mean_default[435:450]))
agent16_def = (np.mean(mean_default[450:465]+mean_default[466:480]))
agent17_def = (np.mean(mean_default[480:496]+mean_default[497:510]))
agent18_def = (np.mean(mean_default[510:527]+mean_default[528:540]))
agent19_def = (np.mean(mean_default[540:558]+mean_default[559:570]))
agent20_def = (np.mean(mean_default[570:589]+mean_default[590:600]))
agent21_def = (np.mean(mean_default[600:620]+mean_default[621:630]))
agent22_def = (np.mean(mean_default[630:651]+mean_default[652:660]))
agent23_def = (np.mean(mean_default[660:672]+mean_default[673:690]))
agent24_def = (np.mean(mean_default[690:713]+mean_default[714:720]))
agent25_def = (np.mean(mean_default[720:744]+mean_default[745:750]))
agent26_def = (np.mean(mean_default[750:775]+mean_default[776:780]))
agent27_def = (np.mean(mean_default[780:806]+mean_default[807:810]))
agent28_def = (np.mean(mean_default[810:837]+mean_default[838:840]))
agent29_def = (np.mean(mean_default[840:868]+mean_default[869:870]))
agent30_def = (np.mean(mean_default[870:899]))


# Per-agent Robustness
agents = ['0', 'Cooperator', 'Defector', 'TitForTat', 'Alternator', 'Adaptor', 'Grudger', 'AvgCopier', 'Appeaser', 'FirmButFair', 'FirstByAnon',
          'TitFor2Tats', '2TitsForTat', 'DefHunter', 'Punisher', 'InvPunisher', 'AdaptorBrief',
             'AdaptorLong', 'AdapTitForTat', 'AntiTitForTat', 'Bully', 'Gradual', 'GradKiller', 'EasyGo',
          'Handshake', 'HardProber', 'ArrQLearner', 'CautQLearner', 'HesitQLearner', 'Ressurection', 'LimRetaliate']

plt.plot(agents, [0.33, agent1_def, agent2_def, agent3_def, agent4_def, agent5_def, agent6_def, agent7_def, agent8_def, agent9_def, agent10_def,
                  agent11_def, agent12_def, agent13_def, agent14_def, agent15_def, agent16_def, agent17_def, agent18_def, agent19_def, agent20_def,
                  agent21_def, agent22_def, agent23_def, agent24_def, agent25_def, agent26_def, agent27_def, agent28_def, agent29_def, agent30_def], label="Default")

plt.boxplot([agent1, agent2, agent3, agent4, agent5, agent6, agent7, agent8, agent9, agent10,
             agent11, agent12, agent13, agent14, agent15, agent16, agent17, agent18, agent19, agent20,
             agent21, agent22, agent23, agent24, agent25, agent26, agent27, agent28, agent29, agent30], labels=agents[1:])

plt.xlabel("Agents")
plt.ylabel("Robustness number")
plt.xticks(rotation = 60)
plt.legend()
plt.show()

# Global Impact
novelty_names = ['nov1', 'nov2', 'nov3', 'nov4', 'nov5', 'nov6', 'nov7', 'nov8', 'nov9', 'nov10', 'nov11', 'nov12', 'nov13',
                 'nov14', 'nov15', 'nov16', 'nov17', 'nov18', 'nov19', 'nov20']

# plt.plot(novelty_names, diff_mean_list, label='mean')
# plt.plot(novelty_names, diff_std_list, label='std err')
plt.errorbar(novelty_names, diff_mean_list, yerr=diff_std_list)
# plt.legend()
plt.xlabel("Novelties")
plt.ylabel("Global impact value")
plt.xticks(rotation=45)
plt.show()


novelty_perf = list()
for i in range(20):
    temp = [agent1[i], agent2[i], agent3[i], agent4[i], agent5[i], agent6[i], agent7[i], agent8[i], agent9[i], agent10[i],
            agent11[i], agent12[i], agent13[i], agent14[i], agent15[i], agent16[i], agent17[i], agent18[i], agent19[i], agent20[i],
            agent21[i], agent22[i], agent23[i], agent24[i], agent25[i], agent26[i], agent27[i], agent28[i], agent29[i], agent30[i]]

    novelty_perf.append(temp)

print("N1 ", novelty_perf[0])
print("N2 ", novelty_perf[1])
print("N3 ", novelty_perf[2])
print("N4 ", novelty_perf[3])
print("N5 ", novelty_perf[4])
print("N6 ", novelty_perf[5])
print("N7 ", novelty_perf[6])
print("N8 ", novelty_perf[7])
print("N9 ", novelty_perf[8])
print("N10 ", novelty_perf[9])
print("N11 ", novelty_perf[10])
print("N12 ", novelty_perf[11])
print("N13 ", novelty_perf[12])


for i in range(len(novelty_perf)):
    for j in range(i, len(novelty_perf)):
        print(i, j, scipy.stats.ttest_rel(novelty_perf[i], novelty_perf[j]))


print("agent1 ", scipy.stats.ttest_1samp(agent1, agent1_def))
print("agent1 ", scipy.stats.ttest_1samp(agent1[2:], agent1_def))

print("agent2 ", scipy.stats.ttest_1samp(agent2, agent2_def))
print("agent2 ", scipy.stats.ttest_1samp(agent2[2:], agent2_def))

print("agent3 ", scipy.stats.ttest_1samp(agent3, agent3_def))
print("agent3 ", scipy.stats.ttest_1samp(agent3[2:], agent3_def))

print("agent4 ", scipy.stats.ttest_1samp(agent4, agent4_def))
print("agent4 ", scipy.stats.ttest_1samp(agent4[2:], agent4_def))

print("agent5 ", scipy.stats.ttest_1samp(agent5, agent5_def))
print("agent5 ", scipy.stats.ttest_1samp(agent5[2:], agent5_def))

print("agent6 ", scipy.stats.ttest_1samp(agent6, agent6_def))
print("agent6 ", scipy.stats.ttest_1samp(agent6[2:], agent6_def))

print("agent7 ", scipy.stats.ttest_1samp(agent7, agent7_def))
print("agent7 ", scipy.stats.ttest_1samp(agent7[2:], agent7_def))

print("agent8 ", scipy.stats.ttest_1samp(agent8, agent8_def))
print("agent8 ", scipy.stats.ttest_1samp(agent8[2:], agent8_def))

print("agent9 ", scipy.stats.ttest_1samp(agent9, agent9_def))
print("agent9 ", scipy.stats.ttest_1samp(agent9[2:], agent9_def))

print("agent10 ", scipy.stats.ttest_1samp(agent10, agent10_def))
print("agent10 ", scipy.stats.ttest_1samp(agent10[2:], agent10_def))

print("agent11 ", scipy.stats.ttest_1samp(agent11, agent11_def))
print("agent11 ", scipy.stats.ttest_1samp(agent11[2:], agent11_def))

print("agent12 ", scipy.stats.ttest_1samp(agent12, agent12_def))
print("agent12 ", scipy.stats.ttest_1samp(agent12[2:], agent12_def))

print("agent13 ", scipy.stats.ttest_1samp(agent13, agent13_def))
print("agent13 ", scipy.stats.ttest_1samp(agent13[2:], agent13_def))

print("agent14 ", scipy.stats.ttest_1samp(agent14, agent14_def))
print("agent14 ", scipy.stats.ttest_1samp(agent14[2:], agent14_def))

print("agent15 ", scipy.stats.ttest_1samp(agent15, agent15_def))
print("agent15 ", scipy.stats.ttest_1samp(agent15[2:], agent15_def))

print("agent16 ", scipy.stats.ttest_1samp(agent16, agent16_def))
print("agent16 ", scipy.stats.ttest_1samp(agent16[2:], agent16_def))

print("agent17 ", scipy.stats.ttest_1samp(agent17, agent17_def))
print("agent17 ", scipy.stats.ttest_1samp(agent17[2:], agent17_def))

print("agent18 ", scipy.stats.ttest_1samp(agent18, agent18_def))
print("agent18 ", scipy.stats.ttest_1samp(agent18[2:], agent18_def))

print("agent19 ", scipy.stats.ttest_1samp(agent19, agent19_def))
print("agent19 ", scipy.stats.ttest_1samp(agent19[2:], agent19_def))

print("agent20 ", scipy.stats.ttest_1samp(agent20, agent20_def))
print("agent20 ", scipy.stats.ttest_1samp(agent20[2:], agent20_def))

print("agent21 ", scipy.stats.ttest_1samp(agent21, agent21_def))
print("agent21 ", scipy.stats.ttest_1samp(agent21[2:], agent21_def))

print("agent22 ", scipy.stats.ttest_1samp(agent22, agent22_def))
print("agent22 ", scipy.stats.ttest_1samp(agent22[2:], agent22_def))

print("agent23 ", scipy.stats.ttest_1samp(agent23, agent23_def))
print("agent23 ", scipy.stats.ttest_1samp(agent23[2:], agent23_def))

print("agent24 ", scipy.stats.ttest_1samp(agent24, agent24_def))
print("agent24 ", scipy.stats.ttest_1samp(agent24[2:], agent25_def))

print("agent25 ", scipy.stats.ttest_1samp(agent25, agent25_def))
print("agent25 ", scipy.stats.ttest_1samp(agent25[2:], agent25_def))

print("agent26 ", scipy.stats.ttest_1samp(agent26, agent26_def))
print("agent26 ", scipy.stats.ttest_1samp(agent26[2:], agent26_def))

print("agent27 ", scipy.stats.ttest_1samp(agent27, agent27_def))
print("agent27 ", scipy.stats.ttest_1samp(agent27[2:], agent27_def))

print("agent28 ", scipy.stats.ttest_1samp(agent28, agent28_def))
print("agent28 ", scipy.stats.ttest_1samp(agent28[2:], agent28_def))

print("agent29 ", scipy.stats.ttest_1samp(agent29, agent29_def))
print("agent29 ", scipy.stats.ttest_1samp(agent29[2:], agent29_def))

print("agent30 ", scipy.stats.ttest_1samp(agent30, agent30_def))
print("agent30 ", scipy.stats.ttest_1samp(agent30[2:], agent30_def))
