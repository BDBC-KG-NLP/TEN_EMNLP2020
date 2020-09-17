import numpy as np

import matplotlib as mpl
mpl.use('Agg')

import matplotlib.pyplot as plt

# import ternary
    
# scale = 1
# figure, tax = ternary.figure(scale=scale)

# # Draw Boundary and Gridlines
# tax.boundary(linewidth=2.0)
# tax.gridlines(color="blue", multiple=0.2)

# # Set Axis labels and Title
# fontsize = 12
# offset = 0.14
# tax.set_title("Various Lines\n", fontsize=fontsize)
# # tax.right_corner_label("X", fontsize=fontsize)
# # tax.top_corner_label("Y", fontsize=fontsize)
# # tax.left_corner_label("Z", fontsize=fontsize)
# tax.left_axis_label("Food", fontsize=fontsize, offset=offset)
# tax.right_axis_label("Area", fontsize=fontsize, offset=offset)
# tax.bottom_axis_label("Price range", fontsize=fontsize, offset=offset)

# words = ['apple', 'orange', 'apple', 'pear']
# x_coords = [0.1, 0.4, 0.7, 0.2]
# y_coords = [0.7, 0.4, 0.1, 0.3]
# z_coords = [0.2, 0.2, 0.2, 0.5]

# for i, word in enumerate(words):
#     x = x_coords[i]
#     y = y_coords[i]
#     z = z_coords[i]
#     tax.scatter([[x, y, z]], marker='D', color='red', label=word)
#     # tax.text(x, y, word, fontsize=9)

# # points = [[0.3, 0.6, 0.1]]
# # tax.scatter(points, marker='D', color='green', label="Green Diamonds")
# # tax.legend()

# tax.ticks(axis='lbr', multiple=0.2, linewidth=2, offset=0.025, tick_formats='%.1f')
# tax.get_axes().axis('off')
# tax.clear_matplotlib_ticks()
# # tax.show()
# tax.savefig('tri.png')


# pred_basic = np.load('1.npz')['goal'][()]
# pred_fgt = np.load('2.npz')['goal'][()]

# keys = pred_basic.keys()
# lens = len(pred_basic[keys[0]])

# basic = {key:[0, 0] for key in keys}
# fgt = {key:[0, 0] for key in keys}

# for i in range(lens):
# 	for key in keys:
# 		if not pred_basic[key][i]:
# 			basic[key][0] += 1
# 			if pred_fgt[key][i]:
# 				basic[key][1] += 1
# 		if not pred_fgt[key][i]:
# 			fgt[key][0] += 1
# 			if pred_basic[key][i]:
# 				fgt[key][1] += 1

# print basic
# print fgt

# def resilience(ll):
# 	wrong = False
# 	count = 0
# 	for i in ll:
# 		if wrong and i:
# 			count += 1
# 		if not i:
# 			wrong = True
# 	return count

# pred_basic = np.load('3.npz')['dgoal'][()]
# pred_fgt = np.load('4.npz')['dgoal'][()]
# keys = pred_basic.keys()
# lens = len(pred_basic[keys[0]])


# basic = {key: 0 for key in keys}
# fgt = {key: 0 for key in keys}

# for i in range(lens):
# 	for s in keys:
# 		# print pred_basic[s][i]
# 		# if resilience(pred_basic[s][i]):
# 		basic[s] += resilience(pred_basic[s][i])
# 		# if resilience(pred_fgt[s][i]):
# 		fgt[s] += resilience(pred_fgt[s][i])

# print basic
# print fgt

x1 = [175, 87, 28, 69]
x2 = [151, 81, 13, 66]
y1 = [30, 14, 15, 7]
y2 = [6, 8, 0, 4]

label_list = ['joint goal', 'food', 'area', 'price range']
# num_list1 = [20, 30, 15, 35]
# num_list2 = [15, 30, 40, 20]
x = np.arange(len(x1))
width = 0.35
rects1 = plt.bar(left=x, height=x1, width=width, alpha=0.6, color='red', label="TEN-X-wrong")
rects2 = plt.bar(left=x, height=y1, width=width, alpha=0.6, color='green', label="TEN-correct")
rects1 = plt.bar(left=[i + width for i in x], height=x2, width=width, alpha=0.6, color='yellow', label="TEN-wrong")
rects2 = plt.bar(left=[i + width for i in x], height=y2, width=width, alpha=0.6, color='blue', label="TEN-X-correct")
plt.ylim(0, 200)
plt.ylabel("Count", fontsize='large')
plt.xticks(x + width/2, label_list, fontsize='large')
plt.xlabel("Goal", fontsize='large')
# plt.title('T')
plt.legend()
plt.savefig('fig.png')

# pcnn_att = [1086, 592, 620, 825, 1075]
# pcnn_att_em = [821, 415, 524, 751, 1687]
# name_list = ['0.0~0.2', '0.2~0.4', '0.4~0.6', '0.6~0.8', '0.8~1.0']
# # pcnn_att = [0.639, 0.694, 0.580, 0.484, 0.220, 0.223, 0.170]
# # pcnn_att_em = [0.622, 0.843, 0.844, 0.649, 0.482, 0.337, 0.300]
# # name_list = ['NA', 'contains', 'nationality', 'company', 'place_lived', 'place_of_death', 'place_of_birth']	
# x = np.arange(len(pcnn_att))
# total_width, n = 0.6, 2
# width = total_width / n
# x = x - (total_width - width) / 2
# plt.xticks(x + width/2, name_list, rotation=8, fontsize='small', fontweight='semibold')
# plt.bar(x, pcnn_att,  width=width, color='gray', alpha=0.9, label='PCNN+ATT')
# plt.bar(x + width, pcnn_att_em, width=width, color='black', alpha=0.9, label='PCNN+ATT+nEM')
# # plt.xlabel('Relation')
# # plt.ylabel('Average score')
# plt.xlabel('Score region')
# plt.ylabel('Counts')		
# plt.legend(loc="upper left")
# # plt.savefig('compare/ave_score.png')
# plt.savefig('compare/count.png')

