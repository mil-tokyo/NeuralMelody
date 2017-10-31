import re
import numpy as np
from math import log


def freq_list_to_dict(data):
	dico = {}
	for x in data:
		if x in dico.keys():
			dico[x] += 1
		else:
			dico[x] = 1
	return dico

def barplot_range(data, filename):
	import plotly.plotly as py
	import plotly.graph_objs as go
	dico = freq_list_to_dict(data)
	data_plot = [go.Bar(
	            x=dico.keys(),
	            y=dico.values()
	    )]
	py.plot(data_plot, filename=filename)
	return


def get_reg_range_weights(ixtoword, min_range, max_range):
	# Define measure outside range
	# def f(x):
	# 	return x
	def f(x):
		return log(10+x)
	ixtocost = []
	ranges_note = []
	for ix, word in ixtoword.items():
		if word == '.':
			ixtocost.append(0)
			continue
		note = int(re.split(r";", word)[0])
		ranges_note.append(note)
		if note >= min_range:
			if note <= max_range:
				ixtocost.append(0)
			else:
				ixtocost.append(f(note-max_range))
		else:
			ixtocost.append(f(min_range-note))
	# barplot_range(ranges_note)
	return np.asarray(ixtocost)

def get_reg_range_cost(P, W):
	# P = preds
	# C = \sum_j WjPj
	reg_range_cost = np.sum(np.multiply(P, W), axis=1)
	return reg_range_cost

def get_reg_range_derivative(P, W, C):
	# P = preds
	# W = reg_range_weights
	# C = reg_range_cost
	#
	# dE / dYi = Pi * [Wi - \sum_{j} Pj . Wj]	
	reg_range_d = np.multiply(P, W) - C.reshape([-1,1])*P
	return reg_range_d