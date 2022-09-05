'''
This file try to provide a solution to the dynamic operator optimization problem given a cost model from my_kernel_cost_model.py


given a cost model of micro-kernels, 
1. get candidate micro-kernel shapes
2. get high performance micro-kernel implementation of given shapes
3. compute the residence of each micro-kernel
4. prune the micro-kernels following the rule that the micro-kernel cannot be dominated by others w.r.t. latency and residence
not sure whether there is need to prune micro-kernels

'''


# some notations: TB stands for thread block

from importlib import reload
# import helper_functions
# helper_functions = reload(helper_functions)
from helper_functions import *

import math
import subprocess
import json
import os
import numpy as np
import tvm
from tvm import te, auto_scheduler, topi
from tvm.auto_scheduler import MeasureInput, MeasureResult

import copy
# import my_task_scheduler
# from xgb_model_py import XGBModel
import logging

import sympy
import multiprocessing
import itertools
import time

# import solution2_fine_tune_cost_redefined as eto



logger = logging.getLogger("auto_scheduler")
# os.environ['CUDA_VISIBLE_DEVICES'] = str(1)




def has_enough_factors(v):
	return not((v>10) and (len(get_factors(v)) == 2))




def get_candidate_micK_shapes(op_paras, TBNum_threshold):
	'''
	Compute the candidate micro-kernel shapes for the given op shapes satisfying the thread block number threshold.
	'''
	micK_Sshapes = list()
	for op_para in op_paras:
		out_space = [op_para['loop'][i] for i in op_para['space_iters']]
		out_size = get_product(out_space)
		blk_sizes = list()
		for i in range(hardware_infor.warp_size, out_size // TBNum_threshold + 1):
			if out_size % i == 0:
				blk_shape_dict = get_combinations(i, op_para['space_tile_knobs'])
				blk_shapes =  dict2list(blk_shape_dict, op_para['space_tile_knobs'])
				micK_Sshapes.append(blk_shapes)
	return micK_Sshapes
		



def get_candidate_micK_shapes_from_outshapes(out_shapes, TBNum_threshold, op_type):
	'''
	Compute the candidate micro-kernel shapes for the given op shapes satisfying the thread block number threshold.
	'''
	micK_Sshapes = list()
	if op_type == 'dense':
		mick_Ssizes = list()
		for out_shape in out_shapes:
			out_size = get_product(out_shape)
			for i in range(hardware_infor.warp_size, out_size // TBNum_threshold + 1):
				if out_size % i == 0:
					if i in mick_Ssizes:
						continue
					mick_Ssizes.append(i)
					blk_shape_dict = get_combinations(i, 
						[f"axis{j}" for j in range(len(out_shape))])
					blk_shapes =  dict2list(blk_shape_dict, 
						[f"axis{j}" for j in range(len(out_shape))])
					micK_Sshapes.append(blk_shapes)
	elif op_type == 'conv2d':
		mick_Ssizes = list()
		for out_shape in out_shapes:
			out_size = get_product(out_shape)
			for i in range(hardware_infor.warp_size, out_size // TBNum_threshold + 1):
				if out_size % i == 0:
					if i in mick_Ssizes:
						continue
					mick_Ssizes.append(i)
					blk_shape_dict = get_combinations(i, 
						[f"axis{j}" for j in range(len(out_shape))])
					blk_shapes =  dict2list(blk_shape_dict, 
						[f"axis{j}" for j in range(len(out_shape))])
					# prune shapes that are longer than out_shape in any dimension
					blk_shapes_saved = list()
					for s in blk_shapes:
						if True not in [out_shape[idx] < s[idx] for idx in range(len(s))]:
							blk_shapes_saved.append(s)
					micK_Sshapes.append(blk_shapes_saved)
	return micK_Sshapes




def get_candidate_micK_Sshapes_from_max_outshape(out_shape, op_type, looplen_ratio = 1):
	'''
	Compute the candidate micro-kernel spatial shapes for the given op shapes satisfying a few pruning rules.
	INPUT:
		looplen_ratio: the ratio between the mick loop and the max op shape in the last pruning rule.
	'''
	micK_Sshapes = list()
	out_size = get_product(out_shape)
	for i in range(4*hardware_infor.warp_size, min(26043, looplen_ratio*out_size+1), 4*hardware_infor.warp_size): # change to step size 4*32=128
		if max(factorint(i).keys()) > 5:
			continue
		blk_shapes = None
		if len(out_shape) == 1:
			blk_shapes = [[i,]]
		else:		
			blk_shape_dict = get_combinations(i, 
				[f"axis{j}" for j in range(len(out_shape))])
			blk_shapes =  dict2list(blk_shape_dict, 
				[f"axis{j}" for j in range(len(out_shape))])
		# prune some bad mick Sshapes
		blk_shapes_saved = list()
		for sp in blk_shapes:
			if (op_type in ['conv2d', 'bmm', 'bmm_nn'] and sp[0]>=4):
				continue
			# if (False not in [has_enough_factors(v) for v in sp]):
			if (False not in [sp[j] <= looplen_ratio*out_shape[j] for j in range(len(out_shape))]):
				blk_shapes_saved.append(sp)
		micK_Sshapes.append(blk_shapes_saved)
	return micK_Sshapes





def get_candidate_micK_Rshapes_from_max_reducshape(reduc_shape, op_type, looplen_ratio = 1):
	'''
	Compute the candidate micro-kernel reduction shapes for the given op shapes satisfying a few pruning rules.
	INPUT:
		looplen_ratio: the ratio between the mick loop and the max op shape in the last pruning rule.
	Note:
		For conv2d operators, we assume the rh, rw axes should not be dynamic. 
	'''
	micK_Sshapes = list()
	reduc_size = get_product(reduc_shape)
	if op_type == 'conv2d':
		_, rh, rw = reduc_shape
		max_Rsize = min([math.ceil(Rsize_limit/rh/rw)*rh*rw for Rsize_limit in [128, looplen_ratio*reduc_size]])
		for i in range(1, max_Rsize//(rh*rw), 1):
			if (i > 1) and (max(factorint(i).keys()) > 5):
				continue
			blk_shapes = [[i, rh, rw]]
			# prune some bad mick Sshapes
			blk_shapes_saved = list()
			for sp in blk_shapes:
				# if (False not in [has_enough_factors(v) for v in sp]):
				if (False not in [sp[j] <= looplen_ratio*reduc_shape[j] for j in range(len(reduc_shape))]):
					blk_shapes_saved.append(sp)
			micK_Sshapes.append(blk_shapes_saved)
		return micK_Sshapes	
	# 
	max_Rsize = min(128, looplen_ratio*reduc_size)
	for i in range(1, max_Rsize, 1):
		if (i > 1) and (max(factorint(i).keys()) > 5):
			continue
		blk_shapes = None
		if len(reduc_shape) == 1:
			blk_shapes = [[i,]]
		else:
			blk_shape_dict = get_combinations(i, 
				[f"axis{j}" for j in range(len(reduc_shape))])
			blk_shapes =  dict2list(blk_shape_dict, 
				[f"axis{j}" for j in range(len(reduc_shape))])
		# prune some bad mick Sshapes
		blk_shapes_saved = list()
		for sp in blk_shapes:
			# if (False not in [has_enough_factors(v) for v in sp]):
			if (False not in [sp[j] <= looplen_ratio*reduc_shape[j] for j in range(len(reduc_shape))]):
				blk_shapes_saved.append(sp)
		micK_Sshapes.append(blk_shapes_saved)
	return micK_Sshapes



# def get_search_policy_params(params=None):
# 	if params is None:
# 		params = auto_scheduler.SketchPolicy.DEFAULT_PARAMS
# 	else:
# 		for key,  value in auto_scheduler.SketchPolicy.DEFAULT_PARAMS.items():
# 			if key not in params:
# 				params[key] = value
# 	return params





# search_policy_params = get_search_policy_params()





# def my_search_one_round(search_policy_params, num_random_states, random_states):
# 	// Get parameters
# 	population = search_policy_params["evolutionary_search_population"]
# 	num_use_measured = min(
# 	  static_cast<int>(measured_states_vector_.size()),
# 	  static_cast<int>(
# 	      GetDoubleParam(params, SketchParamKey::SampleInitPopulation::use_measured_ratio) *
# 	      population));

# 	// 1. Generate sketches
# 	if (sketch_cache_.empty()) {
# 	sketch_cache_ = GenerateSketches();
# 	}

# 	// 2. Sample the init population
# 	Array<State> init_population = SampleInitPopulation(sketch_cache_);

# 	// 3. Perform evolutionary search.
# 	// Also insert already measured good states to the initial population
# 	std::vector<int> indices = Argsort(measured_states_throughputs_);
# 	for (int i = 0; i < num_use_measured; i++) {
# 	init_population.push_back(measured_states_vector_[indices[i]]);
# 	}
# 	// Sample some random states for eps-greedy
# 	if (num_random_states > 0 && random_states != nullptr) {
# 	*random_states = RandomSampleStates(init_population, &rand_gen, num_random_states);
# 	}
# 	return EvolutionarySearch(init_population, num_measure_per_iter_ * 2);





def get_search_policy(tasks, tune_option, tune_mick = False):
	search_policy = 'sketch.xgb'
	if tune_mick:
		search_policy_params = {
		"limit_blk_num" : 1,
		"limit_fetch_num" : -1}
	else:
		search_policy_params = None
	load_model_file = "xgb.pkl"
	load_log_file = None
	adapative_training = False
	disable_cost_model_update = True
	few_shot_learning='base_only'
	policies = my_task_scheduler.make_search_policies(
            search_policy,
            search_policy_params,
            tasks,
            tune_option.num_measures_per_round,
            tune_option.verbose,
            load_model_file,
            load_log_file,
            adapative_training,
            disable_cost_model_update,
            few_shot_learning,
        )
	return policies




def get_cost_model(tune_option):
	disable_cost_model_update = True
	few_shot_learning='base_only'
	load_model_file = "xgb.pkl"
	cost_model = XGBModel(
		num_warmup_sample = tune_option.num_measures_per_round,
		disable_update = disable_cost_model_update,
		few_shot_learning = few_shot_learning
	)
	logger.info("TaskScheduler: Load pretrained model...")
	cost_model.load(load_model_file)
	return cost_model







def measure_inputs(tune_option, task, inputs):
    # results->clear();
    # results->reserve(inputs.size());
    # 
    # // Call builder and runner
    # Array<BuildResult> build_res_batch = builder->Build(inputs, verbose);
    # Array<MeasureResult> result_batch = runner->Run(inputs, build_res_batch, verbose);
    build_res_batch = tune_option.builder.build(inputs, tune_option.verbose)
    result_batch = tune_option.runner.run(inputs, build_res_batch, tune_option.verbose)
    # 
    # // Store result batch
    # for (auto& res : result_batch) {
    # results->push_back(res);
    return result_batch




def measure_states(tune_option, task, states):
	# results->clear();
	# results->reserve(inputs.size());
	# 
	# // Call builder and runner
	# Array<BuildResult> build_res_batch = builder->Build(inputs, verbose);
	# Array<MeasureResult> result_batch = runner->Run(inputs, build_res_batch, verbose);
	result_batch = None
	try:
		inputs = [MeasureInput(task, state) for state in states]
		build_res_batch = tune_option.builder.build(inputs, tune_option.verbose)
		result_batch = tune_option.runner.run(inputs, build_res_batch, tune_option.verbose)
	except Exception as e:
		print(e)
		result_batch = None
	# 
	# // Store result batch
	# for (auto& res : result_batch) {
	# results->push_back(res);
	return result_batch




def build_states_for_diff_tasks_(tune_option, tasks, states):
	try:
		inputs = list()
		for task, state in zip(tasks, states):
			inputs.append(MeasureInput(task, state))
		# inputs = [MeasureInput(task, state) for state in states]
		build_res_batch = tune_option.builder.build(inputs, tune_option.verbose)
		return inputs, build_res_batch
	except Exception as e:
		print(e)
		return (None, None)


def build_states_for_diff_tasks(tune_option, tasks, states):
	ret = build_states_for_diff_tasks_(tune_option, tasks, states)
	if ret != (None, None):
		return ret
	build_res_batch = list()
	inputs = list()
	for task, state in zip(tasks, states):
		inp, build_res = build_states_for_diff_tasks_(tune_option, [task], [state])
		if inp==None:
			inp, build_res = [None], [None]
		inputs = inputs + inp
		build_res_batch = build_res_batch + build_res
	return inputs, build_res_batch



def run_built_states_(tune_option, inputs, build_res_batch):
	try:
		return tune_option.runner.run(inputs, build_res_batch, tune_option.verbose)
	except Exception as e:
		print(e)
		return None


# def run_built_states(tune_option, inputs, build_res_batch):
# 	try:
# 		return run_built_states_(tune_option, inputs, build_res_batch)
# 	except Exception as e:
# 		print(e)
# 		results = list()
# 		for inp, build_res in zip(inputs, build_res_batch):
# 			results = results + run_built_states_(tune_option, [inp], [build_res])
# 		return results



def measure_states_for_diff_tasks(tune_option, tasks, states):
	# results->clear();
	# results->reserve(inputs.size());
	# 
	# // Call builder and runner
	# Array<BuildResult> build_res_batch = builder->Build(inputs, verbose);
	# Array<MeasureResult> result_batch = runner->Run(inputs, build_res_batch, verbose);
	result_batch = None
	try:
		inputs = list()
		for task, state in zip(tasks, states):
			inputs.append(MeasureInput(task, state))
		# inputs = [MeasureInput(task, state) for state in states]
		build_res_batch = tune_option.builder.build(inputs, tune_option.verbose)
		result_batch = tune_option.runner.run(inputs, build_res_batch, tune_option.verbose)
	except Exception as e:
		print(e)
		result_batch = None
	# 
	# // Store result batch
	# for (auto& res : result_batch) {
	# results->push_back(res);
	return result_batch




def fake_tuning(task, tuner = "xgbPretrain", target="cuda", tune_mick = False):
	'''
	This function is used to tune a task, 
	the difference is that this function will not do real hardware measurement.
	INPUT: SearchTask
	OUTPUT: the best state and the corr. normalized throughput, the real latency
	'''
	# log_file = "conv2d.json"
	measure_ctx = auto_scheduler.LocalRPCMeasureContext(min_repeat_ms=300)
	tune_option = auto_scheduler.TuningOptions(
	    num_measure_trials=1000,  # change this to 1000 to achieve the best performance
	    runner=measure_ctx.runner,
	    # measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
	    verbose=0,
	)
	policy = get_search_policy([task], tune_option, tune_mick)[0]
	cost_model = get_cost_model(tune_option)
	measurer = auto_scheduler.measure.ProgramMeasurer(
				tune_option.builder, tune_option.runner,
				tune_option.measure_callbacks, tune_option.verbose
				)
	best_state = policy.search_without_measure(tune_option.num_measure_trials, 
				tune_option.early_stopping,
				tune_option.num_measures_per_round, 
				measurer)
	return best_state, cost_model.predict(task, [best_state]), measure_states(tune_option, task, [best_state])[0]





# @auto_scheduler.register_workload
# def dense_layer(X_shape, Y_shape):
# 	X = te.placeholder(X_shape)#, name="X")
# 	Y = te.placeholder(Y_shape)#, name="Y")
# 	out = topi.nn.dense(X, Y, bias=None, out_dtype=None, auto_scheduler_rewritten_layout="")
# 	return [X, Y, out]




# @auto_scheduler.register_workload
# def dense_layer(X_shape, Y_shape):
# 	X = te.placeholder(X_shape, name="X")
# 	Y = te.placeholder(Y_shape, name="Y")
# 	out = topi.nn.dense(X, Y, bias=None, out_dtype=None, auto_scheduler_rewritten_layout="")
# 	return [X, Y, out]


# @auto_scheduler.register_workload
# def conv2d_nchw(data_shape, kernel_shape, stride, padding, dilation_val, out_dtype):
# 	data = te.placeholder(data_shape)#, name="data")
# 	kernel = te.placeholder(kernel_shape)#, name="kernel")
# 	# bias = te.placeholder((1, CO, 1, 1), name="bias")
# 	conv = topi.nn.conv2d_nchw(data, kernel, stride, padding, dilation=dilation_val, out_dtype=out_dtype)
# 	# out = topi.nn.relu(conv + bias)
# 	return [data, kernel, conv]





# @auto_scheduler.register_workload
# def batch_matmul(X_shape, Y_shape, oshape = None):
# 	X = te.placeholder(X_shape, name="X")
# 	Y = te.placeholder(Y_shape, name="Y")
# 	out = topi.nn.batch_matmul(X, Y, oshape=None, auto_scheduler_rewritten_layout="")
# 	return [X, Y, out]





def get_task_log_file_name(workload_key, tuner = "ansor", target = "cuda", kernel_type = "op", diff_measure_task_wlk="", soft_unroll=False):
	if not soft_unroll:
		if diff_measure_task_wlk == "":
			return f"tuning_log_hub/({workload_key},{target},{tuner},{kernel_type}).json".replace(" ","")
		else:
			return f"tuning_log_hub/({workload_key},{target},{tuner},{kernel_type},{diff_measure_task_wlk}).json".replace(" ","")
	else:
		if diff_measure_task_wlk == "":
			return f"tuning_log_hub/({workload_key},{target},{tuner},{kernel_type},SoftUnroll).json".replace(" ","")
		else:
			return f"tuning_log_hub/({workload_key},{target},{tuner},{kernel_type},{diff_measure_task_wlk},SoftUnroll).json".replace(" ","")




def get_task_log_file_name_old(workload_key, tuner = "ansor", target = "cuda", kernel_type = "op"):
	return f"tuning_log_hub/({workload_key},{target},{tuner}).json".replace(" ","")



def mick_shape_to_str(mick_shape):
	ret = ""
	for l in mick_shape:
		ret = ret + str(l)+" "*(5-len(str(l)))
	return ret




def tune_ops_ansor(mick_shapes, micks, tasks = None, 
	tune_mick = False, limit_fetch_num = False, limit_tileL = False):
	target = tvm.target.Target("cuda")
	kernel_type = "op"
	search_policy_params = None
	if tune_mick:
		kernel_type = "micK"
		search_policy_params = {
		"limit_blk_num" : 1,
		"limit_fetch_num" : -1}
		if limit_fetch_num:
			search_policy_params["limit_fetch_num"] = 1
			kernel_type = "micK1fetch"
		if limit_tileL:
			search_policy_params["limit_tileL"] = 1
	# 
	if tasks == None:
		tasks = list()
		for T in [5, 24, 43, 62, 81, 100, 119, 128]:
			X_shape = (16*T, 768)
			Y_shape = (2304, 768)
			task = auto_scheduler.SearchTask(
			    func=dense_layer, args=(X_shape, Y_shape), target=target
			)
			tasks.append(task)
			print(task.workload_key)
	# for task, mick_shape in zip(tasks, mick_shapes):
	for i in range(len(tasks)):
		task = tasks[i]
		mick_shape = mick_shapes[i]
		if 'conv2d_nchw' in task.workload_key:
			mick_shape = mick_shape[:7]
		mick = micks[i]
		# log_file = "({},cuda,ansor).json".format(task.workload_key).replace(" ","")
		log_file = get_task_log_file_name(mick.workload_key, tuner = "ansor",target = "cuda", kernel_type=kernel_type,
											diff_measure_task_wlk=task.workload_key)
		if os.path.exists(log_file):
			continue
		measure_ctx = auto_scheduler.LocalRPCMeasureContext(min_repeat_ms=300)
		tune_option = auto_scheduler.TuningOptions(
		    num_measure_trials=512,  # change this to 1000 to achieve the best performance
		    runner=measure_ctx.runner,
		    measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
		    verbose=2,
		)
		# task.tune(tune_option)
		cost_model = auto_scheduler.cost_model.XGBModel()
		mick_sp_str = mick_shape_to_str(mick_shape)
		search_policy_params["mick_shape"] = mick_sp_str
		search_policy = auto_scheduler.SketchPolicy(task, cost_model, search_policy_params)
		measurer = auto_scheduler.measure.ProgramMeasurer(
				tune_option.builder, tune_option.runner,
				tune_option.measure_callbacks, tune_option.verbose
				)
		best_state = search_policy.search(tune_option.num_measure_trials, 
				tune_option.early_stopping,
				tune_option.num_measures_per_round, 
				measurer)
        






def tune_ops_tensetPretrain():
	target = tvm.target.Target("cuda")
	tasks = list()
	ret = dict()
	for T in [5, 24, 43, 62, 81, 100, 119, 128]:
		X_shape = (16*T, 768)
		Y_shape = (2304, 768)
		task = auto_scheduler.SearchTask(
		    func=dense_layer, args=(X_shape, Y_shape), target=target
		)
		tasks.append(task)
		print(task.workload_key)
	for task in tasks:
		best_state, pred_score, measure_result = \
			fake_tuning(task, tuner = "xgbPretrain", target="cuda")
		ret[task.workload_key] = (best_state, pred_score, measure_result)
	return ret




def get_out_shapes(op_type):
	out_shapes = None
	if op_type == 'dense':
		out_shapes = [(16*T, 2304) for T in 
			[5, 24, 43, 62, 81, 100, 119, 128]]
	elif op_type == 'conv2d':
		out_shapes = [(1, 32, 16*T, 1080) for T in 
			[5, 24, 43, 62, 81, 100, 119, 128]]
	return out_shapes





def get_Ssp_black_dict(op_type):
	'''
	sdsd
	'''
	Ssp_len = 2
	Rsp_len = 1
	other_params = []
	if op_type in ['bmm', 'bmm_nn']:
		Ssp_len = 3
	elif op_type == 'conv2d':
		Ssp_len = 4
		Rsp_len = 3
		other_params = [(1,1),0,(1,1)] # stride, padding, dilation (any other_params is ok for computing stG)
	# 
	max_out_shape = [26043 for _ in range(Ssp_len)]
	Rsp = [1 for _ in range(Rsp_len)]
	Ssp_groups = get_candidate_micK_Sshapes_from_max_outshape(max_out_shape, op_type, looplen_ratio = 1)
	black_dict = dict()
	for Ssps in Ssp_groups:
		# all Ssps in the same group are of the same block size, i.e., Ssize
		tot_stG_list = [get_feature_for_mick_shape(Ssp + Rsp + other_params, op_type, repl=None) \
							for Ssp in Ssps]
		min_tot_stG = min(tot_stG_list)
		for Sspi, Ssp in enumerate(Ssps):
			if tot_stG_list[Sspi] > min_tot_stG:
				if get_product(Ssp) not in black_dict:
					black_dict[get_product(Ssp)] = list()
				black_dict[get_product(Ssp)].append(Ssp)
	# then we store the black_dict to a file
	with open("Ssp_black_dict_pyfile.py", 'a') as f:
		f.write(f"def get_Ssp_black_dict_{op_type}():\n\treturn {black_dict}\n\n\n")




def filter_mick_shapes_worker_old(firstSspId, common_params):
	'''
		Prune mick shapes whose data read amount is out of shared memory limit. 
		Also prune mick shapes whose stG number is larger than the min stG number of the corresponding Ssize.
	'''
	mick_shapes = list()
	SshapeParaNum, Ssp_groups, Rsps, op_type, other_params = common_params
	SspGrpNum = len(Ssp_groups)
	for Ssp_grpi in range(firstSspId, min(firstSspId + SshapeParaNum, SspGrpNum)):
		# all Ssps in the same group are of the same block size, i.e., Ssize
		tot_stG_list = [get_feature_for_mick_shape(Ssp + Rsps[0] + other_params, op_type, repl=None) \
							for Ssp in Ssp_groups[Ssp_grpi]]
		min_tot_stG = min(tot_stG_list)
		for Sspi, Ssp in enumerate(Ssp_groups[Ssp_grpi]):
			if tot_stG_list[Sspi] > min_tot_stG:
				continue
			for Rsp in Rsps:
				msp = Ssp + Rsp + other_params
				if data_read_amount_PerBlk(op_type, msp) <= (49152 / 4):
					mick_shapes.append(msp)
	return mick_shapes	
	# 
	# below is the code for flattened Ssps, and do not consider the stG feature
	mick_shapes = list()
	SshapeParaNum, Ssps, Rsps, op_type, other_params = common_params
	SspsNum = len(Ssps)
	for Sspi in range(firstSspId, min(firstSspId + SshapeParaNum, SspsNum)):
		Ssp = Ssps[Sspi]
		for Rsp in Rsps:
			msp = Ssp + Rsp + other_params
			if data_read_amount_PerBlk(op_type, msp) <= (49152 / 4):
				mick_shapes.append(msp)
	return mick_shapes
	# 
	# if op_type == 'dense':
	# 	for Sspi in range(firstSspId, min(firstSspId + SshapeParaNum, SspsNum)):
	# 		Ssp = Ssps[Sspi]
	# 		for Rsp in Rsps:
	# 			if data_read_amount_PerBlk(op_type, Ssp, rc = Rsp[0]) <= (49152 / 4):
	# 				mick_shapes.append(Ssp + Rsp)
	# elif op_type == 'conv2d':
	# 	stride, padding, dilation = other_params
	# 	for Sspi in range(firstSspId, min(firstSspId + SshapeParaNum, SspsNum)):
	# 		Ssp = Ssps[Sspi]
	# 		for Rsp in Rsps:
	# 			rc, rh, rw = Rsp
	# 			if data_read_amount_PerBlk(op_type, Ssp, rc = rc, rh = rh, rw = rw, stride=stride, dilation=dilation) <= (49152 / 4):
	# 				mick_shapes.append(Ssp + Rsp + other_params)
	# elif op_type == 'bmm':
	# 	for Sspi in range(firstSspId, min(firstSspId + SshapeParaNum, SspsNum)):
	# 		Ssp = Ssps[Sspi]
	# 		for Rsp in Rsps:
	# 			if data_read_amount_PerBlk(op_type, Ssp, rc = Rsp[0]) <= (49152 / 4):
	# 				mick_shapes.append(Ssp + Rsp)	
	# return mick_shapes




def filter_mick_shapes_worker(firstSspId, common_params):
	'''
		Prune mick shapes whose data read amount is out of shared memory limit. 
		Also prune mick shapes whose stG number is larger than the min stG number of the corresponding Ssize.
		We return the filted msps and also the index range of msps with the same Ssp.
	'''
	# below is the code for flattened Ssps, and do not consider the stG feature
	mick_shapes = list()
	Ssp_poses = dict()
	SshapeParaNum, Ssps, Rsps, op_type, other_params, Ssp_black_dict = common_params
	SspsNum = len(Ssps)
	for Sspi in range(firstSspId, min(firstSspId + SshapeParaNum, SspsNum)):
		Ssp = Ssps[Sspi]
		if (get_product(Ssp) in Ssp_black_dict) and (Ssp in Ssp_black_dict[get_product(Ssp)]):
			continue
		start_i = len(mick_shapes)
		for Rsp in Rsps:
			msp = Ssp + Rsp + other_params
			if data_read_amount_PerBlk(op_type, msp) <= (49152 / 4):
				mick_shapes.append(msp)
		end_i = len(mick_shapes)
		Ssp_poses[tuple(Ssp)] = (start_i, end_i)
	return (mick_shapes, Ssp_poses)





def get_candidate_micro_ops_N_shapes(max_out_shape, max_reduc_shape, op_type, looplen_ratio = 1, other_params = list()):
	'''
		Note: This function now will not return candidate micro-kernel op instances, i.e., all_micK_ops is an empty list.
		INPUT:
			looplen_ratio: the ratio between the mick loop and the max op shape in the last pruning rule.
			other_params: the parameters like stride, padding, dilation for conv2d.
	'''
	all_micK_ops = list()
	# micK_Sshapes = get_candidate_micK_shapes_from_outshapes(out_shapes, 1*SMNum, op_type=op_type)
	micK_Sshapes = get_candidate_micK_Sshapes_from_max_outshape(max_out_shape, op_type, looplen_ratio)
	micK_Rshapes = get_candidate_micK_Rshapes_from_max_reducshape(max_reduc_shape, op_type, looplen_ratio)
	print("total number of micro-kernel space shapes: ", sum([len(j) for j in micK_Sshapes]))
	print("total number of micro-kernel reduction shapes: ", sum([len(j) for j in micK_Rshapes]))
	Sshapes, Rshapes = list(), list()
	# we do not flatten Ssps here
	# Sshapes = micK_Sshapes
	# we do flatten Ssps here
	for i in range(len(micK_Sshapes)):
		Sshapes = Sshapes + sorted(micK_Sshapes[i])
	for i in range(len(micK_Rshapes)):
		Rshapes = Rshapes + sorted(micK_Rshapes[i])
	# combine Sshapes and Rshapes into complete shapes
	Ssp_black_dict = None
	if op_type == 'dense':
		Ssp_black_dict = Ssp_black_dict_pyfile.get_Ssp_black_dict_dense()
	elif op_type == 'bmm':
		Ssp_black_dict = Ssp_black_dict_pyfile.get_Ssp_black_dict_bmm()
	elif op_type == 'bmm_nn':
		Ssp_black_dict = Ssp_black_dict_pyfile.get_Ssp_black_dict_bmm_nn()
	elif op_type == 'conv2d':
		Ssp_black_dict = Ssp_black_dict_pyfile.get_Ssp_black_dict_conv2d()
	# 
	tot_micK_Sshapes_num = len(Sshapes)
	workerNum = 240
	SshapeParaNum = math.ceil(tot_micK_Sshapes_num/workerNum)
	firstSspIds = list(range(0, tot_micK_Sshapes_num, SshapeParaNum))
	common_params = SshapeParaNum, Sshapes, Rshapes, op_type, other_params, Ssp_black_dict
	with multiprocessing.Pool(processes=workerNum) as pool:
		filter_res_list = pool.starmap(filter_mick_shapes_worker, zip(firstSspIds, itertools.repeat(common_params)))
	# 
	mick_shapes = list()
	Ssp_poses = dict()
	for tmp_sp_list, tmp_Ssp_poses in filter_res_list:
		# correct the Ssp poses with the offset of the tmp_sp_list in mick_shapes
		offset = len(mick_shapes)
		for Ssp, poses in tmp_Ssp_poses.items():
			Ssp_poses[Ssp] = [pos+offset for pos in poses]
		mick_shapes = mick_shapes + tmp_sp_list
	# 
	# 
	# mick_shapes = list()
	# for Sshapes in micK_Sshapes:
	# 	for Rshapes in micK_Rshapes:
	# 		mick_shapes = mick_shapes + [sp[0]+sp[1] for sp in itertools.product(Sshapes, Rshapes)]
	print("total number of micro-kernel shapes: ", len(mick_shapes))
	print("total number of Ssps: ", len(Ssp_poses))
	return all_micK_ops, mick_shapes, Ssp_poses # micK_Sshapes
	# 



def fake_tune_mick_ops(all_micK_ops, all_micK_Sshapes, task_set_id, tuner = "xgbPretrain", targetstr="cuda"):
	count=0
	SMNum = 108
	kernel_type="micK"
	tot_op_num = sum([len(i) for i in all_micK_ops])
	target = tvm.target.Target("cuda")
	print(f"total ops to be measured: {tot_op_num}")
	for mick_op_group_i in range(len(all_micK_ops)):
		mick_op_group = all_micK_ops[mick_op_group_i]
		mick_Sshape_group = all_micK_Sshapes[mick_op_group_i]
		# 
		for task_i in range(len(mick_op_group)):
			task = mick_op_group[task_i]
			micK_Sshape = mick_Sshape_group[task_i]
			count+=1
			# print(count)
			# 
			if (count <= math.ceil(tot_op_num / 4) * task_set_id) \
				or (count > math.ceil(tot_op_num / 4) * (1+task_set_id)):
				continue
			print(task.workload_key, micK_Sshape)
			best_state, pred_score, measure_result = \
				fake_tuning(task, tuner, targetstr, tune_mick=True)
			print(pred_score, measure_result)
			log_file = get_task_log_file_name(task.workload_key, tuner, targetstr, kernel_type)
			auto_scheduler.save_records(log_file, [MeasureInput(task, best_state)], [measure_result])
			# 
			# measure the kernels with multiple such micro-kernel
			for n in range(1, 5):
				task = auto_scheduler.SearchTask(
				    func=dense_layer, args=((micK_Sshape[0], 768), (micK_Sshape[1]*(SMNum*n+1), 768)), target=target
				)
				print(task.workload_key)
				measure_ctx = auto_scheduler.LocalRPCMeasureContext(min_repeat_ms=300)
				tune_option = auto_scheduler.TuningOptions(
				    num_measure_trials=1000,  # change this to 1000 to achieve the best performance
				    runner=measure_ctx.runner,
				    # measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
				    verbose=2,
				)
				auto_scheduler.save_records(log_file, [MeasureInput(task, best_state)], 
					measure_states(tune_option, task, [best_state]))





def get_inp_shapes(workload_key, op_type):
	if op_type == 'dense':
		X_shape = json.loads(workload_key)[1]
		Y_shape = json.loads(workload_key)[2]
		return X_shape, Y_shape
	elif op_type in ['bmm', 'bmm_nn']:
		X_shape = json.loads(workload_key)[1]
		Y_shape = json.loads(workload_key)[2]
		return X_shape, Y_shape
	elif op_type == 'conv2d':
		data_shape = json.loads(workload_key)[1]
		kernel_shape = json.loads(workload_key)[2]
		return data_shape, kernel_shape



def get_inp_shapes_from_paramters(out_shape, op_type, params):
	if op_type == 'conv2d':
		rc, rh, rw, stride, padding, dilation = params
		data_shape = (out_shape[0], rc, stride[0]*(out_shape[2]-1)+dilation[0]*(rh-1)+1, stride[1]*(out_shape[3]-1)+dilation[1]*(rw-1)+1,)
		kernel_shape = (out_shape[1], rc, rh, rw)
		return data_shape, kernel_shape




def get_inp_shapes_from_sp(sp, op_type):
	if op_type == 'dense':
		X_shape = (sp[0], sp[2])
		Y_shape = (sp[1], sp[2])
		return X_shape, Y_shape
	elif op_type == 'bmm':
		X_shape = (sp[0], sp[1], sp[3])
		Y_shape = (sp[0], sp[2], sp[3])
		return X_shape, Y_shape
	elif op_type == 'bmm_nn':
		X_shape = (sp[0], sp[1], sp[3])
		Y_shape = (sp[0], sp[3], sp[2])
		return X_shape, Y_shape
	elif op_type == 'conv2d':
		return get_inp_shapes_from_paramters(sp[:4], op_type, sp[4:])



def get_best_micK_for_op(micK_latency_dict, task):
	'''return the padded task as well to apply the micro-kernel'''
	SMNum = 108
	X_shape, Y_shape = get_inp_shapes(task.workload_key)
	#
	out_shape = (X_shape[0], Y_shape[0])
	best_cost = 1e10
	best_mick = None
	best_padded_outshape = None
	# histry = list()
	for s_key, micK_costs in micK_latency_dict.items(): 
		if len(micK_costs) < 5:
			continue
		if 10000000000.0 in micK_costs:
			continue
		micK_Sshape = s_key[:2]
		blk_num = get_product([math.ceil(out_shape[i] / micK_Sshape[i]) for i in range(len(out_shape))])
		# print(out_shape, micK_Sshape, blk_num)
		cost = None
		if blk_num // SMNum < 5:
			cost = micK_costs[blk_num // SMNum]
		else:
			cost = (blk_num // SMNum - 4) * (micK_costs[4] - micK_costs[3]) + micK_costs[4]
		# print(cost, blk_num)
		if (cost < 0):
			print(cost, blk_num, out_shape, micK_Sshape)
			continue
		# histry.append((cost, blk_num))
		if cost < best_cost:
			best_cost = cost
			best_mick = s_key
			best_padded_outshape = [math.ceil(out_shape[i] / micK_Sshape[i]) * micK_Sshape[i]\
								for i in range(len(out_shape))]
	return best_cost, best_mick, best_padded_outshape#, histry




def get_best_micK_for_ops(all_micK_ops):
	target = tvm.target.Target("cuda")
	tuner = "xgbPretrain" 
	targetstr="cuda"
	kernel_type="micK"
	tasks = list()
	ret = dict()
	for T in [5, 24, 43, 62, 81, 100, 119, 128]:
		X_shape = (16*T, 768)
		Y_shape = (2304, 768)
		task = auto_scheduler.SearchTask(
		    func=dense_layer, args=(X_shape, Y_shape), target=target
		)
		tasks.append(task)
		print(task.workload_key)
		# 
	# get the latency dict for each micK in the available logs
	micK_latency_dict = dict()
	for mick_op_group in all_micK_ops:
		for micK in mick_op_group:
			log_file = get_task_log_file_name(micK.workload_key, tuner, targetstr, kernel_type)
			if not os.path.exists(log_file):
				continue
			log_reader = auto_scheduler.RecordReader(log_file)
			X_shape, Y_shape = get_inp_shapes(micK.workload_key)
			s_key =(X_shape[0], Y_shape[0], X_shape[1])
			micK_latency_dict[s_key] = list()
			for lineNO, (inp, res) in enumerate(log_reader):
				costs = [v.value for v in res.costs]
				cost = np.mean(costs)
				micK_latency_dict[s_key].append(cost)
	# 
	print(len(micK_latency_dict))
	# get the best latency for each task
	for task in tasks:
		cost, s_key, padded_outshape = get_best_micK_for_op(micK_latency_dict, task)
		ret[task.workload_key] = (cost, s_key, padded_outshape)
	return ret





def get_micK_GFLOPS_dict(all_micK_ops, micK_latency_dict):
	micK_GFLOPS_dict = dict()
	SMNum = 108
	for mick_op_group in all_micK_ops:
		for micK in mick_op_group:
			X_shape, Y_shape = get_inp_shapes(micK.workload_key)
			s_key =(X_shape[0], Y_shape[0], X_shape[1])
			if s_key not in micK_latency_dict:
				continue
			costs = micK_latency_dict[s_key]
			micK_GFLOPS_dict[s_key] = \
				[micK.compute_dag.flop_ct * (i+1) * SMNum / costs[i] / 1e9 \
				for i in range(len(costs))]
	return micK_GFLOPS_dict


'''
micK_GFLOPS_dict = get_micK_GFLOPS_dict(all_micK_ops, micK_latency_dict)
with open(f"tmp_res_hub/tmp_res16.py", "w") as file:
	file.write(f"def get_res():\n\treturn {micK_GFLOPS_dict}\n")
'''


# def measure_the_micK_on_ops(mapping_res, all_micK_ops):
# 	'''
# 	We should do corresponding padding when measuring the kernels, 
# 	because the running time will be worse if the operator shape is not aligned with the micro-kernel
# 	'''
# 	ret = dict()
# 	target = tvm.target.Target("cuda")
# 	tuner = "xgbPretrain" 
# 	targetstr="cuda"
# 	kernel_type="micK"
# 	tasks = list()
# 	ret = dict()
# 	for T in [5, 24, 43, 62, 81, 100, 119, 128]:
# 		X_shape = (16*T, 768)
# 		Y_shape = (2304, 768)
# 		task = auto_scheduler.SearchTask(
# 		    func=dense_layer, args=(X_shape, Y_shape), target=target
# 		)
# 		tasks.append(task)
# 		print(task.workload_key)
# 		# 
# 	measure_ctx = auto_scheduler.LocalRPCMeasureContext(min_repeat_ms=300)
# 	tune_option = auto_scheduler.TuningOptions(
# 	    num_measure_trials=1000,  # change this to 1000 to achieve the best performance
# 	    runner=measure_ctx.runner,
# 	    # measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
# 	    verbose=2,
# 	)
# 	for task in tasks:
# 		s_key = mapping_res[task.workload_key][1]
# 		padded_outshape = mapping_res[task.workload_key][2]
# 		print(s_key)
# 		micK = auto_scheduler.SearchTask(
# 		    func=dense_layer, args=((s_key[0], s_key[2]), (s_key[1], s_key[2])), target=target
# 		)
# 		log_file = get_task_log_file_name(micK.workload_key, tuner, targetstr, kernel_type)
# 		print(log_file)
# 		tmp = dict()
# 		my_load_all_best_input_from_file(log_file, target, tmp, workload_key=micK.workload_key)
# 		padded_op = auto_scheduler.SearchTask(
# 		    func=dense_layer, args=((padded_outshape[0], 768), \
# 		    	(padded_outshape[1], 768)), target=target
# 		)
# 		print(padded_op.workload_key)
# 		ret[task.workload_key] = measure_states(tune_option, padded_op, [tmp[micK.workload_key][0].state])
# 	return ret



# ours_measured = measure_the_micK_on_ops(ours, all_micK_ops)





def measure_the_micK_on_op(mick_shape, padded_taskshape, only_multi_tile = True,
	tuner="ansor", targetstr="cuda", kernel_type="micK", op_type='', mick_tune_on_op = False):
	'''
	We should do corresponding padding when measuring the kernels, 
	because the running time will be worse if the operator shape is not aligned with the micro-kernel
	Measure the best micro-kernel implementation of the given shape on task.
	If there is no micro-kernel of given shape, return None.
	'''
	target = tvm.target.Target("cuda")
	micK = None
	measured_op = None
	if op_type == 'dense':
		micK = auto_scheduler.SearchTask(
		    func=dense_layer, args=((mick_shape[0], mick_shape[2]), \
		    	(mick_shape[1], mick_shape[2])), target=target
		)
	elif op_type == 'conv2d':
		s = mick_shape
		micK = auto_scheduler.SearchTask(
			func=conv2d_nchw, args=((s[0], 64, s[2]-1+7, s[3]-1+7), (s[1], 64, 7, 7), [1,1], 0, [1,1], "float32"), 
			target=tvm.target.Target("cuda")
		)
		if not os.path.exists(get_task_log_file_name(micK.workload_key, tuner, targetstr, kernel_type)):
			micK = auto_scheduler.SearchTask(
				func=conv2d_nchw, args=((s[0], 64, s[2]-1+7, s[3]-1+7), (s[1], 64, 7, 7), 1, 0, 1, "float32"), 
				target=tvm.target.Target("cuda")
			)
	log_file = get_task_log_file_name(micK.workload_key, tuner, targetstr, kernel_type)
	if mick_tune_on_op:
		if op_type == 'dense':
			rep_layout = [get_symmetric_factor(540), 540//get_symmetric_factor(540), 8]
			op_s = [rep_layout[i] * mick_shape[i] for i in range(3)]
			measured_op = auto_scheduler.SearchTask(
				func=dense_layer, args=((op_s[0], op_s[2]), (op_s[1], op_s[2])), 
				target=target)
			log_file = get_task_log_file_name(micK.workload_key, tuner, targetstr, kernel_type, measured_op.workload_key)
	if not os.path.exists(log_file):
		return None
	if measured_op == None:
		measured_op = micK
	tmp = dict()
	if only_multi_tile:
		my_load_all_best_input_from_file_multiTileOnes(log_file, target, tmp, workload_key=measured_op.workload_key)
	else:
		my_load_all_best_input_from_file(log_file, target, tmp, workload_key=measured_op.workload_key)
	padded_op = None
	if op_type == 'dense':
		padded_op = auto_scheduler.SearchTask(
		    func=dense_layer, \
		    args=((padded_taskshape[0], padded_taskshape[2]), \
		    	(padded_taskshape[1], padded_taskshape[2])), target=target
		)
	elif op_type == 'conv2d':
		s = padded_taskshape
		padded_op = auto_scheduler.SearchTask(
		    func=conv2d_nchw, args=((s[0], 64, s[2]-1+7, s[3]-1+7), (s[1], 64, 7, 7), [1,1], 0, [1,1], "float32"), 
			target=target
		)
	measure_ctx = auto_scheduler.LocalRPCMeasureContext(min_repeat_ms=300)
	tune_option = auto_scheduler.TuningOptions(
	    num_measure_trials=1000,  # change this to 1000 to achieve the best performance
	    runner=measure_ctx.runner,
	    # measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
	    verbose=2,
	)
	ret = measure_states(tune_option, padded_op, [tmp[measured_op.workload_key][0].state])
	if ret != None:
		print("log_file: ", log_file)
		print("padded_op.workload_key: ", padded_op.workload_key)
	return ret





def measure_the_micK_on_op_given_tasks(padded_ops, micK_wlk, only_multi_tile = True,
	tuner="ansor", targetstr="cuda", kernel_type="micK"):
	'''
	We should do corresponding padding when measuring the kernels, 
	because the running time will be worse if the operator shape is not aligned with the micro-kernel
	Measure the best micro-kernel implementation of the given shape on task.
	If there is no micro-kernel of given shape, return None.
	NOTE: return the GFLOPS efficiency.
	'''
	target = tvm.target.Target("cuda")
	log_file = get_task_log_file_name(micK_wlk, tuner, targetstr, kernel_type)
	if not os.path.exists(log_file):
		return None
	print("log_file: ", log_file)
	tmp = dict()
	if only_multi_tile:
		my_load_all_best_input_from_file_multiTileOnes(log_file, target, tmp, workload_key=micK_wlk)
	else:
		my_load_all_best_input_from_file(log_file, target, tmp, workload_key=micK_wlk)
	measure_ctx = auto_scheduler.LocalRPCMeasureContext(min_repeat_ms=300)
	tune_option = auto_scheduler.TuningOptions(
	    num_measure_trials=1000,  # change this to 1000 to achieve the best performance
	    runner=measure_ctx.runner,
	    builder=auto_scheduler.LocalBuilder(n_parallel=60),
	    # measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
	    verbose=2,
	)
	# ret = measure_states(tune_option, padded_op, [tmp[micK.workload_key][0].state])
	# 
	ret_res = list()
	state_to_measure = tmp[micK_wlk][0].state
	# for padded_op in padded_ops:
	# 	ret = measure_states_for_diff_tasks(tune_option, [padded_op], [state_to_measure])
	# 	if ret == None:
	# 		ret_res.append(None)
	# 		continue
	# 	costs = [v.value for v in ret[0].costs]
	# 	cost = np.mean(costs)
	# 	ret_res.append(padded_op.compute_dag.flop_ct / cost / 1e9)
	# 	print(f"ret[{padded_op.workload_key}]={ret_res[-1]}", flush=True)
	# return ret_res
	# 
	for i in range(len(padded_ops)//256+1):
		ops_num = len(padded_ops[i*256:i*256+256])
		ret = measure_states_for_diff_tasks(tune_option, padded_ops[i*256:i*256+256], [state_to_measure for op_i in range(ops_num)])
		if ret == None:
			for padded_op in padded_ops[i*256:i*256+256]:
				tmp_res = measure_states_for_diff_tasks(tune_option, [padded_op], [state_to_measure])
				if tmp_res == None:
					ret_res.append(None)
				else:
					costs = [v.value for v in tmp_res[0].costs]
					cost = np.mean(costs)
					ret_res.append(padded_op.compute_dag.flop_ct / cost / 1e9)
					print(f"ret[{padded_op.workload_key}]={ret_res[-1]}", flush=True)
			continue
		for op_i in range(ops_num):
			padded_op = padded_ops[i*256:i*256+256][op_i]
			# print("log_file: ", log_file)
			# print("padded_op: ", padded_op.workload_key)
			# if ret[op_i]==None:
			# 	ret_res.append(None)
			# 	continue
			costs = [v.value for v in ret[op_i].costs]
			cost = np.mean(costs)
			ret_res.append(padded_op.compute_dag.flop_ct / cost / 1e9)
			print(f"ret[{padded_op.workload_key}]={ret_res[-1]}", flush=True)
	return ret_res





def measure_all_states_of_a_mick_on_op(mick_shape, padded_taskshape, only_multi_tile = True,
	tuner="ansor", targetstr="cuda", kernel_type="micK", op_type=''):
	'''
	We should do corresponding padding when measuring the kernels, 
	because the running time will be worse if the operator shape is not aligned with the micro-kernel
	Measure the best micro-kernel implementation of the given shape on task.
	If there is no micro-kernel of given shape, return None.
	NOTE: return the GFLOPS efficiency.
	'''
	target = tvm.target.Target("cuda")
	micK = None
	if op_type == 'dense':
		micK = auto_scheduler.SearchTask(
		    func=dense_layer, args=((mick_shape[0], mick_shape[2]), \
		    	(mick_shape[1], mick_shape[2])), target=target
		)
	elif op_type == 'conv2d':
		s = mick_shape
		micK = auto_scheduler.SearchTask(
			func=conv2d_nchw, args=((s[0], 64, s[2]-1+7, s[3]-1+7), (s[1], 64, 7, 7), [1,1], 0, [1,1], "float32"), 
			target=tvm.target.Target("cuda")
		)
		if not os.path.exists(get_task_log_file_name(micK.workload_key, tuner, targetstr, kernel_type)):
			micK = auto_scheduler.SearchTask(
				func=conv2d_nchw, args=((s[0], 64, s[2]-1+7, s[3]-1+7), (s[1], 64, 7, 7), 1, 0, 1, "float32"), 
				target=tvm.target.Target("cuda")
			)
	log_file = get_task_log_file_name(micK.workload_key, tuner, targetstr, kernel_type)
	if not os.path.exists(log_file):
		return None
	tmp = dict()
	if only_multi_tile:
		get_all_input_from_file_multiTileOnes(log_file, target, tmp, workload_key=micK.workload_key)
	else:
		my_load_all_best_input_from_file(log_file, target, tmp, workload_key=micK.workload_key)
	padded_op = None
	if op_type == 'dense':
		padded_op = auto_scheduler.SearchTask(
		    func=dense_layer, \
		    args=((padded_taskshape[0], padded_taskshape[2]), \
		    	(padded_taskshape[1], padded_taskshape[2])), target=target
		)
	elif op_type == 'conv2d':
		s = padded_taskshape
		padded_op = auto_scheduler.SearchTask(
		    func=conv2d_nchw, args=((s[0], 64, s[2]-1+7, s[3]-1+7), (s[1], 64, 7, 7), [1,1], 0, [1,1], "float32"), 
			target=target
		)
	measure_ctx = auto_scheduler.LocalRPCMeasureContext(min_repeat_ms=300)
	tune_option = auto_scheduler.TuningOptions(
	    num_measure_trials=1000,  # change this to 1000 to achieve the best performance
	    runner=measure_ctx.runner,
	    builder=auto_scheduler.LocalBuilder(n_parallel=60),
	    # measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
	    verbose=2,
	)
	# ret = measure_states(tune_option, padded_op, [tmp[micK.workload_key][0].state])
	# 
	ret_res = list()
	states_to_measure = [v[0].state for v in tmp[micK.workload_key]]
	# for padded_op in padded_ops:
	# 	ret = measure_states_for_diff_tasks(tune_option, [padded_op], [state_to_measure])
	# 	if ret == None:
	# 		ret_res.append(None)
	# 		continue
	# 	costs = [v.value for v in ret[0].costs]
	# 	cost = np.mean(costs)
	# 	ret_res.append(padded_op.compute_dag.flop_ct / cost / 1e9)
	# 	print(f"ret[{padded_op.workload_key}]={ret_res[-1]}", flush=True)
	# return ret_res
	# 
	for i in range(len(states_to_measure)//256+1):
		states_num = len(states_to_measure[i*256:i*256+256])
		ret = measure_states_for_diff_tasks(tune_option, [padded_op for op_i in range(states_num)], states_to_measure[i*256:i*256+256])
		if ret == None:
			for state_i, state in enumerate(states_to_measure[i*256:i*256+256]):
				tmp_res = measure_states_for_diff_tasks(tune_option, [padded_op], [state])
				if tmp_res == None:
					ret_res.append(None)
				else:
					costs = [v.value for v in tmp_res[0].costs]
					cost = np.mean(costs)
					ret_res.append(cost) # (padded_op.compute_dag.flop_ct / cost / 1e9)
					print(f"ret[{i*256+state_i}]={ret_res[-1]}", flush=True)
			continue
		for state_i in range(states_num):
			# padded_op = padded_ops[op_i]
			# print("log_file: ", log_file)
			# print("padded_op: ", padded_op.workload_key)
			# if ret[op_i]==None:
			# 	ret_res.append(None)
			# 	continue
			costs = [v.value for v in ret[state_i].costs]
			cost = np.mean(costs)
			# ret_res.append(costs) #padded_op.compute_dag.flop_ct / cost / 1e9)
			ret_res.append(cost)
			print(f"ret[{i*256+state_i}]={ret_res[-1]}", flush=True)
	return ret_res



def my_load_all_best_input_from_file(filename, target, best_result_dict, workload_key=None):
	# import tvm
	# from tvm import auto_scheduler
	log_reader = auto_scheduler.RecordReader(filename)
	lineNO = 0
	best_lineNO = -1
	for lineNO, (inp, res) in enumerate(log_reader):
		if res.error_no != auto_scheduler.measure.MeasureErrorNo.NO_ERROR:
			continue
		if target and inp.task.target.kind.name != target.kind.name:
			continue
		# 
		costs = [v.value for v in res.costs]
		cost = np.mean(costs)
		# 
		if (inp.task.workload_key not in best_result_dict) or (cost < best_result_dict[inp.task.workload_key][1]):
			best_result_dict[inp.task.workload_key] = [inp, cost]
			if (workload_key == None) or (inp.task.workload_key == workload_key):
				best_lineNO = lineNO
		# lineNO += 1
	# find the best configuration line as well
	with open(filename, "r") as file:
		for index, line in enumerate(file):
			if index == best_lineNO:
				return line







def my_load_all_best_input_from_file_multiTileOnes(filename, target, best_result_dict, workload_key=None):
	'''
	only find the best states which does multi-level tiling.
	'''
	# import tvm
	# from tvm import auto_scheduler
	log_reader = auto_scheduler.RecordReader(filename)
	lineNO = 0
	best_lineNO = -1
	for lineNO, (inp, res) in enumerate(log_reader):
		cost = None
		if target and inp.task.target.kind.name != target.kind.name:
			continue
		if len(auto_scheduler.measure.recover_measure_input(inp).state.transform_steps) in [5, 7]:
			continue
		if res.error_no != auto_scheduler.measure.MeasureErrorNo.NO_ERROR:
			cost = 1e10
		else:
			costs = [v.value for v in res.costs]
			cost = np.mean(costs)
		# 
		if (inp.task.workload_key not in best_result_dict) or (cost < best_result_dict[inp.task.workload_key][1]):
			best_result_dict[inp.task.workload_key] = [inp, cost]
			if (workload_key == None) or (inp.task.workload_key == workload_key):
				best_lineNO = lineNO
		# lineNO += 1
	# find the best configuration line as well
	with open(filename, "r") as file:
		for index, line in enumerate(file):
			if index == best_lineNO:
				return line






def get_all_input_from_file_multiTileOnes(filename, target, result_dict, workload_key):
	'''
	only find the best states which does multi-level tiling.
	'''
	# import tvm
	# from tvm import auto_scheduler
	log_reader = auto_scheduler.RecordReader(filename)
	# lineNO = 0
	# best_lineNO = -1
	for lineNO, (inp, res) in enumerate(log_reader):
		cost = None
		if target and inp.task.target.kind.name != target.kind.name:
			continue
		if len(auto_scheduler.measure.recover_measure_input(inp).state.transform_steps) in [5, 7]:
			continue
		if res.error_no != auto_scheduler.measure.MeasureErrorNo.NO_ERROR:
			cost = 1e10
		else:
			costs = [v.value for v in res.costs]
			cost = np.mean(costs)
		# 
		# if (inp.task.workload_key not in best_result_dict) or (cost < best_result_dict[inp.task.workload_key][1]):
		# 	best_result_dict[inp.task.workload_key] = [inp, cost]
		# 	if (workload_key == None) or (inp.task.workload_key == workload_key):
		# 		best_lineNO = lineNO
		if (inp.task.workload_key not in result_dict):
			result_dict[inp.task.workload_key] = list()
		result_dict[inp.task.workload_key].append((inp, cost))
		# lineNO += 1
	# find the best configuration line as well
	# with open(filename, "r") as file:
	# 	for index, line in enumerate(file):
	# 		if index == best_lineNO:
	# 			return line





def get_best_kernels_for_ops_by_ansor(tasks = None):
	target = tvm.target.Target("cuda")
	targetstr="cuda"
	tuner = "ansor"
	kernel_type = 'op'
	ret = dict()
	if tasks == None:
		kernel_type = ""
		tasks = list()
		for T in [5, 24, 43, 62, 81, 100, 119, 128]:
			X_shape = (16*T, 768)
			Y_shape = (2304, 768)
			task = auto_scheduler.SearchTask(
			    func=dense_layer, args=(X_shape, Y_shape), target=target
			)
			tasks.append(task)
			print(task.workload_key)
	for task in tasks:
		log_file = None
		if kernel_type == "":
			log_file = get_task_log_file_name_old(task.workload_key, tuner = tuner, target = targetstr, kernel_type = kernel_type)
		else:
			log_file = get_task_log_file_name(task.workload_key, tuner = tuner, target = targetstr, kernel_type = kernel_type)
		my_load_all_best_input_from_file(log_file, target, ret, workload_key=task.workload_key)
	return ret




def get_output_shape_from_wlk(wlk, op_type):
	s_key = None
	if op_type == 'dense':
		X_shape, Y_shape = get_inp_shapes(wlk, op_type=op_type)
		s_key =(X_shape[0], Y_shape[0], X_shape[1])
	elif op_type == 'bmm':
		X_shape, Y_shape = get_inp_shapes(wlk, op_type=op_type)
		s_key =(X_shape[0], X_shape[1], Y_shape[1], X_shape[2])
	elif op_type == 'bmm_nn':
		X_shape, Y_shape = get_inp_shapes(wlk, op_type=op_type)
		s_key =(X_shape[0], X_shape[1], Y_shape[2], Y_shape[1],)
	elif op_type == 'conv2d':
		# ["conv2d_nchw",[8,64,7,7],[306,64,7,7],[3,3],0,[1,1],"float32"]
		Dshape, Kshape = get_inp_shapes(wlk, op_type=op_type)
		stride, padding, dilation = json.loads(wlk)[3:6]
		if isinstance(stride, int):
			stride=(stride,stride)
		if isinstance(dilation, int):
			dilation=(dilation,dilation)
		assert int((Dshape[2]+2*padding-1-dilation[0]*(Kshape[2]-1))/stride[0]+1) == (Dshape[2]+2*padding-1-dilation[0]*(Kshape[2]-1))/stride[0]+1, \
			"not int length"
		assert int((Dshape[3]+2*padding-1-dilation[1]*(Kshape[3]-1))/stride[1]+1) == (Dshape[3]+2*padding-1-dilation[1]*(Kshape[3]-1))/stride[1]+1, \
			"not int length"
		s_key = (Dshape[0], Kshape[0], 
			int((Dshape[2]+2*padding-1-dilation[0]*(Kshape[2]-1))/stride[0]+1), 
			int((Dshape[3]+2*padding-1-dilation[1]*(Kshape[3]-1))/stride[1]+1), Kshape[1], Kshape[2], Kshape[3], 
			tuple(stride), padding, tuple(dilation))
	return s_key




def get_best_GFLOPS_for_micK_by_ansor(all_micK_ops, op_type, kernel_type):
	ret = dict()
	target = tvm.target.Target("cuda")
	# kernel_type = "micK"
	for mick_op_group in all_micK_ops:
		for micK in mick_op_group:
			log_file = get_task_log_file_name(micK.workload_key, tuner = "ansor",target = "cuda", kernel_type=kernel_type)
			if not os.path.exists(log_file):
				continue
			s_key = get_output_shape_from_wlk(micK.workload_key, op_type)
			# if op_type == 'dense':
			# 	X_shape, Y_shape = get_inp_shapes(micK.workload_key, op_type=op_type)
			# 	s_key =(X_shape[0], Y_shape[0], X_shape[1])
			# elif op_type == 'conv2d':
			# 	Dshape, Kshape = get_inp_shapes(micK.workload_key, op_type=op_type)
			# 	s_key = (Dshape[0], Kshape[0], Dshape[2]-6, Dshape[3]-6, Kshape[1], Kshape[2], Kshape[3])
			tmp = dict()
			my_load_all_best_input_from_file_multiTileOnes(log_file, target, tmp, workload_key=micK.workload_key)
			if micK.workload_key not in tmp:
				continue
			ret[s_key] = micK.compute_dag.flop_ct / tmp[micK.workload_key][1] /1e9
	return ret




'''
for k in ours:
	print(ours[k], ours_measured[k][0].costs, ansors[k][1])
'''

'''
ansors = get_best_GFLOPS_for_micK_by_ansor(all_micK_ops)
with open(f"tmp_res_hub/tmp_res17.py", "w") as file:
	file.write(f"def get_res():\n\treturn {ansors}\n")
'''



def measure_micK_diff_rep_layouts(micK, only_multi_tile = True, kernel_type = "micK"):
	'''
	Measure the efficiency of different replication layouts for the given micro-kernel.
	'''
	target = tvm.target.Target("cuda")
	ret = dict()
	log_file = get_task_log_file_name(micK.workload_key, tuner = "ansor",target = "cuda", kernel_type=kernel_type)
	if not os.path.exists(log_file):
		return None
	X_shape, Y_shape = get_inp_shapes(micK.workload_key)
	s_key =(X_shape[0], Y_shape[0], X_shape[1])
	tmp = dict()
	if only_multi_tile:
		my_load_all_best_input_from_file_multiTileOnes(log_file, target, tmp, workload_key=micK.workload_key)
	else:
		my_load_all_best_input_from_file(log_file, target, tmp, workload_key=micK.workload_key)
	best_state = tmp[micK.workload_key][0].state
	ret[(1, 1)] = [micK.compute_dag.flop_ct / tmp[micK.workload_key][1] /1e9]
	# 
	rep_layouts = list()
	blk_nums = [440, 540]
	for blk_num in blk_nums:
		for repN in get_factors(blk_num):
			rep_layouts.append((repN, blk_num//repN))
	# 
	red_lens = [X_shape[1]]
	if kernel_type == "micK1fetch":
		red_lens = [X_shape[1]*n for n in [1, 5, 10, 20, 40, 80]]
	for red_len in red_lens:
		for rep_layout in rep_layouts:
			task = auto_scheduler.SearchTask(
			    func=dense_layer, 
			    args=((X_shape[0]*rep_layout[0], red_len), (Y_shape[0]*rep_layout[1], red_len)), 
			    target=target
			)
			print(task.workload_key)
			if task.workload_key in tmp:
				print("no measure")
				ret[rep_layout] = (task.compute_dag.flop_ct / tmp[task.workload_key][1] / 1e9)
				continue
			# return task, tmp[micK.workload_key][0]
			measure_ctx = auto_scheduler.LocalRPCMeasureContext(min_repeat_ms=300)
			tune_option = auto_scheduler.TuningOptions(
			    num_measure_trials=1000,  # change this to 1000 to achieve the best performance
			    runner=measure_ctx.runner,
			    # measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
			    verbose=2,
			)
			ress = measure_states(tune_option, task, [best_state])
			if ress == None:
				continue
			costs = [v.value for v in ress[0].costs]
			cost = np.mean(costs)
			ret[rep_layout] = (task.compute_dag.flop_ct / cost / 1e9)
			auto_scheduler.save_records(log_file, 
				[MeasureInput(task, best_state)], ress)
	return ret




'''
micK_repLs_multiTiles = dict()
micK1 = auto_scheduler.SearchTask(
			    func=dense_layer, 
			    args=((129, 768), (6, 768)), 
			    target=target
			)


micK_repLs_multiTiles[(129, 6, 768)] = measure_micK_diff_rep_layouts(micK1, only_multi_tile = True, kernel_type = "micK")


micK2 = auto_scheduler.SearchTask(
			    func=dense_layer, 
			    args=((192, 768), (7, 768)), 
			    target=target
			)


micK_repLs_multiTiles[(192, 7, 768)] = measure_micK_diff_rep_layouts(micK2, only_multi_tile = True, kernel_type = "micK")

'''






def measure_micK_curve_(all_micK_ops, diff_measure_task_wlks, only_multi_tile = True, kernel_type = "micK", op_type='', to_measure=False, 
	tuner = 'ansor'):
	'''
		only when to_measure is True, will do measure on hardware, and will not return anything.
		when to_measure is False, will not do measure, and will return the measured GFLOPS.
		so we should first call this function with to_measure=True, and then with to_measure=False.
	'''
	SMNum = 108
	target = tvm.target.Target("cuda")
	# kernel_type = "micK"
	tune_option = None
	if to_measure:
		measure_ctx = auto_scheduler.LocalRPCMeasureContext(min_repeat_ms=300)
		tune_option = auto_scheduler.TuningOptions(
		    num_measure_trials=1000,  # change this to 1000 to achieve the best performance
		    runner=measure_ctx.runner,
		    builder=auto_scheduler.LocalBuilder(n_parallel=60),
		    # measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
		    verbose=2,
		)
	ret = dict()
	micKs_to_measureCurve = list()
	for mick_op_group, diff_measure_task_wlk_grp in zip(all_micK_ops, diff_measure_task_wlks):
		for micK, diff_measure_task_wlk in zip(mick_op_group, diff_measure_task_wlk_grp):
			log_file = get_task_log_file_name(micK.workload_key, tuner = tuner, target = "cuda", kernel_type=kernel_type, \
				diff_measure_task_wlk=diff_measure_task_wlk)
			if not os.path.exists(log_file):
				continue
			else:
				micKs_to_measureCurve.append(micK.workload_key)
	for mick_op_group, diff_measure_task_wlk_grp in zip(all_micK_ops, diff_measure_task_wlks):
		for micK, diff_measure_task_wlk in zip(mick_op_group, diff_measure_task_wlk_grp):
			tasks_to_measure = list()
			if micK.workload_key not in micKs_to_measureCurve:
				continue
			log_file = get_task_log_file_name(micK.workload_key, tuner = tuner, target = "cuda", kernel_type=kernel_type, \
				diff_measure_task_wlk=diff_measure_task_wlk)
			s_key = get_output_shape_from_wlk(micK.workload_key, op_type)
			print(s_key)
			# X_shape, Y_shape = get_inp_shapes(micK.workload_key)
			# s_key =(X_shape[0], Y_shape[0], X_shape[1])
			tmp = dict()
			if only_multi_tile:
				my_load_all_best_input_from_file_multiTileOnes(log_file, target, tmp, workload_key=diff_measure_task_wlk)#micK.workload_key)
			else:
				my_load_all_best_input_from_file(log_file, target, tmp, workload_key=diff_measure_task_wlk) #micK.workload_key)
			if diff_measure_task_wlk not in tmp:
				continue
			best_state = tmp[diff_measure_task_wlk][0].state
			ret[s_key] = list()
			# ret[s_key] = [micK.compute_dag.flop_ct / tmp[micK.workload_key][1] /1e9]
			# 
			red_lens = list()
			if op_type == 'dense':
				red_lens = [s_key[2]]#[X_shape[1]]
				if kernel_type == "micK1fetch":
					# red_lens = [s_key[2]*n for n in [1, 5, 10, 20, 40, 80]]
					red_lens = [s_key[2]*n for n in [2**i for i in range(7)]]
			elif op_type in ['bmm', 'bmm_nn']:
				red_lens = [s_key[3]]#[X_shape[1]]
				if kernel_type == "micK1fetch":
					# red_lens = [s_key[2]*n for n in [1, 5, 10, 20, 40, 80]]
					red_lens = [s_key[3]*n for n in [2**i for i in range(7)]]
			elif op_type == 'conv2d':
				red_lens = [s_key[4:7]]
				if kernel_type == "micK1fetch":
					# red_lens = [s_key[2]*n for n in [1, 5, 10, 20, 40, 80]]
					red_lens = [(s_key[4]*n, ) + s_key[5:7] for n in [2**i for i in range(7)]]
			# 
			interested_blkNs = {0:[48, 108], 1:[110, 216], 2:[220, 324], 3:[330, 432], 4:[440, 540]}
			for red_len in red_lens:
				for n in range(0, 5):
					# we also want to measure different replication layout
					# blk_nums = [SMNum*n+1, SMNum*(n+1)]
					blk_nums = interested_blkNs[n]
					tasks = list()
					for blk_num in blk_nums:
						rep_nums = get_factors(blk_num)
						interested_repNs = [1, blk_num]
						# compute the rep_num with the minimum data read amount
						# interested_repNs = [1, sorted(rep_nums, key=lambda repN: X_shape[0]*repN+Y_shape[0]*blk_num//repN)[0]]
						interested_repNs.append(sorted(rep_nums, key=lambda repN: repN+blk_num//repN)[0])
						if 1 == interested_repNs[-1]:
							interested_repNs[-1] = blk_num
						if op_type in ['bmm', 'bmm_nn']:
							interested_repNs = [(1, interested_repNs[-1], blk_num//interested_repNs[-1]), (blk_num, 1, 1)]
						for repN in interested_repNs:
							task = None
							if op_type == 'dense':
								task = auto_scheduler.SearchTask(
								    func=dense_layer, 
								    args=((s_key[0]*repN, red_len), (s_key[1]*blk_num//repN, red_len)), 
								    target=target
								)
							elif op_type == 'bmm':
								rep_s = [s_key[i]*repN[i] for i in range(len(repN))]
								task = auto_scheduler.SearchTask(
								    func=batch_matmul, 
								    args=((rep_s[0], rep_s[1], red_len), (rep_s[0], rep_s[2], red_len)), 
								    target=target
								)
							elif op_type == 'bmm_nn':
								rep_s = [s_key[i]*repN[i] for i in range(len(repN))]
								task = auto_scheduler.SearchTask(
								    func=batch_matmul_noTrans, 
								    args=((rep_s[0], rep_s[1], red_len), (rep_s[0], red_len, rep_s[2])), 
								    target=target
								)
							elif op_type == 'conv2d':
								rep_layout = [repN, blk_num//repN, 1, 1]
								rep_s = [s_key[i]*rep_layout[i] for i in range(len(rep_layout))]
								# Dshape = (rep_s[0], red_len[0], rep_s[2]-1+red_len[1], rep_s[3]-1+red_len[2])
								# Kshape = tuple([rep_s[1]] + list(red_len))
								Dshape, Kshape = get_inp_shapes_from_paramters(rep_s, op_type, tuple(red_len)+tuple(s_key[7:]))
								task = auto_scheduler.SearchTask(
									func=conv2d_nchw, 
									args=(Dshape, Kshape) + tuple(s_key[7:]) +  ("float32",), 
									target=target
								)
							tasks.append(task)
					for task in tasks:
						print(task.workload_key)
						if task.workload_key in tmp:
							print("no measure")
							ret[s_key].append(task.compute_dag.flop_ct / tmp[task.workload_key][1] / 1e9)
							continue
						tasks_to_measure.append(task)
			# measure the tasks in batch
			if not to_measure:
				assert len(tasks_to_measure) == 0, 'There are tasks not measured.'
			if len(tasks_to_measure) == 0:
				continue
			ress = measure_states_for_diff_tasks(tune_option, tasks_to_measure, [best_state for task in tasks_to_measure])
			if ress == None:
				for task in tasks_to_measure:
					ress = measure_states_for_diff_tasks(tune_option, [task], [best_state])
					if ress == None:
						continue
					costs = [v.value for v in ress[0].costs]
					cost = np.mean(costs)
					# ret[s_key].append(task.compute_dag.flop_ct / cost / 1e9)
					auto_scheduler.save_records(log_file, 
						[MeasureInput(task, best_state)], ress)
			else:
				inputs = [MeasureInput(task, best_state) for task in tasks_to_measure]
				auto_scheduler.save_records(log_file, inputs, ress)
				# for task, ress_i in zip(tasks_to_measure, ress):
				# 	costs = [v.value for v in ress_i.costs]
				# 	cost = np.mean(costs)
				# 	# ret[s_key].append(task.compute_dag.flop_ct / cost / 1e9)
				# 	auto_scheduler.save_records(log_file, 
				# 		[MeasureInput(task, best_state)], [ress_i])
			# 
						# return task, tmp[micK.workload_key][0]
						# measure_ctx = auto_scheduler.LocalRPCMeasureContext(min_repeat_ms=300)
						# tune_option = auto_scheduler.TuningOptions(
						#     num_measure_trials=1000,  # change this to 1000 to achieve the best performance
						#     runner=measure_ctx.runner,
						#     # measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
						#     verbose=2,
						# )
						# ress = measure_states(tune_option, task, [best_state])
						# if ress == None:
						# 	continue
						# costs = [v.value for v in ress[0].costs]
						# cost = np.mean(costs)
						# ret[s_key].append(task.compute_dag.flop_ct / cost / 1e9)
						# auto_scheduler.save_records(log_file, 
						# 	[MeasureInput(task, best_state)], ress)
	if not to_measure:
		return ret




def measure_micK_curve(all_micK_ops, diff_measure_task_wlks, only_multi_tile = True, kernel_type = "micK", op_type='', tuner='ansor'):
	measure_micK_curve_(all_micK_ops, diff_measure_task_wlks, only_multi_tile = only_multi_tile, kernel_type = kernel_type, op_type=op_type, to_measure=True, tuner=tuner)
	ret = measure_micK_curve_(all_micK_ops, diff_measure_task_wlks, only_multi_tile = only_multi_tile, kernel_type = kernel_type, op_type=op_type, to_measure=False, tuner=tuner)
	return ret



'''
micK_curs = measure_micK_curve(all_micK_ops, False)
micK_curs_multiTiles = measure_micK_curve(all_micK_ops, only_multi_tile = True)

with open(f"tmp_res_hub/tmp_res18.py", "w") as file:
	file.write(f"def get_micK_curs():\n\treturn {micK_curs}\n\n")
	file.write(f"def get_micK_curs_multiTiles():\n\treturn {micK_curs_multiTiles}\n\n")


'''


def get_micK_latency_curve(all_micK_ops, only_multi_tile = True):
	SMNum = 108
	target = tvm.target.Target("cuda")
	kernel_type = "micK"
	ret = dict()
	micKs_to_measureCurve = list()
	for mick_op_group in all_micK_ops:
		for micK in mick_op_group:
			log_file = get_task_log_file_name(micK.workload_key, tuner = "ansor",target = "cuda", kernel_type=kernel_type)
			if not os.path.exists(log_file):
				continue
			else:
				micKs_to_measureCurve.append(micK.workload_key)
	for mick_op_group in all_micK_ops:
		for micK in mick_op_group:
			if micK.workload_key not in micKs_to_measureCurve:
				continue
			log_file = get_task_log_file_name(micK.workload_key, tuner = "ansor",target = "cuda", kernel_type=kernel_type)
			X_shape, Y_shape = get_inp_shapes(micK.workload_key)
			s_key =(X_shape[0], Y_shape[0], X_shape[1])
			tmp = dict()
			if only_multi_tile:
				my_load_all_best_input_from_file_multiTileOnes(log_file, target, tmp, workload_key=micK.workload_key)
			else:
				my_load_all_best_input_from_file(log_file, target, tmp, workload_key=micK.workload_key)
			best_state = tmp[micK.workload_key][0].state
			ret[s_key] = [tmp[micK.workload_key][1]]
			# 
			for n in range(1, 5):
				task = auto_scheduler.SearchTask(
				    func=dense_layer, 
				    args=((X_shape[0], 768), (Y_shape[0]*(SMNum*n+1), 768)), 
				    target=target
				)
				print(task.workload_key)
				if task.workload_key in tmp:
					ret[s_key].append(tmp[task.workload_key][1])
					continue
	return ret



'''
micK_curs_multiTiles = dict()
for k, vs in micK_latency_dict.items():
	if k not in micK_curs_multiTiles:
		micK_curs_multiTiles[k] = list()
	for i in range(len(vs)):
		micK_curs_multiTiles[k].append(get_product(k)*2*(i*108+1)/vs[i]/1e9)

'''


'''
micK_latency_dict = get_micK_latency_curve(all_micK_ops, only_multi_tile = True)

import tmp_res19
import tmp_res20
import tmp_res21
import tmp_res22
micK_curs_multiTiles_19 = tmp_res19.get_micK_curs_multiTiles()
micK_curs_multiTiles_20 = tmp_res20.get_micK_curs_multiTiles()
micK_curs_multiTiles_21 = tmp_res21.get_micK_curs_multiTiles()
micK_curs_multiTiles_22 = tmp_res22.get_micK_curs_multiTiles()

micK_latency_dict = dict()
for a in [micK_curs_multiTiles_19, micK_curs_multiTiles_20, micK_curs_multiTiles_21, micK_curs_multiTiles_22]:
	for k, v in a.items():
		micK_latency_dict[k] = list()
		for i in range(len(v)):
			if v[i] < 1:
				micK_latency_dict[k].append(1e10)
			else:
				micK_latency_dict[k].append(get_product(k)*2/v[i]/1e9*(108*i+1))


target = tvm.target.Target("cuda")
tasks = list()
ret = dict()
for T in [5, 24, 43, 62, 81, 100, 119, 128]:
	X_shape = (16*T, 768)
	Y_shape = (2304, 768)
	task = auto_scheduler.SearchTask(
	    func=dense_layer, args=(X_shape, Y_shape), target=target
	)
	tasks.append(task)
	print(task.workload_key)
for task in tasks:
	cost, s_key, padded_outshape = get_best_micK_for_op(micK_latency_dict, task)
	ret[task.workload_key] = (cost, s_key, padded_outshape)

ours = dict()
for task in tasks:
	v = ret[task.workload_key]
	tmp = measure_the_micK_on_op(v[1], v[2]+[768], only_multi_tile = True,
		tuner="ansor", targetstr="cuda", kernel_type="micK")
	if tmp != None:
		ours[task.workload_key] = (v[1], v[2], \
			v[0], tmp)

'''





def func_micK_curv(x, a, b, c, d):
	# we assume the micK latency curve satisfy this function
	# return a * np.emath.logn(b, (x+c)) + d
	return -a*(b **(x+c))+ d




def scale_micK_curve_X(x):
	return x/1e6


def scale_micK_curve_Y(y):
	return y/1e3



def scale_back_micK_curve_Y(y):
	return y*1e3



def fit_micK_curve(X, Y, func):
	'''
	We use this function to fit micK curve. We assume it to be exp-like function.
	Currently, we fit the curve to 1 block efficiency.
	Return the best parameters we found.
	'''
	X = scale_micK_curve_X(np.array(X))
	Y = scale_micK_curve_Y(np.array(Y)) # divided by 1e3, in terms of TFLOPS
	popt, pcov = scipy.optimize.curve_fit(func, X, Y, p0=(4, 0.75, -4, 12))
	print(f"the best params are {popt}")
	print(f"the covar are {pcov}")
	return popt


def func_micK_curv_UB(x, a, b, c, d):
    # fit the upper bound of the curve
    return -a*(b **(x+c))+ d
#     return a * np.log(x+c) / np.log(b) + d

def func_micK_curv_LB(x, a, b, c, d):
#     fit the lower bound of the curve
    # we assume the micK latency curve satisfy this function
#     should not be log, should be like exponential function
    return a*(x+d)+b/(x+d)+c
#     return a * np.log(x+c) / np.log(b) + d



def func_micK_curv_ConvHull(x, a, b):
#     fit the lower bound of the curve
	# this is a piecewise function
	if x>a[-1]:
		# the points outside the coverage of this cost model are invalid, this is because the lw and up curves may not have the same coverage for conv2d
		# maybe we should select other interesting mick shapes.
		return np.array(-1)
	x = np.array(x).astype(float)
	conditions = [x < a[0]] + \
			[((a[i] <= x) and (x < a[i+1])) for i in range(len(a)-1)] + \
			[a[-1]<=x]
	# i = conditions.index(True)
	# choices = [(b[1]-b[0])*(x-a[0])/(a[1]-a[0])+b[0]] + \
	# 		[(b[i+1]-b[i])*(x-a[i])/(a[i+1]-a[i])+b[i] for i in range(len(a)-1)] + \
	# 		[(b[-1]-b[-2])*(x-a[-2])/(a[-1]-a[-2])+b[-2]]
	functions = [lambda v: (b[1]-b[0])*(v-a[0])/(a[1]-a[0])+b[0]] + \
			[lambda v, i=i: (b[i+1]-b[i])*(v-a[i])/(a[i+1]-a[i])+b[i] for i in range(len(a)-1)] + \
			[lambda v: (b[-1]-b[-2])*(v-a[-2])/(a[-1]-a[-2])+b[-2]]
	return np.e**(np.piecewise(x, conditions, functions))



def comp_data_read_amount(mick_shape, rep_layout, op_type):
	'''
		The rep_layout here is only about the replication over spatial axes.
	'''
	if op_type == 'dense':
		return sum([mick_shape[i] * rep_layout[i] for i in range(2)])*mick_shape[2]
	elif op_type in ['bmm', 'bmm_nn']:
		return mick_shape[0]*rep_layout[0]*sum([mick_shape[i] * rep_layout[i] for i in range(1,3)])*mick_shape[3]
	elif op_type == 'conv2d':
		n, c, h, w = [mick_shape[i]*rep_layout[i] for i in range(len(rep_layout))]
		rc, rh, rw, stride, padding, dilation = mick_shape[4:]
		# ret = get_product((s[0], 64, s[2]+6, s[3]+6)) + get_product((s[1], 64, 7, 7))
		sh, sw = stride
		dh, dw = dilation
		ret = get_product((n, rc, sh*(h-1)+dh*(rh-1)+1, sw*(w-1)+dw*(rw-1)+1))+\
			get_product((c, rc, rh, rw))
		return ret




def comp_efficiency_given_rep_and_mick_on_curve_old(func_micK_curvs, popts, curve_rep_layouts, key1, key2, base_mick_shape1, base_mick_shape2, mick_shape):
	'''
	Compute the efficiency of a micro-kernel of given replicatioin layout based on on the curve.
	key1: the key for the upper curve of given replication layout (and replication number).
	key2: the key for the lower curve of given replication layout (and replication number).
	base_mick_shape1, base_mick_shape2: base micro-kernel shapes for upper curve and lower curve respectively.
	'''
	mick_size = get_product(mick_shape) / 1e7
	# comp the efficiency of two base micro-kernel shapes
	E1 = func_micK_curvs[key1](mick_size, *(popts[key1]))
	E2 = func_micK_curvs[key2](mick_size, *(popts[key2]))
	# print(f"key1:{key1}, key2:{key2}, Etop:{E1}, Ebottom:{E2}")
	# the data read amount of the two base micro-kernel shapes
	data1 = comp_data_read_amount(base_mick_shape1, curve_rep_layouts[key1])
	data2 = comp_data_read_amount(base_mick_shape2, curve_rep_layouts[key2])
	data = comp_data_read_amount(mick_shape, curve_rep_layouts[key1])
	print(f"data1:{data1/1e7}, data2:{data2/1e7}, data:{data/1e7}, layout:{curve_rep_layouts[key1]}")
	# print(base_mick_shape1, base_mick_shape2, curve_rep_layouts[key1])
	# print(key1, key2, base_mick_shape1, base_mick_shape2, mick_shape)
	if data1 == data2:
		return (E1+E2)/2*1e3
	latency1 = 1/E1
	latency2 = 1/E2
	latency = (latency1-latency2)*(data-data2)/(data1-data2)+latency2
	E = 1/latency
	print(f"efficiency:{E}")
	return E*1e3






def comp_efficiency_given_rep_and_mick_on_curve_old2(func_micK_curvs, popts, curve_rep_layouts, key1, key2, base_mick_shape1, base_mick_shape2, mick_shape, 
	op_type):
	'''
	Compute the efficiency of a micro-kernel of given replicatioin layout based on on the curve.
	key1: the key for the upper curve of given replication layout (and replication number).
	key2: the key for the lower curve of given replication layout (and replication number).
	base_mick_shape1, base_mick_shape2: base micro-kernel shapes for upper curve and lower curve respectively.
	'''
	if op_type == 'dense':
		mick_size = get_product(mick_shape) / 1e7
	elif op_type == 'conv2d':
		mick_size = get_product(mick_shape[:4]) / 1e4
	# comp the efficiency of two base micro-kernel shapes
	E1 = func_micK_curvs[key1](mick_size, *(popts[key1]))
	E2 = func_micK_curvs[key2](mick_size, *(popts[key2]))
	# print(f"key1:{key1}, key2:{key2}, Etop:{E1}, Ebottom:{E2}")
	# we just assume in any time, E1 (of square mick) should be higher than E2 (of fixed len mick)
	if E1<0:
		E1 = 1/1e10
	if E2<0:
		E2 = 1/1e10
	if E1 < E2:
		tmp = E2
		E2 = E1
		E1 = tmp
	# the data read amount of the two base micro-kernel shapes, only 
	rep_layout = (1, 1)
	if op_type == 'conv2d':
		rep_layout = (1, 1, 1, 1,)
	data1 = comp_data_read_amount(base_mick_shape1, rep_layout, op_type)
	data2 = comp_data_read_amount(base_mick_shape2, rep_layout, op_type)
	data = comp_data_read_amount(mick_shape, rep_layout, op_type)
	# print(f"data1:{data1/1e7}, data2:{data2/1e7}, data:{data/1e7}")#, layout:{curve_rep_layouts[key1]}")
	# print(base_mick_shape1, base_mick_shape2, curve_rep_layouts[key1])
	# print(key1, key2, base_mick_shape1, base_mick_shape2, mick_shape)
	if data1 == data2:
		return (E1+E2)/2*1e3
	latency1 = 1/E1
	latency2 = 1/E2
	latency = (latency1-latency2)*(data-data2)/(data1-data2)+latency2
	E = 1/latency
	# print(f"efficiency:{E}")
	if E < 0:
		E = 1/1e10
	return E*1e3




def comp_efficiency_given_rep_and_mick_on_curve(func_micK_curvs, popts, curve_rep_layouts, key1, key2, base_mick_shape1, base_mick_shape2, mick_shape, 
	op_type):
	'''
	Compute the efficiency of a micro-kernel of given replicatioin layout based on on the curve.
	key1: the key for the upper curve of given replication layout (and replication number).
	key2: the key for the lower curve of given replication layout (and replication number).
	base_mick_shape1, base_mick_shape2: base micro-kernel shapes for upper curve and lower curve respectively.
	'''
	micK_Sshape = None
	if op_type == 'dense':
		micK_Sshape = mick_shape[:2]
	elif op_type == 'conv2d':
		micK_Sshape = mick_shape[:4]
	mick_size = get_product(micK_Sshape) / 1e4 # this is the x value on the GFLOPS curves
	# comp the efficiency of two base micro-kernel shapes
	E1 = func_micK_curvs[key1](mick_size, *(popts[key1]))
	E2 = func_micK_curvs[key2](mick_size, *(popts[key2]))
	# print(f"key1:{key1}, key2:{key2}, Etop:{E1}, Ebottom:{E2}")
	# we just assume in any time, E1 (of square mick) should be higher than E2 (of fixed len mick)
	if E1<0:
		E1 = 1/1e10
	if E2<0:
		E2 = 1/1e10
	if E1 < E2:
		E2 = E1
		# tmp = E2
		# E2 = E1
		# E1 = tmp
	# the data read amount of the two base micro-kernel shapes, only 
	rep_layout = (1, 1)
	if op_type == 'conv2d':
		rep_layout = (1, 1, 1, 1,)
	data1 = comp_data_read_amount(base_mick_shape1, rep_layout, op_type)
	data2 = None
	if isinstance(base_mick_shape2, float):
		data2 = base_mick_shape2
	else:
		data2 = comp_data_read_amount(base_mick_shape2, rep_layout, op_type)
	data = comp_data_read_amount(mick_shape, rep_layout, op_type)
	# print(f"data1:{data1/1e7}, data2:{data2/1e7}, data:{data/1e7}")#, layout:{curve_rep_layouts[key1]}")
	# print(base_mick_shape1, base_mick_shape2, curve_rep_layouts[key1])
	# print(key1, key2, base_mick_shape1, base_mick_shape2, mick_shape)
	if data1 == data2: # it means the min data read amount shape is the max data read amount shape, so this case is valid
		return (E1+E2)/2*1e3
	latency1 = 1/E1
	latency2 = 1/E2
	latency = (latency1-latency2)*(data-data2)/(data1-data2)+latency2
	E = 1/latency
	# print(f"efficiency:{E}")
	if E < 0:
		E = 1/1e10
	return E*1e3




def comp_efficiency_given_rep_and_mick_on_curve_2Rngs(func_micK_curvs, popts, curve_rep_layouts, key1, key2, base_mick_shape1, base_mick_shape2, mick_shape, 
	op_type):
	'''
	Compute the efficiency of a micro-kernel of given replicatioin layout based on on the curve.
	key1: the key for the upper curve of given replication layout (and replication number).
	key2: the key for the lower curve of given replication layout (and replication number).
	base_mick_shape1, base_mick_shape2: base micro-kernel shapes for upper curve and lower curve respectively.
	'''
	micK_Sshape = None
	if op_type == 'dense':
		micK_Sshape = mick_shape[:2]
	elif op_type in ['bmm', 'bmm_nn']:
		micK_Sshape = mick_shape[:3]
	elif op_type == 'conv2d':
		micK_Sshape = mick_shape[:4]
	mick_size = get_product(micK_Sshape) #/ 1e4 # this is the x value on the GFLOPS curves
	# comp the efficiency of two base micro-kernel shapes
	try:
		# E1 = np.e**(func_micK_curvs[key1](mick_size, *(popts[key1])))
		# E2 = np.e**(func_micK_curvs[key2](mick_size, *(popts[key2])))
		E1 = (func_micK_curvs[key1](mick_size, *(popts[key1])))
		E2 = (func_micK_curvs[key2](mick_size, *(popts[key2])))
	except Exception as e:
		print(key1, key2, mick_shape, mick_size)
		print(e)
		assert False
	# E1 = np.e**(func_micK_curvs[key1](mick_size, *(popts[key1])))
	# E2 = np.e**(func_micK_curvs[key2](mick_size, *(popts[key2])))
	# print(f"key1:{key1}, key2:{key2}, Etop:{E1}, Ebottom:{E2}")
	# we just assume in any time, E1 (of square mick) should be higher than E2 (of fixed len mick)
	# if E1<0:
	# 	E1 = 1/1e10
	# if E2<0:
	# 	E2 = 1/1e10
	# the above two cases should not happen after we fit the log curves of GFLOPS 
	if min(E1, E2) < 0:
		# the mick_size is not covered by some curves
		return -1
	# if E1 < E2:
	# 	# E2 = E1
	# 	E1, E2 = E2, E1
	# 	# tmp = E2
	# 	# E2 = E1
	# 	# E1 = tmp
	# the data read amount of the two base micro-kernel shapes, only 
	rep_layout = (1, 1)
	if op_type == 'conv2d':
		rep_layout = (1, 1, 1, 1,)
	elif op_type in ['bmm', 'bmm_nn']:
		rep_layout = (1, 1, 1,)
	data1 = comp_data_read_amount(base_mick_shape1, rep_layout, op_type)
	data2 = None
	if isinstance(base_mick_shape2, float):
		data2 = base_mick_shape2
	else:
		data2 = comp_data_read_amount(base_mick_shape2, rep_layout, op_type)
	data = comp_data_read_amount(mick_shape, rep_layout, op_type)
	if ((E1 < E2) and (data1<data2)) or ((E1 > E2) and (data1>data2)):
		# E2 = E1
		E1, E2 = E2, E1
	# 
	# print(f"data1:{data1/1e7}, data2:{data2/1e7}, data:{data/1e7}")#, layout:{curve_rep_layouts[key1]}")
	# print(base_mick_shape1, base_mick_shape2, curve_rep_layouts[key1])
	# print(key1, key2, base_mick_shape1, base_mick_shape2, mick_shape)
	if data1 == data2: # it means the min data read amount shape is the max data read amount shape, so this case is valid
		if data == data1:
			return (E1+E2)/2#*1e3
		else:
			return -1 # this is an invalid value, and should be filtered by the function calling this method
	latency1 = 1/E1
	latency2 = 1/E2
	latency = (latency1-latency2)*(data-data2)/(data1-data2)+latency2
	if latency == 0:
		# this case should not happen
		return -1 # this is an invalid value, and should be filtered by the function calling this method
	E = 1/latency
	# print(f"efficiency:{E}")
	if E < 0:
		E = 1/1e10
	return E#*1e3



# WE USE THIS VERSION'S RESULT, THIS FUNCTION SHOULD BE THE SAME AS get_min_DataRead_mick_shape EXCEPT NO CACHE IS USED HERE
def get_min_DataRead_mick_shape_noCache(micK_size, onlyInt, op_type, mick_shape):
	'''
	micK_size is the size of output tensor.
	If onlyInt = True, then only consider integers as the loop length.
	'''
	if op_type == 'dense':
		if onlyInt:
			v = get_symmetric_factor(micK_size)
			return (v, micK_size // v)
		else:
			return (micK_size**0.5, micK_size**0.5)
	elif op_type in ['bmm', 'bmm_nn']:
		if onlyInt:
			v = get_symmetric_factor(micK_size)
			return (1, v, micK_size // v)
		else:
			return (1, micK_size**0.5, micK_size**0.5)
	elif op_type == 'conv2d':
		if onlyInt:
			# n1s = get_factors(micK_size)
			# min_cost = None
			# best_s = None
			# for n1 in n1s:
			# 	h = get_symmetric_factor(n1)
			# 	w = n1//h
			# 	n = 1
			# 	c = micK_size//n1
			# 	cost = data_read_amount_PerBlk(op_type, (n, c, h, w)+mick_shape[4:])
			# 	if (min_cost==None) or (cost < min_cost):
			# 		min_cost = cost
			# 		best_s = [n, c, h, w]
			# return tuple(best_s)
			n1s = get_factors(micK_size)
			min_cost = None
			best_s = None
			for n1 in n1s:
				for h in get_factors(n1):
					w = n1//h
					n = 1
					c = micK_size//n1
					cost = data_read_amount_PerBlk(op_type, (n, c, h, w)+mick_shape[4:])
					if (min_cost==None) or (cost < min_cost):
						min_cost = cost
						best_s = [n, c, h, w]
			return tuple(best_s)
		else:
			# to speedup, we just iterate to find the best one
			rc, rh, rw, stride, padding, dilation = mick_shape[4:]
			# the mick shapes here are exactly of the same pattern as the ones we generate for the cost model
			# although in general, there are 3 cases we should consider to find the min data read one
			n1s = get_factors(micK_size)
			min_cost = None
			best_s = None
			sh, sw = stride
			dh, dw = dilation
			c1 = -sh+dh*(rh-1)+1
			c2 = -sw+dw*(rw-1)+1
			# print(c1, c2)
			for n1 in n1s: #(sh*h+c1)*(sw*w+c2)  --> sh*h*c2+sw*w*c1 minimize -->sh*h*c2==sw*w*c1 -->sh*h*c2==sw*n1/h*c1 -->h=(sw*n1*c1/sh/c2)**0.5
				hs = []
				if c1 > 0:
					if c2>0:
						hs=[(sw*n1*c1/sh/c2)**0.5]
					else:
						hs=[n1]
				elif c1 == 0:
					if c2 > 0:
						hs = [1]
					elif c2 <= 0:
						hs = [n1]
				else:
					if c2 < 0:
						hs = [1, n1]
					else:
						hs=[1]
				for h in hs:
					# h = (sw*n1*c1/sh/c2)**0.5 # n1**0.5
					w = n1/h
					n = 1
					c = micK_size/n1
					cost = data_read_amount_PerBlk(op_type, (n, c, h, w)+mick_shape[4:])
					if (min_cost==None) or (cost < min_cost):
						min_cost = cost
						best_s = [n, c, h, w]
			return tuple(best_s)
			# 
			# below is to use the sympy to find the min data read Ssp
			# y = sympy.symbols('y', real=True, positive=True)
			# # 
			# # x = sympy.symbols('x', real=True, positive=True)
			# # micK_size_symb = sympy.symbols('micK_size_symb', real=True, positive=True)
			# # dataRead = (64*(sympy.sqrt(x)+6)*(sympy.sqrt(x)+6) + micK_size_symb*64*49/x).simplify()
			# # dif_y = sympy.diff(dataRead, x).subs(sympy.sqrt(x), y)
			# # dif_y.subs(micK_size_symb, micK_size)
			# # 
			# dif_y = -3136*micK_size/y**4 + 64*(y + 6)/y
			# vs = sympy.solve(dif_y, y)
			# for v in vs:
			# 	val = v.evalf()
			# 	if (isinstance(val, sympy.core.numbers.Float)) and (val > 0):
			# 		val = float(val)
			# 		return (1, micK_size/(val**2), val, val)
			# 
			# 
			# The following implementation is too slow, so we use the code above
			# x = sympy.symbols('x', real=True, positive=True)
			# dataRead = (64*(sympy.sqrt(x)+6)*(sympy.sqrt(x)+6) + micK_size*64*49/x).simplify()
			# dif = sympy.diff(dataRead)
			# # vs = sympy.solve(x**4+6*(x**3)-49*micK_size, x)
			# vs = sympy.solve(dif, x)
			# assert len(vs) == 1, 'Not only one root in finding the min-data-read mick shape!'
			# for v in vs:
			# 	# val = v.evalf()
			# 	# if (isinstance(val, sympy.core.numbers.Float)) and (val > 0):
			# 	# 	val = float(val)
			# 	# 	return (1, micK_size/(val**2), val, val)
			# 	val = float(v)
			# 	return (1, micK_size/val, val**0.5, val**0.5)





def get_min_DataRead_mick_shape(micK_size, onlyInt, op_type, mick_shape, cache):
	'''
	micK_size is the size of output tensor.
	If onlyInt = True, then only consider integers as the loop length.
	CACHE INFOR: THE CACHE DICT IS PER OPTYPE, per reduc shape, and per other params.
	Note: we do not use cache for dense and bmm.
	'''
	# 
	if op_type == 'dense':
		if onlyInt:
			v = get_symmetric_factor(micK_size)
			return (v, micK_size // v)
		else:
			return (micK_size**0.5, micK_size**0.5)
	elif op_type in ['bmm', 'bmm_nn']:
		if onlyInt:
			v = get_symmetric_factor(micK_size)
			return (1, v, micK_size // v)
		else:
			return (1, micK_size**0.5, micK_size**0.5)
	elif op_type == 'conv2d':
		# check the cache first
		cache_key = (micK_size, onlyInt) 
		if cache_key in cache:
			return cache[cache_key]
		if onlyInt:
			# n1s = get_factors(micK_size)
			# min_cost = None
			# best_s = None
			# for n1 in n1s:
			# 	h = get_symmetric_factor(n1)
			# 	w = n1//h
			# 	n = 1
			# 	c = micK_size//n1
			# 	cost = data_read_amount_PerBlk(op_type, (n, c, h, w)+mick_shape[4:])
			# 	if (min_cost==None) or (cost < min_cost):
			# 		min_cost = cost
			# 		best_s = [n, c, h, w]
			# return tuple(best_s)
			n1s = get_factors(micK_size)
			min_cost = None
			best_s = None
			for n1 in n1s:
				for h in get_factors(n1):
					w = n1//h
					n = 1
					c = micK_size//n1
					cost = data_read_amount_PerBlk(op_type, (n, c, h, w)+mick_shape[4:])
					if (min_cost==None) or (cost < min_cost):
						min_cost = cost
						best_s = [n, c, h, w]
			cache[cache_key] = tuple(best_s)
			return cache[cache_key]
		else:
			# to speedup, we just iterate to find the best one
			rc, rh, rw, stride, padding, dilation = mick_shape[4:]
			# the mick shapes here are exactly of the same pattern as the ones we generate for the cost model
			# although in general, there are 3 cases we should consider to find the min data read one
			n1s = get_factors(micK_size)
			min_cost = None
			best_s = None
			sh, sw = stride
			dh, dw = dilation
			c1 = -sh+dh*(rh-1)+1
			c2 = -sw+dw*(rw-1)+1
			# print(c1, c2)
			for n1 in n1s: #(sh*h+c1)*(sw*w+c2)  --> sh*h*c2+sw*w*c1 minimize -->sh*h*c2==sw*w*c1 -->sh*h*c2==sw*n1/h*c1 -->h=(sw*n1*c1/sh/c2)**0.5
				hs = []
				if c1 > 0:
					if c2>0:
						hs=[(sw*n1*c1/sh/c2)**0.5]
					else:
						hs=[n1]
				elif c1 == 0:
					if c2 > 0:
						hs = [1]
					elif c2 <= 0:
						hs = [n1]
				else:
					if c2 < 0:
						hs = [1, n1]
					else:
						hs=[1]
				for h in hs:
					# h = (sw*n1*c1/sh/c2)**0.5 # n1**0.5
					w = n1/h
					n = 1
					c = micK_size/n1
					cost = data_read_amount_PerBlk(op_type, (n, c, h, w)+mick_shape[4:])
					if (min_cost==None) or (cost < min_cost):
						min_cost = cost
						best_s = [n, c, h, w]
			cache[cache_key] = tuple(best_s)
			return cache[cache_key]
			# 
			# below is to use the sympy to find the min data read Ssp
			# y = sympy.symbols('y', real=True, positive=True)
			# # 
			# # x = sympy.symbols('x', real=True, positive=True)
			# # micK_size_symb = sympy.symbols('micK_size_symb', real=True, positive=True)
			# # dataRead = (64*(sympy.sqrt(x)+6)*(sympy.sqrt(x)+6) + micK_size_symb*64*49/x).simplify()
			# # dif_y = sympy.diff(dataRead, x).subs(sympy.sqrt(x), y)
			# # dif_y.subs(micK_size_symb, micK_size)
			# # 
			# dif_y = -3136*micK_size/y**4 + 64*(y + 6)/y
			# vs = sympy.solve(dif_y, y)
			# for v in vs:
			# 	val = v.evalf()
			# 	if (isinstance(val, sympy.core.numbers.Float)) and (val > 0):
			# 		val = float(val)
			# 		return (1, micK_size/(val**2), val, val)
			# 
			# 
			# The following implementation is too slow, so we use the code above
			# x = sympy.symbols('x', real=True, positive=True)
			# dataRead = (64*(sympy.sqrt(x)+6)*(sympy.sqrt(x)+6) + micK_size*64*49/x).simplify()
			# dif = sympy.diff(dataRead)
			# # vs = sympy.solve(x**4+6*(x**3)-49*micK_size, x)
			# vs = sympy.solve(dif, x)
			# assert len(vs) == 1, 'Not only one root in finding the min-data-read mick shape!'
			# for v in vs:
			# 	# val = v.evalf()
			# 	# if (isinstance(val, sympy.core.numbers.Float)) and (val > 0):
			# 	# 	val = float(val)
			# 	# 	return (1, micK_size/(val**2), val, val)
			# 	val = float(v)
			# 	return (1, micK_size/val, val**0.5, val**0.5)







def binary_search_gradient0(lw, up, func):
	'''
		This function searches the xs whose y is closes to 0
		func: the lambda function to compute the gradient.
	'''
	# print(lw, up)
	while lw < up:
		mid = lw + int((up - lw) / 2 * 10)/10 # we enable float number here
		val = func(mid)
		# print(val)
		if val > 0:
			up = mid
		elif val == 0:
			return mid
		else:
			lw = mid + 0.1 # 1
	return lw





# THE FUNCTION IS THE SAME AS get_min_DataRead_rep_layout, EXCEPT THAT IT HAS NO CACHE
def get_min_DataRead_rep_layout_noCache(blk_num, micK_Sshape, op_type, mick_shape):
	'''
	micK_size is the size of output tensor.
	We only return the optimal float rep layout.
	'''
	if op_type == 'dense':
		v = (blk_num*micK_Sshape[1]/micK_Sshape[0])**0.5
		return (v, blk_num/v)
	elif op_type in ['bmm', 'bmm_nn']:
		v = (blk_num*micK_Sshape[2]/micK_Sshape[1])**0.5
		return (1, v, blk_num/v)
	elif op_type == 'conv2d':
		# there are 4 cases that we should consider
		n, c, h, w = micK_Sshape
		rc, rh, rw, stride, padding, dilation = mick_shape[4:]
		sh, sw = stride
		dh, dw = dilation
		offset1 = -sh+dh*(rh-1)+1
		offset2 = -sw+dw*(rw-1)+1
		if offset1<=0 and offset2<=0:
			# the best repl should be [x, y, 1, 1]
			nb = (blk_num*c*rc*rh*rw / (n*rc*(sh*h+offset1)*(sw*w+offset2)))**0.5
			return (nb, blk_num/nb, 1, 1)
		elif offset1<=0 and offset2>=0:
			# the best repl should be [1, y, 1, x]
			nw = (blk_num*c*rc*rh*rw / (n*rc*(sh*h+offset1)*sw*w))**0.5
			return (1, blk_num/nw, 1, nw)
		elif offset1>=0 and offset2<=0:
			# the best repl should be [1, y, x, 1]
			nh = (blk_num*c*rc*rh*rw / (n*rc*(sw*w+offset2)*sh*h))**0.5
			return (1, blk_num/nh, nh, 1)
		# ELSE: in this case, both offsets are > 0, the best repl should be [1, y, x1, x2]
		# # 
		# The method below is too slow, so I change to search the best point directly, i.e., search the point with gradient 0.
		alpha = sh*h*offset2/offset1
		c1 = 2*alpha*h*n*rc*sh
		c2 = alpha*n*offset1*rc + h*n*offset2*rc*sh
		c3 = 2*blk_num*c*rc*rh*rw*sw*w/alpha
		dif = lambda x1: c1*x1 + c2 - c3/x1**3
		# if dif(1)>=0:
		# 	# the dataRead increases as x1, so x1=1
		# 	nh, nw = 1, alpha/(sw*w)
		# 	return (1, blk_num/(nh*nw), nh, nw)
		# if dif(blk_num)<=0:
		# 	# the dataRead decreases as x1, so x1=blk_num
		# 	nh, nw = blk_num, alpha/(sw*w)*blk_num
		# 	return (1, blk_num/(nh*nw), nh, nw)
		# # in this case, dataRead first decreases as x1 and then it increases as x1
		# max_x1 = min(int((c3/c1)**0.25)+1, blk_num)
		# nh = binary_search_gradient0(2, max_x1, dif)
		max_x1 = int((c3/c1)**0.25)+1 # we limit x1 to be any positive real number
		# print(f'c1, c2, c3: {(c1, c2, c3)}')
		nh = binary_search_gradient0(0.1, max_x1, dif)
		nw = alpha/(sw*w)*nh
		return (1, blk_num/(nh*nw), nh, nw)
		# #
		# n = sympy.symbols('n', real=True, positive=True)
		# c = sympy.symbols('c', real=True, positive=True)
		# h = sympy.symbols('h', real=True, positive=True)
		# w = sympy.symbols('w', real=True, positive=True)
		# rc = sympy.symbols('rc', real=True, positive=True)
		# rh = sympy.symbols('rh', real=True, positive=True)
		# rw = sympy.symbols('rw', real=True, positive=True)
		# blk_num = sympy.symbols('blk_num', real=True, positive=True)
		# # in the best repl, sh*h*offset2*x1 = sw*w*offset1*x2
		# x1 = sympy.symbols('x1', real=True, positive=True)
		# sh = sympy.symbols('sh', real=True, positive=True)
		# sw = sympy.symbols('sw', real=True, positive=True)
		# dh = sympy.symbols('dh', real=True, positive=True)
		# dw = sympy.symbols('dw', real=True, positive=True)
		# offset1 = sympy.symbols('offset1', real=True, positive=True)
		# offset2 = sympy.symbols('offset2', real=True, positive=True)
		# # alpha = sh*h*offset2/offset1
		# alpha = sympy.symbols('alpha', real=True, positive=True)
		# dataRead = (n*rc*(sh*h*x1+offset1)*(sh*h*offset2/offset1*x1+offset2)+c*rc*rh*rw*blk_num/x1/(sh*h*offset2/offset1/sw/w*x1)).simplify()
		# dataRead = (n*rc*(sh*h*x1+offset1)*(alpha*x1+offset2)+c*rc*rh*rw*blk_num/x1/(alpha/sw/w*x1)).simplify()
		# # dataRead = ((n*rc*sh*h*sh*h*offset2/offset1*x1**2+n*rc*2*sh*h*offset2*x1+n*rc*offset1*offset2)+c*rc*rh*rw*blk_num/(sh*h*offset2/offset1/sw/w)/(x1**2)).simplify()
		# # dataRead = dataRead.subs(((offset1, -sh+dh*(rh-1)+1), (offset2, -sw+dw*(rw-1)+1))).simplify()
		# dif = sympy.diff(dataRead, x1).simplify()
		# 
		# the code below will not be executed
		x1 = sympy.symbols('x1', real=True, positive=True)
		alpha = sh*h*offset2/offset1
		dif = 2*alpha*h*n*rc*sh*x1 + alpha*n*offset1*rc + h*n*offset2*rc*sh - 2*blk_num*c*rc*rh*rw*sw*w/(alpha*x1**3)
		vs = sympy.solve(dif, x1)
		for v in vs:
			val = v.evalf()
			if (isinstance(val, sympy.core.numbers.Float)) and (val > 0):
				nh = float(val)
				if nh>blk_num:
					nh = blk_num
				nw = alpha/(sw*w)*nh
				return (1, blk_num/(nh*nw), nh, nw)
		# if reach here, then there is no valid nh, we set nh=1, i.e., the minimum possible value
		nh, nw = 1, alpha/(sw*w)
		return (1, blk_num/(nh*nw), nh, nw)
		# 
		# BELOW is the previous equation assuming reduc shape to be (64, 7, 7)
		# y = sympy.symbols('y', real=True, positive=True)
		# # x = sympy.symbols('x', real=True, positive=True)
		# # n_symb = sympy.symbols('n_symb', real=True, positive=True)
		# # c_symb = sympy.symbols('c_symb', real=True, positive=True)
		# # h_symb = sympy.symbols('h_symb', real=True, positive=True)
		# # w_symb = sympy.symbols('w_symb', real=True, positive=True)
		# # blk_num_symb = sympy.symbols('blk_num_symb', real=True, positive=True)
		# # dataRead = (n_symb*64*(h_symb*w_symb*x+36+12*(sympy.sqrt(h_symb*w_symb*x))) + c_symb*blk_num_symb*64*49/x).simplify()
		# # dif = sympy.diff(dataRead, x).simplify()
		# # dif_y = dif.subs(1/sympy.sqrt(x), y)
		# # dif_y = dif_y.subs(((blk_num_symb, blk_num), (n_symb, n), (c_symb, c), (h_symb, h), (w_symb, w)))
		# # 
		# hsqr = h**0.5
		# wsqr = w**0.5
		# dif_y = -3136*blk_num*c*y**4 + 384*hsqr*n*wsqr*y + 64*h*n*w
		# vs = sympy.solve(dif_y, y)
		# for v in vs:
		# 	val = v.evalf()
		# 	if (isinstance(val, sympy.core.numbers.Float)) and (val > 0):
		# 		val = 1/float(val)
		# 		rh = ((w/h)**0.5)*val
		# 		return (1, blk_num/(val**2), rh, val**2/rh)
		# 
		# The following code is too slow, use the code above instead.
		# x = sympy.symbols('x', real=True, positive=True)
		# n, c, h, w = micK_Sshape
		# dataRead = (n*64*(h*w*x+36+12*(sympy.sqrt(h*w*x))) + c*blk_num*64*49/x).simplify()
		# dif = sympy.diff(dataRead)
		# vs = sympy.solve(dif, x)
		# assert len(vs) == 1, 'Not only one root in finding the min-data-read rep layout!'
		# for v in vs:
		# 	# val = v.evalf(3)
		# 	# if (isinstance(val, sympy.core.numbers.Float)) and (val > 0):
		# 	# 	val = float(val)
		# 	# 	rh = (w*val/h)**0.5
		# 	# 	return (1, blk_num/val, rh, val/rh)
		# 	val = float(v)
		# 	rh = (w*val/h)**0.5
		# 	return (1, blk_num/val, rh, val/rh)






def get_min_DataRead_rep_layout(blk_num, micK_Sshape, op_type, mick_shape, cache):
	'''
	micK_size is the size of output tensor.
	We only return the optimal float rep layout.
	CACHE INFO: We do not use cache for dense and bmm. 
				The cache dict is per op_type, per reduc shape, per other params. 
	'''
	if op_type == 'dense':
		v = (blk_num*micK_Sshape[1]/micK_Sshape[0])**0.5
		return (v, blk_num/v)
	elif op_type in ['bmm', 'bmm_nn']:
		v = (blk_num*micK_Sshape[2]/micK_Sshape[1])**0.5
		return (1, v, blk_num/v)
	elif op_type == 'conv2d':
		# there are 4 cases that we should consider
		n, c, h, w = micK_Sshape
		rc, rh, rw, stride, padding, dilation = mick_shape[4:]
		sh, sw = stride
		dh, dw = dilation
		offset1 = -sh+dh*(rh-1)+1
		offset2 = -sw+dw*(rw-1)+1
		if offset1<=0 and offset2<=0:
			# the best repl should be [x, y, 1, 1]
			nb = (blk_num*c*rc*rh*rw / (n*rc*(sh*h+offset1)*(sw*w+offset2)))**0.5
			return (nb, blk_num/nb, 1, 1)
		elif offset1<=0 and offset2>=0:
			# the best repl should be [1, y, 1, x]
			nw = (blk_num*c*rc*rh*rw / (n*rc*(sh*h+offset1)*sw*w))**0.5
			return (1, blk_num/nw, 1, nw)
		elif offset1>=0 and offset2<=0:
			# the best repl should be [1, y, x, 1]
			nh = (blk_num*c*rc*rh*rw / (n*rc*(sw*w+offset2)*sh*h))**0.5
			return (1, blk_num/nh, nh, 1)
		# ELSE: in this case, both offsets are > 0, the best repl should be [1, y, x1, x2]
		# # 
		# check the cache first
		cache_key = (blk_num, tuple(micK_Sshape))
		if cache_key in cache:
			return cache[cache_key]
		# 
		# The method below is too slow, so I change to search the best point directly, i.e., search the point with gradient 0.
		alpha = sh*h*offset2/offset1
		c1 = 2*alpha*h*n*rc*sh
		c2 = alpha*n*offset1*rc + h*n*offset2*rc*sh
		c3 = 2*blk_num*c*rc*rh*rw*sw*w/alpha
		dif = lambda x1: c1*x1 + c2 - c3/x1**3
		# if dif(1)>=0:
		# 	# the dataRead increases as x1, so x1=1
		# 	nh, nw = 1, alpha/(sw*w)
		# 	return (1, blk_num/(nh*nw), nh, nw)
		# if dif(blk_num)<=0:
		# 	# the dataRead decreases as x1, so x1=blk_num
		# 	nh, nw = blk_num, alpha/(sw*w)*blk_num
		# 	return (1, blk_num/(nh*nw), nh, nw)
		# # in this case, dataRead first decreases as x1 and then it increases as x1
		# max_x1 = min(int((c3/c1)**0.25)+1, blk_num)
		# nh = binary_search_gradient0(2, max_x1, dif)
		max_x1 = int((c3/c1)**0.25)+1 # we limit x1 to be any positive real number
		# print(f'c1, c2, c3: {(c1, c2, c3)}')
		nh = binary_search_gradient0(0.1, max_x1, dif)
		nw = alpha/(sw*w)*nh
		cache[cache_key] = (1, blk_num/(nh*nw), nh, nw)
		return cache[cache_key]








def get_max_DataRead_rep_layout(blk_num, micK_Sshape, op_type, mick_shape):
	'''
	micK_size is the size of output tensor.
	We only return the optimal float rep layout.
	'''
	if op_type == 'dense':
		tmp = sorted([1, blk_num], key=lambda repN: comp_data_read_amount(tuple(micK_Sshape)+(768,), (repN, blk_num/repN), op_type))[-1]
		return (tmp, blk_num/tmp)
	# 
	elif op_type in ['bmm', 'bmm_nn']:
		# tmp = sorted([1, blk_num], key=lambda repN: comp_data_read_amount(tuple(micK_Sshape)+(768,), (repN, blk_num/repN), op_type))[-1]
		return (blk_num, 1, 1)
	# 
	elif op_type == 'conv2d':
		# tmp = sorted([1, blk_num], key=lambda repN: comp_data_read_amount(micK_Sshape, (repN, blk_num/repN, 1, 1), op_type))[-1]
		# return (tmp, blk_num/tmp, 1, 1)
		# there are 4 cases to consider, similar to "get_min_DataRead_rep_layout"
		n, c, h, w = micK_Sshape
		rc, rh, rw, stride, padding, dilation = mick_shape[4:]
		sh, sw = stride
		dh, dw = dilation
		offset1 = -sh+dh*(rh-1)+1
		offset2 = -sw+dw*(rw-1)+1
		if offset1>=0 and offset2>=0:
			# the best repl should be [x, y, 1, 1]
			tmp = sorted([1, blk_num], key=lambda repN: comp_data_read_amount(mick_shape, (repN, blk_num/repN, 1, 1), op_type))[-1]
			return (tmp, blk_num/tmp, 1, 1)
		elif offset1>=0 and offset2<=0:
			# the best repl should be [1, y, 1, x]
			tmp = sorted([1, blk_num], key=lambda repN: comp_data_read_amount(mick_shape, (1, blk_num/repN, 1, repN), op_type))[-1]
			return (1, blk_num/tmp, 1, tmp)
		elif offset1<=0 and offset2>=0:
			# the best repl should be [1, y, x, 1]
			tmp = sorted([1, blk_num], key=lambda repN: comp_data_read_amount(mick_shape, (1, blk_num/repN, repN, 1), op_type))[-1]
			return (1, blk_num/tmp, tmp, 1)
		# ELSE: in this case, both offsets are < 0, the best repl should be [1, y, x1, x2]
		# in the best repl, sh*h*offset2*x1 = sw*w*offset1*x2
		# then the diff func should be the same as in "get_min_DataRead_rep_layout", to find the max data read one, we find endpoints for x1.
		alpha = sh*h*offset2/(sw*w*offset1)
		tmp = sorted([1, blk_num], \
			key=lambda repN: comp_data_read_amount(mick_shape, (1, blk_num/(repN*alpha*repN), repN, alpha*repN), op_type))[-1]
		return (1, blk_num/(tmp*alpha*tmp), tmp, alpha*tmp)
		# # 



def my_cost_model_old(func_micK_curvs, popts, curve_rep_layouts, interested_blkNs, mick_shape, task_shape, op_type):
	'''
	compute the cost in terms of s.
	INPUT:
		func_micK_curvs: a dict of functions which fit the micro-kernel latency curves: 
			for each block number (5 in total, from small to large) and each block layout (2 in total, upper bound first, lower bound curve next), 
			there are 2 curve (1 for lower part, 1 for upper part, 20 in total).
		popts: a dict of parameters for the corr. functions.
		curve_rep_layouts: a dict of replication layouts, the replication layout corr. to each func_micK_curv.
		mick_shape: a list of int, the given micro-kernel shape, including reduction part.
		task_shape: a list of int, the given task shape.
	'''
	# first compute the block number
	SMNum = 108
	micK_Sshape = None
	if op_type == 'dense':
		micK_Sshape = mick_shape[:2]
	elif op_type == 'conv2d':
		micK_Sshape = mick_shape[:4]
	out_size = get_product(micK_Sshape)
	rep_layout = [math.ceil(task_shape[i] / micK_Sshape[i]) for i in range(len(micK_Sshape))]
	blk_num = get_product(rep_layout)
	# the max number of thread blocks assigned to one SM
	idx = math.ceil(blk_num / SMNum) - 1
	# real_base_blk_nums = [idx*SMNum+1, (idx+1)*SMNum]
	if idx > 4:
		idx = 4
	Es = dict()
	# ref_base_blk_nums = [idx*SMNum+1, (idx+1)*SMNum]
	ref_base_blk_nums = interested_blkNs[idx]
	# rep_keys = ['line', 'square']
	rep_keys = ['line_h', 'line_v', 'square']
	# the two base micro-kernel shapes
	base_mick_shape1, base_mick_shape2 = None, None
	min_mick_Sshape = get_min_DataRead_mick_shape(out_size, False, op_type)
	if op_type == 'dense':
		base_mick_shape1 = min_mick_Sshape+(768,)
		base_mick_shape2 = (out_size/8, 8, 768)
	elif op_type == 'conv2d':
		base_mick_shape1 = min_mick_Sshape+(64, 7, 7,)
		base_mick_shape2 = (8, out_size/8, 1, 1, 64, 7, 7,)
	# 
	for base_blk_i in range(len(ref_base_blk_nums)):
		base_blk = ref_base_blk_nums[base_blk_i]
		# print("base_blk: ", base_blk)
		for rep_key in rep_keys:
			key1 = (base_blk, rep_key, 'up')
			key2 = (base_blk, rep_key, 'lw')
			# print(f"key1 is {key1}, key2 is {key2}")
			E = comp_efficiency_given_rep_and_mick_on_curve_old2(func_micK_curvs, popts, curve_rep_layouts, 
				key1, key2, base_mick_shape1, base_mick_shape2, mick_shape, op_type)
			Es[(base_blk, rep_key)] = E
	# 
	# comp the efficiency for the target rep_layout of the target micro-kernel
	base_repLs1 = list()
	base_repLs2 = list()
	# base_repNs2 = [1 for base_blk in ref_base_blk_nums]
	for base_blk in ref_base_blk_nums:
		# tmp = sorted(get_factors(base_blk), key=lambda repN: repN*mick_shape[0]+(base_blk//repN)*mick_shape[1])
		# # base_repNs1.append(tmp[0])
		# base_repNs1.append((base_blk*mick_shape[1]/mick_shape[0])**0.5) # consider the float best repN
		# base_repNs2.append(tmp[-1])
		base_repLs1.append(get_min_DataRead_rep_layout(base_blk, micK_Sshape, op_type))
		base_repLs2.append(get_max_DataRead_rep_layout(base_blk, micK_Sshape, op_type))
	# 
	# base_datas1 = [comp_data_read_amount(mick_shape, (base_repNs1[i], ref_base_blk_nums[i]//base_repNs1[i])) for i in range(2)]
	# base_datas2 = [comp_data_read_amount(mick_shape, (base_repNs2[i], ref_base_blk_nums[i]//base_repNs2[i])) for i in range(2)]
	base_datas1 = [comp_data_read_amount(mick_shape, base_repLs1[i], op_type) for i in range(2)]
	base_datas2 = [comp_data_read_amount(mick_shape, base_repLs2[i], op_type) for i in range(2)]
	# 
	datas_sqr = [comp_data_read_amount(mick_shape, curve_rep_layouts[(base_blk, 'square', 'up')], op_type) for base_blk in ref_base_blk_nums]
	datas_h = [comp_data_read_amount(mick_shape, curve_rep_layouts[(base_blk, 'line_h', 'up')], op_type) for base_blk in ref_base_blk_nums]
	datas_v = [comp_data_read_amount(mick_shape, curve_rep_layouts[(base_blk, 'line_v', 'up')], op_type) for base_blk in ref_base_blk_nums]
	# if True in [base_datas1[i] == base_datas2[i] for i in range(2)]:
	# 	return 1e10
	if True in [(datas_sqr[i] == datas_h[i]) and (datas_sqr[i] == datas_v[i]) for i in range(2)]:
		return 1e10
	# 
	# normed_datas = [(datas[i] - base_datas2[i]) / (base_datas1[i] - base_datas2[i]) for i in range(2)]
	# if 0 in normed_datas:
	# 	return 1e10
	normed_datas1 = list()
	normed_datas2 = list()
	base_Es1 = list()
	base_Es2 = list() 
	for i in range(2):
		tmp_data = [datas_sqr[i], datas_h[i], datas_v[i]]
		tmp_labels = ['square', 'line_h', 'line_v']
		sorted_idx = sorted(range(3), key=lambda j: tmp_data[j])
		normed_datas1.append((tmp_data[sorted_idx[0]] - base_datas2[i]) / (base_datas1[i] - base_datas2[i]))
		normed_datas2.append((tmp_data[sorted_idx[-1]] - base_datas2[i]) / (base_datas1[i] - base_datas2[i]))
		base_Es1.append(Es[(ref_base_blk_nums[i], tmp_labels[sorted_idx[0]])])
		base_Es2.append(Es[(ref_base_blk_nums[i], tmp_labels[sorted_idx[-1]])])
	# 
	# base_Es1 = [Es[(base_blk, 'square')] for base_blk in ref_base_blk_nums]
	# base_Es2 = [Es[(base_blk, 'line')] for base_blk in ref_base_blk_nums]
	# 
	# tmp = sorted(get_factors(blk_num), key=lambda repN: repN*mick_shape[0]+(blk_num//repN)*mick_shape[1])
	# # base_repN1 = tmp[0]
	# base_repN1 = (blk_num*mick_shape[1]/mick_shape[0])**0.5 # consider the float best repN
	# base_repN2 = tmp[-1]
	base_repL1 = get_min_DataRead_rep_layout(blk_num, micK_Sshape, op_type)
	base_repL2 = get_max_DataRead_rep_layout(blk_num, micK_Sshape, op_type)
	# base_data1 = comp_data_read_amount(mick_shape, (base_repN1, blk_num//base_repN1))
	# base_data2 = comp_data_read_amount(mick_shape, (base_repN2, blk_num//base_repN2))
	base_data1 = comp_data_read_amount(mick_shape, base_repL1, op_type)
	base_data2 = comp_data_read_amount(mick_shape, base_repL2, op_type)
	# 
	data = comp_data_read_amount(mick_shape, rep_layout, op_type)
	if base_data1 == base_data2:
		# return 1e10
		normed_data = 0 # we assume the bottom points are always on a line in one segment of the rep layout curve
	else:
		normed_data = (data - base_data2) / (base_data1 - base_data2)
	base_Es = [(base_Es1[i]-base_Es2[i])*(normed_data-normed_datas2[i])/(normed_datas1[i]-normed_datas2[i])+base_Es2[i] for i in range(2)]
	# print((real_base_blk_nums[1]-real_base_blk_nums[0]))
	E = (base_Es[1]-base_Es[0])*((blk_num%SMNum)-(ref_base_blk_nums[0]%SMNum))/(ref_base_blk_nums[1]-ref_base_blk_nums[0])+base_Es[0]
	if E < 0:
		return 1e10
	if op_type == 'dense':
		return get_product([rep_layout[i]*micK_Sshape[i] for i in range(len(rep_layout))] + list(mick_shape[2:]))*2/E/1e9
	elif op_type == 'conv2d':
		return get_product([rep_layout[i]*micK_Sshape[i] for i in range(len(rep_layout))] + list(mick_shape[4:]))*2/E/1e9

	


def my_cost_model_fix_reduc(func_micK_curvs, popts, curve_rep_layouts, interested_blkNs, mick_shape, task_shape, op_type, reduc_keys):
	'''
	compute the cost in terms of s.
	INPUT:
		func_micK_curvs: a dict of functions which fit the micro-kernel latency curves: 
			for each block number (5 in total, from small to large) and each block layout (2 in total, upper bound first, lower bound curve next), 
			there are 2 curve (1 for lower part, 1 for upper part, 20 in total).
		popts: a dict of parameters for the corr. functions.
		curve_rep_layouts: a dict of replication layouts, the replication layout corr. to each func_micK_curv.
		mick_shape: a list of int, the given micro-kernel shape, including reduction part.
		task_shape: a list of int, the given task shape.
		reduc_keys: tuple. the setting of reduc_len and reduc_rep_num in the curves we will query.
	'''
	# first compute the block number
	SMNum = 108
	red_len, reduc_rep_num = reduc_keys
	micK_Sshape = None
	if op_type == 'dense':
		micK_Sshape = mick_shape[:2]
	elif op_type == 'conv2d':
		micK_Sshape = mick_shape[:4]
	out_size = get_product(micK_Sshape)
	rep_layout = [math.ceil(task_shape[i] / micK_Sshape[i]) for i in range(len(micK_Sshape))]
	blk_num = get_product(rep_layout)
	# the max number of thread blocks assigned to one SM
	idx = math.ceil(blk_num / SMNum) - 1
	# real_base_blk_nums = [idx*SMNum+1, (idx+1)*SMNum]
	if idx > 4:
		idx = 4
	Es = dict()
	# ref_base_blk_nums = [idx*SMNum+1, (idx+1)*SMNum]
	ref_base_blk_nums = interested_blkNs[idx]
	# rep_keys = ['line', 'square']
	rep_keys = ['line_h', 'line_v', 'square']
	# the two base micro-kernel shapes
	base_mick_shape1, base_mick_shape2 = None, None
	min_mick_Sshape = get_min_DataRead_mick_shape(out_size, False, op_type)
	if op_type == 'dense':
		base_mick_shape1 = min_mick_Sshape+(red_len,)
		base_mick_shape2 = (49152 / 4) # if it is a value, then the value is data read amount per block #(out_size/8, 8, red_len)
	elif op_type == 'conv2d':
		base_mick_shape1 = min_mick_Sshape+(64, 7, 7,)
		base_mick_shape2 = (8, out_size/8, 1, 1, 64, 7, 7,)
	# 
	for base_blk_i in range(len(ref_base_blk_nums)):
		base_blk = ref_base_blk_nums[base_blk_i]
		# print("base_blk: ", base_blk)
		for rep_key in rep_keys:
			key1 = reduc_keys + (base_blk, rep_key, 'up')
			key2 = reduc_keys + (base_blk, rep_key, 'lw')
			# print(f"key1 is {key1}, key2 is {key2}")
			E = comp_efficiency_given_rep_and_mick_on_curve(func_micK_curvs, popts, curve_rep_layouts, 
				key1, key2, base_mick_shape1, base_mick_shape2, mick_shape, op_type)
			Es[(base_blk, rep_key)] = E
	# print(reduc_keys, '='*50)
	# print(Es)
	# 
	# comp the efficiency for the target rep_layout of the target micro-kernel
	base_repLs1 = list()
	base_repLs2 = list()
	# base_repNs2 = [1 for base_blk in ref_base_blk_nums]
	for base_blk in ref_base_blk_nums:
		# tmp = sorted(get_factors(base_blk), key=lambda repN: repN*mick_shape[0]+(base_blk//repN)*mick_shape[1])
		# # base_repNs1.append(tmp[0])
		# base_repNs1.append((base_blk*mick_shape[1]/mick_shape[0])**0.5) # consider the float best repN
		# base_repNs2.append(tmp[-1])
		base_repLs1.append(get_min_DataRead_rep_layout(base_blk, micK_Sshape, op_type))
		base_repLs2.append(get_max_DataRead_rep_layout(base_blk, micK_Sshape, op_type))
	# 
	# base_datas1 = [comp_data_read_amount(mick_shape, (base_repNs1[i], ref_base_blk_nums[i]//base_repNs1[i])) for i in range(2)]
	# base_datas2 = [comp_data_read_amount(mick_shape, (base_repNs2[i], ref_base_blk_nums[i]//base_repNs2[i])) for i in range(2)]
	base_datas1 = [comp_data_read_amount(mick_shape, base_repLs1[i], op_type) for i in range(2)]
	base_datas2 = [comp_data_read_amount(mick_shape, base_repLs2[i], op_type) for i in range(2)]
	# 
	datas_sqr = [comp_data_read_amount(mick_shape, curve_rep_layouts[reduc_keys + (base_blk, 'square', 'up')], op_type) for base_blk in ref_base_blk_nums]
	datas_h = [comp_data_read_amount(mick_shape, curve_rep_layouts[reduc_keys + (base_blk, 'line_h', 'up')], op_type) for base_blk in ref_base_blk_nums]
	datas_v = [comp_data_read_amount(mick_shape, curve_rep_layouts[reduc_keys + (base_blk, 'line_v', 'up')], op_type) for base_blk in ref_base_blk_nums]
	# if True in [base_datas1[i] == base_datas2[i] for i in range(2)]:
	# 	return 1e10
	if True in [(datas_sqr[i] == datas_h[i]) and (datas_sqr[i] == datas_v[i]) for i in range(2)]: # this case is invalid because datas_sqr may not be the min
		return 1e10
	# 
	# normed_datas = [(datas[i] - base_datas2[i]) / (base_datas1[i] - base_datas2[i]) for i in range(2)]
	# if 0 in normed_datas:
	# 	return 1e10
	normed_datas1 = list()
	normed_datas2 = list()
	base_Es1 = list()
	base_Es2 = list() 
	for i in range(2):
		tmp_data = [datas_sqr[i], datas_h[i], datas_v[i]]
		tmp_labels = ['square', 'line_h', 'line_v']
		sorted_idx = sorted(range(3), key=lambda j: tmp_data[j])
		# print(f"sorted_layouts: {sorted_idx}")
		normed_datas1.append((tmp_data[sorted_idx[0]] - base_datas2[i]) / (base_datas1[i] - base_datas2[i]))
		normed_datas2.append((tmp_data[sorted_idx[-1]] - base_datas2[i]) / (base_datas1[i] - base_datas2[i]))
		base_Es1.append(Es[(ref_base_blk_nums[i], tmp_labels[sorted_idx[0]])])
		base_Es2.append(Es[(ref_base_blk_nums[i], tmp_labels[sorted_idx[-1]])])
		if base_Es1[-1]<base_Es2[-1]:
			base_Es2[-1], base_Es1[-1] = base_Es1[-1], base_Es2[-1]
		# if base_Es1[-1]<(base_Es2[-1]*0.9): # we assume base_Es1 should be higher than base_Es2
		# 	return 1e10
	if min(base_Es1+base_Es1) < 0:
		return 1e10
	# print(f"normed_datas1:{normed_datas1}")
	# print(f"normed_datas2:{normed_datas2}")
	# print(f"base_Es1:{base_Es1}")
	# print(f"base_Es2:{base_Es2}")
	# 
	# base_Es1 = [Es[(base_blk, 'square')] for base_blk in ref_base_blk_nums]
	# base_Es2 = [Es[(base_blk, 'line')] for base_blk in ref_base_blk_nums]
	# 
	# tmp = sorted(get_factors(blk_num), key=lambda repN: repN*mick_shape[0]+(blk_num//repN)*mick_shape[1])
	# # base_repN1 = tmp[0]
	# base_repN1 = (blk_num*mick_shape[1]/mick_shape[0])**0.5 # consider the float best repN
	# base_repN2 = tmp[-1]
	base_repL1 = get_min_DataRead_rep_layout(blk_num, micK_Sshape, op_type)
	base_repL2 = get_max_DataRead_rep_layout(blk_num, micK_Sshape, op_type)
	# base_data1 = comp_data_read_amount(mick_shape, (base_repN1, blk_num//base_repN1))
	# base_data2 = comp_data_read_amount(mick_shape, (base_repN2, blk_num//base_repN2))
	base_data1 = comp_data_read_amount(mick_shape, base_repL1, op_type)
	base_data2 = comp_data_read_amount(mick_shape, base_repL2, op_type)
	# 
	data = comp_data_read_amount(mick_shape, rep_layout, op_type)
	if base_data1 == base_data2:
		# return 1e10
		normed_data = 0 # we assume the bottom points are always on a line in one segment of the rep layout curve
	else:
		normed_data = (data - base_data2) / (base_data1 - base_data2)
	# print(f"normed_data:{normed_data}")
	base_Es = [1/((1/base_Es1[i]-1/base_Es2[i])*(normed_data-normed_datas2[i])/(normed_datas1[i]-normed_datas2[i])+1/base_Es2[i]) for i in range(2)]
	# print(f"base_Es:{base_Es}")
	if min(base_Es) < 0:
		return 1e10
	if base_Es[1]<base_Es[0]:
		base_Es[0], base_Es[1] = base_Es[1], base_Es[0]
	# if base_Es[1]<base_Es[0]*0.9: # we assume base_Es[0] < base_Es[1]
	# 	return 1e10
	# print((real_base_blk_nums[1]-real_base_blk_nums[0]))
	# E = (base_Es[1]-base_Es[0])*((blk_num%SMNum)-(ref_base_blk_nums[0]%SMNum))/(ref_base_blk_nums[1]-ref_base_blk_nums[0])+base_Es[0]
	base_compute_amounts = list()
	if op_type == 'dense':
		# compute_amount = get_product(mick_shape)*blk_num*reduc_rep_num*2
		delta_base_blk_nums = [ref_base_blk_nums[i]-(math.ceil(ref_base_blk_nums[i]/SMNum)-1)*SMNum for i in range(2)]
		base_compute_amounts = [get_product(mick_shape)*((math.ceil(blk_num/SMNum)-1)*SMNum+delta_base_blk_nums[i])*reduc_rep_num*2 for i in range(2)]
	latency = sum([base_compute_amounts[i] / base_Es[i] / 1e9 for i in range(2)])/2
	if latency < 0:
		return 1e10
	return latency
	# if E < 0:
	# 	return 1e10
	# if op_type == 'dense':
	# 	# print(get_product([rep_layout[i]*micK_Sshape[i] for i in range(len(rep_layout))] + list(mick_shape[2:]))*reduc_rep_num*2, E)
	# 	return get_product([rep_layout[i]*micK_Sshape[i] for i in range(len(rep_layout))] + list(mick_shape[2:]))*reduc_rep_num*2/E/1e9
	# elif op_type == 'conv2d':
	# 	return get_product([rep_layout[i]*micK_Sshape[i] for i in range(len(rep_layout))] + list(mick_shape[4:]))*reduc_rep_num*2/E/1e9






def my_cost_model_fix_reduc_2Rngs(selected_fixedL_dict, func_micK_curvs, popts, curve_rep_layouts, interested_blkNs, 
	mick_shape, task_shape, op_type, reduc_keys, min_DataRead_mSsp_cache, min_DataRead_repl_cache):
	'''
	compute the cost in terms of s.
	INPUT:
		selected_fixedL_dict: a dict mapping the fixed len of the first insteresting mick shape of lower curve to the corr. red_len.
		func_micK_curvs: a dict of functions which fit the micro-kernel latency curves: 
			for each block number (5 in total, from small to large) and each block layout (2 in total, upper bound first, lower bound curve next), 
			there are 2 curve (1 for lower part, 1 for upper part, 20 in total).
		popts: a dict of parameters for the corr. functions.
		curve_rep_layouts: a dict of replication layouts, the replication layout corr. to each func_micK_curv.
		mick_shape: a list of int, the given micro-kernel shape, including reduction part.
		task_shape: a list of int, the given task shape.
		reduc_keys: tuple. the setting of reduc_len and reduc_rep_num in the curves we will query.
		min_DataRead_mSsp_cache, min_DataRead_repl_cache: two caches for get_min_DataRead_mick_shape and get_min_DataRead_rep_layout respectively.
	CACHE INFOR: key of the result cache for this func should be (mick_shape, Srepl, reduc_keys)

	'''
	# first compute the block number
	SMNum = 108
	Rsp, reduc_rep_num = reduc_keys
	red_len = get_product(Rsp)
	micK_Sshape = None
	sp_len = len(mick_shape)
	if op_type == 'dense':
		micK_Sshape = mick_shape[:2]
	elif op_type == 'conv2d':
		micK_Sshape = mick_shape[:4]
		sp_len = 7
	elif op_type in ['bmm', 'bmm_nn']:
		micK_Sshape = mick_shape[:3]
	out_size = get_product(micK_Sshape)
	rep_layout = [math.ceil(task_shape[i] / micK_Sshape[i]) for i in range(len(micK_Sshape))]
	blk_num = get_product(rep_layout)
	# the max number of thread blocks assigned to one SM
	idx = math.ceil(blk_num / SMNum) - 1
	# real_base_blk_nums = [idx*SMNum+1, (idx+1)*SMNum]
	if idx > 4:
		idx = 4
	Es = dict()
	# ref_base_blk_nums = [idx*SMNum+1, (idx+1)*SMNum]
	ref_base_blk_nums = interested_blkNs[idx]
	# rep_keys = ['line', 'square']
	rep_keys = ['line_h', 'line_v', 'square']
	if op_type in ['bmm', 'bmm_nn']:
		rep_keys = ['line_b', 'square']
	# 
	fixed_len = None
	standard_other_params=tuple()
	if op_type in ['conv2d']:
		fixed_len = selected_fixedL_dict[Rsp]
		standard_other_params = ((1,1),0,(1,1)) # stride, padding, dilation
	else:
		fixed_len = selected_fixedL_dict[red_len]
		reduc_keys = red_len, reduc_rep_num
	# 
	# the two base micro-kernel shapes
	base_mick_shape1, base_mick_shape2 = None, None
	min_mick_Sshape = get_min_DataRead_mick_shape(out_size, False, op_type, mick_shape[:sp_len]+standard_other_params, min_DataRead_mSsp_cache)
	if op_type == 'dense':
		base_mick_shape1 = min_mick_Sshape+(red_len,)
		base_mick_shape2 = (49152 / 4) # if it is a value, then the value is data read amount per block #(out_size/8, 8, red_len)
		# check whether this mick space size is in the range covered by the fixed_len interesting mick shape 
		if comp_data_read_amount([fixed_len, out_size / fixed_len, red_len], (1, 1), op_type) <= (49152 / 4):
			base_mick_shape2 = (fixed_len, out_size / fixed_len, red_len)
		else:
			fixed_len = 'inf'
	elif op_type in ['bmm', 'bmm_nn']:
		base_mick_shape1 = min_mick_Sshape+(red_len,)
		base_mick_shape2 = (49152 / 4) # if it is a value, then the value is data read amount per block #(out_size/8, 8, red_len)
		# check whether this mick space size is in the range covered by the fixed_len interesting mick shape 
		if comp_data_read_amount([1, fixed_len, out_size / fixed_len, red_len], (1, 1, 1), op_type) <= (49152 / 4):
			base_mick_shape2 = (1, fixed_len, out_size / fixed_len, red_len)
		else:
			fixed_len = 'inf'
	elif op_type == 'conv2d':
		base_mick_shape1 = min_mick_Sshape + mick_shape[4:7]+standard_other_params
		base_mick_shape2 = (49152 / 4) # if it is a value, then the value is data read amount per block #(out_size/8, 8, red_len)
		# check whether this mick space size is in the range covered by the fixed_len interesting mick shape 
		# if comp_data_read_amount((fixed_len, out_size / fixed_len, 1, 1) + mick_shape[4:], (1, 1, 1, 1), op_type) <= (49152 / 4):
			# base_mick_shape2 = (fixed_len, out_size / fixed_len, 1, 1) + mick_shape[4:]
		# if comp_data_read_amount((1, fixed_len, 1, out_size / fixed_len) + mick_shape[4:], (1, 1, 1, 1), op_type) <= (49152 / 4):
		# 	base_mick_shape2 = (1, fixed_len, 1, out_size / fixed_len) + mick_shape[4:]
		if comp_data_read_amount((1, out_size / fixed_len, 1, fixed_len) + mick_shape[4:7]+standard_other_params, (1, 1, 1, 1), op_type) <= (49152 / 4):
			base_mick_shape2 = (1, out_size / fixed_len, 1, fixed_len) + mick_shape[4:7]+standard_other_params
		else:
			fixed_len = 'inf'
	# 
	for base_blk_i in range(len(ref_base_blk_nums)):
		base_blk = ref_base_blk_nums[base_blk_i]
		# print("base_blk: ", base_blk)
		for rep_key in rep_keys:
			key1 = reduc_keys + (base_blk, rep_key, 'up', 'inf')
			key2 = reduc_keys + (base_blk, rep_key, 'lw', fixed_len)
			# print(f"key1 is {key1}, key2 is {key2}")
			E = comp_efficiency_given_rep_and_mick_on_curve_2Rngs(func_micK_curvs, popts, curve_rep_layouts, 
				key1, key2, base_mick_shape1, base_mick_shape2, mick_shape, op_type)
			Es[(base_blk, rep_key)] = E
	# print(reduc_keys, '='*50)
	# print(Es)
	# 
	# comp the efficiency for the target rep_layout of the target micro-kernel
	base_repLs1 = list()
	base_repLs2 = list()
	# base_repNs2 = [1 for base_blk in ref_base_blk_nums]
	for base_blk in ref_base_blk_nums:
		# tmp = sorted(get_factors(base_blk), key=lambda repN: repN*mick_shape[0]+(base_blk//repN)*mick_shape[1])
		# # base_repNs1.append(tmp[0])
		# base_repNs1.append((base_blk*mick_shape[1]/mick_shape[0])**0.5) # consider the float best repN
		# base_repNs2.append(tmp[-1])
		base_repLs1.append(get_min_DataRead_rep_layout(base_blk, micK_Sshape, op_type, mick_shape, min_DataRead_repl_cache))
		base_repLs2.append(get_max_DataRead_rep_layout(base_blk, micK_Sshape, op_type, mick_shape))
	# 
	# base_datas1 = [comp_data_read_amount(mick_shape, (base_repNs1[i], ref_base_blk_nums[i]//base_repNs1[i])) for i in range(2)]
	# base_datas2 = [comp_data_read_amount(mick_shape, (base_repNs2[i], ref_base_blk_nums[i]//base_repNs2[i])) for i in range(2)]
	base_datas1 = [comp_data_read_amount(mick_shape, base_repLs1[i], op_type) for i in range(2)]
	base_datas2 = [comp_data_read_amount(mick_shape, base_repLs2[i], op_type) for i in range(2)]
	# 
	datas_sqr, datas_h, datas_v, datas_b = None, None, None, None
	if op_type in ['dense', 'conv2d']:
		datas_sqr = [comp_data_read_amount(mick_shape, curve_rep_layouts[reduc_keys + (base_blk, 'square', 'up', 'inf')], op_type) for base_blk in ref_base_blk_nums]
		datas_h = [comp_data_read_amount(mick_shape, curve_rep_layouts[reduc_keys + (base_blk, 'line_h', 'up', 'inf')], op_type) for base_blk in ref_base_blk_nums]
		datas_v = [comp_data_read_amount(mick_shape, curve_rep_layouts[reduc_keys + (base_blk, 'line_v', 'up', 'inf')], op_type) for base_blk in ref_base_blk_nums]
		if True in [(datas_sqr[i] == datas_h[i]) and (datas_sqr[i] == datas_v[i]) for i in range(2)]: # this case is invalid because datas_sqr may not be the min
			return 1e10
	elif op_type in ['bmm','bmm_nn']:
		datas_sqr = [comp_data_read_amount(mick_shape, curve_rep_layouts[reduc_keys + (base_blk, 'square', 'up', 'inf')], op_type) for base_blk in ref_base_blk_nums]
		datas_b = [comp_data_read_amount(mick_shape, curve_rep_layouts[reduc_keys + (base_blk, 'line_b', 'up', 'inf')], op_type) for base_blk in ref_base_blk_nums]
		if True in [(datas_sqr[i] == datas_b[i]) for i in range(2)]: # this case is invalid because datas_sqr may not be the min
			return 1e10
	# if True in [base_datas1[i] == base_datas2[i] for i in range(2)]:
	# 	return 1e10
	# 
	# normed_datas = [(datas[i] - base_datas2[i]) / (base_datas1[i] - base_datas2[i]) for i in range(2)]
	# if 0 in normed_datas:
	# 	return 1e10
	normed_datas1 = list()
	normed_datas2 = list()
	base_Es1 = list()
	base_Es2 = list() 
	for i in range(2):
		tmp_data, tmp_labels = None, None
		if op_type in ['dense', 'conv2d']:
			tmp_data = [datas_sqr[i], datas_h[i], datas_v[i]]
			tmp_labels = ['square', 'line_h', 'line_v']
		elif op_type in ['bmm','bmm_nn']:
			tmp_data = [datas_sqr[i], datas_b[i]]
			tmp_labels = ['square', 'line_b']
		sorted_idx = sorted(range(len(tmp_labels)), key=lambda j: tmp_data[j])
		# print(f"sorted_layouts: {sorted_idx}")
		normed_datas1.append((tmp_data[sorted_idx[0]] - base_datas2[i]) / (base_datas1[i] - base_datas2[i]))
		normed_datas2.append((tmp_data[sorted_idx[-1]] - base_datas2[i]) / (base_datas1[i] - base_datas2[i]))
		base_Es1.append(Es[(ref_base_blk_nums[i], tmp_labels[sorted_idx[0]])])
		base_Es2.append(Es[(ref_base_blk_nums[i], tmp_labels[sorted_idx[-1]])])
		if base_Es1[-1]<base_Es2[-1]:
			base_Es2[-1], base_Es1[-1] = base_Es1[-1], base_Es2[-1]
		# if base_Es1[-1]<(base_Es2[-1]*0.9): # we assume base_Es1 should be higher than base_Es2
		# 	return 1e10
	if min(base_Es1+base_Es1) < 0:
		return 1e10
	# print(f"normed_datas1:{normed_datas1}")
	# print(f"normed_datas2:{normed_datas2}")
	# print(f"base_Es1:{base_Es1}")
	# print(f"base_Es2:{base_Es2}")
	# 
	# base_Es1 = [Es[(base_blk, 'square')] for base_blk in ref_base_blk_nums]
	# base_Es2 = [Es[(base_blk, 'line')] for base_blk in ref_base_blk_nums]
	# 
	# tmp = sorted(get_factors(blk_num), key=lambda repN: repN*mick_shape[0]+(blk_num//repN)*mick_shape[1])
	# # base_repN1 = tmp[0]
	# base_repN1 = (blk_num*mick_shape[1]/mick_shape[0])**0.5 # consider the float best repN
	# base_repN2 = tmp[-1]
	base_repL1 = get_min_DataRead_rep_layout(blk_num, micK_Sshape, op_type, mick_shape, min_DataRead_repl_cache)
	base_repL2 = get_max_DataRead_rep_layout(blk_num, micK_Sshape, op_type, mick_shape)
	# base_data1 = comp_data_read_amount(mick_shape, (base_repN1, blk_num//base_repN1))
	# base_data2 = comp_data_read_amount(mick_shape, (base_repN2, blk_num//base_repN2))
	base_data1 = comp_data_read_amount(mick_shape, base_repL1, op_type)
	base_data2 = comp_data_read_amount(mick_shape, base_repL2, op_type)
	# 
	data = comp_data_read_amount(mick_shape, rep_layout, op_type)
	if base_data1 == base_data2:
		# return 1e10
		normed_data = 0 # we assume the bottom points are always on a line in one segment of the rep layout curve
	else:
		normed_data = (data - base_data2) / (base_data1 - base_data2)
	# print(f"normed_data:{normed_data}")
	# if (0 in (base_Es1+base_Es2)) or (0 in [normed_datas1[i]-normed_datas2[i] for i in range(2)]):
	# 	print(base_Es1, base_Es2, normed_datas1, normed_datas2)
	try:
		base_Es = [1/((1/base_Es1[i]-1/base_Es2[i])*(normed_data-normed_datas2[i])/(normed_datas1[i]-normed_datas2[i])+1/base_Es2[i]) for i in range(2)]
	except Exception as e:
		print(reduc_keys, mick_shape, task_shape, base_Es1, base_Es2, normed_datas1, normed_datas2, normed_data)
		assert False
		raise e
	# print(f"base_Es:{base_Es}")
	if min(base_Es) < 0:
		return 1e10
	if base_Es[1]<base_Es[0]:
		base_Es[0], base_Es[1] = base_Es[1], base_Es[0]
	# if base_Es[1]<base_Es[0]*0.9: # we assume base_Es[0] < base_Es[1]
	# 	return 1e10
	# print((real_base_blk_nums[1]-real_base_blk_nums[0]))
	# E = (base_Es[1]-base_Es[0])*((blk_num%SMNum)-(ref_base_blk_nums[0]%SMNum))/(ref_base_blk_nums[1]-ref_base_blk_nums[0])+base_Es[0]
	base_compute_amounts = list()
	if op_type in ['dense', 'bmm', 'bmm_nn', 'conv2d']:
		# compute_amount = get_product(mick_shape)*blk_num*reduc_rep_num*2
		delta_base_blk_nums = [ref_base_blk_nums[i]-(math.ceil(ref_base_blk_nums[i]/SMNum)-1)*SMNum for i in range(2)]
		base_compute_amounts = [out_size*red_len*((math.ceil(blk_num/SMNum)-1)*SMNum+delta_base_blk_nums[i])*reduc_rep_num*2 for i in range(2)]
	latency = sum([base_compute_amounts[i] / base_Es[i] / 1e9 for i in range(2)])/2
	if latency < 0:
		return 1e10
	return latency
	# if E < 0:
	# 	return 1e10
	# if op_type == 'dense':
	# 	# print(get_product([rep_layout[i]*micK_Sshape[i] for i in range(len(rep_layout))] + list(mick_shape[2:]))*reduc_rep_num*2, E)
	# 	return get_product([rep_layout[i]*micK_Sshape[i] for i in range(len(rep_layout))] + list(mick_shape[2:]))*reduc_rep_num*2/E/1e9
	# elif op_type == 'conv2d':
	# 	return get_product([rep_layout[i]*micK_Sshape[i] for i in range(len(rep_layout))] + list(mick_shape[4:]))*reduc_rep_num*2/E/1e9




def my_cost_model_full_dynamic(func_micK_curvs, popts, curve_rep_layouts, interested_blkNs, mick_shape, task_shape, op_type):
	'''
	compute the cost in terms of s.
	INPUT:
		func_micK_curvs: a dict of functions which fit the micro-kernel latency curves: 
			for each block number (5 in total, from small to large) and each block layout (2 in total, upper bound first, lower bound curve next), 
			there are 2 curve (1 for lower part, 1 for upper part, 20 in total).
		popts: a dict of parameters for the corr. functions.
		curve_rep_layouts: a dict of replication layouts, the replication layout corr. to each func_micK_curv.
		mick_shape: a list of int, the given micro-kernel shape, including reduction part.
		task_shape: a list of int, the given task shape.
	'''
	# we consider varying reduction loops
	# now, for dense op, we only get curves for reduc len in [12, 18], and reduc rep len in [4, 32]
	mick_shape = tuple(mick_shape)
	micK_Sshape = None
	if op_type == 'dense':
		micK_Sshape = mick_shape[:2]
	red_len = None
	if op_type == 'dense':
		red_len = mick_shape[2]
	reduc_rep_num = None
	if op_type == 'dense':
		reduc_rep_num = math.ceil(task_shape[2] / red_len)
	if data_read_amount_PerBlk(op_type, micK_Sshape, rc = red_len) > (49152 / 4): # not satisfy the shared memory constraint
		return 1e10
	# 
	base_red_lens = [6, 12, 24, 36, 48, 60] #[12, 24, 36] #[12, 18]
	base_reduc_rep_nums = [1, 2, 4, 8, 16, 32, 64] #[8, 16, 32] #[4, 32]
	if red_len < base_red_lens[0]:
		return 1e10
	base_red_len1, base_red_len2 = None, None
	for i in range(len(base_red_lens)-1):
		if red_len in range(base_red_lens[i], base_red_lens[i+1]):
			base_red_len1, base_red_len2 = base_red_lens[i], base_red_lens[i+1]
			break
	if red_len < base_red_lens[0]:
		base_red_len1, base_red_len2 = base_red_lens[:2]
	elif red_len >= base_red_lens[-1]:
		base_red_len1, base_red_len2 = base_red_lens[-2:]
	# 
	base_reduc_rep_num1, base_reduc_rep_num2 = None, None
	for i in range(len(base_reduc_rep_nums)-1):
		if reduc_rep_num in range(base_reduc_rep_nums[i], base_reduc_rep_nums[i+1]):
			base_reduc_rep_num1, base_reduc_rep_num2 = base_reduc_rep_nums[i], base_reduc_rep_nums[i+1]
			break
	if reduc_rep_num < base_reduc_rep_nums[0]:
		base_reduc_rep_num1, base_reduc_rep_num2 = base_reduc_rep_nums[:2]
	elif reduc_rep_num >= base_reduc_rep_nums[-1]:
		base_reduc_rep_num1, base_reduc_rep_num2 = base_reduc_rep_nums[-2:]
	# 
	latencys = dict()
	# 0. get the latency of different base reduc setting.
	for base_reduc_rep_num in [base_reduc_rep_num1, base_reduc_rep_num2]: #base_reduc_rep_nums:
		for base_red_len in [base_red_len1, base_red_len2]:# base_red_lens:
			reduc_keys = (base_red_len, base_reduc_rep_num)
			base_mick_shape = None
			if op_type == 'dense':
				base_mick_shape = mick_shape[:2] + (base_red_len, )
			cost = my_cost_model_fix_reduc(func_micK_curvs, popts, curve_rep_layouts, interested_blkNs, base_mick_shape, task_shape, op_type, reduc_keys)
			latencys[reduc_keys] = cost
	# for k,v in latencys.items():
	# 	print(k,v)
	# 1. compute the latency of the micro-kernel with two base_reduc_rep_nums.
	base_latencys = list()
	# if red_len < 24:
	# 	base_red_len1, base_red_len2 = 12, 24
	# else:
	# 	base_red_len1, base_red_len2 = 24, 36
	for base_reduc_rep_num in [base_reduc_rep_num1, base_reduc_rep_num2]: #base_reduc_rep_nums:
		latency1 = latencys[(base_red_len1, base_reduc_rep_num)]
		latency2 = latencys[(base_red_len2, base_reduc_rep_num)]
		# if latency2 < latency1 * 0.9: # we assume latency2 is higher than latency1
			# return 1e10
		if (red_len>base_red_len2) and (latency2 < latency1):
			latency1, latency2 = latency2, latency1
		base_latencys.append((latency2 - latency1) * (red_len-base_red_len1) / (base_red_len2-base_red_len1) + latency1)
	# print(f"base_latencys:{base_latencys}")
	if min(base_latencys)<=0:
		return 1e10
	# 2. compute the latency of the micro-kernel with the required reduc_rep_num.
	# reduc_rep_num = None
	# if op_type == 'dense':
	# 	reduc_rep_num = math.ceil(task_shape[2] / red_len)
	# if reduc_rep_num<8:
	# 	return 1e10
	# elif reduc_rep_num < 16:
	# 	base_reduc_rep_num1, base_reduc_rep_num2 = 8, 16
	# 	base_latency1, base_latency2 = base_latencys[0], base_latencys[1]
	# else:
	# 	base_reduc_rep_num1, base_reduc_rep_num2 = 16, 32
	# 	base_latency1, base_latency2 = base_latencys[1], base_latencys[2]
	base_latency1, base_latency2 = base_latencys
	# if base_latency2 < base_latency1 * 0.9: # we assume base_latency2 is higher than base_latency1
	# 	return 1e10
	if (reduc_rep_num > base_reduc_rep_num2) and (base_latency1>base_latency2):
		base_latency1, base_latency2 = base_latency2, base_latency1
	# print(base_latency2-base_latency1, reduc_rep_num - base_reduc_rep_num1, base_reduc_rep_num2 - base_reduc_rep_num1)
	latency = (base_latency2-base_latency1) * (reduc_rep_num - base_reduc_rep_num1) / (base_reduc_rep_num2 - base_reduc_rep_num1) + base_latency1
	if latency <= 0:
		return 1e10
	return latency
			




def infer_from_linear_func(base_xs, base_ys, x):
	'''
		This function computes the y value corresponding to the x value based on the linear function defined by the base_xs and base_ys.
		INPUT:
			base_xs: list of two x values.
			base_ys: list of two y values.
		OUTPUT:
			the y value corr. to x.
	'''
	# assert base_xs[1] != base_xs[0], f"infer_from_linear_func error infor: {(base_xs, base_ys, x)}"
	return (base_ys[1]-base_ys[0]) * (x - base_xs[0]) / (base_xs[1] - base_xs[0]) + base_ys[0]





def my_cost_model_full_dynamic_2Rngs(selected_fixedL_dict, func_micK_curvs, popts, curve_rep_layouts, interested_blkNs, mick_shape, task_shape, op_type,
	interested_Rsps, fix_reduc_2Rngs_cache = dict(), min_DataRead_mSsp_caches=dict(), min_DataRead_repl_caches=dict()):
	'''
	Note: the curve fitting of lower part has two ranges.
	compute the cost in terms of s.
	INPUT:
		selected_fixedL_dict: a dict mapping the fixed len of the first insteresting mick shape of lower curve to the corr. red_len.
		func_micK_curvs: a dict of functions which fit the micro-kernel latency curves: 
			for each block number (5 in total, from small to large) and each block layout (2 in total, upper bound first, lower bound curve next), 
			there are 2 curve (1 for lower part, 1 for upper part, 20 in total).
		popts: a dict of parameters for the corr. functions.
		curve_rep_layouts: a dict of replication layouts, the replication layout corr. to each func_micK_curv.
		mick_shape: a list of int, the given micro-kernel shape, including reduction part.
		task_shape: a list of int, the given task shape.
		interested_Rsps: dict {red_len: list of base reduc_shapes}. Each reduc_shape should be a tuple.	
	CACHE: fix_reduc_2Rngs_cache: store the result of calling my_cost_model_fix_reduc_2Rngs to reduce cost computation time.
		The cache key is (micK_Sshape, Srepl, base_Rsp, base_reduc_rep_num), the cache is per mick_shape[sp_len:] (i.e., other parameters)
		min_DataRead_mSsp_caches, min_DataRead_repl_caches: used for my_cost_model_fix_reduc_2Rngs, each is a dict of cache dict.
	'''
	# we consider varying reduction loops
	# now, for dense op, we only get curves for reduc len in [12, 18], and reduc rep len in [4, 32]
	mick_shape = tuple(mick_shape)
	micK_Sshape, red_len, mick_Rshape, reduc_rep_num = None, None, None, None
	sp_len = len(mick_shape)
	Ssp_len = sp_len - 1
	if op_type == 'dense':
		micK_Sshape = mick_shape[:2]
		red_len = mick_shape[2]
		reduc_rep_num = math.ceil(task_shape[2] / red_len)
	elif op_type in ['bmm', 'bmm_nn']:
		micK_Sshape = mick_shape[:3]
		red_len = mick_shape[3]
		reduc_rep_num = math.ceil(task_shape[3] / red_len)
	elif op_type == 'conv2d':
		micK_Sshape = mick_shape[:4]
		mick_Rshape = mick_shape[4:7]
		red_len = get_product(mick_Rshape)
		reduc_rep_num = get_product([math.ceil(task_shape[i] / mick_shape[i]) for i in range(4, 7)])
		sp_len = 7
		Ssp_len = 4
	Srepl = tuple([math.ceil(task_shape[i] / mick_shape[i]) for i in range(Ssp_len)])
	# 
	dataInPerBlk = data_read_amount_PerBlk(op_type, mick_shape)
	if dataInPerBlk > (49152 / 4): # not satisfy the shared memory constraint
		return 1e10
	# 
	base_red_lens = [6] + list(range(12, 128, 12)) + [128] # [6, 12, 24, 36, 48, 60, 72, ]
	# base_red_lens = [6] + list(range(12, 73, 12)) # + [128] # [6, 12, 24, 36, 48, 60, 72, ]
	base_reduc_rep_nums = [2**i for i in range(7)]
	if red_len < base_red_lens[0]:
		return 1e10
	base_red_len1, base_red_len2 = None, None
	for i in range(len(base_red_lens)-1):
		if red_len in range(base_red_lens[i], base_red_lens[i+1]):
			base_red_len1, base_red_len2 = base_red_lens[i], base_red_lens[i+1]
			break
	if red_len < base_red_lens[0]:
		base_red_len1, base_red_len2 = base_red_lens[:2]
	elif red_len >= base_red_lens[-1]:
		base_red_len1, base_red_len2 = base_red_lens[-2:]
	# 
	base_reduc_rep_num1, base_reduc_rep_num2 = None, None
	for i in range(len(base_reduc_rep_nums)-1):
		if reduc_rep_num in range(base_reduc_rep_nums[i], base_reduc_rep_nums[i+1]):
			base_reduc_rep_num1, base_reduc_rep_num2 = base_reduc_rep_nums[i], base_reduc_rep_nums[i+1]
			break
	if reduc_rep_num < base_reduc_rep_nums[0]:
		base_reduc_rep_num1, base_reduc_rep_num2 = base_reduc_rep_nums[:2]
	elif reduc_rep_num >= base_reduc_rep_nums[-1]:
		base_reduc_rep_num1, base_reduc_rep_num2 = base_reduc_rep_nums[-2:]
	# 
	latencys = dict()
	# 0. get the latency of different base reduc setting.
	for base_reduc_rep_num in [base_reduc_rep_num1, base_reduc_rep_num2]: #base_reduc_rep_nums:
		for base_red_len in [base_red_len1, base_red_len2]:# base_red_lens:
			for base_Rsp in interested_Rsps[base_red_len]:
				reduc_keys = (base_Rsp, base_reduc_rep_num)
				base_mick_shape = micK_Sshape + base_Rsp + mick_shape[sp_len:]
				if data_read_amount_PerBlk(op_type, base_mick_shape) > (49152 / 4):
					return 1e10
				# 
				# base_mick_shape = None
				# if op_type in ['dense', 'bmm']:
				# 	base_mick_shape = micK_Sshape + (base_red_len, )
				# 	if data_read_amount_PerBlk(op_type, micK_Sshape, rc = base_red_len) > (49152 / 4):
				# 		return 1e10
				# we check the cache first
				if (tuple(micK_Sshape), Srepl, tuple(base_Rsp), base_reduc_rep_num) in fix_reduc_2Rngs_cache:
					cost = fix_reduc_2Rngs_cache[(tuple(micK_Sshape), Srepl, tuple(base_Rsp), base_reduc_rep_num)]
					if cost == 'fail':
						print(mick_shape, task_shape, base_reduc_rep_num, base_red_len, base_Rsp)
						assert False
					else:
						latencys[reduc_keys] = cost
						continue
				try:
					if base_Rsp not in min_DataRead_mSsp_caches:
						min_DataRead_mSsp_caches[base_Rsp], min_DataRead_repl_caches[base_Rsp] = dict(), dict()
					# 
					cost = my_cost_model_fix_reduc_2Rngs(selected_fixedL_dict, func_micK_curvs, popts, curve_rep_layouts, interested_blkNs, 
						base_mick_shape, task_shape, op_type, reduc_keys, 
						min_DataRead_mSsp_caches[base_Rsp], min_DataRead_repl_caches[base_Rsp])
					# 
					fix_reduc_2Rngs_cache[(tuple(micK_Sshape), Srepl, tuple(base_Rsp), base_reduc_rep_num)] = cost
				except Exception as e:
					print(mick_shape, task_shape, base_reduc_rep_num, base_red_len, base_Rsp)
					fix_reduc_2Rngs_cache[(tuple(micK_Sshape), Srepl, tuple(base_Rsp), base_reduc_rep_num)] = 'fail'
					assert False
					raise e
				# cost = my_cost_model_fix_reduc_2Rngs(selected_fixedL_dict, func_micK_curvs, popts, curve_rep_layouts, interested_blkNs, base_mick_shape, task_shape, op_type, reduc_keys)
				latencys[reduc_keys] = cost
	# for k,v in latencys.items():
	# 	print(k,v)
	# 1. compute the latency of the micro-kernel with two base_reduc_rep_nums.
	base_latencys = list()
	# if red_len < 24:
	# 	base_red_len1, base_red_len2 = 12, 24
	# else:
	# 	base_red_len1, base_red_len2 = 24, 36
	for base_reduc_rep_num in [base_reduc_rep_num1, base_reduc_rep_num2]: #base_reduc_rep_nums:
		if op_type == 'conv2d':
			latency12 = list()
			for base_Rspi in range(2):
				# there are two kinds of base Rsps for a fixed Rsp size
				base_xs = [base_red_len1, base_red_len2]
				base_ys = [latencys[(interested_Rsps[rl][base_Rspi], base_reduc_rep_num)] for rl in [base_red_len1, base_red_len2]]
				if (red_len>base_red_len2) and (base_ys[1] < base_ys[0]):
					base_ys = base_ys[::-1]
				latency12.append(infer_from_linear_func(base_xs, base_ys, red_len))
			# then we compute the latency for the mick_Rshape
			base_xs = [data_read_amount_PerBlk(op_type, micK_Sshape+rl+mick_shape[sp_len:]) for rl in [(1, red_len**0.5, red_len**0.5), (red_len/2, 1, 2)]]
			base_ys = latency12
			if base_xs[0] == base_xs[1]:
				if dataInPerBlk!=base_xs[0]:
					return 1e10
				else:
					base_latencys.append(sum(latency12)/2)
			else:
				if ((base_xs[0]<base_xs[1]) and (base_ys[0]>base_ys[1])) or ((base_xs[0]>base_xs[1]) and (base_ys[0]<base_ys[1])):
					# because we assume the smaller data read amount, the faster the mick
					base_ys = base_ys[::-1]
				base_latencys.append(infer_from_linear_func(base_xs, base_ys, dataInPerBlk))
		else:
			latency1 = latencys[(interested_Rsps[base_red_len1][0], base_reduc_rep_num)]
			latency2 = latencys[(interested_Rsps[base_red_len2][0], base_reduc_rep_num)]
			# if latency2 < latency1 * 0.9: # we assume latency2 is higher than latency1
				# return 1e10
			if (red_len>base_red_len2) and (latency2 < latency1):
				latency1, latency2 = latency2, latency1
			base_latencys.append((latency2 - latency1) * (red_len-base_red_len1) / (base_red_len2-base_red_len1) + latency1)
		# latency1 = latencys[(base_red_len1, base_reduc_rep_num)]
		# latency2 = latencys[(base_red_len2, base_reduc_rep_num)]
		# # if latency2 < latency1 * 0.9: # we assume latency2 is higher than latency1
		# 	# return 1e10
		# if (red_len>base_red_len2) and (latency2 < latency1):
		# 	latency1, latency2 = latency2, latency1
		# base_latencys.append((latency2 - latency1) * (red_len-base_red_len1) / (base_red_len2-base_red_len1) + latency1)
	# print(f"base_latencys:{base_latencys}")
	if min(base_latencys)<=0:
		return 1e10
	# 2. compute the latency of the micro-kernel with the required reduc_rep_num.
	# reduc_rep_num = None
	# if op_type == 'dense':
	# 	reduc_rep_num = math.ceil(task_shape[2] / red_len)
	# if reduc_rep_num<8:
	# 	return 1e10
	# elif reduc_rep_num < 16:
	# 	base_reduc_rep_num1, base_reduc_rep_num2 = 8, 16
	# 	base_latency1, base_latency2 = base_latencys[0], base_latencys[1]
	# else:
	# 	base_reduc_rep_num1, base_reduc_rep_num2 = 16, 32
	# 	base_latency1, base_latency2 = base_latencys[1], base_latencys[2]
	base_latency1, base_latency2 = base_latencys
	# if base_latency2 < base_latency1 * 0.9: # we assume base_latency2 is higher than base_latency1
	# 	return 1e10
	if (reduc_rep_num > base_reduc_rep_num2) and (base_latency1>base_latency2):
		base_latency1, base_latency2 = base_latency2, base_latency1
	# print(base_latency2-base_latency1, reduc_rep_num - base_reduc_rep_num1, base_reduc_rep_num2 - base_reduc_rep_num1)
	latency = (base_latency2-base_latency1) * (reduc_rep_num - base_reduc_rep_num1) / (base_reduc_rep_num2 - base_reduc_rep_num1) + base_latency1
	if latency <= 0:
		return 1e10
	return latency








def comp_cost_worker_inputCompleteShapes(first_end_id, common_params):
	# first_end_id, (end_id_num, flat_all_micK_Sshapes, op_type, task_shapes, func_micK_curvs, popts, curve_rep_layouts, interested_blkNs) = params
	end_id_num, flat_all_micK_shapes, op_type, task_shapes, selected_fixedL_dict, func_micK_curvs, popts, curve_rep_layouts, interested_blkNs, interested_Rsps = common_params
	# with open(f"tmptmptmp/tmp_res{first_end_id}_{op_type}_fd2.py", "w") as file:
	# 	file.write("def update_Edges(Edges):\n")
	sp_len = len(flat_all_micK_shapes[0])
	if op_type == 'conv2d':
		sp_len = 7
	# 
	Edges = dict()
	fix_reduc_2Rngs_caches = dict()
	min_DataRead_mSsp_caches, min_DataRead_repl_caches = dict(), dict()
	for count in range(len(task_shapes)):
		Edges[count] = list()
	for end_id in range(first_end_id, min(first_end_id + end_id_num, len(flat_all_micK_shapes))):
		# if (len(get_factors(s[0])) == 2) or (len(get_factors(s[1])) == 2):
		# 	continue
		mick_shape = flat_all_micK_shapes[end_id]
		# we use a cache here to reduce cost computation time
		cost_cache = dict()
		# we use a cache inside the cost computation function as well
		if tuple(mick_shape[sp_len:]) not in fix_reduc_2Rngs_caches:
			fix_reduc_2Rngs_caches[tuple(mick_shape[sp_len:])] = dict()
		# 
		for src_id in Edges:
			op_s = task_shapes[src_id]
			repl = tuple([math.ceil(op_s[i] / mick_shape[i]) for i in range(sp_len)])
			if repl in cost_cache:
				Edges[src_id].append(cost_cache[repl])
				continue
			# blk_num = get_product([math.ceil(op_s[i] / s[i]) for i in range(2)])
			# in the mick shape op task shape, there may be some other parameters like stride/padding/dilation for conv2d
			# padded_fullshape = [math.ceil(op_s[i] / mick_shape[i]) * mick_shape[i] for i in range(sp_len)] + mick_shape[sp_len:]
			padded_fullshape = [repl[i] * mick_shape[i] for i in range(sp_len)] + mick_shape[sp_len:]
			# cost = my_cost_model(func_micK_curvs, popts, curve_rep_layouts, interested_blkNs, mick_shape, op_s, op_type)
			# cost = my_cost_model_full_dynamic(func_micK_curvs, popts, curve_rep_layouts, interested_blkNs, mick_shape, op_s, op_type)
			cost = my_cost_model_full_dynamic_2Rngs(selected_fixedL_dict, func_micK_curvs, popts, curve_rep_layouts, interested_blkNs, 
				mick_shape, op_s, op_type, interested_Rsps, 
				fix_reduc_2Rngs_caches[tuple(mick_shape[sp_len:])], min_DataRead_mSsp_caches, min_DataRead_repl_caches)
			Edges[src_id].append((end_id, padded_fullshape, cost))
			cost_cache[repl] = (end_id, padded_fullshape, cost)
			# with open(f"tmptmptmp/tmp_res{first_end_id}_{op_type}_fd2.py", "a") as file:
			# 	file.write(f"\tEdges[{src_id}].append(({end_id}, {mick_shape}, {padded_fullshape}, {cost}))\n")
	return Edges












def compute_best_cost_UCB_worker(first_task_id, common_params):
	'''
		Compute the best cost for a task according to the cost dict (Edges).
		Input:
			common_params: (taskParaNum, st, Edges). st is the selected arm set.
			first_task_id: int. The first task id this worker considers.
			taskParaNum: int. The number of tasks a worker needs to consider.
			taskids: list of int. The list of ids of the tasks to be considered.
			Edges: {taskid: list of cost preduction results}. An element in the list of cost prediction results is a tuple (or list), where
				the last value is the predicted cost.
		Output:
			A tuple: the best cost dict.
	'''
	taskParaNum, st, Edges = common_params
	taskNum = len(Edges.keys())
	cost_dict = dict()
	for taski in range(first_task_id, min(taskNum, first_task_id + taskParaNum)):
		costs = Edges[taski][st]
		best_idx = np.argpartition(costs, 0)[0]
		cost_dict[taski] = [costs[best_idx], st[best_idx]]
	# for taski in range(first_task_id, min(taskNum, first_task_id + taskParaNum)):
	# 	cost_dict[taski] = [float("inf"), None]
	# for i in st:
	# 	# Nis[i] = Nis[i] + 1
	# 	# mick_shape = flat_all_micK_Sshapes[i//128] + [i%128+1, ]
	# 	# for taski, op_s in enumerate(task_shapes):
	# 	for taski in range(first_task_id, min(taskNum, first_task_id + taskParaNum)):
	# 		# cost = my_cost_model_full_dynamic_2Rngs(
	# 		# 	selected_fixedL_dict, func_micK_curvs, popts, curve_rep_layouts, interested_blkNs, mick_shape, op_s, op_type)
	# 		cost = Edges[taski][i][-1]
	# 		if cost < cost_dict[taski][0]:
	# 			cost_dict[taski] = [cost, i]
	return cost_dict











def get_best_mick_by_cost_model(Edges, task_shapes, avg_errors = None):
	res = dict()
	if avg_errors == None:
		for task_i, v in Edges.items():
			mick_i = np.argpartition(v, 0)[0]
			cost = v[mick_i]
			res[task_i] = [cost, mick_i]
	else:
		# need to consider prediction error in the predicted cost
		for task_i, v in Edges.items():
			correct_costs = list()
			for mick_i in range(len(v)):
				msp = flat_all_micK_shapes[mick_i]
				Ssp, Rsp = None, None
				if op_type == 'dense':
					Ssp, Rsp = tuple(msp[:2]), (msp[2], )
				elif op_type in ['bmm','bmm_nn']:
					Ssp, Rsp = tuple(msp[:3]), (msp[3], )
				elif op_type == 'conv2d':
					Ssp, Rsp = tuple(msp[:4]), tuple(msp[4:7])
				error_ratio = 1
				if get_product(Ssp) in avg_errors['Ssize']:
					# error_ratio = error_ratio * np.mean(avg_errors['Ssize'][get_product(Ssp)])
					error_ratio = error_ratio * avg_errors['Ssize'][get_product(Ssp)]
				if Rsp in avg_errors['Rsp']:
					# error_ratio = error_ratio * np.mean(avg_errors['Rsp'][Rsp])
					error_ratio = error_ratio * avg_errors['Rsp'][Rsp]
				if get_product(Rsp) in avg_errors['Rsp']:
					error_ratio = error_ratio * avg_errors['Rsp'][get_product(Rsp)]
				correct_costs.append(error_ratio * v[mick_i])
			mick_i = np.argpartition(correct_costs, 0)[0]
			cost = correct_costs[mick_i]
			res[task_i] = [cost, mick_i]	
	return res




def get_repl_to_measure_for_mick(op_type):
	'''
		Since we need to measure a mick on an op when tuning it, this method returns the the required task shape.
	'''
	if op_type == 'dense':
		rep_layout = [get_symmetric_factor(540), 540//get_symmetric_factor(540), 8]
		return rep_layout
	elif op_type in ['bmm', 'bmm_nn']:
		rep_layout = [1, get_symmetric_factor(540), 540//get_symmetric_factor(540), 8]
		return rep_layout
	elif op_type == 'conv2d':
		rep_layout = [get_symmetric_factor(540), 540//get_symmetric_factor(540), 1, 1, 8, 1, 1]
		return rep_layout




def get_tsp_to_measure_for_mick(mick_shape, op_type):
	'''
		Since we need to measure a mick on an op when tuning it, this method returns the the required task shape.
	'''
	rep_layout = get_repl_to_measure_for_mick(op_type)
	tsp = [rep_layout[i] *mick_shape[i] for i in range(len(rep_layout))] + mick_shape[len(rep_layout):]
	return tsp
	# if op_type == 'dense':
	# 	rep_layout = [get_symmetric_factor(540), 540//get_symmetric_factor(540), 8]
	# 	tsp = [rep_layout[i] *mick_shape[i] for i in range(3)]
	# 	return tsp
	# elif op_type == 'bmm':
	# 	rep_layout = [1, get_symmetric_factor(540), 540//get_symmetric_factor(540), 8]
	# 	tsp = [rep_layout[i] *mick_shape[i] for i in range(4)]
	# 	return tsp
	# elif op_type == 'conv2d':
	# 	rep_layout = [get_symmetric_factor(540), 540//get_symmetric_factor(540), 1, 1, 8, 1, 1]
	# 	tsp = [rep_layout[i] *mick_shape[i] for i in range(len(rep_layout))] + mick_shape[len(rep_layout):]
	# 	return tsp



def get_task_to_measure_for_mick(mick_shape, op_type):
	'''
		Since we need to measure a mick on an op when tuning it, this method returns the the required task.
	'''
	if op_type == 'dense':
		# rep_layout = [get_symmetric_factor(540), 540//get_symmetric_factor(540), 8]
		# tsp = [rep_layout[i] *mick_shape[i] for i in range(3)]
		tsp = get_tsp_to_measure_for_mick(mick_shape, op_type)
		task = auto_scheduler.SearchTask(
			    func=dense_layer, args=((tsp[0], tsp[2]), (tsp[1], tsp[2])), target=tvm.target.Target("cuda")
			)
		return task
	elif op_type == 'bmm':
		# rep_layout = [get_symmetric_factor(540), 540//get_symmetric_factor(540), 8]
		# tsp = [rep_layout[i] *mick_shape[i] for i in range(3)]
		tsp = get_tsp_to_measure_for_mick(mick_shape, op_type)
		task = auto_scheduler.SearchTask(
			    func=batch_matmul, args=((tsp[0], tsp[1], tsp[3]), (tsp[0], tsp[2], tsp[3])), target=tvm.target.Target("cuda")
			)
		return task
	elif op_type == 'bmm_nn':
		# rep_layout = [get_symmetric_factor(540), 540//get_symmetric_factor(540), 8]
		# tsp = [rep_layout[i] *mick_shape[i] for i in range(3)]
		tsp = get_tsp_to_measure_for_mick(mick_shape, 'bmm')
		task = auto_scheduler.SearchTask(
			    func=batch_matmul_noTrans, args=((tsp[0], tsp[1], tsp[3]), (tsp[0], tsp[3], tsp[2])), target=tvm.target.Target("cuda")
			)
		return task
	elif op_type == 'conv2d':
		tsp = get_tsp_to_measure_for_mick(mick_shape, op_type)
		n, c, h, w, rc, rh, rw, stride, padding, dilation = tsp
		sh, sw = stride
		dh, dw = dilation
		task = auto_scheduler.SearchTask(
			    func=conv2d_nchw, args=((n, rc, sh*(h-1)+dh*(rh-1)+1, sw*(w-1)+dw*(rw-1)+1), (c, rc, rh, rw), stride, padding, dilation, "float32"), 
				target=tvm.target.Target("cuda")
			)
		return task




def get_padded_op(msp, repl=None, tsp=None, op_type='dense'):
	sp_len = len(msp)
	if op_type == 'conv2d':
		sp_len = 7
	if repl == None:
		assert tsp!=None
		# if op_type == 'dense':
		repl = tuple([math.ceil(tsp[i]/msp[i]) for i in range(sp_len)])
	psp = [repl[i] * msp[i] for i in range(len(repl))] + msp[sp_len:]
	if op_type == 'dense':
		# psp = [repl[i] * msp[i] for i in range(3)]
		task = auto_scheduler.SearchTask(
			    func=dense_layer, args=((psp[0], psp[2]), (psp[1], psp[2])), target=tvm.target.Target("cuda")
			)
		return task
	elif op_type == 'bmm':
		# psp = [repl[i] * msp[i] for i in range(len(repl))]
		task = auto_scheduler.SearchTask(
			    func=batch_matmul, args=((psp[0], psp[1], psp[3]), (psp[0], psp[2], psp[3])), target=tvm.target.Target("cuda")
			)
		return task
	elif op_type == 'bmm_nn':
		# psp = [repl[i] * msp[i] for i in range(len(repl))]
		task = auto_scheduler.SearchTask(
			    func=batch_matmul_noTrans, args=((psp[0], psp[1], psp[3]), (psp[0], psp[3], psp[2])), target=tvm.target.Target("cuda")
			)
		return task
	elif op_type == 'conv2d':
		n, c, h, w, rc, rh, rw, stride, padding, dilation = psp
		sh, sw = stride
		dh, dw = dilation
		task = auto_scheduler.SearchTask(
			    func=conv2d_nchw, args=((n, rc, sh*(h-1)+dh*(rh-1)+1, sw*(w-1)+dw*(rw-1)+1), (c, rc, rh, rw), stride, padding, dilation, "float32"), 
				target=tvm.target.Target("cuda")
			)
		return task




def get_mick_with_Kth_latency(UCB_res, Kth):
	'''
		get the mick shape with the Kth largest latency in the mick shape set selected by UCB.
		Output:
			Will return None if the number of used mick shapes < Kth.
			Otherwise, return the mick id, and the task ids assigned to that mick.
	'''
	tot_cost = dict()
	assigned_tasks = dict()
	for task_i, (cost, mick_i) in UCB_res.items():
		if mick_i not in tot_cost:
			tot_cost[mick_i] = cost
			assigned_tasks[mick_i] = [task_i]
		else:
			tot_cost[mick_i] = tot_cost[mick_i] + cost
			assigned_tasks[mick_i].append(task_i)
	if len(tot_cost.keys()) < Kth:
		return None, None
	ret_mick_id = sorted(tot_cost.keys(), key=lambda mick_i: tot_cost[mick_i])[-Kth]
	return ret_mick_id, assigned_tasks[ret_mick_id]




def get_tot_latency_of_mick(UCB_res, Edges, mick_i):
	tot_cost = 0
	for task_i, (_, i) in UCB_res.items():
		if i == mick_i:
			tot_cost = tot_cost + Edges[task_i][mick_i]
	return tot_cost





# <08.31> add iter_res_log as input
def get_best_mick_with_base_mickset(mick_ids, flat_all_micK_shapes, Edges, 
	dead_micks, op_type, exclude_dead_mick = True, avg_errors=None, 
	task_ids=None, iter_res_log=None):
	'''
		avg_errors is a dict that stores the average prediction error for each reduction shape.
		error = real / prediction.
		task_ids: list of ints. The id of the ops that we would consider in this method.
						If it is None, we consider all the ops to find the best mick.
	'''
	taskNum = len(Edges)
	mickNum = len(Edges[0])
	costs = list()
	if task_ids == None:
		task_ids = list(range(taskNum))
	# 
	# this part is very time-consuming, we make it parallel
	# for mick_i in range(mickNum):
	# 	cost_dict = compute_best_cost_UCB_worker(0, (taskNum, list(mick_ids)+[mick_i], Edges))
	# 	costs.append(sum([c for c, _ in cost_dict.values()]))
	# parallel version is still slow, the version below is much faster
	base_cost_vec = None
	if len(mick_ids) != 0:
		base_cost = compute_best_cost_UCB_worker(0, (taskNum, list(mick_ids), Edges))
		base_cost_vec = [base_cost[task_i][0] for task_i in task_ids]
	else:
		base_cost_vec = [float('inf') for task_i in task_ids]
	for mick_i in range(mickNum):
		cost_vec = [Edges[task_i][mick_i] for task_i in task_ids]
		costs.append(np.sum(np.minimum(base_cost_vec, cost_vec)))
	# 
	if avg_errors == None:
		# we do not consider avg_errors
		# <08.31> add print to help show the selection is accurate
		if iter_res_log != None:
			with open(iter_res_log, 'a') as f:
				top_100_mick_ids = np.argpartition(costs, 1000)[:1000]
				f.write(f"{{'base_mick_ids':{list(mick_ids)}, 'tasks_to_focus':{list(task_ids)}, 'top_mick_ids':{list(top_100_mick_ids)}, 'top_costs_weight':{list(np.array(costs)[top_100_mick_ids])}}},\n")
		# 
		return np.argpartition(costs, 0)[:1]
		# return ret[:10]
	# 
	# THE CODE BELOW IS NOT USED NOW
	topK = len(costs) # - 1 # 8	
	if exclude_dead_mick:
		ret = [mick_i for mick_i in sorted(range(mickNum), key=lambda mick_i: costs[mick_i]) if mick_i not in dead_micks][:topK]
	else:
		# ret = sorted(range(mickNum), key=lambda mick_i: costs[mick_i])[:2*topK]
		ret = np.argpartition(costs, topK-1)[:topK]
	if avg_errors == None:
		# we do not consider avg_errors
		return np.argpartition(costs, 0)[:1]
		# return ret[:10]
	# we need to consider avg_errors
	correct_costs = list()
	for mick_i in ret:
		if mick_i in dead_micks:
			correct_costs.append(costs[mick_i])
		else:
			msp = flat_all_micK_shapes[mick_i]
			Ssp, Rsp = None, None
			if op_type == 'dense':
				Ssp, Rsp = tuple(msp[:2]), (msp[2], )
			elif op_type in ['bmm', 'bmm_nn']:
				Ssp, Rsp = tuple(msp[:3]), (msp[3], )
			elif op_type == 'conv2d':
				Ssp, Rsp = tuple(msp[:4]), tuple(msp[4:7])
			error_ratio = 1
			if get_product(Ssp) in avg_errors['Ssize']:
				# error_ratio = error_ratio * np.mean(avg_errors['Ssize'][get_product(Ssp)])
				error_ratio = error_ratio * avg_errors['Ssize'][get_product(Ssp)]
			if Rsp in avg_errors['Rsp']:
				# error_ratio = error_ratio * np.mean(avg_errors['Rsp'][Rsp])
				error_ratio = error_ratio * avg_errors['Rsp'][Rsp]
			if get_product(Rsp) in avg_errors['Rsp']:
				error_ratio = error_ratio * avg_errors['Rsp'][get_product(Rsp)]
			correct_costs.append(error_ratio * costs[mick_i])
	# correct_costs = list()
	# for mick_i in ret:
	# 	if mick_i in dead_micks:
	# 		correct_costs.append(costs[mick_i])
	# 	else:
	# 		correct_costs.append(avg_errors[mick_i] * costs[mick_i])
	ret = [mick_i for mick_i, _ in sorted(zip(ret, correct_costs), key=lambda x : x[1])]
	return ret[:10] # ret[:topK]




def tune_mick_one_round(mick_ids, policy_dict, num_measures_per_round, measurer, 
	mick_cts, best_costs, best_states, mick_best_cts, dead_micks, mick_costs_history):
	roundNum = 1
	if len(dead_micks) == 0:
		roundNum = 2
	for round_i in range(roundNum):
		# we tune each mick for 256 trials to get a close approximation
		for mick_i in mick_ids:
			# best_state = search_policy.search(tune_option.num_measure_trials, 
			# 		tune_option.early_stopping,
			# 		tune_option.num_measures_per_round, 
			# 		measurer)
			if mick_i in dead_micks:
				continue
			# 
			measure_inputs, measure_results = policy_dict[mick_i].continue_search_one_round(
			    num_measures_per_round, measurer
			)
			# 
			mick_cts[mick_i] += 1
			# 
			for inp, res in zip(measure_inputs, measure_results):
				cost = np.mean([v.value for v in res.costs])
				# cost = array_mean(res.costs)
				if cost < best_costs[mick_i]:
					mick_best_cts[mick_i] = mick_cts[mick_i]
					best_costs[mick_i] = cost
					best_states[mick_i] = inp.state
			# 
			# Stop tuning this task in the rest of the process if its search space has been
			# fully explored or it has no improvement for a long while.
			# no_change_trials = (
			#     self.task_cts[task_idx] - self.task_best_cts[task_idx]
			# ) * self.num_measures_per_round
			if len(measure_inputs) == 0: # or no_change_trials > self.early_stopping_task:
				dead_micks.add(mick_i)
			# 
			mick_costs_history[mick_i].append(best_costs[mick_i])
	for mick_i in mick_ids:
		dead_micks.add(mick_i)




def get_search_policies_for_micks(mick_ids, flat_all_micK_shapes, policy_dict, cost_model, op_type):
	'''
		Create search policies for new micks.
	'''
	# kernel_type = "micK1fetch"
	search_policy_params = {
		"limit_blk_num" : 1,
		"limit_fetch_num" : 1,
		"limit_tileL": 1}
	for mick_i in mick_ids:
		print(mick_i)
		msp = flat_all_micK_shapes[mick_i]
		if mick_i not in policy_dict:
			mick_sp_str = mick_shape_to_str(msp)
			search_policy_params["mick_shape"] = mick_sp_str
			# 
			# if op_type == 'dense':
				# rep_layout = [get_symmetric_factor(540), 540//get_symmetric_factor(540), 8]
				# tsp = [rep_layout[i] *msp[i] for i in range(3)]
				# task = auto_scheduler.SearchTask(
				# 	    func=dense_layer, args=((tsp[0], tsp[2]), (tsp[1], tsp[2])), target=tvm.target.Target("cuda")
				# 	)
			task = get_task_to_measure_for_mick(msp, op_type)
			policy_dict[mick_i] = auto_scheduler.SketchPolicy(task, cost_model, search_policy_params)






def measure_tasks_and_states(tune_option, tasks_to_measure, states_to_measure, return_MeasureRes = False):
	'''
		If return_MeasureRes, we return the measureResult directly; else, we return the running time cost.
	'''
	ret_res = list()
	for i in range(len(states_to_measure)//256+1):
		states_batch = states_to_measure[i*256:i*256+256]
		task_batch = tasks_to_measure[i*256:i*256+256]
		states_num = len(states_batch)
		ret = measure_states_for_diff_tasks(tune_option, task_batch, states_batch)
		if ret == None:
			for state_i, (task, state) in enumerate(zip(task_batch, states_batch)):
				tmp_res = measure_states_for_diff_tasks(tune_option, [task], [state])
				if tmp_res == None:
					if return_MeasureRes:
						ret_res.append(None)
					else:
						ret_res.append(1e10) # it means this cost is invalid
				else:
					if return_MeasureRes:
						ret_res = ret_res + list(tmp_res)
					else:
						costs = [v.value for v in tmp_res[0].costs]
						cost = np.mean(costs)
						ret_res.append(cost) # (padded_op.compute_dag.flop_ct / cost / 1e9)
						# print(f"ret[{i*256+state_i}]={ret_res[-1]}", flush=True)
			continue
		if return_MeasureRes:
			ret_res = ret_res + list(ret)
			continue
		for state_i in range(states_num):
			# padded_op = padded_ops[op_i]
			# print("log_file: ", log_file)
			# print("padded_op: ", padded_op.workload_key)
			# if ret[op_i]==None:
			# 	ret_res.append(None)
			# 	continue
			costs = [v.value for v in ret[state_i].costs]
			cost = np.mean(costs)
			# ret_res.append(costs) #padded_op.compute_dag.flop_ct / cost / 1e9)
			ret_res.append(cost)
			# print(f"ret[{i*256+state_i}]={ret_res[-1]}", flush=True)
	return ret_res



def correct_the_cost_model(tune_option, mick_ids, task_shapes, flat_all_micK_shapes, Edges, best_states, op_type):
	'''
		This function correct the cost vector of mick shapes according to the tuning result.
	'''
	SMNum = 108
	tasks_to_measure, states_to_measure = list(), list()
	result_mapping = dict()
	for mick_i in mick_ids:
		result_mapping[mick_i] = dict()
		msp = flat_all_micK_shapes[mick_i]
		# compute the replication layout range for this mick shape
		rep_layouts = set([tuple([math.ceil(tsp[i]/msp[i]) for i in range(3)]) for tsp in task_shapes])
		# Srepls = list()
		for repl in rep_layouts:
			task = get_padded_op(msp, repl=repl, tsp=None, op_type=op_type)
			tasks_to_measure.append(task)
			states_to_measure.append(best_states[mick_i])
			result_mapping[mick_i][repl] = len(tasks_to_measure) - 1
		# if op_type == 'dense':
		# 	# sort by spatial rep number
		# 	# sorted_repls = sorted(rep_layouts, key=lambda rl: get_product(rl[:2]))
		# 	# max_SrepNum = get_product(sorted_repls[-1][:2])
		# 	# min_SrepNum = get_product(sorted_repls[0][:2])
		# 	# if max_SrepNum//SMNum == min_SrepNum:
		# 	# 	Srepls.append(sorted_repls[-1])
		# 	# 
		# 	# we first try to measure this mick shape for each rep layout
		# 	for repl in rep_layouts:
		# 		tsp = [repl[i] * msp[i] for i in range(3)]
		# 		task = auto_scheduler.SearchTask(
		# 			    func=dense_layer, args=((tsp[0], tsp[2]), (tsp[1], tsp[2])), target=tvm.target.Target("cuda")
		# 			)
		# 		tasks_to_measure.append(task)
		# 		states_to_measure.append(best_states[mick_i])
		# 		result_mapping[mick_i][repl] = len(tasks_to_measure) - 1
	# measure the micks and get results
	costs = measure_tasks_and_states(tune_option, tasks_to_measure, states_to_measure)
	for mick_i in mick_ids:
		msp = flat_all_micK_shapes[mick_i]
		for task_i, tsp in enumerate(task_shapes):
			repl = tuple([math.ceil(tsp[i]/msp[i]) for i in range(3)])
			Edges[task_i][mick_i] = costs[ result_mapping[mick_i][repl] ]






def correct_the_cost_model_approx(tune_option, mick_ids, task_shapes, 
	flat_all_micK_shapes, Edges, ori_Edges, best_costs, op_type, cost_model_params, 
	avg_errors, error_log):
	'''
		This function correct the cost vector of mick shapes according to the tuning result.
		This function will not do the real measure, but use the approximate cost.
	'''
	SMNum = 108
	selected_fixedL_dict, func_micK_curvs, popts, curve_rep_layouts, interested_blkNs = cost_model_params
	for mick_i in mick_ids:
		real_cost = best_costs[mick_i]
		mick_shape = flat_all_micK_shapes[mick_i]
		psp = get_tsp_to_measure_for_mick(mick_shape, op_type)
		pred_cost = my_cost_model_full_dynamic_2Rngs(selected_fixedL_dict, func_micK_curvs, popts, curve_rep_layouts, interested_blkNs, \
			mick_shape, psp, op_type)
		# 
		error_log[tuple(mick_shape)] = (real_cost, pred_cost, real_cost / pred_cost)
		# 
		msp = mick_shape
		Ssp, Rsp = None, None
		if op_type == 'dense':
			Ssp, Rsp = tuple(msp[:2]), (msp[2], )
		error_ratio = 1
		if get_product(Ssp) in avg_errors['Ssize']:
			# error_ratio = error_ratio * np.mean(avg_errors['Ssize'][get_product(Ssp)])
			error_ratio = error_ratio * avg_errors['Ssize'][get_product(Ssp)]
		if Rsp in avg_errors['Rsp']:
			# error_ratio = error_ratio * np.mean(avg_errors['Rsp'][Rsp])
			error_ratio = error_ratio * avg_errors['Rsp'][Rsp]
		pred_cost = pred_cost * error_ratio
		# 
		error_ratio = real_cost / pred_cost
		# # update avg_errors
		# Ssp, Rsp = None, None
		# if op_type == 'dense':
		# 	Ssp, Rsp = tuple(mick_shape[:2]), (mick_shape[2], )
		# if get_product(Ssp) not in avg_errors['Ssize']:
		# 	avg_errors['Ssize'][get_product(Ssp)] = list()
		# if Rsp not in avg_errors['Rsp']:
		# 	avg_errors['Rsp'][Rsp] = list()
		# 	avg_errors['Rsp'][Rsp].append(error_ratio)
		# 	avg_errors['Ssize'][get_product(Ssp)].append(error_ratio)
		# else:
		# 	avg_errors['Ssize'][get_product(Ssp)].append(error_ratio / np.mean(avg_errors['Rsp'][Rsp]))
		# 	avg_errors['Rsp'][Rsp].append(error_ratio)
		# # avg_errors['Ssize'][get_product(Ssp)].append(error_ratio)
		# # avg_errors['Rsp'][Rsp].append(error_ratio)
		# 
		# error_log[tuple(mick_shape)] = (real_cost, pred_cost, error_ratio)
		# 
		# then we update the Edge dict according to the error_ratio
		for task_i, tsp in enumerate(task_shapes):
			Edges[task_i][mick_i] = Edges[task_i][mick_i] * error_ratio
			ori_Edges[task_i][mick_i] = Edges[task_i][mick_i]





def correct_the_cost_model_approx_do_measure(tune_option, mick_ids, task_shapes, 
	flat_all_micK_shapes, Edges, ori_Edges, ori_Edges_with_weight, best_states, op_type, cost_model_params, 
	avg_errors, error_log, res_dict, dead_micks, only_max_repl = True, infer_ratio = 1.03, Ssp_poses=None):
	'''
		This function correct the cost vector of mick shapes according to the tuning result.
		This function will do the real measure, but just measure on the largest assigned op shape in task_shapes.
		res_dict: this is the assignment dict according to ori_Edges.
		only_max_repl: bool. It controls whether the correction is done per repl num group or only based on the max repl num.
		ori_Edges_with_weight: is the Edges after correction at begin (consider avg_errors and padding cost)
	'''
	SMNum = 108
	selected_fixedL_dict, func_micK_curvs, popts, curve_rep_layouts, interested_blkNs, interested_Rsps = cost_model_params
	sp_len = len(flat_all_micK_shapes[0])
	Ssp_len = sp_len - 1
	if op_type == 'conv2d':
		sp_len = 7
		Ssp_len = 4
	for mick_i in mick_ids:
		# first compute the largest op shape assigned to mick_i
		assigned_info = list()
		msp = flat_all_micK_shapes[mick_i]
		if only_max_repl:
			for task_i, (_, i) in res_dict.items():
				if i == mick_i:
					tsp = task_shapes[task_i]
					repl = [math.ceil(tsp[j]/msp[j]) for j in range(sp_len)]
					assigned_info.append((tsp, repl, get_product(repl[:Ssp_len]), get_product(repl)))
		else:
			for tsp in task_shapes:
				repl = [math.ceil(tsp[j]/msp[j]) for j in range(sp_len)]
				assigned_info.append((tsp, repl, get_product(repl[:Ssp_len]), get_product(repl)))
		if len(assigned_info) == 0:
			return False
		# 
		measure_res = dict()
		if only_max_repl:
			largest_repl = sorted(assigned_info, key=lambda info: info[-1])[-1][1]
			task = get_padded_op(msp, repl=largest_repl, tsp=None, op_type=op_type)
			costs = measure_tasks_and_states(tune_option, [task], [best_states[mick_i]])
			real_cost = costs[0]
			psp = [largest_repl[i]*msp[i] for i in range(sp_len)] + msp[sp_len:]
			pred_cost = my_cost_model_full_dynamic_2Rngs(selected_fixedL_dict, func_micK_curvs, popts, curve_rep_layouts, interested_blkNs, \
				msp, psp, op_type, interested_Rsps)
			measure_res['any'] = (real_cost, pred_cost, real_cost / pred_cost)
		else:
			largest_repl_per_group = dict()
			for info in assigned_info:
				key = min(math.ceil(info[-2]/108), 10)
				if key not in largest_repl_per_group:
					largest_repl_per_group[key] = info
				elif info[-1] > largest_repl_per_group[key][-1]:
					largest_repl_per_group[key] = info
			print(f"assigned_info:{assigned_info}")
			print(f"largest_repl_per_group:{largest_repl_per_group}")
			keys = list(largest_repl_per_group.keys())
			repls = [largest_repl_per_group[k][1] for k in keys]
			psps = [[repl[i]*msp[i] for i in range(sp_len)] + msp[sp_len:] for repl in repls]
			print("psps", psps)
			pred_costs = [
						my_cost_model_full_dynamic_2Rngs(selected_fixedL_dict, func_micK_curvs, popts, curve_rep_layouts, interested_blkNs, \
							msp, psp, op_type, interested_Rsps) 
						for psp in psps]
			tasks = [get_padded_op(msp, repl=repl, tsp=None, op_type=op_type) for repl in repls]
			costs = measure_tasks_and_states(tune_option, tasks, [best_states[mick_i] for t in tasks])
			for ki, k in enumerate(keys):
				measure_res[k] = (costs[ki], pred_costs[ki], costs[ki] / pred_costs[ki])
		# real_cost = best_costs[mick_i]
		# 
		# mick_shape = flat_all_micK_shapes[mick_i]
		# # psp = get_tsp_to_measure_for_mick(mick_shape, op_type)
		# psp = [largest_repl[i]*msp[i] for i in range(sp_len)] + msp[sp_len:]
		# pred_cost = my_cost_model_full_dynamic_2Rngs(selected_fixedL_dict, func_micK_curvs, popts, curve_rep_layouts, interested_blkNs, \
		# 	msp, psp, op_type, interested_Rsps)
		# 
		# error_log[tuple(mick_shape)] = (real_cost, pred_cost, real_cost / pred_cost)
		error_log[tuple(msp)] = measure_res
		print("error_log", error_log)
		# 
		# msp = mick_shape
		Ssp, Rsp = None, None
		if op_type == 'dense':
			Ssp, Rsp = tuple(msp[:2]), (msp[2], )
		elif op_type in ['bmm', 'bmm_nn']:
			Ssp, Rsp = tuple(msp[:3]), (msp[3], )
		elif op_type == 'conv2d':
			Ssp, Rsp = tuple(msp[:4]), tuple(msp[4:7])
		error_ratio = 1
		if get_product(Ssp) in avg_errors['Ssize']:
			# error_ratio = error_ratio * np.mean(avg_errors['Ssize'][get_product(Ssp)])
			error_ratio = error_ratio * avg_errors['Ssize'][get_product(Ssp)]
		if Rsp in avg_errors['Rsp']:
			# error_ratio = error_ratio * np.mean(avg_errors['Rsp'][Rsp])
			error_ratio = error_ratio * avg_errors['Rsp'][Rsp]
		if get_product(Rsp) in avg_errors['Rsp']:
			error_ratio = error_ratio * avg_errors['Rsp'][get_product(Rsp)]
		# 
		# pred_cost = pred_cost * error_ratio
		# error_ratio = real_cost / pred_cost
		# 
		error_ratio_list = list()
		if only_max_repl:
			error_ratio_list = [measure_res['any'][-1]/error_ratio]*10
		else:
			keys=sorted(measure_res.keys())
			ki=0
			for i in range(1,11):
				if (i > keys[ki]) and ((ki+1)<len(keys)):
					ki+=1
				error_ratio_list.append(measure_res[keys[ki]][-1]/error_ratio)
		# 
		# # update avg_errors
		# Ssp, Rsp = None, None
		# if op_type == 'dense':
		# 	Ssp, Rsp = tuple(mick_shape[:2]), (mick_shape[2], )
		# if get_product(Ssp) not in avg_errors['Ssize']:
		# 	avg_errors['Ssize'][get_product(Ssp)] = list()
		# if Rsp not in avg_errors['Rsp']:
		# 	avg_errors['Rsp'][Rsp] = list()
		# 	avg_errors['Rsp'][Rsp].append(error_ratio)
		# 	avg_errors['Ssize'][get_product(Ssp)].append(error_ratio)
		# else:
		# 	avg_errors['Ssize'][get_product(Ssp)].append(error_ratio / np.mean(avg_errors['Rsp'][Rsp]))
		# 	avg_errors['Rsp'][Rsp].append(error_ratio)
		# # avg_errors['Ssize'][get_product(Ssp)].append(error_ratio)
		# # avg_errors['Rsp'][Rsp].append(error_ratio)
		# 
		# error_log[tuple(mick_shape)] = (real_cost, pred_cost, error_ratio)
		# 
		# we try to adjust the prediction error for other micro-kernel shapes with the same Ssp, tmp_is are mick ids which can be corrected together
		tmp_is = [tmp_i for tmp_i in range(Ssp_poses[tuple(msp[:Ssp_len])][0], Ssp_poses[tuple(msp[:Ssp_len])][1]) \
			if tmp_i not in dead_micks]
		# 
		# then we update the Edge dict according to the error_ratio
		for task_i, tsp in enumerate(task_shapes):
			key = int(min(math.ceil(get_product([math.ceil(tsp[j]/msp[j]) for j in range(Ssp_len)])/108), 10)-1)
			err = error_ratio_list[key]
			# Edges[task_i][mick_i] = Edges[task_i][mick_i] * err
			# ori_Edges[task_i][mick_i] = ori_Edges[task_i][mick_i] * err # Edges[task_i][mick_i]
			ori_Edges[task_i][mick_i] = ori_Edges[task_i][mick_i] * err * error_ratio
			Edges[task_i][mick_i] = ori_Edges_with_weight[task_i][mick_i] * err # we should compute padding cost as well
			# we try to adjust the prediction error for other micro-kernel shapes with the same Ssp
			if err*error_ratio > infer_ratio:
				for tmp_i in tmp_is:
					# Edges[task_i][tmp_i] = Edges[task_i][tmp_i] * err
					Edges[task_i][tmp_i] = ori_Edges_with_weight[task_i][tmp_i] * err
		# we try to adjust the prediction error for other micro-kernel shapes with the same Rsp and the same Ssize
		# for tmp_i in range(len(flat_all_micK_shapes)):
		# 	if tmp_i in dead_micks:
		# 		continue
		# 	tmsp = flat_all_micK_shapes[tmp_i]
		# 	# if (tmsp[Ssp_len:sp_len] == msp[Ssp_len:sp_len]) and (get_product(msp[:Ssp_len]) == get_product(tmsp[:Ssp_len])):
		# 	# if (get_product(msp[:Ssp_len]) == get_product(tmsp[:Ssp_len])):
		# 	if ((msp[:Ssp_len]) == (tmsp[:Ssp_len])):
		# 		for task_i, tsp in enumerate(task_shapes):
		# 			key = int(min(get_product([math.ceil(tsp[j]/tmsp[j]) for j in range(Ssp_len)])/108, 10)-1)
		# 			err = error_ratio_list[key]
		# 			if err*error_ratio > infer_ratio:
		# 				Edges[task_i][tmp_i] = Edges[task_i][tmp_i] * err
	return True





def measure_final_res(tune_option, UCB_res, task_shapes, flat_all_micK_shapes, Edges, best_states, op_type):
	'''
		This function measure the final selected mick set on hardware.
		The padding cost is also measured.
	'''
	SMNum = 108
	tasks_to_measure, states_to_measure = list(), list()
	result_mapping = dict()
	ret = dict()
	sp_len = len(flat_all_micK_shapes[0])
	if op_type == 'conv2d':
		sp_len = 7
	for task_i, (_, mick_i) in UCB_res.items():
		if mick_i not in result_mapping:
			result_mapping[mick_i] = dict()
		msp = flat_all_micK_shapes[mick_i]
		tsp = task_shapes[task_i]
		repl = tuple([math.ceil(tsp[i]/msp[i]) for i in range(sp_len)])
		# psp = [repl[i] * msp[i] for i in range(sp_len)] + msp[sp_len:]
		if repl in result_mapping[mick_i]:
			continue
		task = get_padded_op(msp, repl=repl, tsp=None, op_type=op_type)
		# task = None
		# if op_type == 'dense':
		# 	task = auto_scheduler.SearchTask(
		# 		    func=dense_layer, args=((psp[0], psp[2]), (psp[1], psp[2])), target=tvm.target.Target("cuda")
		# 		)
		# elif op_type == 'bmm':
		# 	task = auto_scheduler.SearchTask(
		# 	    func=batch_matmul, args=((psp[0], psp[1], psp[3]), (psp[0], psp[2], psp[3])), target=tvm.target.Target("cuda")
		# 	)
		tasks_to_measure.append(task)
		states_to_measure.append(best_states[mick_i])
		result_mapping[mick_i][repl] = len(tasks_to_measure) - 1
	# measure the micks and get results
	costs = measure_tasks_and_states(tune_option, tasks_to_measure, states_to_measure)
	# measure the padding ops
	# padding_costs = measure_padding_ops_for_selection_res(tune_option, UCB_res, task_shapes, flat_all_micK_shapes, op_type, None)
	for task_i, (_, mick_i) in UCB_res.items():
		msp = flat_all_micK_shapes[mick_i]
		tsp = task_shapes[task_i]
		repl = tuple([math.ceil(tsp[i]/msp[i]) for i in range(sp_len)])
		psp = [repl[i] * msp[i] for i in range(sp_len)] + msp[sp_len:]
		ret[task_i] = (msp, psp, Edges[task_i][mick_i], costs[ result_mapping[mick_i][repl] ]) #, padding_costs[task_i], 
			# costs[ result_mapping[mick_i][repl] ]) + sum(padding_costs[task_i]))
	return ret





def measure_final_res_from_logfile(tune_option, UCB_res, tasks, task_shapes, flat_all_micK_shapes, Edges, log_file, op_type):
	'''
		This function measure the final selected mick set on hardware.
	'''
	sp_len = len(flat_all_micK_shapes[0])
	if op_type == 'conv2d':
		sp_len = 7
	tasks_to_measure, states_to_measure = list(), list()
	ret = dict()
	states_dict = dict()
	my_load_all_best_input_from_file_multiTileOnes(log_file, tvm.target.Target("cuda"), states_dict)
	for task_i, tsp in enumerate(task_shapes):
		mick_i = UCB_res[task_i][1]
		msp = flat_all_micK_shapes[mick_i]
		diff_measure_task_wlk = get_task_to_measure_for_mick(msp, op_type).workload_key
		padded_op = get_padded_op(msp, repl=None, tsp=tsp, op_type=op_type)
		state = states_dict[diff_measure_task_wlk][0].state
		# 
		tasks_to_measure.append(padded_op)
		states_to_measure.append(state)
	# measure the micks and get results
	costs = measure_tasks_and_states(tune_option, tasks_to_measure, states_to_measure)
	for task_i, tsp in enumerate(task_shapes):
		mick_i = UCB_res[task_i][1]
		msp = flat_all_micK_shapes[mick_i]
		psp = [math.ceil(tsp[i]/msp[i]) * msp[i] for i in range(sp_len)] + msp[sp_len:]
		# ret[tasks[task_i].workload_key] = (msp, psp, tasks_to_measure[task_i].workload_key, Edges[task_i][mick_i], costs[ task_i ])
		ret[task_i] = (msp, psp, tasks_to_measure[task_i].workload_key, Edges[task_i][mick_i], costs[ task_i ])
	return ret








def query_tot_cost(Edges, UCB_res):
	tot_cost = 0
	for task_i, (_, mick_i) in UCB_res.items():
		tot_cost = tot_cost + Edges[task_i][mick_i]
	return tot_cost


def query_mick_cost(Edges, UCB_res, mick_i):
	tot_cost = 0
	for task_i, (_, tmick_i) in UCB_res.items():
		if tmick_i == mick_i:
			tot_cost = tot_cost + Edges[task_i][mick_i]
	return tot_cost



def change_costs_to_another_costdict(Edges, UCB_res):
	new_res = dict()
	for task_i, (_, mick_i) in UCB_res.items():
		new_res[task_i] = (Edges[task_i][mick_i], mick_i)
	return new_res



def store_cost_res_to_file(writer, Edges, UCB_res, flat_all_micK_shapes):
	for task_i, (predicted_cost, mick_i) in UCB_res.items():
		writer.write(f"{task_i};{flat_all_micK_shapes[mick_i]};{predicted_cost};{Edges[task_i][mick_i]}\n")
	writer.write(f"{sum([v for v, _ in UCB_res.values()])};{query_tot_cost(Edges, UCB_res)}\n")









# ##############################################
# store the final schedules into python files, and the dispatch dictionary to the same file as well
# ##############################################

def write_pysched_to_file(UCB_res, task_shapes, flat_all_micK_shapes, best_states, op_type, file_id):
	mick_ids = sorted(set([mick_i for _, mick_i in UCB_res.values()]))
	sched_func_name_prefix = f'schedule_{op_type}'
	with open(f'FINAL_mick_pysched_hub/{op_type}_expr{file_id}_pysched.py', 'w') as fout:
		fout.write("""from tvm import te\n\n""")
		selected_msps = dict()
		selected_msp_config = dict()
		for mick_i in mick_ids: 
			msp = flat_all_micK_shapes[mick_i]
			selected_msps[mick_i] = msp
			mick = get_padded_op(msp, repl=None, tsp=msp, op_type=op_type)
			pysched = mick.compute_dag.print_python_code_from_state(best_states[mick_i])
			wkl_func_args = ', '.join([t.op.name for t in mick.compute_dag.tensors])
			if op_type == 'conv2d':
				wkl_func_args = wkl_func_args + f', pad_temp'
			fout.write(f'def {sched_func_name_prefix}_{mick_i}({wkl_func_args}, s):\n')
			fout.write('\n'.join(['\t' + line for line in pysched.split('\n')[:-1]]))
			fout.write('\n\n\n')
			selected_msp_config[mick_i] = auto_scheduler.measure_record.dump_record_to_string(MeasureInput(mick, best_states[mick_i]), MeasureResult([0.0], 0,'', 0.0, 0))
		# write the dispatch infor
		fout.write(f"def get_dispatch_dict():\n\treturn {UCB_res}\n")
		fout.write(f"def get_task_shapes():\n\treturn {task_shapes}\n")
		fout.write(f"def get_selected_msps():\n\treturn {selected_msps}\n")
		fout.write(f"def get_selected_msp_config():\n\treturn {selected_msp_config}\n")
		fout.write("def get_selected_msp_scheds():\n\treturn {")
		for mick_i in mick_ids:
			fout.write(f"{mick_i}:{sched_func_name_prefix}_{mick_i}, ")
		fout.write("}\n\n\n")










def get_pad_cost(tsp, psp, op_type):
	inps_pad = get_inp_shapes_from_sp(psp, op_type)
	inps = get_inp_shapes_from_sp(tsp, op_type)
	pad_c0 = get_product(inps[0]) / get_product(inps_pad[0])
	pad_c0 = max(pad_c0, 1-pad_c0)
	pad_c1 = get_product(inps[1]) / get_product(inps_pad[1])
	pad_c1 = max(pad_c1, 1-pad_c1)
	# if (pad_c0*pad_c1) > 0.8*0.8:
	# 	return 1
	# else:
	# 	return 2
	return 1/(pad_c0*pad_c1)









def get_pad_cost_fast(tsp, msp, op_type):
	'''
		This function computes the padding cost by computing the max execution ratio of one statement in the if-else function in
		the stage of data loading from global memory to shared memory.
		I.e., return max(r1 = the ratio of data load of original data over the total number of data load by all blocks, 
						 r2 = the ratio of data load of padded data over the total number of data load by all blocks, )
		To speed up the padding cost computation, we deal with each input seperately.
		For each dimension of an input, we compute the consecutive ranges of all blocks along this dimension with the block index on other dimensions the same.
	'''
	def compute_covered_len(scope, rng):
		'''Compute how many integers in rng ( rng is [) ) is < scope'''
		if rng[0]>scope-1:
			return 0
		else:
			# in this case, at least the first point in rng is covered in scope
			return min(scope-1, rng[1]-1)-rng[0]+1
	# 
	sp_len = len(tsp)
	n, c, h, w, rc, rh, rw, (sh, sw), padding, (dh, dw) = None, None, None, None, None, None, None, (None, None), None, (None, None)
	if op_type == 'conv2d':
		sp_len = 7
		n, c, h, w, rc, rh, rw, (sh, sw), padding, (dh, dw) = msp
	repl = [math.ceil(tsp[i]/msp[i]) for i in range(sp_len)]
	ori_inp_sps = get_inp_shapes_from_sp(tsp, op_type)
	inp_scopes = None
	rng_lists = None
	if op_type in ['dense', 'bmm', 'bmm_nn']:
		rng_lists = [[(ni*msp[0], (ni+1)*msp[0]) for ni in range(rep_num)]
						for rep_num in repl]
		inp_scopes = tsp
	elif op_type == 'conv2d':
		rng_lists = [[(ni*msp[dim_i], (ni+1)*msp[dim_i]) for ni in range(repl[dim_i])]
						for dim_i in range(2)] +\
					[[(sh*hi*h+dh*rhi*rh, sh*((hi+1)*h-1)+dh*((rhi+1)*rh-1)+1) for hi in range(repl[2]) for rhi in range(repl[5])]] + \
					[[(sw*wi*w+dw*rwi*rw, sw*((wi+1)*w-1)+dw*((rwi+1)*rw-1)+1) for wi in range(repl[3]) for rwi in range(repl[6])]] + \
					[[(ni*msp[dim_i], (ni+1)*msp[dim_i]) for ni in range(repl[dim_i])]
						for dim_i in range(4, 7)]
		inp_scopes = tsp[:2]+ori_inp_sps[0][2:]+tsp[4:7]
	# 
	covered_len_lists = list()
	for rng_list, scope in zip(rng_lists, inp_scopes):
		tmp = dict()
		for rng in rng_list:
			key = compute_covered_len(scope, rng)
			if key not in tmp:
				tmp[key] = 0
			tmp[key] = tmp[key]+1
		covered_len_lists.append([k*v for k, v in tmp.items()])
	# 
	inp_covered_len_lists=None
	if op_type == 'dense':
	# compute the valid data load for input 0
		inp_covered_len_lists = [(covered_len_lists[0], [repl[1]], covered_len_lists[2]), ([repl[0]], covered_len_lists[1], covered_len_lists[2])]
	elif op_type in ['bmm', 'bmm_nn']:
		inp_covered_len_lists = [(covered_len_lists[0], covered_len_lists[1], [repl[2]], covered_len_lists[3]), 
							(covered_len_lists[0], [repl[1]], covered_len_lists[2], covered_len_lists[3])]
	elif op_type == 'conv2d':
		inp_covered_len_lists = [(covered_len_lists[0], [repl[1]], covered_len_lists[2], covered_len_lists[3], covered_len_lists[4]), 
							([repl[0]], covered_len_lists[1], [repl[2]], [repl[3]], covered_len_lists[4], covered_len_lists[5], covered_len_lists[6])]
	ori_data_num = 0
	for inp_covered_lens in inp_covered_len_lists:
		ori_data_num = ori_data_num + sum([get_product(tmp) for tmp in itertools.product(*inp_covered_lens)])
	# 
	mick_inp_shapes = get_inp_shapes_from_sp(msp, op_type)
	tot_data_num = get_product(repl)*sum([get_product(sp) for sp in mick_inp_shapes])
	ori_ratio = ori_data_num / tot_data_num
	return 1/(max(ori_ratio, 1-ori_ratio))






def correct_cost_dict(Edges, task_shapes, flat_all_micK_shapes, avg_errors, op_type = 'dense'):
	'''
		Correct the cost dict Edges with the avg_errors.
	'''
	sp_len = len(task_shapes[0])
	if op_type == 'conv2d':
		sp_len = 7
	for task_i, v in Edges.items():
		tsp = task_shapes[task_i]
		correct_costs = list()
		for mick_i in range(len(v)):
			msp = flat_all_micK_shapes[mick_i]
			Ssp, Rsp = None, None
			if op_type == 'dense':
				Ssp, Rsp = tuple(msp[:2]), (msp[2], )
			elif op_type in ['bmm', 'bmm_nn']:
				Ssp, Rsp = tuple(msp[:3]), (msp[3], )
			elif op_type == 'conv2d':
				Ssp, Rsp = tuple(msp[:4]), tuple(msp[4:7])
			error_ratio = 1
			if get_product(Ssp) in avg_errors['Ssize']:
				# error_ratio = error_ratio * np.mean(avg_errors['Ssize'][get_product(Ssp)])
				error_ratio = error_ratio * avg_errors['Ssize'][get_product(Ssp)]
			if Rsp in avg_errors['Rsp']:
				# error_ratio = error_ratio * np.mean(avg_errors['Rsp'][Rsp])
				error_ratio = error_ratio * avg_errors['Rsp'][Rsp]
			if get_product(Rsp) in avg_errors['Rsp']:
				error_ratio = error_ratio * avg_errors['Rsp'][get_product(Rsp)]
			pad_cost = get_pad_cost(tsp, [math.ceil(tsp[i]/msp[i]) * msp[i] for i in range(sp_len)]+msp[sp_len:], op_type)
			# pad_cost = get_pad_cost_fast(tsp, msp, op_type)
			correct_costs.append(error_ratio * v[mick_i] * pad_cost)
		Edges[task_i] = np.array(correct_costs)
	






# ========================================================================================================================
# About using ETO tuning
# ========================================================================================================================
def get_tile_sizes_for_measure(mick, msp, op_type, repl):
	'''
		This function just return the tile size settings of all the possible thread numbers.
	'''
	if op_type == 'conv2d':
		msp = msp[:7]
	tsp = [msp[i]*repl[i] for i in range(len(msp))]
	loop = msp
	op_para = get_op_para_ansor(mick, loop)
	mick_Ssp, reduc_shape, out_shape = None, None, None
	if op_type == 'dense':
		mick_Ssp = msp[:2]
		reduc_shape = [msp[2]]
		out_shape = tsp[:2]
	elif op_type in ['bmm', 'bmm_nn']:
		mick_Ssp = msp[:3]
		reduc_shape = [msp[3]]
		out_shape = tsp[:3]
	elif op_type == 'conv2d':
		mick_Ssp = msp[:4]
		reduc_shape = msp[4:7]
		out_shape = tsp[:4]
	out_size = get_product(mick_Ssp)
	thdNums = list()
	# for thrd_wld in range(1, 64):
	# 	if out_size % thrd_wld != 0:
	# 		continue
	# 	thdNum  = out_size // thrd_wld
	# 	if (thdNum % 32 == 0) and (thdNum<=1024):
	# 		thdNums.append(thdNum)
	# we do not limit the max workload of a thread now
	for thdNum in get_factors(out_size):
		if thdNum % 128 == 0:
			thdNums.append(thdNum)
	if len(thdNums) == 0:
		for thdNum in get_factors(out_size):
			if thdNum % 32 == 0:
				thdNums.append(thdNum)
	thrd_topk = 1
	# tile_knobs = list()
	tile_knobs_dict = dict()
	blk_shape = mick_Ssp
	for thdNum in thdNums:
		# tile_shapes = get_best_thrd_vthrd_shapes([mick_Ssp], thdNum, reduc_shape, op_para, cal_bk_cflct, thrd_topk)
		# blk_shape, thrd_shape, vthrd_shape = tile_shapes[0]
		# # then find the best last two level tiling for this setting
		# # we first test only 4 tile levels
		# tile_knobs.append([[vthrd_shape[i], thrd_shape[i], 1, blk_shape[i] // (vthrd_shape[i] * thrd_shape[i])] for i in range(len(blk_shape))] + \
		# 			[[1, reduc_shape[i]] for i in range(len(reduc_shape))])
		# we also do not consider the vectorization of global mem fetch now
		# 
		# we enumerate all the possible tile size settings of this thdNum, and select the skyline one 
		# in terms of the bank conflict number and the store back transaction number.
		# min_mem_cost = (float('inf'), float('inf'))
		# best_tile_shape = None
		mem_cost_list = list()
		tile_shapes = list()
		# for split_infos in split_info_list:
		# 	thrd_shape = [split_infos[i][1] for i in range(len(mick_Ssp))]
		# 	vthrd_shape = [split_infos[i][0] for i in range(len(mick_Ssp))]
		# 	confl = cal_bk_cflct(blk_shape, thrd_shape, vthrd_shape, reduc_shape, op_para)
		# 	ldSNum = load_per_thread(blk_shape, thrd_shape, vthrd_shape, reduc_shape, op_para)
		# 	stGNum = cal_32B_trans_stG(blk_shape, thrd_shape, vthrd_shape, op_para)
		# 	mem_cost = (stGNum, sum([confl[i]*ldSNum[i] for i in range(len(ldSNum))]))
		# 	if mem_cost < min_mem_cost:
		# 		best_split_infos = list(split_infos)
		# 		min_mem_cost = mem_cost
		thrd_shapes = get_combinations(thdNum, [f"axis{j}" for j in range(len(mick_Ssp))])
		thrd_shapes = dict2list(thrd_shapes, [f"axis{j}" for j in range(len(mick_Ssp))])
		# print(f"thrd_shapes:{thrd_shapes}")
		for thrd_shape in thrd_shapes:
			if False in [(mick_Ssp[i] % thrd_shape[i] == 0) for i in range(len(mick_Ssp))]:
				# this thrd_shape is invalid
				continue
			vthrd_shapes = itertools.product(*[get_factors(mick_Ssp[i] // thrd_shape[i]) for i in range(len(mick_Ssp))])
			# print(f"vthrd_shapes:{list(vthrd_shapes)}")
			for vthrd_shape in vthrd_shapes:
				confl = cal_bk_cflct(blk_shape, thrd_shape, vthrd_shape, reduc_shape, op_para)
				ldSNum = load_per_thread(blk_shape, thrd_shape, vthrd_shape, reduc_shape, op_para)
				stGNum = cal_32B_trans_stG(blk_shape, thrd_shape, vthrd_shape, op_type, out_shape)
				mem_cost = (stGNum, sum([confl[i]*ldSNum[i] for i in range(len(ldSNum))]))
				mem_cost_list.append(mem_cost)
				tile_shapes.append((thrd_shape, vthrd_shape))
				# if mem_cost < min_mem_cost:
				# 	best_tile_shape = (thrd_shape, vthrd_shape)
				# 	min_mem_cost = mem_cost
		# we return the tile size setting related to the best_split_infos
		# print(best_split_infos, mick_Ssp)
		# thrd_shape, vthrd_shape = best_tile_shape
		best_idx = sorted(range(len(mem_cost_list)), key=lambda i: mem_cost_list[i])[:3]
		# best_idx = np.argpartition(mem_cost_list, 1)[:2]
		# print(mem_cost_list)
		# print(best_idx)
		tile_knobs_dict[thdNum] = list()
		for i in best_idx:
			thrd_shape, vthrd_shape = tile_shapes[i]
			# tile_knobs_dict[thdNum] = [best_split_infos[i][:-1]+[1]+best_split_infos[i][-1:] for i in range(len(mick_Ssp))] + best_split_infos[len(mick_Ssp):]	
			tile_knobs_dict[thdNum].append([[vthrd_shape[i], thrd_shape[i], 1, blk_shape[i] // (vthrd_shape[i] * thrd_shape[i])] for i in range(len(blk_shape))] + \
									[[1, reduc_shape[i]] for i in range(len(reduc_shape))])
	return tile_knobs_dict




def gen_state_given_tiles(split_infos, op_type):
	template = None
	if op_type == 'dense':
		template = '{"i": [["[\\"dense_layer\\", [2160, 192], [1280, 192]]", "cuda -keys=cuda,gpu -arch=sm_80 -max_num_threads=1024 -thread_warp_size=32", [-1, 16, 64, 49152, 2147483647, 1024, 8, 32], "", 0, []], [[], [["CHW", 2, "local"], '
		template = template + f'["SP", 2, 0, 2160, {split_infos[0]}, 1], ["SP", 2, 5, 1280, {split_infos[1]}, 1], ["SP", 2, 10, 192, {split_infos[2]}, 1], '
		template = template + '["RE", 2, [0, 5, 1, 6, 2, 7, 10, 11, 3, 8, 12, 4, 9]], ["FSP", 3, 0, 1, 3], ["FSP", 3, 4, 2, 3], ["RE", 3, [0, 4, 1, 5, 2, 6, 3, 7]], ["CA", 2, 3, 5], ["CHR", 1, "shared", [2]], ["CA", 2, 3, 6], ["CHR", 0, "shared", [3]], ["CA", 1, 4, 6], ["FU", 5, [0, 1]], ["AN", 5, 0, 5], ["FU", 5, [1, 2]], ["AN", 5, 1, 4], ["FU", 5, [2, 3]], ["AN", 5, 2, 6], '
		template = template + f'["FU", 3, [0, 1]], ["SP", 3, 0, 24, {[1]}, 1], ["AN", 3, 1, 2], ["FFSP", 3, 0, [2, 1], 1, 1], ["AN", 3, 1, 6], '
		template = template + f'["FU", 1, [0, 1]], ["SP", 1, 0, 240, {[1]}, 1], ["AN", 1, 1, 2], ["FFSP", 1, 0, [2, 1], 1, 1], ["AN", 1, 1, 6], '
		template = template + f'["PR", 4, 0, "auto_unroll_max_step${2500}"]]]], '
		template = template + '"r": [[7.01212e-05], 0, 3.12254, 1653324698], "v": "v0.6"}\n'
	elif op_type == 'bmm':
		template = '{"i": [["[\\"batch_matmul\\", [1, 1296, 1024], [1, 960, 1024]]", "cuda -keys=cuda,gpu -arch=sm_80 -max_num_threads=1024 -thread_warp_size=32", [-1, 16, 64, 49152, 2147483647, 1024, 8, 32], "", 0, []], [[], [["CHW", 2, "local"], '
		template = template + f'["SP", 2, 0, 1, {split_infos[0]}, 1], ["SP", 2, 5, 1296, {split_infos[1]}, 1], ["SP", 2, 10, 960, {split_infos[2]}, 1], ["SP", 2, 15, 1024, {split_infos[3]}, 1], '
		template = template + '["RE", 2, [0, 5, 10, 1, 6, 11, 2, 7, 12, 15, 16, 3, 8, 13, 17, 4, 9, 14]], ["FSP", 3, 0, 1, 3], ["FSP", 3, 4, 2, 3], ["FSP", 3, 8, 3, 3], ["RE", 3, [0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11]], ["CA", 2, 3, 8], ["CHR", 1, "shared", [2]], ["CA", 2, 3, 9], ["CHR", 0, "shared", [3]], ["CA", 1, 4, 9], ["FU", 5, [0, 1, 2]], ["AN", 5, 0, 5], ["FU", 5, [1, 2, 3]], ["AN", 5, 1, 4], ["FU", 5, [2, 3, 4]], ["AN", 5, 2, 6], '
		template = template + f'["FU", 3, [0, 1, 2]], ["SP", 3, 0, 128, {[1]}, 1], ["AN", 3, 1, 2], ["FFSP", 3, 0, [3, 2, 1], 1, 1], ["AN", 3, 1, 6], '
		template = template + f'["FU", 1, [0, 1, 2]], ["SP", 1, 0, 256, {[1]}, 1], ["AN", 1, 1, 2], ["FFSP", 1, 0, [3, 2, 1], 1, 1], ["AN", 1, 1, 6], '
		template = template + f'["PR", 4, 0, "auto_unroll_max_step${2500}"]]]], '
		template = template + '"r": [[0.000337517], 0, 13.701, 1653860304], "v": "v0.6"}\n'
	elif op_type == 'bmm_nn':
		template = '{"i": [["[\\"batch_matmul_noTrans\\", [1, 1728, 192], [1, 192, 960]]", "cuda -keys=cuda,gpu -arch=sm_80 -max_num_threads=1024 -thread_warp_size=32", [-1, 16, 64, 49152, 2147483647, 1024, 8, 32], "", 0, []], [[], [["CHW", 2, "local"], '
		template = template + f'["SP", 2, 0, 1, {split_infos[0]}, 1], ["SP", 2, 5, 1728, {split_infos[1]}, 1], ["SP", 2, 10, 960, {split_infos[2]}, 1], ["SP", 2, 15, 192, {split_infos[3]}, 1], '
		template = template + '["RE", 2, [0, 5, 10, 1, 6, 11, 2, 7, 12, 15, 16, 3, 8, 13, 17, 4, 9, 14]], ["FSP", 3, 0, 1, 3], ["FSP", 3, 4, 2, 3], ["FSP", 3, 8, 3, 3], ["RE", 3, [0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11]], ["CA", 2, 3, 8], ["CHR", 1, "shared", [2]], ["CA", 2, 3, 9], ["CHR", 0, "shared", [3]], ["CA", 1, 4, 9], ["FU", 5, [0, 1, 2]], ["AN", 5, 0, 5], ["FU", 5, [1, 2, 3]], ["AN", 5, 1, 4], ["FU", 5, [2, 3, 4]], ["AN", 5, 2, 6], '
		template = template + f'["FU", 3, [0, 1, 2]], ["SP", 3, 0, 24, {[1]}, 1], ["AN", 3, 1, 2], ["FFSP", 3, 0, [3, 2, 1], 1, 1], ["AN", 3, 1, 6], '
		template = template + f'["FU", 1, [0, 1, 2]], ["SP", 1, 0, 192, {[1]}, 1], ["AN", 1, 1, 2], ["FFSP", 1, 0, [3, 2, 1], 1, 1], ["AN", 1, 1, 6], '
		template = template + f'["PR", 4, 0, "auto_unroll_max_step${2500}"]]]], '
		template = template + '"r": [[4.56453e-05], 0, 2.23268, 1656173832], "v": "v0.6"}\n'
	elif op_type == 'conv2d':
		template = '{"i": [["[\\"conv2d_nchw\\", [108, 8, 8, 12], [2080, 8, 8, 12], [1, 1], 0, [1, 1], \\"float32\\"]", "cuda -keys=cuda,gpu -arch=sm_80 -max_num_threads=1024 -thread_warp_size=32", [-1, 16, 64, 49152, 2147483647, 1024, 8, 32], "", 0, []], [[], [["CHW", 3, "local"], '
		template = template + f'["SP", 3, 0, 108, {split_infos[0]}, 1], ["SP", 3, 5, 2080, {split_infos[1]}, 1], ["SP", 3, 10, 1, {split_infos[2]}, 1], ["SP", 3, 15, 1, {split_infos[3]}, 1], ["SP", 3, 20, 8, {split_infos[4]}, 1], ["SP", 3, 23, 8, {split_infos[5]}, 1], ["SP", 3, 26, 12, {split_infos[6]}, 1], '
		template = template + '["RE", 3, [0, 5, 10, 15, 1, 6, 11, 16, 2, 7, 12, 17, 20, 23, 26, 21, 24, 27, 3, 8, 13, 18, 22, 25, 28, 4, 9, 14, 19]], ["FSP", 4, 0, 1, 3], ["FSP", 4, 4, 2, 3], ["FSP", 4, 8, 3, 3], ["FSP", 4, 12, 4, 3], ["RE", 4, [0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15]], ["CA", 3, 4, 11], ["CHR", 2, "shared", [3]], ["CA", 3, 4, 14], ["CHR", 1, "shared", [4]], ["CA", 2, 5, 14], ["CI", 1], ["FU", 6, [0, 1, 2, 3]], ["AN", 6, 0, 5], ["FU", 6, [1, 2, 3, 4]], ["AN", 6, 1, 4], ["FU", 6, [2, 3, 4, 5]], ["AN", 6, 2, 6], '
		template = template + f'["FU", 4, [0, 1, 2, 3]], ["SP", 4, 0, 96, {[1]}, 1], ["AN", 4, 1, 2], ["FFSP", 4, 0, [4, 3, 2, 1], 1, 1], ["AN", 4, 1, 6], '
		template = template + f'["FU", 2, [0, 1, 2, 3]], ["SP", 2, 0, 192, {[1]}, 1], ["AN", 2, 1, 2], ["FFSP", 2, 0, [4, 3, 2, 1], 1, 1], ["AN", 2, 1, 6], '
		template = template + f'["PR", 5, 0, "auto_unroll_max_step${2500}"]]]], '
		template = template + '"r": [[0.00011504], 0, 2.72844, 1655046694], "v": "v0.6"}\n'
	elif op_type == 'padding':
		template = '{"i": [["[\\"padding\\", [2048, 768], [2100, 768]]", "cuda -keys=cuda,gpu -arch=sm_80 -max_num_threads=1024 -thread_warp_size=32", [-1, 16, 64, 49152, 2147483647, 1024, 8, 32], "", 0, []], [[], [["FU", 1, [0, 1]], '
		template = template + f'["SP", 1, 0, 1612800, {split_infos[0]}, 1], ["AN", 1, 0, 5], ["AN", 1, 1, 6]]]], '
		template = template + '"r": [[2.08407e-05], 0, 1.52146, 1657117451], "v": "v0.6"}\n'
	# 
	# print(template)
	inp, res = tvm.auto_scheduler.measure_record.load_record_from_string(template)
	return inp.state




def split_list_into_rngs(keys, rng_len):
	'''
		Split the keys into several groups, where the value range of a group is [n*rng_len+1,(n+1)*rng_len]. 
		keys: list of values.
		rng_len: the length of each range.
	'''
	keys = sorted(keys)
	rng_i = 1
	ret = [list()]
	for k in keys:
		if k <= rng_i * rng_len:
			ret[-1].append(k)
		else:
			while(k > rng_i * rng_len):
				rng_i+=1
			ret.append([k])
	return ret






# def get_max_alive_variables_given_split_info(split_infos, op_type):
# 	'''
# 	Compute the average number of alive variables along all the loop iterations.
# 	'''
# 	if op_type == "batch_matmul":
# 		# we only need to compute the alive range ratio for each variable and then sum them up
# 		(i0, ti, i1, i2), (j0, tj, j1, j2), (k0, tk, k1, k2), (r1, r2) = split_infos
# 		# r0 = config_dict.fetch_num
# 		loops = [r1, i1, j1, k1, r2, i2, j2, k2, i0, j0, k0]
# 		loops_size = get_product(loops)
# 		X_size = loops_size / k0 / k1 / k2
# 		Y_size = loops_size / j0 / j1 / j2
# 		O_size = loops_size / r1 / r2
# 		ret = 0
# 		ret = ret + (-xy2idx([0 for i in range(len(loops))], loops) + \
# 					xy2idx((0, 0, 0, k1-1, 0, 0, 0, k2-1, 0, 0, k0-1), loops)) / loops_size * X_size
# 		ret = ret + (-xy2idx([0 for i in range(len(loops))], loops) + \
# 					xy2idx((0, 0, j1-1, 0, 0, 0, j2-1, 0, 0, j0-1, 0), loops)) / loops_size * Y_size
# 		ret = ret + O_size
# 		return ret
# 	elif op_type == "dense":
# 		(i0, ti, i1, i2), (j0, tj, j1, j2), (r1, r2) = split_infos
# 		# r0 = config_dict.fetch_num
# 		loops = [r1, i1, j1, r2, i2, j2, i0, j0]
# 		loops_size = get_product(loops)
# 		X_size = loops_size / j0 / j1 / j2
# 		Y_size = loops_size / i0 / i1 / i2
# 		O_size = loops_size / r1 / r2
# 		ret = 0
# 		ret = ret + (-xy2idx([0 for i in range(len(loops))], loops) + \
# 					xy2idx((0, 0, j1-1, 0, 0, j2-1, 0, j0-1), loops)) / loops_size * X_size
# 		ret = ret + (-xy2idx([0 for i in range(len(loops))], loops) + \
# 					xy2idx((0, i1-1, 0, 0, i2-1, 0, i0-1, 0), loops)) / loops_size * Y_size
# 		ret = ret + O_size
# 		return ret		







def eto_tune(mick_ids, flat_all_micK_shapes, op_type, tune_option, 
	mick_cts, best_costs, best_states, mick_best_cts, dead_micks, mick_costs_history, log_file):
	for mick_i in mick_ids:
		if mick_i in dead_micks:
			continue
		msp = flat_all_micK_shapes[mick_i]
		print(f"msp:{msp}")
		mick = None
		if op_type == 'dense':
			mick = auto_scheduler.SearchTask(
			    func=dense_layer, args=((msp[0], msp[2]), (msp[1], msp[2])), target=tvm.target.Target("cuda")
			)
		elif op_type == 'bmm':
			mick = auto_scheduler.SearchTask(
			    func=batch_matmul, args=((msp[0], msp[1], msp[3]), (msp[0], msp[2], msp[3])), target=tvm.target.Target("cuda")
			)
		elif op_type == 'bmm_nn':
			mick = auto_scheduler.SearchTask(
			    func=batch_matmul_noTrans, args=((msp[0], msp[1], msp[3]), (msp[0], msp[3], msp[2])), target=tvm.target.Target("cuda")
			)
		elif op_type == 'conv2d':
			n, c, h, w, rc, rh, rw, stride, padding, dilation = msp
			sh, sw = stride
			dh, dw = dilation
			mick = auto_scheduler.SearchTask(
				    func=conv2d_nchw, args=((n, rc, sh*(h-1)+dh*(rh-1)+1, sw*(w-1)+dw*(rw-1)+1), (c, rc, rh, rw), stride, padding, dilation, "float32"), 
					target=tvm.target.Target("cuda")
				)
		# 
		task = get_task_to_measure_for_mick(msp, op_type)
		repl = get_repl_to_measure_for_mick(op_type)
		# tile_knob_list = get_tile_sizes_for_measure(mick, msp, op_type)
		# for split_infos in tile_knob_list:
		# 	print(split_infos)
		# 	state = gen_state_given_tiles(split_infos, op_type)
		# 	states_to_measure.append(state)
		# 	tasks_to_measure.append(task)
		# MeasureRess = measure_tasks_and_states(tune_option, tasks_to_measure, states_to_measure, return_MeasureRes = True)
		# 
		tile_knobs_dict = get_tile_sizes_for_measure(mick, msp, op_type, repl)
		print(tile_knobs_dict)
		thdNum_groups = split_list_into_rngs(tile_knobs_dict.keys(), 32*4)
		for thdNum_group in thdNum_groups:
			states_to_measure, tasks_to_measure = list(), list()
			for thdNum in thdNum_group:
				# split_infos = tile_knobs_dict[thdNum]
				for split_infos in tile_knobs_dict[thdNum]:
					print(split_infos)
					state = gen_state_given_tiles(split_infos, op_type)
					states_to_measure.append(state)
					tasks_to_measure.append(task)
			MeasureRess = measure_tasks_and_states(tune_option, tasks_to_measure, states_to_measure, return_MeasureRes = True)
			best_cost_updated = False
			mick_cts[mick_i] += 1
			for state, res in zip(states_to_measure, MeasureRess):
				if res == None:
					continue
				auto_scheduler.save_records(log_file, [MeasureInput(task, state)], [res])
				cost = np.mean([v.value for v in res.costs])
				if cost < best_costs[mick_i]:
					best_cost_updated = True
					mick_best_cts[mick_i] = mick_cts[mick_i]
					best_costs[mick_i] = cost
					best_states[mick_i] = state
			mick_costs_history[mick_i].append(best_costs[mick_i])
			if not best_cost_updated:
				break
		# 
		# mick_cts[mick_i] += 1
		# for state, res in zip(states_to_measure, MeasureRess):
		# 	if res == None:
		# 		continue
		# 	auto_scheduler.save_records(log_file, [MeasureInput(task, state)], [res])
		# 	cost = np.mean([v.value for v in res.costs])
		# 	if cost < best_costs[mick_i]:
		# 		mick_best_cts[mick_i] = mick_cts[mick_i]
		# 		best_costs[mick_i] = cost
		# 		best_states[mick_i] = state
		# mick_costs_history[mick_i].append(best_costs[mick_i])
	for mick_i in mick_ids:
		dead_micks.add(mick_i)







def measure_padding_ops_for_selection_res(tune_option, UCB_res, task_shapes, flat_all_micK_shapes, op_type, padding_state=None):
	'''
		This function tunes the required padding operators for the mick selection result.
		Return: a list of the best padding state for each task.
	'''
	# we do not tune padding, but directly set the tile size to 1024
	split_infos = [[1024]]
	state = gen_state_given_tiles(split_infos, 'padding')
	# 
	ret = dict()
	sp_len = len(task_shapes[0])
	if op_type == 'conv2d':
		sp_len = 7
	for task_i, (_, mick_i) in UCB_res.items():
		msp = flat_all_micK_shapes[mick_i]
		tsp = task_shapes[task_i]
		psp = tuple([math.ceil(tsp[i]/msp[i])*msp[i] for i in range(sp_len)]+msp[sp_len:])
		ori_inp_sps = get_inp_shapes_from_sp(tsp, op_type)
		new_inp_sps = get_inp_shapes_from_sp(psp, op_type)
		best_costs = list()
		for ori_sp, new_sp in zip(ori_inp_sps, new_inp_sps):
			if ori_sp == new_sp:
				best_costs.append(0)
				continue
			# we measure the cost using the state from padding_state
			# state = padding_state[new_sp]
			task = auto_scheduler.SearchTask(
					func=padding_layer, args=(ori_sp, new_sp), target=tvm.target.Target("cuda")
				)
			costs = measure_tasks_and_states(tune_option, [task], [state])
			best_costs.append(costs[0])
		ret[task_i] = best_costs
	return ret



def get_feature_for_mick_shape_old(msp, op_type, repl):
	'''
		This function is to verify the assumption that the three memory feature may help predict the latency of different mick shape.
	'''
	mick, task, out_shape = None, None, None
	tsp = [msp[i]*repl[i] for i in range(len(msp))]
	assert op_type == 'dense'
	if op_type == 'dense':
		mick = auto_scheduler.SearchTask(
		    func=dense_layer, args=((msp[0], msp[2]), (msp[1], msp[2])), target=tvm.target.Target("cuda")
		)
		task = auto_scheduler.SearchTask(
		    func=dense_layer, args=((tsp[0], tsp[2]), (tsp[1], tsp[2])), target=tvm.target.Target("cuda")
		)
		out_shape = tsp[:2]
	elif op_type == 'bmm':
		mick = auto_scheduler.SearchTask(
		    func=batch_matmul, args=((msp[0], msp[1], msp[3]), (msp[0], msp[2], msp[3])), target=tvm.target.Target("cuda")
		)
		task = auto_scheduler.SearchTask(
		    func=batch_matmul, args=((tsp[0], tsp[1], tsp[3]), (tsp[0], tsp[2], tsp[3])), target=tvm.target.Target("cuda")
		)
		out_shape = tsp[:3]
	tile_knobs_dict = get_tile_sizes_for_measure(mick, msp, op_type)
	print(tile_knobs_dict)
	# we consider the thread number >= 128 (if any) or the largest thread number (if all thread numbers < 128)
	thdNum = None
	for k, v in tile_knobs_dict.items():
		if k >= 128:
			thdNum = k
	if thdNum == None:
		thdNum = max(tile_knobs_dict.keys())
	# we compute the memory features according to the "best" split info related to thdNum (i.e., the first one)
	loop = msp
	op_para = get_op_para_ansor(task, tsp)
	split_infos = tile_knobs_dict[thdNum][0]
	blk_shape = [get_product(split_infos[i]) for i in range(2)]
	thrd_shape = [split_infos[i][1] for i in range(2)]
	vthrd_shape = [split_infos[i][0] for i in range(2)]
	reduc_shape = [msp[2]]
	blk_num = get_product(repl[:-1])
	confl = cal_bk_cflct(blk_shape, thrd_shape, vthrd_shape, reduc_shape, op_para)
	ldSNum = load_per_thread(blk_shape, thrd_shape, vthrd_shape, reduc_shape, op_para)
	# stGNum = cal_32B_trans_stG(blk_shape, thrd_shape, vthrd_shape, op_para)
	stGNum = cal_32B_trans_stG(blk_shape, thrd_shape, vthrd_shape, op_type, out_shape)
	ldGNum = sum(cal_32B_trans(blk_shape, op_para, simplify = True, load_reduc_shape = reduc_shape))
	mem_cost = (stGNum*blk_num, sum([confl[i]*ldSNum[i] for i in range(len(ldSNum))])*get_product(repl), ldGNum*get_product(repl))
	return mem_cost








def get_feature_for_mick_shape(msp, op_type, repl=None):
	'''
		This function is to verify the assumption that the three memory feature may help predict the latency of different mick shape.
	'''
	if repl == None:
		repl = get_repl_to_measure_for_mick(op_type)
	tsp = [msp[i]*repl[i] for i in range(len(repl))] + msp[len(repl):]
	# mick = get_padded_op(msp, repl=None, tsp=msp, op_type=op_type)
	sp_len = len(msp)
	Ssp_len = sp_len - 1
	if op_type == 'conv2d':
		Ssp_len = 4
		sp_len = 7
	out_shape = tsp[:Ssp_len]
	# 
	# tile_knobs_dict = get_tile_sizes_for_measure(mick, msp, op_type, repl)
	# print(f"tile_knobs_dict.keys(): {tile_knobs_dict.keys()}")
	# we consider the thread number >= 128 (if any) or the largest thread number (if all thread numbers < 128)
	blk_size = get_product(msp[:Ssp_len])
	thdNum = None
	thdNums = None
	if blk_size % 128 == 0:
		thdNum = 128
		# thdNums = [k for k in get_factors(blk_size) if k%128==0 and k>=128][:3]
	else:
		tmp_ks = [k for k in get_factors(blk_size) if k%32==0]
		# print(tmp_ks)
		for ki, k in enumerate(tmp_ks):
			if k >= 128:
				if (k-128) < (128-tmp_ks[ki-1]):
					thdNum = k
				else:
					thdNum = tmp_ks[ki-1]
				# thdNums = tmp_ks[max(0, ki-2):ki+2] 
				break
		if max(tmp_ks) < 128:
			thdNum = max(tmp_ks)
			# thdNums = tmp_ks[-3:] 
	thdNums = [thdNum]
	# print(f"thdNums: {thdNums}")
	# 
	# thdNums = [k for k in get_factors(blk_size) if k%32==0]
	mick_Ssp = msp[:Ssp_len]
	blk_shape = mick_Ssp
	reduc_shape = msp[Ssp_len:sp_len]
	# op_para = get_op_para_ansor(mick, msp)
	min_tot_stG = float('inf')
	for thdNum in thdNums:
		mem_cost_list, tile_shapes = list(), list()
		thrd_shapes = get_combinations(thdNum, [f"axis{j}" for j in range(len(mick_Ssp))])
		thrd_shapes = dict2list(thrd_shapes, [f"axis{j}" for j in range(len(mick_Ssp))])
		# print(f"thrd_shapes:{thrd_shapes}")
		for thrd_shape in thrd_shapes:
			if False in [(mick_Ssp[i] % thrd_shape[i] == 0) for i in range(len(mick_Ssp))]:
				# this thrd_shape is invalid
				continue
			vthrd_shapes = itertools.product(*[get_factors(mick_Ssp[i] // thrd_shape[i]) for i in range(len(mick_Ssp))])
			# print(f"vthrd_shapes:{list(vthrd_shapes)}")
			for vthrd_shape in vthrd_shapes:
				# confl = cal_bk_cflct(blk_shape, thrd_shape, vthrd_shape, reduc_shape, op_para)
				# ldSNum = load_per_thread(blk_shape, thrd_shape, vthrd_shape, reduc_shape, op_para)
				stGNum = cal_32B_trans_stG(blk_shape, thrd_shape, vthrd_shape, op_type, out_shape)
				# mem_cost = (stGNum, sum([confl[i]*ldSNum[i] for i in range(len(ldSNum))]))
				mem_cost = (stGNum,)
				mem_cost_list.append(mem_cost)
				tile_shapes.append((thrd_shape, vthrd_shape))
				# if mem_cost < min_mem_cost:
				# 	best_tile_shape = (thrd_shape, vthrd_shape)
				# 	min_mem_cost = mem_cost
		# we return the tile size setting related to the best_split_infos
		# print(best_split_infos, mick_Ssp)
		# thrd_shape, vthrd_shape = best_tile_shape
		# best_idx = sorted(range(len(mem_cost_list)), key=lambda i: mem_cost_list[i])[:3]
		best_idx = np.argpartition(mem_cost_list, 0)[0]
		thrd_shape, vthrd_shape = tile_shapes[best_idx[0]]
		mem_cost = mem_cost_list[best_idx[0]]
		# print(get_product(blk_shape)/get_product(thrd_shape)* mem_cost[0])
		tot_stG = get_product(blk_shape)/get_product(thrd_shape)* mem_cost[0]
		if tot_stG < min_tot_stG:
			min_tot_stG = tot_stG
		# return (get_product(thrd_shape), 
		# 	get_product(blk_shape)/get_product(thrd_shape), mem_cost[0], mem_cost[1])
	return min_tot_stG
	# 
	# we compute the memory features according to the "best" split info related to thdNum (i.e., the first one)
	# split_infos = tile_knobs_dict[thdNum][0] # we only check the data of the first candidate split info
	# split_infos = split_infos[:Ssp_len]
	# blk_shape = [get_product(tile) for tile in split_infos]
	# assert blk_shape == msp[:Ssp_len]
	# thrd_shape = [tile[1] for tile in split_infos]
	# vthrd_shape = [tile[0] for tile in split_infos]
	# reduc_shape = msp[Ssp_len:sp_len]
	# # 
	# op_para = get_op_para_ansor(mick, msp)
	# confl = cal_bk_cflct(blk_shape, thrd_shape, vthrd_shape, reduc_shape, op_para)
	# ldSNum = load_per_thread(blk_shape, thrd_shape, vthrd_shape, reduc_shape, op_para)
	# stGNum = cal_32B_trans_stG(blk_shape, thrd_shape, vthrd_shape, op_type, out_shape)
	# mem_cost = (get_product(thrd_shape), 
	# 	get_product(blk_shape)/get_product(thrd_shape), stGNum, sum([confl[i]*ldSNum[i] for i in range(len(ldSNum))]))
	# return (mem_cost)










def search_mick_with_feadback(cost_model_params, Edges, tasks, all_micK_shapes, op_type, log_file, iter_res_log, file_id, Ssp_poses, overall_start_time):
	'''
		This method search a good set of micro-kernel shapes for a set of operator shapes.
		1. We run UCB according to the original cost model.
		2. We measure the mick_shapes on hardware.
		3. We tune other mick_shapes close to those selected ones. 
		4. We correct the cost model according to the measured results.
		5. We rerun the UCB according to the updated cost model, and go to step 3, until we run out of time.
		Input:
			Edges: the original cost model.
	'''
	start_time = time.time()
	taskNum = len(tasks)
	task_shapes = list()
	for count, task in enumerate(tasks):
		# X_shape, Y_shape = get_inp_shapes(task.workload_key)
		# s_key = (X_shape[0], Y_shape[0], X_shape[1])
		s_key = get_output_shape_from_wlk(task.workload_key, op_type)
		task_shapes.append(s_key)


	policy_dict = dict()
	cost_model = auto_scheduler.cost_model.XGBModel()
	measure_ctx = auto_scheduler.LocalRPCMeasureContext(min_repeat_ms=300)
	tune_option = auto_scheduler.TuningOptions(
	    num_measure_trials=1000,  # change this to 1000 to achieve the best performance
	    runner=measure_ctx.runner,
	    measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
	    verbose=0,
	)


	measurer = auto_scheduler.measure.ProgramMeasurer(
				tune_option.builder, tune_option.runner,
				tune_option.measure_callbacks, tune_option.verbose
				)


	flat_all_micK_shapes = all_micK_shapes
	num_measures_per_round = tune_option.num_measures_per_round # // 2
	mickNum = len(flat_all_micK_shapes)
	mick_cts = [0 for _ in range(mickNum)]
	# mick_best_cts[i] saves the round task i found the best latency
	mick_best_cts = [0 for _ in range(mickNum)]
	# task_costs_history[i] saves the latency history of task i
	mick_costs_history = [[] for _ in range(mickNum)]
	# best_costs[i] saves the best latency of task i
	best_costs = 1e10 * np.ones(mickNum)
	best_states = [None for _ in range(mickNum)]
	dead_micks = set()

	update_num = 10
	with open(iter_res_log, 'a') as f:
		f.write("start a new run!"+'-'*50 + '\n')


	avg_errors = None
	if op_type == 'dense':
		avg_errors = {'Ssize': {32: 1.0214696308566988, 64: 1.045592479114492, 96: 0.8343696874721203, 128: 0.7698130620905423, 160: 0.8767722920885559, 192: 0.8540447110826465, 256: 0.8494444194638382, 288: 0.9536521385245709, 320: 1.0172428271551377, 384: 0.956747266122913, 480: 0.9948164103884493, 512: 0.9218962737026414, 576: 1.006149352687396, 640: 0.9281286837699441, 768: 0.9515114883957175, 800: 1.29262495463531, 864: 1.1394622608388487, 960: 1.0104593519709046, 1024: 0.9004924649814064, 1152: 0.8902262693495125, 1280: 0.9167099751946637, 1440: 0.9297260251531607, 1536: 0.9091561761604776, 1600: 1.0319091061481505, 1728: 0.9643938123325433, 1920: 1.0073993149366227, 2048: 0.9731589215624579, 2304: 1.0048368293513714, 2400: 0.987226096824682, 2560: 1.0669511840756476, 2592: 1.0413641543068606, 2880: 1.02850127458238, 3072: 0.9663645295478047, 3200: 1.2964020286107065, 3456: 1.2390750018233039, 3840: 1.0940487664814549, 4000: 1.2708302024104285, 4096: 1.0029956503451507, 4320: 1.0446201221640092, 4608: 0.9942145292262636, 4800: 1.1649724752951143, 5120: 0.9864503632637284, 5184: 1.123693007039693, 5760: 1.0368404058581726, 6144: 0.9856377787936311, 6400: 1.011761821432629, 6912: 1.0212256157145672, 7200: 1.0381989976087431, 7680: 1.0216585887534768, 7776: 1.2184166244495296, 8000: 1.2929924835928825, 8192: 0.9954618822549653, 8640: 1.0849665130882176, 9216: 1.0612509090590567, 9600: 1.00710208379765, 10240: 1.0913616529452028, 10368: 1.1014552414777685, 11520: 1.0342768997129952, 12000: 1.0619993783932828, 12288: 1.0192951804340586, 12800: 1.4209754495497964, 12960: 1.1862957427736642, 13824: 1.122646248865884, 14400: 1.1919269445755876, 15360: 1.1446314815030512, 15552: 1.2266362740533354, 16000: 1.3481953548377292, 16384: 1.0834094358529558, 17280: 1.0878972697673395, 18432: 1.0470998103931999, 19200: 1.0361298648709658, 20000: 1.0829008228791177, 20480: 1.0066264931606297, 20736: 1.0132747972389533, 21600: 1.0411890191777808, 23040: 0.9789400195126984, 23328: 1.0662363962516295, 24000: 1.0152010247005732, 24576: 0.9446564832097044, 25600: 0.9371181678598541, 25920: 0.9875082069292697}, 'Rsp': {(24,): 0.9864503632637284, (12,): 1.0172261096520885, (15,): 1.0308406416828917, (16,): 1.0743272199939775, (18,): 1.0318244647355042, (20,): 0.9868225192685836, (25,): 1.0001002020476235, (27,): 1.0083256505438167, (30,): 1.012749391221672, (32,): 1.278432403664182, (36,): 1.0268537737423424, (40,): 1.0043927146265066, (45,): 1.0092287563140259, (48,): 1.0036060224048478, (50,): 0.9924959098565611, (54,): 1.0188271180189452, (60,): 1.0418499715328406, (64,): 1.4695339847463393, (72,): 1.0059252084629924, (75,): 1.0114233483871524, (80,): 1.1995154567719752, (81,): 1.0062793413712159, (6,): 1.0187721197565016, (8,): 1.02973015618515, (9,): 1.049648912331954, (10,): 1.038277670523981, (90,): 0.7857037842331424, (96,): 1.0046874446459662, (100,): 0.7519698029709507, (108,): 1.0187134602323493, (120,): 1.0159152499224813, (125,): 0.6585427569111081, (128,): 1.0037881872497398}}
	elif op_type == 'bmm':
		avg_errors = {'Ssize': {32: 1.0176097185277906, 64: 1.0288994880091213, 96: 0.8426443595777292, 128: 0.7571662779561824, 160: 0.8532493465664276, 192: 0.8384008844267786, 256: 0.839111675364786, 288: 0.9417677908420868, 320: 1.012712404414941, 384: 0.9485282128354215, 480: 0.9812960641690706, 512: 0.9003962208373306, 576: 1.0072627829094112, 640: 0.9151519100513762, 768: 0.9508898952793744, 800: 1.279281392835785, 864: 1.1284161413374807, 960: 1.004555871699807, 1024: 0.8906971927871383, 1152: 0.8765226174483574, 1280: 0.8990110287682678, 1440: 0.9065612229103365, 1536: 0.8830280720681428, 1600: 1.0180247815113965, 1728: 0.9332694780575803, 1920: 1.0050974908519406, 2048: 0.9356737239261032, 2304: 0.9589107959957236, 2400: 0.955105209852458, 2560: 1.0192667478339583, 2592: 1.014331803833898, 2880: 0.969324472916845, 3072: 0.9496303982657172, 3200: 1.2771750376269657, 3456: 1.2283925495693135, 3840: 1.0919287064893444, 4000: 1.2692543831768057, 4096: 1.0049796951386265, 4320: 1.0409300137652773, 4608: 0.9932326690860256, 4800: 1.0639024818522573, 5120: 0.9858056401703911, 5184: 1.1356133984964312, 5760: 1.0345878837213098, 6144: 0.9824446094195717, 6400: 1.0057263392077904, 6912: 1.1406222160374369, 7200: 1.0514392542107096, 7680: 1.0730006695846346, 7776: 1.2142477177985804, 8000: 1.2765496351885333, 8192: 0.9986446058149828, 8640: 1.0653747321336866, 9216: 0.9972849595080133, 9600: 1.0017004533457319, 10240: 0.9944728203362209, 10368: 1.1512820265444108, 11520: 1.0220379408870157, 12000: 1.090222425404644, 12288: 1.0001199885775138, 12800: 1.4023675841263248, 12960: 1.143399696133763, 13824: 1.1865483404457482, 14400: 1.0140201192130878, 15360: 1.0897008425853287, 15552: 1.1740039461088065, 16000: 1.3890495280867008, 16384: 1.0750474071354685, 17280: 1.0689821434057598, 18432: 1.0437518048030137, 19200: 1.0374682812517413, 20000: 1.0841126721243575, 20480: 1.0080748206067953, 20736: 1.0109540991295467, 21600: 1.0460800647488824, 23040: 0.9838878831548975, 23328: 1.0495159124461306, 24000: 1.0150595956746886, 24576: 0.9369764538911551, 25600: 0.9656522466897478, 25920: 0.9790737647108937}, 'Rsp': {(24,): 0.9858056401703911, (6,): 1.0203366933879199, (8,): 1.019618703319162, (9,): 1.0525220730436422, (10,): 1.0353771456762826, (12,): 1.0912375739037532, (15,): 1.0350036645360703, (16,): 1.0790164216050997, (18,): 1.0342487396265727, (20,): 0.9919517882824961, (25,): 0.997349606015983, (27,): 1.0042049197811551, (30,): 1.0139083487677247, (32,): 1.2681435172223483, (36,): 1.0184161121575457, (40,): 1.0007053129961054, (45,): 1.0105410763529359, (48,): 1.0035254675555254, (50,): 0.9960876514515554, (54,): 1.028537106247306, (60,): 1.033451611036558, (64,): 1.489758526940214, (72,): 1.023801457615359, (75,): 1.0310822308329093, (80,): 1.2327054703876097, (81,): 1.0354261320021396, (90,): 0.7854203679878975, (96,): 1.0036801009940586, (100,): 0.7523667960451703, (108,): 1.017847446941305, (120,): 1.0132421291402827, (125,): 0.6575749812608305, (128,): 1.003021397984787}}
	elif op_type == 'bmm_nn':
		avg_errors = {'Ssize': {32: 1.1607406861501768, 64: 1.127880672302462, 96: 0.9231357228774195, 128: 1.009102438189609, 160: 1.1100507460663158, 192: 0.9253814927614425, 256: 1.0074962614675456, 288: 1.0097547985851043, 320: 1.0285683608048468, 384: 1.1003114198131345, 480: 1.1999685818414372, 512: 1.0564082214369586, 576: 1.0169969845246336, 640: 1.5259224781728054, 768: 1.066180695797167, 800: 1.573786375970281, 864: 1.5682950397332796, 960: 1.063988160763565, 1024: 0.8498467422161217, 1152: 0.9545310813248662, 1280: 0.8415255286077149, 1440: 0.9667282609112736, 1536: 0.8580663170426123, 1600: 1.011569373365247, 1728: 0.9988835164957676, 1920: 0.9745855668611741, 2048: 0.9031442488417949, 2304: 0.9565226384246905, 2400: 1.11242944519462, 2560: 0.9921000288911905, 2592: 1.0599479245241603, 2880: 1.0499682271570825, 3072: 1.0012068325158412, 3200: 1.9665263980857741, 3456: 1.9246378627425713, 3840: 1.3570541720376064, 4000: 1.2661363636972274, 4096: 1.0030659698766196, 4320: 1.18244358331586, 4608: 0.9627213581481765, 4800: 1.1990452262673559, 5120: 0.9766725226259192, 5184: 1.108687744248711, 5760: 1.0197406674161786, 6144: 0.9659435407447541, 6400: 1.0059727532311766, 6912: 1.0258357430465972, 7200: 1.2077507126068783, 7680: 1.0035385440753521, 7776: 1.3119344838233502, 8000: 1.3136172222363727, 8192: 0.9729193370372459, 8640: 1.1565025688099664, 9216: 0.9855743263619776, 9600: 1.0351715303483002, 10240: 1.0173843729788776, 10368: 1.0197105229833427, 11520: 1.0175781253773601, 12000: 1.1126842219242283, 12288: 0.9957949735428318, 12800: 1.190714424812476, 12960: 1.3568561875387912, 13824: 1.247008777210848, 14400: 1.167907996338686, 15360: 1.0252253812175083, 15552: 1.1850965465558465, 16000: 2.0579052040887236, 16384: 0.9671870546978222, 17280: 1.2932631043755303, 18432: 0.9623083566391222, 19200: 1.1518818741289567, 20000: 1.2627500029301515, 20480: 1.0050122389725251, 20736: 1.0069448247142452, 21600: 1.080163080614124, 23040: 0.9777951463746883, 23328: 1.0438333288517176, 24000: 1.0772704746118915, 24576: 0.9213893419194037}, 'Rsp': {(6,): 0.9814965104912557, (8,): 0.9908037022525991, (9,): 1.0366925424230995, (10,): 1.0102956205907783, (12,): 0.9826417909754236, (15,): 1.0170111701765097, (16,): 0.9634937053104822, (18,): 1.0137598818778593, (20,): 0.9972380682400224, (24,): 0.9766725226259192, (25,): 0.9844378707536289, (27,): 0.993924882795097, (30,): 0.9759333388420665, (32,): 0.9474041305283946, (36,): 0.9774573181299077, (40,): 0.9688130001403968, (45,): 1.0282555827829976, (48,): 0.9977333693967011, (50,): 1.0347943237345152, (54,): 1.019982729410486, (60,): 1.0155254250543584, (64,): 0.9724698534149178, (72,): 0.9905015638243764, (75,): 1.021105906911335, (80,): 0.9877059640782966, (81,): 1.0235243685350415, (90,): 1.022051117409556, (96,): 1.0228866704166464, (100,): 1.051681667616403, (108,): 1.022028682967015, (120,): 1.0412196168894319, (125,): 1.0513937059218714, (128,): 1.0213560216876265}}
	elif op_type == 'conv2d':
		avg_errors = {'Ssize': {32: 1.0427239794288254, 64: 0.6178082273541335, 96: 0.5075432874362927, 128: 0.9839883909076406, 160: 0.43070646111226174, 192: 0.44594868670397325, 256: 0.6669705881786568, 288: 0.43278443638611747, 320: 0.4158451828307742, 384: 0.6420820153997777, 480: 0.6653282330548522, 512: 0.8177267355005797, 576: 0.7104371880337164, 640: 1.0045484504047122, 768: 1.0224742743286908, 800: 0.9312919505659867, 864: 0.8390415729077711, 960: 0.7674865592669354, 1024: 0.8066820626184429, 1152: 0.844361474974047, 1280: 0.9166467564627142, 1440: 1.1013684481671535, 1536: 1.0192493824227413, 1600: 1.1052237521078145, 1728: 1.04912966131926, 1920: 1.0246321194640935, 2048: 0.9750999859824957, 2304: 0.9699704548186993, 2400: 1.222971700622945, 2560: 0.9693893701665708, 2592: 1.175667687385915, 2880: 1.1584156145870121, 3072: 1.0053151277148757, 3200: 1.1771045862986453, 3456: 1.0334802778392937, 3840: 1.1124744520757899, 4000: 1.3608134256707718, 4096: 0.9700847866210714, 4320: 1.1562454648007081, 4608: 0.9799350102958435, 4800: 1.2432355490747082, 5120: 1.083090771779669, 5184: 1.4680800625661121, 5760: 0.9968313728490399, 6144: 0.9910192973155355, 6400: 1.6766191515553401, 6912: 0.9883472385289105, 7200: 1.24009829178734, 7680: 1.0043236420723964, 7776: 1.3151832606351013, 8000: 1.1675045105769075, 8192: 0.9757560596668949, 8640: 1.5462390123721002, 9216: 1.003994167700363, 9600: 1.149959124239653, 10240: 0.9813975790336907, 10368: 1.9346190384448538, 11520: 1.3976367524794304, 12000: 1.2383174670708263, 12288: 1.0406668244648516, 12800: 1.0792762608738593, 12960: 1.4244848539458377, 13824: 1.0026276248900516, 14400: 1.312293932260893, 15360: 1.0094718565274519, 15552: 1.316182506073853, 16384: 1.0112932959351357, 17280: 1.0218094828402506, 18432: 1.0127250988285168}, 'Rsp': {6: 0.9741804422996774, 7: 0.9534334362354849, 8: 1.0389559090042235, 9: 1.0232540859306996, 10: 1.0619535860452864, 11: 1.0405343676685301, 12: 1.0299328371954988, 13: 1.028237040835579, 14: 1.0230693634550765, 15: 0.9735794060040709, 16: 0.9742618961729653, 17: 0.9972177001447046, 18: 0.9809978992910501, 19: 1.0371649492709998, 20: 0.964978947873505, 21: 0.9861271322965678, 22: 1.053211521630452, 23: 0.9888151079298257, 24: 0.9513675637710464, 25: 0.9589065311429342, 26: 1.0021100814108235, 27: 0.9746806204210842, 28: 0.978595915529926, 29: 1.016606399707906, 30: 0.9943570876662849, 31: 1.0136853018729612, 32: 0.9819629573979212, 33: 1.012189372333284, 34: 1.0502299554851675, 35: 1.0101493297782547, 36: 0.9916738313258153, 37: 1.0214460077753016, 38: 1.0229960063874715, 39: 1.0160721922952158, 40: 0.9957102527429647, 41: 0.9681740499704551, 42: 0.9862594834455454, 43: 0.9937844032249954, 44: 0.9940287985882732, 45: 0.965821634362358, 46: 1.0700187947933435, 47: 1.0600902671493164, 48: 0.9933279900145503, 49: 0.9805849242970464, 50: 1.006007623797361, 51: 1.1145538097164336, 52: 1.037468711177509, 53: 1.0686462580015814, 54: 1.0220060072117463, 55: 1.0634021869750883, 56: 1.0473768027194177, 57: 1.0888905067382328, 58: 1.109277869951779, 59: 1.0815224614374441, 60: 0.9368413670633304, 61: 0.9608956390829956, 62: 0.9611646929162242, 63: 0.954756760095103, 64: 0.9053157201874695, 65: 0.9626982143767587, 66: 0.9603751860458701, 67: 0.9786123134003084, 68: 0.9472497052561087, 69: 0.9915296329821892, 70: 0.9498072639903333, 71: 0.9643780308837449, 72: 0.9467560529475819, 73: 0.9600306242205007, 74: 0.938171409187894, 75: 0.960420236107214, 76: 0.9701319377655068, 77: 0.979836244001061, 78: 0.9421965747592607, 79: 0.9574196751132217, 80: 0.9439194282768848, 81: 0.9656990314181749, 82: 0.961219062310585, 83: 0.9861526903038892, 84: 0.9747604377754417, 85: 0.9819707669889988, 86: 0.9635237010945857, 87: 0.9713348816122479, 88: 0.9351944240076813, 89: 0.9647425799332855, 90: 0.9349575993420121, 91: 0.9635575663228679, 92: 0.9418256001606058, 93: 0.9781082896014697, 94: 0.9493364275251599, 95: 0.9932657385567581, 96: 1.2844531074165721, 97: 0.968173840793926, 98: 0.9537088253670634, 99: 1.0213549522760719, 100: 0.9677254481127218, 101: 0.9810535559276535, 102: 0.9553652630461045, 103: 0.9802349161443118, 104: 0.9387958887462997, 105: 0.9969984059589421, 106: 0.9410978639368656, 107: 0.9790941644517808, 108: 0.9348801801879957, 109: 0.9718955916251119, 110: 0.9772740309709361, 111: 0.9738636830826087, 112: 0.9271993557046047, 113: 0.9719054428521212, 114: 0.9499218549028772, 115: 0.9833732158960233, 116: 0.9282353138318402, 117: 0.9955711311544169, 118: 1.0244224894507485, 119: 0.9897966248299661, 120: 0.9796555623563592, 121: 1.014489762334747, 122: 0.9284145701687272, 123: 0.9883598373967858, 124: 0.9393741455269102, 125: 1.004026539600878, 126: 0.9690673887609199, 127: 1.0130978338161118, 128: 0.9453494843957576}}



	ori_Edges = copy.deepcopy(Edges)
	correct_cost_dict(Edges, task_shapes, flat_all_micK_shapes, avg_errors, op_type = op_type)
	ori_Edges_with_weight = copy.deepcopy(Edges)



	# avg_errors = None
	res = get_best_mick_by_cost_model(Edges, task_shapes, avg_errors=None)
	mick_ids = set([v[1] for v in res.values()])
	

	res = compute_best_cost_UCB_worker(0, (len(Edges), list(mick_ids), ori_Edges))


	print("mick_ids: ", mick_ids)
	print(f"UCB ends, predicted cost with prediction error: {query_tot_cost(Edges, res)}")
	print(f"UCB ends, predicted cost without prediction error: {query_tot_cost(ori_Edges, res)}")


	use_eto = True

	error_log = dict()
	Kth = 1
	roundNum = len(mick_ids)
	round_i = 1
	micks_pred_right = set() # stores the micks whose prediction by the cost model is close to the real cost
	pred_riht_threshold = 0.97
	infer_ratio = 1.03
	while (Kth <= roundNum): # (round_i < 2*roundNum) or (Kth <= roundNum)
		print(f"Kth:{Kth} round_i:{round_i}--------------------------------------------")
		round_i += 1
		mick_i_to_tune, tasks_to_focus = get_mick_with_Kth_latency(res, Kth)
		if mick_i_to_tune == None:
			break


		if mick_i_to_tune in micks_pred_right:
			# this mick is already tuned and its performance is as expected
			Kth += 1
			continue


		print(mick_i_to_tune, flat_all_micK_shapes[mick_i_to_tune])
		print(f"tasks_to_focus:{tasks_to_focus}")
		print(f"ori res: {res}")

		# 
		# we need to find mick shapes to replace this one
		mick_ids.remove(mick_i_to_tune)

		replace_cands = get_best_mick_with_base_mickset(mick_ids, flat_all_micK_shapes, Edges, dead_micks, op_type, 
				exclude_dead_mick = False, avg_errors = None, task_ids=tasks_to_focus, iter_res_log=iter_res_log)

		# 
		if (mick_i_to_tune in dead_micks) and (replace_cands[0] == mick_i_to_tune):
			print("no other mick is better now")
			Kth += 1
			mick_ids.add(mick_i_to_tune)
			continue
		trial_i = 0
		while(True):
			trial_i += 1
			print(f"replace_cands: {replace_cands}, {[flat_all_micK_shapes[mick_i] for mick_i in replace_cands]}")
			valid_cands = set()
			# tune the cand micks and update the cost and select the best one
			for mick_i in replace_cands[:1]:
				print(f"replace cand: {flat_all_micK_shapes[mick_i]}")
				# tmp_res = compute_best_cost_UCB_worker(0, (taskNum, list(mick_ids)+[mick_i], Edges))
				# if get_tot_latency_of_mick(tmp_res, Edges, mick_i) == 0:

				tmp_res = compute_best_cost_UCB_worker(0, (taskNum, list(mick_ids)+[mick_i], ori_Edges))
				tmp_res2 = compute_best_cost_UCB_worker(0, (taskNum, list(mick_ids)+[mick_i], Edges))


				if (get_tot_latency_of_mick(tmp_res, ori_Edges, mick_i) == 0) and (get_tot_latency_of_mick(tmp_res2, Edges, mick_i) == 0):
					print("no assigned task, so no need to tune")
					valid_cands.add(mick_i)
					break
				if mick_i in dead_micks:
					print("already tuned")
					valid_cands.add(mick_i)
					break
				if use_eto:
					eto_tune([mick_i], flat_all_micK_shapes, op_type, tune_option, 
						mick_cts, best_costs, best_states, mick_best_cts, dead_micks, mick_costs_history, log_file)
				else:
					get_search_policies_for_micks([mick_i], flat_all_micK_shapes, policy_dict, cost_model, op_type)
					tune_mick_one_round([mick_i], policy_dict, num_measures_per_round, measurer, 
						mick_cts, best_costs, best_states, mick_best_cts, dead_micks, mick_costs_history)
				predicted_cost = sum([tmp_res[task_i][0] for task_i in tasks_to_focus])
				# 
				success = correct_the_cost_model_approx_do_measure(tune_option, [mick_i], task_shapes, 
					flat_all_micK_shapes, Edges, ori_Edges, ori_Edges_with_weight, best_states, op_type, cost_model_params, 
					avg_errors, error_log, tmp_res, dead_micks, only_max_repl=False, infer_ratio=infer_ratio, Ssp_poses=Ssp_poses)
				if not success:
					# in this case, it means according to tmp_res, no task will be assigned to mick_i
					correct_the_cost_model_approx_do_measure(tune_option, [mick_i], task_shapes, 
						flat_all_micK_shapes, Edges, ori_Edges, ori_Edges_with_weight, best_states, op_type, cost_model_params, 
						avg_errors, error_log, tmp_res2, dead_micks, only_max_repl=False, infer_ratio=infer_ratio, Ssp_poses=Ssp_poses)
				# 


				tmp_res3 = compute_best_cost_UCB_worker(0, (taskNum, list(mick_ids)+[mick_i], ori_Edges))


				real_cost = sum([tmp_res3[task_i][0] for task_i in tasks_to_focus])




				# 
				print(f"predicted cost v.s. updated cost of this mick: {predicted_cost}, {real_cost}")
				print(f"tmp_res:{tmp_res}")
				print(f"tmp_res3:{tmp_res3}")
				print("="*50)
				if predicted_cost >= pred_riht_threshold*real_cost:
					valid_cands.add(mick_i)
					micks_pred_right.add(mick_i)
					break
			if len(valid_cands) > 0:
				break
			# 

			replace_cands = get_best_mick_with_base_mickset(mick_ids, flat_all_micK_shapes, Edges, dead_micks, op_type, 
				exclude_dead_mick = False, avg_errors = None, task_ids=tasks_to_focus, iter_res_log=iter_res_log)

		# 
		mick_ids.update(valid_cands)
		Kth = 1


		res = compute_best_cost_UCB_worker(0, (len(Edges), list(mick_ids), ori_Edges))

		mick_ids = set([v[1] for v in res.values()]) # squeeze the mick id set
		print(f"To find replace mick TRIAL Num: {trial_i}")
		print(f"mick tuned, predicted cost with prediction error: {query_tot_cost(Edges, res)}")
		print(f"mick tuned, predicted cost without prediction error: {query_tot_cost(ori_Edges, res)}")



	# In the end, we need to reselect from the dead micro-kernels, in case some better mick assignment is missed due to the search region restriction
	res = compute_best_cost_UCB_worker(0, (len(Edges), list(dead_micks), ori_Edges))

	end_time = time.time()

	# measure the final selected mick shape set
	ret = measure_final_res(tune_option, res, task_shapes, flat_all_micK_shapes, Edges, best_states, op_type)

	# 
	write_pysched_to_file(res, task_shapes, flat_all_micK_shapes, best_states, op_type, file_id)
	# 
	with open(iter_res_log, 'a') as f:
		f.write(f"running time of this round {end_time - start_time} seconds\n")
		f.write(f"total search time of this dataset {end_time - overall_start_time} seconds\n")
	return ret





def select_best_micK_shapes_full_dynamic_2Rngs(tasks, all_micK_shapes, max_micK_num, op_type, file_id, Ssp_poses, overall_start_time):
	'''
	Note: 2Rngs means the cost model for lower part curve fitting seperates
	the spatial loop sizes into two ranges, and uses two curves for them
	individually.

	This function is to fully depends on the micK curve pattern 
	and select the best set of micro-kernel shapes for the given
	dynamic operator shapes.
	
	We use a straight-forward greedy algorithm here, not consider max_micK_num.
	'''
	# edges = dict(), edges[src_id] = [(end_id, cost), ...]
	# cost in terms of s.
	Edges = dict()
	ret = dict()
	task_shapes = list()
	SMNum = 108
	for count, task in enumerate(tasks):
		s_key = get_output_shape_from_wlk(task.workload_key, op_type)
		task_shapes.append(s_key)
		Edges[count] = list()


	flat_all_micK_shapes = all_micK_shapes



	# 
	# cost_model = func_micK_curv
	# popt = [2.97046454, 0.69860168, -4.07368524, 13.7136372]
	func_micK_curvs = dict()
	popts = dict()
	curve_rep_layouts = dict()
	# base_blk_nums = [SMNum*i+1 for i in range(5)] + [(i+1)*SMNum for i in range(5)]
	interested_blkNs = {0:[48, 108], 1:[110, 216], 2:[220, 324], 3:[330, 432], 4:[440, 540]}
	base_blk_nums = list()
	for k in range(5):
		base_blk_nums = base_blk_nums + interested_blkNs[k]


	base_reduc_rep_nums = [2**i for i in range(7)]
	base_reduc_lens = [6, ] + list(range(12, 121, 12)) + [128, ] # [6, 12, 24, 36, 48, 60, 72, ]#18,]
	# base_reduc_lens = [6, ] + list(range(12, 73, 12)) # + [128, ] # [6, 12, 24, 36, 48, 60, 72, ]#18,]
	fix_lens = list(range(1, 22))+["inf"] #[1, 2, 3, 4, 5, 6, 7, 8, "inf"] # [1, 2, 3, 4, 5, 6, "inf"]
	# rep_keys = ['line', 'square']
	rep_keys = ['line_h', 'line_v', 'square']
	if op_type in ['bmm', 'bmm_nn']:
		rep_keys = ['line_b', 'square'] # one rep layout is only replicate along batch axis, the other one is square over h and w

	interested_Rsps = dict()
	if op_type == 'conv2d':
		# for conv2d, we draw curves for each reduction loop shape, instead of each reduction loop size
		base_red_shapes = list()
		for r_size in base_reduc_lens:
			rw = get_symmetric_factor(r_size)
			rh = r_size // rw
			base_red_shapes.append((1, rh, rw))
			# base_red_shapes.append((r_size, 1, 1))
			rw = min(factorint(r_size).keys())
			base_red_shapes.append((r_size//rw, 1, rw))
			interested_Rsps[r_size] = base_red_shapes[-2:] #[(1, rh, rw), (r_size, 1, 1)]
		base_reduc_lens = base_red_shapes
	else:
		for r_size in base_reduc_lens:
			interested_Rsps[r_size] = [(r_size, )]




	curve_poses = ['up', 'lw']
	for reduc_len in base_reduc_lens:
		for reduc_rep_num in base_reduc_rep_nums:
			for blk in base_blk_nums:
				for repK in rep_keys:
					for pos in curve_poses:
						for fixlen in fix_lens:
							key = (reduc_len, reduc_rep_num, blk, repK, pos, fixlen)
							if pos == 'up':
								func_micK_curvs[key] = func_micK_curv_ConvHull # func_micK_curv_LB #func_micK_curv_UB
							else:
								func_micK_curvs[key] = func_micK_curv_ConvHull # func_micK_curv_LB
							if repK == 'square':
								tmp = sorted(get_factors(blk), key=lambda repN: repN+blk//repN)[0]
								if tmp == 1:
									curve_rep_layouts[key] = (blk, 1)
								else:
									curve_rep_layouts[key] = (tmp, blk//tmp)
							elif repK == 'line_h':
								curve_rep_layouts[key] = (1, blk)
							elif repK == 'line_v':
								curve_rep_layouts[key] = (blk, 1)
							elif repK == 'line_b':
								curve_rep_layouts[key] = (blk, 1, 1)
							if op_type == 'conv2d':
								curve_rep_layouts[key] = curve_rep_layouts[key] + (1,1)
							if (op_type in ['bmm','bmm_nn']) and (repK == 'square'):
								curve_rep_layouts[key] = (1, ) + curve_rep_layouts[key]


	if op_type == 'dense':
		popts = cost_model_params_dense_convhull3.get_popts()
	elif op_type == 'bmm':
		popts = cost_model_params_bmm_convhull1.get_popts()
	elif op_type == 'bmm_nn':
		popts = cost_model_params_bmm_nn_convhull1.get_popts()
	elif op_type == 'conv2d':
		popts = cost_model_params_conv2d_convhull6.get_popts() 


	selected_fixedL_dict = None
	if op_type == 'dense':
		selected_fixedL_dict = {6: 2, 12: 4, 24: 8, 36: 8, 48: 8, 60: 4, 72: 6, 84: 2, 96: 4, 108: 8, 120: 4, 128: 16}
	elif op_type == 'bmm':
		selected_fixedL_dict = {108: 8, 128: 8, 6: 2, 12: 4, 24: 8, 36: 8, 48: 8, 60: 4, 72: 6, 84: 2, 96: 4, 120: 4}
	elif op_type == 'bmm_nn':
		selected_fixedL_dict = {6: 1, 12: 2, 24: 4, 36: 2, 48: 8, 60: 4, 72: 6, 84: 2, 96: 4, 108: 8, 120: 4, 128: 8}
	elif op_type == 'conv2d':
		selected_fixedL_dict = {(1, 2, 3): 1, (3, 1, 2): 1, (1, 3, 4): 1, (6, 1, 2): 1, (1, 4, 6): 2, (12, 1, 2): 2, (1, 6, 6): 2, (18, 1, 2): 2, (1, 6, 8): 4, (24, 1, 2): 4, (1, 6, 10): 4, (30, 1, 2): 4, (1, 8, 9): 4, (36, 1, 2): 4, (1, 7, 12): 2, (42, 1, 2): 2, (1, 8, 12): 8, (48, 1, 2): 8, (1, 9, 12): 2, (54, 1, 2): 2, (1, 10, 12): 8, (60, 1, 2): 8, (1, 8, 16): 8, (64, 1, 2): 16}




	tot_micK_shapes_num = len(flat_all_micK_shapes)
	workerNum = 240
	end_id_num = math.ceil(tot_micK_shapes_num/workerNum)
	first_end_ids = list(range(0, tot_micK_shapes_num, end_id_num))
	common_params = (end_id_num, flat_all_micK_shapes, op_type, task_shapes, selected_fixedL_dict, func_micK_curvs, popts, curve_rep_layouts, interested_blkNs, interested_Rsps)
	with multiprocessing.Pool(processes=workerNum) as pool:
		Edges_list = pool.starmap(comp_cost_worker_inputCompleteShapes, zip(first_end_ids, itertools.repeat(common_params)))


	for tmp_Edges in Edges_list:
		for k in Edges:
			Edges[k] = Edges[k] + tmp_Edges[k]


	for k in Edges:
		Edges[k] = sorted(Edges[k], key=lambda vi: vi[0])


	Edges_full_info = Edges
	Edges_only_cost = dict()
	for k in Edges:
		Edges_only_cost[k] = np.array([v[-1] for v in Edges[k]])


	Edges = Edges_only_cost # used in the UCB algorithm


	cost_model_params = selected_fixedL_dict, func_micK_curvs, popts, curve_rep_layouts, interested_blkNs, interested_Rsps
	log_file, iter_res_log = f"FINAL_feedback_search_log_hub/{file_id}_{op_type}.log", \
		f"FINAL_feedback_search_log_hub/{file_id}_{op_type}.csv"

	with open(iter_res_log, 'a') as f:
		f.write("The task shapes of this dataset:\n")
		for tsp in task_shapes:
			f.write(f"{tsp}\n")

	# ansors = None
	# if op_type == 'dense':
	# 	ansors = get_best_kernels_for_ops_by_ansor(None)
	# else:
	# 	ansors = get_best_kernels_for_ops_by_ansor(tasks)
	ansors = None
	# if (len(tasks) >= 128) or ('F' in file_id):
	if True:
		ansors = dict()
		for task in tasks:
			ansors[task.workload_key] = (None, 1)
	else:
		try:
			ansors = get_best_kernels_for_ops_by_ansor(tasks)	
		except Exception as e:
			assert op_type == 'dense'
			ansors = get_best_kernels_for_ops_by_ansor(None)
			assert len(tasks) == len(ansors)
			for task in tasks:
				assert task.workload_key in ansors
		# 
		# ansors = get_best_kernels_for_ops_by_ansor(tasks)
		for k, v in ansors.items():
			print(k, v)

	with open(iter_res_log, 'a') as f:
		f.write(f"running time before search with feedback: {time.time()-overall_start_time} seconds\n")
	# return
	ret = search_mick_with_feadback(cost_model_params, Edges, tasks, all_micK_shapes, op_type, log_file, iter_res_log, file_id, Ssp_poses, overall_start_time)
	with open(iter_res_log, 'a') as f:
		for k, v in ret.items():
			f.write(f"{k};{v[:-1]};{v[-1]};{tasks[k].compute_dag.flop_ct/v[-1]/1e9};{ansors[tasks[k].workload_key][1]};{tasks[k].compute_dag.flop_ct/ansors[tasks[k].workload_key][1]/1e9}\n")

	return








def data_read_amount_PerBlk(op_type, mick_shape):
	'''
		This function directly uses mick_shape as the input, and the mick_shape may contain some parameter infor (e.g., stride of conv2d)
	'''
	if op_type=='conv2d':
		n, c, h, w, rc, rh, rw, stride, padding, dilation = mick_shape
		sh, sw = stride
		dh, dw = dilation
		ret = get_product((n, rc, sh*(h-1)+dh*(rh-1)+1, sw*(w-1)+dw*(rw-1)+1))+\
			get_product((c, rc, rh, rw))
		# ret = get_product((mick_Sshape[0], rc, mick_Sshape[2]+rh-1, mick_Sshape[3]+rw-1))+\
		# 	get_product((mick_Sshape[1], rc, rh, rw))
		return ret
	elif op_type == 'dense':
		mick_Sshape, rc = mick_shape[:2], mick_shape[2]
		ret = sum(mick_Sshape)*rc
		return ret
	elif op_type in ['bmm', 'bmm_nn']:
		mick_Sshape, rc = mick_shape[:3], mick_shape[3]
		ret = sum(mick_Sshape[1:])*mick_Sshape[0]*rc
		return ret






def cacheline_num_SingBlk(op_type, micKs):
	ret = dict()
	for micK in micKs:
		s_key = None
		tot_transNums = None
		loop = get_loops_ansor([micK])[0]
		op_para = get_op_para_ansor(micK, loop)
		if op_type == 'conv2d':
			s_key = get_output_shape_from_wlk(micK.workload_key, op_type)
			# Dshape, Kshape = get_inp_shapes(micK.workload_key, op_type=op_type)
			# s_key = (Dshape[0], Kshape[0], Dshape[2]-6, Dshape[3]-6, Kshape[1], Kshape[2], Kshape[3])
			tot_transNums = cal_32B_trans(s_key[:4], op_para, simplify = False, load_reduc_shape = None)
		ret[s_key] =  sum(tot_transNums)
	return ret


'''
CLNums = cacheline_num_SingBlk(op_type, interested)
'''


def get_symmetric_factor(v):
	symmetric_factor = None
	for i in range(1,v+1):
		if (v%i==0) and (v//i)<=i:
			symmetric_factor = i
			break
	return symmetric_factor









def update_mick_to_tune_infor(sp, op_type, micks, mick_shapes, tasks, tuner='ansor'):
	# sp is the micro-kernel shape
	if op_type == 'dense':
		mick = auto_scheduler.SearchTask(
			    func=dense_layer, args=((sp[0], sp[2]), (sp[1], sp[2])), target=tvm.target.Target("cuda")
			)
		mick_shape = sp
		# rep_layout = [get_symmetric_factor(540), 540//get_symmetric_factor(540), 8]
		# tsp = [rep_layout[i] *mick_shape[i] for i in range(3)]
		# task = auto_scheduler.SearchTask(
		# 	    func=dense_layer, args=((tsp[0], tsp[2]), (tsp[1], tsp[2])), target=tvm.target.Target("cuda")
		# 	)
		task = get_task_to_measure_for_mick(mick_shape, op_type)
		log_file = get_task_log_file_name(mick.workload_key, tuner = tuner,target = "cuda", kernel_type='micK1fetch',
											diff_measure_task_wlk=task.workload_key)
		# log_file = get_task_log_file_name(task.workload_key, tuner = "ansor",target = "cuda", kernel_type='micK1fetch')
		if os.path.exists(log_file):
			return
		micks.append(mick)
		mick_shapes.append(mick_shape)
		tasks.append(task)
	elif op_type == 'bmm':
		mick = auto_scheduler.SearchTask(
			    func=batch_matmul, args=((sp[0], sp[1], sp[3]), (sp[0], sp[2], sp[3])), target=tvm.target.Target("cuda")
			)
		mick_shape = sp
		task = get_task_to_measure_for_mick(mick_shape, op_type)
		log_file = get_task_log_file_name(mick.workload_key, tuner = tuner,target = "cuda", kernel_type='micK1fetch',
											diff_measure_task_wlk=task.workload_key)
		# log_file = get_task_log_file_name(task.workload_key, tuner = "ansor",target = "cuda", kernel_type='micK1fetch')
		if os.path.exists(log_file):
			return
		micks.append(mick)
		mick_shapes.append(mick_shape)
		tasks.append(task)
	elif op_type == 'bmm_nn':
		mick = auto_scheduler.SearchTask(
			    func=batch_matmul_noTrans, args=((sp[0], sp[1], sp[3]), (sp[0], sp[3], sp[2])), target=tvm.target.Target("cuda")
			)
		mick_shape = sp
		task = get_task_to_measure_for_mick(mick_shape, op_type)
		log_file = get_task_log_file_name(mick.workload_key, tuner = tuner,target = "cuda", kernel_type='micK1fetch',
											diff_measure_task_wlk=task.workload_key)
		# log_file = get_task_log_file_name(task.workload_key, tuner = "ansor",target = "cuda", kernel_type='micK1fetch')
		if os.path.exists(log_file):
			return
		micks.append(mick)
		mick_shapes.append(mick_shape)
		tasks.append(task)
	elif op_type == 'conv2d':
		# padding is an int, stride and dilation are two 2-tuples.
		n, c, h, w, rc, rh, rw, stride, padding, dilation = sp
		sh, sw = stride
		dh, dw = dilation
		mick = auto_scheduler.SearchTask(
			    func=conv2d_nchw, args=((n, rc, sh*(h-1)+dh*(rh-1)+1, sw*(w-1)+dw*(rw-1)+1), (c, rc, rh, rw), stride, padding, dilation, "float32"), 
				target=tvm.target.Target("cuda")
			)
		mick_shape = sp
		task = get_task_to_measure_for_mick(mick_shape, op_type)
		log_file = get_task_log_file_name(mick.workload_key, tuner = tuner, target = "cuda", kernel_type='micK1fetch',
											diff_measure_task_wlk=task.workload_key)
		# log_file = get_task_log_file_name(task.workload_key, tuner = "ansor",target = "cuda", kernel_type='micK1fetch')
		if os.path.exists(log_file):
			return
		micks.append(mick)
		mick_shapes.append(mick_shape)
		tasks.append(task)






def process_dietcode_csv():
	'''
		Convert the format of the dietcode csv result file.
	'''
	import csv
	dc_ret = dict()
	vendor_ret = dict()
	with open('temp_workspace.csv', 'r') as f:
		reader = csv.reader(f)
		head = next(reader)
		for line in reader:
			sp = [int(i) for i in line[1][1:-1].split(',')]
			# sp = [sp[0], sp[2], sp[1]]
			if line[0] == 'Vendor':
				vendor_ret[tuple(sp)] = float(line[2])
			elif line[0] == 'DietCode':
				dc_ret[tuple(sp)] = float(line[2])
	with open('converted_workspace.csv', 'w') as f:
		writer = csv.writer(f, delimiter=';')
		writer.writerow(['op_shape', 'DietCode', 'Vendor'])
		for k in vendor_ret:
			line = [k, dc_ret[k], vendor_ret[k]]
			writer.writerow(line)
		f.write(f'dc_cost_dict = {dc_ret}\n\n')
		f.write(f'vendor_cost_dict = {vendor_ret}\n\n')
	return dc_ret, vendor_ret








def get_tasks_in_DietCode(op_type, sp_expr = None, dyn_range=None):
	'''
	dyn_range: the range of the dynamic variable
	'''
	if dyn_range == None:
		dyn_range = list(range(5, 128, 19))+[128]
	# 
	tasks = None
	target = tvm.target.Target("cuda")
	if op_type == 'dense':
		if sp_expr == None:
			sp_expr = lambda t: (16*t, 2304, 768)
		tasks = list()
		for T in dyn_range: #range(1, 129): #[5, 24, 43, 62, 81, 100, 119, 128]:
			# X_shape = (16*T, 768)
			# Y_shape = (2304, 768)
			sp = sp_expr(T)
			X_shape, Y_shape = (sp[0], sp[2]), (sp[1], sp[2])
			task = auto_scheduler.SearchTask(
			    func=dense_layer, args=(X_shape, Y_shape), target=target
			)
			tasks.append(task)
			print(task.workload_key)
		# ansors = get_best_kernels_for_ops_by_ansor()
	elif op_type == 'bmm':
		if sp_expr == None:
			sp_expr = lambda t: (192, t, t, 64)
		tasks = list()
		# Ts = list(range(5, 128, 19))
		# Ts.append(128)
		for T in dyn_range: #range(1, 129): #[5, 24, 43, 62, 81, 100, 119, 128]:
			# X_shape = (192, T, 64)
			# Y_shape = (192, T, 64)
			sp = sp_expr(T)
			X_shape, Y_shape = (sp[0], sp[1], sp[3]), (sp[0], sp[2], sp[3])
			task = auto_scheduler.SearchTask(
			    func=batch_matmul, args=(X_shape, Y_shape), target=target
			)
			tasks.append(task)
			print(task.workload_key)
	elif op_type == 'bmm_nn':
		if sp_expr == None:
			sp_expr = lambda t: (192, t, 64, t)
		tasks = list()
		# Ts = list(range(5, 128, 19))
		# Ts.append(128)
		for T in dyn_range: #range(1, 129): #[5, 24, 43, 62, 81, 100, 119, 128]:
			# X_shape = (192, T, T)
			# Y_shape = (192, T, 64)
			sp = sp_expr(T)
			X_shape, Y_shape = (sp[0], sp[1], sp[3]), (sp[0], sp[3], sp[2])
			task = auto_scheduler.SearchTask(
			    func=batch_matmul_noTrans, args=(X_shape, Y_shape), target=target
			)
			tasks.append(task)
			print(task.workload_key)		
	elif op_type == 'conv2d':
		# tasks = list()
		# Ts = list(range(5, 128, 19))
		# Ts.append(128)
		# n, c, w, rc, rh, rw = 1, 32, 1080, 64, 7, 7
		# # we will need to change the dynamic shape to something like batch=16, and (h,w) are varying in range(1, 1024).
		# sh, sw = [1, 1]
		# dh, dw = [1, 1]
		# stride, padding, dilation = [1, 1], 0, [1, 1]
		# for T in Ts:
		# 	h = 16*T
		# 	task = auto_scheduler.SearchTask(
		# 		    func=conv2d_nchw, args=((n, rc, sh*(h-1)+dh*(rh-1)+1, sw*(w-1)+dw*(rw-1)+1), (c, rc, rh, rw), stride, padding, dilation, "float32"), 
		# 			target=tvm.target.Target("cuda")
		# 		)
		# 	tasks.append(task)
		# 	print(task.workload_key)
		# 
		# ----------------------------------------------------
		# first check the case when sp_expr!=None
		if sp_expr!=None:
			tasks = list()
			# Ts = list(range(5, 128, 19))
			# Ts.append(128)
			for T in dyn_range:
				sp = sp_expr(T)
				n, c, h, w, rc, rh, rw, stride, padding, dilation = sp
				sh, sw = stride
				dh, dw = dilation
				task = auto_scheduler.SearchTask(
					    func=conv2d_nchw, args=((n, rc, sh*(h-1)+dh*(rh-1)+1, sw*(w-1)+dw*(rw-1)+1), (c, rc, rh, rw), stride, padding, dilation, "float32"), 
						target=tvm.target.Target("cuda")
					)
				tasks.append(task)
				print(task.workload_key)
			return tasks
		# 
		# the op set below is too large to run successfully----------------
		tasks = list()
		Ts = list(range(5, 128, 19))
		Ts.append(128)
		n, c, rc, rh, rw = 16, 64, 64, 3, 3
		# we will need to change the dynamic shape to something like batch=16, and (h,w) are varying in range(1, 1024).
		sh, sw = [1, 1]
		dh, dw = [1, 1]
		stride, padding, dilation = [1, 1], 0, [1, 1]
		for T in Ts:
			h = 8*T
			w = h
			task = auto_scheduler.SearchTask(
				    func=conv2d_nchw, args=((n, rc, sh*(h-1)+dh*(rh-1)+1, sw*(w-1)+dw*(rw-1)+1), (c, rc, rh, rw), stride, padding, dilation, "float32"), 
					target=tvm.target.Target("cuda")
				)
			tasks.append(task)
			print(task.workload_key)
		# this op set can run successfuly-----------------------
		tasks = list()
		Ts = list(range(5, 128, 19))
		Ts.append(128)
		n, c, w, rc, rh, rw = 16, 64, 256, 64, 3, 3
		# we will need to change the dynamic shape to something like batch=16, and (h,w) are varying in range(1, 1024).
		sh, sw = [1, 1]
		dh, dw = [1, 1]
		stride, padding, dilation = [1, 1], 0, [1, 1]
		for T in Ts:
			h = 2*T
			task = auto_scheduler.SearchTask(
				    func=conv2d_nchw, args=((n, rc, sh*(h-1)+dh*(rh-1)+1, sw*(w-1)+dw*(rw-1)+1), (c, rc, rh, rw), stride, padding, dilation, "float32"), 
					target=tvm.target.Target("cuda")
				)
			tasks.append(task)
			print(task.workload_key)
		# the op set below has h and w both being dynamic----------------------
		tasks = list()
		Ts = list(range(5, 128, 19))
		Ts.append(128)
		n, c, rc, rh, rw = 16, 64, 64, 3, 3
		# we will need to change the dynamic shape to something like batch=16, and (h,w) are varying in range(1, 1024).
		sh, sw = [1, 1]
		dh, dw = [1, 1]
		stride, padding, dilation = [1, 1], 0, [1, 1]
		for T in Ts:
			h = 2*T
			w = h
			task = auto_scheduler.SearchTask(
				    func=conv2d_nchw, args=((n, rc, sh*(h-1)+dh*(rh-1)+1, sw*(w-1)+dw*(rw-1)+1), (c, rc, rh, rw), stride, padding, dilation, "float32"), 
					target=tvm.target.Target("cuda")
				)
			tasks.append(task)
			print(task.workload_key)
		# below we generate more tasks from resnet 50
		tasks = list()
		Ts = list(range(5, 128, 19))
		Ts.append(128)
		sh, sw = [1, 1]
		dh, dw = [1, 1]
		stride, padding, dilation = [1, 1], 0, [1, 1]
		# op 1: (56, 56, 64, 64, 1, 1, 0)
		# op 2: (14, 14, 256, 256, 3, 1, 1)
		# op 3: (7, 7, 512, 512, 3, 1, 1)
		# op 4: (56, 56, 64, 256, 1, 1, 0)
		# op 5: (56, 56, 256, 64, 1, 1, 0)
		# op 5: (56, 56, 256, 64, 3, 1, 0)
		# op 6: (28, 28, 128, 512, 1, 1, 0)
		# op 7: (28, 28, 512, 128, 1, 1, 0)
		# op 7: (28, 28, 512, 128, 3, 1, 0)
		# op 8: (14, 14, 256, 1024, 1, 1, 0)
		# op 9: (14, 14, 1024, 256, 1, 1, 0)
		# op 9: (14, 14, 1024, 256, 3, 1, 0)
		# op 10: (7, 7, 512, 2048, 1, 1, 0)
		# op 11: (7, 7, 2048, 512, 1, 1, 0)
		# op 11: (7, 7, 2048, 512, 3, 1, 0)
		sps = list()
		for T in Ts:
			sps.append((16, 64, 2*T, 2*T, 64, 1, 1))
			sps.append((16, 128, T, T, 128, 3, 3))
			sps.append((16, 256, 2*T, 2*T, 64, 1, 1))
			sps.append((16, 64, 2*T, 2*T, 256, 1, 1))
			sps.append((16, 64, 2*T, 2*T, 256, 3, 3))
			sps.append((16, 512, T, T, 128, 1, 1))
			sps.append((16, 128, T, T, 512, 1, 1))
			sps.append((16, 128, T, T, 512, 3, 3))
		for sp in sps:
			n, c, h, w, rc, rh, rw = sp
			task = auto_scheduler.SearchTask(
				    func=conv2d_nchw, args=((n, rc, sh*(h-1)+dh*(rh-1)+1, sw*(w-1)+dw*(rw-1)+1), (c, rc, rh, rw), stride, padding, dilation, "float32"), 
					target=tvm.target.Target("cuda")
				)
			tasks.append(task)
			print(task.workload_key)
	return tasks






def get_max_Ssp_Rsp_from_tasks(tasks, op_type):
	'''
		This function get the max Ssp and the max Rsp from the tasks.
	'''
	task_shapes = list()
	for count, task in enumerate(tasks):
		s_key = get_output_shape_from_wlk(task.workload_key, op_type)
		task_shapes.append(s_key)
	sp_len = len(task_shapes[0])
	Ssp_len = sp_len - 1
	if op_type == 'conv2d':
		sp_len = 7
		Ssp_len = 4
	max_Ssp = [max(sp[i] for sp in task_shapes) for i in range(Ssp_len)]
	max_Rsp = [max(sp[i] for sp in task_shapes) for i in range(Ssp_len, sp_len)]
	return max_Ssp, max_Rsp




def get_search_time_of_ansor(task, op_type):
	log_reader = None
	tuner = "ansor"
	targetstr = "cuda"
	kernel_type='op'
	log_file = get_task_log_file_name(task.workload_key, tuner = tuner, target = targetstr, kernel_type=kernel_type, diff_measure_task_wlk="")
	if not os.path.exists(log_file):
		log_file = get_task_log_file_name_old(task.workload_key, tuner = tuner, target = targetstr, kernel_type = "")
	# 
	log_reader = auto_scheduler.RecordReader(log_file)
	best_lineNO = -1
	start_time, end_time = None, None
	for lineNO, (inp, res) in enumerate(log_reader):
		if lineNO == 0:
			start_time = res.timestamp
		elif lineNO == 999:
			end_time = res.timestamp
		elif lineNO>=1000:
			assert False, "check this log_file, seems more than 1000 measurement trials."
	return end_time - start_time

	




def get_cost_model_params(op_type):
	func_micK_curvs = dict()
	popts = dict()
	curve_rep_layouts = dict()
	# base_blk_nums = [SMNum*i+1 for i in range(5)] + [(i+1)*SMNum for i in range(5)]
	interested_blkNs = {0:[48, 108], 1:[110, 216], 2:[220, 324], 3:[330, 432], 4:[440, 540]}
	base_blk_nums = list()
	for k in range(5):
		base_blk_nums = base_blk_nums + interested_blkNs[k]


	base_reduc_rep_nums = [2**i for i in range(7)]
	base_reduc_lens = [6, ] + list(range(12, 121, 12)) + [128, ] # [6, 12, 24, 36, 48, 60, 72, ]#18,]
	# base_reduc_lens = [6, ] + list(range(12, 73, 12)) # + [128, ] # [6, 12, 24, 36, 48, 60, 72, ]#18,]
	fix_lens = list(range(1, 22))+["inf"] #[1, 2, 3, 4, 5, 6, 7, 8, "inf"] # [1, 2, 3, 4, 5, 6, "inf"]
	# rep_keys = ['line', 'square']
	rep_keys = ['line_h', 'line_v', 'square']
	if op_type in ['bmm', 'bmm_nn']:
		rep_keys = ['line_b', 'square'] # one rep layout is only replicate along batch axis, the other one is square over h and w

	interested_Rsps = dict()
	if op_type == 'conv2d':
		# for conv2d, we draw curves for each reduction loop shape, instead of each reduction loop size
		base_red_shapes = list()
		for r_size in base_reduc_lens:
			rw = get_symmetric_factor(r_size)
			rh = r_size // rw
			base_red_shapes.append((1, rh, rw))
			# base_red_shapes.append((r_size, 1, 1))
			rw = min(factorint(r_size).keys())
			base_red_shapes.append((r_size//rw, 1, rw))
			interested_Rsps[r_size] = base_red_shapes[-2:] #[(1, rh, rw), (r_size, 1, 1)]
		base_reduc_lens = base_red_shapes
	else:
		for r_size in base_reduc_lens:
			interested_Rsps[r_size] = [(r_size, )]




	curve_poses = ['up', 'lw']
	for reduc_len in base_reduc_lens:
		for reduc_rep_num in base_reduc_rep_nums:
			for blk in base_blk_nums:
				for repK in rep_keys:
					for pos in curve_poses:
						for fixlen in fix_lens:
							key = (reduc_len, reduc_rep_num, blk, repK, pos, fixlen)
							if pos == 'up':
								func_micK_curvs[key] = func_micK_curv_ConvHull # func_micK_curv_LB #func_micK_curv_UB
							else:
								func_micK_curvs[key] = func_micK_curv_ConvHull # func_micK_curv_LB
							if repK == 'square':
								tmp = sorted(get_factors(blk), key=lambda repN: repN+blk//repN)[0]
								if tmp == 1:
									curve_rep_layouts[key] = (blk, 1)
								else:
									curve_rep_layouts[key] = (tmp, blk//tmp)
							elif repK == 'line_h':
								curve_rep_layouts[key] = (1, blk)
							elif repK == 'line_v':
								curve_rep_layouts[key] = (blk, 1)
							elif repK == 'line_b':
								curve_rep_layouts[key] = (blk, 1, 1)
							if op_type == 'conv2d':
								curve_rep_layouts[key] = curve_rep_layouts[key] + (1,1)
							if (op_type in ['bmm','bmm_nn']) and (repK == 'square'):
								curve_rep_layouts[key] = (1, ) + curve_rep_layouts[key]


	if op_type == 'dense':
		popts = cost_model_params_dense_convhull3.get_popts()
	elif op_type == 'bmm':
		popts = cost_model_params_bmm_convhull1.get_popts()
	elif op_type == 'bmm_nn':
		popts = cost_model_params_bmm_nn_convhull1.get_popts()
	elif op_type == 'conv2d':
		popts = cost_model_params_conv2d_convhull6.get_popts() 


	selected_fixedL_dict = None
	if op_type == 'dense':
		selected_fixedL_dict = {6: 2, 12: 4, 24: 8, 36: 8, 48: 8, 60: 4, 72: 6, 84: 2, 96: 4, 108: 8, 120: 4, 128: 16}
	elif op_type == 'bmm':
		selected_fixedL_dict = {108: 8, 128: 8, 6: 2, 12: 4, 24: 8, 36: 8, 48: 8, 60: 4, 72: 6, 84: 2, 96: 4, 120: 4}
	elif op_type == 'bmm_nn':
		selected_fixedL_dict = {6: 1, 12: 2, 24: 4, 36: 2, 48: 8, 60: 4, 72: 6, 84: 2, 96: 4, 108: 8, 120: 4, 128: 8}
	elif op_type == 'conv2d':
		selected_fixedL_dict = {(1, 2, 3): 1, (3, 1, 2): 1, (1, 3, 4): 1, (6, 1, 2): 1, (1, 4, 6): 2, (12, 1, 2): 2, (1, 6, 6): 2, (18, 1, 2): 2, (1, 6, 8): 4, (24, 1, 2): 4, (1, 6, 10): 4, (30, 1, 2): 4, (1, 8, 9): 4, (36, 1, 2): 4, (1, 7, 12): 2, (42, 1, 2): 2, (1, 8, 12): 8, (48, 1, 2): 8, (1, 9, 12): 2, (54, 1, 2): 2, (1, 10, 12): 8, (60, 1, 2): 8, (1, 8, 16): 8, (64, 1, 2): 16}

	return selected_fixedL_dict, func_micK_curvs, popts, curve_rep_layouts, interested_blkNs, interested_Rsps





def analyse_log_for_pred_error(tasks, file_id, op_type):
	'''
		This function analyse the log file when running the search with feedback.
		1. we get all the states we checked during search.
		2. we measure the states w.r.t each task on hardware.
		3. we store the real cost in a dictory and save it in a file.
	'''
	log_file = f"FINAL_feedback_search_log_hub/{file_id}_{op_type}.log"
	tmp = dict()
	my_load_all_best_input_from_file_multiTileOnes(log_file, tvm.target.Target("cuda"), tmp)
	# 
	log_file_pred_error = f"FINAL_feedback_search_log_hub/PredErrorAnalyse_ETO2.py"
	measure_ctx = auto_scheduler.LocalRPCMeasureContext(min_repeat_ms=300)
	tune_option = auto_scheduler.TuningOptions(
	    num_measure_trials=1000,  # change this to 1000 to achieve the best performance
	    runner=measure_ctx.runner,
	    measure_callbacks=[auto_scheduler.RecordToFile(log_file_pred_error)],
	    verbose=0,
	)
	# 
	tasks_to_measure = list()
	states_to_measure = list()
	ret_keys = list()
	pred_dict = dict()
	selected_fixedL_dict, func_micK_curvs, popts, curve_rep_layouts, interested_blkNs, interested_Rsps = get_cost_model_params(op_type)
	# 
	for state_key, (inp, cost) in tmp.items():
		for task in tasks:
			state_sp = get_output_shape_from_wlk(state_key, op_type)
			tsp = get_output_shape_from_wlk(task.workload_key, op_type)
			rep_layout = get_repl_to_measure_for_mick(op_type)
			mick_shape = [state_sp[i] // rep_layout[i] for i in range(len(rep_layout))] + list(state_sp[len(rep_layout):])
			# tasks_to_measure.append(task)
			repl = tuple([math.ceil(tsp[i]/mick_shape[i]) for i in range(len(rep_layout))])
			# psp = [repl[i] * msp[i] for i in range(sp_len)] + msp[sp_len:]
			padded_op = get_padded_op(mick_shape, repl=repl, tsp=None, op_type=op_type)
			tasks_to_measure.append(padded_op)
			# 
			states_to_measure.append(inp.state)
			ret_keys.append((tuple(mick_shape), tuple(tsp)))
			# 
			print((tuple(mick_shape), tuple(tsp)))
			cost = my_cost_model_full_dynamic_2Rngs(selected_fixedL_dict, func_micK_curvs, popts, curve_rep_layouts, interested_blkNs, 
				mick_shape, tsp, op_type, interested_Rsps)
			pred_dict[(tuple(mick_shape), tuple(tsp))] = cost
	# 
	with open(log_file_pred_error, "a") as f:
		f.write(f'def get_pred_cost_{file_id}_{op_type}():\n\treturn {pred_dict}\n\n\n')
	# 
	costs = measure_tasks_and_states(tune_option, tasks_to_measure, states_to_measure)
	ret = dict()
	for key, cost in zip(ret_keys, costs):
		ret[key] = cost
	with open(log_file_pred_error, "a") as f:
		f.write(f'def get_real_cost_{file_id}_{op_type}():\n\treturn {ret}\n\n\n')
	# 
	errors = dict()
	for k in ret:
		errors[k] = ret[k] / pred_dict[k]
	with open(log_file_pred_error, "a") as f:
		f.write(f'def get_pred_error_{file_id}_{op_type}():\n\treturn {errors}\n\n\n')









def eto_tune_selected_msps(flat_all_micK_shapes, op_type, file_id, cuda_i):
	mick_ids = [4306, 4331, 2287, 2322, 308, 309, 311, 313, 316, 317, 319, 2418, 2422, 2423, 2424, 2425, 2426, 2428, 2431, 2434, 2436, 597, 618, 623, 626, 628, 631, 634, 636, 637, 639, 640, 642, 771, 777, 785, 798, 802, 803, 804, 805, 806, 808, 809, 811, 812, 813, 814, 816, 819, 820, 821, 822, 835, 839, 840, 841, 842, 843, 845, 846, 848, 849, 850, 851, 853, 856, 857, 859, 888, 2937, 2938, 2939, 2940, 2941, 2942, 2944, 2945, 2947, 2950, 2960, 2962, 2963, 2964, 2965, 2966, 2967, 2968, 2969, 2970, 2971, 2973, 2974, 2976, 2977, 2979, 2981, 2984, 2996, 2999, 3000, 3001, 3002, 3003, 3004, 3006, 3007, 3009, 3012, 3061, 3065, 3066, 3067, 3069, 3074, 3085, 3086, 3088, 3089, 3090, 3091, 3092, 3093, 3094, 3095, 3096, 3097, 3099, 3102, 3105, 3126, 1103, 1124, 1125, 1126, 1127, 1129, 1130, 1132, 1134, 1135, 1137, 3188, 1141, 1140, 1143, 1146, 3216, 3217, 3219, 1172, 3224, 1177, 1178, 1180, 3307, 3334, 1309, 1317, 1487, 3419, 3420, 3446, 3484, 3485, 3486, 3487, 3488, 3489, 3490, 3491, 3492, 3493, 3494, 3495, 3496, 3497, 3498, 3499, 3500, 3501, 3502, 3503, 3504, 3505, 3506, 3507, 3508, 3509, 3510, 3511, 3512, 3513, 3514, 3515, 3516, 3517, 3518, 3519, 3520, 3521, 3522, 3523, 3524, 3525, 3526, 3527, 3528, 1481, 3529, 1483, 3530, 3531, 3532, 3533, 3534, 1489, 3535, 3536, 3537, 3538, 3539, 3540, 3541, 3542, 3543, 3544, 3545, 3546, 3547, 3548, 3549, 3550, 3551, 3552, 3553, 3554, 3555, 3556, 3557, 3558, 3559, 3560, 3561, 3562, 1518, 3563, 1520, 3564, 3565, 1523, 1524, 3566, 3567, 3568, 3569, 3577, 3578, 3570, 3580, 3572, 3574, 3575, 3576, 1526, 1555, 1557, 3606, 3608, 3609, 3610, 3611, 3612, 3613, 3614, 3615, 3616, 3617, 3618, 3619, 3620, 3621, 3622, 3623, 3624, 3625, 3626, 3627, 3628, 3629, 3630, 3631, 3571, 3636, 3637, 3573, 3663, 3664, 3735, 3738, 3739, 3740, 3741, 3742, 3743, 3748, 3767, 1563, 3836, 3840, 3841, 3842, 3844, 3863, 1817, 3867, 3868, 3869, 1819, 3871, 1822, 1825, 1823, 3876, 1946, 1948, 1951, 1960, 1961, 1965, 1966, 1967, 1968, 1969, 1971, 1974, 1977, 1979, 1985, 4033, 1997, 1998, 2002, 2003, 2004, 2005, 2006, 2008, 2011, 2014, 2016, 2022]
	# mick_ids = [139, 176, 263, 264, 265, 266, 268, 269, 270, 271, 272, 273, 274, 276, 277, 278, 279, 280, 282, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 313, 314, 315, 316, 317, 319, 320, 321, 322, 417, 438, 439, 440, 441, 442, 443, 444, 445, 446, 448, 449, 450, 451, 452, 453, 454, 456, 457, 459, 460, 462, 567, 581, 582, 583, 584, 585, 586, 587, 588, 589, 591, 592, 594, 595, 596, 597, 599, 600, 602, 603, 605, 606, 608, 615, 616, 617, 618, 619, 620, 621, 622, 623, 624, 625, 626, 627, 628, 629, 630, 631, 632, 633, 634, 635, 636, 637, 638, 639, 640, 641, 642, 643, 644, 645, 655, 656, 657, 658, 659, 660, 661, 662, 663, 665, 666, 668, 669, 670, 671, 673, 674, 676, 677, 679, 734, 735, 736, 737, 739, 747, 760, 761, 762, 763, 764, 765, 766, 767, 768, 769, 770, 771, 772, 773, 774, 775, 776, 777, 779, 780, 782, 783, 784, 785, 786, 787, 788, 794, 795, 796, 797, 798, 799, 800, 801, 802, 803, 804, 805, 806, 807, 808, 809, 810, 811, 812, 813, 814, 815, 816, 817, 818, 819, 820, 821, 822, 823, 824, 825, 832, 833, 834, 835, 836, 837, 838, 839, 840, 841, 842, 843, 844, 845, 846, 847, 848, 849, 850, 851, 852, 853, 854, 855, 856, 857, 858, 859, 860, 861, 862, 872, 873, 874, 875, 876, 877, 878, 879, 880, 881, 882, 883, 884, 885, 886, 887, 888, 890, 891, 894, 896, 899, 946, 947, 948, 949, 951, 954, 957, 959, 963, 965, 994, 1000, 1002, 1031, 1037, 1039, 1090, 1091, 1092, 1093, 1094, 1095, 1097, 1098, 1099, 1100, 1101, 1102, 1103, 1105, 1116, 1117, 1118, 1119, 1120, 1121, 1122, 1123, 1124, 1125, 1126, 1127, 1128, 1129, 1130, 1131, 1132, 1133, 1134, 1135, 1136, 1137, 1138, 1139, 1140, 1141, 1142, 1143, 1144, 1145, 1146, 1156, 1157, 1158, 1159, 1160, 1161, 1162, 1163, 1164, 1165, 1166, 1167, 1169, 1170, 1171, 1172, 1174, 1175, 1176, 1177, 1178, 1179, 1180, 1181, 1182, 1183, 1266, 1267, 1268, 1269, 1270, 1272, 1273, 1275, 1276, 1277, 1278, 1280, 1293, 1296, 1297, 1298, 1299, 1300, 1301, 1303, 1304, 1306, 1307, 1308, 1309, 1311, 1312, 1313, 1314, 1315, 1317, 1318, 1320, 1340, 1346, 1348, 1351, 1352, 1354, 1357, 1439, 1440, 1441, 1442, 1444, 1447, 1450, 1452, 1465, 1466, 1467, 1468, 1469, 1470, 1471, 1472, 1473, 1474, 1475, 1476, 1477, 1478, 1479, 1480, 1481, 1483, 1484, 1485, 1486, 1487, 1488, 1489, 1490, 1491, 1492, 1501, 1502, 1503, 1504, 1505, 1506, 1507, 1508, 1509, 1510, 1511, 1512, 1513, 1514, 1515, 1516, 1517, 1518, 1519, 1520, 1521, 1522, 1523, 1524, 1525, 1526, 1527, 1528, 1529, 1539, 1540, 1541, 1542, 1543, 1544, 1545, 1546, 1547, 1548, 1549, 1550, 1551, 1552, 1553, 1554, 1555, 1557, 1558, 1560, 1561, 1563, 1564, 1566, 1620, 1622, 1651, 1653, 1659, 1688, 1690, 1694, 1696, 1774, 1775, 1776, 1777, 1778, 1780, 1786, 1788, 1789, 1791, 1801, 1802, 1803, 1804, 1805, 1806, 1807, 1808, 1809, 1810, 1811, 1812, 1816, 1817, 1819, 1820, 1821, 1822, 1823, 1825, 1826, 1827, 1828, 1930, 1931, 1932, 1933, 1934, 1935, 1936, 1937, 1938, 1939, 1940, 1941, 1943, 1944, 1945, 1946, 1948, 1949, 1950, 1951, 1960, 1961, 1965, 1966, 1967, 1968, 1969, 1971, 1974, 1977, 1979, 1980, 1981, 1982, 1983, 1985, 1988, 1997, 1998, 2002, 2003, 2004, 2005, 2006, 2008, 2011, 2014, 2016, 2022, 2104, 2106, 2109, 2135, 2137, 2143, 2256, 2276, 2277, 2278, 2279, 2281, 2287, 2289, 2290, 2291, 2292, 2293, 2295, 2311, 2312, 2313, 2314, 2316, 2322, 2324, 2325, 2327, 2328, 2330, 2418, 2422, 2423, 2424, 2425, 2426, 2428, 2431, 2434, 2436, 2658, 2659, 2660, 2661, 2662, 2663, 2664, 2665, 2666, 2667, 2668, 2669, 2670, 2671, 2672, 2673, 2674, 2675, 2676, 2677, 2678, 2679, 2680, 2681, 2682, 2683, 2684, 2685, 2686, 2687, 2688, 2689, 2690, 2691, 2692, 2693, 2694, 2695, 2696, 2697, 2698, 2699, 2700, 2701, 2702, 2703, 2704, 2705, 2706, 2707, 2708, 2709, 2710, 2711, 2712, 2713, 2714, 2715, 2716, 2717, 2718, 2719, 2720, 2721, 2722, 2723, 2724, 2725, 2726, 2727, 2728, 2729, 2730, 2731, 2732, 2733, 2734, 2735, 2736, 2737, 2738, 2739, 2740, 2741, 2742, 2743, 2744, 2745, 2746, 2747, 2748, 2749, 2750, 2751, 2752, 2753, 2754, 2755, 2756, 2757, 2758, 2759, 2760, 2761, 2762, 2763, 2764, 2765, 2766, 2767, 2768, 2769, 2770, 2771, 2772, 2773, 2774, 2775, 2776, 2777, 2778, 2779, 2780, 2781, 2782, 2783, 2784, 2785, 2786, 2787, 2788, 2789, 2790, 2791, 2792, 2793, 2794, 2795, 2796, 2797, 2798, 2799, 2800, 2801, 2802, 2803, 2804, 2805, 2806, 2807, 2808, 2809, 2810, 2811, 2812, 2813, 2814, 2815, 2816, 2817, 2818, 2819, 2820, 2821, 2822, 2823, 2824, 2825, 2826, 2827, 2828, 2829, 2830, 2831, 2832, 2833, 2834, 2835, 2836, 2837, 2838, 2839, 2840, 2841, 2842, 2843, 2844, 2845, 2846, 2847, 2848, 2849, 2850, 2851, 2852, 2853, 2854, 2855, 2856, 2857, 2858, 2859, 2860, 2861, 2862, 2863, 2864, 2865, 2866, 2867, 2868, 2869, 2870, 2871, 2872, 2873, 2874, 2875, 2876, 2877, 2878, 2879, 2880, 2881, 2882, 2883, 2884, 2885, 2886, 2887, 2888, 2889, 2890, 2891, 2892, 2893, 2894, 2895, 2896, 2897, 2898, 2899, 2900, 2901, 2902, 2903, 2904, 2905, 2906, 2907, 2908, 2909, 2910, 2911, 2912, 2913, 2914, 2915, 2916, 2917, 2918, 2919, 2920, 2921, 2922, 2923, 2924, 2925, 2926, 2927, 2928, 2929, 2930, 2931, 2932, 2933, 2934, 2935, 2936, 2937, 2938, 2939, 2940, 2941, 2942, 2943, 2944, 2945, 2946, 2947, 2948, 2949, 2950, 2951, 2952, 2953, 2954, 2955, 2956, 2957, 2958, 2959, 2960, 2961, 2962, 2963, 2964, 2965, 2966, 2967, 2968, 2969, 2970, 2971, 2972, 2973, 2974, 2975, 2976, 2977, 2978, 2979, 2980, 2981, 2982, 2983, 2984, 2985, 2986, 2987, 2988, 2989, 2990, 2991, 2992, 2993, 2994, 2995, 2996, 2997, 2998, 2999, 3000, 3001, 3002, 3003, 3004, 3005, 3006, 3007, 3008, 3009, 3010, 3011, 3012, 3013, 3014, 3015, 3016, 3017, 3018, 3019, 3020, 3021, 3022, 3024, 3026, 3061, 3066, 3067, 3069, 3074, 3085, 3086, 3088, 3089, 3090, 3091, 3092, 3093, 3094, 3095, 3096, 3097, 3099, 3102, 3105, 3126, 3188, 3206, 3207, 3208, 3209, 3210, 3211, 3216, 3217, 3219, 3224, 3307, 3315, 3318, 3319, 3320, 3321, 3322, 3323, 3324, 3325, 3326, 3327, 3328, 3329, 3330, 3331, 3332, 3333, 3334, 3335, 3336, 3337, 3338, 3339, 3340, 3341, 3342, 3343, 3344, 3345, 3346, 3347, 3348, 3349, 3350, 3351, 3352, 3353, 3354, 3355, 3356, 3357, 3358, 3359, 3360, 3361, 3362, 3363, 3364, 3365, 3366, 3367, 3368, 3369, 3370, 3371, 3372, 3373, 3374, 3375, 3376, 3377, 3378, 3379, 3380, 3381, 3382, 3383, 3384, 3385, 3386, 3387, 3388, 3389, 3390, 3391, 3392, 3393, 3394, 3395, 3396, 3397, 3398, 3399, 3400, 3401, 3402, 3403, 3404, 3405, 3406, 3407, 3408, 3409, 3410, 3411, 3412, 3413, 3414, 3415, 3416, 3417, 3418, 3419, 3420, 3421, 3422, 3423, 3424, 3425, 3426, 3427, 3428, 3429, 3430, 3431, 3432, 3433, 3434, 3435, 3436, 3437, 3438, 3439, 3440, 3441, 3442, 3443, 3444, 3445, 3446, 3447, 3448, 3449, 3450, 3451, 3452, 3453, 3454, 3455, 3456, 3457, 3458, 3459, 3460, 3461, 3462, 3463, 3464, 3465, 3466, 3467, 3468, 3469, 3470, 3471, 3472, 3473, 3474, 3475, 3476, 3477, 3478, 3479, 3480, 3481, 3482, 3483, 3484, 3485, 3486, 3487, 3488, 3489, 3490, 3491, 3492, 3493, 3494, 3495, 3496, 3497, 3498, 3499, 3500, 3501, 3502, 3503, 3504, 3505, 3506, 3507, 3508, 3509, 3510, 3511, 3512, 3513, 3514, 3515, 3516, 3517, 3518, 3519, 3520, 3521, 3522, 3523, 3524, 3525, 3526, 3527, 3528, 3529, 3530, 3531, 3532, 3533, 3534, 3535, 3536, 3537, 3538, 3539, 3540, 3541, 3542, 3543, 3544, 3545, 3546, 3547, 3548, 3549, 3550, 3551, 3552, 3553, 3554, 3555, 3556, 3557, 3558, 3559, 3560, 3561, 3562, 3563, 3564, 3565, 3566, 3567, 3568, 3569, 3570, 3571, 3572, 3573, 3574, 3575, 3576, 3577, 3578, 3579, 3580, 3581, 3582, 3583, 3584, 3585, 3586, 3587, 3588, 3589, 3590, 3591, 3592, 3593, 3594, 3595, 3596, 3597, 3598, 3599, 3600, 3601, 3602, 3603, 3604, 3605, 3606, 3607, 3608, 3609, 3610, 3611, 3612, 3613, 3614, 3615, 3616, 3617, 3618, 3619, 3620, 3621, 3622, 3623, 3624, 3625, 3626, 3627, 3628, 3629, 3630, 3631, 3632, 3633, 3634, 3635, 3636, 3637, 3638, 3639, 3640, 3641, 3642, 3643, 3644, 3645, 3646, 3647, 3648, 3649, 3650, 3651, 3652, 3653, 3654, 3655, 3656, 3657, 3658, 3659, 3660, 3661, 3662, 3663, 3664, 3665, 3666, 3667, 3668, 3669, 3670, 3671, 3672, 3673, 3674, 3675, 3676, 3677, 3678, 3679, 3680, 3681, 3682, 3683, 3684, 3685, 3686, 3687, 3688, 3689, 3690, 3691, 3692, 3693, 3694, 3695, 3696, 3697, 3698, 3699, 3700, 3701, 3702, 3703, 3704, 3705, 3706, 3707, 3708, 3709, 3710, 3711, 3712, 3713, 3714, 3715, 3716, 3717, 3718, 3719, 3720, 3721, 3722, 3723, 3724, 3725, 3726, 3727, 3728, 3729, 3730, 3731, 3732, 3733, 3734, 3735, 3736, 3737, 3738, 3739, 3740, 3741, 3742, 3743, 3744, 3745, 3746, 3747, 3748, 3749, 3750, 3751, 3752, 3753, 3754, 3755, 3756, 3757, 3758, 3759, 3760, 3761, 3762, 3763, 3764, 3765, 3766, 3767, 3768, 3769, 3770, 3771, 3772, 3773, 3774, 3775, 3776, 3777, 3778, 3779, 3780, 3781, 3782, 3783, 3784, 3785, 3786, 3787, 3788, 3789, 3790, 3791, 3792, 3793, 3794, 3795, 3796, 3797, 3798, 3799, 3800, 3801, 3802, 3803, 3804, 3805, 3806, 3807, 3808, 3809, 3810, 3811, 3812, 3813, 3814, 3815, 3816, 3817, 3818, 3819, 3820, 3821, 3822, 3823, 3824, 3825, 3826, 3827, 3828, 3829, 3830, 3831, 3832, 3833, 3834, 3835, 3836, 3837, 3838, 3839, 3840, 3841, 3842, 3843, 3844, 3845, 3846, 3847, 3848, 3849, 3850, 3851, 3852, 3853, 3854, 3855, 3856, 3857, 3858, 3859, 3860, 3861, 3862, 3863, 3864, 3865, 3866, 3867, 3868, 3869, 3870, 3871, 3872, 3873, 3874, 3875, 3876, 3877, 3878, 3879, 3880, 3881, 3882, 3883, 3884, 3885, 3886, 3887, 3888, 3889, 3890, 3891, 3892, 3893, 3894, 3895, 3896, 3897, 3898, 3899, 3900, 3901, 3902, 3903, 3904, 3905, 3906, 3907, 3908, 3909, 3910, 3911, 3912, 3913, 3914, 3915, 3916, 3917, 3918, 3919, 3920, 3921, 3922, 3923, 3924, 3925, 3926, 3927, 3928, 3929, 3930, 3931, 3932, 3933, 3934, 3935, 3936, 3937, 3938, 3939, 3940, 3941, 3942, 3943, 3944, 3945, 3946, 3947, 3948, 3949, 3950, 3951, 3952, 3953, 3954, 3955, 3956, 3957, 3958, 3959, 3960, 3961, 3962, 3963, 3964, 3965, 3966, 3967, 3968, 3969, 3970, 3971, 3972, 3973, 3974, 3975, 3976, 3977, 3978, 3979, 3980, 3981, 3982, 3983, 3984, 3985, 3986, 3987, 3988, 3989, 3990, 3991, 3992, 3993, 3994, 3995, 3996, 3997, 3998, 3999, 4000, 4001, 4002, 4003, 4004, 4005, 4006, 4007, 4008, 4009, 4010, 4011, 4012, 4013, 4014, 4015, 4016, 4017, 4018, 4019, 4020, 4021, 4022, 4023, 4024, 4025, 4026, 4027, 4028, 4029, 4030, 4031, 4032, 4033, 4034, 4035, 4036, 4037, 4038, 4039, 4040, 4041, 4042, 4043, 4044, 4045, 4046, 4047, 4048, 4049, 4050, 4051, 4052, 4053, 4054, 4055, 4056, 4057, 4058, 4059, 4060, 4061, 4062, 4063, 4064, 4065, 4066, 4067, 4068, 4069, 4070, 4071, 4072, 4073, 4074, 4075, 4076, 4077, 4306, 4331]
	# 
	log_file = f"FINAL_check_accuracy/{file_id}AllCand1k{cuda_i}_{op_type}.log"
	# 
	measure_ctx = auto_scheduler.LocalRPCMeasureContext(min_repeat_ms=300)
	tune_option = auto_scheduler.TuningOptions(
	    num_measure_trials=1000,  # change this to 1000 to achieve the best performance
	    runner=measure_ctx.runner,
	    measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
	    verbose=0,
	)
	# 
	measurer = auto_scheduler.measure.ProgramMeasurer(
				tune_option.builder, tune_option.runner,
				tune_option.measure_callbacks, tune_option.verbose
				)
	# 
	# flat_all_micK_shapes = all_micK_shapes
	# num_measures_per_round = tune_option.num_measures_per_round # // 2
	mickNum = len(flat_all_micK_shapes)
	mick_cts = [0 for _ in range(mickNum)]
	# mick_best_cts[i] saves the round task i found the best latency
	mick_best_cts = [0 for _ in range(mickNum)]
	# task_costs_history[i] saves the latency history of task i
	mick_costs_history = [[] for _ in range(mickNum)]
	# best_costs[i] saves the best latency of task i
	best_costs = 1e10 * np.ones(mickNum)
	best_states = [None for _ in range(mickNum)]
	dead_micks = set()
	mick_ids_to_tune = mick_ids[math.ceil(len(mick_ids)/3)*(cuda_i-1): math.ceil(len(mick_ids)/3)*(cuda_i+0)]
	for mick_i in mick_ids_to_tune:
		# tune this mick
		eto_tune([mick_i], flat_all_micK_shapes, op_type, tune_option, 
			mick_cts, best_costs, best_states, mick_best_cts, dead_micks, mick_costs_history, log_file)
	# 
	res = dict()
	tsps = [[0,0,0] for i in mick_ids_to_tune]
	for i, mick_i in enumerate(mick_ids_to_tune):
		res[i] = (-1, mick_i)
	# 
	write_pysched_to_file(res, tsps, flat_all_micK_shapes, best_states, op_type, f"{file_id}AllCand1k{cuda_i}")	







dense_layer = dense_layer
batch_matmul = batch_matmul
batch_matmul_noTrans = batch_matmul_noTrans
conv2d_nchw = conv2d_nchw


'''
from solution0 import *
nohup python3 solution0.py --cuda 0 > solution0_nohupout0.log 2>&1 &
nohup python3 solution0.py --cuda 1 > solution0_nohupout1.log 2>&1 &
nohup python3 solution0.py --cuda 2 > solution0_nohupout2.log 2>&1 &
nohup python3 solution0.py --cuda 3 > solution0_nohupout3.log 2>&1 &
'''

import cost_model_params_dense_convhull3, cost_model_params_bmm_convhull1, cost_model_params_bmm_nn_convhull1, cost_model_params_conv2d_convhull6


import Ssp_black_dict_pyfile

'''
cost_model_params_dense4: two ranges of lower curves
cost_model_params_dense5: two ranges of lower curves + full coverage of micK reduction l
cost_model_params_conv2d_convhull1: the cost model based on the ETO backend
'''


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--cuda", type=int)

	args = parser.parse_args()
	os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda)
	os.environ['OPENBLAS_NUM_THREADS'] = '1'
	# BELOW we first generate the Ssp black dict python file
	# get_Ssp_black_dict('dense')
	# get_Ssp_black_dict('bmm')
	# get_Ssp_black_dict('bmm_nn')
	# get_Ssp_black_dict('conv2d')

	# =====================================================================================
	# We try to prove that predicted_cost+ weight is close to the real cost
	run_setting = [(None, 'RAccuracy_')] 

	for dyn_rng, fhead in run_setting:
		# if fhead not in ['RF08272B_', 'RF08274M_', 'RF08274B_']:
		# 	continue
		op_type = 'dense'
		other_params = []
		sp_exprs = [None]
		file_id = 1
		for sp_expr in sp_exprs:
			start_time = time.time()
			tasks = get_tasks_in_DietCode(op_type, sp_expr = sp_expr, dyn_range=dyn_rng)
			max_out_shape, max_reduc_shape = get_max_Ssp_Rsp_from_tasks(tasks, op_type)
			looplen_ratio = 1
			all_micK_ops, all_micK_shapes, Ssp_poses = get_candidate_micro_ops_N_shapes(max_out_shape, max_reduc_shape, op_type, looplen_ratio, other_params)
			# select_best_micK_shapes_full_dynamic_2Rngs(tasks, all_micK_shapes, len(tasks), op_type, f'{fhead}{file_id}', Ssp_poses, start_time)
			# analyse_log_for_pred_error(tasks, f'R3_{file_id}', op_type)
			# collect_data_to_analyse_pred_error_model(Edges_vers, tasks, all_micK_shapes, op_type, f'{fhead}{file_id}')
			eto_tune_selected_msps(all_micK_shapes, op_type, f'{fhead}{file_id}', args.cuda)
			end_time = time.time()
			print(f"\nTotal running time on this dataset: {end_time - start_time}\n")
			file_id+=1
			print(max([v[1]-v[0] for v in Ssp_poses.values()]))


	# BELOW we automatically running the search with feedback algorithm
	# =====================================================================================
	run_setting = [(None, 'R1'), (list(range(1, 129)), 'RF1_')]

	for dyn_rng, fhead in run_setting:
		op_type = 'dense'
		other_params = []
		sp_exprs = [None]
		file_id = 1
		for sp_expr in sp_exprs:
			start_time = time.time()
			tasks = get_tasks_in_DietCode(op_type, sp_expr = sp_expr, dyn_range=dyn_rng)
			max_out_shape, max_reduc_shape = get_max_Ssp_Rsp_from_tasks(tasks, op_type)
			looplen_ratio = 1
			all_micK_ops, all_micK_shapes, Ssp_poses = get_candidate_micro_ops_N_shapes(max_out_shape, max_reduc_shape, op_type, looplen_ratio, other_params)
			select_best_micK_shapes_full_dynamic_2Rngs(tasks, all_micK_shapes, len(tasks), op_type, f'{fhead}{file_id}', Ssp_poses, start_time)
			# analyse_log_for_pred_error(tasks, f'R3_{file_id}', op_type)
			end_time = time.time()
			print(f"\nTotal running time on this dataset: {end_time - start_time}\n")
			file_id+=1
			print(max([v[1]-v[0] for v in Ssp_poses.values()]))


		op_type = 'bmm'
		other_params = []
		sp_exprs = [None]
		file_id = 1
		for sp_expr in sp_exprs:
			start_time = time.time()
			tasks = get_tasks_in_DietCode(op_type, sp_expr = sp_expr, dyn_range=dyn_rng)
			max_out_shape, max_reduc_shape = get_max_Ssp_Rsp_from_tasks(tasks, op_type)
			looplen_ratio = 1
			all_micK_ops, all_micK_shapes, Ssp_poses = get_candidate_micro_ops_N_shapes(max_out_shape, max_reduc_shape, op_type, looplen_ratio, other_params)
			select_best_micK_shapes_full_dynamic_2Rngs(tasks, all_micK_shapes, len(tasks), op_type, f'{fhead}{file_id}', Ssp_poses, start_time)
			# analyse_log_for_pred_error(tasks, f'R3_{file_id}', op_type)
			end_time = time.time()
			print(f"\nTotal running time on this dataset: {end_time - start_time}\n")
			file_id+=1
			print(max([v[1]-v[0] for v in Ssp_poses.values()]))


	for dyn_rng, fhead in run_setting:
		op_type = 'bmm_nn'
		other_params = []
		sp_exprs = [None]
		file_id = 1
		for sp_expr in sp_exprs:
			start_time = time.time()
			tasks = get_tasks_in_DietCode(op_type, sp_expr = sp_expr, dyn_range=dyn_rng)
			max_out_shape, max_reduc_shape = get_max_Ssp_Rsp_from_tasks(tasks, op_type)
			looplen_ratio = 1
			all_micK_ops, all_micK_shapes, Ssp_poses = get_candidate_micro_ops_N_shapes(max_out_shape, max_reduc_shape, op_type, looplen_ratio, other_params)
			select_best_micK_shapes_full_dynamic_2Rngs(tasks, all_micK_shapes, len(tasks), op_type, f'{fhead}{file_id}', Ssp_poses, start_time)
			# analyse_log_for_pred_error(tasks, f'R3_{file_id}', op_type)
			end_time = time.time()
			print(f"\nTotal running time on this dataset: {end_time - start_time}\n")
			file_id+=1
			print(max([v[1]-v[0] for v in Ssp_poses.values()]))

	# below is to tune BERT automatically------------------------------
	for dyn_rng, fhead in run_setting:
		start_time = time.time()
		other_params = []
		file_id = f"{fhead}BERTbase1"
		tasks = list()
		op_type = 'dense'
		sp_exprs = [
			lambda t: [16*t, 768, 768],
			lambda t: [16*t, 768, 3072],
			lambda t: [16*t, 3072, 768],
			]


		for sp_expr in sp_exprs:
			tasks = tasks + get_tasks_in_DietCode(op_type, sp_expr, dyn_range=dyn_rng)


		max_out_shape, max_reduc_shape = get_max_Ssp_Rsp_from_tasks(tasks, op_type)
		looplen_ratio = 1
		all_micK_ops, all_micK_shapes, Ssp_poses = get_candidate_micro_ops_N_shapes(max_out_shape, max_reduc_shape, op_type, looplen_ratio, other_params)
		select_best_micK_shapes_full_dynamic_2Rngs(tasks, all_micK_shapes, len(tasks), op_type, file_id, Ssp_poses, start_time)
		# analyse_log_for_pred_error(tasks, file_id, op_type)
		# 
		start_time2 = time.time()
		tasks = list()
		op_type = 'bmm'
		sp_exprs = [
			lambda t: [16*12, t, t, 768//12],
			lambda t: [16*12, t, 768//12, t],
			]


		for sp_expr in sp_exprs:
			tasks = tasks + get_tasks_in_DietCode(op_type, sp_expr, dyn_range=dyn_rng)


		max_out_shape, max_reduc_shape = get_max_Ssp_Rsp_from_tasks(tasks, op_type)
		looplen_ratio = 1
		all_micK_ops, all_micK_shapes, Ssp_poses = get_candidate_micro_ops_N_shapes(max_out_shape, max_reduc_shape, op_type, looplen_ratio, other_params)
		select_best_micK_shapes_full_dynamic_2Rngs(tasks, all_micK_shapes, len(tasks), op_type, file_id, Ssp_poses, start_time2)
		# analyse_log_for_pred_error(tasks, file_id, op_type)
		end_time = time.time()
		print(f"\nTotal running time on this dataset: {end_time - start_time}\n")
	#-------------------------------------------------------------
	# =====================================================================================
	for dyn_rng, fhead in run_setting:
		op_type = 'conv2d'
		other_params = [(1, 1), 0, (1, 1)]
		sp_exprs = [
			lambda t: [16, 128, t, t, 128, 3, 3]+other_params,
			lambda t: [16, 256, 2*t, 2*t, 64, 1, 1]+other_params,
		]
		file_id = 1
		for sp_expr in sp_exprs:
			start_time = time.time()
			tasks = get_tasks_in_DietCode(op_type, sp_expr = sp_expr, dyn_range=dyn_rng)
			max_out_shape, max_reduc_shape = get_max_Ssp_Rsp_from_tasks(tasks, op_type)
			looplen_ratio = 1
			all_micK_ops, all_micK_shapes, Ssp_poses = get_candidate_micro_ops_N_shapes(max_out_shape, max_reduc_shape, op_type, looplen_ratio, other_params)
			select_best_micK_shapes_full_dynamic_2Rngs(tasks, all_micK_shapes, len(tasks), op_type, f'{fhead}{file_id}', Ssp_poses, start_time)
			# analyse_log_for_pred_error(tasks, f'R3_{file_id}', op_type)
			end_time = time.time()
			print(f"\nTotal running time on this dataset: {end_time - start_time}\n")
			file_id+=1
			print(max([v[1]-v[0] for v in Ssp_poses.values()]))


		op_type = 'conv2d'
		other_params = [(2, 2), 0, (1, 1)]
		sp_exprs = [lambda t: [16, 64, 4*t, 4*t, 3, 7, 7]+[(2, 2), 0, (1, 1)]]
		for sp_expr in sp_exprs:
			start_time = time.time()
			tasks = get_tasks_in_DietCode(op_type, sp_expr = sp_expr, dyn_range=dyn_rng)
			max_out_shape, max_reduc_shape = get_max_Ssp_Rsp_from_tasks(tasks, op_type)
			looplen_ratio = 1
			all_micK_ops, all_micK_shapes, Ssp_poses = get_candidate_micro_ops_N_shapes(max_out_shape, max_reduc_shape, op_type, looplen_ratio, other_params)
			select_best_micK_shapes_full_dynamic_2Rngs(tasks, all_micK_shapes, len(tasks), op_type, f'{fhead}{file_id}', Ssp_poses, start_time)
			# analyse_log_for_pred_error(tasks, f'R3_{file_id}', op_type)
			end_time = time.time()
			print(f"\nTotal running time on this dataset: {end_time - start_time}\n")
			file_id+=1
			print(max([v[1]-v[0] for v in Ssp_poses.values()]))


