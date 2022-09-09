from solution0 import *
from scipy.spatial import ConvexHull



################################################################################################
# DENSE
def get_data_for_dense():
	op_type = 'dense'
	tuner = 'ansor' # 'eto'
	dataReads = [(49152 / 4)]
	red_lens = [6] + list(range(12, 128, 12)) + [128]
	# red_lens = [108]
	tasks = list() # only used for measuring micks when micks are being tuned
	mick_shapes = list()
	micks = list()
	# get min data read amount shapes
	interested0 = list()
	interested1 = list()
	for k in range(32, 26043, 32):
		# we only consider space sizes with factors 2, 3, 5 now
		if max(factorint(k).keys()) > 5:
			continue
		n = get_symmetric_factor(k)
		m = k // n
		# if (len(get_factors(n)) > 2) and (len(get_factors(m)) > 2):
		if n == m:
			interested0.append((n, m))
		# if max(factorint(k).keys()) > 5:
		# 	continue
		else:
			interested1.append((n, m))
		if n==m:
			for i, f in enumerate(get_factors(k)):
				if get_factors(k)[i+1] == n:
					interested1.append((k//f, f))
					break
	selected_fixedL_dict = dict()
	for red_len in red_lens:
		# get min data read amount tasks
		for interested_Ssps in [interested0, interested1]:
			sps = list()
			for sp in interested_Ssps: #interested0:
				if data_read_amount_PerBlk(op_type, sp+(red_len,)) > (49152 / 4):
					continue
				sps.append(sp)
			sp_num = len(sps)
			print("sp_num: ", sp_num, sps, flush=True)
			step_size = max(10, sp_num)//10
			step_size2 = math.ceil(sp_num/10)
			if abs(10-len(sps[::step_size]))> abs(10-len(sps[::step_size2])):
				step_size = step_size2
			print(f"red_len: {red_len}; selected sps num: {len(sps[::step_size])}")
			for sp in sps[::step_size]:
			# for sp in sps[::max(40, sp_num)//40]:
				mick_shape = list(sp) + [red_len]
				update_mick_to_tune_infor(mick_shape, op_type, micks, mick_shapes, tasks, tuner=tuner)
		# get shapes with fixed loop len 1
		for fixedL in range(1, 21):
			# we only consider mick space sizes with factors 2, 3, 5 now.
			factor_limit = 5
			if red_len == 108:
				factor_limit = 13
			if (fixedL > 1) and (max(factorint(fixedL).keys()) > factor_limit):
				continue
			sps = list()
			for sp in [(fixedL, k//fixedL) for k in range(32, 26043, 32) if k%fixedL == 0]:
				# we only consider mick space sizes with factors 2, 3, 5 now.
				if max(factorint(get_product(sp)).keys()) > factor_limit:
					continue
				if data_read_amount_PerBlk(op_type, sp+(red_len,)) > (49152 / 4):
					continue
				# if (fixedL == 1) or (data_read_amount_PerBlk(op_type, [sp[0]-1, get_product(sp)/(sp[0]-1)], rc = red_len) > (49152 / 4)):
				# 	sps.append(sp)
				sps.append(sp)
			if len(sps) == 0:
				continue
			sorted_shapes = sorted(sps, key=lambda sp: data_read_amount_PerBlk(op_type, sp+(red_len,)) )
			if data_read_amount_PerBlk(op_type, sorted_shapes[-1]+(red_len,)) >= (dataReads[0]-400):
				selected_fixedL_dict[red_len] = fixedL
				sp_num = len(sps)
				print(f"red_len:{red_len}, selected_fixedL:{fixedL}", flush=True)
				print("sp_num: ", sp_num, sps[-1], flush=True)
				print(f"sps: {sps}", flush=True)
				step_size = max(10, sp_num)//10
				step_size2 = math.ceil(sp_num/10)
				if abs(10-len(sps[::step_size]))> abs(10-len(sps[::step_size2])):
					step_size = step_size2
				print(f"red_len: {red_len}; selected sps num: {len(sps[::step_size])}")
				for sp in sps[::step_size]:
				# for sp in sps[::max(20, sp_num)//20]:
					mick_shape = list(sp) + [red_len]
					update_mick_to_tune_infor(mick_shape, op_type, micks, mick_shapes, tasks, tuner=tuner)
				break
		# get max data read amount tasks
		# if red_len != 72:
		# 	continue
		for dataRead in dataReads:
			factor_limit = 5
			if red_len == 108:
				factor_limit = 13
			shapes = list()
			for i in range(1, math.ceil(dataRead//red_len/2)):
				n = i
				for m in range(int((dataRead-400)//red_len) - i, \
					int(dataRead//red_len) - i + 1):
					# n, m  = i, int(dataRead//red_len) - i
					# if (n > 10 and len(get_factors(n)) <= 2) or (m > 10 and len(get_factors(m)) <= 2):
					# 	continue
					if (n*m<26043) and (n<=m) and ((n*m)%32==0) and (max(factorint(n*m).keys()) <= factor_limit):
						shapes.append((n, m))
			print(len(shapes))#, [get_product(sp) for sp in shapes], shapes)
			sorted_shapes = sorted(shapes, key=lambda sp: data_read_amount_PerBlk(op_type, sp+(red_len,)) )
			assert data_read_amount_PerBlk(op_type, sorted_shapes[-1]+(red_len,)) <= (49152 / 4)
			sorted_shapes = sorted(shapes, key=lambda sp: get_product(sp[:2]) )
			max_idx = None
			for idx, sp in enumerate(sorted_shapes):
				if get_product(sp[:2]) > min(10000, get_product(sorted_shapes[-1][:2]) / 2):
					max_idx = idx
					break
			# 
			for sps in [sorted_shapes[:max_idx], sorted_shapes[max_idx:]]:
				sp_num = len(sps)
				print("sp_num of range 2 lw curve: ", sp_num, sps[-1], flush=True)
				print(f"sps: {sps}", flush=True)
				print(f"sp Ssizes: {[get_product(sp) for sp in sps]}", flush=True)
				step_size = max(10, sp_num)//10
				step_size2 = math.ceil(sp_num/10)
				if abs(10-len(sps[::step_size]))> abs(10-len(sps[::step_size2])):
					step_size = step_size2
				print(f"red_len: {red_len}; selected sps num: {len(sps[::step_size])}")
				for sp in sps[::step_size]:
				# for sp in sps[::max(20, sp_num)//20]:
					mick_shape = list(sp) + [red_len]
					update_mick_to_tune_infor(mick_shape, op_type, micks, mick_shapes, tasks, tuner=tuner)			
	tot_op_num = len(tasks)
	print(tot_op_num, flush = True)
	ansors = dict()
	measure_ctx = auto_scheduler.LocalRPCMeasureContext(min_repeat_ms=300)
	tune_option = auto_scheduler.TuningOptions(
	    num_measure_trials=1000,  # change this to 1000 to achieve the best performance
	    runner=measure_ctx.runner,
	    builder=auto_scheduler.LocalBuilder(n_parallel=60),
	    # measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
	    verbose=2,
	)
	# 
	for count, task in enumerate(tasks):
		micK = micks[count]
		mick_shape = mick_shapes[count]
		# if count % 1 != 0:
		# 	continue
		# if ((count+1) <= math.ceil(tot_op_num / 4) * (0+args.cuda)) \
		# 	or ((count+1) > math.ceil(tot_op_num / 4) * (1+args.cuda)):
		# 	continue
		print(micK.workload_key, flush=True)
		# if micK.workload_key in ['["dense_layer", [2462, 4], [8, 4]]', '["dense_layer", [2762, 4], [8, 4]]']:
		# 	continue
		# tune_ops_ansor(tasks = [micK], tune_mick = True, limit_fetch_num = True)
		tune_ops_ansor([mick_shape], [micK], tasks = [task], tune_mick = True, limit_fetch_num = True)
		# 
		# mick_cts, best_costs, best_states, mick_best_cts, dead_micks, mick_costs_history = [0], [1e10], [None], [0], set(), [[]]
		# log_file = get_task_log_file_name(micK.workload_key, 
		# 	tuner = tuner,target = "cuda", kernel_type='micK1fetch', diff_measure_task_wlk=task.workload_key)
		# eto_tune([0], [mick_shape], op_type, tune_option, 
		# 	mick_cts, best_costs, best_states, mick_best_cts, dead_micks, mick_costs_history, log_file)
		# 
		tmp_ansors = measure_micK_curve([[micK]], [[task.workload_key]], only_multi_tile = True, kernel_type = "micK1fetch", op_type = op_type, tuner=tuner)
		for k, v in tmp_ansors.items():
			ansors[k] = v
		# 
		# tmp_ansors = measure_micK_curve([[micK]], [[task.workload_key]], only_multi_tile = True, kernel_type = "micK1fetch", op_type = op_type)
		# for k, v in tmp_ansors.items():
		# 	ansors[k] = v
	return ansors
################################################################################################
# DENSE collect data for prediction error
def get_pred_errors_for_dense():
	op_type = 'dense'
	tasks, mick_shapes, micks = list(), list(), list() # tasks only used for measuring micks when micks are being tuned
	Ssps = [[80, 64],[48, 48]]
	red_len = 24 # 40 # 24
	# get min data read amount shapes
	interested0 = list()
	tuner = 'ansor' # 'eto'
	avg_errors_keys={'Ssize':dict(), 'Rsp':dict()}
	for k in range(32, 26043, 32):
		# we only consider space sizes with factors 2, 3, 5 now
		if max(factorint(k).keys()) > 5:
			continue
		n = get_symmetric_factor(k)
		m = k // n
		# if (len(get_factors(n)) > 2) and (len(get_factors(m)) > 2):
		interested0.append((n, m))
	# different Ssps with a good fixed red_len
	for sp in interested0:
		mick_shape = list(sp) + [red_len]
		if (data_read_amount_PerBlk(op_type, mick_shape) > (49152 / 4)):
			continue
		update_mick_to_tune_infor(mick_shape, op_type, micks, mick_shapes, tasks, tuner=tuner)
		avg_errors_keys['Ssize'][get_product(sp)] = tuple(mick_shape)
	# different red_len with a good Ssp
	for k in range(1, 129):
		if (k > 1) and (max(factorint(k).keys()) > 5):
			continue
		for Ssp in Ssps:	
			# if (k, ) in avg_errors['Rsp']:
			# 	continue
			mick_shape = list(Ssp) + [k]
			if (data_read_amount_PerBlk(op_type, mick_shape) > (49152 / 4)):
				continue
			# mick_shape = list(Ssp) + [k]
			update_mick_to_tune_infor(mick_shape, op_type, micks, mick_shapes, tasks, tuner=tuner)
			avg_errors_keys['Rsp'][(k,)] = tuple(mick_shape)
			break
	tot_op_num = len(tasks)
	print(tot_op_num, flush = True)
	ansors = dict()
	preds = dict()
	measure_ctx = auto_scheduler.LocalRPCMeasureContext(min_repeat_ms=300)
	tune_option = auto_scheduler.TuningOptions(
	    num_measure_trials=1000,  # change this to 1000 to achieve the best performance
	    runner=measure_ctx.runner,
	    builder=auto_scheduler.LocalBuilder(n_parallel=60),
	    # measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
	    verbose=2,
	)
	selected_fixedL_dict, func_micK_curvs, popts, curve_rep_layouts, interested_blkNs, interested_Rsps = get_cost_model_params(op_type)
	for count, task in enumerate(tasks):
		micK = micks[count]
		mick_shape = mick_shapes[count]
		# if count % 1 != 0:
		# 	continue
		# if ((count+1) <= math.ceil(tot_op_num / 4) * (0+args.cuda)) \
		# 	or ((count+1) > math.ceil(tot_op_num / 4) * (1+args.cuda)):
		# 	continue
		print(micK.workload_key, flush=True)
		# if micK.workload_key in ['["dense_layer", [2462, 4], [8, 4]]', '["dense_layer", [2762, 4], [8, 4]]']:
		# 	continue
		# 
		tune_ops_ansor([mick_shape], [micK], tasks = [task], tune_mick = True, limit_fetch_num = True)
		# tmp_ansors = measure_micK_curve([[micK]], [[task.workload_key]], only_multi_tile = True, kernel_type = "micK1fetch", op_type = op_type)
		# for k, v in tmp_ansors.items():
		# 	ansors[k] = v	
		# 
		# mick_cts, best_costs, best_states, mick_best_cts, dead_micks, mick_costs_history = [0], [1e10], [None], [0], set(), [[]]
		# log_file = get_task_log_file_name(micK.workload_key, 
		# 	tuner = tuner,target = "cuda", kernel_type='micK1fetch', diff_measure_task_wlk=task.workload_key)
		# eto_tune([0], [mick_shape], op_type, tune_option, 
		# 	mick_cts, best_costs, best_states, mick_best_cts, dead_micks, mick_costs_history, log_file)
		# 
		# tmp_ansors = measure_micK_curve([[micK]], [[task.workload_key]], only_multi_tile = True, kernel_type = "micK1fetch", op_type = op_type, tuner=tuner)
		# for k, v in tmp_ansors.items():
		# 	ansors[k] = v
		# 
		log_file = get_task_log_file_name(micK.workload_key, tuner = tuner,target = "cuda", kernel_type='micK1fetch',
											diff_measure_task_wlk=task.workload_key)
		if not os.path.exists(log_file):
			continue
		tmp = dict()
		_ = my_load_all_best_input_from_file_multiTileOnes(log_file, tvm.target.Target("cuda"), tmp, workload_key=task.workload_key)
		real_cost = tmp[task.workload_key][1]
		ansors[tuple(mick_shape)] = real_cost
		# 
		tsp = get_output_shape_from_wlk(task.workload_key, op_type)
		pred_cost = my_cost_model_full_dynamic_2Rngs(selected_fixedL_dict, 
			func_micK_curvs, popts, curve_rep_layouts, interested_blkNs, 
			mick_shape, tsp, op_type, interested_Rsps)
		preds[tuple(mick_shape)] = pred_cost
	return ansors, preds, avg_errors_keys
	################################################################################################


################################################################################################
# BMM: collect mick results for cost model
def get_data_for_bmm():
	op_type = 'bmm'
	tuner = 'ansor' # 'eto'
	dataReads = [(49152 / 4)]
	batch_num = 1	
	red_lens = [6, ] + list(range(12, 129, 12)) + [128, ]
	# red_lens = [108, 128]
	# red_lens = list(range(72, 128, 12))[1:] + [128, ]
	tasks = list() # only used for measuring micks when micks are being tuned
	mick_shapes = list()
	micks = list()
	# get min data read amount shapes
	interested0 = list()
	interested1 = list()
	for k in range(32, 26043, 32):
		# we only consider space sizes with factors 2, 3, 5 now
		if max(factorint(k).keys()) > 5:
			continue
		n = get_symmetric_factor(k)
		m = k // n
		# if (len(get_factors(n)) > 2) and (len(get_factors(m)) > 2):
		if n == m:
			interested0.append((batch_num, n, m))
		# if max(factorint(k).keys()) > 5:
		# 	continue
		else:
			interested1.append((batch_num, n, m))
		if n==m:
			for i, f in enumerate(get_factors(k)):
				if get_factors(k)[i+1] == n:
					interested1.append((batch_num, k//f, f))
					break
	selected_fixedL_dict = dict()
	for red_len in red_lens:
		# get min data read amount tasks
		for interested_Ssps in [interested0, interested1]:
			sps = list()
			for sp in interested_Ssps: #interested0:
				if data_read_amount_PerBlk(op_type, sp+(red_len,)) > (49152 / 4):
					continue
				sps.append(sp)
			sp_num = len(sps)
			print("sp_num: ", sp_num, sps, flush=True)
			step_size = max(10, sp_num)//10
			step_size2 = math.ceil(sp_num/10)
			if abs(10-len(sps[::step_size]))> abs(10-len(sps[::step_size2])):
				step_size = step_size2
			print(f"red_len: {red_len}; selected sps num: {len(sps[::step_size])}")
			for sp in sps[::step_size]:
			# for sp in sps[::max(40, sp_num)//40]:
				mick_shape = list(sp) + [red_len]
				update_mick_to_tune_infor(mick_shape, op_type, micks, mick_shapes, tasks, tuner=tuner)
		# get shapes with fixed loop len 1
		factor_limit = 5
		if red_len in [108, 128]:
			factor_limit = 13
		for fixedL in range(1, 21):
			# we only consider mick space sizes with factors 2, 3, 5 now.
			if (fixedL > 1) and (max(factorint(fixedL).keys()) > factor_limit):
				continue
			sps = list()
			for sp in [(batch_num, fixedL, k//fixedL) for k in range(32, 26043, 32) if k%fixedL == 0]:
				# we only consider mick space sizes with factors 2, 3, 5 now.
				if max(factorint(get_product(sp)).keys()) > factor_limit:
					continue
				if data_read_amount_PerBlk(op_type, sp+(red_len,)) > (49152 / 4):
					continue
				# if (fixedL == 1) or (data_read_amount_PerBlk(op_type, [sp[0]-1, get_product(sp)/(sp[0]-1)], rc = red_len) > (49152 / 4)):
				# 	sps.append(sp)
				sps.append(sp)
			sorted_shapes = sorted(sps, key=lambda sp: data_read_amount_PerBlk(op_type, sp+(red_len,)) )
			if data_read_amount_PerBlk(op_type, sorted_shapes[-1]+(red_len,)) >= (dataReads[0]-400):
				selected_fixedL_dict[red_len] = fixedL
				sp_num = len(sps)
				print(f"red_len:{red_len}, selected_fixedL:{fixedL}", flush=True)
				print("sp_num: ", sp_num, sps[-1], flush=True)
				print(f"sps: {sps}", flush=True)
				step_size = max(10, sp_num)//10
				step_size2 = math.ceil(sp_num/10)
				if abs(10-len(sps[::step_size]))> abs(10-len(sps[::step_size2])):
					step_size = step_size2
				print(f"red_len: {red_len}; selected sps num: {len(sps[::step_size])}")
				for sp in sps[::step_size]:
				# for sp in sps[::max(20, sp_num)//20]:
					mick_shape = list(sp) + [red_len]
					update_mick_to_tune_infor(mick_shape, op_type, micks, mick_shapes, tasks, tuner=tuner)
				break
		# # get max data read amount tasks
		# if red_len != 72:
		# 	continue
		for dataRead in dataReads:
			shapes = list()
			for i in range(1, math.ceil(dataRead//red_len/2)):
				n = i
				for m in range(int((dataRead-400)//red_len) - i, \
					int(dataRead//red_len) - i + 1):
					# n, m  = i, int(dataRead//red_len) - i
					# if (n > 10 and len(get_factors(n)) <= 2) or (m > 10 and len(get_factors(m)) <= 2):
					# 	continue
					if (n*m<26043) and (n<=m) and ((n*m)%32==0) and (max(factorint(n*m).keys()) <= factor_limit):
						shapes.append((batch_num, n, m))
			print(len(shapes))#, [get_product(sp) for sp in shapes], shapes)
			sorted_shapes = sorted(shapes, key=lambda sp: data_read_amount_PerBlk(op_type, sp+(red_len,)) )
			assert data_read_amount_PerBlk(op_type, sorted_shapes[-1]+(red_len,)) <= (49152 / 4)
			sorted_shapes = sorted(shapes, key=lambda sp: get_product(sp) )
			max_idx = None
			for idx, sp in enumerate(sorted_shapes):
				if get_product(sp) > min(10000, get_product(sorted_shapes[-1]) / 2):
					max_idx = idx
					break
			for sps in [sorted_shapes[:max_idx], sorted_shapes[max_idx:]]:
				sp_num = len(sps)
				print("sp_num of range 2 lw curve: ", sp_num, sps[-1], flush=True)
				print(f"sps: {sps}", flush=True)
				print(f"sp Ssizes: {[get_product(sp) for sp in sps]}", flush=True)
				step_size = max(10, sp_num)//10
				step_size2 = math.ceil(sp_num/10)
				if abs(10-len(sps[::step_size]))> abs(10-len(sps[::step_size2])):
					step_size = step_size2
				print(f"red_len: {red_len}; selected sps num: {len(sps[::step_size])}")
				for sp in sps[::step_size]:
				# for sp in sps[::max(20, sp_num)//20]:
					mick_shape = list(sp) + [red_len]
					update_mick_to_tune_infor(mick_shape, op_type, micks, mick_shapes, tasks, tuner=tuner)			
	tot_op_num = len(tasks)
	print(tot_op_num, flush = True)
	ansors = dict()
	measure_ctx = auto_scheduler.LocalRPCMeasureContext(min_repeat_ms=300)
	tune_option = auto_scheduler.TuningOptions(
	    num_measure_trials=1000,  # change this to 1000 to achieve the best performance
	    runner=measure_ctx.runner,
	    builder=auto_scheduler.LocalBuilder(n_parallel=60),
	    # measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
	    verbose=2,
	)
	# 
	for count, task in enumerate(tasks):
		micK = micks[count]
		mick_shape = mick_shapes[count]
		# if count % 1 != 0:
		# 	continue
		# if ((count+1) <= math.ceil(tot_op_num / 1) * (0+args.cuda)) \
		# 	or ((count+1) > math.ceil(tot_op_num / 1) * (1+args.cuda)):
		# 	continue
		print(micK.workload_key, flush=True)
		# if micK.workload_key in ['["dense_layer", [2462, 4], [8, 4]]', '["dense_layer", [2762, 4], [8, 4]]']:
		# 	continue
		tune_ops_ansor([mick_shape], [micK], tasks = [task], tune_mick = True, limit_fetch_num = True)
		# tmp_ansors = measure_micK_curve([[micK]], [[task.workload_key]], only_multi_tile = True, kernel_type = "micK1fetch", op_type = op_type)
		# for k, v in tmp_ansors.items():
		# 	ansors[k] = v
		# 
		# mick_cts, best_costs, best_states, mick_best_cts, dead_micks, mick_costs_history = [0], [1e10], [None], [0], set(), [[]]
		# log_file = get_task_log_file_name(micK.workload_key, 
		# 	tuner = tuner,target = "cuda", kernel_type='micK1fetch', diff_measure_task_wlk=task.workload_key)
		# eto_tune([0], [mick_shape], op_type, tune_option, 
		# 	mick_cts, best_costs, best_states, mick_best_cts, dead_micks, mick_costs_history, log_file)
		# 
		tmp_ansors = measure_micK_curve([[micK]], [[task.workload_key]], only_multi_tile = True, kernel_type = "micK1fetch", op_type = op_type, tuner=tuner)
		for k, v in tmp_ansors.items():
			ansors[k] = v
	return ansors
################################################################################################
# BMM collect data for prediction error
def get_pred_errors_for_bmm():
	op_type = 'bmm'
	tuner = 'ansor' # 'eto'
	tasks, mick_shapes, micks = list(), list(), list() # tasks only used for measuring micks when micks are being tuned
	batch_num = 1
	Ssps = [[batch_num, 80, 64], [batch_num, 48, 48]]
	red_len = 24 # 40 # 24
	# get min data read amount shapes
	avg_errors_keys={'Ssize':dict(), 'Rsp':dict()}
	interested0 = list()
	for k in range(32, 26043, 32):
		# we only consider space sizes with factors 2, 3, 5 now
		if max(factorint(k).keys()) > 5:
			continue
		n = get_symmetric_factor(k)
		m = k // n
		# if (len(get_factors(n)) > 2) and (len(get_factors(m)) > 2):
		interested0.append((batch_num, n, m))
	# different Ssps with a good fixed red_len
	for sp in interested0:
		mick_shape = list(sp) + [red_len]
		if (data_read_amount_PerBlk(op_type, sp+(red_len,)) > (49152 / 4)):
			continue
		update_mick_to_tune_infor(mick_shape, op_type, micks, mick_shapes, tasks, tuner=tuner)
		avg_errors_keys['Ssize'][get_product(sp)] = tuple(mick_shape)
	# different red_len with a good Ssp
	for k in range(1, 129):
		if (k > 1) and (max(factorint(k).keys()) > 5):
			continue
		for Ssp in Ssps:
			# if (k, ) in avg_errors['Rsp']:
			# 	continue
			if (data_read_amount_PerBlk(op_type, Ssp+[k]) > (49152 / 4)):
				continue
			mick_shape = list(Ssp) + [k]
			update_mick_to_tune_infor(mick_shape, op_type, micks, mick_shapes, tasks, tuner=tuner)
			avg_errors_keys['Rsp'][(k,)] = tuple(mick_shape)
			break
	tot_op_num = len(tasks)
	print(tot_op_num, flush = True)
	ansors = dict()
	preds = dict()
	measure_ctx = auto_scheduler.LocalRPCMeasureContext(min_repeat_ms=300)
	tune_option = auto_scheduler.TuningOptions(
	    num_measure_trials=1000,  # change this to 1000 to achieve the best performance
	    runner=measure_ctx.runner,
	    builder=auto_scheduler.LocalBuilder(n_parallel=60),
	    # measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
	    verbose=2,
	)
	selected_fixedL_dict, func_micK_curvs, popts, curve_rep_layouts, interested_blkNs, interested_Rsps = get_cost_model_params(op_type)
	for count, task in enumerate(tasks):
		micK = micks[count]
		mick_shape = mick_shapes[count]
		# if count % 1 != 0:
		# 	continue
		# if ((count+1) <= math.ceil(tot_op_num / 4) * (0+args.cuda)) \
		# 	or ((count+1) > math.ceil(tot_op_num / 4) * (1+args.cuda)):
		# 	continue
		print(micK.workload_key, flush=True)
		# if micK.workload_key in ['["dense_layer", [2462, 4], [8, 4]]', '["dense_layer", [2762, 4], [8, 4]]']:
		# 	continue
		tune_ops_ansor([mick_shape], [micK], tasks = [task], tune_mick = True, limit_fetch_num = True)
		# tmp_ansors = measure_micK_curve([[micK]], [[task.workload_key]], only_multi_tile = True, kernel_type = "micK1fetch", op_type = op_type)
		# for k, v in tmp_ansors.items():
		# 	ansors[k] = v	
		# 
		# mick_cts, best_costs, best_states, mick_best_cts, dead_micks, mick_costs_history = [0], [1e10], [None], [0], set(), [[]]
		# log_file = get_task_log_file_name(micK.workload_key, 
		# 	tuner = tuner,target = "cuda", kernel_type='micK1fetch', diff_measure_task_wlk=task.workload_key)
		# eto_tune([0], [mick_shape], op_type, tune_option, 
		# 	mick_cts, best_costs, best_states, mick_best_cts, dead_micks, mick_costs_history, log_file)
		# 
		log_file = get_task_log_file_name(micK.workload_key, tuner = tuner,target = "cuda", kernel_type='micK1fetch',
											diff_measure_task_wlk=task.workload_key)
		if not os.path.exists(log_file):
			continue
		tmp = dict()
		_ = my_load_all_best_input_from_file_multiTileOnes(log_file, tvm.target.Target("cuda"), tmp, workload_key=task.workload_key)
		real_cost = tmp[task.workload_key][1]
		ansors[tuple(mick_shape)] = real_cost
		# 
		# tmp_ansors = measure_micK_curve([[micK]], [[task.workload_key]], only_multi_tile = True, kernel_type = "micK1fetch", op_type = op_type, tuner=tuner)
		# for k, v in tmp_ansors.items():
		# 	ansors[k] = v
		# 
		# 
		tsp = get_output_shape_from_wlk(task.workload_key, op_type)
		pred_cost = my_cost_model_full_dynamic_2Rngs(selected_fixedL_dict, 
			func_micK_curvs, popts, curve_rep_layouts, interested_blkNs, 
			mick_shape, tsp, op_type, interested_Rsps)
		preds[tuple(mick_shape)] = pred_cost
	return ansors, preds, avg_errors_keys
	# ################################################################################################

################################################################################################
# BMM No-Transposed version: collect mick results for cost model
def get_data_for_bmm_nn():
	op_type = 'bmm_nn'
	tuner = 'eto' # 'ansor' # 'eto'
	dataReads = [(49152 / 4)]
	batch_num = 1
	red_lens = [6, ] + list(range(12, 129, 12)) + [128, ]
	# red_lens = [108, 128]
	# red_lens = [24, 120]
	# red_lens = list(range(72, 128, 12))[1:] + [128, ]
	tasks = list() # only used for measuring micks when micks are being tuned
	mick_shapes = list()
	micks = list()
	# get min data read amount shapes
	interested0 = list()
	interested1 = list()
	for k in range(128, 26043, 128): #(32, 26043, 32):
		# we only consider space sizes with factors 2, 3, 5 now
		if max(factorint(k).keys()) > 5:
			continue
		n = get_symmetric_factor(k)
		m = k // n
		# if (len(get_factors(n)) > 2) and (len(get_factors(m)) > 2):
		if n == m:
			interested0.append((batch_num, n, m))
		# if max(factorint(k).keys()) > 5:
		# 	continue
		else:
			interested1.append((batch_num, n, m))
		if n==m:
			for i, f in enumerate(get_factors(k)):
				if get_factors(k)[i+1] == n:
					interested1.append((batch_num, k//f, f))
					break
	selected_fixedL_dict = dict()
	for red_len in red_lens:
		# get min data read amount tasks
		print("="*50)
		for interested_Ssps in [interested0, interested1]:
			sps = list()
			for sp in interested_Ssps: #interested0:
				if data_read_amount_PerBlk(op_type, sp+(red_len,)) > (49152 / 4):
					continue
				sps.append(sp)
			sp_num = len(sps)
			print("sp_num: ", sp_num, sps, flush=True)
			step_size = max(10, sp_num)//10
			step_size2 = math.ceil(sp_num/10)
			if abs(10-len(sps[::step_size]))> abs(10-len(sps[::step_size2])):
				step_size = step_size2
			print(f"red_len: {red_len}; selected sps num: {len(sps[::step_size])}")
			for sp in sps[::step_size]:
			# for sp in sps[::max(40, sp_num)//40]:
				mick_shape = list(sp) + [red_len]
				update_mick_to_tune_infor(mick_shape, op_type, micks, mick_shapes, tasks, tuner=tuner)
		# get shapes with fixed loop len 1
		print("="*50)
		for fixedL in range(1, 21):
			# we only consider mick space sizes with factors 2, 3, 5 now.
			if (fixedL > 1) and (max(factorint(fixedL).keys()) > 13):#5):
				continue
			sps = list()
			# for sp in [(batch_num, fixedL, k//fixedL) for k in range(32, 26043, 32) if k%fixedL == 0]:
			for sp in [(batch_num, fixedL, k//fixedL) for k in range(128, 26043, 128) if k%fixedL == 0]:
				# we only consider mick space sizes with factors 2, 3, 5 now.
				if max(factorint(get_product(sp)).keys()) > 13: #5:
					continue
				if data_read_amount_PerBlk(op_type, sp+(red_len,)) > (49152 / 4):
					continue
				# if (fixedL == 1) or (data_read_amount_PerBlk(op_type, [sp[0]-1, get_product(sp)/(sp[0]-1)], rc = red_len) > (49152 / 4)):
				# 	sps.append(sp)
				sps.append(sp)
			sorted_shapes = sorted(sps, key=lambda sp: data_read_amount_PerBlk(op_type, sp+(red_len,)) )
			if data_read_amount_PerBlk(op_type, sorted_shapes[-1]+(red_len,)) >= (dataReads[0]-400):
				selected_fixedL_dict[red_len] = fixedL
				sp_num = len(sps)
				print(f"red_len:{red_len}, selected_fixedL:{fixedL}", flush=True)
				print("sp_num: ", sp_num, sps[-1], flush=True)
				print(f"sps: {sps}", flush=True)
				step_size = max(10, sp_num)//10
				step_size2 = math.ceil(sp_num/10)
				if abs(10-len(sps[::step_size]))> abs(10-len(sps[::step_size2])):
					step_size = step_size2
				print(f"selected sps num: {len(sps[::step_size])}")
				for sp in sps[::step_size]:
				# for sp in sps[::max(20, sp_num)//20]:
					mick_shape = list(sp) + [red_len]
					update_mick_to_tune_infor(mick_shape, op_type, micks, mick_shapes, tasks, tuner=tuner)
				break
		# # get max data read amount tasks
		# if red_len != 72:
		# 	continue
		print("="*50)
		for dataRead in dataReads:
			shapes = list()
			for i in range(1, math.ceil(dataRead//red_len/2)):
				n = i
				for m in range(int((dataRead-400)//red_len) - i, \
					int(dataRead//red_len) - i + 1):
					# n, m  = i, int(dataRead//red_len) - i
					# if (n > 10 and len(get_factors(n)) <= 2) or (m > 10 and len(get_factors(m)) <= 2):
					# 	continue
					# if (n*m<26043) and (n<=m) and ((n*m)%32==0) and (max(factorint(n*m).keys()) <= 13): #5):
					if (n*m<26043) and (n<=m) and ((n*m)%128==0) and (max(factorint(n*m).keys()) <= 13): #5):
						shapes.append((batch_num, n, m))
			print(len(shapes))#, [get_product(sp) for sp in shapes], shapes)
			sorted_shapes = sorted(shapes, key=lambda sp: data_read_amount_PerBlk(op_type, sp+(red_len,)) )
			assert data_read_amount_PerBlk(op_type, sorted_shapes[-1]+(red_len,)) <= (49152 / 4)
			sorted_shapes = sorted(shapes, key=lambda sp: get_product(sp) )
			max_idx = None
			for idx, sp in enumerate(sorted_shapes):
				if get_product(sp) > min(10000, get_product(sorted_shapes[-1]) / 2):
					max_idx = idx
					break
			for sps in [sorted_shapes[:max_idx], sorted_shapes[max_idx:]]:
				sp_num = len(sps)
				print("sp_num of range 2 lw curve: ", sp_num, sps[-1], flush=True)
				print(f"sps: {sps}", flush=True)
				print(f"sp Ssizes: {[get_product(sp) for sp in sps]}", flush=True)
				step_size = max(10, sp_num)//10
				step_size2 = math.ceil(sp_num/10)
				if abs(10-len(sps[::step_size]))> abs(10-len(sps[::step_size2])):
					step_size = step_size2
				print(f"red_len: {red_len}; selected sps num: {len(sps[::step_size])}")
				for sp in sps[::step_size]:
				# for sp in sps[::max(20, sp_num)//20]:
					mick_shape = list(sp) + [red_len]
					update_mick_to_tune_infor(mick_shape, op_type, micks, mick_shapes, tasks, tuner=tuner)			
	tot_op_num = len(tasks)
	print(tot_op_num, flush = True)
	ansors = dict()
	measure_ctx = auto_scheduler.LocalRPCMeasureContext(min_repeat_ms=300)
	tune_option = auto_scheduler.TuningOptions(
	    num_measure_trials=1000,  # change this to 1000 to achieve the best performance
	    runner=measure_ctx.runner,
	    builder=auto_scheduler.LocalBuilder(n_parallel=60),
	    # measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
	    verbose=2,
	)
	# 
	for count, task in enumerate(tasks):
		micK = micks[count]
		mick_shape = mick_shapes[count]
		# if count % 1 != 0:
		# 	continue
		if ((count+1) <= math.ceil(tot_op_num / 4) * (0+args.cuda)) \
			or ((count+1) > math.ceil(tot_op_num / 4) * (1+args.cuda)):
			continue
		print(micK.workload_key, flush=True)
		if micK.workload_key in ['["dense_layer", [2462, 4], [8, 4]]', '["dense_layer", [2762, 4], [8, 4]]']:
			continue
		# 
		# tune_ops_ansor([mick_shape], [micK], tasks = [task], tune_mick = True, limit_fetch_num = True)
		# 
		mick_cts, best_costs, best_states, mick_best_cts, dead_micks, mick_costs_history = [0], [1e10], [None], [0], set(), [[]]
		log_file = get_task_log_file_name(micK.workload_key, 
			tuner = tuner,target = "cuda", kernel_type='micK1fetch', diff_measure_task_wlk=task.workload_key)
		eto_tune([0], [mick_shape], op_type, tune_option, 
			mick_cts, best_costs, best_states, mick_best_cts, dead_micks, mick_costs_history, log_file)
		# 
		tmp_ansors = measure_micK_curve([[micK]], [[task.workload_key]], only_multi_tile = True, kernel_type = "micK1fetch", op_type = op_type, tuner=tuner)
		for k, v in tmp_ansors.items():
			ansors[k] = v
	return ansors
################################################################################################
# BMM No-Transposed version: collect data for prediction error
def get_pred_errors_for_bmm_nn():
	op_type = 'bmm_nn'
	tuner = 'eto' # 'ansor' # 'eto'
	tasks, mick_shapes, micks = list(), list(), list() # tasks only used for measuring micks when micks are being tuned
	batch_num = 1
	Ssps = [[batch_num, 80, 64], [batch_num, 48, 48]]
	red_len = 24 # 40 # 24
	# get min data read amount shapes
	interested0 = list()
	avg_errors_keys={'Ssize':dict(), 'Rsp':dict()}
	for k in range(32, 26043, 32):
		# we only consider space sizes with factors 2, 3, 5 now
		if max(factorint(k).keys()) > 5:
			continue
		n = get_symmetric_factor(k)
		m = k // n
		# if (len(get_factors(n)) > 2) and (len(get_factors(m)) > 2):
		interested0.append((batch_num, n, m))
	# different Ssps with a good fixed red_len
	for sp in interested0:
		mick_shape = list(sp) + [red_len]
		if (data_read_amount_PerBlk(op_type, sp+(red_len,)) > (49152 / 4)):
			continue
		update_mick_to_tune_infor(mick_shape, op_type, micks, mick_shapes, tasks, tuner=tuner)
		avg_errors_keys['Ssize'][get_product(sp)] = tuple(mick_shape)
	# different red_len with a good Ssp
	for k in range(1, 129):
		if (k > 1) and (max(factorint(k).keys()) > 5):
			continue
		for Ssp in Ssps:
			# if (k, ) in avg_errors['Rsp']:
			# 	continue
			if (data_read_amount_PerBlk(op_type, Ssp+[k]) > (49152 / 4)):
				continue
			mick_shape = list(Ssp) + [k]
			update_mick_to_tune_infor(mick_shape, op_type, micks, mick_shapes, tasks, tuner=tuner)
			avg_errors_keys['Rsp'][(k,)] = tuple(mick_shape)
			break
	tot_op_num = len(tasks)
	print(tot_op_num, flush = True)
	ansors = dict()
	preds = dict()
	measure_ctx = auto_scheduler.LocalRPCMeasureContext(min_repeat_ms=300)
	tune_option = auto_scheduler.TuningOptions(
	    num_measure_trials=1000,  # change this to 1000 to achieve the best performance
	    runner=measure_ctx.runner,
	    builder=auto_scheduler.LocalBuilder(n_parallel=60),
	    # measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
	    verbose=2,
	)
	selected_fixedL_dict, func_micK_curvs, popts, curve_rep_layouts, interested_blkNs, interested_Rsps = get_cost_model_params(op_type)	
	for count, task in enumerate(tasks):
		micK = micks[count]
		mick_shape = mick_shapes[count]
		# if count % 1 != 0:
		# 	continue
		# if ((count+1) <= math.ceil(tot_op_num / 4) * (0+args.cuda)) \
		# 	or ((count+1) > math.ceil(tot_op_num / 4) * (1+args.cuda)):
		# 	continue
		print(micK.workload_key, flush=True)
		# if micK.workload_key in ['["dense_layer", [2462, 4], [8, 4]]', '["dense_layer", [2762, 4], [8, 4]]']:
		# 	continue
		# tune_ops_ansor([mick_shape], [micK], tasks = [task], tune_mick = True, limit_fetch_num = True)
		# tmp_ansors = measure_micK_curve([[micK]], [[task.workload_key]], only_multi_tile = True, kernel_type = "micK1fetch", op_type = op_type)
		# for k, v in tmp_ansors.items():
		# 	ansors[k] = v	
		# 
		mick_cts, best_costs, best_states, mick_best_cts, dead_micks, mick_costs_history = [0], [1e10], [None], [0], set(), [[]]
		log_file = get_task_log_file_name(micK.workload_key, 
			tuner = tuner,target = "cuda", kernel_type='micK1fetch', diff_measure_task_wlk=task.workload_key)
		eto_tune([0], [mick_shape], op_type, tune_option, 
			mick_cts, best_costs, best_states, mick_best_cts, dead_micks, mick_costs_history, log_file)
		# 
		log_file = get_task_log_file_name(micK.workload_key, tuner = tuner,target = "cuda", kernel_type='micK1fetch',
											diff_measure_task_wlk=task.workload_key)
		if not os.path.exists(log_file):
			continue
		tmp = dict()
		_ = my_load_all_best_input_from_file_multiTileOnes(log_file, tvm.target.Target("cuda"), tmp, workload_key=task.workload_key)
		real_cost = tmp[task.workload_key][1]
		ansors[tuple(mick_shape)] = real_cost
		# 
		# tmp_ansors = measure_micK_curve([[micK]], [[task.workload_key]], only_multi_tile = True, kernel_type = "micK1fetch", op_type = op_type, tuner=tuner)
		# for k, v in tmp_ansors.items():
		# 	ansors[k] = v
		# 
		# 
		tsp = get_output_shape_from_wlk(task.workload_key, op_type)
		pred_cost = my_cost_model_full_dynamic_2Rngs(selected_fixedL_dict, 
			func_micK_curvs, popts, curve_rep_layouts, interested_blkNs, 
			mick_shape, tsp, op_type, interested_Rsps)
		preds[tuple(mick_shape)] = pred_cost
	return ansors, preds, avg_errors_keys
	# ################################################################################################

################################################################################################
# TUNE CONV2D MICRO-KERNELS: BUILD COST MODEL only contain the shapes that we are interested in, 
# The default stride, padding, dilation is 1, 0, 1
def get_data_for_conv2d():	
	op_type = 'conv2d'
	dataReads = [(49152 / 4)]
	red_sizes = [6] + list(range(12, 128, 12)) + [128]
	red_shapes = list()
	tasks, mick_shapes, micks = list(), list(), list()
	tuner = 'eto' # 'ansor' # 'eto'
	other_params = [(1,1), 0, (1,1)]
	# compute the interesting reduc shapes for each mick reduction size 
	for r_size in red_sizes:
		rw = get_symmetric_factor(r_size)
		rh = r_size // rw
		red_shapes.append([1, rh, rw])
		rw = min(factorint(r_size).keys())
		red_shapes.append([r_size//rw, 1, rw])
		# red_shapes.append((r_size, 1, 1))
	# 
	selected_fixedL_dict = dict()
	for red_shape in red_shapes:
		interested0 = list()
		interested1 = list()
		# 
		# we compute the up curve mick_shape for each reduc_shape, and make it with the minimum data read amount.
		for size in [32]+list(range(128, 26043, 128)): #range(32, 26043, 32):
			# compute the shape with the minimum data read amount
			if max(factorint(size).keys()) > 5:
				continue
			n1s = get_factors(size)
			min_cost = None
			best_s = None
			for n1 in n1s:
				for h in get_factors(n1):
					w = n1//h
					n = 1
					c = size//n1
					cost = data_read_amount_PerBlk(op_type, [n, c, h, w]+red_shape+other_params)
					if (min_cost==None) or (cost < min_cost):
						min_cost = cost
						best_s = [n, c, h, w]
			if best_s[2]*(red_shape[2]-1) == best_s[3]*(red_shape[1]-1):
				interested0.append(best_s)
			else:
				interested1.append(best_s)
		# get min data read amount tasks
		only_interested1_valid = False
		for interested_Ssps in [interested0, interested1]:
			sps = list()
			for sp in interested_Ssps: #interested0:
				if data_read_amount_PerBlk(op_type, sp+red_shape+other_params) > (49152 / 4):
					continue
				sps.append(sp)
			sp_num = len(sps)
			print("sp_num: ", sp_num, sps, flush=True)
			if sp_num == 0:
				only_interested1_valid == True
				continue
			sample_num = 10
			if only_interested1_valid:
				sample_num = 20
			step_size = max(sample_num, sp_num)//sample_num
			step_size2 = math.ceil(sp_num/sample_num)
			if abs(sample_num-len(sps[::step_size]))> abs(sample_num-len(sps[::step_size2])):
				step_size = step_size2
			print(f"red_shape: {red_shape}; selected sps num: {len(sps[::step_size])}")
			for sp in sps[::step_size]:
			# for sp in sps[::max(40, sp_num)//40]:
				mick_shape = sp+red_shape+other_params # a mick_shape is a tuple of (spatial loops, reduction loops, stride, padding, dilation)
				update_mick_to_tune_infor(mick_shape, op_type, micks, mick_shapes, tasks, tuner=tuner)
		# get shapes with fixed loop len 1
		for fixedL in range(1, 21):
			if (fixedL > 1) and (max(factorint(fixedL).keys()) > 13):#5):
				continue
			if 32%fixedL!=0:
				continue
			sps = list()
			for sp in [[1, k//fixedL, 1, fixedL] for k in range(32, 26043, 32) if k%fixedL == 0]: # changed from (fixedL, k//fixedL, 1, 1)
				# we only consider mick space sizes with factors 2, 3, 5 now.
				# if max(factorint(get_product(sp)).keys()) > 13: #5:
				# 	continue
				if data_read_amount_PerBlk(op_type, sp+red_shape+other_params) > (49152 / 4):
					continue
				# if (fixedL == 1) or (data_read_amount_PerBlk(op_type, [sp[0]-1, get_product(sp)/(sp[0]-1)], rc = red_len) > (49152 / 4)):
				# 	sps.append(sp)
				sps.append(sp)
			sorted_shapes = sorted(sps, key=lambda sp: data_read_amount_PerBlk(op_type, sp+red_shape+other_params) )
			if data_read_amount_PerBlk(op_type, sorted_shapes[-1]+red_shape+other_params) >= (dataReads[0]-400):
				selected_fixedL_dict[tuple(red_shape)] = fixedL
				sp_num = len(sps)
				print(f"red_shape:{red_shape}, selected_fixedL:{fixedL}", flush=True)
				print("sp_num: ", sp_num, sps[-1], flush=True)
				print(f"sps: {sps}", flush=True)
				step_size = max(10, sp_num)//10
				step_size2 = math.ceil(sp_num/10)
				if abs(10-len(sps[::step_size]))> abs(10-len(sps[::step_size2])):
					step_size = step_size2
				print(f"selected sps num: {len(sps[::step_size])}")
				for sp in sps[::step_size]:
				# for sp in sps[::max(20, sp_num)//20]:
					mick_shape = sp+red_shape+other_params
					update_mick_to_tune_infor(mick_shape, op_type, micks, mick_shapes, tasks, tuner=tuner)
				if sps[-1] not in sps[::step_size]:
					mick_shape = sps[-1]+red_shape+other_params
					update_mick_to_tune_infor(mick_shape, op_type, micks, mick_shapes, tasks, tuner=tuner)
				break
		# # get max data read amount tasks
		rc, rh, rw = red_shape
		for dataRead in dataReads:
			shapes = list()
			for i in range(1, math.ceil((dataRead-(rw-1)*rc*rh)/(rc*rh))):
				n = i
				for m in range(max(int((dataRead-400-rc*rh*(n-1+rw))//(rc*rh*rw)), 1), \
					max(int((dataRead-rc*rh*(n-1+rw))//(rc*rh*rw)), 0) + 1):
					# n, m  = i, int(dataRead//red_len) - i
					# if (n > 10 and len(get_factors(n)) <= 2) or (m > 10 and len(get_factors(m)) <= 2):
					# 	continue
					if (n*m<26043) and ((n*m)%32==0) and (max(factorint(n*m).keys()) <= 13): #5):
						shapes.append([1, m, 1, n]) # n->n, m->c, h = w = 1 --> change to n->w, m->c, n=h=1
			print(len(shapes))#, [get_product(sp) for sp in shapes], shapes)
			sorted_shapes = sorted(shapes, key=lambda sp: data_read_amount_PerBlk(op_type, sp+red_shape+other_params) )
			assert data_read_amount_PerBlk(op_type, sorted_shapes[-1]+red_shape+other_params) <= (49152 / 4)
			sorted_shapes = sorted(shapes, key=lambda sp: get_product(sp) )
			max_idx = None
			for idx, sp in enumerate(sorted_shapes):
				if get_product(sp) > min(10000, get_product(sorted_shapes[-1]) / 2):
					max_idx = idx
					break
			for sps in [sorted_shapes[:max_idx], sorted_shapes[max_idx:]]:
				sp_num = len(sps)
				print("sp_num of range 2 lw curve: ", sp_num, sps[-1], flush=True)
				print(f"sps: {sps}", flush=True)
				print(f"sp Ssizes: {[get_product(sp) for sp in sps]}", flush=True)
				step_size = max(10, sp_num)//10
				step_size2 = math.ceil(sp_num/10)
				if abs(10-len(sps[::step_size]))> abs(10-len(sps[::step_size2])):
					step_size = step_size2
				print(f"rc, rh, rw: {rc}, {rh}, {rw}; selected sps num: {len(sps[::step_size])}")
				for sp in sps[::step_size]:
				# for sp in sps[::max(20, sp_num)//20]:
					mick_shape = sp+red_shape+other_params
					update_mick_to_tune_infor(mick_shape, op_type, micks, mick_shapes, tasks, tuner=tuner)			
		# 
	tot_op_num = len(tasks)
	print(tot_op_num, flush = True)
	ansors = dict()
	# 
	measure_ctx = auto_scheduler.LocalRPCMeasureContext(min_repeat_ms=300)
	tune_option = auto_scheduler.TuningOptions(
	    num_measure_trials=1000,  # change this to 1000 to achieve the best performance
	    runner=measure_ctx.runner,
	    builder=auto_scheduler.LocalBuilder(n_parallel=60),
	    # measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
	    verbose=2,
	)
	# 
	for count, task in enumerate(tasks):
		micK = micks[count]
		mick_shape = mick_shapes[count]
		# if count % 1 != 0:
		# 	continue
		if ((count+1) <= math.ceil(tot_op_num / 4) * (0+args.cuda)) \
			or ((count+1) > math.ceil(tot_op_num / 4) * (1+args.cuda)):
			continue
		print(micK.workload_key, flush=True)
		# if micK.workload_key in ['["conv2d_nchw", [1, 1, 44, 47], [16, 1, 9, 12], [1, 1], 0, [1, 1], "float32"]']:
		# 	continue
		# tune_ops_ansor([mick_shape], [micK], tasks = [task], tune_mick = True, limit_fetch_num = True)
		# tmp_ansors = measure_micK_curve([[micK]], [[task.workload_key]], only_multi_tile = True, kernel_type = "micK1fetch", op_type = op_type)
		# for k, v in tmp_ansors.items():
		# 	ansors[k] = v
		# CHANGE TO USE ETO TUNIGN INSTEAD OF ANSOR
		mick_cts, best_costs, best_states, mick_best_cts, dead_micks, mick_costs_history = [0], [1e10], [None], [0], set(), [[]]
		log_file = get_task_log_file_name(micK.workload_key, 
			tuner = tuner,target = "cuda", kernel_type='micK1fetch', diff_measure_task_wlk=task.workload_key)
		eto_tune([0], [mick_shape], op_type, tune_option, 
			mick_cts, best_costs, best_states, mick_best_cts, dead_micks, mick_costs_history, log_file)
		tmp_ansors = measure_micK_curve([[micK]], [[task.workload_key]], only_multi_tile = True, kernel_type = "micK1fetch", op_type = op_type, tuner=tuner)
		for k, v in tmp_ansors.items():
			ansors[k] = v
	return ansors
	################################################################################################

################################################################################################
# CONV2D collect data for prediction error
def get_pred_errors_for_conv2d():
	op_type = 'conv2d'
	tasks, mick_shapes, micks = list(), list(), list() # tasks only used for measuring micks when micks are being tuned
	Ssps = [[1, 16, 20, 16], [1, 48, 8, 6]]
	r_size = 24 # 40 # 24
	rw = get_symmetric_factor(r_size)
	rh = r_size // rw
	red_shape = [1, rh, rw]
	other_params = [(1,1), 0, (1,1)]
	tuner = 'eto'
	# get min data read amount shapes
	avg_errors_keys={'Ssize':dict(), 'Rsp':dict()}
	interested0 = list()
	for size in range(32, 26043, 32):
		# 
		if max(factorint(size).keys()) > 5:
			continue
		n1s = get_factors(size)
		min_cost = None
		best_s = None
		for n1 in n1s:
			for h in get_factors(n1):
				w = n1//h
				n = 1
				c = size//n1
				cost = data_read_amount_PerBlk(op_type, [n, c, h, w]+red_shape+other_params)
				if (min_cost==None) or (cost < min_cost):
					min_cost = cost
					best_s = [n, c, h, w]
		interested0.append(best_s)
	# 
	# different Ssps with a good fixed red_len
	for sp in interested0:
		mick_shape = list(sp) + red_shape + other_params
		if data_read_amount_PerBlk(op_type, mick_shape) > (49152 / 4):
			continue
		update_mick_to_tune_infor(mick_shape, op_type, micks, mick_shapes, tasks, tuner=tuner)
		avg_errors_keys['Ssize'][get_product(sp)] = tuple(mick_shape)
	# different red_len with a good Ssp
	interested1 = list()
	for r_size in range(1, 129):
		rw = get_symmetric_factor(r_size)
		rh = r_size // rw
		interested1.append([1, rh, rw])
	# for k in range(1, 129):
	# 	if (k > 1) and (max(factorint(k).keys()) > 5):
	# 		continue
	for Rsp in interested1:
		for Ssp in Ssps:
			# if (k, ) in avg_errors['Rsp']:
			# 	continue
			mick_shape = list(Ssp) + Rsp + other_params
			if (data_read_amount_PerBlk(op_type, mick_shape) > (49152 / 4)):
				continue
			update_mick_to_tune_infor(mick_shape, op_type, micks, mick_shapes, tasks, tuner=tuner)
			avg_errors_keys['Rsp'][get_product(Rsp)] = tuple(mick_shape)
			break
	tot_op_num = len(tasks)
	print(tot_op_num, flush = True)
	ansors = dict()
	preds = dict()
	# 
	measure_ctx = auto_scheduler.LocalRPCMeasureContext(min_repeat_ms=300)
	tune_option = auto_scheduler.TuningOptions(
	    num_measure_trials=1000,  # change this to 1000 to achieve the best performance
	    runner=measure_ctx.runner,
	    builder=auto_scheduler.LocalBuilder(n_parallel=60),
	    # measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
	    verbose=2,
	)
	# 
	selected_fixedL_dict, func_micK_curvs, popts, curve_rep_layouts, interested_blkNs, interested_Rsps = get_cost_model_params(op_type)
	for count, task in enumerate(tasks):
		micK = micks[count]
		mick_shape = mick_shapes[count]
		# if count % 1 != 0:
		# 	continue
		# if ((count+1) <= math.ceil(tot_op_num / 3) * (0+args.cuda)) \
		# 	or ((count+1) > math.ceil(tot_op_num / 3) * (1+args.cuda)):
		# 	continue
		print(micK.workload_key, flush=True)
		# if micK.workload_key in ['["dense_layer", [2462, 4], [8, 4]]', '["dense_layer", [2762, 4], [8, 4]]']:
		# 	continue
		# tune_ops_ansor(tasks = [micK], tune_mick = True, limit_fetch_num = True)
		# 
		# tune_ops_ansor([mick_shape], [micK], tasks = [task], tune_mick = True, limit_fetch_num = True)
		# 
		mick_cts, best_costs, best_states, mick_best_cts, dead_micks, mick_costs_history = [0], [1e10], [None], [0], set(), [[]]
		log_file = get_task_log_file_name(micK.workload_key, 
			tuner = tuner,target = "cuda", kernel_type='micK1fetch', diff_measure_task_wlk=task.workload_key)
		eto_tune([0], [mick_shape], op_type, tune_option, 
			mick_cts, best_costs, best_states, mick_best_cts, dead_micks, mick_costs_history, log_file)
		# 
		# tmp_ansors = measure_micK_curve([[micK]], [[task.workload_key]], only_multi_tile = True, kernel_type = "micK1fetch", op_type = op_type)
		# for k, v in tmp_ansors.items():
		# 	ansors[k] = v	
		# 
		log_file = get_task_log_file_name(micK.workload_key, tuner = tuner,target = "cuda", kernel_type='micK1fetch',
											diff_measure_task_wlk=task.workload_key)
		if not os.path.exists(log_file):
			continue
		tmp = dict()
		_ = my_load_all_best_input_from_file_multiTileOnes(log_file, tvm.target.Target("cuda"), tmp, workload_key=task.workload_key)
		real_cost = tmp[task.workload_key][1]
		ansors[tuple(mick_shape)] = real_cost
		# 
		tsp = get_output_shape_from_wlk(task.workload_key, op_type)
		pred_cost = my_cost_model_full_dynamic_2Rngs(selected_fixedL_dict, 
			func_micK_curvs, popts, curve_rep_layouts, interested_blkNs, 
			mick_shape, tsp, op_type, interested_Rsps)
		preds[tuple(mick_shape)] = pred_cost	
	return ansors, preds, avg_errors_keys
	################################################################################################




def get_cost_model_params(micK_curs_multiTiles, op_type):
	# 
	# return the points on the upper part of the convex hull of the original point set
	def get_up_part_of_ConvexHull(xs, ys):
	    points = list(zip(xs, ys))
	    hull = ConvexHull(points)
	    up_xs = list()
	    up_ys = list()
	    for i, idx in enumerate(hull.vertices):
	        x = xs[idx]
	        y = ys[idx]
	        next_idx = hull.vertices[0]
	        if i < (len(hull.vertices) - 1): 
	            next_idx = hull.vertices[i+1]
	        if xs[next_idx] < x:
	            if x not in up_xs:
	                up_xs.append(x)
	                up_ys.append(y)
	            if xs[next_idx] not in up_xs:
	                up_xs.append(xs[next_idx])
	                up_ys.append(ys[next_idx])
	    sorted_idx = sorted(range(len(up_xs)), key=lambda i: up_xs[i])
	    return np.array(up_xs)[sorted_idx], np.array(up_ys)[sorted_idx]
	# 
	def get_close_points_from_ConvexHull(xs, ys, ratio=1):
	#     ratio defines how close the points from the hull we want
	    hull_xs, hull_ys = get_up_part_of_ConvexHull(xs, ys)
	    if (len(hull_xs)>1) and (hull_ys[0]>=hull_ys[1]):
	        xs = [xs[i] for i in range(len(xs)) if (xs[i] != hull_xs[0] and ys[i] != hull_ys[0])]
	        ys = [ys[i] for i in range(len(xs)) if (xs[i] != hull_xs[0] and ys[i] != hull_ys[0])]
	        hull_xs, hull_ys = get_up_part_of_ConvexHull(xs, ys)
	    ret_xs, ret_ys = list(), list()
	    for x,y in zip(xs,ys):
	        x1, x2 = None, None
	        y1, y2 = None, None
	        for i in range(len(hull_xs) - 1):
	            if (x >=hull_xs[i] and x <= hull_xs[i+1]):
	                x1, x2 = hull_xs[i], hull_xs[i+1]
	                y1, y2 = hull_ys[i], hull_ys[i+1]
	                break
	        assert x1!=None
	        ytmp = (y1-y2)*(x-x2)/(x1-x2)+y2
	        if y>=(ratio*ytmp):
	            ret_xs.append(x)
	            ret_ys.append(y)
	    point_dict = dict()
	    for i, x in enumerate(ret_xs):
	        if x not in point_dict:
	            point_dict[x] = list()
	        point_dict[x].append(ret_ys[i])
	    ret_xs, ret_ys = list(), list()
	    for x in sorted(point_dict.keys()):
	        ret_xs = ret_xs + [x for i in [point_dict[x]]]
	        ret_ys = ret_ys + sorted(point_dict[x], reverse=True)
	#     sorted_idx = sorted(range(len(ret_xs)), key=lambda i: ret_xs[i])
	#     return np.array(ret_xs)[sorted_idx], np.array(ret_ys)[sorted_idx]
	    return np.array(ret_xs), np.array(ret_ys)
	    
	SMNum=108
	popts = dict()
	curve_rep_layouts = dict()
	# base_blk_nums = [SMNum*i+1 for i in range(5)] + [(i+1)*SMNum for i in range(5)]
	interested_blkNs = {0:[48, 108], 1:[110, 216], 2:[220, 324], 3:[330, 432], 4:[440, 540]}
	base_blk_nums = list()
	for k in range(5):
		base_blk_nums = base_blk_nums + interested_blkNs[k]

	base_reduc_rep_nums = [2**i for i in range(7)]
	base_reduc_lens = [6, ] + list(range(12, 121, 12)) + [128, ] # [6, 12, 24, 36, 48, 60, 72, ]#18,]
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
	xs_dict = dict()
	ys_dict = dict()
	# valid_xs_dict = dict()
	# valid_ys_dict = dict()
	shape_dict = dict()
	# sqr_xs_dict = dict()
	# sqr_ys_dict = dict()


	for reduc_len in base_reduc_lens:
		for reduc_rep_num in base_reduc_rep_nums:
			for blk in base_blk_nums:
				for repK in rep_keys:
					for pos in curve_poses:
						for fixlen in fix_lens:
							key = (reduc_len, reduc_rep_num, blk, repK, pos, fixlen)
	                        xs_dict[key] = list()
	                        ys_dict[key] = list()
	                        valid_xs_dict[key] = list()
	                        valid_ys_dict[key] = list()
	                        shape_dict[key] = list()
	                        sqr_xs_dict[key] = list()
	                        sqr_ys_dict[key] = list()
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

	selected_fixedL_dict = None
	if op_type == 'dense':
		selected_fixedL_dict = {6: 2, 12: 4, 24: 8, 36: 8, 48: 8, 60: 4, 72: 6, 84: 2, 96: 4, 108: 8, 120: 4, 128: 16}
	elif op_type == 'bmm':
		selected_fixedL_dict = {108: 8, 128: 8, 6: 2, 12: 4, 24: 8, 36: 8, 48: 8, 60: 4, 72: 6, 84: 2, 96: 4, 120: 4}
	elif op_type == 'bmm_nn':
		selected_fixedL_dict = {6: 1, 12: 2, 24: 4, 36: 2, 48: 8, 60: 4, 72: 6, 84: 2, 96: 4, 108: 8, 120: 4, 128: 8}
	elif op_type == 'conv2d':
		selected_fixedL_dict = {(1, 2, 3): 1, (3, 1, 2): 1, (1, 3, 4): 1, (6, 1, 2): 1, (1, 4, 6): 2, (12, 1, 2): 2, (1, 6, 6): 2, (18, 1, 2): 2, (1, 6, 8): 4, (24, 1, 2): 4, (1, 6, 10): 4, (30, 1, 2): 4, (1, 8, 9): 4, (36, 1, 2): 4, (1, 7, 12): 2, (42, 1, 2): 2, (1, 8, 12): 8, (48, 1, 2): 8, (1, 9, 12): 2, (54, 1, 2): 2, (1, 10, 12): 8, (60, 1, 2): 8, (1, 8, 16): 8, (64, 1, 2): 16}


	for k_i, (k, v) in enumerate(micK_curs_multiTiles.items()):
		if (len(v) != 210) or (op_type in ['bmm', 'bmm_nn'] and len(v) != 140):
			continue
		poses = list()
		Rsp = None
		in_lw1, in_lw2 = False, False
        if op_type == 'dense':
			sshape = k[:2]
			size = get_product(sshape)
			n = get_symmetric_factor(size)
			m = size // n
			best_s = [n, m]
			if best_s == list(sshape):
				poses.append('up')
			# 
			Rsp = k[-1]
			if (k[0] == selected_fixedL_dict[k[-1]]):
				in_lw1 = True
			if (sum(k[:2])>=(49152 / 4 - 400)//k[2]):
				in_lw2 = True
			if in_lw2 or in_lw1:
				poses.append('lw')
		elif op_type in ['bmm', 'bmm_nn']:
			sshape = k[:3]
			size = get_product(sshape)
			n = get_symmetric_factor(size)
			m = size // n
			best_s = [1, n, m]
			extra_s = None
			if n == m:
				for i, f in enumerate(get_factors(size)):
					if get_factors(size)[i+1] == n:
						extra_s = [1, size//f, f]
						break
			if (best_s == list(sshape)) or ([1, m, n] == list(sshape)) or ((extra_s != None) and (extra_s == list(sshape))):
				poses.append('up')
			# 
			Rsp = k[-1]
			if (k[1] == selected_fixedL_dict[k[-1]]):
				in_lw1 = True
			if (sum(k[1:3])>=(49152 / 4 - 400)//(k[0]*k[3])):
				in_lw2 = True
			if in_lw2 or in_lw1:
				poses.append('lw')
			# 
		elif op_type == 'conv2d':
			rc, rh, rw, stride, padding, dilation = k[-6:]
			sshape = k[:4]
			poses = list()
			size = get_product(sshape)
			data_read = data_read_amount_PerBlk(op_type, k)
			#         
			n1s = get_factors(size)
			min_cost = None
			best_s = None
			for n1 in n1s:
				for h in get_factors(n1):
					w = n1//h
					n = 1
					c = size//n1
					cost = data_read_amount_PerBlk(op_type, [n, c, h, w]+[rc, rh, rw, stride, padding, dilation])
					if (min_cost==None) or (cost < min_cost):
						min_cost = cost
						best_s = [n, c, h, w]
			#         
			if (best_s == list(sshape)) or (data_read<=min_cost):
			    poses.append('up')
			#   
			Rsp = (rc, rh, rw)    
			if ((k[3] == selected_fixedL_dict[(rc, rh, rw)]) and (k[0] == 1) and (k[2] == 1)):
				in_lw1 = True
			if ((k[1] >= max(int((49152 / 4 - 400-rc*rh*(k[3]-1+rw))//(rc*rh*rw)), 1)) and (k[0] == 1) and (k[2] == 1)):
				in_lw2 = True
			# 
			if in_lw2 or \
				(data_read>=(49152 / 4 - 400)) or \
				in_lw1:
				poses.append('lw')
		# 
        if len(poses) == 0:
            continue
        count = 0
        for reduc_rep_num in base_reduc_rep_nums:
            for n in range(0, 5):
                blk_nums = interested_blkNs[n]
                for blk_num in blk_nums:
                    for repK in rep_keys:
                        if v[count] > 1:
                            for pos in poses:
                                fixlen = ["inf"]
                                if (pos == 'lw') and in_lw1:#(micK_curs_multiTiles == micK_curs_multiTiles_56):#(sum(k[:2])<(49152 / 4 - 400)//k[2]):#(k[0] <= 5):
                                    fixlen = [selected_fixedL_dict[Rsp]] # [k[0]]
                                if (pos == 'lw') and (in_lw2 or \
                                    (data_read>=(49152 / 4 - 400))):
                                    fixlen.append('inf')
                                for fl in fixlen:
                                    key = (Rsp, reduc_rep_num, blk_num, repK, pos, fl)
                                    if k in shape_dict[key]:
                                        continue
                                    xs_dict[key].append(get_product(sshape))
                                    ys_dict[key].append(v[count])
                                    shape_dict[key].append(k)                              
                        count+=1

	markers = ['v', '<']
	fixlenColors = ['purple']*(len(fix_lens)-1)+['green']

	for reduc_len in base_reduc_lens:
	    for reduc_rep_num in base_reduc_rep_nums:
	        for blk in base_blk_nums:
	            for repK in rep_keys:
	                for pos, marker in zip(curve_poses, markers):
	                    for fixlen, fixlenColor in zip(fix_lens, fixlenColors):
	                        key = (reduc_len, reduc_rep_num, blk, repK, pos, fixlen)
	                        Xs = np.array(xs_dict[key])#/1e4
	                        Ys = np.log(np.array(ys_dict[key]))#/1e3
	                        # Ys_latency = np.array([x*1e4/y/1e3/1e9 for x, y in zip(Xs, Ys)])
	                        # sqr_Xs = np.array(sqr_xs_dict[key])#/1e4
	                        # sqr_Ys = np.log(np.array(sqr_ys_dict[key]))#/1e3
	                        # valid_Xs = np.array(valid_xs_dict[key])#/1e4
	                        # valid_Ys = np.log(np.array(valid_ys_dict[key]))#/1e3
	                        if len(list(Ys[:10]))>0:
	                            print("="*50)
	                            print(key)
	                            print(list(Ys[:10]))
	                        # func_micK_curv = func_micK_curvs[key]
	                        popt = None
	                        if pos == 'up':
	                            if len(Xs) == 0:
	                                continue
	                            Xs_fit, Ys_fit = get_close_points_from_ConvexHull(Xs, Ys, ratio=1)#(sqr_Xs, sqr_Ys, ratio=1)
	                            popt = (list(Xs_fit), list(Ys_fit))
	                        else: 
	                            if len(Xs) == 0:
	                                continue
	                            Xs_fit, Ys_fit = get_close_points_from_ConvexHull(Xs, Ys, ratio=1)
	                            print(list(Xs), list(Ys))
	                            popt = (list(Xs_fit), list(Ys_fit))
	                        popts[key] = popt
	                        print(f"the best params are {popt}")
	return popts








def get_init_pred_errors(ansors, preds, avg_errors_keys):
	errors = {'Ssize':dict(), 'Rsp':dict()}
	for kind, vs in avg_errors_keys.items():
		for k, msp in vs.items():
			if preds[msp] == 1e10:
				continue
			errors[kind][k] = ansors[msp] / preds[msp]
	return errors




def get_data_for_cost_model(op_type):
	if op_type == 'dense':
		return get_data_for_dense()
	elif op_type == 'bmm':
		return get_data_for_bmm()
	elif op_type == 'bmm_nn':
		return get_data_for_bmm_nn()
	elif op_type == 'conv2d':
		return get_data_for_conv2d()


def get_data_for_pred_errors(op_type):
	if op_type == 'dense':
		return get_pred_errors_for_dense()
	elif op_type == 'bmm':
		return get_pred_errors_for_bmm()
	elif op_type == 'bmm_nn':
		return get_pred_errors_for_bmm_nn()
	elif op_type == 'conv2d':
		return get_pred_errors_for_conv2d()	



if __name__ == "__main__":
	# get cost model parameters
	for op_type in ['dense', 'bmm', 'bmm_nn', 'conv2d']:
		ansors = get_data_for_cost_model(op_type)
		popts = get_cost_model_params(ansors, op_type)
		with open(f'cost_model_params_{op_type}.py', 'w') as f:
			f.write(f'def get_popts():\n\treturn {popts}\n')
	# 
	# get init pred errors
	for op_type in ['dense', 'bmm', 'bmm_nn', 'conv2d']:
		ansors, preds, avg_errors_keys = get_data_for_pred_errors(op_type)
		errors = get_init_pred_errors(ansors, preds, avg_errors_keys)
		with open(f'pred_errors_{op_type}.py', 'w') as f:
			f.write(f'def get_pred_errors():\n\treturn {errors}\n')


