from solution0 import *



	################################################################################################
	# DENSE only test the shapes with the largest space size (around 25000) and fixed reduction length, to see which kind of shape is valid
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
	# TUNE CONV2D MICRO-KERNELS: BUILD COST MODEL only test the shapes that we are interested in, 
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
