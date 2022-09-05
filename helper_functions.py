import tvm
from tvm import te, auto_scheduler, topi

import copy






from itertools import combinations
from sympy.ntheory import factorint



class HARDWARE_INFOR(object):
	"""docstring for HARDWARE_INFOR"""
	def __init__(self):
		super(HARDWARE_INFOR, self).__init__()
		self.instruc_warp_scheduler = 4
		self.warp_size = 32






hardware_infor = HARDWARE_INFOR()		




@auto_scheduler.register_workload
def dense_layer(X_shape, Y_shape):
	X = te.placeholder(X_shape, name="X")
	Y = te.placeholder(Y_shape, name="Y")
	out = topi.nn.dense(X, Y, bias=None, out_dtype=None, auto_scheduler_rewritten_layout="")
	return [X, Y, out]


@auto_scheduler.register_workload
def conv2d_nchw(data_shape, kernel_shape, stride, padding, dilation_val, out_dtype):
	data = te.placeholder(data_shape, name="data")
	kernel = te.placeholder(kernel_shape, name="kernel")
	# bias = te.placeholder((1, CO, 1, 1), name="bias")
	conv = topi.nn.conv2d_nchw(data, kernel, stride, padding, dilation=dilation_val, out_dtype=out_dtype)
	# out = topi.nn.relu(conv + bias)
	return [data, kernel, conv]





@auto_scheduler.register_workload
def batch_matmul(X_shape, Y_shape, oshape = None):
	X = te.placeholder(X_shape, name="X")
	Y = te.placeholder(Y_shape, name="Y")
	out = topi.nn.batch_matmul(X, Y, oshape=None, auto_scheduler_rewritten_layout="")
	return [X, Y, out]



@auto_scheduler.register_workload
def batch_matmul_noTrans(X_shape, Y_shape, oshape = None):
	X = te.placeholder(X_shape, name="X")
	Y = te.placeholder(Y_shape, name="Y")
	out = topi.nn.batch_matmul(X, Y, oshape=None, transpose_b = False, auto_scheduler_rewritten_layout="")
	return [X, Y, out]






def get_product(elements):
	'''
		Get the product of the elements. 
		INPUT:	elements: list of ints.
	'''
	product = 1
	for i in elements:
		product = product * i
	return product




def get_factors(value):
	ret = list()
	for i in range(1, value + 1):
		if value % i == 0:
			ret.append(i)
	return ret






def get_op_para_ansor(task, loop):
	'''
		Get op parameter dictionary. task is of type SearchTask in Ansor
	'''
	import json
	def get_workload_dict(task):
		workload_dict = json.loads(s='{"wlk": ' + task.workload_key + '}')["wlk"]
		return workload_dict
	# 
	op_para = None
	workload = get_workload_dict(task)
	if workload[0] == 'conv2d_nchw':
		stride = workload[3]
		dilation = workload[5]
		_, _, kh, kw = workload[2]
		op_para = dict()
		op_para['kh'] = kh 
		op_para['kw'] = kw 
		op_para['stride'] = stride
		op_para['dilation'] = dilation
		op_para['op_type'] = "conv2d"
		op_para['loop'] = loop
		op_para['space_iters'] = [0, 1, 2, 3]
		op_para['reduc_iters'] = [4, 5, 6]
		op_para['load_step_iter'] = 4 # the axis divided for loading input data
		op_para['space_tile_knobs'] = ['tile_n', 'tile_f', 'tile_y', 'tile_x']
		op_para['reduc_tile_knobs'] = ['tile_rc', 'tile_ry', 'tile_rx']
		op_para['load_step_knob'] = 'tile_rc'
	elif workload[0] == 'group_conv2d_nchw':
		stride = workload[3]
		dilation = workload[5]
		groups = workload[6]
		_, in_channel, _, _ = workload[1]
		num_filter, _, kh, kw = workload[2]
		op_para = dict()
		op_para['kh'] = kh
		op_para['kw'] = kw
		op_para['stride'] = stride
		op_para['dilation'] = dilation
		op_para['groups'] = groups
		op_para['in_channel'] = in_channel
		op_para['num_filter'] = num_filter
		op_para['op_type'] = "group_conv2d"
		op_para['loop'] = loop
		op_para['space_iters'] = [0, 1, 2, 3]
		op_para['reduc_iters'] = [4, 5, 6]
		op_para['load_step_iter'] = 4 # the axis divided for loading input data
		op_para['space_tile_knobs'] = ['tile_n', 'tile_f', 'tile_y', 'tile_x']
		op_para['reduc_tile_knobs'] = ['tile_rc', 'tile_ry', 'tile_rx']
		op_para['load_step_knob'] = 'tile_rc'
		# knob1 in space knobs (i.e., tile_f) must cover k groups in blk shape
		op_para['tile_by_group_knob_idx'] = 1 
	elif workload[0] == 'depthwise_conv2d_nchw':
		stride = workload[3]
		dilation = workload[5]
		_, in_channel, _, _ = workload[1]
		_, channel_multiplier, kh, kw = workload[2]
		op_para = dict()
		op_para['kh'] = kh 
		op_para['kw'] = kw 
		op_para['stride'] = stride
		op_para['dilation'] = dilation
		op_para['in_channel'] = in_channel
		op_para['channel_multiplier'] = channel_multiplier
		op_para['op_type'] = "depthwise_conv2d"
		op_para['loop'] = loop
		op_para['space_iters'] = [0, 1, 2, 3]
		op_para['reduc_iters'] = [4, 5]
		op_para['load_step_iter'] = None # the axis divided for loading input data
		op_para['space_tile_knobs'] = ['tile_n', 'tile_f', 'tile_y', 'tile_x']
		op_para['reduc_tile_knobs'] = ['tile_ry', 'tile_rx']
		op_para['load_step_knob'] = None
	elif workload[0] == 'conv2d_transpose_nchw':
		_, _, kh, kw = workload[2]
		op_para = dict()
		op_para['kh'] = kh 
		op_para['kw'] = kw 
		op_para['op_type'] = "transpose_conv2d"
		op_para['loop'] = loop
		op_para['space_iters'] = [0, 1, 2, 3]
		op_para['reduc_iters'] = [4, 5, 6]
		op_para['load_step_iter'] = 4 # the axis divided for loading input data
		op_para['space_tile_knobs'] = ['tile_n', 'tile_f', 'tile_y', 'tile_x']
		op_para['reduc_tile_knobs'] = ['tile_rc', 'tile_ry', 'tile_rx']
		op_para['load_step_knob'] = 'tile_rc'		
	elif workload[0] == 'conv2d_capsule_nhwijc':
		# this op has not topi topi implementation, so we cannot extract autoTVM tasks
		# conv2d_capsule_nhwijc(N, H, W, CI, CO, kernel_size, stride=1, padding=0, capsule_size=4)
		# workload_key example: '["conv2d_capsule_nhwijc", 1, 16, 16, 32, 32, 3, 2, 1, 4]'
		op_para = dict()
		op_para['kh'] = workload[2] 
		op_para['kw'] = workload[3] 
		op_para['stride'] = workload[7] # it is int, not list
		op_para['op_type'] = "conv2d_capsule"
		op_para['loop'] = loop
		op_para['space_iters'] = [0, 1, 2, 3, 4, 5]
		op_para['reduc_iters'] = [6, 7, 8, 9]
		op_para['load_step_iter'] = 9 # the axis divided for loading input data
		op_para['space_tile_knobs'] = ['tile_n', 'tile_y', 'tile_x', 
										'tile_capi', 'tile_capj', 'tile_f']
		op_para['reduc_tile_knobs'] = ['tile_ry', 'tile_rx', 'tile_rcapk','tile_rc']
		op_para['load_step_knob'] = 'tile_rc'	
	elif workload[0] == 'conv1d_ncw':
		# we get op parameter of conv1d from ansor search task (not autotvm task)
		stride = workload[3]
		dilation = workload[5]
		_, _, kw = workload[2]
		op_para = dict()
		op_para['kw'] = kw
		op_para['stride'] = stride
		op_para['dilation'] = dilation
		op_para['op_type'] = "conv1d"
		op_para['loop'] = loop
		op_para['space_iters'] = [0, 1, 2]
		op_para['reduc_iters'] = [3, 4]
		op_para['load_step_iter'] = 3
		op_para['space_tile_knobs'] = ['tile_n', 'tile_f', 'tile_x']
		op_para['reduc_tile_knobs'] = ['tile_rc', 'tile_rx']
		op_para['load_step_knob'] = 'tile_rc'
	elif workload[0] == 'conv3d_ncdhw':
		stride = workload[3]
		dilation = workload[5]
		_, _, kd, kh, kw = workload[2]
		op_para = dict()
		op_para['kd'] = kd
		op_para['kh'] = kh
		op_para['kw'] = kw
		op_para['stride'] = stride
		op_para['dilation'] = dilation
		op_para['op_type'] = "conv3d"
		op_para['loop'] = loop
		op_para['space_iters'] = [0, 1, 2, 3, 4]
		op_para['reduc_iters'] = [5, 6, 7, 8]
		op_para['load_step_iter'] = 5
		op_para['space_tile_knobs'] = ['tile_n', 'tile_f', 'tile_d', 'tile_y', 'tile_x']
		op_para['reduc_tile_knobs'] = ['tile_rc', 'tile_rd', 'tile_ry', 'tile_rx']
		op_para['load_step_knob'] = 'tile_rc'
	elif workload[0] == 'batch_matmul':
		# output shape is [batch, M, N], iteration space is b, y, x, k.
		op_para = dict()
		op_para['X_shape'] = workload[1]
		op_para['Y_shape'] = workload[2]
		op_para['op_type'] = "batch_matmul"
		op_para['loop'] = loop
		op_para['space_iters'] = [0, 1, 2]
		op_para['reduc_iters'] = [3]
		op_para['load_step_iter'] = 3
		op_para['space_tile_knobs'] = ['tile_n', 'tile_y', 'tile_x']
		op_para['reduc_tile_knobs'] = ['tile_k']
		op_para['load_step_knob'] = 'tile_k'
	elif workload[0] == 'batch_matmul_noTrans':
		# output shape is [batch, M, N], iteration space is b, y, x, k.
		op_para = dict()
		op_para['X_shape'] = workload[1]
		op_para['Y_shape'] = workload[2]
		op_para['op_type'] = "batch_matmul_noTrans"
		op_para['loop'] = loop
		op_para['space_iters'] = [0, 1, 2]
		op_para['reduc_iters'] = [3]
		op_para['load_step_iter'] = 3
		op_para['space_tile_knobs'] = ['tile_n', 'tile_y', 'tile_x']
		op_para['reduc_tile_knobs'] = ['tile_k']
		op_para['load_step_knob'] = 'tile_k'
	elif workload[0] == 'dense_layer':
		# output shape is [batch, M, N], iteration space is b, y, x, k.
		op_para = dict()
		op_para['X_shape'] = workload[1]
		op_para['Y_shape'] = workload[2]
		op_para['op_type'] = "dense"
		op_para['loop'] = loop
		op_para['space_iters'] = [0, 1]
		op_para['reduc_iters'] = [2]
		op_para['load_step_iter'] = 2
		op_para['space_tile_knobs'] = ['tile_y', 'tile_x']
		op_para['reduc_tile_knobs'] = ['tile_k']
		op_para['load_step_knob'] = 'tile_k'
	elif workload[0] == 'frobenius_norm':
		op_para = dict()
		op_para['N'] = workload[1]
		op_para['H'] = workload[2]
		op_para['W'] = workload[3]
		op_para['op_type'] = "frobenius_norm"
		op_para['loop'] = [workload[1], workload[2], workload[3]]
		op_para['space_iters'] = [0]
		op_para['reduc_iters'] = [1, 2]
		op_para['max_loop_to_tile'] = 2 # there are at most 2 unfused loops to tile
	elif "conv2d_winograd" in task.compute_dag.init_state.stages[-1].__str__():
		# deal with winograd conv2d in nhwc layout
		op_para = dict()
		op_para['op_type'] = "conv2d_winograd"
		op_para['loop_set'] = loop
		op_para['loop'] = None # dynamically set it to the loop being optimized in 'loop_set'
		op_para['space_iters_set'] = [[2, 3], [0, 1, 2, 3], [2, 3], [0, 1, 2, 3]]
		op_para['reduc_iters_set'] = [[0, 1, 4, 5], [4,], [0, 1, 4, 5], []] # for datapack and inverse, [0,1] are unrolled space axes.
		op_para['space_iters'] = None
		op_para['reduc_iters'] = None
		# below is for space and reduc tile knobs of all stages
		# # the space axes for the ConstTensor stage: data_pack, are first tiled, reordered, fused, and then tiled
		op_para['space_tile_knobs_set'] = [['tile_p_datapack', 'tile_ci_datapack', 'tile_fuseS_datapack'], 
										['tile_eps', 'tile_nu', 'tile_p', 'tile_co'],
										['tile_p_inverse', 'tile_co_inverse', 'tile_fuseS_inverse'], 
										['tile_fuseS_winograd']]
		op_para['reduc_tile_knobs_set'] = [[], ['tile_ci'], [], []]
		op_para['space_tile_knobs'] = None
		op_para['reduc_tile_knobs'] = None		
		# below is for the stage: bgemm
		op_para['X_shape'] = copy.deepcopy(loop[0][:4]) # from data pack
		op_para['Y_shape'] = copy.deepcopy(loop[0][:2]) + [loop[1][3], loop[0][3]]
		op_para['load_step_iter'] = 4
		op_para['load_step_knob'] = 'tile_ci'
		op_para['auto_unroll_keys'] = ["auto_unroll_max_step_datapack", "auto_unroll_max_step_bgemm", "auto_unroll_max_step_inverse"]
	# 
	# print("OOOOOOOP_PARA: ", op_para)
	return op_para







def get_loops_ansor(tasks):
	'''
	Get the loop extents of the given tasks using Ansor.
	INPUT:
		tasks:
			list of SearchTasks in Ansor. 
	OUTPUT:
		loops:
			list of loops. each loop is a list of its iterator extents.
	'''
	def find_target_stage_id(tstage, stages):
		''' find the index of the corresponding stage with name `tstage` in a list of stages `stages`. '''
		target_stage_id = None
		for stage_id in range(len(stages)):
			if tstage in stages[stage_id].__str__():
				target_stage_id = stage_id
				break
		assert target_stage_id!=None, "error when extracting loops for winograd conv2d!"
		return target_stage_id
	# 
	count = 1
	loops = list()
	search_tasks = list()
	for i in range(len(tasks)):
		task = tasks[i]
		init_state = task.compute_dag.get_init_state()
		# 
		if count % 1000 == 0:
			print('*'*50)
			print('TASK NO: ', count)
			print('*'*50)
			print(task.compute_dag)
			print(init_state)
		count = count + 1
		# 
		# deal with winograd
		if "conv2d_winograd" in init_state.stages[-1].__str__():
			loop = list()
			# there 4 loops in winograd conv2d
			target_stages = ["data_pack", "bgemm", "inverse", "conv2d_winograd"]
			for tstage in target_stages:
				tstage_id = find_target_stage_id(tstage, init_state.stages)
				iters = init_state.stages[tstage_id].iters
				loop.append([int(iter_i.range.extent) for iter_i in iters])
		else:
			loop = list()
			iters = init_state.stages[-1].iters
			for iter_i in iters:
				loop.append(int(iter_i.range.extent))
		loops.append(loop)
		# print(task.workload_key)
	return loops















def get_input_len_for_conv(out_len, stride, dilation, reduc_len):
	'''
		A helper founction to compute the input length (1d) 
		if we want to get the given #out_length points in convolution.
		INPUT:
			out_len: int. The number of output points we want to computed.
			stride: int. The stride value.
			dilation: int. The dilation value.
			reduc_len: int. The reduction length. 
				For example, kernel width is 3, and we only read 2 into shared mem. So the reduc_len = 2.
		OUTPUT:
			in_len: int.
	'''
	return ((out_len - 1) * stride + (reduc_len - 1) * dilation + 1)








def cal_bk_cflct(blk_shape, thrd_shape, vthrd_shape, reduc_shape, op_para):
	'''
	Calculate how many bank conflicts are there when loading data from shared memory to registers/local memory.
	According to the documentation, P100 has 32 banks, each has 4B bandwidth per clock cycle.
	In a warp, there are 32 threads, where 2 threads accessing any address within the same 32-bit word (even though the two addresses fall in the same bank) would not cause bank conflicts.
	# 
	INPUT:
		blk_shape:
			list of int. The output shape the gpu block is responsible for;
		thrd_shape:
			list of int. The thread shape in a block;
		vthrd_shape:
			list of int. The virtual thread shape in a block;
		reduc_shape:
			list of int. For 2d convolution ops, reduc_shape is the shape of reduction axes 
							each time a block needs to load.
		op_para:
			a dictionary of parameters, which are necessary for us to compute the amount of memory transactions, also INCLUDES op_type and op loop structure.
	OUTPUT:
		list of conflict-free shared_mem access for loading one data from each input. 
		The order is (#conflict-free access in data, ~ in kernel) for convolution or (~ in X, ~ in Y) for batch matmul.
	'''
	def xy2idx(coordinates, space_shape):
		''' 
			transform the coordinates to the unique 1 dimension id in the given space.
			coordinates: the coordinates in the global env, list of ints.
			space_shape: the extent of the space axes corr. to the coordinates, list of ints. THE EXTENTS are listed from outer axes to inner axes.
		'''
		idx = coordinates[0]
		for i in range(1, len(coordinates)):
			idx = idx * space_shape[i] + coordinates[i]
		return idx
	def idx2xy(space_shape, idx):
		'''
			values are listed from outer axes to inner axes. 
			transform the index of a point to the coordinates.
		'''
		coordinates = [0 for i in range(len(space_shape))]
		for i in range(len(space_shape)-1, -1, -1):
			coordinates[i] = idx % space_shape[i]
			idx = idx // space_shape[i]
		return coordinates
	# 
	def nmem_req(coordinate_list, space_shape, bkn, bkwidth = 32):
		'''
			get the cost of warps loading once, i.e., the memory request number: the number of warps + max numer of bank conflicts on one bank (in terms of a warp) when all threads in the block are loading data.
			coordinate_list:
				list of list of coordinates, each coordinate is a list of ints, from outer to inner axes, in the space; each coordinate list is the address requests in a warp.
			space_shape:
				the shape of the space where coordinates are in, extents are listed from outer to inner axes.
			bkn:
				the number of banks in the shared memory. 32 for Nvidia P100.
			bkwidth:
				the bank width, 32-bits for P100.
			RMK:
				When compute the number of bank conflict, consider all warps, but for each warp, we only consider the the first data point the threads need to load:
				Because (1) bank conflict only happens among threads in a warp;
				(2), memory request should be served as a whole, so every time only one request can be served? (NOT SURE, JUST GUESS).
		'''
		# the total number of shared memory requests
		n_req = 0
		for warp_cords in coordinate_list:
			# store the different address requested by a warp in each bank
			req_dict = dict()
			for i in range(bkn):
				req_dict[i] = 0
			# we calculate the number of bank conflicts in each warp
			checked_idx = list()
			# accessed_bk = list()
			for xy in warp_cords:
				idx = xy2idx(xy, space_shape)
				if idx not in checked_idx:
					checked_idx.append(idx)
					bkNO = idx % bkn
					req_dict[bkNO] = req_dict[bkNO] + 1
					# if bkNO in accessed_bk:
					# 	nconflict = nconflict + 1
					# else:
					# 	accessed_bk.append(bkNO)
			n_req = n_req + max(req_dict.values())
		return n_req
	# 
	# get the total number of threads in a gpu block
	warp_size = 32 # the number of threads in a warp
	bank_num = 32 # the number of banks in shared memory
	thrd_num = 1
	for i in thrd_shape:
		thrd_num = thrd_num * i
	# 
	tot_nbkcflct = None
	op_type = op_para['op_type']
	if op_type == "conv2d":
		# the list of the first coordinates threads in warps want to access
		cord_list_d = list() 
		cord_list_k = list()
		tot_nbkcflct = 0
		stride, dilation = op_para['stride'], op_para['dilation']
		# get the consecutive block a thread is responsible for
		iter_shape = list()
		for i in range(len(blk_shape)):
			iter_shape.append(blk_shape[i] // (vthrd_shape[i] * thrd_shape[i]))
		# get the data and kernel shape in shared memory
		shape_d = [blk_shape[0], reduc_shape[0], 
					get_input_len_for_conv(blk_shape[2], stride[0], dilation[0], reduc_shape[1]), 
					get_input_len_for_conv(blk_shape[3], stride[1], dilation[1], reduc_shape[2])]
		shape_k = [blk_shape[1], reduc_shape[0], reduc_shape[1], reduc_shape[2]]
		# offset_d = 0
		# offset_k = blk_shape[0] *  reduc * ((blk_shape[2] - 1) * stride[0] + kh) * ((blk_shape[3] - 1) * stride[1] + kw)
		# we need to get the coordinates every warp requests
		warp_cords_d = None
		warp_cords_k = None
		for ti in range(thrd_num):
			if ti % warp_size == 0:
				# store the old warp request addresses
				if warp_cords_d != None:
					cord_list_d.append(warp_cords_d)
				if warp_cords_k != None:
					cord_list_k.append(warp_cords_k)
				# new a list to store the coordinates a warp want to access
				warp_cords_d = list()
				warp_cords_k = list()
			thrd_cord = idx2xy(thrd_shape, ti)
			cord_d = [thrd_cord[0] * iter_shape[0], 0, 
						conv_inputIdex(thrd_cord[2] * iter_shape[2], 0, stride[0], dilation[0]),
						conv_inputIdex(thrd_cord[3] * iter_shape[3], 0, stride[1], dilation[1])]
			cord_k = [thrd_cord[1] * iter_shape[1], 0, 0, 0]
			warp_cords_d.append(cord_d)
			warp_cords_k.append(cord_k)
		cord_list_d.append(warp_cords_d)
		cord_list_k.append(warp_cords_k)
		# tot_nbkcflct = tot_nbkcflct + nmem_req(cord_list_d, shape_d, bank_num)
		# tot_nbkcflct = tot_nbkcflct + nmem_req(cord_list_k, shape_k, bank_num)
		tot_nbkcflct = [nmem_req(cord_list_d, shape_d, bank_num), \
			nmem_req(cord_list_k, shape_k, bank_num)]
	elif op_type == "group_conv2d":
		# the list of the first coordinates threads in warps want to access
		cord_list_d = list() 
		cord_list_k = list()
		tot_nbkcflct = 0
		stride, dilation = op_para['stride'], op_para['dilation']
		# get the consecutive block a thread is responsible for
		iter_shape = list()
		for i in range(len(blk_shape)):
			iter_shape.append(blk_shape[i] // (vthrd_shape[i] * thrd_shape[i]))
		# get the data and kernel shape in shared memory
		filter_a_group = op_para['num_filter'] // op_para['groups']
		inchnl_a_group = op_para['in_channel'] // op_para['groups']
		# we use the first block as the standard to compute shape_d[1]
		shape_d = [blk_shape[0], 
					 ((blk_shape[1] - 1) // filter_a_group + 1) * reduc_shape[0],
					get_input_len_for_conv(blk_shape[2], stride[0], dilation[0], reduc_shape[1]), 
					get_input_len_for_conv(blk_shape[3], stride[1], dilation[1], reduc_shape[2])]
		shape_k = [blk_shape[1], reduc_shape[0], reduc_shape[1], reduc_shape[2]]
		# we need to get the coordinates every warp requests
		warp_cords_d = None
		warp_cords_k = None
		for ti in range(thrd_num):
			if ti % warp_size == 0:
				# store the old warp request addresses
				if warp_cords_d != None:
					cord_list_d.append(warp_cords_d)
				if warp_cords_k != None:
					cord_list_k.append(warp_cords_k)
				# new a list to store the coordinates a warp want to access
				warp_cords_d = list()
				warp_cords_k = list()
			thrd_cord = idx2xy(thrd_shape, ti)
			cord_d = [thrd_cord[0] * iter_shape[0], 0, 
						conv_inputIdex(thrd_cord[2] * iter_shape[2], 0, stride[0], dilation[0]),
						conv_inputIdex(thrd_cord[3] * iter_shape[3], 0, stride[1], dilation[1])]
			cord_k = [thrd_cord[1] * iter_shape[1], 0, 0, 0]
			warp_cords_d.append(cord_d)
			warp_cords_k.append(cord_k)
		cord_list_d.append(warp_cords_d)
		cord_list_k.append(warp_cords_k)
		# tot_nbkcflct = tot_nbkcflct + nmem_req(cord_list_d, shape_d, bank_num)
		# tot_nbkcflct = tot_nbkcflct + nmem_req(cord_list_k, shape_k, bank_num)
		tot_nbkcflct = [nmem_req(cord_list_d, shape_d, bank_num), \
			nmem_req(cord_list_k, shape_k, bank_num)]
	elif op_type == "depthwise_conv2d":
		# the list of the first coordinates threads in warps want to access
		cord_list_d = list() 
		cord_list_k = list()
		tot_nbkcflct = 0
		stride, dilation = op_para['stride'], op_para['dilation']
		# get the consecutive block a thread is responsible for
		iter_shape = list()
		for i in range(len(blk_shape)):
			iter_shape.append(blk_shape[i] // (vthrd_shape[i] * thrd_shape[i]))
		# get the data and kernel shape in shared memory
		# we use the first block as the standard to compute shape_d[1]
		shape_d = [blk_shape[0], 
					(blk_shape[1] - 1) // op_para['channel_multiplier'] + 1, 
					get_input_len_for_conv(blk_shape[2], stride[0], dilation[0], reduc_shape[0]), 
					get_input_len_for_conv(blk_shape[3], stride[1], dilation[1], reduc_shape[1])]
		# we change the shape of shape_k to 3D from 4D to make the computation easier
		shape_k = [blk_shape[1], reduc_shape[0], reduc_shape[1]]
		# we need to get the coordinates every warp requests
		warp_cords_d = None
		warp_cords_k = None
		for ti in range(thrd_num):
			if ti % warp_size == 0:
				# store the old warp request addresses
				if warp_cords_d != None:
					cord_list_d.append(warp_cords_d)
				if warp_cords_k != None:
					cord_list_k.append(warp_cords_k)
				# new a list to store the coordinates a warp want to access
				warp_cords_d = list()
				warp_cords_k = list()
			thrd_cord = idx2xy(thrd_shape, ti)
			cord_d = [thrd_cord[0] * iter_shape[0], 
						thrd_cord[1] * iter_shape[1] // op_para['channel_multiplier'], 
						conv_inputIdex(thrd_cord[2] * iter_shape[2], 0, stride[0], dilation[0]),
						conv_inputIdex(thrd_cord[3] * iter_shape[3], 0, stride[1], dilation[1])]
			cord_k = [thrd_cord[1] * iter_shape[1], 0, 0]
			warp_cords_d.append(cord_d)
			warp_cords_k.append(cord_k)
		cord_list_d.append(warp_cords_d)
		cord_list_k.append(warp_cords_k)
		# tot_nbkcflct = tot_nbkcflct + nmem_req(cord_list_d, shape_d, bank_num)
		# tot_nbkcflct = tot_nbkcflct + nmem_req(cord_list_k, shape_k, bank_num)
		tot_nbkcflct = [nmem_req(cord_list_d, shape_d, bank_num), \
			nmem_req(cord_list_k, shape_k, bank_num)]
	elif op_type == "transpose_conv2d":
		# the list of the first coordinates threads in warps want to access
		cord_list_d = list() 
		cord_list_k = list()
		tot_nbkcflct = 0
		# get the consecutive block a thread is responsible for
		iter_shape = list()
		for i in range(len(blk_shape)):
			iter_shape.append(blk_shape[i] // (vthrd_shape[i] * thrd_shape[i]))
		# get the data and kernel shape in shared memory
		shape_d = [blk_shape[0], reduc_shape[0], 
					get_input_len_for_conv(blk_shape[2], 1, 1, reduc_shape[1]), 
					get_input_len_for_conv(blk_shape[3], 1, 1, reduc_shape[2])]
		shape_k = [blk_shape[1], reduc_shape[0], reduc_shape[1], reduc_shape[2]]
		# we need to get the coordinates every warp requests
		warp_cords_d = None
		warp_cords_k = None
		for ti in range(thrd_num):
			if ti % warp_size == 0:
				# store the old warp request addresses
				if warp_cords_d != None:
					cord_list_d.append(warp_cords_d)
				if warp_cords_k != None:
					cord_list_k.append(warp_cords_k)
				# new a list to store the coordinates a warp want to access
				warp_cords_d = list()
				warp_cords_k = list()
			thrd_cord = idx2xy(thrd_shape, ti)
			cord_d = [thrd_cord[0] * iter_shape[0], 0, 
						conv_inputIdex(thrd_cord[2] * iter_shape[2], 0, 1, 1),
						conv_inputIdex(thrd_cord[3] * iter_shape[3], 0, 1, 1)]
			cord_k = [thrd_cord[1] * iter_shape[1], 0, 0, 0]
			warp_cords_d.append(cord_d)
			warp_cords_k.append(cord_k)
		cord_list_d.append(warp_cords_d)
		cord_list_k.append(warp_cords_k)
		# tot_nbkcflct = tot_nbkcflct + nmem_req(cord_list_d, shape_d, bank_num)
		# tot_nbkcflct = tot_nbkcflct + nmem_req(cord_list_k, shape_k, bank_num)
		tot_nbkcflct = [nmem_req(cord_list_d, shape_d, bank_num), \
			nmem_req(cord_list_k, shape_k, bank_num)]
	elif op_type == "conv2d_capsule":
		# the list of the first coordinates threads in warps want to access
		cord_list_d = list() 
		cord_list_k = list()
		tot_nbkcflct = 0
		stride = op_para['stride']
		# get the consecutive block a thread is responsible for
		iter_shape = list()
		for i in range(len(blk_shape)):
			iter_shape.append(blk_shape[i] // (vthrd_shape[i] * thrd_shape[i]))
		# get the data and kernel shape in shared memory
		shape_d = [blk_shape[0],  
					get_input_len_for_conv(blk_shape[1], stride, 1, reduc_shape[0]), 
					get_input_len_for_conv(blk_shape[2], stride, 1, reduc_shape[1]), 
					blk_shape[3], reduc_shape[2], reduc_shape[3]]
		shape_k = [reduc_shape[0], reduc_shape[1], reduc_shape[2], blk_shape[4], reduc_shape[3], blk_shape[5]]
		# we need to get the coordinates every warp requests
		warp_cords_d = None
		warp_cords_k = None
		for ti in range(thrd_num):
			if ti % warp_size == 0:
				# store the old warp request addresses
				if warp_cords_d != None:
					cord_list_d.append(warp_cords_d)
				if warp_cords_k != None:
					cord_list_k.append(warp_cords_k)
				# new a list to store the coordinates a warp want to access
				warp_cords_d = list()
				warp_cords_k = list()
			thrd_cord = idx2xy(thrd_shape, ti)
			cord_d = [thrd_cord[0] * iter_shape[0], 
						conv_inputIdex(thrd_cord[1] * iter_shape[1], 0, stride, 1),
						conv_inputIdex(thrd_cord[2] * iter_shape[2], 0, stride, 1),
						thrd_cord[3] * iter_shape[3], 0, 0]
			cord_k = [0, 0, 0, thrd_cord[4] * iter_shape[4], 0, thrd_cord[5] * iter_shape[5]]
			warp_cords_d.append(cord_d)
			warp_cords_k.append(cord_k)
		cord_list_d.append(warp_cords_d)
		cord_list_k.append(warp_cords_k)
		# tot_nbkcflct = tot_nbkcflct + nmem_req(cord_list_d, shape_d, bank_num)
		# tot_nbkcflct = tot_nbkcflct + nmem_req(cord_list_k, shape_k, bank_num)
		tot_nbkcflct = [nmem_req(cord_list_d, shape_d, bank_num), \
			nmem_req(cord_list_k, shape_k, bank_num)]
	elif op_type == "conv1d":
		# the list of the first coordinates threads in warps want to access
		cord_list_d = list() 
		cord_list_k = list()
		tot_nbkcflct = 0
		stride, dilation = op_para['stride'], op_para['dilation']
		# get the consecutive block a thread is responsible for
		iter_shape = list()
		for i in range(len(blk_shape)):
			iter_shape.append(blk_shape[i] // (vthrd_shape[i] * thrd_shape[i]))
		# get the data and kernel shape in shared memory
		shape_d = [blk_shape[0], reduc_shape[0], 
					get_input_len_for_conv(blk_shape[2], stride[0], dilation[0], reduc_shape[1])]
		shape_k = [blk_shape[1], reduc_shape[0], reduc_shape[1]]
		# we need to get the coordinates every warp requests
		warp_cords_d = None
		warp_cords_k = None
		for ti in range(thrd_num):
			if ti % warp_size == 0:
				# store the old warp request addresses
				if warp_cords_d != None:
					cord_list_d.append(warp_cords_d)
				if warp_cords_k != None:
					cord_list_k.append(warp_cords_k)
				# new a list to store the coordinates a warp want to access
				warp_cords_d = list()
				warp_cords_k = list()
			thrd_cord = idx2xy(thrd_shape, ti)
			cord_d = [thrd_cord[0] * iter_shape[0], 0, 
						conv_inputIdex(thrd_cord[2] * iter_shape[2], 0, stride[0], dilation[0])]
			cord_k = [thrd_cord[1] * iter_shape[1], 0, 0]
			warp_cords_d.append(cord_d)
			warp_cords_k.append(cord_k)
		cord_list_d.append(warp_cords_d)
		cord_list_k.append(warp_cords_k)
		# tot_nbkcflct = tot_nbkcflct + nmem_req(cord_list_d, shape_d, bank_num)
		# tot_nbkcflct = tot_nbkcflct + nmem_req(cord_list_k, shape_k, bank_num)
		tot_nbkcflct = [nmem_req(cord_list_d, shape_d, bank_num), \
			nmem_req(cord_list_k, shape_k, bank_num)]
	elif op_type == "conv3d":
		# the list of the first coordinates threads in warps want to access
		cord_list_d = list() 
		cord_list_k = list()
		tot_nbkcflct = 0
		stride, dilation = op_para['stride'], op_para['dilation']
		# get the consecutive block a thread is responsible for
		iter_shape = list()
		for i in range(len(blk_shape)):
			iter_shape.append(blk_shape[i] // (vthrd_shape[i] * thrd_shape[i]))
		# get the data and kernel shape in shared memory
		shape_d = [blk_shape[0], reduc_shape[0], 
					get_input_len_for_conv(blk_shape[2], stride[0], dilation[0], reduc_shape[1]), 
					get_input_len_for_conv(blk_shape[3], stride[1], dilation[1], reduc_shape[2]), 
					get_input_len_for_conv(blk_shape[4], stride[2], dilation[2], reduc_shape[3])]
		shape_k = [blk_shape[1], reduc_shape[0], reduc_shape[1], reduc_shape[2], reduc_shape[3]]
		# we need to get the coordinates every warp requests
		warp_cords_d = None
		warp_cords_k = None
		for ti in range(thrd_num):
			if ti % warp_size == 0:
				# store the old warp request addresses
				if warp_cords_d != None:
					cord_list_d.append(warp_cords_d)
				if warp_cords_k != None:
					cord_list_k.append(warp_cords_k)
				# new a list to store the coordinates a warp want to access
				warp_cords_d = list()
				warp_cords_k = list()
			thrd_cord = idx2xy(thrd_shape, ti)
			cord_d = [thrd_cord[0] * iter_shape[0], 0, 
						conv_inputIdex(thrd_cord[2] * iter_shape[2], 0, stride[0], dilation[0]),
						conv_inputIdex(thrd_cord[3] * iter_shape[3], 0, stride[1], dilation[1]),
						conv_inputIdex(thrd_cord[4] * iter_shape[4], 0, stride[2], dilation[2])]
			cord_k = [thrd_cord[1] * iter_shape[1], 0, 0, 0, 0]
			warp_cords_d.append(cord_d)
			warp_cords_k.append(cord_k)
		cord_list_d.append(warp_cords_d)
		cord_list_k.append(warp_cords_k)
		# tot_nbkcflct = tot_nbkcflct + nmem_req(cord_list_d, shape_d, bank_num)
		# tot_nbkcflct = tot_nbkcflct + nmem_req(cord_list_k, shape_k, bank_num)
		tot_nbkcflct = [nmem_req(cord_list_d, shape_d, bank_num), \
			nmem_req(cord_list_k, shape_k, bank_num)]		
	elif op_type == "batch_matmul":
		cord_list_A = list() 
		cord_list_B = list()
		tot_nbkcflct = 0
		# get the consecutive block a thread is responsible for
		iter_shape = list()
		for i in range(len(blk_shape)):
			iter_shape.append(blk_shape[i] // (vthrd_shape[i] * thrd_shape[i]))
		# get the data and kernel shape in shared memory
		shape_A = [blk_shape[0] if op_para['X_shape'][0] != 1 else 1, blk_shape[1], reduc_shape[0]]
		shape_B = [blk_shape[0] if op_para['Y_shape'][0] != 1 else 1, blk_shape[2], reduc_shape[0]]
		# we need to get the coordinates every warp requests
		warp_cords_A = None
		warp_cords_B = None
		for ti in range(thrd_num):
			if ti % warp_size == 0:
				# store the old warp request addresses
				if warp_cords_A != None:
					cord_list_A.append(warp_cords_A)
				if warp_cords_B != None:
					cord_list_B.append(warp_cords_B)
				# new a list to store the coordinates a warp want to access
				warp_cords_A = list()
				warp_cords_B = list()
			thrd_cord = idx2xy(thrd_shape, ti)
			cord_A = [(thrd_cord[0] * iter_shape[0]) if (op_para['X_shape'] != 1) else 0, thrd_cord[1] * iter_shape[1], 0]
			cord_B = [(thrd_cord[0] * iter_shape[0]) if (op_para['Y_shape'] != 1) else 0, thrd_cord[2] * iter_shape[2], 0]
			warp_cords_A.append(cord_A)
			warp_cords_B.append(cord_B)
		cord_list_A.append(warp_cords_A)
		cord_list_B.append(warp_cords_B)
		# tot_nbkcflct = tot_nbkcflct + nmem_req(cord_list_A, shape_A, bank_num)
		# tot_nbkcflct = tot_nbkcflct + nmem_req(cord_list_B, shape_B, bank_num)
		tot_nbkcflct = [nmem_req(cord_list_A, shape_A, bank_num), \
			nmem_req(cord_list_B, shape_B, bank_num)]	
	elif op_type == "batch_matmul_noTrans":
		cord_list_A = list() 
		cord_list_B = list()
		tot_nbkcflct = 0
		# get the consecutive block a thread is responsible for
		iter_shape = list()
		for i in range(len(blk_shape)):
			iter_shape.append(blk_shape[i] // (vthrd_shape[i] * thrd_shape[i]))
		# get the data and kernel shape in shared memory
		shape_A = [blk_shape[0] if op_para['X_shape'][0] != 1 else 1, blk_shape[1], reduc_shape[0]]
		shape_B = [blk_shape[0] if op_para['Y_shape'][0] != 1 else 1, reduc_shape[0], blk_shape[2]]
		# we need to get the coordinates every warp requests
		warp_cords_A = None
		warp_cords_B = None
		for ti in range(thrd_num):
			if ti % warp_size == 0:
				# store the old warp request addresses
				if warp_cords_A != None:
					cord_list_A.append(warp_cords_A)
				if warp_cords_B != None:
					cord_list_B.append(warp_cords_B)
				# new a list to store the coordinates a warp want to access
				warp_cords_A = list()
				warp_cords_B = list()
			thrd_cord = idx2xy(thrd_shape, ti)
			cord_A = [(thrd_cord[0] * iter_shape[0]) if (op_para['X_shape'] != 1) else 0, thrd_cord[1] * iter_shape[1], 0]
			cord_B = [(thrd_cord[0] * iter_shape[0]) if (op_para['Y_shape'] != 1) else 0, 0, thrd_cord[2] * iter_shape[2]]
			warp_cords_A.append(cord_A)
			warp_cords_B.append(cord_B)
		cord_list_A.append(warp_cords_A)
		cord_list_B.append(warp_cords_B)
		# tot_nbkcflct = tot_nbkcflct + nmem_req(cord_list_A, shape_A, bank_num)
		# tot_nbkcflct = tot_nbkcflct + nmem_req(cord_list_B, shape_B, bank_num)
		tot_nbkcflct = [nmem_req(cord_list_A, shape_A, bank_num), \
			nmem_req(cord_list_B, shape_B, bank_num)]	
	elif op_type == "dense":
		cord_list_A = list() 
		cord_list_B = list()
		tot_nbkcflct = 0
		# get the consecutive block a thread is responsible for
		iter_shape = list()
		for i in range(len(blk_shape)):
			iter_shape.append(blk_shape[i] // (vthrd_shape[i] * thrd_shape[i]))
		# get the data and kernel shape in shared memory
		shape_A = [blk_shape[0], reduc_shape[0]]
		shape_B = [blk_shape[1], reduc_shape[0]]
		# we need to get the coordinates every warp requests
		warp_cords_A = None
		warp_cords_B = None
		for ti in range(thrd_num):
			if ti % warp_size == 0:
				# store the old warp request addresses
				if warp_cords_A != None:
					cord_list_A.append(warp_cords_A)
				if warp_cords_B != None:
					cord_list_B.append(warp_cords_B)
				# new a list to store the coordinates a warp want to access
				warp_cords_A = list()
				warp_cords_B = list()
			thrd_cord = idx2xy(thrd_shape, ti)
			cord_A = [thrd_cord[0] * iter_shape[0], 0]
			cord_B = [thrd_cord[1] * iter_shape[1], 0]
			warp_cords_A.append(cord_A)
			warp_cords_B.append(cord_B)
		cord_list_A.append(warp_cords_A)
		cord_list_B.append(warp_cords_B)
		# tot_nbkcflct = tot_nbkcflct + nmem_req(cord_list_A, shape_A, bank_num)
		# tot_nbkcflct = tot_nbkcflct + nmem_req(cord_list_B, shape_B, bank_num)
		tot_nbkcflct = [nmem_req(cord_list_A, shape_A, bank_num), \
			nmem_req(cord_list_B, shape_B, bank_num)]		
	elif op_type == "conv2d_winograd":
		cord_list_A = list() 
		cord_list_B = list()
		tot_nbkcflct = 0
		# get the consecutive block a thread is responsible for
		iter_shape = list()
		for i in range(len(blk_shape)):
			iter_shape.append(blk_shape[i] // (vthrd_shape[i] * thrd_shape[i]))
		# get the data and kernel shape in shared memory
		shape_A = [blk_shape[0], blk_shape[1], blk_shape[2], reduc_shape[0]]
		shape_B = [blk_shape[0], blk_shape[1], blk_shape[3], reduc_shape[0]]
		# we need to get the coordinates every warp requests
		warp_cords_A = None
		warp_cords_B = None
		for ti in range(thrd_num):
			if ti % warp_size == 0:
				# store the old warp request addresses
				if warp_cords_A != None:
					cord_list_A.append(warp_cords_A)
				if warp_cords_B != None:
					cord_list_B.append(warp_cords_B)
				# new a list to store the coordinates a warp want to access
				warp_cords_A = list()
				warp_cords_B = list()
			thrd_cord = idx2xy(thrd_shape, ti)
			cord_A = [thrd_cord[0] * iter_shape[0], thrd_cord[1] * iter_shape[1], thrd_cord[2] * iter_shape[2], 0]
			cord_B = [thrd_cord[0] * iter_shape[0], thrd_cord[1] * iter_shape[1], thrd_cord[3] * iter_shape[3], 0]
			warp_cords_A.append(cord_A)
			warp_cords_B.append(cord_B)
		cord_list_A.append(warp_cords_A)
		cord_list_B.append(warp_cords_B)
		# tot_nbkcflct = tot_nbkcflct + nmem_req(cord_list_A, shape_A, bank_num)
		# tot_nbkcflct = tot_nbkcflct + nmem_req(cord_list_B, shape_B, bank_num)
		tot_nbkcflct = [nmem_req(cord_list_A, shape_A, bank_num), \
			nmem_req(cord_list_B, shape_B, bank_num)]	
	# 
	return tot_nbkcflct






def load_per_thread(blk_shape, thrd_shape, vthrd_shape, reduc_shape, op_para, load_shape = None):
	'''	
		Get the number of data a thread needs to load from shared memory to its registers. 
		RMK:
			The value in any shapes are listed from outer to inner axes.
			We calculate for the thread with id 0 in the thread block.
			If not given, we assume every time, the reduc axes we load, i.e., load_shape = reduc_shape
		OUTPUT:
			return the list of the number of data to load for each input. (RMK!)
	'''
	def get_conv_InputIdxSet(out_idx_range, reduc_idx_range, stride, dilation):
		'''
			RMK: The index starts from 0.
			Get the indice in one dimension (y or x) in conv op that a thread needs as input.
			INPUT:
				out_idx_range:	[int, int). The start and end index (not included) of the output point we want to compute.
				reduc_idx:		[int, int). The start and end index (not included) of the reduction element we want to compute for each output point.
				stride:			int. The stride value.
				dilation:		int. The dilation value.
		'''
		# print(out_idx_range, reduc_idx_range, stride, dilation)
		indice = set()
		for i in range(out_idx_range[0], out_idx_range[1]):
			for r in range(reduc_idx_range[0], reduc_idx_range[1]):
				# print(i, r, type(i), type(r), type(stride), type(dilation))
				indice.add(conv_inputIdex(i, r, stride, dilation))
		# print(indice)
		return indice
	# 
	def area_size(xyranges):
		''' 
			get the size of the given xyranges, which is a list of ranges on each axis. each axis has a list of ranges, which may overlap each other. 
			RMK:
				We assume the ranges for an axis are listed in an increasding order.
				The range can be list of [lower bound, upper bound], Or the ranges for an axis is just a set of actual indice.
		'''
		tot_size = 1
		for ranges in xyranges: # ranges are for one axis
			if isinstance(ranges, set):
				tot_size = tot_size * len(ranges)
				continue
			# else, the ranges are a list of [lower, upper] bounds
			size = 0
			end = -1
			for i in range(len(ranges)):
				rng = ranges[i]
				if rng[0] > end:
					size = size + rng[1] - rng[0] + 1
					end = rng[1]
				else:
					if rng[1] > end:
						size = size + rng[1] - end
						end = rng[1]
			tot_size = tot_size * size
		return tot_size
	# 
	if load_shape == None:
		load_shape = reduc_shape
	ret = None
	if op_para['op_type'] == "conv2d":
		stride, dilation = op_para['stride'], op_para['dilation']
		iter_shape = [blk_shape[i] // (thrd_shape[i] * vthrd_shape[i]) for i in range(len(blk_shape))]
		xyranges_d = [list() for i in range(4)]
		xyranges_k = [list() for i in range(4)]
		axis = 0 # n 
		vthrd_range = iter_shape[axis] * thrd_shape[axis]
		for vi in range(vthrd_shape[axis]):
			xyranges_d[0].append([vi * vthrd_range, vi * vthrd_range + iter_shape[axis] - 1 ])
		axis = 1 # f
		vthrd_range = iter_shape[axis] * thrd_shape[axis]
		for vi in range(vthrd_shape[axis]):
			xyranges_k[0].append([vi * vthrd_range, vi * vthrd_range + iter_shape[axis] - 1 ])
		axis = 2 # y
		xyranges_d[2] = set()
		vthrd_range = iter_shape[axis] * thrd_shape[axis]
		for vi in range(vthrd_shape[axis]):
			# xyranges_d[2].append([conv_inputIdex(vi * vthrd_range, 0, stride[0], dilation[0]), 
			# 	conv_inputIdex(vi * vthrd_range + iter_shape[axis] - 1, reduc_shape[1] - 1, stride[0], dilation[0])])
			xyranges_d[2].update(get_conv_InputIdxSet([vi * vthrd_range, vi * vthrd_range + iter_shape[axis]], \
				[0, reduc_shape[1]], stride[0], dilation[0]))
		axis = 3 # x
		xyranges_d[3] = set()
		vthrd_range = iter_shape[axis] * thrd_shape[axis]
		for vi in range(vthrd_shape[axis]):
			# xyranges_d[3].append([conv_inputIdex(vi * vthrd_range, 0, stride[1], dilation[1]), 
			# 	conv_inputIdex(vi * vthrd_range + iter_shape[axis] - 1, reduc_shape[2] - 1, stride[1], dilation[1])])			
			xyranges_d[3].update(get_conv_InputIdxSet([vi * vthrd_range, vi * vthrd_range + iter_shape[axis]], \
				[0, reduc_shape[2]], stride[1], dilation[1]))
		xyranges_d[1].append([0, reduc_shape[0] - 1])
		xyranges_k[1].append([0, reduc_shape[0]-1])
		xyranges_k[2].append([0, reduc_shape[1]-1])
		xyranges_k[3].append([0, reduc_shape[2]-1])
		# get the amount of data to load
		# ret = area_size(xyranges_d) + area_size(xyranges_k)
		ret = [area_size(xyranges_d), area_size(xyranges_k)]
	elif op_para['op_type'] == "group_conv2d":
		stride, dilation = op_para['stride'], op_para['dilation']
		filter_a_group = op_para['num_filter'] // op_para['groups']
		inchnl_a_group = op_para['in_channel'] // op_para['groups']
		iter_shape = [blk_shape[i] // (thrd_shape[i] * vthrd_shape[i]) for i in range(len(blk_shape))]
		xyranges_d = [list() for i in range(4)]
		xyranges_k = [list() for i in range(4)]
		axis = 0 # n 
		vthrd_range = iter_shape[axis] * thrd_shape[axis]
		for vi in range(vthrd_shape[axis]):
			xyranges_d[0].append([vi * vthrd_range, vi * vthrd_range + iter_shape[axis] - 1 ])
		axis = 1 # f
		vthrd_range = iter_shape[axis] * thrd_shape[axis]
		for vi in range(vthrd_shape[axis]):
			# xyranges_d[1].append([(vi * vthrd_range) // filter_a_group * reduc_shape[0], 
			# 					(vi * vthrd_range + iter_shape[axis] - 1) // filter_a_group * reduc_shape[0] + reduc_shape[0] - 1])
			for start_d1 in range((vi * vthrd_range) // filter_a_group, (vi * vthrd_range + iter_shape[axis] - 1) // filter_a_group + 1):
				xyranges_d[1].append([start_d1 * load_shape[0], start_d1 * load_shape[0] + reduc_shape[0] - 1])
			xyranges_k[0].append([vi * vthrd_range, vi * vthrd_range + iter_shape[axis] - 1 ])
		axis = 2 # y
		xyranges_d[2] = set()
		vthrd_range = iter_shape[axis] * thrd_shape[axis]
		for vi in range(vthrd_shape[axis]):
			# xyranges_d[2].append([conv_inputIdex(vi * vthrd_range, 0, stride[0], dilation[0]), 
			# 	conv_inputIdex(vi * vthrd_range + iter_shape[axis] - 1, reduc_shape[1] - 1, stride[0], dilation[0])])
			xyranges_d[2].update(get_conv_InputIdxSet([vi * vthrd_range, vi * vthrd_range + iter_shape[axis]], \
				[0, reduc_shape[1]], stride[0], dilation[0]))
		axis = 3 # x
		xyranges_d[3] = set()
		vthrd_range = iter_shape[axis] * thrd_shape[axis]
		for vi in range(vthrd_shape[axis]):
			# xyranges_d[3].append([conv_inputIdex(vi * vthrd_range, 0, stride[1], dilation[1]), 
			# 	conv_inputIdex(vi * vthrd_range + iter_shape[axis] - 1, reduc_shape[2] - 1, stride[1], dilation[1])])			
			xyranges_d[3].update(get_conv_InputIdxSet([vi * vthrd_range, vi * vthrd_range + iter_shape[axis]], \
				[0, reduc_shape[2]], stride[1], dilation[1]))
		xyranges_k[1].append([0, reduc_shape[0]-1])
		xyranges_k[2].append([0, reduc_shape[1]-1])
		xyranges_k[3].append([0, reduc_shape[2]-1])
		# get the amount of data to load
		# ret = area_size(xyranges_d) + area_size(xyranges_k)
		ret = [area_size(xyranges_d), area_size(xyranges_k)]
	elif op_para['op_type'] == "depthwise_conv2d":
		stride, dilation = op_para['stride'], op_para['dilation']
		iter_shape = [blk_shape[i] // (thrd_shape[i] * vthrd_shape[i]) for i in range(len(blk_shape))]
		xyranges_d = [list() for i in range(4)]
		xyranges_k = [list() for i in range(3)]
		axis = 0 # n 
		vthrd_range = iter_shape[axis] * thrd_shape[axis]
		for vi in range(vthrd_shape[axis]):
			xyranges_d[0].append([vi * vthrd_range, vi * vthrd_range + iter_shape[axis] - 1 ])
		axis = 1 # f
		vthrd_range = iter_shape[axis] * thrd_shape[axis]
		for vi in range(vthrd_shape[axis]):
			xyranges_d[1].append([(vi * vthrd_range) // op_para['channel_multiplier'], 
								(vi * vthrd_range + iter_shape[axis] - 1) // op_para['channel_multiplier']])
			xyranges_k[0].append([vi * vthrd_range, vi * vthrd_range + iter_shape[axis] - 1 ])
		axis = 2 # y
		xyranges_d[2] = set()
		vthrd_range = iter_shape[axis] * thrd_shape[axis]
		for vi in range(vthrd_shape[axis]):
			# xyranges_d[2].append([conv_inputIdex(vi * vthrd_range, 0, stride[0], dilation[0]), 
			# 	conv_inputIdex(vi * vthrd_range + iter_shape[axis] - 1, reduc_shape[0] - 1, stride[0], dilation[0])])
			xyranges_d[2].update(get_conv_InputIdxSet([vi * vthrd_range, vi * vthrd_range + iter_shape[axis]], \
				[0, reduc_shape[0]], stride[0], dilation[0]))
		axis = 3 # x
		xyranges_d[3] = set()
		vthrd_range = iter_shape[axis] * thrd_shape[axis]
		for vi in range(vthrd_shape[axis]):
			# xyranges_d[3].append([conv_inputIdex(vi * vthrd_range, 0, stride[1], dilation[1]), 
			# 	conv_inputIdex(vi * vthrd_range + iter_shape[axis] - 1, reduc_shape[1] - 1, stride[1], dilation[1])])			
			xyranges_d[3].update(get_conv_InputIdxSet([vi * vthrd_range, vi * vthrd_range + iter_shape[axis]], \
				[0, reduc_shape[1]], stride[1], dilation[1]))
		xyranges_k[1].append([0, reduc_shape[0]-1])
		xyranges_k[2].append([0, reduc_shape[1]-1])
		# get the amount of data to load
		# ret = area_size(xyranges_d) + area_size(xyranges_k)
		ret = [area_size(xyranges_d), area_size(xyranges_k)]
	elif op_para['op_type'] == "transpose_conv2d":
		# stride, dilation = op_para['stride'], op_para['dilation']
		iter_shape = [blk_shape[i] // (thrd_shape[i] * vthrd_shape[i]) for i in range(len(blk_shape))]
		xyranges_d = [list() for i in range(4)]
		xyranges_k = [list() for i in range(4)]
		axis = 0 # n 
		vthrd_range = iter_shape[axis] * thrd_shape[axis]
		for vi in range(vthrd_shape[axis]):
			xyranges_d[0].append([vi * vthrd_range, vi * vthrd_range + iter_shape[axis] - 1 ])
		axis = 1 # f
		vthrd_range = iter_shape[axis] * thrd_shape[axis]
		for vi in range(vthrd_shape[axis]):
			xyranges_k[0].append([vi * vthrd_range, vi * vthrd_range + iter_shape[axis] - 1 ])
		axis = 2 # y
		xyranges_d[2] = set()
		vthrd_range = iter_shape[axis] * thrd_shape[axis]
		for vi in range(vthrd_shape[axis]):
			# xyranges_d[2].append([conv_inputIdex(vi * vthrd_range, 0, 1, 1), 
			# 	conv_inputIdex(vi * vthrd_range + iter_shape[axis] - 1, reduc_shape[1] - 1, 1, 1)])
			xyranges_d[2].update(get_conv_InputIdxSet([vi * vthrd_range, vi * vthrd_range + iter_shape[axis]], \
				[0, reduc_shape[1]], 1, 1))
		axis = 3 # x
		xyranges_d[3] = set()
		vthrd_range = iter_shape[axis] * thrd_shape[axis]
		for vi in range(vthrd_shape[axis]):
			# xyranges_d[3].append([conv_inputIdex(vi * vthrd_range, 0, 1, 1), 
			# 	conv_inputIdex(vi * vthrd_range + iter_shape[axis] - 1, reduc_shape[2] - 1, 1, 1)])			
			xyranges_d[3].update(get_conv_InputIdxSet([vi * vthrd_range, vi * vthrd_range + iter_shape[axis]], \
				[0, reduc_shape[2]], 1, 1))
		xyranges_d[1].append([0, reduc_shape[0] - 1])
		xyranges_k[1].append([0, reduc_shape[0]-1])
		xyranges_k[2].append([0, reduc_shape[1]-1])
		xyranges_k[3].append([0, reduc_shape[2]-1])
		# get the amount of data to load
		# ret = area_size(xyranges_d) + area_size(xyranges_k)
		ret = [area_size(xyranges_d), area_size(xyranges_k)]
	elif op_para['op_type'] == "conv2d_capsule":
		stride = op_para['stride']
		iter_shape = [blk_shape[i] // (thrd_shape[i] * vthrd_shape[i]) for i in range(len(blk_shape))]
		xyranges_d = [list() for i in range(6)]
		xyranges_k = [list() for i in range(6)]
		axis = 0 # n 
		vthrd_range = iter_shape[axis] * thrd_shape[axis]
		for vi in range(vthrd_shape[axis]):
			xyranges_d[0].append([vi * vthrd_range, vi * vthrd_range + iter_shape[axis] - 1 ])
		axis = 1 # y
		xyranges_d[1] = set()
		vthrd_range = iter_shape[axis] * thrd_shape[axis]
		for vi in range(vthrd_shape[axis]):
			# xyranges_d[1].append([conv_inputIdex(vi * vthrd_range, 0, stride, 1), 
			# 	conv_inputIdex(vi * vthrd_range + iter_shape[axis] - 1, reduc_shape[0] - 1, stride, 1)])
			xyranges_d[1].update(get_conv_InputIdxSet([vi * vthrd_range, vi * vthrd_range + iter_shape[axis]], \
				[0, reduc_shape[0]], stride, 1))
		axis = 2 # x
		xyranges_d[2] = set()
		vthrd_range = iter_shape[axis] * thrd_shape[axis]
		for vi in range(vthrd_shape[axis]):
			# xyranges_d[2].append([conv_inputIdex(vi * vthrd_range, 0, stride, 1), 
			# 	conv_inputIdex(vi * vthrd_range + iter_shape[axis] - 1, reduc_shape[1] - 1, stride, 1)])			
			xyranges_d[2].update(get_conv_InputIdxSet([vi * vthrd_range, vi * vthrd_range + iter_shape[axis]], \
				[0, reduc_shape[1]], stride, 1))
		axis = 3 # cap_i 
		vthrd_range = iter_shape[axis] * thrd_shape[axis]
		for vi in range(vthrd_shape[axis]):
			xyranges_d[3].append([vi * vthrd_range, vi * vthrd_range + iter_shape[axis] - 1 ])
		axis = 4 # cap_j 
		vthrd_range = iter_shape[axis] * thrd_shape[axis]
		for vi in range(vthrd_shape[axis]):
			xyranges_k[3].append([vi * vthrd_range, vi * vthrd_range + iter_shape[axis] - 1 ])
		axis = 5 # f
		vthrd_range = iter_shape[axis] * thrd_shape[axis]
		for vi in range(vthrd_shape[axis]):
			xyranges_k[5].append([vi * vthrd_range, vi * vthrd_range + iter_shape[axis] - 1 ])
		xyranges_d[4].append([0, reduc_shape[2] - 1])
		xyranges_d[5].append([0, reduc_shape[3] - 1])
		xyranges_k[0].append([0, reduc_shape[0]-1])
		xyranges_k[1].append([0, reduc_shape[1]-1])
		xyranges_k[2].append([0, reduc_shape[2]-1])
		xyranges_k[4].append([0, reduc_shape[3]-1])
		# get the amount of data to load
		# ret = area_size(xyranges_d) + area_size(xyranges_k)
		ret = [area_size(xyranges_d), area_size(xyranges_k)]		
	elif op_para['op_type'] == "conv1d":
		stride, dilation = op_para['stride'], op_para['dilation']
		iter_shape = [blk_shape[i] // (thrd_shape[i] * vthrd_shape[i]) for i in range(len(blk_shape))]
		xyranges_d = [list() for i in range(3)]
		xyranges_k = [list() for i in range(3)]
		axis = 0 # n 
		vthrd_range = iter_shape[axis] * thrd_shape[axis]
		for vi in range(vthrd_shape[axis]):
			xyranges_d[0].append([vi * vthrd_range, vi * vthrd_range + iter_shape[axis] - 1 ])
		axis = 1 # f
		vthrd_range = iter_shape[axis] * thrd_shape[axis]
		for vi in range(vthrd_shape[axis]):
			xyranges_k[0].append([vi * vthrd_range, vi * vthrd_range + iter_shape[axis] - 1 ])
		axis = 2 # x
		xyranges_d[2] = set()
		vthrd_range = iter_shape[axis] * thrd_shape[axis]
		for vi in range(vthrd_shape[axis]):
			# xyranges_d[2].append([conv_inputIdex(vi * vthrd_range, 0, stride[0], dilation[0]), 
			# 	conv_inputIdex(vi * vthrd_range + iter_shape[axis] - 1, reduc_shape[1] - 1, stride[0], dilation[0])])	
			xyranges_d[2].update(get_conv_InputIdxSet([vi * vthrd_range, vi * vthrd_range + iter_shape[axis]], \
				[0, reduc_shape[1]], stride[0], dilation[0]))
		xyranges_d[1].append([0, reduc_shape[0] - 1])
		xyranges_k[1].append([0, reduc_shape[0]-1])
		xyranges_k[2].append([0, reduc_shape[1]-1])
		# get the amount of data to load
		# ret = area_size(xyranges_d) + area_size(xyranges_k)
		ret = [area_size(xyranges_d), area_size(xyranges_k)]
	elif op_para['op_type'] == "conv3d":
		stride, dilation = op_para['stride'], op_para['dilation']
		iter_shape = [blk_shape[i] // (thrd_shape[i] * vthrd_shape[i]) for i in range(len(blk_shape))]
		xyranges_d = [list() for i in range(5)]
		xyranges_k = [list() for i in range(5)]
		axis = 0 # n 
		vthrd_range = iter_shape[axis] * thrd_shape[axis]
		for vi in range(vthrd_shape[axis]):
			xyranges_d[0].append([vi * vthrd_range, vi * vthrd_range + iter_shape[axis] - 1 ])
		axis = 1 # f
		vthrd_range = iter_shape[axis] * thrd_shape[axis]
		for vi in range(vthrd_shape[axis]):
			xyranges_k[0].append([vi * vthrd_range, vi * vthrd_range + iter_shape[axis] - 1 ])
		axis = 2 # z
		xyranges_d[2] = set()
		vthrd_range = iter_shape[axis] * thrd_shape[axis]
		for vi in range(vthrd_shape[axis]):
			# xyranges_d[2].append([conv_inputIdex(vi * vthrd_range, 0, stride[0], dilation[0]), 
			# 	conv_inputIdex(vi * vthrd_range + iter_shape[axis] - 1, reduc_shape[1] - 1, stride[0], dilation[0])])
			xyranges_d[2].update(get_conv_InputIdxSet([vi * vthrd_range, vi * vthrd_range + iter_shape[axis]], \
				[0, reduc_shape[1]], stride[0], dilation[0]))
		axis = 3 # y
		xyranges_d[3] = set()
		vthrd_range = iter_shape[axis] * thrd_shape[axis]
		for vi in range(vthrd_shape[axis]):
			# xyranges_d[3].append([conv_inputIdex(vi * vthrd_range, 0, stride[1], dilation[1]), 
			# 	conv_inputIdex(vi * vthrd_range + iter_shape[axis] - 1, reduc_shape[2] - 1, stride[1], dilation[1])])			
			xyranges_d[3].update(get_conv_InputIdxSet([vi * vthrd_range, vi * vthrd_range + iter_shape[axis]], \
				[0, reduc_shape[2]], stride[1], dilation[1]))
		axis = 4 # x
		xyranges_d[4] = set()
		vthrd_range = iter_shape[axis] * thrd_shape[axis]
		for vi in range(vthrd_shape[axis]):
			# xyranges_d[4].append([conv_inputIdex(vi * vthrd_range, 0, stride[2], dilation[2]), 
			# 	conv_inputIdex(vi * vthrd_range + iter_shape[axis] - 1, reduc_shape[3] - 1, stride[2], dilation[2])])		
			xyranges_d[4].update(get_conv_InputIdxSet([vi * vthrd_range, vi * vthrd_range + iter_shape[axis]], \
				[0, reduc_shape[3]], stride[2], dilation[2]))	
		xyranges_d[1].append([0, reduc_shape[0] - 1])
		xyranges_k[1].append([0, reduc_shape[0]-1])
		xyranges_k[2].append([0, reduc_shape[1]-1])
		xyranges_k[3].append([0, reduc_shape[2]-1])
		xyranges_k[4].append([0, reduc_shape[3]-1])
		# get the amount of data to load
		# ret = area_size(xyranges_d) + area_size(xyranges_k)
		ret = [area_size(xyranges_d), area_size(xyranges_k)]
	elif op_para['op_type'] == "batch_matmul":
		iter_shape = [blk_shape[i] // (thrd_shape[i] * vthrd_shape[i]) for i in range(len(blk_shape))]
		xyranges_A = [list() for i in range(3)]
		xyranges_B = [list() for i in range(3)]
		axis = 0 # n 
		for vi in range(vthrd_shape[axis]):
			xyranges_A[0].append([vi * iter_shape[axis] * thrd_shape[axis], vi * iter_shape[axis] * thrd_shape[axis] + iter_shape[axis] - 1 ] if (op_para['X_shape'][0] != 1) else [0, 0])
			xyranges_B[0].append([vi * iter_shape[axis] * thrd_shape[axis], vi * iter_shape[axis] * thrd_shape[axis] + iter_shape[axis] - 1 ] if (op_para['Y_shape'][0] != 1) else [0, 0])
		axis = 1 # y
		for vi in range(vthrd_shape[axis]):
			xyranges_A[1].append([vi * iter_shape[axis] * thrd_shape[axis], vi * iter_shape[axis] * thrd_shape[axis] + iter_shape[axis] - 1 ])
		axis = 2 # x
		for vi in range(vthrd_shape[axis]):
			xyranges_B[1].append([vi * iter_shape[axis] * thrd_shape[axis], vi * iter_shape[axis] * thrd_shape[axis] + iter_shape[axis] - 1 ])
		xyranges_A[2].append([0, reduc_shape[0]-1])
		xyranges_B[2].append([0, reduc_shape[0]-1])
		# get the amount of data to load
		# ret = area_size(xyranges_A) + area_size(xyranges_B)
		ret = [area_size(xyranges_A), area_size(xyranges_B)]
	elif op_para['op_type'] == "batch_matmul_noTrans":
		iter_shape = [blk_shape[i] // (thrd_shape[i] * vthrd_shape[i]) for i in range(len(blk_shape))]
		xyranges_A = [list() for i in range(3)]
		xyranges_B = [list() for i in range(3)]
		axis = 0 # n 
		for vi in range(vthrd_shape[axis]):
			xyranges_A[0].append([vi * iter_shape[axis] * thrd_shape[axis], vi * iter_shape[axis] * thrd_shape[axis] + iter_shape[axis] - 1 ] if (op_para['X_shape'][0] != 1) else [0, 0])
			xyranges_B[0].append([vi * iter_shape[axis] * thrd_shape[axis], vi * iter_shape[axis] * thrd_shape[axis] + iter_shape[axis] - 1 ] if (op_para['Y_shape'][0] != 1) else [0, 0])
		axis = 1 # y
		for vi in range(vthrd_shape[axis]):
			xyranges_A[1].append([vi * iter_shape[axis] * thrd_shape[axis], vi * iter_shape[axis] * thrd_shape[axis] + iter_shape[axis] - 1 ])
		axis = 2 # x
		for vi in range(vthrd_shape[axis]):
			xyranges_B[2].append([vi * iter_shape[axis] * thrd_shape[axis], vi * iter_shape[axis] * thrd_shape[axis] + iter_shape[axis] - 1 ])
		xyranges_A[2].append([0, reduc_shape[0]-1])
		xyranges_B[1].append([0, reduc_shape[0]-1])
		# get the amount of data to load
		# ret = area_size(xyranges_A) + area_size(xyranges_B)
		ret = [area_size(xyranges_A), area_size(xyranges_B)]
	elif op_para['op_type'] == "dense":
		iter_shape = [blk_shape[i] // (thrd_shape[i] * vthrd_shape[i]) for i in range(len(blk_shape))]
		xyranges_A = [list() for i in range(2)]
		xyranges_B = [list() for i in range(2)]
		axis = 0 # y
		for vi in range(vthrd_shape[axis]):
			xyranges_A[0].append([vi * iter_shape[axis] * thrd_shape[axis], vi * iter_shape[axis] * thrd_shape[axis] + iter_shape[axis] - 1 ])
		axis = 1 # x
		for vi in range(vthrd_shape[axis]):
			xyranges_B[0].append([vi * iter_shape[axis] * thrd_shape[axis], vi * iter_shape[axis] * thrd_shape[axis] + iter_shape[axis] - 1 ])
		xyranges_A[1].append([0, reduc_shape[0]-1])
		xyranges_B[1].append([0, reduc_shape[0]-1])
		# get the amount of data to load
		# ret = area_size(xyranges_A) + area_size(xyranges_B)
		ret = [area_size(xyranges_A), area_size(xyranges_B)]
	elif op_para['op_type'] == "conv2d_winograd":
		iter_shape = [blk_shape[i] // (thrd_shape[i] * vthrd_shape[i]) for i in range(len(blk_shape))]
		xyranges_A = [list() for i in range(4)]
		xyranges_B = [list() for i in range(4)]
		axis = 0 # n 
		for vi in range(vthrd_shape[axis]):
			xyranges_A[0].append([vi * iter_shape[axis] * thrd_shape[axis], vi * iter_shape[axis] * thrd_shape[axis] + iter_shape[axis] - 1 ])
			xyranges_B[0].append([vi * iter_shape[axis] * thrd_shape[axis], vi * iter_shape[axis] * thrd_shape[axis] + iter_shape[axis] - 1 ])
		axis = 1 # y
		for vi in range(vthrd_shape[axis]):
			xyranges_A[1].append([vi * iter_shape[axis] * thrd_shape[axis], vi * iter_shape[axis] * thrd_shape[axis] + iter_shape[axis] - 1 ])
			xyranges_B[1].append([vi * iter_shape[axis] * thrd_shape[axis], vi * iter_shape[axis] * thrd_shape[axis] + iter_shape[axis] - 1 ])
		axis = 2 # x
		for vi in range(vthrd_shape[axis]):
			xyranges_A[2].append([vi * iter_shape[axis] * thrd_shape[axis], vi * iter_shape[axis] * thrd_shape[axis] + iter_shape[axis] - 1 ])
		axis = 3 # z
		for vi in range(vthrd_shape[axis]):
			xyranges_B[2].append([vi * iter_shape[axis] * thrd_shape[axis], vi * iter_shape[axis] * thrd_shape[axis] + iter_shape[axis] - 1 ])
		xyranges_A[3].append([0, reduc_shape[0]-1])
		xyranges_B[3].append([0, reduc_shape[0]-1])
		# get the amount of data to load
		# ret = area_size(xyranges_A) + area_size(xyranges_B)
		ret = [area_size(xyranges_A), area_size(xyranges_B)]
	# 
	return ret








def cal_32B_trans_stG(blk_shape, thrd_shape, vthrd_shape, op_type, out_shape):
	'''
	Calculate a block needs how many memory transactions to store data from registers to global memory.
	According to the documatation, P100 cache global memory in L2 cache, and conducts 32-byte memory transactions,
	so we count the number of 32B transactions a given block shape, thrd_shape and vthrd_shape would request.
	Every time, a thread would request a word, which is 4 byte (4B) for float32 number.
	# 
	INPUT:
		blk_shape:
			list of int. The block shape.
		# thrd_size:
		# 	int. how many threads are there in this block.
		op_para:
			a dictionary of parameters, which are necessary for us to compute the amount of memory transactions, also INCLUDES op_type and op loop structure.
		simplify:
			This func is the simplified version already.
			bool. If True, then we use one block's #(cache_line request) to approximate the actual amount; else we count the sum over all blocks.
		out_shape: 
			list of int. The output tensor shape after replicating operators.
	# 
	RMK:
		We only calculate the number of cache lines assuming perfect vectorization of memory load
	'''
	def xy2idx(coordinates, glob_shape):
		''' 
			transform the coordinates in global mem to the unique 1 dimension id in the global memory env 
			coordinates: the coordinates in the global env, list of ints.
			glob_shape:	the extent of the global data space axes corr. to the coordinates, list of ints. THE EXTENTS are listed from outer axes to inner axes.
		'''
		idx = coordinates[0]
		for i in range(1, len(coordinates)):
			idx = idx * glob_shape[i] + coordinates[i]
		return idx
	# 
	def idx2xy(space_shape, idx):
		'''
			values are listed from outer axes to inner axes. 
			transform the index of a point to the coordinates.
		'''
		coordinates = [0 for i in range(len(space_shape))]
		for i in range(len(space_shape)-1, -1, -1):
			coordinates[i] = idx % space_shape[i]
			idx = idx // space_shape[i]
		return coordinates
	# 
	def nmem_req(coordinate_list, space_shape, cl_size):
		'''
			get the cost of warps loading once, i.e., the memory request number: the number of warps + max numer of bank conflicts on one bank (in terms of a warp) when all threads in the block are loading data.
			coordinate_list:
				list of list of coordinates, each coordinate is a list of ints, from outer to inner axes, in the space; each coordinate list is the address requests in a warp.
			space_shape:
				the shape of the space where coordinates are in, extents are listed from outer to inner axes.
			cl_size:
				the cache line size in terms of the number of f32 data in a transaction. 8 for Nvidia P100.
			RMK:
				When compute the number of store transactions, consider all warps, but for each warp, we only consider the the first data point the threads need to load:
				# Because (1) bank conflict only happens among threads in a warp;
				# (2), memory request should be served as a whole, so every time only one request can be served? (NOT SURE, JUST GUESS).
		'''
		# the total number of shared memory requests
		n_req = 0
		for warp_cords in coordinate_list:
			# store the different address requested by a warp in each bank
			req_set = set([xy2idx(xy, space_shape) // cl_size for xy in warp_cords])
			n_req = n_req + len(req_set)
		return n_req
	# 
	# we first get the number of points a thread need to load
	warp_size = 32 # the number of threads in a warp
	cl_size = 8 # the cache line size
	tot_n_cl = 0 # the total number of cache lines required
	# op_type = op_para['op_type']
	thrd_num = get_product(thrd_shape)
	if op_type in ["dense", "bmm", "bmm_nn", "conv2d"]:
		cord_list_O = list()
		# get the consecutive block a thread is responsible for
		iter_shape = list()
		for i in range(len(blk_shape)):
			iter_shape.append(blk_shape[i] // (vthrd_shape[i] * thrd_shape[i]))
		# get the data and kernel shape in shared memory
		# out_shape = blk_shape
		# we need to get the coordinates every warp requests
		warp_cords_O = None
		for ti in range(thrd_num):
			if ti % warp_size == 0:
				# store the old warp request addresses
				if warp_cords_O != None:
					cord_list_O.append(warp_cords_O)
				# new a list to store the coordinates a warp want to access
				warp_cords_O = list()
			thrd_cord = idx2xy(thrd_shape, ti)
			cord_O = [thrd_cord[i] * iter_shape[i] for i in range(len(blk_shape))]
			warp_cords_O.append(cord_O)
		cord_list_O.append(warp_cords_O)
		# tot_nbkcflct = tot_nbkcflct + nmem_req(cord_list_A, shape_A, bank_num)
		# tot_nbkcflct = tot_nbkcflct + nmem_req(cord_list_B, shape_B, bank_num)
		tot_n_cl = nmem_req(cord_list_O, out_shape, cl_size)		
		return tot_n_cl










def get_combinations(tot, groups):
	'''
	Get all combinations to divide tot into groups so that the product of values assigned to groups is tot.
	INPUT:
		num:
			int. The tot number of things to be dicided into groups
		groups:
			list of strings. The groups to partition tot into.
	OUTPUT:
		dictionary of list storing all possible value in groups.
	'''
	fact_dict = factorint(int(tot))
	# use itertools.combinations() to generate all combinations
	last_dict = dict()
	for group in groups:
		last_dict[group] = list()
		last_dict[group].append(1)
	# 
	for factor in fact_dict.keys():
		delta_dict = dict()
		for group in groups:
			delta_dict[group] = list()
		positions = list(range( 1, fact_dict[factor] + len(groups) ))
		for select_psts in combinations(positions, len(groups) - 1):
			amounts = sorted(select_psts)
			for i in range( len( last_dict[groups[0]] ) ):
				# print(fact_dict, fact, f_i)
				# iterate over all partial combinations 
				for f_i in range(len(groups)):
					group = groups[f_i]
					if f_i == 0:
						amount = amounts[f_i] - 1
					elif (f_i < len(groups) - 1):
						amount = amounts[f_i] - amounts[f_i - 1] - 1
					else:
						amount = fact_dict[factor] + len(groups) - amounts[f_i - 1] - 1
					delta_dict[group].append( last_dict[group][i] * (factor ** amount ) )
		last_dict = delta_dict
	# Now last_dict stores all the possible delta changes on the cand_features
	return last_dict







def dict2list(from_d, keys):
	''' 
		Transform dictionary of list to list of list. E.g., transform {'a':[1, 2], 'b':[3, 4]} to [[1, 3], [2, 4]], where keys = ['a', 'b']
		keys:
			list of strings, the data in different keys are listed in the order of keys.
		RMK:
			Every key must have a list value of the same length.
	'''
	size = None
	for key in keys:
		if size == None:
			size = len(from_d[key])
		else:
			if size != len(from_d[key]):
				assert False, "the dictionary does not have the same length on each key"
	ret = list()
	for i in range(size):
		tmp = list()
		for j in range(len(keys)):
			tmp.append(from_d[keys[j]][i])
		ret.append(tmp)
	return ret




