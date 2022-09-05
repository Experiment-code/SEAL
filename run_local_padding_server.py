'''
This file contains the scripts 
to run schedules with local padding in DietCode's environment
'''

import filecmp
from flaky import flaky
import logging

logger = logging.getLogger(__name__)

from shared import CUDAContext, dietcode_decor, NoLocalPadding, tolerance

from ops.shared.utils import get_time_evaluator_results_rpc_wrapper



logger = logging.getLogger(__name__)

from shared import CUDAContext, dietcode_decor, NoLocalPadding, tolerance

from ops.shared.utils import get_time_evaluator_results_rpc_wrapper


import os
import numpy as np

import tvm
from tvm import te, auto_scheduler, topi
from tvm.topi.testing import conv2d_nchw_python, dense

import argparse




def get_product(elements):
	'''
		Get the product of the elements. 
		INPUT:	elements: list of ints.
	'''
	product = 1
	for i in elements:
		product = product * i
	return product





@flaky(max_runs=3)
@dietcode_decor
def test_my_run_local_padding(pysched_lib, pysched_lib_str):
	# from codegen import dense_expr1_pysched
	from ops.dense.fixture import Dense, cuBLASDenseFixture
	# 
	op_type = 'dense'
	tsps = pysched_lib.get_task_shapes() # list
	selected_msps = pysched_lib.get_selected_msps() # dict()
	dispatch_dict = pysched_lib.get_dispatch_dict()
	selected_msp_scheds = pysched_lib.get_selected_msp_scheds()
	# 
	# THE ORIGIANL MEASUREMENT CODE FROM DIETCODE CANNOT RUN
	# for task_i, tsp in enumerate(tsps):
	# 	mick_i = dispatch_dict[task_i][1]
	# 	sched = selected_msp_scheds[mick_i]
	# 	GFLOPs = 2 *get_product(tsp) / 1e9
	# 	wkl_func_args = (tsp[0], tsp[2], tsp[1])
	# 	cublas_fixture = cuBLASDenseFixture(*wkl_func_args)
	# 	local_padding_perf_results = get_time_evaluator_results_rpc_wrapper(
	#                                      wkl_func=Dense,
	#                                      wkl_func_args=tsp,
	#                                      sched_func_or_str=sched,
	#                                      fixture=cublas_fixture,
	#                                      print_kernel=False,
	#                                      log_kernel_filename="temp_workspace.log",
	#                                      verify_correctness=True
	#                                  )
	# 	local_padding_gflops = GFLOPs / np.average(local_padding_perf_results)
	# 	logger.info(f"Local Padding: {local_padding_tflops} (GFLOPS)   {local_padding_tflops} (seconds)")
	# 
	# 
	# use our own code to measure latency---------------
	cost_dict = dict()
	for task_i, tsp in enumerate(tsps):
		mick_i = dispatch_dict[task_i][1]
		sched = selected_msp_scheds[mick_i]
		msp = selected_msps[mick_i]
		# GFLOPs = 2 *get_product(tsp) / 1e9
		wkl_func_args = (tsp[0], tsp[2], tsp[1])
		# 
		tensor_args = Dense(*wkl_func_args) 
		s = te.create_schedule(tensor_args[-1].op)
		sched(*tensor_args, s)
		kernel = tvm.build(s, tensor_args, target=tvm.target.Target("cuda"))
		# 
		# prepare data and check correctness
		data_np = np.random.uniform(size=(tsp[0], tsp[2])).astype(np.float32)
		weight_np = np.random.uniform(size=(tsp[1], tsp[2])).astype(np.float32)
		# out_np = dense(data_np, weight_np, None)
		out_np = np.random.uniform(size=(tsp[0], tsp[1])).astype(np.float32)
		# 
		dev = tvm.cuda()
		data_tvm = tvm.nd.array(data_np, device=dev)
		weight_tvm = tvm.nd.array(weight_np, device=dev)
		out_tvm = tvm.nd.empty(out_np.shape, device=dev)
		kernel(data_tvm, weight_tvm, out_tvm)
		# 
		# check results
		# np.testing.assert_allclose(out_np, out_tvm.numpy(), rtol=1e-3)
		# evaluate execution time
		# first warmup
		warmup_evaluator = kernel.time_evaluator(kernel.entry_name, dev, number=300, repeat=1, min_repeat_ms=300)
		warmup_evaluator(data_tvm, weight_tvm, out_tvm)
		time_evaluator = kernel.time_evaluator(kernel.entry_name, dev,
					number=100, repeat=10, min_repeat_ms=100)
		our_cost = np.average(time_evaluator(data_tvm, weight_tvm, out_tvm).results)
		# logger.info(f"Local Padding: {our_cost} (seconds)")
		print(f"msp: {msp};  the avg run time of our kernel: {our_cost}")
		# ---------------------------------------------
		# run Vendor library
		X = te.placeholder((tsp[0], tsp[2]), name='X')
		W = te.placeholder((tsp[1], tsp[2]), name='W')
		Y = tvm.contrib.cublas.matmul(X, W, transa=False, transb=True)
		tensor_args_cublas = [X, W, Y]
		sched_cublas = te.create_schedule(Y.op)
		cublas_kernel = tvm.build(sched_cublas, [X, W, Y], target=tvm.target.Target("cuda"))
		out_tvm = tvm.nd.empty(out_np.shape, device=dev)
		cublas_kernel(data_tvm, weight_tvm, out_tvm)
		# check results of cublas
		# np.testing.assert_allclose(out_np, out_tvm.numpy(), rtol=1e-3)
		warmup_evaluator = cublas_kernel.time_evaluator(cublas_kernel.entry_name, dev, number=300, repeat=1, min_repeat_ms=300)
		warmup_evaluator(data_tvm, weight_tvm, out_tvm)
		time_evaluator = cublas_kernel.time_evaluator(cublas_kernel.entry_name, dev,
					number=100, repeat=10, min_repeat_ms=100)
		vendor_cost = np.average(time_evaluator(data_tvm, weight_tvm, out_tvm).results)
		# logger.info(f"CUBLAS: {vendor_cost} (seconds)")
		print(f"msp: {msp};  the avg run time of vendor kernel: {vendor_cost}")
		# 
		cost_dict[(task_i,tuple(tsp))] = (msp, our_cost, vendor_cost)
	with open(f'measured_cost_hub/cost_{pysched_lib_str}.csv', 'a') as f:
		f.write("start a new measure:------------------------\n")
		f.write(f'ours_localPad;vendor\n')
		for k, v in cost_dict.items():
			f.write(f'{k};{v[0]};{v[1]};{v[2]}\n')
		f.write(f"cost_dict = {cost_dict}\n")
		f.write(f"tot_cost = {sum([v[1]for v in cost_dict.values()])}\n")






def test_my_run_local_padding_MeasurePairs(measure_pairs, tsps, selected_msps, selected_msp_scheds, pysched_lib_str):
	# from codegen import dense_expr1_pysched
	from ops.dense.fixture import Dense, cuBLASDenseFixture
	# 
	op_type = 'dense'
	# tsps = pysched_lib.get_task_shapes() # list
	# selected_msps = pysched_lib.get_selected_msps() # dict()
	# dispatch_dict = pysched_lib.get_dispatch_dict()
	# selected_msp_scheds = pysched_lib.get_selected_msp_scheds()
	# 
	# use our own code to measure latency---------------
	cost_dict = dict()
	for task_i, mick_i in measure_pairs:
		# mick_i = dispatch_dict[task_i][1]
		sched = selected_msp_scheds[mick_i]
		msp = selected_msps[mick_i]
		tsp = tsps[task_i]
		# GFLOPs = 2 *get_product(tsp) / 1e9
		wkl_func_args = (tsp[0], tsp[2], tsp[1])
		# 
		tensor_args = Dense(*wkl_func_args) 
		s = te.create_schedule(tensor_args[-1].op)
		sched(*tensor_args, s)
		kernel = tvm.build(s, tensor_args, target=tvm.target.Target("cuda"))
		# 
		# prepare data and check correctness
		data_np = np.random.uniform(size=(tsp[0], tsp[2])).astype(np.float32)
		weight_np = np.random.uniform(size=(tsp[1], tsp[2])).astype(np.float32)
		# out_np = dense(data_np, weight_np, None)
		out_np = np.random.uniform(size=(tsp[0], tsp[1])).astype(np.float32)
		# 
		dev = tvm.cuda()
		data_tvm = tvm.nd.array(data_np, device=dev)
		weight_tvm = tvm.nd.array(weight_np, device=dev)
		out_tvm = tvm.nd.empty(out_np.shape, device=dev)
		kernel(data_tvm, weight_tvm, out_tvm)
		# 
		# check results
		# np.testing.assert_allclose(out_np, out_tvm.numpy(), rtol=1e-3)
		# evaluate execution time
		# first warmup
		warmup_evaluator = kernel.time_evaluator(kernel.entry_name, dev, number=300, repeat=1, min_repeat_ms=300)
		warmup_evaluator(data_tvm, weight_tvm, out_tvm)
		time_evaluator = kernel.time_evaluator(kernel.entry_name, dev,
					number=100, repeat=10, min_repeat_ms=100)
		our_cost = np.average(time_evaluator(data_tvm, weight_tvm, out_tvm).results)
		# logger.info(f"Local Padding: {our_cost} (seconds)")
		print(f"msp: {msp};  the avg run time of our kernel: {our_cost}")
		# ---------------------------------------------
		# run Vendor library
		X = te.placeholder((tsp[0], tsp[2]), name='X')
		W = te.placeholder((tsp[1], tsp[2]), name='W')
		Y = tvm.contrib.cublas.matmul(X, W, transa=False, transb=True)
		tensor_args_cublas = [X, W, Y]
		sched_cublas = te.create_schedule(Y.op)
		cublas_kernel = tvm.build(sched_cublas, [X, W, Y], target=tvm.target.Target("cuda"))
		out_tvm = tvm.nd.empty(out_np.shape, device=dev)
		cublas_kernel(data_tvm, weight_tvm, out_tvm)
		# check results of cublas
		# np.testing.assert_allclose(out_np, out_tvm.numpy(), rtol=1e-3)
		warmup_evaluator = cublas_kernel.time_evaluator(cublas_kernel.entry_name, dev, number=300, repeat=1, min_repeat_ms=300)
		warmup_evaluator(data_tvm, weight_tvm, out_tvm)
		time_evaluator = cublas_kernel.time_evaluator(cublas_kernel.entry_name, dev,
					number=100, repeat=10, min_repeat_ms=100)
		vendor_cost = np.average(time_evaluator(data_tvm, weight_tvm, out_tvm).results)
		# logger.info(f"CUBLAS: {vendor_cost} (seconds)")
		print(f"msp: {msp};  the avg run time of vendor kernel: {vendor_cost}")
		# 
		cost_dict[(task_i,mick_i)] = (msp, our_cost, vendor_cost,tsp)
	with open(f'measured_cost_hub/cost_{pysched_lib_str}.csv', 'a') as f:
		f.write("start a new measure:------------------------\n")
		f.write(f'ours_localPad;vendor\n')
		for k, v in cost_dict.items():
			f.write(f'{k};{v[0]};{v[1]};{v[2]};{v[3]}\n')
		f.write(f"cost_dict = {cost_dict}\n")
		f.write(f"tot_cost = {sum([v[1]for v in cost_dict.values()])}\n")







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







@flaky(max_runs=3)
@dietcode_decor
def test_my_run_local_padding_bmm(pysched_lib, pysched_lib_str, op_type):
	# from codegen import dense_expr1_pysched
	# from ops.dense.fixture import Dense, cuBLASDenseFixture
	# 
	# op_type = 'bmm'
	tsps = pysched_lib.get_task_shapes() # list
	selected_msps = pysched_lib.get_selected_msps() # dict()
	dispatch_dict = pysched_lib.get_dispatch_dict()
	selected_msp_scheds = pysched_lib.get_selected_msp_scheds()
	# 
	# use our own code to measure latency---------------
	cost_dict = dict()
	for task_i, tsp in enumerate(tsps):
		mick_i = dispatch_dict[task_i][1]
		sched = selected_msp_scheds[mick_i]
		msp = selected_msps[mick_i]
		# GFLOPs = 2 *get_product(tsp) / 1e9
		wkl_func_args = (tsp[0], tsp[2], tsp[1])
		if op_type == 'bmm':
			wkl_func_args = ((tsp[0], tsp[1], tsp[3]), (tsp[0], tsp[2], tsp[3]))
		elif op_type == 'bmm_nn':
			wkl_func_args = ((tsp[0], tsp[1], tsp[3]), (tsp[0], tsp[3], tsp[2]))
		op_func = batch_matmul
		if op_type == 'bmm_nn':
			op_func = batch_matmul_noTrans
		# 
		tensor_args = op_func(*wkl_func_args) 
		s = te.create_schedule(tensor_args[-1].op)
		sched(*tensor_args, s)
		kernel = tvm.build(s, tensor_args, target=tvm.target.Target("cuda"))
		# 
		# prepare data and check correctness
		data_np = np.random.uniform(size=wkl_func_args[0]).astype(np.float32)
		weight_np = np.random.uniform(size=wkl_func_args[1]).astype(np.float32)
		# out_np = None
		# if op_type == 'bmm':
		# 	out_np = tvm.topi.testing.batch_matmul(data_np, weight_np, out_dtype=None, trans_x=False, trans_y=True)
		# else:
		# 	out_np = tvm.topi.testing.batch_matmul(data_np, weight_np, out_dtype=None, trans_x=False, trans_y=False)
		out_np = np.random.uniform(size=(tsp[0], tsp[1], tsp[2])).astype(np.float32)
		# 
		dev = tvm.cuda()
		data_tvm = tvm.nd.array(data_np, device=dev)
		weight_tvm = tvm.nd.array(weight_np, device=dev)
		out_tvm = tvm.nd.empty(out_np.shape, device=dev)
		kernel(data_tvm, weight_tvm, out_tvm)
		# 
		# check results
		# np.testing.assert_allclose(out_np, out_tvm.numpy(), rtol=1e-3)
		# evaluate execution time
		# first warmup
		warmup_evaluator = kernel.time_evaluator(kernel.entry_name, dev, number=300, repeat=1, min_repeat_ms=300)
		warmup_evaluator(data_tvm, weight_tvm, out_tvm)
		time_evaluator = kernel.time_evaluator(kernel.entry_name, dev,
					number=100, repeat=10, min_repeat_ms=100)
		our_cost = np.average(time_evaluator(data_tvm, weight_tvm, out_tvm).results)
		# logger.info(f"Local Padding: {our_cost} (seconds)")
		print(f"msp: {msp};  the avg run time of our kernel: {our_cost}")
		# ---------------------------------------------
		# run Vendor library
		X = te.placeholder(wkl_func_args[0], name='X')
		W = te.placeholder(wkl_func_args[1], name='W')
		Y = None
		if op_type == 'bmm':
			Y = tvm.contrib.cublas.batch_matmul(X, W, transa=False, transb=True, dtype=None)
		else:
			Y = tvm.contrib.cublas.batch_matmul(X, W, transa=False, transb=False, dtype=None)
		tensor_args_cublas = [X, W, Y]
		sched_cublas = te.create_schedule(Y.op)
		cublas_kernel = tvm.build(sched_cublas, [X, W, Y], target=tvm.target.Target("cuda"))
		out_tvm = tvm.nd.empty(out_np.shape, device=dev)
		cublas_kernel(data_tvm, weight_tvm, out_tvm)
		# check results of cublas
		# np.testing.assert_allclose(out_np, out_tvm.numpy(), rtol=1e-3)
		warmup_evaluator = cublas_kernel.time_evaluator(cublas_kernel.entry_name, dev, number=300, repeat=1, min_repeat_ms=300)
		warmup_evaluator(data_tvm, weight_tvm, out_tvm)
		time_evaluator = cublas_kernel.time_evaluator(cublas_kernel.entry_name, dev,
					number=100, repeat=10, min_repeat_ms=100)
		vendor_cost = np.average(time_evaluator(data_tvm, weight_tvm, out_tvm).results)
		# logger.info(f"CUBLAS: {vendor_cost} (seconds)")
		print(f"msp: {msp};  the avg run time of vendor kernel: {vendor_cost}")
		# 
		cost_dict[tuple(tsp)] = (msp, our_cost, vendor_cost)
	with open(f'measured_cost_hub/cost_{pysched_lib_str}.csv', 'a') as f:
		f.write("start a new measure:------------------------\n")
		f.write(f'ours_localPad;vendor\n')
		for k, v in cost_dict.items():
			f.write(f'{k};{v[0]};{v[1]};{v[2]}\n')
		f.write(f"cost_dict = {cost_dict}\n")
		f.write(f"tot_cost = {sum([v[1]for v in cost_dict.values()])}\n")








@auto_scheduler.register_workload
def conv2d_nchw(data_shape, kernel_shape, stride, padding, dilation_val, out_dtype):
	data = te.placeholder(data_shape, name="data")
	kernel = te.placeholder(kernel_shape, name="kernel")
	# bias = te.placeholder((1, CO, 1, 1), name="bias")
	conv = topi.nn.conv2d_nchw(data, kernel, stride, padding, dilation=dilation_val, out_dtype=out_dtype)
	# out = topi.nn.relu(conv + bias)
	return [data, kernel, conv]







@flaky(max_runs=3)
@dietcode_decor
def test_my_run_local_padding_conv2d(conv2d_pysched_lib, conv2d_pysched_lib_str):
	# from codegen import conv2d_expr43_pysched
	# 
	op_type = 'conv2d'
	tsps = conv2d_pysched_lib.get_task_shapes() # list
	selected_msps = conv2d_pysched_lib.get_selected_msps() # dict()
	dispatch_dict = conv2d_pysched_lib.get_dispatch_dict()
	selected_msp_scheds = conv2d_pysched_lib.get_selected_msp_scheds()
	selected_msp_config = conv2d_pysched_lib.get_selected_msp_config()
	# 
	# use our own code to measure latency---------------
	cost_dict = dict()
	for task_i, tsp in enumerate(tsps):
		mick_i = dispatch_dict[task_i][1]
		sched = selected_msp_scheds[mick_i]
		msp = selected_msps[mick_i]
		# GFLOPs = 2 *get_product(tsp[:7]) / 1e9
		n, c, h, w, rc, rh, rw, (sh, sw), padding, (dh, dw) = tsp
		wkl_func_args = ((n, rc, sh*(h-1)+dh*(rh-1)+1, sw*(w-1)+dw*(rw-1)+1), (c, rc, rh, rw), (sh, sw), padding, (dh, dw), "float32")
		# 
		tensor_args = conv2d_nchw(*wkl_func_args) 
		s = te.create_schedule(tensor_args[-1].op)
		sched(*tensor_args, tensor_args[-1].op.input_tensors[0], s)
		# 
		kernel = tvm.build(s, tensor_args, target=tvm.target.Target("cuda"))
		# 
		# prepare data and check correctness
		data_np = np.random.uniform(size=(n, rc, sh*(h-1)+dh*(rh-1)+1, sw*(w-1)+dw*(rw-1)+1)).astype(np.float32)
		weight_np = np.random.uniform(size=(c, rc, rh, rw)).astype(np.float32)
		# out_np = conv2d_nchw_python(data_np, weight_np, (sh, sw), padding)
		out_np = np.random.uniform(size=(n, c, h, w)).astype(np.float32)
		# 
		dev = tvm.cuda()
		data_tvm = tvm.nd.array(data_np, device=dev)
		weight_tvm = tvm.nd.array(weight_np, device=dev)
		out_tvm = tvm.nd.empty(out_np.shape, device=dev)
		kernel(data_tvm, weight_tvm, out_tvm)
		# 
		# check results
		# np.testing.assert_allclose(out_np, out_tvm.numpy(), rtol=1e-3)
		# evaluate execution time
		# first warmup
		# warmup_evaluator = kernel.time_evaluator(kernel.entry_name, dev, number=300, repeat=1, min_repeat_ms=300)
		warmup_evaluator = kernel.time_evaluator(kernel.entry_name, dev, number=3, repeat=1, min_repeat_ms=300)
		warmup_evaluator(data_tvm, weight_tvm, out_tvm)
		# time_evaluator = kernel.time_evaluator(kernel.entry_name, dev,
		# 			number=100, repeat=10, min_repeat_ms=100)
		time_evaluator = kernel.time_evaluator(kernel.entry_name, dev,
					number=3, repeat=1, min_repeat_ms=300)
		our_cost = np.average(time_evaluator(data_tvm, weight_tvm, out_tvm).results)
		# logger.info(f"Local Padding: {our_cost} (seconds)")
		print(f"msp: {msp};  the avg run time of our kernel: {our_cost}")
		cost_dict[tuple(tsp)] = (msp, our_cost,)
		# ---------------------------------------------
		# run Vendor library
		# X = te.placeholder((n, rc, sh*(h-1)+dh*(rh-1)+1, sw*(w-1)+dw*(rw-1)+1), name='X')
		# W = te.placeholder((c, rc, rh, rw), name='W')
		# Y = tvm.contrib.cudnn.conv_forward(X, W, padding, (sh, sw), (dh, dw), 0, 0, -1, None, groups=1)
		# tensor_args_cudnn = [X, W, Y]
		# sched_cudnn = te.create_schedule(Y.op)
		# cudnn_kernel = tvm.build(sched_cudnn, [X, W, Y], target=tvm.target.Target("cuda"))
		# out_tvm = tvm.nd.empty(out_np.shape, device=dev)
		# cudnn_kernel(data_tvm, weight_tvm, out_tvm)
		# # check results of cudnn
		# np.testing.assert_allclose(out_np, out_tvm.numpy(), rtol=1e-3)
		# warmup_evaluator = cudnn_kernel.time_evaluator(cudnn_kernel.entry_name, dev, number=300, repeat=1, min_repeat_ms=300)
		# warmup_evaluator(data_tvm, weight_tvm, out_tvm)
		# time_evaluator = cudnn_kernel.time_evaluator(cudnn_kernel.entry_name, dev,
		# 			number=100, repeat=10, min_repeat_ms=100)
		# vendor_cost = np.average(time_evaluator(data_tvm, weight_tvm, out_tvm).results)
		# # logger.info(f"CUDNN: {vendor_cost} (seconds)")
		# print(f"the avg run time of vendor kernel: {vendor_cost}")
		# # 
		# # cost_dict[tuple(tsp)] = (our_cost, vendor_cost)
		# cost_dict[tuple(tsp)] = (vendor_cost,)
		# ---------------------------------------------Use another method to run CUDNN backend
		# tensor_args_cudnn2 = conv2d_nchw(*wkl_func_args)
		# sched_cudnn2 = te.create_schedule(tensor_args_cudnn2[-1].op)
		# cudnn_kernel2 = tvm.build(sched_cudnn2, tensor_args_cudnn2, target=tvm.target.Target("cuda -libs=cudnn"))
		# dev2 = tvm.device("cuda -libs=cudnn", 0)
		# data_tvm2 = tvm.nd.array(data_np, device=dev2)
		# weight_tvm2 = tvm.nd.array(weight_np, device=dev2)
		# out_tvm2 = tvm.nd.empty(out_np.shape, device=dev2)
		# cudnn_kernel2(data_tvm2, weight_tvm2, out_tvm2)
		# # check results of cudnn
		# np.testing.assert_allclose(out_np, out_tvm.numpy(), rtol=1e-3)
		# warmup_evaluator = cudnn_kernel2.time_evaluator(cudnn_kernel2.entry_name, dev2, number=300, repeat=1, min_repeat_ms=300)
		# warmup_evaluator(data_tvm2, weight_tvm2, out_tvm2)
		# time_evaluator = cudnn_kernel2.time_evaluator(cudnn_kernel2.entry_name, dev2,
		# 			number=100, repeat=10, min_repeat_ms=100)
		# vendor_cost2 = np.average(time_evaluator(data_tvm2, weight_tvm2, out_tvm2).results)
		# # logger.info(f"CUDNN2: {vendor_cost2} (seconds)")
		# print(f"the avg run time of vendor kernel: {vendor_cost2}")
		# # 
		# cost_dict[tuple(tsp)] = (our_cost, vendor_cost, vendor_cost2)
	# 
	with open(f'measured_cost_hub/cost_{conv2d_pysched_lib_str}.csv', 'a') as f:
		f.write("start a new measure:------------------------\n")
		f.write(f'ours_localPad;vendor;vendor2\n')
		print(cost_dict)
		for k, v in cost_dict.items():
			f.write(f'{k};{v[0]};{v[1]}\n')
		f.write(f"cost_dict = {cost_dict}\n")
		f.write(f"tot_cost = {sum([v[1]for v in cost_dict.values()])}\n")










if __name__ == "__main__":
	from codegen import dense_exprR11_pysched
	test_my_run_local_padding(dense_exprR11_pysched, "dense_exprR11_pysched")

