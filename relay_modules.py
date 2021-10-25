import tvm
from tvm import relay, autotvm
#from tvm.contrib import graph_runtime
from tvm.contrib import graph_executor
from tvm.relay.testing import layers
from tvm.relay.testing.init import create_workload
import numpy as np
import time

def timestamp(prefix="NONE", mod=False):
    if mod:
        prefix = prefix + str( time.clock_gettime(time.CLOCK_REALTIME)*1000 )
        print(prefix)
    else:
        return time.clock_gettime(time.CLOCK_REALTIME)*1000

dtype="float32"
##################################################################################
###Conv Block in Relay                                                           #
##################################################################################
def Relay_SelectActivation(input_data, act='ReLU6'):
    assert act in ['ReLU','ReLU6']
    if act == 'ReLU':
        return relay.nn.relu(input_data)
    elif act == 'ReLU6':
        return relay.clip(input_data, a_min=0., a_max=6.)


def Relay_ConvBnAct(config_key, act='ReLU6', conv_type='stem', input_data=None):
    assert conv_type in ['stem','head']
    if conv_type == 'stem':
        in_s = 224
    elif conv_type == 'head':
        in_s = 7

    in_c = config_key[0]
    out_c = config_key[1]
    k = config_key[2]
    s = config_key[3]
    if input_data==None:
        input_data = relay.var("data", shape=(1, in_c, in_s, in_s), dtype=dtype)
    conv_weight = relay.var(conv_type+".conv_weight", shape=(out_c, in_c, k, k), dtype=dtype)

    conv = layers.conv2d(data=input_data, weight=conv_weight, channels=out_c,
                         strides=[s,s],padding=[k//2,k//2],kernel_size=[k,k],name=conv_type+".conv")
    bn = layers.batch_norm_infer(data=conv, name=conv_type+".bn")
    net = Relay_SelectActivation(bn, act=act)
    return net


def Relay_Linear(last_c, num_class=1000, biase=True, input_data=None):
    if input_data==None:
        input_data = relay.var("data", shape=(last_c,))
    linear_weight = relay.var("linear_weight", shape=(num_class, last_c))
    linear_bias = relay.var("linear_bias", shape=(num_class,))
    linear = relay.nn.dense(input_data, linear_weight)
    net = relay.add(linear, linear_bias)
    return net


def Relay_HeadConv_Linear(headconv_key, act='ReLU6', input_data=None):
    last_c = headconv_key[1]
    if input_data==None:
        head_conv = Relay_ConvBnAct(headconv_key, act=act, conv_type='head')
    else:
        head_conv = Relay_ConvBnAct(headconv_key, act=act, conv_type='head', input_data=input_data)

    avg = relay.nn.adaptive_avg_pool2d(head_conv, output_size=[1,1])
    flat = relay.nn.batch_flatten(avg)
    net = Relay_Linear(last_c, input_data=flat)
    return net


##################################################################################
###MBCOnv Block in Relay                                                         #
##################################################################################
def Relay_MBConvBlock(config_key, block_name, act='ReLU6', input_data=None):
    in_s = config_key[0]
    k = config_key[1]
    s = config_key[2]
    in_c = config_key[3]
    exp_c = config_key[4]
    out_c = config_key[5]

    if input_data==None:
        input_data = relay.var("data", shape=(1, in_c, in_s, in_s), dtype=dtype)
    residual_connection = (in_c == out_c and s == 1)

    #Point-wise1
    if in_c != exp_c:
        pw1_weight = relay.var(block_name + ".pw1_weight", shape=(exp_c, in_c, 1, 1), dtype=dtype)
        pw1 = layers.conv2d(data=input_data,weight=pw1_weight,channels=exp_c,
                            strides=[1,1],padding=[0,0],kernel_size=[1,1],name=block_name+".pw1")
        bn1 = layers.batch_norm_infer(data=pw1, name=block_name+".bn1")
        relu1 = Relay_SelectActivation(bn1, act=act)
    else:
        relu1 = input_data

    dw_weight = relay.var(block_name + ".dw_weight", shape=(exp_c, 1, k, k), dtype=dtype)
    pw2_weight = relay.var(block_name + ".pw2_weight", shape=(out_c, exp_c, 1, 1), dtype=dtype)

    #Depth-wise
    dw = layers.conv2d(data=relu1,weight=dw_weight,channels=exp_c,groups=exp_c,
                       kernel_size=[k,k],strides=[s,s],padding=[k//2,k//2],name=block_name+".dw")
    bn2 = layers.batch_norm_infer(data=dw, name=block_name+".bn2")
    relu2 = Relay_SelectActivation(bn2, act=act)

    #Point-wse2
    pw2 = layers.conv2d(data=relu2,weight=pw2_weight,channels=out_c,
                        kernel_size=[1,1],strides=[1,1],padding=[0,0],name=block_name+".pw2")
    bn3 = layers.batch_norm_infer(data=pw2, name=block_name+".bn3")

    if residual_connection:
        net = relay.add(input_data, bn3)
    else:
        net = bn3

    return net


def inverted_residual_block(input_data, block_name, k, s, in_c, exp_c, out_c, act='ReLU6'):
    use_res_connect = s == 1 and in_c == out_c
    #Point-wise
    if in_c != exp_c:
        pw1_weight = relay.var(block_name + "_pw1_weight", shape=(exp_c, in_c, 1, 1))
        pw1 = layers.conv2d(data=input_data,
                                weight=pw1_weight,
                                channels=exp_c,
                                kernel_size=[1,1],
                                strides=[1,1],
                                padding=[0,0],
                                name=block_name + "_pw1")
        bn1 = layers.batch_norm_infer(data=pw1,
                            name=block_name + "_bn1")
        relu1 = Relay_SelectActivation(bn1, act=act)
    else:
        relu1 = input_data

    dw_weight = relay.var(block_name + "_dw_weight", shape=(exp_c, 1, k, k))
    pw2_weight = relay.var(block_name + "_pw2_weight", shape=(out_c, exp_c, 1, 1))
    
    #Depth-wise
    dw = layers.conv2d(data=relu1,
                        weight=dw_weight,
                        channels=exp_c,
                        groups=exp_c,
                        kernel_size=[k,k],
                        strides=[s,s],
                        padding=[k//2, k//2],
                        name=block_name + "_dw")
    bn2 = layers.batch_norm_infer(data=dw,
                        name=block_name + "_bn2")
    relu2 = Relay_SelectActivation(bn2, act=act)

    #Point-wise
    pw2 = layers.conv2d(data=relu2,
                        weight=pw2_weight,
                        channels=out_c,
                        kernel_size=[1,1],
                        strides=[1,1],
                        padding=[0,0],
                        name=block_name + "_pw2")
    bn3 = layers.batch_norm_infer(data=pw2,
                        name=block_name + "_bn3")
    
    if use_res_connect:
        out = relay.add(input_data, bn3)
    else:
        out = bn3

    return out



##################################################################################
###TVM Runtime Build Utils                                                       #
##################################################################################
def build_module(module, args):
    tgt = tvm.target.create('llvm -mtriple=aarch64-linux-gnu')
    ctx = tvm.cpu()
    Func = relay.Function(args, module)
    mod = tvm.IRModule.from_expr(Func)
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(mod, tgt)
    module = graph_executor.GraphModule(lib["default"](ctx))
    return module

def get_workload(net):
    Func = relay.Function(relay.analysis.free_vars(net), net)
    return create_workload(Func)


def get_runtime_module(mod, params, opt_level=3, target='cpu'):
    assert target in ['cpu', 'mali', 'adreno', 'cuda', 'trt']
    config = None
    relay.backend.compile_engine.get().clear()

    if target=='trt':
        from tvm.relay.op.contrib.tensorrt import partition_for_tensorrt
        mod, config = partition_for_tensorrt(mod, params)
        config = {'relay.ext.tensorrt.options': config}

    with tvm.transform.PassContext(opt_level=opt_level, config=config):
        if target in ['cpu', 'cuda', 'trt']:
            if target == 'cpu':
                tgt = tvm.target.create('llvm -mtriple=aarch64-linux-gnu')
                ctx = tvm.cpu()
            else:
                tgt = tvm.target.cuda()
                ctx = tvm.gpu()
            #g, m, p = relay.build(mod, tgt, params=params)
            lib = relay.build(mod, tgt, params=params)
        else:
            tgt_host = tvm.target.create('llvm -mtriple=aarch64-linux-gnu')
            ctx = tvm.cl()
            if target == 'mali':
                tgt = tvm.target.create('opencl -device=mali')
            elif target == 'adreno':
                tgt = tvm.target.create('opencl')
            lib = relay.build(mod, tgt, tgt_host, params=params)

    module = graph_executor.GraphModule(lib["default"](ctx))
    return module


def get_runtime_module_tuned(mod, params, tuned_log, opt_level=3, target='cpu'):
    assert target in ['cpu', 'mali', 'adreno', 'cuda', 'trt']
    config = None
    relay.backend.compile_engine.get().clear()

    if target=='trt':
        from tvm.relay.op.contrib.tensorrt import partition_for_tensorrt
        mod, config = partition_for_tensorrt(mod, params)
        config = {'relay.ext.tensorrt.options': config}

    with autotvm.apply_history_best(tuned_log):
        with tvm.transform.PassContext(opt_level=opt_level, config=config):
            if target in ['cpu', 'cuda', 'trt']:
                if target == 'cpu':
                    tgt = tvm.target.create('llvm -mtriple=aarch64-linux-gnu')
                    ctx = tvm.cpu()
                else:
                    tgt = tvm.target.cuda()
                    ctx = tvm.gpu()
                #g, m, p = relay.build(mod, tgt, params=params)
                lib = relay.build(mod, tgt, params=params)
            else:
                tgt_host = tvm.target.create('llvm -mtriple=aarch64-linux-gnu')
                ctx = tvm.cl()
                if target == 'mali':
                    tgt = tvm.target.create('opencl -device=mali')
                elif target == 'adreno':
                    tgt = tvm.target.create('opencl')
                lib = relay.build(mod, tgt, tgt_host, params=params)

        module = graph_executor.GraphModule(lib["default"](ctx))
        return module




##################################################################################
###Get Runtime Modules                                                           #
##################################################################################
def get_mbconv(input_shape, k, s, in_c, exp_c, out_c, act, block_name):
    data = relay.var("data", shape=input_shape, dtype="float32")
    net = inverted_residual_block(data, block_name, k, s, in_c, exp_c, out_c, act)
    Func = relay.Function(relay.analysis.free_vars(net), net)
    return create_workload(Func)


def get_mbconvs(config_keys:list, act, block_name):
    net = None
    count = 0
    for config in config_keys:
        blk=block_name + str(count)
        net = Relay_MBConvBlock(config, blk, act, net)
        count += 1
    Func = relay.Function(relay.analysis.free_vars(net), net)
    return create_workload(Func)


def get_stemconv(config_key, act, block_name):
    net = Relay_ConvBnAct(config_key, act, conv_type='stem')
    Func = relay.Function(relay.analysis.free_vars(net), net)
    return create_workload(Func)


def get_headconv_linear(config_key, act, block_name):
    net = Relay_HeadConv_Linear(config_key, act)
    Func = relay.Function(relay.analysis.free_vars(net), net)
    return create_workload(Func)


def get_relay_sync(output_shape, sync_type, concat_ratio=None):
    if sync_type == 'add':
        lhs = relay.var("lhs", relay.TensorType(output_shape, "float32"))
        rhs = relay.var("rhs", relay.TensorType(output_shape, "float32"))
        lhs_shape = rhs_shape = output_shape
        mod = relay.add(lhs, rhs)
        sync_module = build_module(mod, [lhs,rhs])
    elif sync_type == 'concat':
        assert concat_ratio in [0.1, 0.2, 0.3, 0.4, 0.5]
        lhs_c = int(output_shape[1] * concat_ratio)
        if lhs_c % 2 != 0:
            lhs_c += 1
        rhs_c = output_shape[1] - lhs_c
        lhs_shape = (output_shape[0],) + (lhs_c,) + output_shape[2:]
        rhs_shape = (output_shape[0],) + (rhs_c,) + output_shape[2:]
        lhs = relay.var("lhs", relay.TensorType(lhs_shape, "float32"))
        rhs = relay.var("rhs", relay.TensorType(rhs_shape, "float32"))
        mod = relay.concatenate([lhs,rhs], axis=1)
        sync_module = build_module(mod, [lhs,rhs])
    return sync_module, lhs_shape, rhs_shape



##################################################################################
###Run Runtime Modules                                                           #
##################################################################################
def record_execution(module, input_shape, ctx, repeat=310, warmup=10):
    record = []
    for i in range(repeat):
        data = np.random.uniform(-1, 1, size=input_shape).astype("float32")
        module.set_input("data", data)
        start = timestamp()
        module.run()
        if 'cl' in str(ctx):# or 'gpu' in str(ctx):
            ctx.sync()
        end = timestamp()
        lat = end-start
        if i >= warmup:
            record.append(lat)
    return record
    #return list_to_stat(record)


def record_data_transfer_no_numpy(module, input_shape, ctx, repeat=110, warmup=10):
    h2d_record = []
    d2h_record = []
    for i in range(repeat):
        data = tvm.nd.array(np.random.uniform(-1, 1, size=input_shape).astype("float32"), ctx)
        h2d_start = timestamp()
        module.set_input("data", data)
        h2d_end = timestamp()
        module.run()
        d2h_start = timestamp()
        module.get_output(0)
        d2h_end = timestamp()
        if i >= warmup:
            h2d_record.append(h2d_end - h2d_start)
            d2h_record.append(d2h_end - d2h_start)
    return list_to_stat(h2d_record), list_to_stat(d2h_record)


def record_data_transfer_as_numpy(module, input_shape, ctx, repeat=110, warmup=10):
    h2d_record = []
    d2h_record = []
    for i in range(repeat):
        data = np.random.uniform(-1, 1, size=input_shape).astype("float32")
        h2d_start = timestamp()
        module.set_input("data", data)
        h2d_end = timestamp()
        module.run()
        d2h_start = timestamp()
        module.get_output(0).asnumpy()
        d2h_end = timestamp()
        if i >= warmup:
            h2d_record.append(h2d_end - h2d_start)
            d2h_record.append(d2h_end - d2h_start)
    return list_to_stat(h2d_record), list_to_stat(d2h_record)


def record_relay_sync(sync_module, lhs_shape, rhs_shape, ctx, repeat=110, warmup=10):
    record = dict()
    record["set"] = []
    record["run"] = []
    record["get"] = []
    for i in range(repeat):
        lhs = tvm.nd.array(np.random.uniform(-1, 1, size=lhs_shape).astype("float32"), ctx)
        rhs = tvm.nd.array(np.random.uniform(-1, 1, size=rhs_shape).astype("float32"), ctx)
        #set
        a = timestamp()
        sync_module.set_input("lhs", lhs)
        sync_module.set_input("rhs", rhs)
        b = timestamp()
        #run
        sync_module.run()
        c = timestamp()
        sync_module.get_output(0)
        d = timestamp()
        if i >= warmup:
            record["set"].append(b-a)
            record["run"].append(c-b)
            record["get"].append(c-d)
    for k in record:
        record[k] = list_to_stat(record[k])
    return record

    
def record_numpy_sync(sync_type, lhs_shape, rhs_shape, repeat=110, warmup=10):
    record = []
    for i in range(repeat):
        lhs = np.random.uniform(-1, 1, size=lhs_shape).astype("float32")
        rhs = np.random.uniform(-1, 1, size=rhs_shape).astype("float32")
        if sync_type == 'add':
            start = timestamp()
            np.add(lhs, rhs)
            end = timestamp()
        elif sync_type == 'concat':
            start = timestamp()
            np.concatenate((lhs,rhs), axis=1)
            end = timestamp()
        if i >= warmup:
            record.append(end-start)
    return list_to_stat(record)


def list_to_stat(record: list):
    stat = dict()
    #stat['record'] = record
    stat['mean'] = np.mean(record)
    stat['median'] = np.median(record)
    record.sort()
    stat['record'] = record
    return stat
