��
��
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( �
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
�
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( �

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ��
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.11.02v2.11.0-rc2-17-gd5b57ca93e58��
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_2
[
count_2/Read/ReadVariableOpReadVariableOpcount_2*
_output_shapes
: *
dtype0
b
total_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_2
[
total_2/Read/ReadVariableOpReadVariableOptotal_2*
_output_shapes
: *
dtype0
n
learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namelearning_rate
g
!learning_rate/Read/ReadVariableOpReadVariableOplearning_rate*
_output_shapes
: *
dtype0
f
	iterationVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	iteration
_
iteration/Read/ReadVariableOpReadVariableOp	iteration*
_output_shapes
: *
dtype0	
z
output_layer/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:`*"
shared_nameoutput_layer/bias
s
%output_layer/bias/Read/ReadVariableOpReadVariableOpoutput_layer/bias*
_output_shapes
:`*
dtype0
�
output_layer/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�`*$
shared_nameoutput_layer/kernel
|
'output_layer/kernel/Read/ReadVariableOpReadVariableOpoutput_layer/kernel*
_output_shapes
:	�`*
dtype0
o
layer4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_namelayer4/bias
h
layer4/bias/Read/ReadVariableOpReadVariableOplayer4/bias*
_output_shapes	
:�*
dtype0
x
layer4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*
shared_namelayer4/kernel
q
!layer4/kernel/Read/ReadVariableOpReadVariableOplayer4/kernel* 
_output_shapes
:
��*
dtype0
o
layer3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_namelayer3/bias
h
layer3/bias/Read/ReadVariableOpReadVariableOplayer3/bias*
_output_shapes	
:�*
dtype0
x
layer3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*
shared_namelayer3/kernel
q
!layer3/kernel/Read/ReadVariableOpReadVariableOplayer3/kernel* 
_output_shapes
:
��*
dtype0
o
layer2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_namelayer2/bias
h
layer2/bias/Read/ReadVariableOpReadVariableOplayer2/bias*
_output_shapes	
:�*
dtype0
x
layer2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*
shared_namelayer2/kernel
q
!layer2/kernel/Read/ReadVariableOpReadVariableOplayer2/kernel* 
_output_shapes
:
��*
dtype0
o
layer1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_namelayer1/bias
h
layer1/bias/Read/ReadVariableOpReadVariableOplayer1/bias*
_output_shapes	
:�*
dtype0
x
layer1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*
shared_namelayer1/kernel
q
!layer1/kernel/Read/ReadVariableOpReadVariableOplayer1/kernel* 
_output_shapes
:
��*
dtype0
o
layer0/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_namelayer0/bias
h
layer0/bias/Read/ReadVariableOpReadVariableOplayer0/bias*
_output_shapes	
:�*
dtype0
w
layer0/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	k�*
shared_namelayer0/kernel
p
!layer0/kernel/Read/ReadVariableOpReadVariableOplayer0/kernel*
_output_shapes
:	k�*
dtype0
z
serving_default_input_1Placeholder*'
_output_shapes
:���������k*
dtype0*
shape:���������k
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1layer0/kernellayer0/biaslayer1/kernellayer1/biaslayer2/kernellayer2/biaslayer3/kernellayer3/biaslayer4/kernellayer4/biasoutput_layer/kerneloutput_layer/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������`*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *.
f)R'
%__inference_signature_wrapper_1024533

NoOpNoOp
�<
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�;
value�;B�; B�;
�
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer_with_weights-3
layer-3
layer_with_weights-4
layer-4
layer_with_weights-5
layer-5
layer-6
	variables
	trainable_variables

regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures
#_self_saveable_object_factories*
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias
#_self_saveable_object_factories*
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
* &call_and_return_all_conditional_losses

!kernel
"bias
##_self_saveable_object_factories*
�
$	variables
%trainable_variables
&regularization_losses
'	keras_api
(__call__
*)&call_and_return_all_conditional_losses

*kernel
+bias
#,_self_saveable_object_factories*
�
-	variables
.trainable_variables
/regularization_losses
0	keras_api
1__call__
*2&call_and_return_all_conditional_losses

3kernel
4bias
#5_self_saveable_object_factories*
�
6	variables
7trainable_variables
8regularization_losses
9	keras_api
:__call__
*;&call_and_return_all_conditional_losses

<kernel
=bias
#>_self_saveable_object_factories*
�
?	variables
@trainable_variables
Aregularization_losses
B	keras_api
C__call__
*D&call_and_return_all_conditional_losses

Ekernel
Fbias
#G_self_saveable_object_factories*
�
H	variables
Itrainable_variables
Jregularization_losses
K	keras_api
L__call__
*M&call_and_return_all_conditional_losses
#N_self_saveable_object_factories* 
Z
0
1
!2
"3
*4
+5
36
47
<8
=9
E10
F11*
Z
0
1
!2
"3
*4
+5
36
47
<8
=9
E10
F11*
* 
�
Onon_trainable_variables

Players
Qmetrics
Rlayer_regularization_losses
Slayer_metrics
	variables
	trainable_variables

regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
Ttrace_0
Utrace_1
Vtrace_2
Wtrace_3* 
6
Xtrace_0
Ytrace_1
Ztrace_2
[trace_3* 
* 
O
\
_variables
]_iterations
^_learning_rate
__update_step_xla*

`serving_default* 
* 

0
1*

0
1*
* 
�
anon_trainable_variables

blayers
cmetrics
dlayer_regularization_losses
elayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

ftrace_0* 

gtrace_0* 
]W
VARIABLE_VALUElayer0/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUElayer0/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

!0
"1*

!0
"1*
* 
�
hnon_trainable_variables

ilayers
jmetrics
klayer_regularization_losses
llayer_metrics
	variables
trainable_variables
regularization_losses
__call__
* &call_and_return_all_conditional_losses
& "call_and_return_conditional_losses*

mtrace_0* 

ntrace_0* 
]W
VARIABLE_VALUElayer1/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUElayer1/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

*0
+1*

*0
+1*
* 
�
onon_trainable_variables

players
qmetrics
rlayer_regularization_losses
slayer_metrics
$	variables
%trainable_variables
&regularization_losses
(__call__
*)&call_and_return_all_conditional_losses
&)"call_and_return_conditional_losses*

ttrace_0* 

utrace_0* 
]W
VARIABLE_VALUElayer2/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUElayer2/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

30
41*

30
41*
* 
�
vnon_trainable_variables

wlayers
xmetrics
ylayer_regularization_losses
zlayer_metrics
-	variables
.trainable_variables
/regularization_losses
1__call__
*2&call_and_return_all_conditional_losses
&2"call_and_return_conditional_losses*

{trace_0* 

|trace_0* 
]W
VARIABLE_VALUElayer3/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUElayer3/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

<0
=1*

<0
=1*
* 
�
}non_trainable_variables

~layers
metrics
 �layer_regularization_losses
�layer_metrics
6	variables
7trainable_variables
8regularization_losses
:__call__
*;&call_and_return_all_conditional_losses
&;"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
]W
VARIABLE_VALUElayer4/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUElayer4/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

E0
F1*

E0
F1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
?	variables
@trainable_variables
Aregularization_losses
C__call__
*D&call_and_return_all_conditional_losses
&D"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
c]
VARIABLE_VALUEoutput_layer/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEoutput_layer/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
H	variables
Itrainable_variables
Jregularization_losses
L__call__
*M&call_and_return_all_conditional_losses
&M"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 
* 
* 
5
0
1
2
3
4
5
6*

�0
�1
�2*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

]0*
SM
VARIABLE_VALUE	iteration0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUElearning_rate3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
<
�	variables
�	keras_api

�total

�count*
<
�	variables
�	keras_api

�total

�count*
M
�	variables
�	keras_api

�total

�count
�
_fn_kwargs*

�0
�1*

�	variables*
UO
VARIABLE_VALUEtotal_24keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_24keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

�0
�1*

�	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*

�0
�1*

�	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename!layer0/kernel/Read/ReadVariableOplayer0/bias/Read/ReadVariableOp!layer1/kernel/Read/ReadVariableOplayer1/bias/Read/ReadVariableOp!layer2/kernel/Read/ReadVariableOplayer2/bias/Read/ReadVariableOp!layer3/kernel/Read/ReadVariableOplayer3/bias/Read/ReadVariableOp!layer4/kernel/Read/ReadVariableOplayer4/bias/Read/ReadVariableOp'output_layer/kernel/Read/ReadVariableOp%output_layer/bias/Read/ReadVariableOpiteration/Read/ReadVariableOp!learning_rate/Read/ReadVariableOptotal_2/Read/ReadVariableOpcount_2/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOpConst*!
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *)
f$R"
 __inference__traced_save_1024898
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamelayer0/kernellayer0/biaslayer1/kernellayer1/biaslayer2/kernellayer2/biaslayer3/kernellayer3/biaslayer4/kernellayer4/biasoutput_layer/kerneloutput_layer/bias	iterationlearning_ratetotal_2count_2total_1count_1totalcount* 
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *,
f'R%
#__inference__traced_restore_1024968��
�

�
C__inference_layer3_layer_call_and_return_conditional_losses_1024167

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:����������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
C__inference_layer2_layer_call_and_return_conditional_losses_1024745

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:����������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�5
�	
G__inference_sequential_layer_call_and_return_conditional_losses_1024638

inputs8
%layer0_matmul_readvariableop_resource:	k�5
&layer0_biasadd_readvariableop_resource:	�9
%layer1_matmul_readvariableop_resource:
��5
&layer1_biasadd_readvariableop_resource:	�9
%layer2_matmul_readvariableop_resource:
��5
&layer2_biasadd_readvariableop_resource:	�9
%layer3_matmul_readvariableop_resource:
��5
&layer3_biasadd_readvariableop_resource:	�9
%layer4_matmul_readvariableop_resource:
��5
&layer4_biasadd_readvariableop_resource:	�>
+output_layer_matmul_readvariableop_resource:	�`:
,output_layer_biasadd_readvariableop_resource:`
identity��layer0/BiasAdd/ReadVariableOp�layer0/MatMul/ReadVariableOp�layer1/BiasAdd/ReadVariableOp�layer1/MatMul/ReadVariableOp�layer2/BiasAdd/ReadVariableOp�layer2/MatMul/ReadVariableOp�layer3/BiasAdd/ReadVariableOp�layer3/MatMul/ReadVariableOp�layer4/BiasAdd/ReadVariableOp�layer4/MatMul/ReadVariableOp�#output_layer/BiasAdd/ReadVariableOp�"output_layer/MatMul/ReadVariableOp�
layer0/MatMul/ReadVariableOpReadVariableOp%layer0_matmul_readvariableop_resource*
_output_shapes
:	k�*
dtype0x
layer0/MatMulMatMulinputs$layer0/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
layer0/BiasAdd/ReadVariableOpReadVariableOp&layer0_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
layer0/BiasAddBiasAddlayer0/MatMul:product:0%layer0/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������_
layer0/ReluRelulayer0/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
layer1/MatMul/ReadVariableOpReadVariableOp%layer1_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
layer1/MatMulMatMullayer0/Relu:activations:0$layer1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
layer1/BiasAdd/ReadVariableOpReadVariableOp&layer1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
layer1/BiasAddBiasAddlayer1/MatMul:product:0%layer1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������_
layer1/ReluRelulayer1/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
layer2/MatMul/ReadVariableOpReadVariableOp%layer2_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
layer2/MatMulMatMullayer1/Relu:activations:0$layer2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
layer2/BiasAdd/ReadVariableOpReadVariableOp&layer2_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
layer2/BiasAddBiasAddlayer2/MatMul:product:0%layer2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������_
layer2/ReluRelulayer2/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
layer3/MatMul/ReadVariableOpReadVariableOp%layer3_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
layer3/MatMulMatMullayer2/Relu:activations:0$layer3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
layer3/BiasAdd/ReadVariableOpReadVariableOp&layer3_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
layer3/BiasAddBiasAddlayer3/MatMul:product:0%layer3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������_
layer3/ReluRelulayer3/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
layer4/MatMul/ReadVariableOpReadVariableOp%layer4_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
layer4/MatMulMatMullayer3/Relu:activations:0$layer4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
layer4/BiasAdd/ReadVariableOpReadVariableOp&layer4_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
layer4/BiasAddBiasAddlayer4/MatMul:product:0%layer4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������_
layer4/ReluRelulayer4/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
"output_layer/MatMul/ReadVariableOpReadVariableOp+output_layer_matmul_readvariableop_resource*
_output_shapes
:	�`*
dtype0�
output_layer/MatMulMatMullayer4/Relu:activations:0*output_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������`�
#output_layer/BiasAdd/ReadVariableOpReadVariableOp,output_layer_biasadd_readvariableop_resource*
_output_shapes
:`*
dtype0�
output_layer/BiasAddBiasAddoutput_layer/MatMul:product:0+output_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������`j
output_layer/ReluReluoutput_layer/BiasAdd:output:0*
T0*'
_output_shapes
:���������`e

re_lu/ReluReluoutput_layer/Relu:activations:0*
T0*'
_output_shapes
:���������`g
IdentityIdentityre_lu/Relu:activations:0^NoOp*
T0*'
_output_shapes
:���������`�
NoOpNoOp^layer0/BiasAdd/ReadVariableOp^layer0/MatMul/ReadVariableOp^layer1/BiasAdd/ReadVariableOp^layer1/MatMul/ReadVariableOp^layer2/BiasAdd/ReadVariableOp^layer2/MatMul/ReadVariableOp^layer3/BiasAdd/ReadVariableOp^layer3/MatMul/ReadVariableOp^layer4/BiasAdd/ReadVariableOp^layer4/MatMul/ReadVariableOp$^output_layer/BiasAdd/ReadVariableOp#^output_layer/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:���������k: : : : : : : : : : : : 2>
layer0/BiasAdd/ReadVariableOplayer0/BiasAdd/ReadVariableOp2<
layer0/MatMul/ReadVariableOplayer0/MatMul/ReadVariableOp2>
layer1/BiasAdd/ReadVariableOplayer1/BiasAdd/ReadVariableOp2<
layer1/MatMul/ReadVariableOplayer1/MatMul/ReadVariableOp2>
layer2/BiasAdd/ReadVariableOplayer2/BiasAdd/ReadVariableOp2<
layer2/MatMul/ReadVariableOplayer2/MatMul/ReadVariableOp2>
layer3/BiasAdd/ReadVariableOplayer3/BiasAdd/ReadVariableOp2<
layer3/MatMul/ReadVariableOplayer3/MatMul/ReadVariableOp2>
layer4/BiasAdd/ReadVariableOplayer4/BiasAdd/ReadVariableOp2<
layer4/MatMul/ReadVariableOplayer4/MatMul/ReadVariableOp2J
#output_layer/BiasAdd/ReadVariableOp#output_layer/BiasAdd/ReadVariableOp2H
"output_layer/MatMul/ReadVariableOp"output_layer/MatMul/ReadVariableOp:O K
'
_output_shapes
:���������k
 
_user_specified_nameinputs
�

�
,__inference_sequential_layer_call_fn_1024562

inputs
unknown:	k�
	unknown_0:	�
	unknown_1:
��
	unknown_2:	�
	unknown_3:
��
	unknown_4:	�
	unknown_5:
��
	unknown_6:	�
	unknown_7:
��
	unknown_8:	�
	unknown_9:	�`

unknown_10:`
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������`*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_sequential_layer_call_and_return_conditional_losses_1024215o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������``
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:���������k: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������k
 
_user_specified_nameinputs
�

�
I__inference_output_layer_layer_call_and_return_conditional_losses_1024201

inputs1
matmul_readvariableop_resource:	�`-
biasadd_readvariableop_resource:`
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�`*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������`r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:`*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������`P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������`a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������`w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
(__inference_layer2_layer_call_fn_1024734

inputs
unknown:
��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_layer2_layer_call_and_return_conditional_losses_1024150p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�5
�	
G__inference_sequential_layer_call_and_return_conditional_losses_1024685

inputs8
%layer0_matmul_readvariableop_resource:	k�5
&layer0_biasadd_readvariableop_resource:	�9
%layer1_matmul_readvariableop_resource:
��5
&layer1_biasadd_readvariableop_resource:	�9
%layer2_matmul_readvariableop_resource:
��5
&layer2_biasadd_readvariableop_resource:	�9
%layer3_matmul_readvariableop_resource:
��5
&layer3_biasadd_readvariableop_resource:	�9
%layer4_matmul_readvariableop_resource:
��5
&layer4_biasadd_readvariableop_resource:	�>
+output_layer_matmul_readvariableop_resource:	�`:
,output_layer_biasadd_readvariableop_resource:`
identity��layer0/BiasAdd/ReadVariableOp�layer0/MatMul/ReadVariableOp�layer1/BiasAdd/ReadVariableOp�layer1/MatMul/ReadVariableOp�layer2/BiasAdd/ReadVariableOp�layer2/MatMul/ReadVariableOp�layer3/BiasAdd/ReadVariableOp�layer3/MatMul/ReadVariableOp�layer4/BiasAdd/ReadVariableOp�layer4/MatMul/ReadVariableOp�#output_layer/BiasAdd/ReadVariableOp�"output_layer/MatMul/ReadVariableOp�
layer0/MatMul/ReadVariableOpReadVariableOp%layer0_matmul_readvariableop_resource*
_output_shapes
:	k�*
dtype0x
layer0/MatMulMatMulinputs$layer0/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
layer0/BiasAdd/ReadVariableOpReadVariableOp&layer0_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
layer0/BiasAddBiasAddlayer0/MatMul:product:0%layer0/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������_
layer0/ReluRelulayer0/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
layer1/MatMul/ReadVariableOpReadVariableOp%layer1_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
layer1/MatMulMatMullayer0/Relu:activations:0$layer1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
layer1/BiasAdd/ReadVariableOpReadVariableOp&layer1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
layer1/BiasAddBiasAddlayer1/MatMul:product:0%layer1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������_
layer1/ReluRelulayer1/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
layer2/MatMul/ReadVariableOpReadVariableOp%layer2_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
layer2/MatMulMatMullayer1/Relu:activations:0$layer2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
layer2/BiasAdd/ReadVariableOpReadVariableOp&layer2_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
layer2/BiasAddBiasAddlayer2/MatMul:product:0%layer2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������_
layer2/ReluRelulayer2/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
layer3/MatMul/ReadVariableOpReadVariableOp%layer3_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
layer3/MatMulMatMullayer2/Relu:activations:0$layer3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
layer3/BiasAdd/ReadVariableOpReadVariableOp&layer3_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
layer3/BiasAddBiasAddlayer3/MatMul:product:0%layer3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������_
layer3/ReluRelulayer3/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
layer4/MatMul/ReadVariableOpReadVariableOp%layer4_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
layer4/MatMulMatMullayer3/Relu:activations:0$layer4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
layer4/BiasAdd/ReadVariableOpReadVariableOp&layer4_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
layer4/BiasAddBiasAddlayer4/MatMul:product:0%layer4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������_
layer4/ReluRelulayer4/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
"output_layer/MatMul/ReadVariableOpReadVariableOp+output_layer_matmul_readvariableop_resource*
_output_shapes
:	�`*
dtype0�
output_layer/MatMulMatMullayer4/Relu:activations:0*output_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������`�
#output_layer/BiasAdd/ReadVariableOpReadVariableOp,output_layer_biasadd_readvariableop_resource*
_output_shapes
:`*
dtype0�
output_layer/BiasAddBiasAddoutput_layer/MatMul:product:0+output_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������`j
output_layer/ReluReluoutput_layer/BiasAdd:output:0*
T0*'
_output_shapes
:���������`e

re_lu/ReluReluoutput_layer/Relu:activations:0*
T0*'
_output_shapes
:���������`g
IdentityIdentityre_lu/Relu:activations:0^NoOp*
T0*'
_output_shapes
:���������`�
NoOpNoOp^layer0/BiasAdd/ReadVariableOp^layer0/MatMul/ReadVariableOp^layer1/BiasAdd/ReadVariableOp^layer1/MatMul/ReadVariableOp^layer2/BiasAdd/ReadVariableOp^layer2/MatMul/ReadVariableOp^layer3/BiasAdd/ReadVariableOp^layer3/MatMul/ReadVariableOp^layer4/BiasAdd/ReadVariableOp^layer4/MatMul/ReadVariableOp$^output_layer/BiasAdd/ReadVariableOp#^output_layer/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:���������k: : : : : : : : : : : : 2>
layer0/BiasAdd/ReadVariableOplayer0/BiasAdd/ReadVariableOp2<
layer0/MatMul/ReadVariableOplayer0/MatMul/ReadVariableOp2>
layer1/BiasAdd/ReadVariableOplayer1/BiasAdd/ReadVariableOp2<
layer1/MatMul/ReadVariableOplayer1/MatMul/ReadVariableOp2>
layer2/BiasAdd/ReadVariableOplayer2/BiasAdd/ReadVariableOp2<
layer2/MatMul/ReadVariableOplayer2/MatMul/ReadVariableOp2>
layer3/BiasAdd/ReadVariableOplayer3/BiasAdd/ReadVariableOp2<
layer3/MatMul/ReadVariableOplayer3/MatMul/ReadVariableOp2>
layer4/BiasAdd/ReadVariableOplayer4/BiasAdd/ReadVariableOp2<
layer4/MatMul/ReadVariableOplayer4/MatMul/ReadVariableOp2J
#output_layer/BiasAdd/ReadVariableOp#output_layer/BiasAdd/ReadVariableOp2H
"output_layer/MatMul/ReadVariableOp"output_layer/MatMul/ReadVariableOp:O K
'
_output_shapes
:���������k
 
_user_specified_nameinputs
�U
�
#__inference__traced_restore_1024968
file_prefix1
assignvariableop_layer0_kernel:	k�-
assignvariableop_1_layer0_bias:	�4
 assignvariableop_2_layer1_kernel:
��-
assignvariableop_3_layer1_bias:	�4
 assignvariableop_4_layer2_kernel:
��-
assignvariableop_5_layer2_bias:	�4
 assignvariableop_6_layer3_kernel:
��-
assignvariableop_7_layer3_bias:	�4
 assignvariableop_8_layer4_kernel:
��-
assignvariableop_9_layer4_bias:	�:
'assignvariableop_10_output_layer_kernel:	�`3
%assignvariableop_11_output_layer_bias:`'
assignvariableop_12_iteration:	 +
!assignvariableop_13_learning_rate: %
assignvariableop_14_total_2: %
assignvariableop_15_count_2: %
assignvariableop_16_total_1: %
assignvariableop_17_count_1: #
assignvariableop_18_total: #
assignvariableop_19_count: 
identity_21��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_3�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�	
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*=
value4B2B B B B B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*h
_output_shapesV
T:::::::::::::::::::::*#
dtypes
2	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOpassignvariableop_layer0_kernelIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOpassignvariableop_1_layer0_biasIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOp assignvariableop_2_layer1_kernelIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOpassignvariableop_3_layer1_biasIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOp assignvariableop_4_layer2_kernelIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOpassignvariableop_5_layer2_biasIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOp assignvariableop_6_layer3_kernelIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOpassignvariableop_7_layer3_biasIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOp assignvariableop_8_layer4_kernelIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOpassignvariableop_9_layer4_biasIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOp'assignvariableop_10_output_layer_kernelIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOp%assignvariableop_11_output_layer_biasIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_12AssignVariableOpassignvariableop_12_iterationIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOp!assignvariableop_13_learning_rateIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOpassignvariableop_14_total_2Identity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOpassignvariableop_15_count_2Identity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOpassignvariableop_16_total_1Identity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOpassignvariableop_17_count_1Identity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_18AssignVariableOpassignvariableop_18_totalIdentity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOpassignvariableop_19_countIdentity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 �
Identity_20Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_21IdentityIdentity_20:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_21Identity_21:output:0*=
_input_shapes,
*: : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
�
(__inference_layer1_layer_call_fn_1024714

inputs
unknown:
��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_layer1_layer_call_and_return_conditional_losses_1024133p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�"
�
G__inference_sequential_layer_call_and_return_conditional_losses_1024215

inputs!
layer0_1024117:	k�
layer0_1024119:	�"
layer1_1024134:
��
layer1_1024136:	�"
layer2_1024151:
��
layer2_1024153:	�"
layer3_1024168:
��
layer3_1024170:	�"
layer4_1024185:
��
layer4_1024187:	�'
output_layer_1024202:	�`"
output_layer_1024204:`
identity��layer0/StatefulPartitionedCall�layer1/StatefulPartitionedCall�layer2/StatefulPartitionedCall�layer3/StatefulPartitionedCall�layer4/StatefulPartitionedCall�$output_layer/StatefulPartitionedCall�
layer0/StatefulPartitionedCallStatefulPartitionedCallinputslayer0_1024117layer0_1024119*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_layer0_layer_call_and_return_conditional_losses_1024116�
layer1/StatefulPartitionedCallStatefulPartitionedCall'layer0/StatefulPartitionedCall:output:0layer1_1024134layer1_1024136*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_layer1_layer_call_and_return_conditional_losses_1024133�
layer2/StatefulPartitionedCallStatefulPartitionedCall'layer1/StatefulPartitionedCall:output:0layer2_1024151layer2_1024153*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_layer2_layer_call_and_return_conditional_losses_1024150�
layer3/StatefulPartitionedCallStatefulPartitionedCall'layer2/StatefulPartitionedCall:output:0layer3_1024168layer3_1024170*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_layer3_layer_call_and_return_conditional_losses_1024167�
layer4/StatefulPartitionedCallStatefulPartitionedCall'layer3/StatefulPartitionedCall:output:0layer4_1024185layer4_1024187*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_layer4_layer_call_and_return_conditional_losses_1024184�
$output_layer/StatefulPartitionedCallStatefulPartitionedCall'layer4/StatefulPartitionedCall:output:0output_layer_1024202output_layer_1024204*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������`*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_output_layer_layer_call_and_return_conditional_losses_1024201�
re_lu/PartitionedCallPartitionedCall-output_layer/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������`* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_re_lu_layer_call_and_return_conditional_losses_1024212m
IdentityIdentityre_lu/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`�
NoOpNoOp^layer0/StatefulPartitionedCall^layer1/StatefulPartitionedCall^layer2/StatefulPartitionedCall^layer3/StatefulPartitionedCall^layer4/StatefulPartitionedCall%^output_layer/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:���������k: : : : : : : : : : : : 2@
layer0/StatefulPartitionedCalllayer0/StatefulPartitionedCall2@
layer1/StatefulPartitionedCalllayer1/StatefulPartitionedCall2@
layer2/StatefulPartitionedCalllayer2/StatefulPartitionedCall2@
layer3/StatefulPartitionedCalllayer3/StatefulPartitionedCall2@
layer4/StatefulPartitionedCalllayer4/StatefulPartitionedCall2L
$output_layer/StatefulPartitionedCall$output_layer/StatefulPartitionedCall:O K
'
_output_shapes
:���������k
 
_user_specified_nameinputs
�
�
(__inference_layer3_layer_call_fn_1024754

inputs
unknown:
��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_layer3_layer_call_and_return_conditional_losses_1024167p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
C__inference_layer4_layer_call_and_return_conditional_losses_1024785

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:����������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�"
�
G__inference_sequential_layer_call_and_return_conditional_losses_1024465
input_1!
layer0_1024433:	k�
layer0_1024435:	�"
layer1_1024438:
��
layer1_1024440:	�"
layer2_1024443:
��
layer2_1024445:	�"
layer3_1024448:
��
layer3_1024450:	�"
layer4_1024453:
��
layer4_1024455:	�'
output_layer_1024458:	�`"
output_layer_1024460:`
identity��layer0/StatefulPartitionedCall�layer1/StatefulPartitionedCall�layer2/StatefulPartitionedCall�layer3/StatefulPartitionedCall�layer4/StatefulPartitionedCall�$output_layer/StatefulPartitionedCall�
layer0/StatefulPartitionedCallStatefulPartitionedCallinput_1layer0_1024433layer0_1024435*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_layer0_layer_call_and_return_conditional_losses_1024116�
layer1/StatefulPartitionedCallStatefulPartitionedCall'layer0/StatefulPartitionedCall:output:0layer1_1024438layer1_1024440*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_layer1_layer_call_and_return_conditional_losses_1024133�
layer2/StatefulPartitionedCallStatefulPartitionedCall'layer1/StatefulPartitionedCall:output:0layer2_1024443layer2_1024445*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_layer2_layer_call_and_return_conditional_losses_1024150�
layer3/StatefulPartitionedCallStatefulPartitionedCall'layer2/StatefulPartitionedCall:output:0layer3_1024448layer3_1024450*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_layer3_layer_call_and_return_conditional_losses_1024167�
layer4/StatefulPartitionedCallStatefulPartitionedCall'layer3/StatefulPartitionedCall:output:0layer4_1024453layer4_1024455*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_layer4_layer_call_and_return_conditional_losses_1024184�
$output_layer/StatefulPartitionedCallStatefulPartitionedCall'layer4/StatefulPartitionedCall:output:0output_layer_1024458output_layer_1024460*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������`*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_output_layer_layer_call_and_return_conditional_losses_1024201�
re_lu/PartitionedCallPartitionedCall-output_layer/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������`* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_re_lu_layer_call_and_return_conditional_losses_1024212m
IdentityIdentityre_lu/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`�
NoOpNoOp^layer0/StatefulPartitionedCall^layer1/StatefulPartitionedCall^layer2/StatefulPartitionedCall^layer3/StatefulPartitionedCall^layer4/StatefulPartitionedCall%^output_layer/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:���������k: : : : : : : : : : : : 2@
layer0/StatefulPartitionedCalllayer0/StatefulPartitionedCall2@
layer1/StatefulPartitionedCalllayer1/StatefulPartitionedCall2@
layer2/StatefulPartitionedCalllayer2/StatefulPartitionedCall2@
layer3/StatefulPartitionedCalllayer3/StatefulPartitionedCall2@
layer4/StatefulPartitionedCalllayer4/StatefulPartitionedCall2L
$output_layer/StatefulPartitionedCall$output_layer/StatefulPartitionedCall:P L
'
_output_shapes
:���������k
!
_user_specified_name	input_1
�
�
.__inference_output_layer_layer_call_fn_1024794

inputs
unknown:	�`
	unknown_0:`
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������`*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_output_layer_layer_call_and_return_conditional_losses_1024201o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������``
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�.
�
 __inference__traced_save_1024898
file_prefix,
(savev2_layer0_kernel_read_readvariableop*
&savev2_layer0_bias_read_readvariableop,
(savev2_layer1_kernel_read_readvariableop*
&savev2_layer1_bias_read_readvariableop,
(savev2_layer2_kernel_read_readvariableop*
&savev2_layer2_bias_read_readvariableop,
(savev2_layer3_kernel_read_readvariableop*
&savev2_layer3_bias_read_readvariableop,
(savev2_layer4_kernel_read_readvariableop*
&savev2_layer4_bias_read_readvariableop2
.savev2_output_layer_kernel_read_readvariableop0
,savev2_output_layer_bias_read_readvariableop(
$savev2_iteration_read_readvariableop	,
(savev2_learning_rate_read_readvariableop&
"savev2_total_2_read_readvariableop&
"savev2_count_2_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop
savev2_const

identity_1��MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: �	
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*=
value4B2B B B B B B B B B B B B B B B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0(savev2_layer0_kernel_read_readvariableop&savev2_layer0_bias_read_readvariableop(savev2_layer1_kernel_read_readvariableop&savev2_layer1_bias_read_readvariableop(savev2_layer2_kernel_read_readvariableop&savev2_layer2_bias_read_readvariableop(savev2_layer3_kernel_read_readvariableop&savev2_layer3_bias_read_readvariableop(savev2_layer4_kernel_read_readvariableop&savev2_layer4_bias_read_readvariableop.savev2_output_layer_kernel_read_readvariableop,savev2_output_layer_bias_read_readvariableop$savev2_iteration_read_readvariableop(savev2_learning_rate_read_readvariableop"savev2_total_2_read_readvariableop"savev2_count_2_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableopsavev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *#
dtypes
2	�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*�
_input_shapes�
�: :	k�:�:
��:�:
��:�:
��:�:
��:�:	�`:`: : : : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	k�:!

_output_shapes	
:�:&"
 
_output_shapes
:
��:!

_output_shapes	
:�:&"
 
_output_shapes
:
��:!

_output_shapes	
:�:&"
 
_output_shapes
:
��:!

_output_shapes	
:�:&	"
 
_output_shapes
:
��:!


_output_shapes	
:�:%!

_output_shapes
:	�`: 

_output_shapes
:`:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�

�
I__inference_output_layer_layer_call_and_return_conditional_losses_1024805

inputs1
matmul_readvariableop_resource:	�`-
biasadd_readvariableop_resource:`
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�`*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������`r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:`*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������`P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������`a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������`w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
^
B__inference_re_lu_layer_call_and_return_conditional_losses_1024815

inputs
identityF
ReluReluinputs*
T0*'
_output_shapes
:���������`Z
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:���������`"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������`:O K
'
_output_shapes
:���������`
 
_user_specified_nameinputs
�

�
,__inference_sequential_layer_call_fn_1024591

inputs
unknown:	k�
	unknown_0:	�
	unknown_1:
��
	unknown_2:	�
	unknown_3:
��
	unknown_4:	�
	unknown_5:
��
	unknown_6:	�
	unknown_7:
��
	unknown_8:	�
	unknown_9:	�`

unknown_10:`
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������`*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_sequential_layer_call_and_return_conditional_losses_1024374o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������``
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:���������k: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������k
 
_user_specified_nameinputs
�

�
C__inference_layer0_layer_call_and_return_conditional_losses_1024705

inputs1
matmul_readvariableop_resource:	k�.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	k�*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:����������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������k: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������k
 
_user_specified_nameinputs
�@
�
"__inference__wrapped_model_1024098
input_1C
0sequential_layer0_matmul_readvariableop_resource:	k�@
1sequential_layer0_biasadd_readvariableop_resource:	�D
0sequential_layer1_matmul_readvariableop_resource:
��@
1sequential_layer1_biasadd_readvariableop_resource:	�D
0sequential_layer2_matmul_readvariableop_resource:
��@
1sequential_layer2_biasadd_readvariableop_resource:	�D
0sequential_layer3_matmul_readvariableop_resource:
��@
1sequential_layer3_biasadd_readvariableop_resource:	�D
0sequential_layer4_matmul_readvariableop_resource:
��@
1sequential_layer4_biasadd_readvariableop_resource:	�I
6sequential_output_layer_matmul_readvariableop_resource:	�`E
7sequential_output_layer_biasadd_readvariableop_resource:`
identity��(sequential/layer0/BiasAdd/ReadVariableOp�'sequential/layer0/MatMul/ReadVariableOp�(sequential/layer1/BiasAdd/ReadVariableOp�'sequential/layer1/MatMul/ReadVariableOp�(sequential/layer2/BiasAdd/ReadVariableOp�'sequential/layer2/MatMul/ReadVariableOp�(sequential/layer3/BiasAdd/ReadVariableOp�'sequential/layer3/MatMul/ReadVariableOp�(sequential/layer4/BiasAdd/ReadVariableOp�'sequential/layer4/MatMul/ReadVariableOp�.sequential/output_layer/BiasAdd/ReadVariableOp�-sequential/output_layer/MatMul/ReadVariableOp�
'sequential/layer0/MatMul/ReadVariableOpReadVariableOp0sequential_layer0_matmul_readvariableop_resource*
_output_shapes
:	k�*
dtype0�
sequential/layer0/MatMulMatMulinput_1/sequential/layer0/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
(sequential/layer0/BiasAdd/ReadVariableOpReadVariableOp1sequential_layer0_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
sequential/layer0/BiasAddBiasAdd"sequential/layer0/MatMul:product:00sequential/layer0/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������u
sequential/layer0/ReluRelu"sequential/layer0/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
'sequential/layer1/MatMul/ReadVariableOpReadVariableOp0sequential_layer1_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
sequential/layer1/MatMulMatMul$sequential/layer0/Relu:activations:0/sequential/layer1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
(sequential/layer1/BiasAdd/ReadVariableOpReadVariableOp1sequential_layer1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
sequential/layer1/BiasAddBiasAdd"sequential/layer1/MatMul:product:00sequential/layer1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������u
sequential/layer1/ReluRelu"sequential/layer1/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
'sequential/layer2/MatMul/ReadVariableOpReadVariableOp0sequential_layer2_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
sequential/layer2/MatMulMatMul$sequential/layer1/Relu:activations:0/sequential/layer2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
(sequential/layer2/BiasAdd/ReadVariableOpReadVariableOp1sequential_layer2_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
sequential/layer2/BiasAddBiasAdd"sequential/layer2/MatMul:product:00sequential/layer2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������u
sequential/layer2/ReluRelu"sequential/layer2/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
'sequential/layer3/MatMul/ReadVariableOpReadVariableOp0sequential_layer3_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
sequential/layer3/MatMulMatMul$sequential/layer2/Relu:activations:0/sequential/layer3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
(sequential/layer3/BiasAdd/ReadVariableOpReadVariableOp1sequential_layer3_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
sequential/layer3/BiasAddBiasAdd"sequential/layer3/MatMul:product:00sequential/layer3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������u
sequential/layer3/ReluRelu"sequential/layer3/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
'sequential/layer4/MatMul/ReadVariableOpReadVariableOp0sequential_layer4_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
sequential/layer4/MatMulMatMul$sequential/layer3/Relu:activations:0/sequential/layer4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
(sequential/layer4/BiasAdd/ReadVariableOpReadVariableOp1sequential_layer4_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
sequential/layer4/BiasAddBiasAdd"sequential/layer4/MatMul:product:00sequential/layer4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������u
sequential/layer4/ReluRelu"sequential/layer4/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
-sequential/output_layer/MatMul/ReadVariableOpReadVariableOp6sequential_output_layer_matmul_readvariableop_resource*
_output_shapes
:	�`*
dtype0�
sequential/output_layer/MatMulMatMul$sequential/layer4/Relu:activations:05sequential/output_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������`�
.sequential/output_layer/BiasAdd/ReadVariableOpReadVariableOp7sequential_output_layer_biasadd_readvariableop_resource*
_output_shapes
:`*
dtype0�
sequential/output_layer/BiasAddBiasAdd(sequential/output_layer/MatMul:product:06sequential/output_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������`�
sequential/output_layer/ReluRelu(sequential/output_layer/BiasAdd:output:0*
T0*'
_output_shapes
:���������`{
sequential/re_lu/ReluRelu*sequential/output_layer/Relu:activations:0*
T0*'
_output_shapes
:���������`r
IdentityIdentity#sequential/re_lu/Relu:activations:0^NoOp*
T0*'
_output_shapes
:���������`�
NoOpNoOp)^sequential/layer0/BiasAdd/ReadVariableOp(^sequential/layer0/MatMul/ReadVariableOp)^sequential/layer1/BiasAdd/ReadVariableOp(^sequential/layer1/MatMul/ReadVariableOp)^sequential/layer2/BiasAdd/ReadVariableOp(^sequential/layer2/MatMul/ReadVariableOp)^sequential/layer3/BiasAdd/ReadVariableOp(^sequential/layer3/MatMul/ReadVariableOp)^sequential/layer4/BiasAdd/ReadVariableOp(^sequential/layer4/MatMul/ReadVariableOp/^sequential/output_layer/BiasAdd/ReadVariableOp.^sequential/output_layer/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:���������k: : : : : : : : : : : : 2T
(sequential/layer0/BiasAdd/ReadVariableOp(sequential/layer0/BiasAdd/ReadVariableOp2R
'sequential/layer0/MatMul/ReadVariableOp'sequential/layer0/MatMul/ReadVariableOp2T
(sequential/layer1/BiasAdd/ReadVariableOp(sequential/layer1/BiasAdd/ReadVariableOp2R
'sequential/layer1/MatMul/ReadVariableOp'sequential/layer1/MatMul/ReadVariableOp2T
(sequential/layer2/BiasAdd/ReadVariableOp(sequential/layer2/BiasAdd/ReadVariableOp2R
'sequential/layer2/MatMul/ReadVariableOp'sequential/layer2/MatMul/ReadVariableOp2T
(sequential/layer3/BiasAdd/ReadVariableOp(sequential/layer3/BiasAdd/ReadVariableOp2R
'sequential/layer3/MatMul/ReadVariableOp'sequential/layer3/MatMul/ReadVariableOp2T
(sequential/layer4/BiasAdd/ReadVariableOp(sequential/layer4/BiasAdd/ReadVariableOp2R
'sequential/layer4/MatMul/ReadVariableOp'sequential/layer4/MatMul/ReadVariableOp2`
.sequential/output_layer/BiasAdd/ReadVariableOp.sequential/output_layer/BiasAdd/ReadVariableOp2^
-sequential/output_layer/MatMul/ReadVariableOp-sequential/output_layer/MatMul/ReadVariableOp:P L
'
_output_shapes
:���������k
!
_user_specified_name	input_1
�

�
C__inference_layer1_layer_call_and_return_conditional_losses_1024133

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:����������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
C__inference_layer1_layer_call_and_return_conditional_losses_1024725

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:����������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
C__inference_layer3_layer_call_and_return_conditional_losses_1024765

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:����������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
C__inference_layer4_layer_call_and_return_conditional_losses_1024184

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:����������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
,__inference_sequential_layer_call_fn_1024242
input_1
unknown:	k�
	unknown_0:	�
	unknown_1:
��
	unknown_2:	�
	unknown_3:
��
	unknown_4:	�
	unknown_5:
��
	unknown_6:	�
	unknown_7:
��
	unknown_8:	�
	unknown_9:	�`

unknown_10:`
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������`*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_sequential_layer_call_and_return_conditional_losses_1024215o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������``
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:���������k: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:���������k
!
_user_specified_name	input_1
�

�
%__inference_signature_wrapper_1024533
input_1
unknown:	k�
	unknown_0:	�
	unknown_1:
��
	unknown_2:	�
	unknown_3:
��
	unknown_4:	�
	unknown_5:
��
	unknown_6:	�
	unknown_7:
��
	unknown_8:	�
	unknown_9:	�`

unknown_10:`
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������`*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *+
f&R$
"__inference__wrapped_model_1024098o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������``
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:���������k: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:���������k
!
_user_specified_name	input_1
�"
�
G__inference_sequential_layer_call_and_return_conditional_losses_1024500
input_1!
layer0_1024468:	k�
layer0_1024470:	�"
layer1_1024473:
��
layer1_1024475:	�"
layer2_1024478:
��
layer2_1024480:	�"
layer3_1024483:
��
layer3_1024485:	�"
layer4_1024488:
��
layer4_1024490:	�'
output_layer_1024493:	�`"
output_layer_1024495:`
identity��layer0/StatefulPartitionedCall�layer1/StatefulPartitionedCall�layer2/StatefulPartitionedCall�layer3/StatefulPartitionedCall�layer4/StatefulPartitionedCall�$output_layer/StatefulPartitionedCall�
layer0/StatefulPartitionedCallStatefulPartitionedCallinput_1layer0_1024468layer0_1024470*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_layer0_layer_call_and_return_conditional_losses_1024116�
layer1/StatefulPartitionedCallStatefulPartitionedCall'layer0/StatefulPartitionedCall:output:0layer1_1024473layer1_1024475*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_layer1_layer_call_and_return_conditional_losses_1024133�
layer2/StatefulPartitionedCallStatefulPartitionedCall'layer1/StatefulPartitionedCall:output:0layer2_1024478layer2_1024480*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_layer2_layer_call_and_return_conditional_losses_1024150�
layer3/StatefulPartitionedCallStatefulPartitionedCall'layer2/StatefulPartitionedCall:output:0layer3_1024483layer3_1024485*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_layer3_layer_call_and_return_conditional_losses_1024167�
layer4/StatefulPartitionedCallStatefulPartitionedCall'layer3/StatefulPartitionedCall:output:0layer4_1024488layer4_1024490*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_layer4_layer_call_and_return_conditional_losses_1024184�
$output_layer/StatefulPartitionedCallStatefulPartitionedCall'layer4/StatefulPartitionedCall:output:0output_layer_1024493output_layer_1024495*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������`*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_output_layer_layer_call_and_return_conditional_losses_1024201�
re_lu/PartitionedCallPartitionedCall-output_layer/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������`* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_re_lu_layer_call_and_return_conditional_losses_1024212m
IdentityIdentityre_lu/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`�
NoOpNoOp^layer0/StatefulPartitionedCall^layer1/StatefulPartitionedCall^layer2/StatefulPartitionedCall^layer3/StatefulPartitionedCall^layer4/StatefulPartitionedCall%^output_layer/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:���������k: : : : : : : : : : : : 2@
layer0/StatefulPartitionedCalllayer0/StatefulPartitionedCall2@
layer1/StatefulPartitionedCalllayer1/StatefulPartitionedCall2@
layer2/StatefulPartitionedCalllayer2/StatefulPartitionedCall2@
layer3/StatefulPartitionedCalllayer3/StatefulPartitionedCall2@
layer4/StatefulPartitionedCalllayer4/StatefulPartitionedCall2L
$output_layer/StatefulPartitionedCall$output_layer/StatefulPartitionedCall:P L
'
_output_shapes
:���������k
!
_user_specified_name	input_1
�
^
B__inference_re_lu_layer_call_and_return_conditional_losses_1024212

inputs
identityF
ReluReluinputs*
T0*'
_output_shapes
:���������`Z
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:���������`"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������`:O K
'
_output_shapes
:���������`
 
_user_specified_nameinputs
�
�
(__inference_layer0_layer_call_fn_1024694

inputs
unknown:	k�
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_layer0_layer_call_and_return_conditional_losses_1024116p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������k: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������k
 
_user_specified_nameinputs
�

�
C__inference_layer2_layer_call_and_return_conditional_losses_1024150

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:����������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
(__inference_layer4_layer_call_fn_1024774

inputs
unknown:
��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_layer4_layer_call_and_return_conditional_losses_1024184p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
,__inference_sequential_layer_call_fn_1024430
input_1
unknown:	k�
	unknown_0:	�
	unknown_1:
��
	unknown_2:	�
	unknown_3:
��
	unknown_4:	�
	unknown_5:
��
	unknown_6:	�
	unknown_7:
��
	unknown_8:	�
	unknown_9:	�`

unknown_10:`
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������`*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_sequential_layer_call_and_return_conditional_losses_1024374o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������``
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:���������k: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:���������k
!
_user_specified_name	input_1
�
C
'__inference_re_lu_layer_call_fn_1024810

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������`* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_re_lu_layer_call_and_return_conditional_losses_1024212`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������`"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������`:O K
'
_output_shapes
:���������`
 
_user_specified_nameinputs
�

�
C__inference_layer0_layer_call_and_return_conditional_losses_1024116

inputs1
matmul_readvariableop_resource:	k�.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	k�*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:����������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������k: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������k
 
_user_specified_nameinputs
�"
�
G__inference_sequential_layer_call_and_return_conditional_losses_1024374

inputs!
layer0_1024342:	k�
layer0_1024344:	�"
layer1_1024347:
��
layer1_1024349:	�"
layer2_1024352:
��
layer2_1024354:	�"
layer3_1024357:
��
layer3_1024359:	�"
layer4_1024362:
��
layer4_1024364:	�'
output_layer_1024367:	�`"
output_layer_1024369:`
identity��layer0/StatefulPartitionedCall�layer1/StatefulPartitionedCall�layer2/StatefulPartitionedCall�layer3/StatefulPartitionedCall�layer4/StatefulPartitionedCall�$output_layer/StatefulPartitionedCall�
layer0/StatefulPartitionedCallStatefulPartitionedCallinputslayer0_1024342layer0_1024344*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_layer0_layer_call_and_return_conditional_losses_1024116�
layer1/StatefulPartitionedCallStatefulPartitionedCall'layer0/StatefulPartitionedCall:output:0layer1_1024347layer1_1024349*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_layer1_layer_call_and_return_conditional_losses_1024133�
layer2/StatefulPartitionedCallStatefulPartitionedCall'layer1/StatefulPartitionedCall:output:0layer2_1024352layer2_1024354*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_layer2_layer_call_and_return_conditional_losses_1024150�
layer3/StatefulPartitionedCallStatefulPartitionedCall'layer2/StatefulPartitionedCall:output:0layer3_1024357layer3_1024359*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_layer3_layer_call_and_return_conditional_losses_1024167�
layer4/StatefulPartitionedCallStatefulPartitionedCall'layer3/StatefulPartitionedCall:output:0layer4_1024362layer4_1024364*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_layer4_layer_call_and_return_conditional_losses_1024184�
$output_layer/StatefulPartitionedCallStatefulPartitionedCall'layer4/StatefulPartitionedCall:output:0output_layer_1024367output_layer_1024369*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������`*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_output_layer_layer_call_and_return_conditional_losses_1024201�
re_lu/PartitionedCallPartitionedCall-output_layer/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������`* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_re_lu_layer_call_and_return_conditional_losses_1024212m
IdentityIdentityre_lu/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`�
NoOpNoOp^layer0/StatefulPartitionedCall^layer1/StatefulPartitionedCall^layer2/StatefulPartitionedCall^layer3/StatefulPartitionedCall^layer4/StatefulPartitionedCall%^output_layer/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:���������k: : : : : : : : : : : : 2@
layer0/StatefulPartitionedCalllayer0/StatefulPartitionedCall2@
layer1/StatefulPartitionedCalllayer1/StatefulPartitionedCall2@
layer2/StatefulPartitionedCalllayer2/StatefulPartitionedCall2@
layer3/StatefulPartitionedCalllayer3/StatefulPartitionedCall2@
layer4/StatefulPartitionedCalllayer4/StatefulPartitionedCall2L
$output_layer/StatefulPartitionedCall$output_layer/StatefulPartitionedCall:O K
'
_output_shapes
:���������k
 
_user_specified_nameinputs"�
L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
;
input_10
serving_default_input_1:0���������k9
re_lu0
StatefulPartitionedCall:0���������`tensorflow/serving/predict:߶
�
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer_with_weights-3
layer-3
layer_with_weights-4
layer-4
layer_with_weights-5
layer-5
layer-6
	variables
	trainable_variables

regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures
#_self_saveable_object_factories"
_tf_keras_sequential
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias
#_self_saveable_object_factories"
_tf_keras_layer
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
* &call_and_return_all_conditional_losses

!kernel
"bias
##_self_saveable_object_factories"
_tf_keras_layer
�
$	variables
%trainable_variables
&regularization_losses
'	keras_api
(__call__
*)&call_and_return_all_conditional_losses

*kernel
+bias
#,_self_saveable_object_factories"
_tf_keras_layer
�
-	variables
.trainable_variables
/regularization_losses
0	keras_api
1__call__
*2&call_and_return_all_conditional_losses

3kernel
4bias
#5_self_saveable_object_factories"
_tf_keras_layer
�
6	variables
7trainable_variables
8regularization_losses
9	keras_api
:__call__
*;&call_and_return_all_conditional_losses

<kernel
=bias
#>_self_saveable_object_factories"
_tf_keras_layer
�
?	variables
@trainable_variables
Aregularization_losses
B	keras_api
C__call__
*D&call_and_return_all_conditional_losses

Ekernel
Fbias
#G_self_saveable_object_factories"
_tf_keras_layer
�
H	variables
Itrainable_variables
Jregularization_losses
K	keras_api
L__call__
*M&call_and_return_all_conditional_losses
#N_self_saveable_object_factories"
_tf_keras_layer
v
0
1
!2
"3
*4
+5
36
47
<8
=9
E10
F11"
trackable_list_wrapper
v
0
1
!2
"3
*4
+5
36
47
<8
=9
E10
F11"
trackable_list_wrapper
 "
trackable_list_wrapper
�
Onon_trainable_variables

Players
Qmetrics
Rlayer_regularization_losses
Slayer_metrics
	variables
	trainable_variables

regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
Ttrace_0
Utrace_1
Vtrace_2
Wtrace_32�
,__inference_sequential_layer_call_fn_1024242
,__inference_sequential_layer_call_fn_1024562
,__inference_sequential_layer_call_fn_1024591
,__inference_sequential_layer_call_fn_1024430�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zTtrace_0zUtrace_1zVtrace_2zWtrace_3
�
Xtrace_0
Ytrace_1
Ztrace_2
[trace_32�
G__inference_sequential_layer_call_and_return_conditional_losses_1024638
G__inference_sequential_layer_call_and_return_conditional_losses_1024685
G__inference_sequential_layer_call_and_return_conditional_losses_1024465
G__inference_sequential_layer_call_and_return_conditional_losses_1024500�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zXtrace_0zYtrace_1zZtrace_2z[trace_3
�B�
"__inference__wrapped_model_1024098input_1"�
���
FullArgSpec
args� 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
j
\
_variables
]_iterations
^_learning_rate
__update_step_xla"
experimentalOptimizer
,
`serving_default"
signature_map
 "
trackable_dict_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
anon_trainable_variables

blayers
cmetrics
dlayer_regularization_losses
elayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
ftrace_02�
(__inference_layer0_layer_call_fn_1024694�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zftrace_0
�
gtrace_02�
C__inference_layer0_layer_call_and_return_conditional_losses_1024705�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zgtrace_0
 :	k�2layer0/kernel
:�2layer0/bias
 "
trackable_dict_wrapper
.
!0
"1"
trackable_list_wrapper
.
!0
"1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
hnon_trainable_variables

ilayers
jmetrics
klayer_regularization_losses
llayer_metrics
	variables
trainable_variables
regularization_losses
__call__
* &call_and_return_all_conditional_losses
& "call_and_return_conditional_losses"
_generic_user_object
�
mtrace_02�
(__inference_layer1_layer_call_fn_1024714�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zmtrace_0
�
ntrace_02�
C__inference_layer1_layer_call_and_return_conditional_losses_1024725�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zntrace_0
!:
��2layer1/kernel
:�2layer1/bias
 "
trackable_dict_wrapper
.
*0
+1"
trackable_list_wrapper
.
*0
+1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
onon_trainable_variables

players
qmetrics
rlayer_regularization_losses
slayer_metrics
$	variables
%trainable_variables
&regularization_losses
(__call__
*)&call_and_return_all_conditional_losses
&)"call_and_return_conditional_losses"
_generic_user_object
�
ttrace_02�
(__inference_layer2_layer_call_fn_1024734�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zttrace_0
�
utrace_02�
C__inference_layer2_layer_call_and_return_conditional_losses_1024745�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zutrace_0
!:
��2layer2/kernel
:�2layer2/bias
 "
trackable_dict_wrapper
.
30
41"
trackable_list_wrapper
.
30
41"
trackable_list_wrapper
 "
trackable_list_wrapper
�
vnon_trainable_variables

wlayers
xmetrics
ylayer_regularization_losses
zlayer_metrics
-	variables
.trainable_variables
/regularization_losses
1__call__
*2&call_and_return_all_conditional_losses
&2"call_and_return_conditional_losses"
_generic_user_object
�
{trace_02�
(__inference_layer3_layer_call_fn_1024754�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z{trace_0
�
|trace_02�
C__inference_layer3_layer_call_and_return_conditional_losses_1024765�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z|trace_0
!:
��2layer3/kernel
:�2layer3/bias
 "
trackable_dict_wrapper
.
<0
=1"
trackable_list_wrapper
.
<0
=1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
}non_trainable_variables

~layers
metrics
 �layer_regularization_losses
�layer_metrics
6	variables
7trainable_variables
8regularization_losses
:__call__
*;&call_and_return_all_conditional_losses
&;"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
(__inference_layer4_layer_call_fn_1024774�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
C__inference_layer4_layer_call_and_return_conditional_losses_1024785�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
!:
��2layer4/kernel
:�2layer4/bias
 "
trackable_dict_wrapper
.
E0
F1"
trackable_list_wrapper
.
E0
F1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
?	variables
@trainable_variables
Aregularization_losses
C__call__
*D&call_and_return_all_conditional_losses
&D"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
.__inference_output_layer_layer_call_fn_1024794�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
I__inference_output_layer_layer_call_and_return_conditional_losses_1024805�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
&:$	�`2output_layer/kernel
:`2output_layer/bias
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
H	variables
Itrainable_variables
Jregularization_losses
L__call__
*M&call_and_return_all_conditional_losses
&M"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
'__inference_re_lu_layer_call_fn_1024810�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
B__inference_re_lu_layer_call_and_return_conditional_losses_1024815�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
Q
0
1
2
3
4
5
6"
trackable_list_wrapper
8
�0
�1
�2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
,__inference_sequential_layer_call_fn_1024242input_1"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
,__inference_sequential_layer_call_fn_1024562inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
,__inference_sequential_layer_call_fn_1024591inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
,__inference_sequential_layer_call_fn_1024430input_1"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
G__inference_sequential_layer_call_and_return_conditional_losses_1024638inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
G__inference_sequential_layer_call_and_return_conditional_losses_1024685inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
G__inference_sequential_layer_call_and_return_conditional_losses_1024465input_1"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
G__inference_sequential_layer_call_and_return_conditional_losses_1024500input_1"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
'
]0"
trackable_list_wrapper
:	 2	iteration
: 2learning_rate
�2��
���
FullArgSpec2
args*�'
jself

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
�B�
%__inference_signature_wrapper_1024533input_1"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
(__inference_layer0_layer_call_fn_1024694inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_layer0_layer_call_and_return_conditional_losses_1024705inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
(__inference_layer1_layer_call_fn_1024714inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_layer1_layer_call_and_return_conditional_losses_1024725inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
(__inference_layer2_layer_call_fn_1024734inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_layer2_layer_call_and_return_conditional_losses_1024745inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
(__inference_layer3_layer_call_fn_1024754inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_layer3_layer_call_and_return_conditional_losses_1024765inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
(__inference_layer4_layer_call_fn_1024774inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_layer4_layer_call_and_return_conditional_losses_1024785inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
.__inference_output_layer_layer_call_fn_1024794inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
I__inference_output_layer_layer_call_and_return_conditional_losses_1024805inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
'__inference_re_lu_layer_call_fn_1024810inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
B__inference_re_lu_layer_call_and_return_conditional_losses_1024815inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
R
�	variables
�	keras_api

�total

�count"
_tf_keras_metric
R
�	variables
�	keras_api

�total

�count"
_tf_keras_metric
c
�	variables
�	keras_api

�total

�count
�
_fn_kwargs"
_tf_keras_metric
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper�
"__inference__wrapped_model_1024098o!"*+34<=EF0�-
&�#
!�
input_1���������k
� "-�*
(
re_lu�
re_lu���������`�
C__inference_layer0_layer_call_and_return_conditional_losses_1024705d/�,
%�"
 �
inputs���������k
� "-�*
#� 
tensor_0����������
� �
(__inference_layer0_layer_call_fn_1024694Y/�,
%�"
 �
inputs���������k
� ""�
unknown�����������
C__inference_layer1_layer_call_and_return_conditional_losses_1024725e!"0�-
&�#
!�
inputs����������
� "-�*
#� 
tensor_0����������
� �
(__inference_layer1_layer_call_fn_1024714Z!"0�-
&�#
!�
inputs����������
� ""�
unknown�����������
C__inference_layer2_layer_call_and_return_conditional_losses_1024745e*+0�-
&�#
!�
inputs����������
� "-�*
#� 
tensor_0����������
� �
(__inference_layer2_layer_call_fn_1024734Z*+0�-
&�#
!�
inputs����������
� ""�
unknown�����������
C__inference_layer3_layer_call_and_return_conditional_losses_1024765e340�-
&�#
!�
inputs����������
� "-�*
#� 
tensor_0����������
� �
(__inference_layer3_layer_call_fn_1024754Z340�-
&�#
!�
inputs����������
� ""�
unknown�����������
C__inference_layer4_layer_call_and_return_conditional_losses_1024785e<=0�-
&�#
!�
inputs����������
� "-�*
#� 
tensor_0����������
� �
(__inference_layer4_layer_call_fn_1024774Z<=0�-
&�#
!�
inputs����������
� ""�
unknown�����������
I__inference_output_layer_layer_call_and_return_conditional_losses_1024805dEF0�-
&�#
!�
inputs����������
� ",�)
"�
tensor_0���������`
� �
.__inference_output_layer_layer_call_fn_1024794YEF0�-
&�#
!�
inputs����������
� "!�
unknown���������`�
B__inference_re_lu_layer_call_and_return_conditional_losses_1024815_/�,
%�"
 �
inputs���������`
� ",�)
"�
tensor_0���������`
� 
'__inference_re_lu_layer_call_fn_1024810T/�,
%�"
 �
inputs���������`
� "!�
unknown���������`�
G__inference_sequential_layer_call_and_return_conditional_losses_1024465v!"*+34<=EF8�5
.�+
!�
input_1���������k
p 

 
� ",�)
"�
tensor_0���������`
� �
G__inference_sequential_layer_call_and_return_conditional_losses_1024500v!"*+34<=EF8�5
.�+
!�
input_1���������k
p

 
� ",�)
"�
tensor_0���������`
� �
G__inference_sequential_layer_call_and_return_conditional_losses_1024638u!"*+34<=EF7�4
-�*
 �
inputs���������k
p 

 
� ",�)
"�
tensor_0���������`
� �
G__inference_sequential_layer_call_and_return_conditional_losses_1024685u!"*+34<=EF7�4
-�*
 �
inputs���������k
p

 
� ",�)
"�
tensor_0���������`
� �
,__inference_sequential_layer_call_fn_1024242k!"*+34<=EF8�5
.�+
!�
input_1���������k
p 

 
� "!�
unknown���������`�
,__inference_sequential_layer_call_fn_1024430k!"*+34<=EF8�5
.�+
!�
input_1���������k
p

 
� "!�
unknown���������`�
,__inference_sequential_layer_call_fn_1024562j!"*+34<=EF7�4
-�*
 �
inputs���������k
p 

 
� "!�
unknown���������`�
,__inference_sequential_layer_call_fn_1024591j!"*+34<=EF7�4
-�*
 �
inputs���������k
p

 
� "!�
unknown���������`�
%__inference_signature_wrapper_1024533z!"*+34<=EF;�8
� 
1�.
,
input_1!�
input_1���������k"-�*
(
re_lu�
re_lu���������`