•ѕ

ЋЫ
B
AssignVariableOp
resource
value"dtype"
dtypetypeИ
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
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(И
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
dtypetypeИ
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
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
9
Softmax
logits"T
softmax"T"
Ttype:
2
Њ
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
executor_typestring И
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
Ц
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 И"serve*2.5.02v2.5.0-rc3-213-ga4dfb8d1a718ыэ
u
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	Р2*
shared_namedense/kernel
n
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes
:	Р2*
dtype0
l

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*
shared_name
dense/bias
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes
:2*
dtype0
x
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:22*
shared_namedense_1/kernel
q
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
_output_shapes

:22*
dtype0
p
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*
shared_namedense_1/bias
i
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes
:2*
dtype0
x
dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2
*
shared_namedense_2/kernel
q
"dense_2/kernel/Read/ReadVariableOpReadVariableOpdense_2/kernel*
_output_shapes

:2
*
dtype0
p
dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_namedense_2/bias
i
 dense_2/bias/Read/ReadVariableOpReadVariableOpdense_2/bias*
_output_shapes
:
*
dtype0
l
RMSprop/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_nameRMSprop/iter
e
 RMSprop/iter/Read/ReadVariableOpReadVariableOpRMSprop/iter*
_output_shapes
: *
dtype0	
n
RMSprop/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameRMSprop/decay
g
!RMSprop/decay/Read/ReadVariableOpReadVariableOpRMSprop/decay*
_output_shapes
: *
dtype0
~
RMSprop/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameRMSprop/learning_rate
w
)RMSprop/learning_rate/Read/ReadVariableOpReadVariableOpRMSprop/learning_rate*
_output_shapes
: *
dtype0
t
RMSprop/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_nameRMSprop/momentum
m
$RMSprop/momentum/Read/ReadVariableOpReadVariableOpRMSprop/momentum*
_output_shapes
: *
dtype0
j
RMSprop/rhoVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameRMSprop/rho
c
RMSprop/rho/Read/ReadVariableOpReadVariableOpRMSprop/rho*
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
Н
RMSprop/dense/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	Р2*)
shared_nameRMSprop/dense/kernel/rms
Ж
,RMSprop/dense/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/dense/kernel/rms*
_output_shapes
:	Р2*
dtype0
Д
RMSprop/dense/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*'
shared_nameRMSprop/dense/bias/rms
}
*RMSprop/dense/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/dense/bias/rms*
_output_shapes
:2*
dtype0
Р
RMSprop/dense_1/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:22*+
shared_nameRMSprop/dense_1/kernel/rms
Й
.RMSprop/dense_1/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/dense_1/kernel/rms*
_output_shapes

:22*
dtype0
И
RMSprop/dense_1/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*)
shared_nameRMSprop/dense_1/bias/rms
Б
,RMSprop/dense_1/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/dense_1/bias/rms*
_output_shapes
:2*
dtype0
Р
RMSprop/dense_2/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2
*+
shared_nameRMSprop/dense_2/kernel/rms
Й
.RMSprop/dense_2/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/dense_2/kernel/rms*
_output_shapes

:2
*
dtype0
И
RMSprop/dense_2/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*)
shared_nameRMSprop/dense_2/bias/rms
Б
,RMSprop/dense_2/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/dense_2/bias/rms*
_output_shapes
:
*
dtype0

NoOpNoOp
І$
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*в#
valueЎ#B’# Bќ#
Н
layer-0
layer-1
layer-2
layer_with_weights-0
layer-3
layer_with_weights-1
layer-4
layer_with_weights-2
layer-5
	optimizer
regularization_losses
	trainable_variables

	variables
	keras_api

signatures

_init_input_shape
R
regularization_losses
trainable_variables
	variables
	keras_api
R
regularization_losses
trainable_variables
	variables
	keras_api
h

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
h

kernel
bias
regularization_losses
trainable_variables
 	variables
!	keras_api
h

"kernel
#bias
$regularization_losses
%trainable_variables
&	variables
'	keras_api
Б
(iter
	)decay
*learning_rate
+momentum
,rho	rmsV	rmsW	rmsX	rmsY	"rmsZ	#rms[
 
*
0
1
2
3
"4
#5
*
0
1
2
3
"4
#5
≠
-layer_metrics
.metrics
/non_trainable_variables
regularization_losses
	trainable_variables

0layers
1layer_regularization_losses

	variables
 
 
 
 
 
≠
2layer_metrics
3metrics
4non_trainable_variables
regularization_losses
trainable_variables

5layers
6layer_regularization_losses
	variables
 
 
 
≠
7layer_metrics
8metrics
9non_trainable_variables
regularization_losses
trainable_variables

:layers
;layer_regularization_losses
	variables
XV
VARIABLE_VALUEdense/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
dense/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
≠
<layer_metrics
=metrics
>non_trainable_variables
regularization_losses
trainable_variables

?layers
@layer_regularization_losses
	variables
ZX
VARIABLE_VALUEdense_1/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_1/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
≠
Alayer_metrics
Bmetrics
Cnon_trainable_variables
regularization_losses
trainable_variables

Dlayers
Elayer_regularization_losses
 	variables
ZX
VARIABLE_VALUEdense_2/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_2/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

"0
#1

"0
#1
≠
Flayer_metrics
Gmetrics
Hnon_trainable_variables
$regularization_losses
%trainable_variables

Ilayers
Jlayer_regularization_losses
&	variables
KI
VARIABLE_VALUERMSprop/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUERMSprop/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUERMSprop/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUERMSprop/momentum-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUERMSprop/rho(optimizer/rho/.ATTRIBUTES/VARIABLE_VALUE
 

K0
L1
 
*
0
1
2
3
4
5
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
4
	Mtotal
	Ncount
O	variables
P	keras_api
D
	Qtotal
	Rcount
S
_fn_kwargs
T	variables
U	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

M0
N1

O	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

Q0
R1

T	variables
ГА
VARIABLE_VALUERMSprop/dense/kernel/rmsTlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUERMSprop/dense/bias/rmsRlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
ЕВ
VARIABLE_VALUERMSprop/dense_1/kernel/rmsTlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
А~
VARIABLE_VALUERMSprop/dense_1/bias/rmsRlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
ЕВ
VARIABLE_VALUERMSprop/dense_2/kernel/rmsTlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
А~
VARIABLE_VALUERMSprop/dense_2/bias/rmsRlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
И
serving_default_imagePlaceholder*/
_output_shapes
:€€€€€€€€€*
dtype0*$
shape:€€€€€€€€€
Т
StatefulPartitionedCallStatefulPartitionedCallserving_default_imagedense/kernel
dense/biasdense_1/kerneldense_1/biasdense_2/kerneldense_2/bias*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€
*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8В *-
f(R&
$__inference_signature_wrapper_222690
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
ї
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOp"dense_2/kernel/Read/ReadVariableOp dense_2/bias/Read/ReadVariableOp RMSprop/iter/Read/ReadVariableOp!RMSprop/decay/Read/ReadVariableOp)RMSprop/learning_rate/Read/ReadVariableOp$RMSprop/momentum/Read/ReadVariableOpRMSprop/rho/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp,RMSprop/dense/kernel/rms/Read/ReadVariableOp*RMSprop/dense/bias/rms/Read/ReadVariableOp.RMSprop/dense_1/kernel/rms/Read/ReadVariableOp,RMSprop/dense_1/bias/rms/Read/ReadVariableOp.RMSprop/dense_2/kernel/rms/Read/ReadVariableOp,RMSprop/dense_2/bias/rms/Read/ReadVariableOpConst*"
Tin
2	*
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
GPU 2J 8В *(
f#R!
__inference__traced_save_223210
Т
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense/kernel
dense/biasdense_1/kerneldense_1/biasdense_2/kerneldense_2/biasRMSprop/iterRMSprop/decayRMSprop/learning_rateRMSprop/momentumRMSprop/rhototalcounttotal_1count_1RMSprop/dense/kernel/rmsRMSprop/dense/bias/rmsRMSprop/dense_1/kernel/rmsRMSprop/dense_1/bias/rmsRMSprop/dense_2/kernel/rmsRMSprop/dense_2/bias/rms*!
Tin
2*
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
GPU 2J 8В *+
f&R$
"__inference__traced_restore_223283Ў†
Ђ

ф
C__inference_dense_1_layer_call_and_return_conditional_losses_223095

inputs0
matmul_readvariableop_resource:22-
biasadd_readvariableop_resource:2
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:22*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€22
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:2*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€22	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€22
ReluЧ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€22

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€2: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€2
 
_user_specified_nameinputs
ў
Ж
-__inference_custom_model_layer_call_fn_222623	
image
unknown:	Р2
	unknown_0:2
	unknown_1:22
	unknown_2:2
	unknown_3:2

	unknown_4:

identityИҐStatefulPartitionedCallЂ
StatefulPartitionedCallStatefulPartitionedCallimageunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€
*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_custom_model_layer_call_and_return_conditional_losses_2225912
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':€€€€€€€€€: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:V R
/
_output_shapes
:€€€€€€€€€

_user_specified_nameimage
Щ
Х
(__inference_dense_2_layer_call_fn_223124

inputs
unknown:2

	unknown_0:

identityИҐStatefulPartitionedCallу
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_2224712
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€2: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€2
 
_user_specified_nameinputs
Ў

i
J__inference_gaussian_noise_layer_call_and_return_conditional_losses_222548

inputs
identityИD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapem
random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2
random_normal/meanq
random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL>2
random_normal/stddev’
"random_normal/RandomStandardNormalRandomStandardNormalShape:output:0*
T0*/
_output_shapes
:€€€€€€€€€*
dtype0*
seed±€е)*
seed2Пнa2$
"random_normal/RandomStandardNormal≥
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*/
_output_shapes
:€€€€€€€€€2
random_normal/mulУ
random_normalAddrandom_normal/mul:z:0random_normal/mean:output:0*
T0*/
_output_shapes
:€€€€€€€€€2
random_normalh
addAddV2inputsrandom_normal:z:0*
T0*/
_output_shapes
:€€€€€€€€€2
addc
IdentityIdentityadd:z:0*
T0*/
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€:W S
/
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
ў

i
J__inference_gaussian_noise_layer_call_and_return_conditional_losses_223043

inputs
identityИD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapem
random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2
random_normal/meanq
random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL>2
random_normal/stddev÷
"random_normal/RandomStandardNormalRandomStandardNormalShape:output:0*
T0*/
_output_shapes
:€€€€€€€€€*
dtype0*
seed±€е)*
seed2Хыс2$
"random_normal/RandomStandardNormal≥
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*/
_output_shapes
:€€€€€€€€€2
random_normal/mulУ
random_normalAddrandom_normal/mul:z:0random_normal/mean:output:0*
T0*/
_output_shapes
:€€€€€€€€€2
random_normalh
addAddV2inputsrandom_normal:z:0*
T0*/
_output_shapes
:€€€€€€€€€2
addc
IdentityIdentityadd:z:0*
T0*/
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€:W S
/
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
Ю
f
J__inference_gaussian_noise_layer_call_and_return_conditional_losses_223032

inputs
identityb
IdentityIdentityinputs*
T0*/
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€:W S
/
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
ў
Ж
-__inference_custom_model_layer_call_fn_222493	
image
unknown:	Р2
	unknown_0:2
	unknown_1:22
	unknown_2:2
	unknown_3:2

	unknown_4:

identityИҐStatefulPartitionedCallЂ
StatefulPartitionedCallStatefulPartitionedCallimageunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€
*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_custom_model_layer_call_and_return_conditional_losses_2224782
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':€€€€€€€€€: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:V R
/
_output_shapes
:€€€€€€€€€

_user_specified_nameimage
е
_
C__inference_flatten_layer_call_and_return_conditional_losses_223059

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:€€€€€€€€€Р2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:€€€€€€€€€Р2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€:W S
/
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
С[
¶
"__inference__traced_restore_223283
file_prefix0
assignvariableop_dense_kernel:	Р2+
assignvariableop_1_dense_bias:23
!assignvariableop_2_dense_1_kernel:22-
assignvariableop_3_dense_1_bias:23
!assignvariableop_4_dense_2_kernel:2
-
assignvariableop_5_dense_2_bias:
)
assignvariableop_6_rmsprop_iter:	 *
 assignvariableop_7_rmsprop_decay: 2
(assignvariableop_8_rmsprop_learning_rate: -
#assignvariableop_9_rmsprop_momentum: )
assignvariableop_10_rmsprop_rho: #
assignvariableop_11_total: #
assignvariableop_12_count: %
assignvariableop_13_total_1: %
assignvariableop_14_count_1: ?
,assignvariableop_15_rmsprop_dense_kernel_rms:	Р28
*assignvariableop_16_rmsprop_dense_bias_rms:2@
.assignvariableop_17_rmsprop_dense_1_kernel_rms:22:
,assignvariableop_18_rmsprop_dense_1_bias_rms:2@
.assignvariableop_19_rmsprop_dense_2_kernel_rms:2
:
,assignvariableop_20_rmsprop_dense_2_bias_rms:

identity_22ИҐAssignVariableOpҐAssignVariableOp_1ҐAssignVariableOp_10ҐAssignVariableOp_11ҐAssignVariableOp_12ҐAssignVariableOp_13ҐAssignVariableOp_14ҐAssignVariableOp_15ҐAssignVariableOp_16ҐAssignVariableOp_17ҐAssignVariableOp_18ҐAssignVariableOp_19ҐAssignVariableOp_2ҐAssignVariableOp_20ҐAssignVariableOp_3ҐAssignVariableOp_4ҐAssignVariableOp_5ҐAssignVariableOp_6ҐAssignVariableOp_7ҐAssignVariableOp_8ҐAssignVariableOp_9Ђ
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Ј

value≠
B™
B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB(optimizer/rho/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesЇ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value6B4B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesЩ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*l
_output_shapesZ
X::::::::::::::::::::::*$
dtypes
2	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

IdentityЬ
AssignVariableOpAssignVariableOpassignvariableop_dense_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1Ґ
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2¶
AssignVariableOp_2AssignVariableOp!assignvariableop_2_dense_1_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3§
AssignVariableOp_3AssignVariableOpassignvariableop_3_dense_1_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4¶
AssignVariableOp_4AssignVariableOp!assignvariableop_4_dense_2_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5§
AssignVariableOp_5AssignVariableOpassignvariableop_5_dense_2_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_6§
AssignVariableOp_6AssignVariableOpassignvariableop_6_rmsprop_iterIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7•
AssignVariableOp_7AssignVariableOp assignvariableop_7_rmsprop_decayIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8≠
AssignVariableOp_8AssignVariableOp(assignvariableop_8_rmsprop_learning_rateIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9®
AssignVariableOp_9AssignVariableOp#assignvariableop_9_rmsprop_momentumIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10І
AssignVariableOp_10AssignVariableOpassignvariableop_10_rmsprop_rhoIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11°
AssignVariableOp_11AssignVariableOpassignvariableop_11_totalIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12°
AssignVariableOp_12AssignVariableOpassignvariableop_12_countIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13£
AssignVariableOp_13AssignVariableOpassignvariableop_13_total_1Identity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14£
AssignVariableOp_14AssignVariableOpassignvariableop_14_count_1Identity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15і
AssignVariableOp_15AssignVariableOp,assignvariableop_15_rmsprop_dense_kernel_rmsIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16≤
AssignVariableOp_16AssignVariableOp*assignvariableop_16_rmsprop_dense_bias_rmsIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17ґ
AssignVariableOp_17AssignVariableOp.assignvariableop_17_rmsprop_dense_1_kernel_rmsIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18і
AssignVariableOp_18AssignVariableOp,assignvariableop_18_rmsprop_dense_1_bias_rmsIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19ґ
AssignVariableOp_19AssignVariableOp.assignvariableop_19_rmsprop_dense_2_kernel_rmsIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20і
AssignVariableOp_20AssignVariableOp,assignvariableop_20_rmsprop_dense_2_bias_rmsIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_209
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpђ
Identity_21Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_21Я
Identity_22IdentityIdentity_21:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_22"#
identity_22Identity_22:output:0*?
_input_shapes.
,: : : : : : : : : : : : : : : : : : : : : : 2$
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
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202(
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
Є(
к
!__inference__wrapped_model_222405	
imageD
1custom_model_dense_matmul_readvariableop_resource:	Р2@
2custom_model_dense_biasadd_readvariableop_resource:2E
3custom_model_dense_1_matmul_readvariableop_resource:22B
4custom_model_dense_1_biasadd_readvariableop_resource:2E
3custom_model_dense_2_matmul_readvariableop_resource:2
B
4custom_model_dense_2_biasadd_readvariableop_resource:

identityИҐ)custom_model/dense/BiasAdd/ReadVariableOpҐ(custom_model/dense/MatMul/ReadVariableOpҐ+custom_model/dense_1/BiasAdd/ReadVariableOpҐ*custom_model/dense_1/MatMul/ReadVariableOpҐ+custom_model/dense_2/BiasAdd/ReadVariableOpҐ*custom_model/dense_2/MatMul/ReadVariableOpЙ
custom_model/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€  2
custom_model/flatten/Const¶
custom_model/flatten/ReshapeReshapeimage#custom_model/flatten/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€Р2
custom_model/flatten/Reshape«
(custom_model/dense/MatMul/ReadVariableOpReadVariableOp1custom_model_dense_matmul_readvariableop_resource*
_output_shapes
:	Р2*
dtype02*
(custom_model/dense/MatMul/ReadVariableOpЋ
custom_model/dense/MatMulMatMul%custom_model/flatten/Reshape:output:00custom_model/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€22
custom_model/dense/MatMul≈
)custom_model/dense/BiasAdd/ReadVariableOpReadVariableOp2custom_model_dense_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02+
)custom_model/dense/BiasAdd/ReadVariableOpЌ
custom_model/dense/BiasAddBiasAdd#custom_model/dense/MatMul:product:01custom_model/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€22
custom_model/dense/BiasAddС
custom_model/dense/ReluRelu#custom_model/dense/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€22
custom_model/dense/Reluћ
*custom_model/dense_1/MatMul/ReadVariableOpReadVariableOp3custom_model_dense_1_matmul_readvariableop_resource*
_output_shapes

:22*
dtype02,
*custom_model/dense_1/MatMul/ReadVariableOp—
custom_model/dense_1/MatMulMatMul%custom_model/dense/Relu:activations:02custom_model/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€22
custom_model/dense_1/MatMulЋ
+custom_model/dense_1/BiasAdd/ReadVariableOpReadVariableOp4custom_model_dense_1_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02-
+custom_model/dense_1/BiasAdd/ReadVariableOp’
custom_model/dense_1/BiasAddBiasAdd%custom_model/dense_1/MatMul:product:03custom_model/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€22
custom_model/dense_1/BiasAddЧ
custom_model/dense_1/ReluRelu%custom_model/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€22
custom_model/dense_1/Reluћ
*custom_model/dense_2/MatMul/ReadVariableOpReadVariableOp3custom_model_dense_2_matmul_readvariableop_resource*
_output_shapes

:2
*
dtype02,
*custom_model/dense_2/MatMul/ReadVariableOp”
custom_model/dense_2/MatMulMatMul'custom_model/dense_1/Relu:activations:02custom_model/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€
2
custom_model/dense_2/MatMulЋ
+custom_model/dense_2/BiasAdd/ReadVariableOpReadVariableOp4custom_model_dense_2_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02-
+custom_model/dense_2/BiasAdd/ReadVariableOp’
custom_model/dense_2/BiasAddBiasAdd%custom_model/dense_2/MatMul:product:03custom_model/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€
2
custom_model/dense_2/BiasAdd†
custom_model/dense_2/SoftmaxSoftmax%custom_model/dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€
2
custom_model/dense_2/SoftmaxЗ
IdentityIdentity&custom_model/dense_2/Softmax:softmax:0*^custom_model/dense/BiasAdd/ReadVariableOp)^custom_model/dense/MatMul/ReadVariableOp,^custom_model/dense_1/BiasAdd/ReadVariableOp+^custom_model/dense_1/MatMul/ReadVariableOp,^custom_model/dense_2/BiasAdd/ReadVariableOp+^custom_model/dense_2/MatMul/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':€€€€€€€€€: : : : : : 2V
)custom_model/dense/BiasAdd/ReadVariableOp)custom_model/dense/BiasAdd/ReadVariableOp2T
(custom_model/dense/MatMul/ReadVariableOp(custom_model/dense/MatMul/ReadVariableOp2Z
+custom_model/dense_1/BiasAdd/ReadVariableOp+custom_model/dense_1/BiasAdd/ReadVariableOp2X
*custom_model/dense_1/MatMul/ReadVariableOp*custom_model/dense_1/MatMul/ReadVariableOp2Z
+custom_model/dense_2/BiasAdd/ReadVariableOp+custom_model/dense_2/BiasAdd/ReadVariableOp2X
*custom_model/dense_2/MatMul/ReadVariableOp*custom_model/dense_2/MatMul/ReadVariableOp:V R
/
_output_shapes
:€€€€€€€€€

_user_specified_nameimage
≠

у
A__inference_dense_layer_call_and_return_conditional_losses_222437

inputs1
matmul_readvariableop_resource:	Р2-
biasadd_readvariableop_resource:2
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	Р2*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€22
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:2*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€22	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€22
ReluЧ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€22

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€Р: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:€€€€€€€€€Р
 
_user_specified_nameinputs
ОЖ
∞
__inference_train_step_223028

data_0

data_1D
1custom_model_dense_matmul_readvariableop_resource:	Р2@
2custom_model_dense_biasadd_readvariableop_resource:2E
3custom_model_dense_1_matmul_readvariableop_resource:22B
4custom_model_dense_1_biasadd_readvariableop_resource:2E
3custom_model_dense_2_matmul_readvariableop_resource:2
B
4custom_model_dense_2_biasadd_readvariableop_resource:
&
assignaddvariableop_resource: (
assignaddvariableop_1_resource: .
$rmsprop_cast_readvariableop_resource: 0
&rmsprop_cast_1_readvariableop_resource: 0
&rmsprop_cast_2_readvariableop_resource: E
2rmsprop_rmsprop_update_mul_readvariableop_resource:	Р2B
4rmsprop_rmsprop_update_1_mul_readvariableop_resource:2F
4rmsprop_rmsprop_update_2_mul_readvariableop_resource:22B
4rmsprop_rmsprop_update_3_mul_readvariableop_resource:2F
4rmsprop_rmsprop_update_4_mul_readvariableop_resource:2
B
4rmsprop_rmsprop_update_5_mul_readvariableop_resource:
6
,rmsprop_rmsprop_assignaddvariableop_resource:	 (
assignaddvariableop_2_resource: (
assignaddvariableop_3_resource: 

identity_2

identity_3ИҐAssignAddVariableOpҐAssignAddVariableOp_1ҐAssignAddVariableOp_2ҐAssignAddVariableOp_3ҐRMSprop/Cast/ReadVariableOpҐRMSprop/Cast_1/ReadVariableOpҐRMSprop/Cast_2/ReadVariableOpҐ#RMSprop/RMSprop/AssignAddVariableOpҐ'RMSprop/RMSprop/update/AssignVariableOpҐ)RMSprop/RMSprop/update/AssignVariableOp_1Ґ%RMSprop/RMSprop/update/ReadVariableOpҐ*RMSprop/RMSprop/update/Sqrt/ReadVariableOpҐ)RMSprop/RMSprop/update/mul/ReadVariableOpҐ)RMSprop/RMSprop/update_1/AssignVariableOpҐ+RMSprop/RMSprop/update_1/AssignVariableOp_1Ґ'RMSprop/RMSprop/update_1/ReadVariableOpҐ,RMSprop/RMSprop/update_1/Sqrt/ReadVariableOpҐ+RMSprop/RMSprop/update_1/mul/ReadVariableOpҐ)RMSprop/RMSprop/update_2/AssignVariableOpҐ+RMSprop/RMSprop/update_2/AssignVariableOp_1Ґ'RMSprop/RMSprop/update_2/ReadVariableOpҐ,RMSprop/RMSprop/update_2/Sqrt/ReadVariableOpҐ+RMSprop/RMSprop/update_2/mul/ReadVariableOpҐ)RMSprop/RMSprop/update_3/AssignVariableOpҐ+RMSprop/RMSprop/update_3/AssignVariableOp_1Ґ'RMSprop/RMSprop/update_3/ReadVariableOpҐ,RMSprop/RMSprop/update_3/Sqrt/ReadVariableOpҐ+RMSprop/RMSprop/update_3/mul/ReadVariableOpҐ)RMSprop/RMSprop/update_4/AssignVariableOpҐ+RMSprop/RMSprop/update_4/AssignVariableOp_1Ґ'RMSprop/RMSprop/update_4/ReadVariableOpҐ,RMSprop/RMSprop/update_4/Sqrt/ReadVariableOpҐ+RMSprop/RMSprop/update_4/mul/ReadVariableOpҐ)RMSprop/RMSprop/update_5/AssignVariableOpҐ+RMSprop/RMSprop/update_5/AssignVariableOp_1Ґ'RMSprop/RMSprop/update_5/ReadVariableOpҐ,RMSprop/RMSprop/update_5/Sqrt/ReadVariableOpҐ+RMSprop/RMSprop/update_5/mul/ReadVariableOpҐ)custom_model/dense/BiasAdd/ReadVariableOpҐ(custom_model/dense/MatMul/ReadVariableOpҐ+custom_model/dense_1/BiasAdd/ReadVariableOpҐ*custom_model/dense_1/MatMul/ReadVariableOpҐ+custom_model/dense_2/BiasAdd/ReadVariableOpҐ*custom_model/dense_2/MatMul/ReadVariableOpҐdiv_no_nan/ReadVariableOpҐdiv_no_nan/ReadVariableOp_1Ґdiv_no_nan_1/ReadVariableOpҐdiv_no_nan_1/ReadVariableOp_1Я
!custom_model/gaussian_noise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"             2#
!custom_model/gaussian_noise/Shape•
.custom_model/gaussian_noise/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    20
.custom_model/gaussian_noise/random_normal/mean©
0custom_model/gaussian_noise/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL>22
0custom_model/gaussian_noise/random_normal/stddev°
>custom_model/gaussian_noise/random_normal/RandomStandardNormalRandomStandardNormal*custom_model/gaussian_noise/Shape:output:0*
T0*&
_output_shapes
: *
dtype0*
seed±€е)*
seed2…йї2@
>custom_model/gaussian_noise/random_normal/RandomStandardNormalЪ
-custom_model/gaussian_noise/random_normal/mulMulGcustom_model/gaussian_noise/random_normal/RandomStandardNormal:output:09custom_model/gaussian_noise/random_normal/stddev:output:0*
T0*&
_output_shapes
: 2/
-custom_model/gaussian_noise/random_normal/mulъ
)custom_model/gaussian_noise/random_normalAdd1custom_model/gaussian_noise/random_normal/mul:z:07custom_model/gaussian_noise/random_normal/mean:output:0*
T0*&
_output_shapes
: 2+
)custom_model/gaussian_noise/random_normal≥
custom_model/gaussian_noise/addAddV2data_0-custom_model/gaussian_noise/random_normal:z:0*
T0*&
_output_shapes
: 2!
custom_model/gaussian_noise/addЙ
custom_model/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€  2
custom_model/flatten/Constї
custom_model/flatten/ReshapeReshape#custom_model/gaussian_noise/add:z:0#custom_model/flatten/Const:output:0*
T0*
_output_shapes
:	 Р2
custom_model/flatten/Reshape«
(custom_model/dense/MatMul/ReadVariableOpReadVariableOp1custom_model_dense_matmul_readvariableop_resource*
_output_shapes
:	Р2*
dtype02*
(custom_model/dense/MatMul/ReadVariableOp¬
custom_model/dense/MatMulMatMul%custom_model/flatten/Reshape:output:00custom_model/dense/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

: 22
custom_model/dense/MatMul≈
)custom_model/dense/BiasAdd/ReadVariableOpReadVariableOp2custom_model_dense_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02+
)custom_model/dense/BiasAdd/ReadVariableOpƒ
custom_model/dense/BiasAddBiasAdd#custom_model/dense/MatMul:product:01custom_model/dense/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

: 22
custom_model/dense/BiasAddИ
custom_model/dense/ReluRelu#custom_model/dense/BiasAdd:output:0*
T0*
_output_shapes

: 22
custom_model/dense/Reluћ
*custom_model/dense_1/MatMul/ReadVariableOpReadVariableOp3custom_model_dense_1_matmul_readvariableop_resource*
_output_shapes

:22*
dtype02,
*custom_model/dense_1/MatMul/ReadVariableOp»
custom_model/dense_1/MatMulMatMul%custom_model/dense/Relu:activations:02custom_model/dense_1/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

: 22
custom_model/dense_1/MatMulЋ
+custom_model/dense_1/BiasAdd/ReadVariableOpReadVariableOp4custom_model_dense_1_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02-
+custom_model/dense_1/BiasAdd/ReadVariableOpћ
custom_model/dense_1/BiasAddBiasAdd%custom_model/dense_1/MatMul:product:03custom_model/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

: 22
custom_model/dense_1/BiasAddО
custom_model/dense_1/ReluRelu%custom_model/dense_1/BiasAdd:output:0*
T0*
_output_shapes

: 22
custom_model/dense_1/Reluћ
*custom_model/dense_2/MatMul/ReadVariableOpReadVariableOp3custom_model_dense_2_matmul_readvariableop_resource*
_output_shapes

:2
*
dtype02,
*custom_model/dense_2/MatMul/ReadVariableOp 
custom_model/dense_2/MatMulMatMul'custom_model/dense_1/Relu:activations:02custom_model/dense_2/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

: 
2
custom_model/dense_2/MatMulЋ
+custom_model/dense_2/BiasAdd/ReadVariableOpReadVariableOp4custom_model_dense_2_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02-
+custom_model/dense_2/BiasAdd/ReadVariableOpћ
custom_model/dense_2/BiasAddBiasAdd%custom_model/dense_2/MatMul:product:03custom_model/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

: 
2
custom_model/dense_2/BiasAddЧ
custom_model/dense_2/SoftmaxSoftmax%custom_model/dense_2/BiasAdd:output:0*
T0*
_output_shapes

: 
2
custom_model/dense_2/Softmaxk
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2
ExpandDims/dimp

ExpandDims
ExpandDimsdata_1ExpandDims/dim:output:0*
T0*
_output_shapes

: 2

ExpandDims°
$sparse_categorical_crossentropy/CastCastExpandDims:output:0*

DstT0	*

SrcT0*
_output_shapes

: 2&
$sparse_categorical_crossentropy/CastЯ
%sparse_categorical_crossentropy/ShapeConst*
_output_shapes
:*
dtype0*
valueB"    
   2'
%sparse_categorical_crossentropy/Shape±
-sparse_categorical_crossentropy/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€2/
-sparse_categorical_crossentropy/Reshape/shapeд
'sparse_categorical_crossentropy/ReshapeReshape(sparse_categorical_crossentropy/Cast:y:06sparse_categorical_crossentropy/Reshape/shape:output:0*
T0	*
_output_shapes
: 2)
'sparse_categorical_crossentropy/Reshapeљ
3sparse_categorical_crossentropy/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€25
3sparse_categorical_crossentropy/strided_slice/stackЄ
5sparse_categorical_crossentropy/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 27
5sparse_categorical_crossentropy/strided_slice/stack_1Є
5sparse_categorical_crossentropy/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5sparse_categorical_crossentropy/strided_slice/stack_2Ґ
-sparse_categorical_crossentropy/strided_sliceStridedSlice.sparse_categorical_crossentropy/Shape:output:0<sparse_categorical_crossentropy/strided_slice/stack:output:0>sparse_categorical_crossentropy/strided_slice/stack_1:output:0>sparse_categorical_crossentropy/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2/
-sparse_categorical_crossentropy/strided_slice±
1sparse_categorical_crossentropy/Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€23
1sparse_categorical_crossentropy/Reshape_1/shape/0М
/sparse_categorical_crossentropy/Reshape_1/shapePack:sparse_categorical_crossentropy/Reshape_1/shape/0:output:06sparse_categorical_crossentropy/strided_slice:output:0*
N*
T0*
_output_shapes
:21
/sparse_categorical_crossentropy/Reshape_1/shapeл
)sparse_categorical_crossentropy/Reshape_1Reshape%custom_model/dense_2/BiasAdd:output:08sparse_categorical_crossentropy/Reshape_1/shape:output:0*
T0*
_output_shapes

: 
2+
)sparse_categorical_crossentropy/Reshape_1а
Isparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/ShapeConst*
_output_shapes
:*
dtype0*
valueB: 2K
Isparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/ShapeО
gsparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits#SparseSoftmaxCrossEntropyWithLogits2sparse_categorical_crossentropy/Reshape_1:output:00sparse_categorical_crossentropy/Reshape:output:0*
T0*$
_output_shapes
: : 
2i
gsparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogitsѓ
3sparse_categorical_crossentropy/weighted_loss/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?25
3sparse_categorical_crossentropy/weighted_loss/Constј
1sparse_categorical_crossentropy/weighted_loss/MulMulnsparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits:loss:0<sparse_categorical_crossentropy/weighted_loss/Const:output:0*
T0*
_output_shapes
: 23
1sparse_categorical_crossentropy/weighted_loss/MulЄ
5sparse_categorical_crossentropy/weighted_loss/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 27
5sparse_categorical_crossentropy/weighted_loss/Const_1Е
1sparse_categorical_crossentropy/weighted_loss/SumSum5sparse_categorical_crossentropy/weighted_loss/Mul:z:0>sparse_categorical_crossentropy/weighted_loss/Const_1:output:0*
T0*
_output_shapes
: 23
1sparse_categorical_crossentropy/weighted_loss/SumЇ
:sparse_categorical_crossentropy/weighted_loss/num_elementsConst*
_output_shapes
: *
dtype0*
value	B : 2<
:sparse_categorical_crossentropy/weighted_loss/num_elements€
?sparse_categorical_crossentropy/weighted_loss/num_elements/CastCastCsparse_categorical_crossentropy/weighted_loss/num_elements:output:0*

DstT0*

SrcT0*
_output_shapes
: 2A
?sparse_categorical_crossentropy/weighted_loss/num_elements/Cast™
2sparse_categorical_crossentropy/weighted_loss/RankConst*
_output_shapes
: *
dtype0*
value	B : 24
2sparse_categorical_crossentropy/weighted_loss/RankЄ
9sparse_categorical_crossentropy/weighted_loss/range/startConst*
_output_shapes
: *
dtype0*
value	B : 2;
9sparse_categorical_crossentropy/weighted_loss/range/startЄ
9sparse_categorical_crossentropy/weighted_loss/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2;
9sparse_categorical_crossentropy/weighted_loss/range/delta“
3sparse_categorical_crossentropy/weighted_loss/rangeRangeBsparse_categorical_crossentropy/weighted_loss/range/start:output:0;sparse_categorical_crossentropy/weighted_loss/Rank:output:0Bsparse_categorical_crossentropy/weighted_loss/range/delta:output:0*
_output_shapes
: 25
3sparse_categorical_crossentropy/weighted_loss/rangeМ
3sparse_categorical_crossentropy/weighted_loss/Sum_1Sum:sparse_categorical_crossentropy/weighted_loss/Sum:output:0<sparse_categorical_crossentropy/weighted_loss/range:output:0*
T0*
_output_shapes
: 25
3sparse_categorical_crossentropy/weighted_loss/Sum_1Ъ
3sparse_categorical_crossentropy/weighted_loss/valueDivNoNan<sparse_categorical_crossentropy/weighted_loss/Sum_1:output:0Csparse_categorical_crossentropy/weighted_loss/num_elements/Cast:y:0*
T0*
_output_shapes
: 25
3sparse_categorical_crossentropy/weighted_loss/value_
ShapeConst*
_output_shapes
:*
dtype0*
valueB"       2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2в
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
CastCaststrided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
Castu
MulMul7sparse_categorical_crossentropy/weighted_loss/value:z:0Cast:y:0*
T0*
_output_shapes
: 2
MulN
RankConst*
_output_shapes
: *
dtype0*
value	B : 2
Rank\
range/startConst*
_output_shapes
: *
dtype0*
value	B : 2
range/start\
range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
range/deltal
rangeRangerange/start:output:0Rank:output:0range/delta:output:0*
_output_shapes
: 2
rangeK
SumSumMul:z:0range:output:0*
T0*
_output_shapes
: 2
SumР
AssignAddVariableOpAssignAddVariableOpassignaddvariableop_resourceSum:output:0*
_output_shapes
 *
dtype02
AssignAddVariableOpR
Rank_1Const*
_output_shapes
: *
dtype0*
value	B : 2
Rank_1`
range_1/startConst*
_output_shapes
: *
dtype0*
value	B : 2
range_1/start`
range_1/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
range_1/deltav
range_1Rangerange_1/start:output:0Rank_1:output:0range_1/delta:output:0*
_output_shapes
: 2	
range_1R
Sum_1SumCast:y:0range_1:output:0*
T0*
_output_shapes
: 2
Sum_1Ѓ
AssignAddVariableOp_1AssignAddVariableOpassignaddvariableop_1_resourceSum_1:output:0^AssignAddVariableOp*
_output_shapes
 *
dtype02
AssignAddVariableOp_1Q
onesConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2
ones’
Ggradient_tape/sparse_categorical_crossentropy/weighted_loss/value/ShapeConst*
_output_shapes
: *
dtype0*
valueB 2I
Ggradient_tape/sparse_categorical_crossentropy/weighted_loss/value/Shapeў
Igradient_tape/sparse_categorical_crossentropy/weighted_loss/value/Shape_1Const*
_output_shapes
: *
dtype0*
valueB 2K
Igradient_tape/sparse_categorical_crossentropy/weighted_loss/value/Shape_1•
Wgradient_tape/sparse_categorical_crossentropy/weighted_loss/value/BroadcastGradientArgsBroadcastGradientArgsPgradient_tape/sparse_categorical_crossentropy/weighted_loss/value/Shape:output:0Rgradient_tape/sparse_categorical_crossentropy/weighted_loss/value/Shape_1:output:0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€2Y
Wgradient_tape/sparse_categorical_crossentropy/weighted_loss/value/BroadcastGradientArgsЭ
Lgradient_tape/sparse_categorical_crossentropy/weighted_loss/value/div_no_nanDivNoNanones:output:0Csparse_categorical_crossentropy/weighted_loss/num_elements/Cast:y:0*
T0*
_output_shapes
: 2N
Lgradient_tape/sparse_categorical_crossentropy/weighted_loss/value/div_no_nanж
Egradient_tape/sparse_categorical_crossentropy/weighted_loss/value/SumSumPgradient_tape/sparse_categorical_crossentropy/weighted_loss/value/div_no_nan:z:0\gradient_tape/sparse_categorical_crossentropy/weighted_loss/value/BroadcastGradientArgs:r0:0*
T0*
_output_shapes
: 2G
Egradient_tape/sparse_categorical_crossentropy/weighted_loss/value/Sumд
Igradient_tape/sparse_categorical_crossentropy/weighted_loss/value/ReshapeReshapeNgradient_tape/sparse_categorical_crossentropy/weighted_loss/value/Sum:output:0Pgradient_tape/sparse_categorical_crossentropy/weighted_loss/value/Shape:output:0*
T0*
_output_shapes
: 2K
Igradient_tape/sparse_categorical_crossentropy/weighted_loss/value/Reshapeф
Egradient_tape/sparse_categorical_crossentropy/weighted_loss/value/NegNeg<sparse_categorical_crossentropy/weighted_loss/Sum_1:output:0*
T0*
_output_shapes
: 2G
Egradient_tape/sparse_categorical_crossentropy/weighted_loss/value/NegЁ
Ngradient_tape/sparse_categorical_crossentropy/weighted_loss/value/div_no_nan_1DivNoNanIgradient_tape/sparse_categorical_crossentropy/weighted_loss/value/Neg:y:0Csparse_categorical_crossentropy/weighted_loss/num_elements/Cast:y:0*
T0*
_output_shapes
: 2P
Ngradient_tape/sparse_categorical_crossentropy/weighted_loss/value/div_no_nan_1ж
Ngradient_tape/sparse_categorical_crossentropy/weighted_loss/value/div_no_nan_2DivNoNanRgradient_tape/sparse_categorical_crossentropy/weighted_loss/value/div_no_nan_1:z:0Csparse_categorical_crossentropy/weighted_loss/num_elements/Cast:y:0*
T0*
_output_shapes
: 2P
Ngradient_tape/sparse_categorical_crossentropy/weighted_loss/value/div_no_nan_2Щ
Egradient_tape/sparse_categorical_crossentropy/weighted_loss/value/mulMulones:output:0Rgradient_tape/sparse_categorical_crossentropy/weighted_loss/value/div_no_nan_2:z:0*
T0*
_output_shapes
: 2G
Egradient_tape/sparse_categorical_crossentropy/weighted_loss/value/mulг
Ggradient_tape/sparse_categorical_crossentropy/weighted_loss/value/Sum_1SumIgradient_tape/sparse_categorical_crossentropy/weighted_loss/value/mul:z:0\gradient_tape/sparse_categorical_crossentropy/weighted_loss/value/BroadcastGradientArgs:r1:0*
T0*
_output_shapes
: 2I
Ggradient_tape/sparse_categorical_crossentropy/weighted_loss/value/Sum_1м
Kgradient_tape/sparse_categorical_crossentropy/weighted_loss/value/Reshape_1ReshapePgradient_tape/sparse_categorical_crossentropy/weighted_loss/value/Sum_1:output:0Rgradient_tape/sparse_categorical_crossentropy/weighted_loss/value/Shape_1:output:0*
T0*
_output_shapes
: 2M
Kgradient_tape/sparse_categorical_crossentropy/weighted_loss/value/Reshape_1ў
Igradient_tape/sparse_categorical_crossentropy/weighted_loss/Reshape/shapeConst*
_output_shapes
: *
dtype0*
valueB 2K
Igradient_tape/sparse_categorical_crossentropy/weighted_loss/Reshape/shapeЁ
Kgradient_tape/sparse_categorical_crossentropy/weighted_loss/Reshape/shape_1Const*
_output_shapes
: *
dtype0*
valueB 2M
Kgradient_tape/sparse_categorical_crossentropy/weighted_loss/Reshape/shape_1а
Cgradient_tape/sparse_categorical_crossentropy/weighted_loss/ReshapeReshapeRgradient_tape/sparse_categorical_crossentropy/weighted_loss/value/Reshape:output:0Tgradient_tape/sparse_categorical_crossentropy/weighted_loss/Reshape/shape_1:output:0*
T0*
_output_shapes
: 2E
Cgradient_tape/sparse_categorical_crossentropy/weighted_loss/Reshape…
Agradient_tape/sparse_categorical_crossentropy/weighted_loss/ConstConst*
_output_shapes
: *
dtype0*
valueB 2C
Agradient_tape/sparse_categorical_crossentropy/weighted_loss/Const«
@gradient_tape/sparse_categorical_crossentropy/weighted_loss/TileTileLgradient_tape/sparse_categorical_crossentropy/weighted_loss/Reshape:output:0Jgradient_tape/sparse_categorical_crossentropy/weighted_loss/Const:output:0*
T0*
_output_shapes
: 2B
@gradient_tape/sparse_categorical_crossentropy/weighted_loss/Tileд
Kgradient_tape/sparse_categorical_crossentropy/weighted_loss/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:2M
Kgradient_tape/sparse_categorical_crossentropy/weighted_loss/Reshape_1/shapeя
Egradient_tape/sparse_categorical_crossentropy/weighted_loss/Reshape_1ReshapeIgradient_tape/sparse_categorical_crossentropy/weighted_loss/Tile:output:0Tgradient_tape/sparse_categorical_crossentropy/weighted_loss/Reshape_1/shape:output:0*
T0*
_output_shapes
:2G
Egradient_tape/sparse_categorical_crossentropy/weighted_loss/Reshape_1‘
Cgradient_tape/sparse_categorical_crossentropy/weighted_loss/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2E
Cgradient_tape/sparse_categorical_crossentropy/weighted_loss/Const_1”
Bgradient_tape/sparse_categorical_crossentropy/weighted_loss/Tile_1TileNgradient_tape/sparse_categorical_crossentropy/weighted_loss/Reshape_1:output:0Lgradient_tape/sparse_categorical_crossentropy/weighted_loss/Const_1:output:0*
T0*
_output_shapes
: 2D
Bgradient_tape/sparse_categorical_crossentropy/weighted_loss/Tile_1є
?gradient_tape/sparse_categorical_crossentropy/weighted_loss/MulMulKgradient_tape/sparse_categorical_crossentropy/weighted_loss/Tile_1:output:0<sparse_categorical_crossentropy/weighted_loss/Const:output:0*
T0*
_output_shapes
: 2A
?gradient_tape/sparse_categorical_crossentropy/weighted_loss/MulП
`gradient_tape/sparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2b
`gradient_tape/sparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/ExpandDims/dim£
\gradient_tape/sparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/ExpandDims
ExpandDimsCgradient_tape/sparse_categorical_crossentropy/weighted_loss/Mul:z:0igradient_tape/sparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/ExpandDims/dim:output:0*
T0*
_output_shapes

: 2^
\gradient_tape/sparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/ExpandDimsє
Ugradient_tape/sparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/mulMulegradient_tape/sparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/ExpandDims:output:0rsparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits:backprop:0*
T0*
_output_shapes

: 
2W
Ugradient_tape/sparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/mulї
3gradient_tape/sparse_categorical_crossentropy/ShapeConst*
_output_shapes
:*
dtype0*
valueB"    
   25
3gradient_tape/sparse_categorical_crossentropy/Shapeї
5gradient_tape/sparse_categorical_crossentropy/ReshapeReshapeYgradient_tape/sparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/mul:z:0<gradient_tape/sparse_categorical_crossentropy/Shape:output:0*
T0*
_output_shapes

: 
27
5gradient_tape/sparse_categorical_crossentropy/Reshapeд
6gradient_tape/custom_model/dense_2/BiasAdd/BiasAddGradBiasAddGrad>gradient_tape/sparse_categorical_crossentropy/Reshape:output:0*
T0*
_output_shapes
:
28
6gradient_tape/custom_model/dense_2/BiasAdd/BiasAddGradР
)gradient_tape/custom_model/dense_2/MatMulMatMul>gradient_tape/sparse_categorical_crossentropy/Reshape:output:02custom_model/dense_2/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

: 2*
transpose_b(2+
)gradient_tape/custom_model/dense_2/MatMulЙ
+gradient_tape/custom_model/dense_2/MatMul_1MatMul'custom_model/dense_1/Relu:activations:0>gradient_tape/sparse_categorical_crossentropy/Reshape:output:0*
T0*
_output_shapes

:2
*
transpose_a(2-
+gradient_tape/custom_model/dense_2/MatMul_1н
+gradient_tape/custom_model/dense_1/ReluGradReluGrad3gradient_tape/custom_model/dense_2/MatMul:product:0'custom_model/dense_1/Relu:activations:0*
T0*
_output_shapes

: 22-
+gradient_tape/custom_model/dense_1/ReluGradЁ
6gradient_tape/custom_model/dense_1/BiasAdd/BiasAddGradBiasAddGrad7gradient_tape/custom_model/dense_1/ReluGrad:backprops:0*
T0*
_output_shapes
:228
6gradient_tape/custom_model/dense_1/BiasAdd/BiasAddGradЙ
)gradient_tape/custom_model/dense_1/MatMulMatMul7gradient_tape/custom_model/dense_1/ReluGrad:backprops:02custom_model/dense_1/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

: 2*
transpose_b(2+
)gradient_tape/custom_model/dense_1/MatMulА
+gradient_tape/custom_model/dense_1/MatMul_1MatMul%custom_model/dense/Relu:activations:07gradient_tape/custom_model/dense_1/ReluGrad:backprops:0*
T0*
_output_shapes

:22*
transpose_a(2-
+gradient_tape/custom_model/dense_1/MatMul_1з
)gradient_tape/custom_model/dense/ReluGradReluGrad3gradient_tape/custom_model/dense_1/MatMul:product:0%custom_model/dense/Relu:activations:0*
T0*
_output_shapes

: 22+
)gradient_tape/custom_model/dense/ReluGrad„
4gradient_tape/custom_model/dense/BiasAdd/BiasAddGradBiasAddGrad5gradient_tape/custom_model/dense/ReluGrad:backprops:0*
T0*
_output_shapes
:226
4gradient_tape/custom_model/dense/BiasAdd/BiasAddGradч
'gradient_tape/custom_model/dense/MatMulMatMul%custom_model/flatten/Reshape:output:05gradient_tape/custom_model/dense/ReluGrad:backprops:0*
T0*
_output_shapes
:	Р2*
transpose_a(2)
'gradient_tape/custom_model/dense/MatMulЧ
RMSprop/Cast/ReadVariableOpReadVariableOp$rmsprop_cast_readvariableop_resource*
_output_shapes
: *
dtype02
RMSprop/Cast/ReadVariableOp§
RMSprop/IdentityIdentity#RMSprop/Cast/ReadVariableOp:value:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*
_output_shapes
: 2
RMSprop/IdentityЭ
RMSprop/Cast_1/ReadVariableOpReadVariableOp&rmsprop_cast_1_readvariableop_resource*
_output_shapes
: *
dtype02
RMSprop/Cast_1/ReadVariableOp™
RMSprop/Identity_1Identity%RMSprop/Cast_1/ReadVariableOp:value:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*
_output_shapes
: 2
RMSprop/Identity_1Л
RMSprop/NegNegRMSprop/Identity:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*
_output_shapes
: 2
RMSprop/NegС
RMSprop/ConstConst",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *Хњ÷32
RMSprop/ConstЭ
RMSprop/Cast_2/ReadVariableOpReadVariableOp&rmsprop_cast_2_readvariableop_resource*
_output_shapes
: *
dtype02
RMSprop/Cast_2/ReadVariableOp™
RMSprop/Identity_2Identity%RMSprop/Cast_2/ReadVariableOp:value:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*
_output_shapes
: 2
RMSprop/Identity_2С
RMSprop/sub/xConst",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *  А?2
RMSprop/sub/x•
RMSprop/subSubRMSprop/sub/x:output:0RMSprop/Identity_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*
_output_shapes
: 2
RMSprop/sub 
)RMSprop/RMSprop/update/mul/ReadVariableOpReadVariableOp2rmsprop_rmsprop_update_mul_readvariableop_resource*
_output_shapes
:	Р2*
dtype02+
)RMSprop/RMSprop/update/mul/ReadVariableOp≠
RMSprop/RMSprop/update/mulMulRMSprop/Identity_1:output:01RMSprop/RMSprop/update/mul/ReadVariableOp:value:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*D
_class:
86loc:@custom_model/dense/MatMul/ReadVariableOp/resource*
_output_shapes
:	Р22
RMSprop/RMSprop/update/mulЩ
RMSprop/RMSprop/update/SquareSquare1gradient_tape/custom_model/dense/MatMul:product:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*D
_class:
86loc:@custom_model/dense/MatMul/ReadVariableOp/resource*
_output_shapes
:	Р22
RMSprop/RMSprop/update/SquareХ
RMSprop/RMSprop/update/mul_1MulRMSprop/sub:z:0!RMSprop/RMSprop/update/Square:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*D
_class:
86loc:@custom_model/dense/MatMul/ReadVariableOp/resource*
_output_shapes
:	Р22
RMSprop/RMSprop/update/mul_1°
RMSprop/RMSprop/update/addAddV2RMSprop/RMSprop/update/mul:z:0 RMSprop/RMSprop/update/mul_1:z:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*D
_class:
86loc:@custom_model/dense/MatMul/ReadVariableOp/resource*
_output_shapes
:	Р22
RMSprop/RMSprop/update/addэ
'RMSprop/RMSprop/update/AssignVariableOpAssignVariableOp2rmsprop_rmsprop_update_mul_readvariableop_resourceRMSprop/RMSprop/update/add:z:0*^RMSprop/RMSprop/update/mul/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*D
_class:
86loc:@custom_model/dense/MatMul/ReadVariableOp/resource*
_output_shapes
 *
dtype02)
'RMSprop/RMSprop/update/AssignVariableOpѓ
RMSprop/RMSprop/update/mul_2MulRMSprop/Identity:output:01gradient_tape/custom_model/dense/MatMul:product:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*D
_class:
86loc:@custom_model/dense/MatMul/ReadVariableOp/resource*
_output_shapes
:	Р22
RMSprop/RMSprop/update/mul_2к
*RMSprop/RMSprop/update/Sqrt/ReadVariableOpReadVariableOp2rmsprop_rmsprop_update_mul_readvariableop_resource(^RMSprop/RMSprop/update/AssignVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*D
_class:
86loc:@custom_model/dense/MatMul/ReadVariableOp/resource*
_output_shapes
:	Р2*
dtype02,
*RMSprop/RMSprop/update/Sqrt/ReadVariableOpФ
RMSprop/RMSprop/update/SqrtSqrt2RMSprop/RMSprop/update/Sqrt/ReadVariableOp:value:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*D
_class:
86loc:@custom_model/dense/MatMul/ReadVariableOp/resource*
_output_shapes
:	Р22
RMSprop/RMSprop/update/SqrtЬ
RMSprop/RMSprop/update/add_1AddV2RMSprop/RMSprop/update/Sqrt:y:0RMSprop/Const:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*D
_class:
86loc:@custom_model/dense/MatMul/ReadVariableOp/resource*
_output_shapes
:	Р22
RMSprop/RMSprop/update/add_1≠
RMSprop/RMSprop/update/truedivRealDiv RMSprop/RMSprop/update/mul_2:z:0 RMSprop/RMSprop/update/add_1:z:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*D
_class:
86loc:@custom_model/dense/MatMul/ReadVariableOp/resource*
_output_shapes
:	Р22 
RMSprop/RMSprop/update/truedivЅ
%RMSprop/RMSprop/update/ReadVariableOpReadVariableOp1custom_model_dense_matmul_readvariableop_resource*
_output_shapes
:	Р2*
dtype02'
%RMSprop/RMSprop/update/ReadVariableOp∞
RMSprop/RMSprop/update/subSub-RMSprop/RMSprop/update/ReadVariableOp:value:0"RMSprop/RMSprop/update/truediv:z:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*D
_class:
86loc:@custom_model/dense/MatMul/ReadVariableOp/resource*
_output_shapes
:	Р22
RMSprop/RMSprop/update/subІ
)RMSprop/RMSprop/update/AssignVariableOp_1AssignVariableOp1custom_model_dense_matmul_readvariableop_resourceRMSprop/RMSprop/update/sub:z:0&^RMSprop/RMSprop/update/ReadVariableOp)^custom_model/dense/MatMul/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*D
_class:
86loc:@custom_model/dense/MatMul/ReadVariableOp/resource*
_output_shapes
 *
dtype02+
)RMSprop/RMSprop/update/AssignVariableOp_1Ћ
+RMSprop/RMSprop/update_1/mul/ReadVariableOpReadVariableOp4rmsprop_rmsprop_update_1_mul_readvariableop_resource*
_output_shapes
:2*
dtype02-
+RMSprop/RMSprop/update_1/mul/ReadVariableOpѓ
RMSprop/RMSprop/update_1/mulMulRMSprop/Identity_1:output:03RMSprop/RMSprop/update_1/mul/ReadVariableOp:value:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*E
_class;
97loc:@custom_model/dense/BiasAdd/ReadVariableOp/resource*
_output_shapes
:22
RMSprop/RMSprop/update_1/mul•
RMSprop/RMSprop/update_1/SquareSquare=gradient_tape/custom_model/dense/BiasAdd/BiasAddGrad:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*E
_class;
97loc:@custom_model/dense/BiasAdd/ReadVariableOp/resource*
_output_shapes
:22!
RMSprop/RMSprop/update_1/SquareЧ
RMSprop/RMSprop/update_1/mul_1MulRMSprop/sub:z:0#RMSprop/RMSprop/update_1/Square:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*E
_class;
97loc:@custom_model/dense/BiasAdd/ReadVariableOp/resource*
_output_shapes
:22 
RMSprop/RMSprop/update_1/mul_1•
RMSprop/RMSprop/update_1/addAddV2 RMSprop/RMSprop/update_1/mul:z:0"RMSprop/RMSprop/update_1/mul_1:z:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*E
_class;
97loc:@custom_model/dense/BiasAdd/ReadVariableOp/resource*
_output_shapes
:22
RMSprop/RMSprop/update_1/addИ
)RMSprop/RMSprop/update_1/AssignVariableOpAssignVariableOp4rmsprop_rmsprop_update_1_mul_readvariableop_resource RMSprop/RMSprop/update_1/add:z:0,^RMSprop/RMSprop/update_1/mul/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*E
_class;
97loc:@custom_model/dense/BiasAdd/ReadVariableOp/resource*
_output_shapes
 *
dtype02+
)RMSprop/RMSprop/update_1/AssignVariableOpї
RMSprop/RMSprop/update_1/mul_2MulRMSprop/Identity:output:0=gradient_tape/custom_model/dense/BiasAdd/BiasAddGrad:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*E
_class;
97loc:@custom_model/dense/BiasAdd/ReadVariableOp/resource*
_output_shapes
:22 
RMSprop/RMSprop/update_1/mul_2о
,RMSprop/RMSprop/update_1/Sqrt/ReadVariableOpReadVariableOp4rmsprop_rmsprop_update_1_mul_readvariableop_resource*^RMSprop/RMSprop/update_1/AssignVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*E
_class;
97loc:@custom_model/dense/BiasAdd/ReadVariableOp/resource*
_output_shapes
:2*
dtype02.
,RMSprop/RMSprop/update_1/Sqrt/ReadVariableOpЦ
RMSprop/RMSprop/update_1/SqrtSqrt4RMSprop/RMSprop/update_1/Sqrt/ReadVariableOp:value:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*E
_class;
97loc:@custom_model/dense/BiasAdd/ReadVariableOp/resource*
_output_shapes
:22
RMSprop/RMSprop/update_1/SqrtЮ
RMSprop/RMSprop/update_1/add_1AddV2!RMSprop/RMSprop/update_1/Sqrt:y:0RMSprop/Const:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*E
_class;
97loc:@custom_model/dense/BiasAdd/ReadVariableOp/resource*
_output_shapes
:22 
RMSprop/RMSprop/update_1/add_1±
 RMSprop/RMSprop/update_1/truedivRealDiv"RMSprop/RMSprop/update_1/mul_2:z:0"RMSprop/RMSprop/update_1/add_1:z:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*E
_class;
97loc:@custom_model/dense/BiasAdd/ReadVariableOp/resource*
_output_shapes
:22"
 RMSprop/RMSprop/update_1/truedivЅ
'RMSprop/RMSprop/update_1/ReadVariableOpReadVariableOp2custom_model_dense_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02)
'RMSprop/RMSprop/update_1/ReadVariableOpі
RMSprop/RMSprop/update_1/subSub/RMSprop/RMSprop/update_1/ReadVariableOp:value:0$RMSprop/RMSprop/update_1/truediv:z:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*E
_class;
97loc:@custom_model/dense/BiasAdd/ReadVariableOp/resource*
_output_shapes
:22
RMSprop/RMSprop/update_1/sub≤
+RMSprop/RMSprop/update_1/AssignVariableOp_1AssignVariableOp2custom_model_dense_biasadd_readvariableop_resource RMSprop/RMSprop/update_1/sub:z:0(^RMSprop/RMSprop/update_1/ReadVariableOp*^custom_model/dense/BiasAdd/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*E
_class;
97loc:@custom_model/dense/BiasAdd/ReadVariableOp/resource*
_output_shapes
 *
dtype02-
+RMSprop/RMSprop/update_1/AssignVariableOp_1ѕ
+RMSprop/RMSprop/update_2/mul/ReadVariableOpReadVariableOp4rmsprop_rmsprop_update_2_mul_readvariableop_resource*
_output_shapes

:22*
dtype02-
+RMSprop/RMSprop/update_2/mul/ReadVariableOpі
RMSprop/RMSprop/update_2/mulMulRMSprop/Identity_1:output:03RMSprop/RMSprop/update_2/mul/ReadVariableOp:value:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*F
_class<
:8loc:@custom_model/dense_1/MatMul/ReadVariableOp/resource*
_output_shapes

:222
RMSprop/RMSprop/update_2/mulҐ
RMSprop/RMSprop/update_2/SquareSquare5gradient_tape/custom_model/dense_1/MatMul_1:product:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*F
_class<
:8loc:@custom_model/dense_1/MatMul/ReadVariableOp/resource*
_output_shapes

:222!
RMSprop/RMSprop/update_2/SquareЬ
RMSprop/RMSprop/update_2/mul_1MulRMSprop/sub:z:0#RMSprop/RMSprop/update_2/Square:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*F
_class<
:8loc:@custom_model/dense_1/MatMul/ReadVariableOp/resource*
_output_shapes

:222 
RMSprop/RMSprop/update_2/mul_1™
RMSprop/RMSprop/update_2/addAddV2 RMSprop/RMSprop/update_2/mul:z:0"RMSprop/RMSprop/update_2/mul_1:z:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*F
_class<
:8loc:@custom_model/dense_1/MatMul/ReadVariableOp/resource*
_output_shapes

:222
RMSprop/RMSprop/update_2/addЙ
)RMSprop/RMSprop/update_2/AssignVariableOpAssignVariableOp4rmsprop_rmsprop_update_2_mul_readvariableop_resource RMSprop/RMSprop/update_2/add:z:0,^RMSprop/RMSprop/update_2/mul/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*F
_class<
:8loc:@custom_model/dense_1/MatMul/ReadVariableOp/resource*
_output_shapes
 *
dtype02+
)RMSprop/RMSprop/update_2/AssignVariableOpЄ
RMSprop/RMSprop/update_2/mul_2MulRMSprop/Identity:output:05gradient_tape/custom_model/dense_1/MatMul_1:product:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*F
_class<
:8loc:@custom_model/dense_1/MatMul/ReadVariableOp/resource*
_output_shapes

:222 
RMSprop/RMSprop/update_2/mul_2у
,RMSprop/RMSprop/update_2/Sqrt/ReadVariableOpReadVariableOp4rmsprop_rmsprop_update_2_mul_readvariableop_resource*^RMSprop/RMSprop/update_2/AssignVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*F
_class<
:8loc:@custom_model/dense_1/MatMul/ReadVariableOp/resource*
_output_shapes

:22*
dtype02.
,RMSprop/RMSprop/update_2/Sqrt/ReadVariableOpЫ
RMSprop/RMSprop/update_2/SqrtSqrt4RMSprop/RMSprop/update_2/Sqrt/ReadVariableOp:value:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*F
_class<
:8loc:@custom_model/dense_1/MatMul/ReadVariableOp/resource*
_output_shapes

:222
RMSprop/RMSprop/update_2/Sqrt£
RMSprop/RMSprop/update_2/add_1AddV2!RMSprop/RMSprop/update_2/Sqrt:y:0RMSprop/Const:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*F
_class<
:8loc:@custom_model/dense_1/MatMul/ReadVariableOp/resource*
_output_shapes

:222 
RMSprop/RMSprop/update_2/add_1ґ
 RMSprop/RMSprop/update_2/truedivRealDiv"RMSprop/RMSprop/update_2/mul_2:z:0"RMSprop/RMSprop/update_2/add_1:z:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*F
_class<
:8loc:@custom_model/dense_1/MatMul/ReadVariableOp/resource*
_output_shapes

:222"
 RMSprop/RMSprop/update_2/truediv∆
'RMSprop/RMSprop/update_2/ReadVariableOpReadVariableOp3custom_model_dense_1_matmul_readvariableop_resource*
_output_shapes

:22*
dtype02)
'RMSprop/RMSprop/update_2/ReadVariableOpє
RMSprop/RMSprop/update_2/subSub/RMSprop/RMSprop/update_2/ReadVariableOp:value:0$RMSprop/RMSprop/update_2/truediv:z:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*F
_class<
:8loc:@custom_model/dense_1/MatMul/ReadVariableOp/resource*
_output_shapes

:222
RMSprop/RMSprop/update_2/subµ
+RMSprop/RMSprop/update_2/AssignVariableOp_1AssignVariableOp3custom_model_dense_1_matmul_readvariableop_resource RMSprop/RMSprop/update_2/sub:z:0(^RMSprop/RMSprop/update_2/ReadVariableOp+^custom_model/dense_1/MatMul/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*F
_class<
:8loc:@custom_model/dense_1/MatMul/ReadVariableOp/resource*
_output_shapes
 *
dtype02-
+RMSprop/RMSprop/update_2/AssignVariableOp_1Ћ
+RMSprop/RMSprop/update_3/mul/ReadVariableOpReadVariableOp4rmsprop_rmsprop_update_3_mul_readvariableop_resource*
_output_shapes
:2*
dtype02-
+RMSprop/RMSprop/update_3/mul/ReadVariableOp±
RMSprop/RMSprop/update_3/mulMulRMSprop/Identity_1:output:03RMSprop/RMSprop/update_3/mul/ReadVariableOp:value:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*G
_class=
;9loc:@custom_model/dense_1/BiasAdd/ReadVariableOp/resource*
_output_shapes
:22
RMSprop/RMSprop/update_3/mul©
RMSprop/RMSprop/update_3/SquareSquare?gradient_tape/custom_model/dense_1/BiasAdd/BiasAddGrad:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*G
_class=
;9loc:@custom_model/dense_1/BiasAdd/ReadVariableOp/resource*
_output_shapes
:22!
RMSprop/RMSprop/update_3/SquareЩ
RMSprop/RMSprop/update_3/mul_1MulRMSprop/sub:z:0#RMSprop/RMSprop/update_3/Square:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*G
_class=
;9loc:@custom_model/dense_1/BiasAdd/ReadVariableOp/resource*
_output_shapes
:22 
RMSprop/RMSprop/update_3/mul_1І
RMSprop/RMSprop/update_3/addAddV2 RMSprop/RMSprop/update_3/mul:z:0"RMSprop/RMSprop/update_3/mul_1:z:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*G
_class=
;9loc:@custom_model/dense_1/BiasAdd/ReadVariableOp/resource*
_output_shapes
:22
RMSprop/RMSprop/update_3/addК
)RMSprop/RMSprop/update_3/AssignVariableOpAssignVariableOp4rmsprop_rmsprop_update_3_mul_readvariableop_resource RMSprop/RMSprop/update_3/add:z:0,^RMSprop/RMSprop/update_3/mul/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*G
_class=
;9loc:@custom_model/dense_1/BiasAdd/ReadVariableOp/resource*
_output_shapes
 *
dtype02+
)RMSprop/RMSprop/update_3/AssignVariableOpњ
RMSprop/RMSprop/update_3/mul_2MulRMSprop/Identity:output:0?gradient_tape/custom_model/dense_1/BiasAdd/BiasAddGrad:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*G
_class=
;9loc:@custom_model/dense_1/BiasAdd/ReadVariableOp/resource*
_output_shapes
:22 
RMSprop/RMSprop/update_3/mul_2р
,RMSprop/RMSprop/update_3/Sqrt/ReadVariableOpReadVariableOp4rmsprop_rmsprop_update_3_mul_readvariableop_resource*^RMSprop/RMSprop/update_3/AssignVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*G
_class=
;9loc:@custom_model/dense_1/BiasAdd/ReadVariableOp/resource*
_output_shapes
:2*
dtype02.
,RMSprop/RMSprop/update_3/Sqrt/ReadVariableOpШ
RMSprop/RMSprop/update_3/SqrtSqrt4RMSprop/RMSprop/update_3/Sqrt/ReadVariableOp:value:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*G
_class=
;9loc:@custom_model/dense_1/BiasAdd/ReadVariableOp/resource*
_output_shapes
:22
RMSprop/RMSprop/update_3/Sqrt†
RMSprop/RMSprop/update_3/add_1AddV2!RMSprop/RMSprop/update_3/Sqrt:y:0RMSprop/Const:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*G
_class=
;9loc:@custom_model/dense_1/BiasAdd/ReadVariableOp/resource*
_output_shapes
:22 
RMSprop/RMSprop/update_3/add_1≥
 RMSprop/RMSprop/update_3/truedivRealDiv"RMSprop/RMSprop/update_3/mul_2:z:0"RMSprop/RMSprop/update_3/add_1:z:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*G
_class=
;9loc:@custom_model/dense_1/BiasAdd/ReadVariableOp/resource*
_output_shapes
:22"
 RMSprop/RMSprop/update_3/truediv√
'RMSprop/RMSprop/update_3/ReadVariableOpReadVariableOp4custom_model_dense_1_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02)
'RMSprop/RMSprop/update_3/ReadVariableOpґ
RMSprop/RMSprop/update_3/subSub/RMSprop/RMSprop/update_3/ReadVariableOp:value:0$RMSprop/RMSprop/update_3/truediv:z:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*G
_class=
;9loc:@custom_model/dense_1/BiasAdd/ReadVariableOp/resource*
_output_shapes
:22
RMSprop/RMSprop/update_3/subЄ
+RMSprop/RMSprop/update_3/AssignVariableOp_1AssignVariableOp4custom_model_dense_1_biasadd_readvariableop_resource RMSprop/RMSprop/update_3/sub:z:0(^RMSprop/RMSprop/update_3/ReadVariableOp,^custom_model/dense_1/BiasAdd/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*G
_class=
;9loc:@custom_model/dense_1/BiasAdd/ReadVariableOp/resource*
_output_shapes
 *
dtype02-
+RMSprop/RMSprop/update_3/AssignVariableOp_1ѕ
+RMSprop/RMSprop/update_4/mul/ReadVariableOpReadVariableOp4rmsprop_rmsprop_update_4_mul_readvariableop_resource*
_output_shapes

:2
*
dtype02-
+RMSprop/RMSprop/update_4/mul/ReadVariableOpі
RMSprop/RMSprop/update_4/mulMulRMSprop/Identity_1:output:03RMSprop/RMSprop/update_4/mul/ReadVariableOp:value:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*F
_class<
:8loc:@custom_model/dense_2/MatMul/ReadVariableOp/resource*
_output_shapes

:2
2
RMSprop/RMSprop/update_4/mulҐ
RMSprop/RMSprop/update_4/SquareSquare5gradient_tape/custom_model/dense_2/MatMul_1:product:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*F
_class<
:8loc:@custom_model/dense_2/MatMul/ReadVariableOp/resource*
_output_shapes

:2
2!
RMSprop/RMSprop/update_4/SquareЬ
RMSprop/RMSprop/update_4/mul_1MulRMSprop/sub:z:0#RMSprop/RMSprop/update_4/Square:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*F
_class<
:8loc:@custom_model/dense_2/MatMul/ReadVariableOp/resource*
_output_shapes

:2
2 
RMSprop/RMSprop/update_4/mul_1™
RMSprop/RMSprop/update_4/addAddV2 RMSprop/RMSprop/update_4/mul:z:0"RMSprop/RMSprop/update_4/mul_1:z:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*F
_class<
:8loc:@custom_model/dense_2/MatMul/ReadVariableOp/resource*
_output_shapes

:2
2
RMSprop/RMSprop/update_4/addЙ
)RMSprop/RMSprop/update_4/AssignVariableOpAssignVariableOp4rmsprop_rmsprop_update_4_mul_readvariableop_resource RMSprop/RMSprop/update_4/add:z:0,^RMSprop/RMSprop/update_4/mul/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*F
_class<
:8loc:@custom_model/dense_2/MatMul/ReadVariableOp/resource*
_output_shapes
 *
dtype02+
)RMSprop/RMSprop/update_4/AssignVariableOpЄ
RMSprop/RMSprop/update_4/mul_2MulRMSprop/Identity:output:05gradient_tape/custom_model/dense_2/MatMul_1:product:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*F
_class<
:8loc:@custom_model/dense_2/MatMul/ReadVariableOp/resource*
_output_shapes

:2
2 
RMSprop/RMSprop/update_4/mul_2у
,RMSprop/RMSprop/update_4/Sqrt/ReadVariableOpReadVariableOp4rmsprop_rmsprop_update_4_mul_readvariableop_resource*^RMSprop/RMSprop/update_4/AssignVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*F
_class<
:8loc:@custom_model/dense_2/MatMul/ReadVariableOp/resource*
_output_shapes

:2
*
dtype02.
,RMSprop/RMSprop/update_4/Sqrt/ReadVariableOpЫ
RMSprop/RMSprop/update_4/SqrtSqrt4RMSprop/RMSprop/update_4/Sqrt/ReadVariableOp:value:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*F
_class<
:8loc:@custom_model/dense_2/MatMul/ReadVariableOp/resource*
_output_shapes

:2
2
RMSprop/RMSprop/update_4/Sqrt£
RMSprop/RMSprop/update_4/add_1AddV2!RMSprop/RMSprop/update_4/Sqrt:y:0RMSprop/Const:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*F
_class<
:8loc:@custom_model/dense_2/MatMul/ReadVariableOp/resource*
_output_shapes

:2
2 
RMSprop/RMSprop/update_4/add_1ґ
 RMSprop/RMSprop/update_4/truedivRealDiv"RMSprop/RMSprop/update_4/mul_2:z:0"RMSprop/RMSprop/update_4/add_1:z:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*F
_class<
:8loc:@custom_model/dense_2/MatMul/ReadVariableOp/resource*
_output_shapes

:2
2"
 RMSprop/RMSprop/update_4/truediv∆
'RMSprop/RMSprop/update_4/ReadVariableOpReadVariableOp3custom_model_dense_2_matmul_readvariableop_resource*
_output_shapes

:2
*
dtype02)
'RMSprop/RMSprop/update_4/ReadVariableOpє
RMSprop/RMSprop/update_4/subSub/RMSprop/RMSprop/update_4/ReadVariableOp:value:0$RMSprop/RMSprop/update_4/truediv:z:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*F
_class<
:8loc:@custom_model/dense_2/MatMul/ReadVariableOp/resource*
_output_shapes

:2
2
RMSprop/RMSprop/update_4/subµ
+RMSprop/RMSprop/update_4/AssignVariableOp_1AssignVariableOp3custom_model_dense_2_matmul_readvariableop_resource RMSprop/RMSprop/update_4/sub:z:0(^RMSprop/RMSprop/update_4/ReadVariableOp+^custom_model/dense_2/MatMul/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*F
_class<
:8loc:@custom_model/dense_2/MatMul/ReadVariableOp/resource*
_output_shapes
 *
dtype02-
+RMSprop/RMSprop/update_4/AssignVariableOp_1Ћ
+RMSprop/RMSprop/update_5/mul/ReadVariableOpReadVariableOp4rmsprop_rmsprop_update_5_mul_readvariableop_resource*
_output_shapes
:
*
dtype02-
+RMSprop/RMSprop/update_5/mul/ReadVariableOp±
RMSprop/RMSprop/update_5/mulMulRMSprop/Identity_1:output:03RMSprop/RMSprop/update_5/mul/ReadVariableOp:value:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*G
_class=
;9loc:@custom_model/dense_2/BiasAdd/ReadVariableOp/resource*
_output_shapes
:
2
RMSprop/RMSprop/update_5/mul©
RMSprop/RMSprop/update_5/SquareSquare?gradient_tape/custom_model/dense_2/BiasAdd/BiasAddGrad:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*G
_class=
;9loc:@custom_model/dense_2/BiasAdd/ReadVariableOp/resource*
_output_shapes
:
2!
RMSprop/RMSprop/update_5/SquareЩ
RMSprop/RMSprop/update_5/mul_1MulRMSprop/sub:z:0#RMSprop/RMSprop/update_5/Square:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*G
_class=
;9loc:@custom_model/dense_2/BiasAdd/ReadVariableOp/resource*
_output_shapes
:
2 
RMSprop/RMSprop/update_5/mul_1І
RMSprop/RMSprop/update_5/addAddV2 RMSprop/RMSprop/update_5/mul:z:0"RMSprop/RMSprop/update_5/mul_1:z:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*G
_class=
;9loc:@custom_model/dense_2/BiasAdd/ReadVariableOp/resource*
_output_shapes
:
2
RMSprop/RMSprop/update_5/addК
)RMSprop/RMSprop/update_5/AssignVariableOpAssignVariableOp4rmsprop_rmsprop_update_5_mul_readvariableop_resource RMSprop/RMSprop/update_5/add:z:0,^RMSprop/RMSprop/update_5/mul/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*G
_class=
;9loc:@custom_model/dense_2/BiasAdd/ReadVariableOp/resource*
_output_shapes
 *
dtype02+
)RMSprop/RMSprop/update_5/AssignVariableOpњ
RMSprop/RMSprop/update_5/mul_2MulRMSprop/Identity:output:0?gradient_tape/custom_model/dense_2/BiasAdd/BiasAddGrad:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*G
_class=
;9loc:@custom_model/dense_2/BiasAdd/ReadVariableOp/resource*
_output_shapes
:
2 
RMSprop/RMSprop/update_5/mul_2р
,RMSprop/RMSprop/update_5/Sqrt/ReadVariableOpReadVariableOp4rmsprop_rmsprop_update_5_mul_readvariableop_resource*^RMSprop/RMSprop/update_5/AssignVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*G
_class=
;9loc:@custom_model/dense_2/BiasAdd/ReadVariableOp/resource*
_output_shapes
:
*
dtype02.
,RMSprop/RMSprop/update_5/Sqrt/ReadVariableOpШ
RMSprop/RMSprop/update_5/SqrtSqrt4RMSprop/RMSprop/update_5/Sqrt/ReadVariableOp:value:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*G
_class=
;9loc:@custom_model/dense_2/BiasAdd/ReadVariableOp/resource*
_output_shapes
:
2
RMSprop/RMSprop/update_5/Sqrt†
RMSprop/RMSprop/update_5/add_1AddV2!RMSprop/RMSprop/update_5/Sqrt:y:0RMSprop/Const:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*G
_class=
;9loc:@custom_model/dense_2/BiasAdd/ReadVariableOp/resource*
_output_shapes
:
2 
RMSprop/RMSprop/update_5/add_1≥
 RMSprop/RMSprop/update_5/truedivRealDiv"RMSprop/RMSprop/update_5/mul_2:z:0"RMSprop/RMSprop/update_5/add_1:z:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*G
_class=
;9loc:@custom_model/dense_2/BiasAdd/ReadVariableOp/resource*
_output_shapes
:
2"
 RMSprop/RMSprop/update_5/truediv√
'RMSprop/RMSprop/update_5/ReadVariableOpReadVariableOp4custom_model_dense_2_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02)
'RMSprop/RMSprop/update_5/ReadVariableOpґ
RMSprop/RMSprop/update_5/subSub/RMSprop/RMSprop/update_5/ReadVariableOp:value:0$RMSprop/RMSprop/update_5/truediv:z:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*G
_class=
;9loc:@custom_model/dense_2/BiasAdd/ReadVariableOp/resource*
_output_shapes
:
2
RMSprop/RMSprop/update_5/subЄ
+RMSprop/RMSprop/update_5/AssignVariableOp_1AssignVariableOp4custom_model_dense_2_biasadd_readvariableop_resource RMSprop/RMSprop/update_5/sub:z:0(^RMSprop/RMSprop/update_5/ReadVariableOp,^custom_model/dense_2/BiasAdd/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*G
_class=
;9loc:@custom_model/dense_2/BiasAdd/ReadVariableOp/resource*
_output_shapes
 *
dtype02-
+RMSprop/RMSprop/update_5/AssignVariableOp_1Ц
RMSprop/RMSprop/group_depsNoOp*^RMSprop/RMSprop/update/AssignVariableOp_1,^RMSprop/RMSprop/update_1/AssignVariableOp_1,^RMSprop/RMSprop/update_2/AssignVariableOp_1,^RMSprop/RMSprop/update_3/AssignVariableOp_1,^RMSprop/RMSprop/update_4/AssignVariableOp_1,^RMSprop/RMSprop/update_5/AssignVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
 2
RMSprop/RMSprop/group_depsН
RMSprop/RMSprop/ConstConst^RMSprop/RMSprop/group_deps*
_output_shapes
: *
dtype0	*
value	B	 R2
RMSprop/RMSprop/Const“
#RMSprop/RMSprop/AssignAddVariableOpAssignAddVariableOp,rmsprop_rmsprop_assignaddvariableop_resourceRMSprop/RMSprop/Const:output:0*
_output_shapes
 *
dtype0	2%
#RMSprop/RMSprop/AssignAddVariableOpo
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2
ExpandDims_1/dimv
ExpandDims_1
ExpandDimsdata_1ExpandDims_1/dim:output:0*
T0*
_output_shapes

: 2
ExpandDims_1y
SqueezeSqueezeExpandDims_1:output:0*
T0*
_output_shapes
: *
squeeze_dims

€€€€€€€€€2	
Squeezeo
ArgMax/dimensionConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2
ArgMax/dimensionВ
ArgMaxArgMax&custom_model/dense_2/Softmax:softmax:0ArgMax/dimension:output:0*
T0*
_output_shapes
: 2
ArgMax]
Cast_1CastArgMax:output:0*

DstT0*

SrcT0	*
_output_shapes
: 2
Cast_1Z
EqualEqualSqueeze:output:0
Cast_1:y:0*
T0*
_output_shapes
: 2
EqualW
Cast_2Cast	Equal:z:0*

DstT0*

SrcT0
*
_output_shapes
: 2
Cast_2X
ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
ConstR
Sum_2Sum
Cast_2:y:0Const:output:0*
T0*
_output_shapes
: 2
Sum_2Ш
AssignAddVariableOp_2AssignAddVariableOpassignaddvariableop_2_resourceSum_2:output:0*
_output_shapes
 *
dtype02
AssignAddVariableOp_2N
SizeConst*
_output_shapes
: *
dtype0*
value	B : 2
SizeW
Cast_3CastSize:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
Cast_3ђ
AssignAddVariableOp_3AssignAddVariableOpassignaddvariableop_3_resource
Cast_3:y:0^AssignAddVariableOp_2*
_output_shapes
 *
dtype02
AssignAddVariableOp_3°
div_no_nan/ReadVariableOpReadVariableOpassignaddvariableop_resource^AssignAddVariableOp*
_output_shapes
: *
dtype02
div_no_nan/ReadVariableOp©
div_no_nan/ReadVariableOp_1ReadVariableOpassignaddvariableop_1_resource^AssignAddVariableOp_1*
_output_shapes
: *
dtype02
div_no_nan/ReadVariableOp_1Н

div_no_nanDivNoNan!div_no_nan/ReadVariableOp:value:0#div_no_nan/ReadVariableOp_1:value:0*
T0*
_output_shapes
: 2

div_no_nanQ
IdentityIdentitydiv_no_nan:z:0*
T0*
_output_shapes
: 2

Identity©
div_no_nan_1/ReadVariableOpReadVariableOpassignaddvariableop_2_resource^AssignAddVariableOp_2*
_output_shapes
: *
dtype02
div_no_nan_1/ReadVariableOp≠
div_no_nan_1/ReadVariableOp_1ReadVariableOpassignaddvariableop_3_resource^AssignAddVariableOp_3*
_output_shapes
: *
dtype02
div_no_nan_1/ReadVariableOp_1Х
div_no_nan_1DivNoNan#div_no_nan_1/ReadVariableOp:value:0%div_no_nan_1/ReadVariableOp_1:value:0*
T0*
_output_shapes
: 2
div_no_nan_1W

Identity_1Identitydiv_no_nan_1:z:0*
T0*
_output_shapes
: 2

Identity_1э

Identity_2IdentityIdentity_1:output:0^AssignAddVariableOp^AssignAddVariableOp_1^AssignAddVariableOp_2^AssignAddVariableOp_3^RMSprop/Cast/ReadVariableOp^RMSprop/Cast_1/ReadVariableOp^RMSprop/Cast_2/ReadVariableOp$^RMSprop/RMSprop/AssignAddVariableOp(^RMSprop/RMSprop/update/AssignVariableOp*^RMSprop/RMSprop/update/AssignVariableOp_1&^RMSprop/RMSprop/update/ReadVariableOp+^RMSprop/RMSprop/update/Sqrt/ReadVariableOp*^RMSprop/RMSprop/update/mul/ReadVariableOp*^RMSprop/RMSprop/update_1/AssignVariableOp,^RMSprop/RMSprop/update_1/AssignVariableOp_1(^RMSprop/RMSprop/update_1/ReadVariableOp-^RMSprop/RMSprop/update_1/Sqrt/ReadVariableOp,^RMSprop/RMSprop/update_1/mul/ReadVariableOp*^RMSprop/RMSprop/update_2/AssignVariableOp,^RMSprop/RMSprop/update_2/AssignVariableOp_1(^RMSprop/RMSprop/update_2/ReadVariableOp-^RMSprop/RMSprop/update_2/Sqrt/ReadVariableOp,^RMSprop/RMSprop/update_2/mul/ReadVariableOp*^RMSprop/RMSprop/update_3/AssignVariableOp,^RMSprop/RMSprop/update_3/AssignVariableOp_1(^RMSprop/RMSprop/update_3/ReadVariableOp-^RMSprop/RMSprop/update_3/Sqrt/ReadVariableOp,^RMSprop/RMSprop/update_3/mul/ReadVariableOp*^RMSprop/RMSprop/update_4/AssignVariableOp,^RMSprop/RMSprop/update_4/AssignVariableOp_1(^RMSprop/RMSprop/update_4/ReadVariableOp-^RMSprop/RMSprop/update_4/Sqrt/ReadVariableOp,^RMSprop/RMSprop/update_4/mul/ReadVariableOp*^RMSprop/RMSprop/update_5/AssignVariableOp,^RMSprop/RMSprop/update_5/AssignVariableOp_1(^RMSprop/RMSprop/update_5/ReadVariableOp-^RMSprop/RMSprop/update_5/Sqrt/ReadVariableOp,^RMSprop/RMSprop/update_5/mul/ReadVariableOp*^custom_model/dense/BiasAdd/ReadVariableOp)^custom_model/dense/MatMul/ReadVariableOp,^custom_model/dense_1/BiasAdd/ReadVariableOp+^custom_model/dense_1/MatMul/ReadVariableOp,^custom_model/dense_2/BiasAdd/ReadVariableOp+^custom_model/dense_2/MatMul/ReadVariableOp^div_no_nan/ReadVariableOp^div_no_nan/ReadVariableOp_1^div_no_nan_1/ReadVariableOp^div_no_nan_1/ReadVariableOp_1*
T0*
_output_shapes
: 2

Identity_2ы

Identity_3IdentityIdentity:output:0^AssignAddVariableOp^AssignAddVariableOp_1^AssignAddVariableOp_2^AssignAddVariableOp_3^RMSprop/Cast/ReadVariableOp^RMSprop/Cast_1/ReadVariableOp^RMSprop/Cast_2/ReadVariableOp$^RMSprop/RMSprop/AssignAddVariableOp(^RMSprop/RMSprop/update/AssignVariableOp*^RMSprop/RMSprop/update/AssignVariableOp_1&^RMSprop/RMSprop/update/ReadVariableOp+^RMSprop/RMSprop/update/Sqrt/ReadVariableOp*^RMSprop/RMSprop/update/mul/ReadVariableOp*^RMSprop/RMSprop/update_1/AssignVariableOp,^RMSprop/RMSprop/update_1/AssignVariableOp_1(^RMSprop/RMSprop/update_1/ReadVariableOp-^RMSprop/RMSprop/update_1/Sqrt/ReadVariableOp,^RMSprop/RMSprop/update_1/mul/ReadVariableOp*^RMSprop/RMSprop/update_2/AssignVariableOp,^RMSprop/RMSprop/update_2/AssignVariableOp_1(^RMSprop/RMSprop/update_2/ReadVariableOp-^RMSprop/RMSprop/update_2/Sqrt/ReadVariableOp,^RMSprop/RMSprop/update_2/mul/ReadVariableOp*^RMSprop/RMSprop/update_3/AssignVariableOp,^RMSprop/RMSprop/update_3/AssignVariableOp_1(^RMSprop/RMSprop/update_3/ReadVariableOp-^RMSprop/RMSprop/update_3/Sqrt/ReadVariableOp,^RMSprop/RMSprop/update_3/mul/ReadVariableOp*^RMSprop/RMSprop/update_4/AssignVariableOp,^RMSprop/RMSprop/update_4/AssignVariableOp_1(^RMSprop/RMSprop/update_4/ReadVariableOp-^RMSprop/RMSprop/update_4/Sqrt/ReadVariableOp,^RMSprop/RMSprop/update_4/mul/ReadVariableOp*^RMSprop/RMSprop/update_5/AssignVariableOp,^RMSprop/RMSprop/update_5/AssignVariableOp_1(^RMSprop/RMSprop/update_5/ReadVariableOp-^RMSprop/RMSprop/update_5/Sqrt/ReadVariableOp,^RMSprop/RMSprop/update_5/mul/ReadVariableOp*^custom_model/dense/BiasAdd/ReadVariableOp)^custom_model/dense/MatMul/ReadVariableOp,^custom_model/dense_1/BiasAdd/ReadVariableOp+^custom_model/dense_1/MatMul/ReadVariableOp,^custom_model/dense_2/BiasAdd/ReadVariableOp+^custom_model/dense_2/MatMul/ReadVariableOp^div_no_nan/ReadVariableOp^div_no_nan/ReadVariableOp_1^div_no_nan_1/ReadVariableOp^div_no_nan_1/ReadVariableOp_1*
T0*
_output_shapes
: 2

Identity_3"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : : : : : : : : : : : : : : : : : : : 2*
AssignAddVariableOpAssignAddVariableOp2.
AssignAddVariableOp_1AssignAddVariableOp_12.
AssignAddVariableOp_2AssignAddVariableOp_22.
AssignAddVariableOp_3AssignAddVariableOp_32:
RMSprop/Cast/ReadVariableOpRMSprop/Cast/ReadVariableOp2>
RMSprop/Cast_1/ReadVariableOpRMSprop/Cast_1/ReadVariableOp2>
RMSprop/Cast_2/ReadVariableOpRMSprop/Cast_2/ReadVariableOp2J
#RMSprop/RMSprop/AssignAddVariableOp#RMSprop/RMSprop/AssignAddVariableOp2R
'RMSprop/RMSprop/update/AssignVariableOp'RMSprop/RMSprop/update/AssignVariableOp2V
)RMSprop/RMSprop/update/AssignVariableOp_1)RMSprop/RMSprop/update/AssignVariableOp_12N
%RMSprop/RMSprop/update/ReadVariableOp%RMSprop/RMSprop/update/ReadVariableOp2X
*RMSprop/RMSprop/update/Sqrt/ReadVariableOp*RMSprop/RMSprop/update/Sqrt/ReadVariableOp2V
)RMSprop/RMSprop/update/mul/ReadVariableOp)RMSprop/RMSprop/update/mul/ReadVariableOp2V
)RMSprop/RMSprop/update_1/AssignVariableOp)RMSprop/RMSprop/update_1/AssignVariableOp2Z
+RMSprop/RMSprop/update_1/AssignVariableOp_1+RMSprop/RMSprop/update_1/AssignVariableOp_12R
'RMSprop/RMSprop/update_1/ReadVariableOp'RMSprop/RMSprop/update_1/ReadVariableOp2\
,RMSprop/RMSprop/update_1/Sqrt/ReadVariableOp,RMSprop/RMSprop/update_1/Sqrt/ReadVariableOp2Z
+RMSprop/RMSprop/update_1/mul/ReadVariableOp+RMSprop/RMSprop/update_1/mul/ReadVariableOp2V
)RMSprop/RMSprop/update_2/AssignVariableOp)RMSprop/RMSprop/update_2/AssignVariableOp2Z
+RMSprop/RMSprop/update_2/AssignVariableOp_1+RMSprop/RMSprop/update_2/AssignVariableOp_12R
'RMSprop/RMSprop/update_2/ReadVariableOp'RMSprop/RMSprop/update_2/ReadVariableOp2\
,RMSprop/RMSprop/update_2/Sqrt/ReadVariableOp,RMSprop/RMSprop/update_2/Sqrt/ReadVariableOp2Z
+RMSprop/RMSprop/update_2/mul/ReadVariableOp+RMSprop/RMSprop/update_2/mul/ReadVariableOp2V
)RMSprop/RMSprop/update_3/AssignVariableOp)RMSprop/RMSprop/update_3/AssignVariableOp2Z
+RMSprop/RMSprop/update_3/AssignVariableOp_1+RMSprop/RMSprop/update_3/AssignVariableOp_12R
'RMSprop/RMSprop/update_3/ReadVariableOp'RMSprop/RMSprop/update_3/ReadVariableOp2\
,RMSprop/RMSprop/update_3/Sqrt/ReadVariableOp,RMSprop/RMSprop/update_3/Sqrt/ReadVariableOp2Z
+RMSprop/RMSprop/update_3/mul/ReadVariableOp+RMSprop/RMSprop/update_3/mul/ReadVariableOp2V
)RMSprop/RMSprop/update_4/AssignVariableOp)RMSprop/RMSprop/update_4/AssignVariableOp2Z
+RMSprop/RMSprop/update_4/AssignVariableOp_1+RMSprop/RMSprop/update_4/AssignVariableOp_12R
'RMSprop/RMSprop/update_4/ReadVariableOp'RMSprop/RMSprop/update_4/ReadVariableOp2\
,RMSprop/RMSprop/update_4/Sqrt/ReadVariableOp,RMSprop/RMSprop/update_4/Sqrt/ReadVariableOp2Z
+RMSprop/RMSprop/update_4/mul/ReadVariableOp+RMSprop/RMSprop/update_4/mul/ReadVariableOp2V
)RMSprop/RMSprop/update_5/AssignVariableOp)RMSprop/RMSprop/update_5/AssignVariableOp2Z
+RMSprop/RMSprop/update_5/AssignVariableOp_1+RMSprop/RMSprop/update_5/AssignVariableOp_12R
'RMSprop/RMSprop/update_5/ReadVariableOp'RMSprop/RMSprop/update_5/ReadVariableOp2\
,RMSprop/RMSprop/update_5/Sqrt/ReadVariableOp,RMSprop/RMSprop/update_5/Sqrt/ReadVariableOp2Z
+RMSprop/RMSprop/update_5/mul/ReadVariableOp+RMSprop/RMSprop/update_5/mul/ReadVariableOp2V
)custom_model/dense/BiasAdd/ReadVariableOp)custom_model/dense/BiasAdd/ReadVariableOp2T
(custom_model/dense/MatMul/ReadVariableOp(custom_model/dense/MatMul/ReadVariableOp2Z
+custom_model/dense_1/BiasAdd/ReadVariableOp+custom_model/dense_1/BiasAdd/ReadVariableOp2X
*custom_model/dense_1/MatMul/ReadVariableOp*custom_model/dense_1/MatMul/ReadVariableOp2Z
+custom_model/dense_2/BiasAdd/ReadVariableOp+custom_model/dense_2/BiasAdd/ReadVariableOp2X
*custom_model/dense_2/MatMul/ReadVariableOp*custom_model/dense_2/MatMul/ReadVariableOp26
div_no_nan/ReadVariableOpdiv_no_nan/ReadVariableOp2:
div_no_nan/ReadVariableOp_1div_no_nan/ReadVariableOp_12:
div_no_nan_1/ReadVariableOpdiv_no_nan_1/ReadVariableOp2>
div_no_nan_1/ReadVariableOp_1div_no_nan_1/ReadVariableOp_1:N J
&
_output_shapes
: 
 
_user_specified_namedata/0:B>

_output_shapes
: 
 
_user_specified_namedata/1
б2
ё
__inference__traced_save_223210
file_prefix+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop-
)savev2_dense_2_kernel_read_readvariableop+
'savev2_dense_2_bias_read_readvariableop+
'savev2_rmsprop_iter_read_readvariableop	,
(savev2_rmsprop_decay_read_readvariableop4
0savev2_rmsprop_learning_rate_read_readvariableop/
+savev2_rmsprop_momentum_read_readvariableop*
&savev2_rmsprop_rho_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop7
3savev2_rmsprop_dense_kernel_rms_read_readvariableop5
1savev2_rmsprop_dense_bias_rms_read_readvariableop9
5savev2_rmsprop_dense_1_kernel_rms_read_readvariableop7
3savev2_rmsprop_dense_1_bias_rms_read_readvariableop9
5savev2_rmsprop_dense_2_kernel_rms_read_readvariableop7
3savev2_rmsprop_dense_2_bias_rms_read_readvariableop
savev2_const

identity_1ИҐMergeV2CheckpointsП
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1Л
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard¶
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename•
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Ј

value≠
B™
B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB(optimizer/rho/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesі
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value6B4B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesн
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop)savev2_dense_2_kernel_read_readvariableop'savev2_dense_2_bias_read_readvariableop'savev2_rmsprop_iter_read_readvariableop(savev2_rmsprop_decay_read_readvariableop0savev2_rmsprop_learning_rate_read_readvariableop+savev2_rmsprop_momentum_read_readvariableop&savev2_rmsprop_rho_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop3savev2_rmsprop_dense_kernel_rms_read_readvariableop1savev2_rmsprop_dense_bias_rms_read_readvariableop5savev2_rmsprop_dense_1_kernel_rms_read_readvariableop3savev2_rmsprop_dense_1_bias_rms_read_readvariableop5savev2_rmsprop_dense_2_kernel_rms_read_readvariableop3savev2_rmsprop_dense_2_bias_rms_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *$
dtypes
2	2
SaveV2Ї
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes°
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*Л
_input_shapesz
x: :	Р2:2:22:2:2
:
: : : : : : : : : :	Р2:2:22:2:2
:
: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	Р2: 

_output_shapes
:2:$ 

_output_shapes

:22: 

_output_shapes
:2:$ 

_output_shapes

:2
: 

_output_shapes
:
:

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	Р2: 

_output_shapes
:2:$ 

_output_shapes

:22: 

_output_shapes
:2:$ 

_output_shapes

:2
: 

_output_shapes
:
:

_output_shapes
: 
А
З
H__inference_custom_model_layer_call_and_return_conditional_losses_222644	
image
dense_222628:	Р2
dense_222630:2 
dense_1_222633:22
dense_1_222635:2 
dense_2_222638:2

dense_2_222640:

identityИҐdense/StatefulPartitionedCallҐdense_1/StatefulPartitionedCallҐdense_2/StatefulPartitionedCallн
gaussian_noise/PartitionedCallPartitionedCallimage*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_gaussian_noise_layer_call_and_return_conditional_losses_2224162 
gaussian_noise/PartitionedCallу
flatten/PartitionedCallPartitionedCall'gaussian_noise/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€Р* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_2224242
flatten/PartitionedCallЯ
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_222628dense_222630*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_2224372
dense/StatefulPartitionedCallѓ
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_222633dense_1_222635*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_2224542!
dense_1/StatefulPartitionedCall±
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_222638dense_2_222640*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_2224712!
dense_2/StatefulPartitionedCallа
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':€€€€€€€€€: : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall:V R
/
_output_shapes
:€€€€€€€€€

_user_specified_nameimage
Г
И
H__inference_custom_model_layer_call_and_return_conditional_losses_222478

inputs
dense_222438:	Р2
dense_222440:2 
dense_1_222455:22
dense_1_222457:2 
dense_2_222472:2

dense_2_222474:

identityИҐdense/StatefulPartitionedCallҐdense_1/StatefulPartitionedCallҐdense_2/StatefulPartitionedCallо
gaussian_noise/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_gaussian_noise_layer_call_and_return_conditional_losses_2224162 
gaussian_noise/PartitionedCallу
flatten/PartitionedCallPartitionedCall'gaussian_noise/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€Р* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_2224242
flatten/PartitionedCallЯ
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_222438dense_222440*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_2224372
dense/StatefulPartitionedCallѓ
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_222455dense_1_222457*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_2224542!
dense_1/StatefulPartitionedCall±
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_222472dense_2_222474*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_2224712!
dense_2/StatefulPartitionedCallа
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':€€€€€€€€€: : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
©
э
$__inference_signature_wrapper_222690	
image
unknown:	Р2
	unknown_0:2
	unknown_1:22
	unknown_2:2
	unknown_3:2

	unknown_4:

identityИҐStatefulPartitionedCallД
StatefulPartitionedCallStatefulPartitionedCallimageunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€
*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8В **
f%R#
!__inference__wrapped_model_2224052
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':€€€€€€€€€: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:V R
/
_output_shapes
:€€€€€€€€€

_user_specified_nameimage
≠

у
A__inference_dense_layer_call_and_return_conditional_losses_223075

inputs1
matmul_readvariableop_resource:	Р2-
biasadd_readvariableop_resource:2
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	Р2*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€22
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:2*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€22	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€22
ReluЧ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€22

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€Р: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:€€€€€€€€€Р
 
_user_specified_nameinputs
«
±
H__inference_custom_model_layer_call_and_return_conditional_losses_222591

inputs
dense_222575:	Р2
dense_222577:2 
dense_1_222580:22
dense_1_222582:2 
dense_2_222585:2

dense_2_222587:

identityИҐdense/StatefulPartitionedCallҐdense_1/StatefulPartitionedCallҐdense_2/StatefulPartitionedCallҐ&gaussian_noise/StatefulPartitionedCallЖ
&gaussian_noise/StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_gaussian_noise_layer_call_and_return_conditional_losses_2225482(
&gaussian_noise/StatefulPartitionedCallы
flatten/PartitionedCallPartitionedCall/gaussian_noise/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€Р* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_2224242
flatten/PartitionedCallЯ
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_222575dense_222577*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_2224372
dense/StatefulPartitionedCallѓ
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_222580dense_1_222582*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_2224542!
dense_1/StatefulPartitionedCall±
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_222585dense_2_222587*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_2224712!
dense_2/StatefulPartitionedCallЙ
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall'^gaussian_noise/StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':€€€€€€€€€: : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2P
&gaussian_noise/StatefulPartitionedCall&gaussian_noise/StatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
Ђ

ф
C__inference_dense_1_layer_call_and_return_conditional_losses_222454

inputs0
matmul_readvariableop_resource:22-
biasadd_readvariableop_resource:2
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:22*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€22
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:2*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€22	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€22
ReluЧ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€22

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€2: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€2
 
_user_specified_nameinputs
Ш
Ф
&__inference_dense_layer_call_fn_223084

inputs
unknown:	Р2
	unknown_0:2
identityИҐStatefulPartitionedCallс
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_2224372
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€22

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€Р: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:€€€€€€€€€Р
 
_user_specified_nameinputs
е
_
C__inference_flatten_layer_call_and_return_conditional_losses_222424

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:€€€€€€€€€Р2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:€€€€€€€€€Р2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€:W S
/
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
№
З
-__inference_custom_model_layer_call_fn_222768

inputs
unknown:	Р2
	unknown_0:2
	unknown_1:22
	unknown_2:2
	unknown_3:2

	unknown_4:

identityИҐStatefulPartitionedCallђ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€
*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_custom_model_layer_call_and_return_conditional_losses_2224782
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':€€€€€€€€€: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
и
K
/__inference_gaussian_noise_layer_call_fn_223048

inputs
identity–
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_gaussian_noise_layer_call_and_return_conditional_losses_2224162
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€:W S
/
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
№
З
-__inference_custom_model_layer_call_fn_222785

inputs
unknown:	Р2
	unknown_0:2
	unknown_1:22
	unknown_2:2
	unknown_3:2

	unknown_4:

identityИҐStatefulPartitionedCallђ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€
*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_custom_model_layer_call_and_return_conditional_losses_2225912
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':€€€€€€€€€: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
Ю
f
J__inference_gaussian_noise_layer_call_and_return_conditional_losses_222416

inputs
identityb
IdentityIdentityinputs*
T0*/
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€:W S
/
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
у
ц
H__inference_custom_model_layer_call_and_return_conditional_losses_222717

inputs7
$dense_matmul_readvariableop_resource:	Р23
%dense_biasadd_readvariableop_resource:28
&dense_1_matmul_readvariableop_resource:225
'dense_1_biasadd_readvariableop_resource:28
&dense_2_matmul_readvariableop_resource:2
5
'dense_2_biasadd_readvariableop_resource:

identityИҐdense/BiasAdd/ReadVariableOpҐdense/MatMul/ReadVariableOpҐdense_1/BiasAdd/ReadVariableOpҐdense_1/MatMul/ReadVariableOpҐdense_2/BiasAdd/ReadVariableOpҐdense_2/MatMul/ReadVariableOpo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€  2
flatten/ConstА
flatten/ReshapeReshapeinputsflatten/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€Р2
flatten/Reshape†
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	Р2*
dtype02
dense/MatMul/ReadVariableOpЧ
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€22
dense/MatMulЮ
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02
dense/BiasAdd/ReadVariableOpЩ
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€22
dense/BiasAddj

dense/ReluReludense/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€22

dense/Relu•
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:22*
dtype02
dense_1/MatMul/ReadVariableOpЭ
dense_1/MatMulMatMuldense/Relu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€22
dense_1/MatMul§
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02 
dense_1/BiasAdd/ReadVariableOp°
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€22
dense_1/BiasAddp
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€22
dense_1/Relu•
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

:2
*
dtype02
dense_2/MatMul/ReadVariableOpЯ
dense_2/MatMulMatMuldense_1/Relu:activations:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€
2
dense_2/MatMul§
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02 
dense_2/BiasAdd/ReadVariableOp°
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€
2
dense_2/BiasAddy
dense_2/SoftmaxSoftmaxdense_2/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€
2
dense_2/Softmaxђ
IdentityIdentitydense_2/Softmax:softmax:0^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':€€€€€€€€€: : : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp:W S
/
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
≥

ф
C__inference_dense_2_layer_call_and_return_conditional_losses_222471

inputs0
matmul_readvariableop_resource:2
-
biasadd_readvariableop_resource:

identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:2
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€
2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€
2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€
2	
SoftmaxЦ
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€2: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€2
 
_user_specified_nameinputs
ф
h
/__inference_gaussian_noise_layer_call_fn_223053

inputs
identityИҐStatefulPartitionedCallи
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_gaussian_noise_layer_call_and_return_conditional_losses_2225482
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
ћ
D
(__inference_flatten_layer_call_fn_223064

inputs
identity¬
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€Р* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_2224242
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:€€€€€€€€€Р2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€:W S
/
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
Щ
Х
(__inference_dense_1_layer_call_fn_223104

inputs
unknown:22
	unknown_0:2
identityИҐStatefulPartitionedCallу
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_2224542
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€22

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€2: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€2
 
_user_specified_nameinputs
ƒ
∞
H__inference_custom_model_layer_call_and_return_conditional_losses_222665	
image
dense_222649:	Р2
dense_222651:2 
dense_1_222654:22
dense_1_222656:2 
dense_2_222659:2

dense_2_222661:

identityИҐdense/StatefulPartitionedCallҐdense_1/StatefulPartitionedCallҐdense_2/StatefulPartitionedCallҐ&gaussian_noise/StatefulPartitionedCallЕ
&gaussian_noise/StatefulPartitionedCallStatefulPartitionedCallimage*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_gaussian_noise_layer_call_and_return_conditional_losses_2225482(
&gaussian_noise/StatefulPartitionedCallы
flatten/PartitionedCallPartitionedCall/gaussian_noise/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€Р* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_2224242
flatten/PartitionedCallЯ
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_222649dense_222651*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_2224372
dense/StatefulPartitionedCallѓ
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_222654dense_1_222656*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_2224542!
dense_1/StatefulPartitionedCall±
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_222659dense_2_222661*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_2224712!
dense_2/StatefulPartitionedCallЙ
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall'^gaussian_noise/StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':€€€€€€€€€: : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2P
&gaussian_noise/StatefulPartitionedCall&gaussian_noise/StatefulPartitionedCall:V R
/
_output_shapes
:€€€€€€€€€

_user_specified_nameimage
й)
ц
H__inference_custom_model_layer_call_and_return_conditional_losses_222751

inputs7
$dense_matmul_readvariableop_resource:	Р23
%dense_biasadd_readvariableop_resource:28
&dense_1_matmul_readvariableop_resource:225
'dense_1_biasadd_readvariableop_resource:28
&dense_2_matmul_readvariableop_resource:2
5
'dense_2_biasadd_readvariableop_resource:

identityИҐdense/BiasAdd/ReadVariableOpҐdense/MatMul/ReadVariableOpҐdense_1/BiasAdd/ReadVariableOpҐdense_1/MatMul/ReadVariableOpҐdense_2/BiasAdd/ReadVariableOpҐdense_2/MatMul/ReadVariableOpb
gaussian_noise/ShapeShapeinputs*
T0*
_output_shapes
:2
gaussian_noise/ShapeЛ
!gaussian_noise/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!gaussian_noise/random_normal/meanП
#gaussian_noise/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL>2%
#gaussian_noise/random_normal/stddevГ
1gaussian_noise/random_normal/RandomStandardNormalRandomStandardNormalgaussian_noise/Shape:output:0*
T0*/
_output_shapes
:€€€€€€€€€*
dtype0*
seed±€е)*
seed2Ждљ23
1gaussian_noise/random_normal/RandomStandardNormalп
 gaussian_noise/random_normal/mulMul:gaussian_noise/random_normal/RandomStandardNormal:output:0,gaussian_noise/random_normal/stddev:output:0*
T0*/
_output_shapes
:€€€€€€€€€2"
 gaussian_noise/random_normal/mulѕ
gaussian_noise/random_normalAdd$gaussian_noise/random_normal/mul:z:0*gaussian_noise/random_normal/mean:output:0*
T0*/
_output_shapes
:€€€€€€€€€2
gaussian_noise/random_normalХ
gaussian_noise/addAddV2inputs gaussian_noise/random_normal:z:0*
T0*/
_output_shapes
:€€€€€€€€€2
gaussian_noise/addo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€  2
flatten/ConstР
flatten/ReshapeReshapegaussian_noise/add:z:0flatten/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€Р2
flatten/Reshape†
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	Р2*
dtype02
dense/MatMul/ReadVariableOpЧ
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€22
dense/MatMulЮ
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02
dense/BiasAdd/ReadVariableOpЩ
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€22
dense/BiasAddj

dense/ReluReludense/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€22

dense/Relu•
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:22*
dtype02
dense_1/MatMul/ReadVariableOpЭ
dense_1/MatMulMatMuldense/Relu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€22
dense_1/MatMul§
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02 
dense_1/BiasAdd/ReadVariableOp°
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€22
dense_1/BiasAddp
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€22
dense_1/Relu•
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

:2
*
dtype02
dense_2/MatMul/ReadVariableOpЯ
dense_2/MatMulMatMuldense_1/Relu:activations:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€
2
dense_2/MatMul§
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02 
dense_2/BiasAdd/ReadVariableOp°
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€
2
dense_2/BiasAddy
dense_2/SoftmaxSoftmaxdense_2/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€
2
dense_2/Softmaxђ
IdentityIdentitydense_2/Softmax:softmax:0^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':€€€€€€€€€: : : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp:W S
/
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
≥

ф
C__inference_dense_2_layer_call_and_return_conditional_losses_223115

inputs0
matmul_readvariableop_resource:2
-
biasadd_readvariableop_resource:

identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:2
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€
2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€
2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€
2	
SoftmaxЦ
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€2: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€2
 
_user_specified_nameinputs"ћL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Ѓ
serving_defaultЪ
?
image6
serving_default_image:0€€€€€€€€€;
dense_20
StatefulPartitionedCall:0€€€€€€€€€
tensorflow/serving/predict:фњ
“2
layer-0
layer-1
layer-2
layer_with_weights-0
layer-3
layer_with_weights-1
layer-4
layer_with_weights-2
layer-5
	optimizer
regularization_losses
	trainable_variables

	variables
	keras_api

signatures
\_default_save_signature
*]&call_and_return_all_conditional_losses
^__call__
_
train_step"џ/
_tf_keras_networkњ/{"name": "custom_model", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "CustomModel", "config": {"name": "custom_model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "image"}, "name": "image", "inbound_nodes": []}, {"class_name": "GaussianNoise", "config": {"name": "gaussian_noise", "trainable": true, "dtype": "float32", "stddev": 0.2}, "name": "gaussian_noise", "inbound_nodes": [[["image", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten", "inbound_nodes": [[["gaussian_noise", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 50, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense", "inbound_nodes": [[["flatten", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 50, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_1", "inbound_nodes": [[["dense", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 10, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_2", "inbound_nodes": [[["dense_1", 0, 0, {}]]]}], "input_layers": [["image", 0, 0]], "output_layers": [["dense_2", 0, 0]]}, "shared_object_id": 12, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 28, 28, 1]}, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 28, 28, 1]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 28, 28, 1]}, "float32", "image"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "CustomModel", "config": {"name": "custom_model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "image"}, "name": "image", "inbound_nodes": [], "shared_object_id": 0}, {"class_name": "GaussianNoise", "config": {"name": "gaussian_noise", "trainable": true, "dtype": "float32", "stddev": 0.2}, "name": "gaussian_noise", "inbound_nodes": [[["image", 0, 0, {}]]], "shared_object_id": 1}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten", "inbound_nodes": [[["gaussian_noise", 0, 0, {}]]], "shared_object_id": 2}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 50, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 3}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 4}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense", "inbound_nodes": [[["flatten", 0, 0, {}]]], "shared_object_id": 5}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 50, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 6}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 7}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_1", "inbound_nodes": [[["dense", 0, 0, {}]]], "shared_object_id": 8}, {"class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 10, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 9}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 10}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_2", "inbound_nodes": [[["dense_1", 0, 0, {}]]], "shared_object_id": 11}], "input_layers": [["image", 0, 0]], "output_layers": [["dense_2", 0, 0]]}}, "training_config": {"loss": {"class_name": "SparseCategoricalCrossentropy", "config": {"reduction": "auto", "name": "sparse_categorical_crossentropy", "from_logits": false}, "shared_object_id": 14}, "metrics": [[{"class_name": "MeanMetricWrapper", "config": {"name": "accuracy", "dtype": "float32", "fn": "sparse_categorical_accuracy"}, "shared_object_id": 15}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "RMSprop", "config": {"name": "RMSprop", "learning_rate": 0.0010000000474974513, "decay": 0.0, "rho": 0.8999999761581421, "momentum": 0.0, "epsilon": 1e-07, "centered": false}}}}
М
_init_input_shape"т
_tf_keras_input_layer“{"class_name": "InputLayer", "name": "image", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "image"}}
Х
regularization_losses
trainable_variables
	variables
	keras_api
*`&call_and_return_all_conditional_losses
a__call__"Ж
_tf_keras_layerм{"name": "gaussian_noise", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "GaussianNoise", "config": {"name": "gaussian_noise", "trainable": true, "dtype": "float32", "stddev": 0.2}, "inbound_nodes": [[["image", 0, 0, {}]]], "shared_object_id": 1}
ƒ
regularization_losses
trainable_variables
	variables
	keras_api
*b&call_and_return_all_conditional_losses
c__call__"µ
_tf_keras_layerЫ{"name": "flatten", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "inbound_nodes": [[["gaussian_noise", 0, 0, {}]]], "shared_object_id": 2, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}, "shared_object_id": 16}}
ч

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
*d&call_and_return_all_conditional_losses
e__call__"“
_tf_keras_layerЄ{"name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 50, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 3}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 4}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["flatten", 0, 0, {}]]], "shared_object_id": 5, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 784}}, "shared_object_id": 17}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 784]}}
ч

kernel
bias
regularization_losses
trainable_variables
 	variables
!	keras_api
*f&call_and_return_all_conditional_losses
g__call__"“
_tf_keras_layerЄ{"name": "dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 50, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 6}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 7}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["dense", 0, 0, {}]]], "shared_object_id": 8, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 50}}, "shared_object_id": 18}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 50]}}
ю

"kernel
#bias
$regularization_losses
%trainable_variables
&	variables
'	keras_api
*h&call_and_return_all_conditional_losses
i__call__"ў
_tf_keras_layerњ{"name": "dense_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 10, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 9}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 10}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["dense_1", 0, 0, {}]]], "shared_object_id": 11, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 50}}, "shared_object_id": 19}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 50]}}
Ф
(iter
	)decay
*learning_rate
+momentum
,rho	rmsV	rmsW	rmsX	rmsY	"rmsZ	#rms["
	optimizer
 "
trackable_list_wrapper
J
0
1
2
3
"4
#5"
trackable_list_wrapper
J
0
1
2
3
"4
#5"
trackable_list_wrapper
 
-layer_metrics
.metrics
/non_trainable_variables
regularization_losses
	trainable_variables

0layers
1layer_regularization_losses

	variables
^__call__
\_default_save_signature
*]&call_and_return_all_conditional_losses
&]"call_and_return_conditional_losses"
_generic_user_object
,
jserving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
≠
2layer_metrics
3metrics
4non_trainable_variables
regularization_losses
trainable_variables

5layers
6layer_regularization_losses
	variables
a__call__
*`&call_and_return_all_conditional_losses
&`"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
≠
7layer_metrics
8metrics
9non_trainable_variables
regularization_losses
trainable_variables

:layers
;layer_regularization_losses
	variables
c__call__
*b&call_and_return_all_conditional_losses
&b"call_and_return_conditional_losses"
_generic_user_object
:	Р22dense/kernel
:22
dense/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
≠
<layer_metrics
=metrics
>non_trainable_variables
regularization_losses
trainable_variables

?layers
@layer_regularization_losses
	variables
e__call__
*d&call_and_return_all_conditional_losses
&d"call_and_return_conditional_losses"
_generic_user_object
 :222dense_1/kernel
:22dense_1/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
≠
Alayer_metrics
Bmetrics
Cnon_trainable_variables
regularization_losses
trainable_variables

Dlayers
Elayer_regularization_losses
 	variables
g__call__
*f&call_and_return_all_conditional_losses
&f"call_and_return_conditional_losses"
_generic_user_object
 :2
2dense_2/kernel
:
2dense_2/bias
 "
trackable_list_wrapper
.
"0
#1"
trackable_list_wrapper
.
"0
#1"
trackable_list_wrapper
≠
Flayer_metrics
Gmetrics
Hnon_trainable_variables
$regularization_losses
%trainable_variables

Ilayers
Jlayer_regularization_losses
&	variables
i__call__
*h&call_and_return_all_conditional_losses
&h"call_and_return_conditional_losses"
_generic_user_object
:	 (2RMSprop/iter
: (2RMSprop/decay
: (2RMSprop/learning_rate
: (2RMSprop/momentum
: (2RMSprop/rho
 "
trackable_dict_wrapper
.
K0
L1"
trackable_list_wrapper
 "
trackable_list_wrapper
J
0
1
2
3
4
5"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
‘
	Mtotal
	Ncount
O	variables
P	keras_api"Э
_tf_keras_metricВ{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}, "shared_object_id": 20}
Ю
	Qtotal
	Rcount
S
_fn_kwargs
T	variables
U	keras_api"„
_tf_keras_metricЉ{"class_name": "MeanMetricWrapper", "name": "accuracy", "dtype": "float32", "config": {"name": "accuracy", "dtype": "float32", "fn": "sparse_categorical_accuracy"}, "shared_object_id": 15}
:  (2total
:  (2count
.
M0
N1"
trackable_list_wrapper
-
O	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
Q0
R1"
trackable_list_wrapper
-
T	variables"
_generic_user_object
):'	Р22RMSprop/dense/kernel/rms
": 22RMSprop/dense/bias/rms
*:(222RMSprop/dense_1/kernel/rms
$:"22RMSprop/dense_1/bias/rms
*:(2
2RMSprop/dense_2/kernel/rms
$:"
2RMSprop/dense_2/bias/rms
е2в
!__inference__wrapped_model_222405Љ
Л≤З
FullArgSpec
argsЪ 
varargsjargs
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *,Ґ)
'К$
image€€€€€€€€€
о2л
H__inference_custom_model_layer_call_and_return_conditional_losses_222717
H__inference_custom_model_layer_call_and_return_conditional_losses_222751
H__inference_custom_model_layer_call_and_return_conditional_losses_222644
H__inference_custom_model_layer_call_and_return_conditional_losses_222665ј
Ј≤≥
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
В2€
-__inference_custom_model_layer_call_fn_222493
-__inference_custom_model_layer_call_fn_222768
-__inference_custom_model_layer_call_fn_222785
-__inference_custom_model_layer_call_fn_222623ј
Ј≤≥
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
≈2¬
__inference_train_step_223028†
Ч≤У
FullArgSpec
argsЪ
jself
jdata
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
“2ѕ
J__inference_gaussian_noise_layer_call_and_return_conditional_losses_223032
J__inference_gaussian_noise_layer_call_and_return_conditional_losses_223043і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
Ь2Щ
/__inference_gaussian_noise_layer_call_fn_223048
/__inference_gaussian_noise_layer_call_fn_223053і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
н2к
C__inference_flatten_layer_call_and_return_conditional_losses_223059Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
“2ѕ
(__inference_flatten_layer_call_fn_223064Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
л2и
A__inference_dense_layer_call_and_return_conditional_losses_223075Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
–2Ќ
&__inference_dense_layer_call_fn_223084Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
н2к
C__inference_dense_1_layer_call_and_return_conditional_losses_223095Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
“2ѕ
(__inference_dense_1_layer_call_fn_223104Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
н2к
C__inference_dense_2_layer_call_and_return_conditional_losses_223115Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
“2ѕ
(__inference_dense_2_layer_call_fn_223124Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
…B∆
$__inference_signature_wrapper_222690image"Ф
Н≤Й
FullArgSpec
argsЪ 
varargs
 
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 Ш
!__inference__wrapped_model_222405s"#6Ґ3
,Ґ)
'К$
image€€€€€€€€€
™ "1™.
,
dense_2!К
dense_2€€€€€€€€€
ї
H__inference_custom_model_layer_call_and_return_conditional_losses_222644o"#>Ґ;
4Ґ1
'К$
image€€€€€€€€€
p 

 
™ "%Ґ"
К
0€€€€€€€€€

Ъ ї
H__inference_custom_model_layer_call_and_return_conditional_losses_222665o"#>Ґ;
4Ґ1
'К$
image€€€€€€€€€
p

 
™ "%Ґ"
К
0€€€€€€€€€

Ъ Љ
H__inference_custom_model_layer_call_and_return_conditional_losses_222717p"#?Ґ<
5Ґ2
(К%
inputs€€€€€€€€€
p 

 
™ "%Ґ"
К
0€€€€€€€€€

Ъ Љ
H__inference_custom_model_layer_call_and_return_conditional_losses_222751p"#?Ґ<
5Ґ2
(К%
inputs€€€€€€€€€
p

 
™ "%Ґ"
К
0€€€€€€€€€

Ъ У
-__inference_custom_model_layer_call_fn_222493b"#>Ґ;
4Ґ1
'К$
image€€€€€€€€€
p 

 
™ "К€€€€€€€€€
У
-__inference_custom_model_layer_call_fn_222623b"#>Ґ;
4Ґ1
'К$
image€€€€€€€€€
p

 
™ "К€€€€€€€€€
Ф
-__inference_custom_model_layer_call_fn_222768c"#?Ґ<
5Ґ2
(К%
inputs€€€€€€€€€
p 

 
™ "К€€€€€€€€€
Ф
-__inference_custom_model_layer_call_fn_222785c"#?Ґ<
5Ґ2
(К%
inputs€€€€€€€€€
p

 
™ "К€€€€€€€€€
£
C__inference_dense_1_layer_call_and_return_conditional_losses_223095\/Ґ,
%Ґ"
 К
inputs€€€€€€€€€2
™ "%Ґ"
К
0€€€€€€€€€2
Ъ {
(__inference_dense_1_layer_call_fn_223104O/Ґ,
%Ґ"
 К
inputs€€€€€€€€€2
™ "К€€€€€€€€€2£
C__inference_dense_2_layer_call_and_return_conditional_losses_223115\"#/Ґ,
%Ґ"
 К
inputs€€€€€€€€€2
™ "%Ґ"
К
0€€€€€€€€€

Ъ {
(__inference_dense_2_layer_call_fn_223124O"#/Ґ,
%Ґ"
 К
inputs€€€€€€€€€2
™ "К€€€€€€€€€
Ґ
A__inference_dense_layer_call_and_return_conditional_losses_223075]0Ґ-
&Ґ#
!К
inputs€€€€€€€€€Р
™ "%Ґ"
К
0€€€€€€€€€2
Ъ z
&__inference_dense_layer_call_fn_223084P0Ґ-
&Ґ#
!К
inputs€€€€€€€€€Р
™ "К€€€€€€€€€2®
C__inference_flatten_layer_call_and_return_conditional_losses_223059a7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€
™ "&Ґ#
К
0€€€€€€€€€Р
Ъ А
(__inference_flatten_layer_call_fn_223064T7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€
™ "К€€€€€€€€€РЇ
J__inference_gaussian_noise_layer_call_and_return_conditional_losses_223032l;Ґ8
1Ґ.
(К%
inputs€€€€€€€€€
p 
™ "-Ґ*
#К 
0€€€€€€€€€
Ъ Ї
J__inference_gaussian_noise_layer_call_and_return_conditional_losses_223043l;Ґ8
1Ґ.
(К%
inputs€€€€€€€€€
p
™ "-Ґ*
#К 
0€€€€€€€€€
Ъ Т
/__inference_gaussian_noise_layer_call_fn_223048_;Ґ8
1Ґ.
(К%
inputs€€€€€€€€€
p 
™ " К€€€€€€€€€Т
/__inference_gaussian_noise_layer_call_fn_223053_;Ґ8
1Ґ.
(К%
inputs€€€€€€€€€
p
™ " К€€€€€€€€€§
$__inference_signature_wrapper_222690|"#?Ґ<
Ґ 
5™2
0
image'К$
image€€€€€€€€€"1™.
,
dense_2!К
dense_2€€€€€€€€€
љ
__inference_train_step_223028Ы"#MN*,+VWXYZ[(QRHҐE
>Ґ;
9Ґ6
К
data/0 
К
data/1 
™ "9™6

accuracyК
accuracy 

lossК

loss 