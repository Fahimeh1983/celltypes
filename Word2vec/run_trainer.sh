#Set the gpu
gpu_device="cuda:1"

#check the walk properties
length=10000
N=1
p=1
q=1

#Check walk filename
walk_filename="walk_0.csv"

#Directories
walk_type="Directed_Weighted_node2vec"
project_name="NPP_GNN_project"
layer_class="single_layer"
layer="Pdyn-Oprd1"
roi="VISp"

#run hyperparams
window=2
batch_size=2000
num_workers=4
embedding_size=2
learning_rate=0.001
n_epochs=1000


source activate py374

rm IO_path.csv

python -m IO_check --N $N --length $length --p $p --q $q --walk_filename $walk_filename \
 --roi $roi --project_name $project_name --layer_class $layer_class --layer $layer --walk_type $walk_type \
 --window $window --batch_size $batch_size --embedding_size $embedding_size --learning_rate $learning_rate \
--n_epochs $n_epochs 

python -m trainer --IO_files "IO_path.csv" --window $window --batch_size $batch_size --num_workers $num_workers --embedding_size \
$embedding_size --learning_rate $learning_rate --n_epochs $n_epochs --gpu_device $gpu_device
